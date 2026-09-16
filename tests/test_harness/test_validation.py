"""Tests for harness/validation.py.

Every constraint below is taken from the domain rather than invented. The
canonical value sets come from retail's db.json (all 500 users have
country="USA" and one of 13 two-letter states); the preconditions are quoted
from policy.md; the failing arguments are the ones W1 actually recorded.

No LLM, no benchmark, no network.
"""

import pytest

from policygate.harness.config import HarnessConfig
from policygate.harness.types import FailureCode, ToolCall
from policygate.harness.validation import (
    ArgumentSchema,
    DomainConstraints,
    KnownObject,
    Precondition,
    Validator,
    ValueConstraint,
    observe,
    world_from,
)

# Two-letter states as stored; "New York" and "United States" are what task 22
# wrote in both failing trials, and "NY"/"USA" what the passing trial wrote.
COUNTRY = ValueConstraint(
    field="country", allowed=frozenset({"USA"}), aliases={"United States": "USA"}
)
STATE = ValueConstraint(
    field="state",
    allowed=frozenset({"NY", "CO", "TX", "PA", "AZ", "CA", "IN", "FL"}),
    aliases={"New York": "NY", "Colorado": "CO"},
)

CONSTRAINTS = DomainConstraints(
    schemas={
        "return_delivered_order_items": ArgumentSchema(
            tool="return_delivered_order_items",
            required=frozenset({"order_id", "item_ids", "payment_method_id"}),
        ),
    },
    value_constraints=(COUNTRY, STATE),
    referent_fields={
        "return_delivered_order_items": "order_id",
        "cancel_pending_order": "order_id",
        "exchange_delivered_order_items": "order_id",
    },
    preconditions=(
        # policy.md: "An order can only be returned if its status is 'delivered'"
        Precondition(
            name="return requires delivered",
            tool="return_delivered_order_items",
            attribute="status",
            allowed_values=frozenset({"delivered"}),
        ),
        # policy.md: "An order can only be cancelled if its status is 'pending'"
        Precondition(
            name="cancel requires pending",
            tool="cancel_pending_order",
            attribute="status",
            allowed_values=frozenset({"pending"}),
        ),
    ),
)

USER = "mia_garcia_4516"
DELIVERED = KnownObject(id="#W5490111", attributes={"status": "delivered"}, owner_id=USER)
PENDING = KnownObject(id="#W7387996", attributes={"status": "pending"}, owner_id=USER)
# Listed by a user profile, never fetched: known to exist, status unknown.
LISTED = KnownObject(id="#W9000001", owner_id=USER)
FOREIGN = KnownObject(id="#W8000002", attributes={"status": "delivered"}, owner_id="noah_ito_3850")
WORLD = world_from([DELIVERED, PENDING, LISTED, FOREIGN])


def validator(**overrides) -> Validator:
    return Validator(CONSTRAINTS, HarnessConfig.full(**overrides))


def call(name: str, **args) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=args)


def codes(result) -> set[FailureCode]:
    return {f.code for f in result.failures}


class TestCanonicalValues:
    """task 22: same address, non-stored surface form, hash comparison fails."""

    def test_the_exact_arguments_of_the_failing_trial_are_refused(self):
        result = validator().review(
            call("modify_user_address", state="New York", country="United States")
        )
        assert result.passed is False
        assert codes(result) == {FailureCode.VALUE_NOT_CANONICAL}

    def test_the_arguments_of_the_passing_trial_are_accepted(self):
        assert validator().review(call("modify_user_address", state="NY", country="USA")).passed

    def test_the_refusal_names_the_value_to_write(self):
        result = validator().review(call("modify_user_address", country="United States"))
        assert result.failures[0].expected == "USA"
        assert "USA" in result.failures[0].message

    def test_every_offending_field_is_reported_at_once(self):
        # One error per turn would burn turns and make the harness look like the
        # cause of the extra steps.
        result = validator().review(
            call("modify_user_address", state="New York", country="United States")
        )
        assert {f.field for f in result.failures} == {"state", "country"}

    def test_case_differences_are_resolved_to_the_stored_form(self):
        result = validator().review(call("modify_user_address", country="usa"))
        assert result.failures[0].expected == "USA"

    def test_an_unrecognisable_value_lists_what_is_allowed(self):
        result = validator().review(call("modify_user_address", country="Freedonia"))
        assert result.failures[0].expected == ["USA"]

    def test_unconstrained_fields_are_left_alone(self):
        assert validator().review(call("modify_user_address", address1="101 Highway")).passed

    def test_the_call_is_never_rewritten(self):
        # Repairing arguments would mean the recorded action is not the one the
        # model chose, which is what the whole attribution rests on.
        c = call("modify_user_address", country="United States")
        validator().review(c)
        assert c.arguments["country"] == "United States"


class TestReferentIdentity:
    def test_an_invented_id_is_refused(self):
        # 17 of 31 "Order not found" errors used an id that had never appeared
        # in any tool result -- #12345, #67890 and similar.
        result = validator().review(
            call("return_delivered_order_items", order_id="#12345",
                 item_ids=["x"], payment_method_id="p"),
            subject=USER,
            world=WORLD,
        )
        assert FailureCode.REFERENT_UNKNOWN in codes(result)

    def test_another_users_order_is_refused(self):
        # policy.md: "must deny any requests for tasks related to any other user".
        result = validator().review(
            call("return_delivered_order_items", order_id=FOREIGN.id,
                 item_ids=["x"], payment_method_id="p"),
            subject=USER,
            world=WORLD,
        )
        assert codes(result) == {FailureCode.REFERENT_NOT_OWNED}

    def test_any_of_the_users_own_orders_passes_whichever_was_looked_up_last(self):
        """W4 v1 regression: the check compared the referent with the last order
        fetched, and refused 107 reference actions of multi-order tasks.

        Here the session has looked at both of the user's orders, the pending one
        last, and returns items from the delivered one -- task 58's shape.
        """
        result = validator().review(
            call("return_delivered_order_items", order_id=DELIVERED.id,
                 item_ids=["x"], payment_method_id="p"),
            subject=USER,
            world=world_from([DELIVERED, PENDING]),
        )
        assert result.passed

    def test_ownership_is_not_judged_before_anyone_is_authenticated(self):
        # Identity-first is gating's rule; reporting it again here as "not owned
        # by None" would be a second, misleading account of the same problem.
        result = validator().review(
            call("return_delivered_order_items", order_id=FOREIGN.id,
                 item_ids=["x"], payment_method_id="p"),
            world=WORLD,
        )
        assert result.passed

    def test_ownership_is_not_judged_when_the_result_did_not_say(self):
        unowned = KnownObject(id="#W7000003", attributes={"status": "delivered"})
        result = validator().review(
            call("return_delivered_order_items", order_id=unowned.id,
                 item_ids=["x"], payment_method_id="p"),
            subject=USER,
            world=world_from([unowned]),
        )
        assert result.passed

    def test_the_users_own_order_passes(self):
        result = validator().review(
            call("return_delivered_order_items", order_id="#W5490111",
                 item_ids=["x"], payment_method_id="p"),
            subject=USER,
            world=WORLD,
        )
        assert result.passed

    def test_a_missing_referent_argument_is_reported(self):
        # cancel_pending_order has no registered schema, so this failure can only
        # come from the referent check.
        result = validator().review(
            call("cancel_pending_order"), subject=USER, world=WORLD
        )
        assert FailureCode.SCHEMA_INVALID in codes(result)

    def test_a_missing_referent_is_reported_once_not_twice(self):
        # The schema check already names a missing required argument; a second
        # failure on the same field reads as two separate things to fix.
        result = validator().review(
            call("return_delivered_order_items", item_ids=["x"], payment_method_id="p"),
            subject=USER,
            world=WORLD,
        )
        assert [f.field for f in result.failures].count("order_id") == 1

    def test_the_referent_check_still_reports_it_when_no_schema_covers_it(self):
        result = validator(schema_check_enabled=False).review(
            call("return_delivered_order_items", item_ids=["x"]),
            subject=USER,
            world=WORLD,
        )
        assert any(f.field == "order_id" for f in result.failures)

    def test_tools_without_a_declared_referent_are_not_checked(self):
        assert validator().review(call("get_product_details", product_id="p1"), world=WORLD).passed

    def test_facts_come_from_tool_results_not_from_what_was_said(self):
        # An id the user mentioned but that no lookup returned is not known.
        result = validator().review(
            call("cancel_pending_order", order_id="#W0000001"), world=WORLD
        )
        assert FailureCode.REFERENT_UNKNOWN in codes(result)


class TestPreconditions:
    """Quoted from policy.md, not inferred."""

    def test_returning_a_pending_order_is_refused(self):
        result = validator().review(
            call("return_delivered_order_items", order_id="#W7387996",
                 item_ids=["x"], payment_method_id="p"),
            subject=USER,
            world=WORLD,
        )
        assert FailureCode.PRECONDITION_UNMET in codes(result)

    def test_returning_a_delivered_order_passes(self):
        result = validator().review(
            call("return_delivered_order_items", order_id="#W5490111",
                 item_ids=["x"], payment_method_id="p"),
            subject=USER,
            world=WORLD,
        )
        assert result.passed

    def test_cancelling_a_delivered_order_is_refused(self):
        result = validator().review(
            call("cancel_pending_order", order_id="#W5490111"),
            subject=USER,
            world=WORLD,
        )
        assert FailureCode.PRECONDITION_UNMET in codes(result)

    def test_the_refusal_states_both_the_observed_and_the_required_status(self):
        result = validator().review(
            call("cancel_pending_order", order_id="#W5490111"),
            subject=USER,
            world=WORLD,
        )
        failure = next(f for f in result.failures if f.code is FailureCode.PRECONDITION_UNMET)
        assert failure.observed == "delivered"
        assert failure.expected == ["pending"]

    def test_an_unretrieved_status_is_refused_with_a_message_the_model_can_act_on(self):
        # W4 v1 told the model "has status=None"; it should say to look it up.
        result = validator().review(
            call("return_delivered_order_items", order_id=LISTED.id,
                 item_ids=["x"], payment_method_id="p"),
            subject=USER,
            world=WORLD,
        )
        failure = next(f for f in result.failures if f.code is FailureCode.PRECONDITION_UNMET)
        assert "has not been retrieved" in failure.message
        assert "None" not in failure.message

    def test_preconditions_are_skipped_for_an_object_that_is_not_known(self):
        # "status must be delivered" about a nonexistent order would be noise on
        # top of the real problem.
        result = validator().review(
            call("return_delivered_order_items", order_id="#12345",
                 item_ids=["x"], payment_method_id="p"),
            world=WORLD,
        )
        assert FailureCode.PRECONDITION_UNMET not in codes(result)


class TestArgumentSchema:
    def test_a_missing_required_argument_is_reported(self):
        result = validator().review(
            call("return_delivered_order_items", order_id="#W5490111"),
            subject=USER,
            world=WORLD,
        )
        assert FailureCode.SCHEMA_INVALID in codes(result)
        assert {f.field for f in result.failures if f.code is FailureCode.SCHEMA_INVALID} == {
            "item_ids",
            "payment_method_id",
        }

    def test_an_unknown_argument_is_reported(self):
        result = validator().review(
            call("return_delivered_order_items", order_id="#W5490111", item_ids=["x"],
                 payment_method_id="p", refund_to="paypal"),
            subject=USER,
            world=WORLD,
        )
        assert any(f.field == "refund_to" for f in result.failures)

    def test_tools_without_a_declared_schema_are_not_checked(self):
        assert validator().review(call("calculate", expression="1+1")).passed


class TestSwitchesAreIndependent:
    """Each mechanism answers a different W1 failure; one shared flag could not
    attribute an improvement to any of them."""

    @pytest.mark.parametrize(
        "switch,expected",
        [
            ("value_canonicalization_enabled", FailureCode.VALUE_NOT_CANONICAL),
            # The order is known but owned by USER, and the session acts for
            # someone else, so the referent failure is NOT_OWNED, not UNKNOWN.
            ("referent_check_enabled", FailureCode.REFERENT_NOT_OWNED),
            ("precondition_check_enabled", FailureCode.PRECONDITION_UNMET),
        ],
    )
    def test_disabling_one_silences_only_its_own_code(self, switch, expected):
        args = dict(order_id="#W7387996", item_ids=["x"], payment_method_id="p", country="Nowhere")
        on = validator().review(
            call("return_delivered_order_items", **args), subject="noah_ito_3850", world=WORLD
        )
        off = validator(**{switch: False}).review(
            call("return_delivered_order_items", **args), subject="noah_ito_3850", world=WORLD
        )
        assert expected in codes(on)
        assert expected not in codes(off)
        assert codes(off) == codes(on) - {expected}

    def test_all_checks_off_passes_anything(self):
        v = validator(
            schema_check_enabled=False,
            value_canonicalization_enabled=False,
            referent_check_enabled=False,
            precondition_check_enabled=False,
        )
        result = v.review(
            call("return_delivered_order_items", order_id="#12345", country="Freedonia"),
            subject=USER,
            world=WORLD,
        )
        assert result.passed

    def test_the_baseline_arm_validates_nothing(self):
        v = Validator(CONSTRAINTS, HarnessConfig.baseline())
        assert v.review(call("modify_user_address", country="United States")).passed


class TestNonStringValues:
    def test_a_constrained_field_holding_a_number_is_refused(self):
        # A constrained field is an enumeration of strings; state=42 passing
        # would be a hole in the check meant to close exactly that.
        result = validator().review(call("modify_user_address", state=42))
        assert FailureCode.VALUE_NOT_CANONICAL in codes(result)

    def test_the_refusal_names_the_type_it_got(self):
        result = validator().review(call("modify_user_address", country=None))
        assert "NoneType" in result.failures[0].message

    def test_a_list_is_refused_too(self):
        result = validator().review(call("modify_user_address", state=["NY"]))
        assert result.passed is False


class TestCompositionWithGating:
    """Validation runs after gating and assumes it. Pinned here so the ordering
    is not merely implied by the code."""

    def test_an_unregistered_tool_passes_every_check(self):
        # Refusing it is GateRule.UNKNOWN_TOOL's job; duplicating the refusal
        # would double-count it in the ablation table.
        assert validator().review(call("totally_unknown_tool", anything=1)).passed

    def test_which_means_validation_alone_does_not_refuse_unknown_tools(self):
        # The consequence an ablation is meant to expose, not a defect.
        assert validator().review(call("drop_all_orders")).passed


class TestWorldFrom:
    def test_conflicting_observations_of_one_object_are_rejected(self):
        with pytest.raises(ValueError, match="conflicting observations"):
            world_from(
                [
                    KnownObject(id="#W1", attributes={"status": "pending"}),
                    KnownObject(id="#W1", attributes={"status": "cancelled"}),
                ]
            )

    def test_an_identical_repeat_is_harmless(self):
        obj = KnownObject(id="#W1", attributes={"status": "pending"})
        assert world_from([obj, obj]) == {"#W1": obj}

    def test_assigning_into_the_mapping_is_how_an_object_changes(self):
        world = world_from([KnownObject(id="#W1", attributes={"status": "pending"})])
        world["#W1"] = KnownObject(id="#W1", attributes={"status": "cancelled"})
        assert world["#W1"].attributes["status"] == "cancelled"


class TestRepeatedObservations:
    """W4 v2: a profile listing replaced an order fetched earlier and erased its status."""

    def test_a_listing_without_status_keeps_the_status_already_retrieved(self):
        world = {}
        observe(world, [PENDING])
        observe(world, [KnownObject(id=PENDING.id, owner_id=USER)])  # get_user_details
        assert world[PENDING.id].attributes == {"status": "pending"}

    def test_a_newer_status_replaces_the_old_one(self):
        world = {}
        observe(world, [PENDING])
        observe(world, [KnownObject(id=PENDING.id, attributes={"status": "cancelled"})])
        assert world[PENDING.id].attributes["status"] == "cancelled"
        assert world[PENDING.id].owner_id == USER  # not erased by a result that omits it

    def test_the_write_is_not_refused_after_the_profile_is_read_again(self):
        world = {}
        observe(world, [PENDING])
        observe(world, [KnownObject(id=PENDING.id, owner_id=USER)])
        assert validator().review(call("cancel_pending_order", order_id=PENDING.id),
                                  subject=USER, world=world).passed

    def test_two_different_objects_cannot_be_folded(self):
        with pytest.raises(ValueError, match="not another observation"):
            PENDING.observed_again(DELIVERED)


class TestEveryFailureSaysWhatToDoNext:
    """W4 v4: the refusal has to carry an action, not only a diagnosis."""

    CASES = [
        ("missing argument", "return_delivered_order_items",
         {"order_id": "#W5490111"}),
        ("unknown argument", "return_delivered_order_items",
         {"order_id": "#W5490111", "item_ids": ["x"], "payment_method_id": "p",
          "refund_to": "paypal"}),
        ("non-canonical value", "modify_user_address", {"state": "New York"}),
        ("non-string value", "modify_user_address", {"state": 42}),
        ("unknown referent", "return_delivered_order_items",
         {"order_id": "#12345", "item_ids": ["x"], "payment_method_id": "p"}),
        ("another user's order", "return_delivered_order_items",
         {"order_id": "#W8000002", "item_ids": ["x"], "payment_method_id": "p"}),
        ("status never retrieved", "return_delivered_order_items",
         {"order_id": "#W9000001", "item_ids": ["x"], "payment_method_id": "p"}),
        ("precondition unmet", "cancel_pending_order", {"order_id": "#W5490111"}),
    ]

    @pytest.mark.parametrize("name,tool,args", CASES)
    def test_every_failure_carries_a_remedy(self, name, tool, args):
        result = validator().review(call(tool, **args), subject=USER, world=WORLD)
        assert not result.passed, name
        for failure in result.failures:
            assert failure.remedy, f"{name}: {failure.code} has no remedy"
            assert failure.remedy != failure.message, name

    def test_the_remedy_names_the_value_to_send(self):
        result = validator().review(call("modify_user_address", state="New York"))
        assert "'NY'" in result.failures[0].remedy

    def test_the_remedy_for_an_unknown_order_points_at_what_is_known(self):
        result = validator().review(
            call("cancel_pending_order", order_id="#12345"), subject=USER, world=WORLD
        )
        remedy = next(f.remedy for f in result.failures
                      if f.code is FailureCode.REFERENT_UNKNOWN)
        assert PENDING.id in remedy

    def test_a_passing_call_has_nothing_to_remedy(self):
        assert validator().review(
            call("cancel_pending_order", order_id=PENDING.id), subject=USER, world=WORLD
        ).failures == ()

