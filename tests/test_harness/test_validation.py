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

DELIVERED = KnownObject(id="#W5490111", attributes={"status": "delivered"})
PENDING = KnownObject(id="#W7387996", attributes={"status": "pending"})
WORLD = world_from([DELIVERED, PENDING])


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
            scoped_target="#W5490111",
            world=WORLD,
        )
        assert FailureCode.REFERENT_UNKNOWN in codes(result)

    def test_a_known_id_that_is_not_the_scoped_one_is_refused(self):
        # task 14: items returned from a different order than the one discussed.
        result = validator().review(
            call("return_delivered_order_items", order_id="#W7387996",
                 item_ids=["x"], payment_method_id="p"),
            scoped_target="#W5490111",
            world=WORLD,
        )
        assert FailureCode.REFERENT_AMBIGUOUS in codes(result)

    def test_the_scoped_object_passes(self):
        result = validator().review(
            call("return_delivered_order_items", order_id="#W5490111",
                 item_ids=["x"], payment_method_id="p"),
            scoped_target="#W5490111",
            world=WORLD,
        )
        assert result.passed

    def test_a_missing_referent_argument_is_reported(self):
        result = validator().review(
            call("cancel_pending_order"), scoped_target="#W7387996", world=WORLD
        )
        assert FailureCode.SCHEMA_INVALID in codes(result)

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
            scoped_target="#W7387996",
            world=WORLD,
        )
        assert FailureCode.PRECONDITION_UNMET in codes(result)

    def test_returning_a_delivered_order_passes(self):
        result = validator().review(
            call("return_delivered_order_items", order_id="#W5490111",
                 item_ids=["x"], payment_method_id="p"),
            scoped_target="#W5490111",
            world=WORLD,
        )
        assert result.passed

    def test_cancelling_a_delivered_order_is_refused(self):
        result = validator().review(
            call("cancel_pending_order", order_id="#W5490111"),
            scoped_target="#W5490111",
            world=WORLD,
        )
        assert FailureCode.PRECONDITION_UNMET in codes(result)

    def test_the_refusal_states_both_the_observed_and_the_required_status(self):
        result = validator().review(
            call("cancel_pending_order", order_id="#W5490111"),
            scoped_target="#W5490111",
            world=WORLD,
        )
        failure = next(f for f in result.failures if f.code is FailureCode.PRECONDITION_UNMET)
        assert failure.observed == "delivered"
        assert failure.expected == ["pending"]

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
            scoped_target="#W5490111",
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
            scoped_target="#W5490111",
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
            # These args name a known order that is not the scoped one, so the
            # referent failure here is AMBIGUOUS rather than UNKNOWN.
            ("referent_uniqueness_enabled", FailureCode.REFERENT_AMBIGUOUS),
            ("precondition_check_enabled", FailureCode.PRECONDITION_UNMET),
        ],
    )
    def test_disabling_one_silences_only_its_own_code(self, switch, expected):
        args = dict(order_id="#W7387996", item_ids=["x"], payment_method_id="p", country="Nowhere")
        on = validator().review(
            call("return_delivered_order_items", **args), scoped_target="#W5490111", world=WORLD
        )
        off = validator(**{switch: False}).review(
            call("return_delivered_order_items", **args), scoped_target="#W5490111", world=WORLD
        )
        assert expected in codes(on)
        assert expected not in codes(off)
        assert codes(off) == codes(on) - {expected}

    def test_all_checks_off_passes_anything(self):
        v = validator(
            schema_check_enabled=False,
            value_canonicalization_enabled=False,
            referent_uniqueness_enabled=False,
            precondition_check_enabled=False,
        )
        result = v.review(
            call("return_delivered_order_items", order_id="#12345", country="Freedonia"),
            scoped_target="#W5490111",
            world=WORLD,
        )
        assert result.passed

    def test_the_baseline_arm_validates_nothing(self):
        v = Validator(CONSTRAINTS, HarnessConfig.baseline())
        assert v.review(call("modify_user_address", country="United States")).passed
