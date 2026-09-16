"""Tests for harness/gating.py.

The catalogue mirrors retail's real tool types, checked against the domain
rather than assumed: identity lookups are typed READ exactly like the profile
reads, and transfer_to_human_agents is typed GENERIC alongside calculate. Both
are why ToolRole exists.

No LLM, no benchmark, no network.
"""

import pytest

from policygate.harness.config import HarnessConfig
from policygate.harness.gating import GateRule, ToolGate
from policygate.harness.state import SessionState
from policygate.harness.types import (
    ClauseRef,
    DecisionOutcome,
    ToolCall,
    ToolKind,
    ToolRole,
    ToolSpec,
)

RETAIL_CATALOGUE = (
    # Identity lookups are READ in tau2, same as the profile reads below.
    ToolSpec(name="find_user_id_by_email", kind=ToolKind.READ, role=ToolRole.IDENTITY_LOOKUP),
    ToolSpec(name="find_user_id_by_name_zip", kind=ToolKind.READ, role=ToolRole.IDENTITY_LOOKUP),
    ToolSpec(name="get_user_details", kind=ToolKind.READ),
    ToolSpec(name="get_order_details", kind=ToolKind.READ),
    ToolSpec(name="get_product_details", kind=ToolKind.READ),
    ToolSpec(name="return_delivered_order_items", kind=ToolKind.WRITE),
    ToolSpec(name="exchange_delivered_order_items", kind=ToolKind.WRITE),
    ToolSpec(name="cancel_pending_order", kind=ToolKind.WRITE),
    ToolSpec(name="calculate", kind=ToolKind.GENERIC),
    # GENERIC in tau2, so only its role marks it as the way out.
    ToolSpec(name="transfer_to_human_agents", kind=ToolKind.GENERIC, role=ToolRole.ESCALATION),
)

WRITES = {"return_delivered_order_items", "exchange_delivered_order_items", "cancel_pending_order"}


def gate(**overrides) -> ToolGate:
    return ToolGate(RETAIL_CATALOGUE, HarnessConfig.full(**overrides))


def call(name: str, **args) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=args)


class TestVisibilityByState:
    def test_unverified_offers_only_identity_lookups_and_escalation(self):
        # policy.md: authenticate first, "even when the user already provides
        # the user id". Reads are withheld because they return the profile of a
        # subject nobody has confirmed.
        assert gate().visible_tools(SessionState.UNVERIFIED) == {
            "find_user_id_by_email",
            "find_user_id_by_name_zip",
            "transfer_to_human_agents",
        }

    def test_verified_opens_reads_but_not_writes(self):
        visible = gate().visible_tools(SessionState.VERIFIED)
        assert "get_order_details" in visible
        assert not (visible & WRITES)

    def test_scoped_opens_everything(self):
        assert gate().visible_tools(SessionState.SCOPED) == {s.name for s in RETAIL_CATALOGUE}

    def test_pending_approval_freezes_the_session(self):
        assert gate().visible_tools(SessionState.PENDING_APPROVAL) == frozenset()

    def test_transferred_offers_nothing(self):
        assert gate().visible_tools(SessionState.TRANSFERRED) == frozenset()

    def test_escalation_stays_available_while_the_session_is_live(self):
        for state in (SessionState.UNVERIFIED, SessionState.VERIFIED, SessionState.SCOPED):
            assert "transfer_to_human_agents" in gate().visible_tools(state)

    def test_identity_lookups_remain_available_after_verification(self):
        # Harmless, and the model may legitimately re-look-up the same user.
        assert "find_user_id_by_email" in gate().visible_tools(SessionState.VERIFIED)


class TestDisabledGatingIsAGenuineControl:
    def test_state_tools_ignore_the_switch_so_pruning_can_run_alone(self):
        """Pruning reads state_tools; before the split it pruned nothing without gating."""
        off = ToolGate(RETAIL_CATALOGUE, HarnessConfig.baseline())
        on = gate()
        for state in SessionState:
            assert off.state_tools(state) == on.visible_tools(state)

    def test_everything_is_visible_in_every_state(self):
        g = gate(gating_enabled=False)
        for state in SessionState:
            assert g.visible_tools(state) == {s.name for s in RETAIL_CATALOGUE}

    def test_every_call_is_allowed_and_says_so(self):
        g = gate(gating_enabled=False)
        d = g.review_call(call("return_delivered_order_items"), SessionState.UNVERIFIED)
        assert d.allowed
        assert d.rule == GateRule.DISABLED

    def test_even_an_unknown_tool_passes_when_gating_is_off(self):
        # The baseline arm must not quietly keep half the mechanism.
        g = gate(gating_enabled=False)
        assert g.review_call(call("no_such_tool"), SessionState.SCOPED).allowed


class TestRefusalsNameTheirRule:
    def test_a_read_before_verification_cites_identity_first(self):
        d = gate().review_call(call("get_user_details"), SessionState.UNVERIFIED)
        assert d.outcome is DecisionOutcome.DENY
        assert d.rule == GateRule.IDENTITY_FIRST

    def test_a_write_without_a_unique_target_cites_scope(self):
        # task 14: items returned from a different order than the one discussed.
        d = gate().review_call(call("return_delivered_order_items"), SessionState.VERIFIED)
        assert d.rule == GateRule.WRITE_REQUIRES_SCOPE

    def test_anything_during_a_pending_approval_cites_the_freeze(self):
        d = gate().review_call(call("get_order_details"), SessionState.PENDING_APPROVAL)
        assert d.rule == GateRule.FROZEN_PENDING_APPROVAL

    def test_anything_after_a_transfer_cites_the_ended_session(self):
        d = gate().review_call(call("get_order_details"), SessionState.TRANSFERRED)
        assert d.rule == GateRule.SESSION_ENDED

    def test_an_unknown_tool_is_named_as_such(self):
        d = gate().review_call(call("delete_everything"), SessionState.SCOPED)
        assert d.rule == GateRule.UNKNOWN_TOOL

    def test_refusals_explain_what_to_do_instead(self):
        d = gate().review_call(call("get_user_details"), SessionState.UNVERIFIED)
        assert "identity" in d.reason

    def test_an_allowed_call_is_attributed_too(self):
        d = gate().review_call(call("get_order_details"), SessionState.VERIFIED)
        assert d.allowed
        assert d.rule == GateRule.ALLOWED


class TestOneCallPerTurn:
    """Both retail and airline state it; W1 found it violated in 78% of runs."""

    def test_a_single_call_passes(self):
        review = gate().review_turn([call("get_order_details")], SessionState.VERIFIED)
        assert review.refused is False
        assert len(review.allowed_calls) == 1

    def test_two_calls_refuse_the_whole_turn(self):
        # Not "keep the first, drop the rest": executing part of a proposal
        # would leave the world half-changed against an intent never expressed.
        review = gate().review_turn(
            [call("get_order_details"), call("get_product_details")], SessionState.VERIFIED
        )
        assert review.allowed_calls == ()
        assert {d.rule for d in review.decisions} == {GateRule.ONE_CALL_PER_TURN}

    def test_one_decision_is_returned_per_proposed_call(self):
        review = gate().review_turn([call("get_order_details")] * 3, SessionState.VERIFIED)
        assert len(review.decisions) == 3

    def test_the_rule_has_its_own_switch(self):
        # It must survive gating being ablated, or the two effects cannot be
        # attributed separately.
        g = gate(gating_enabled=False)
        review = g.review_turn(
            [call("get_order_details"), call("get_product_details")], SessionState.VERIFIED
        )
        assert review.allowed_calls == ()

    def test_disabling_the_rule_lets_parallel_calls_through(self):
        g = gate(parallel_call_block_enabled=False)
        review = g.review_turn(
            [call("get_order_details"), call("get_product_details")], SessionState.VERIFIED
        )
        assert len(review.allowed_calls) == 2

    def test_an_empty_turn_is_not_refused(self):
        assert gate().review_turn([], SessionState.VERIFIED).refused is False


class TestCatalogue:
    def test_duplicate_tools_are_rejected(self):
        spec = ToolSpec(name="dup", kind=ToolKind.READ)
        with pytest.raises(ValueError, match="duplicate tool"):
            ToolGate([spec, spec], HarnessConfig.full())

    def test_an_empty_catalogue_offers_nothing(self):
        g = ToolGate([], HarnessConfig.full())
        assert g.visible_tools(SessionState.SCOPED) == frozenset()

    def test_clauses_are_attached_to_the_rule_that_cites_them(self):
        clause = ClauseRef(doc_id="retail/policy.md", clause="authentication")
        g = ToolGate(
            RETAIL_CATALOGUE,
            HarnessConfig.full(),
            clauses={GateRule.IDENTITY_FIRST: (clause,)},
        )
        d = g.review_call(call("get_user_details"), SessionState.UNVERIFIED)
        assert d.clauses == (clause,)

    def test_decisions_carry_no_clauses_when_none_were_supplied(self):
        assert gate().review_call(call("get_user_details"), SessionState.UNVERIFIED).clauses == ()


class TestVisibilityAndEnforcementAgree:
    def test_every_visible_tool_is_allowed_and_every_hidden_one_is_refused(self):
        # Prevention and enforcement must not disagree, or the counts reported
        # for "how often prevention leaked" would be meaningless.
        g = gate()
        for state in SessionState:
            visible = g.visible_tools(state)
            for spec in RETAIL_CATALOGUE:
                decision = g.review_call(call(spec.name), state)
                assert decision.allowed is (spec.name in visible)


class TestEveryRefusalSaysWhatToDoNext:
    """W4 v4: a refusal that says only why was followed by a reference action 30%
    of the time, against 48% for errors the environment itself returned."""

    def test_each_refusing_rule_carries_a_remedy(self):
        g = gate()
        refusals = [
            g.review_call(call("no_such_tool"), SessionState.VERIFIED),
            g.review_call(call("get_user_details"), SessionState.UNVERIFIED),
            g.review_call(call("cancel_pending_order"), SessionState.VERIFIED),
            g.review_call(call("get_user_details"), SessionState.TRANSFERRED),
            g.review_call(call("get_user_details"), SessionState.PENDING_APPROVAL),
            g.review_turn([call("get_user_details"), call("get_order_details")],
                          SessionState.SCOPED).decisions[0],
        ]
        assert {d.outcome for d in refusals} == {DecisionOutcome.DENY}
        assert len({d.rule for d in refusals}) == len(refusals)  # one per rule, no overlap
        for d in refusals:
            assert d.remedy, d.rule
            assert d.remedy[0].islower(), d.remedy  # a clause, appended after the reason

    def test_the_identity_remedy_names_the_lookups_that_exist(self):
        d = gate().review_call(call("get_user_details"), SessionState.UNVERIFIED)
        assert "find_user_id_by_email" in d.remedy

    def test_the_parallel_remedy_names_the_call_to_keep(self):
        review = gate().review_turn(
            [call("get_order_details"), call("get_product_details")], SessionState.SCOPED
        )
        assert "get_order_details" in review.decisions[0].remedy

    def test_an_allowed_call_has_no_remedy(self):
        assert gate().review_call(call("get_user_details"), SessionState.VERIFIED).remedy is None

