"""Tests for harness/state.py.

Grounded in the retail policy text rather than invented: identity must be
established by a lookup, only one user may be served per conversation, and a
transfer ends the agent's involvement. Each rule below cites the behaviour it
protects.

No LLM, no benchmark, no network.
"""

import pytest

from policygate.harness.state import (
    ALLOWED_TRANSITIONS,
    SessionFacts,
    SessionState,
    SessionTracker,
    StateError,
    derive_state,
    is_legal_transition,
)


class TestDeriveStateIsPure:
    def test_equal_facts_give_equal_state(self):
        a = SessionFacts(user_id="u1", candidate_targets=frozenset({"#W1"}))
        b = SessionFacts(user_id="u1", candidate_targets=frozenset({"#W1"}))
        assert derive_state(a) == derive_state(b) == SessionState.SCOPED

    def test_state_does_not_depend_on_how_the_facts_were_reached(self):
        # The whole point of gating: the route through the conversation, and so
        # the wording along it, cannot change the legal action set.
        direct = SessionTracker(SessionFacts(user_id="u1", candidate_targets=frozenset({"#W1"})))

        stepwise = SessionTracker()
        stepwise.record_identity("u1")
        stepwise.record_targets(["#W1", "#W2"])
        stepwise.record_targets(["#W1"])

        assert direct.state == stepwise.state
        assert direct.facts == stepwise.facts

    def test_no_identity_means_unverified(self):
        assert derive_state(SessionFacts()) is SessionState.UNVERIFIED

    def test_transfer_outranks_everything(self):
        facts = SessionFacts(
            user_id="u1",
            candidate_targets=frozenset({"#W1"}),
            pending_approval_id="r1",
            transferred=True,
        )
        assert derive_state(facts) is SessionState.TRANSFERRED

    def test_pending_approval_outranks_scope(self):
        facts = SessionFacts(
            user_id="u1", candidate_targets=frozenset({"#W1"}), pending_approval_id="r1"
        )
        assert derive_state(facts) is SessionState.PENDING_APPROVAL


class TestReferentScoping:
    """task 14: the agent returned items from a different order than the one
    under discussion. An ambiguous referent must not reach write range."""

    def test_one_candidate_is_scoped(self):
        t = SessionTracker()
        t.record_identity("u1")
        assert t.record_targets(["#W5490111"]) is SessionState.SCOPED

    def test_two_candidates_are_not_scoped(self):
        t = SessionTracker()
        t.record_identity("u1")
        assert t.record_targets(["#W5490111", "#W7387996"]) is SessionState.VERIFIED

    def test_no_candidate_is_not_scoped(self):
        t = SessionTracker()
        t.record_identity("u1")
        assert t.record_targets([]) is SessionState.VERIFIED

    def test_scope_can_narrow_then_widen_again(self):
        t = SessionTracker()
        t.record_identity("u1")
        assert t.record_targets(["#W1"]) is SessionState.SCOPED
        assert t.record_targets(["#W1", "#W2"]) is SessionState.VERIFIED

    def test_targets_are_replaced_not_accumulated(self):
        t = SessionTracker()
        t.record_identity("u1")
        t.record_targets(["#W1", "#W2"])
        t.record_targets(["#W2"])
        assert t.facts.candidate_targets == frozenset({"#W2"})


class TestIdentityRules:
    """policy.md: authentication first, and one user per conversation."""

    def test_cannot_scope_before_an_identity_exists(self):
        with pytest.raises(StateError, match="before an identity"):
            SessionFacts(candidate_targets=frozenset({"#W1"}))

    def test_tracker_refuses_to_scope_before_verifying(self):
        with pytest.raises(StateError):
            SessionTracker().record_targets(["#W1"])

    def test_rebinding_to_another_user_is_refused(self):
        t = SessionTracker()
        t.record_identity("mia_garcia_4516")
        with pytest.raises(StateError, match="one user per conversation"):
            t.record_identity("yusuf_rossi_9620")

    def test_rebinding_to_the_same_user_is_harmless(self):
        t = SessionTracker()
        t.record_identity("u1")
        assert t.record_identity("u1") is SessionState.VERIFIED


class TestApprovalGateHoldsOneWrite:
    def test_opening_an_approval_freezes_the_session(self):
        t = SessionTracker()
        t.record_identity("u1")
        t.record_targets(["#W1"])
        assert t.open_approval("r1") is SessionState.PENDING_APPROVAL

    def test_closing_restores_the_underlying_state(self):
        t = SessionTracker()
        t.record_identity("u1")
        t.record_targets(["#W1"])
        t.open_approval("r1")
        assert t.close_approval() is SessionState.SCOPED

    def test_only_one_approval_at_a_time(self):
        t = SessionTracker()
        t.record_identity("u1")
        t.record_targets(["#W1"])
        t.open_approval("r1")
        with pytest.raises(StateError, match="already pending"):
            t.open_approval("r2")

    def test_closing_nothing_is_an_error(self):
        with pytest.raises(StateError, match="no approval"):
            SessionTracker().close_approval()


class TestTransferIsTerminal:
    """task 12's correct outcome. Once handed over, the agent is done."""

    def test_transfer_is_reachable_even_before_verifying(self):
        assert SessionTracker().record_transfer() is SessionState.TRANSFERRED

    def test_nothing_may_be_recorded_after_a_transfer(self):
        t = SessionTracker()
        t.record_transfer()
        with pytest.raises(StateError, match="transferred to a human"):
            t.record_identity("u1")

    def test_no_transitions_lead_out_of_transferred(self):
        assert ALLOWED_TRANSITIONS[SessionState.TRANSFERRED] == frozenset()


class TestTransitionGraph:
    def test_staying_put_is_always_legal(self):
        for s in SessionState:
            assert is_legal_transition(s, s)

    def test_cannot_jump_from_unverified_to_scoped(self):
        assert not is_legal_transition(SessionState.UNVERIFIED, SessionState.SCOPED)

    def test_every_state_has_an_entry(self):
        assert set(ALLOWED_TRANSITIONS) == set(SessionState)

    def test_transfer_is_reachable_from_every_live_state(self):
        for s in SessionState:
            if s is SessionState.TRANSFERRED:
                continue
            assert SessionState.TRANSFERRED in ALLOWED_TRANSITIONS[s]


class TestHistory:
    def test_records_changes_in_order(self):
        t = SessionTracker()
        t.record_identity("u1")
        t.record_targets(["#W1"])
        t.record_transfer()
        assert [h.after for h in t.history] == [
            SessionState.VERIFIED,
            SessionState.SCOPED,
            SessionState.TRANSFERRED,
        ]

    def test_ignores_steps_that_change_nothing(self):
        t = SessionTracker()
        t.record_identity("u1")
        t.record_targets([])  # still VERIFIED
        assert len(t.history) == 1

    def test_each_entry_carries_the_facts_behind_it(self):
        t = SessionTracker()
        t.record_identity("u1")
        assert t.history[0].facts.user_id == "u1"
        assert t.history[0].trigger == "identity_resolved"

    def test_recorded_facts_are_immutable_snapshots(self):
        t = SessionTracker()
        t.record_identity("u1")
        snapshot = t.history[0].facts
        t.record_targets(["#W1"])
        assert snapshot.candidate_targets == frozenset()
