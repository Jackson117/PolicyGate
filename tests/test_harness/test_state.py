"""Tests for harness/state.py.

Grounded in the domains' policy text rather than invented. Retail requires an
identity lookup "even when the user already provides the user id" and allows one
user per conversation; airline requires neither and says the agent obtains the
user id from the user. Each rule below cites the behaviour it protects.

No LLM, no benchmark, no network.
"""

import pytest

from policygate.harness.state import (
    AIRLINE_PROFILE,
    ALLOWED_TRANSITIONS,
    RETAIL_PROFILE,
    DomainStateProfile,
    SessionFacts,
    SessionState,
    SessionTracker,
    StateError,
    derive_state,
    is_legal_transition,
)
from policygate.harness.types import ToolCall

W1 = ToolCall(id="c1", name="return_delivered_order_items", arguments={"order_id": "#W1"})
W2 = ToolCall(id="c2", name="return_delivered_order_items", arguments={"order_id": "#W2"})
CANCEL_W1 = ToolCall(id="c3", name="cancel_pending_order", arguments={"order_id": "#W1"})


def scoped_tracker(target: str = "#W1") -> SessionTracker:
    t = SessionTracker()
    t.record_identity("u1")
    t.record_targets([target])
    return t


class TestDeriveStateIsPure:
    def test_equal_facts_give_equal_state(self):
        a = SessionFacts(user_id="u1", candidate_targets=frozenset({"#W1"}))
        b = SessionFacts(user_id="u1", candidate_targets=frozenset({"#W1"}))
        assert derive_state(a, RETAIL_PROFILE) == derive_state(b, RETAIL_PROFILE)

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

    def test_transfer_outranks_everything(self):
        facts = SessionFacts(
            user_id="u1",
            candidate_targets=frozenset({"#W1"}),
            pending_approval_id="r1",
            pending_action_fingerprint=W1.fingerprint(),
            transferred=True,
        )
        assert derive_state(facts, RETAIL_PROFILE) is SessionState.TRANSFERRED

    def test_pending_approval_outranks_scope(self):
        facts = SessionFacts(
            user_id="u1",
            candidate_targets=frozenset({"#W1"}),
            pending_approval_id="r1",
            pending_action_fingerprint=W1.fingerprint(),
        )
        assert derive_state(facts, RETAIL_PROFILE) is SessionState.PENDING_APPROVAL


class TestDomainProfile:
    """Retail and airline genuinely differ at the front of the conversation."""

    def test_retail_starts_unverified(self):
        assert SessionTracker().state is SessionState.UNVERIFIED

    def test_airline_starts_verified_because_it_has_no_authentication_step(self):
        assert SessionTracker(profile=AIRLINE_PROFILE).state is SessionState.VERIFIED

    def test_airline_may_scope_without_an_identity_lookup(self):
        t = SessionTracker(profile=AIRLINE_PROFILE)
        assert t.record_targets(["ABC123"]) is SessionState.SCOPED

    def test_retail_may_not_scope_without_an_identity_lookup(self):
        with pytest.raises(StateError, match="before an identity"):
            SessionTracker().record_targets(["#W1"])

    def test_single_subject_lock_is_profile_driven(self):
        t = SessionTracker(profile=AIRLINE_PROFILE)
        t.record_identity("u1")
        assert t.record_identity("u2") is SessionState.VERIFIED  # airline: no lock

    def test_a_profile_can_mix_the_two_flags(self):
        p = DomainStateProfile(name="x", requires_identity_lookup=False,
                               single_subject_per_session=True)
        t = SessionTracker(profile=p)
        t.record_identity("u1")
        with pytest.raises(StateError, match="one user per conversation"):
            t.record_identity("u2")


class TestReferentScoping:
    """task 14: the agent returned items from a different order than the one
    under discussion. An ambiguous referent must not reach write range."""

    def test_one_candidate_is_scoped(self):
        assert scoped_tracker("#W5490111").state is SessionState.SCOPED

    def test_two_candidates_are_not_scoped(self):
        t = SessionTracker()
        t.record_identity("u1")
        assert t.record_targets(["#W5490111", "#W7387996"]) is SessionState.VERIFIED

    def test_no_candidate_is_not_scoped(self):
        t = SessionTracker()
        t.record_identity("u1")
        assert t.record_targets([]) is SessionState.VERIFIED

    def test_scope_can_narrow_then_widen_again(self):
        t = scoped_tracker()
        assert t.record_targets(["#W1", "#W2"]) is SessionState.VERIFIED

    def test_targets_are_replaced_not_accumulated(self):
        t = SessionTracker()
        t.record_identity("u1")
        t.record_targets(["#W1", "#W2"])
        t.record_targets(["#W2"])
        assert t.facts.candidate_targets == frozenset({"#W2"})


class TestIdentityRules:
    """retail policy.md: authentication first, one user per conversation."""

    def test_rebinding_to_another_user_is_refused(self):
        t = SessionTracker()
        t.record_identity("mia_garcia_4516")
        with pytest.raises(StateError, match="one user per conversation"):
            t.record_identity("yusuf_rossi_9620")

    def test_rebinding_to_the_same_user_is_harmless(self):
        t = SessionTracker()
        t.record_identity("u1")
        assert t.record_identity("u1") is SessionState.VERIFIED


class TestApprovalIsBoundToOneAction:
    """The gate reviews an action, not a session."""

    def test_opening_an_approval_freezes_the_session(self):
        assert scoped_tracker().open_approval("r1", W1) is SessionState.PENDING_APPROVAL

    def test_granting_authorises_the_reviewed_fingerprint(self):
        t = scoped_tracker()
        t.open_approval("r1", W1)
        assert t.resolve_approval(True) is SessionState.SCOPED
        assert t.facts.authorized_action == W1.fingerprint()

    def test_rejecting_authorises_nothing(self):
        t = scoped_tracker()
        t.open_approval("r1", W1)
        t.resolve_approval(False)
        assert t.facts.authorized_action is None

    def test_only_one_approval_at_a_time(self):
        t = scoped_tracker()
        t.open_approval("r1", W1)
        with pytest.raises(StateError, match="already pending"):
            t.open_approval("r2", W2)

    def test_resolving_nothing_is_an_error(self):
        with pytest.raises(StateError, match="no approval"):
            scoped_tracker().resolve_approval(True)


class TestExecutionSpendsTheAuthorisation:
    def test_an_unapproved_write_is_refused(self):
        with pytest.raises(StateError, match="no approval"):
            scoped_tracker().record_execution(W1)

    def test_approving_one_write_does_not_authorise_another_target(self):
        t = scoped_tracker()
        t.open_approval("r1", W1)
        t.resolve_approval(True)
        with pytest.raises(StateError, match="does not match the approved action"):
            t.record_execution(W2)

    def test_approving_one_write_does_not_authorise_another_tool(self):
        # "user switches from a return to a cancellation" -- same order, new action.
        t = scoped_tracker()
        t.open_approval("r1", W1)
        t.resolve_approval(True)
        with pytest.raises(StateError, match="does not match the approved action"):
            t.record_execution(CANCEL_W1)

    def test_the_approved_write_goes_through(self):
        t = scoped_tracker()
        t.open_approval("r1", W1)
        t.resolve_approval(True)
        assert t.record_execution(W1) is SessionState.SCOPED

    def test_an_authorisation_is_spent_once(self):
        t = scoped_tracker()
        t.open_approval("r1", W1)
        t.resolve_approval(True)
        t.record_execution(W1)
        with pytest.raises(StateError, match="no approval"):
            t.record_execution(W1)

    def test_several_writes_are_fine_if_each_is_approved(self):
        # task 28's passing trial made four returns; multiple writes are normal,
        # what must not be reusable is a single approval.
        t = scoped_tracker()
        for call in (W1, CANCEL_W1):
            t.open_approval(f"r-{call.id}", call)
            t.resolve_approval(True)
            t.record_execution(call)
        assert t.facts.authorized_action is None


class TestMovingScopeVoidsApproval:
    """Reviewing a write against one order and performing it against another is
    the machine version of task 14."""

    def test_changing_target_while_pending_voids_the_approval(self):
        t = scoped_tracker()
        t.open_approval("r1", W1)
        assert t.record_targets(["#W2"]) is SessionState.SCOPED
        assert t.facts.pending_approval_id is None
        assert t.facts.pending_action_fingerprint is None

    def test_voiding_is_recorded_so_the_audit_trail_shows_it(self):
        t = scoped_tracker()
        t.open_approval("r1", W1)
        t.record_targets(["#W2"])
        assert t.history[-1].trigger == "targets_updated_voiding_pending_approval"

    def test_changing_target_voids_a_granted_authorisation(self):
        t = scoped_tracker()
        t.open_approval("r1", W1)
        t.resolve_approval(True)
        t.record_targets(["#W2"])
        assert t.facts.authorized_action is None
        assert t.history[-1].trigger == "targets_updated_voiding_authorization"

    def test_a_voided_approval_cannot_be_executed(self):
        t = scoped_tracker()
        t.open_approval("r1", W1)
        t.resolve_approval(True)
        t.record_targets(["#W2"])
        with pytest.raises(StateError, match="no approval"):
            t.record_execution(W1)

    def test_setting_the_same_targets_again_leaves_the_approval_alone(self):
        t = scoped_tracker()
        t.open_approval("r1", W1)
        t.record_targets(["#W1"])
        assert t.facts.pending_approval_id == "r1"


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
        # Self-loops are handled here rather than listed in the table, which
        # only encodes changes.
        for s in SessionState:
            assert is_legal_transition(s, s)

    def test_cannot_jump_from_unverified_to_scoped(self):
        assert not is_legal_transition(SessionState.UNVERIFIED, SessionState.SCOPED)

    def test_every_state_has_an_entry(self):
        assert set(ALLOWED_TRANSITIONS) == set(SessionState)

    def test_transfer_is_reachable_from_every_live_state(self):
        for s in SessionState:
            if s is not SessionState.TRANSFERRED:
                assert SessionState.TRANSFERRED in ALLOWED_TRANSITIONS[s]


class TestHistoryRecordsFactChanges:
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

    def test_a_target_swap_is_recorded_even_though_the_label_does_not_move(self):
        # The defect this replaced: SCOPED -> SCOPED with a different order was
        # invisible, losing the most important fact of the turn.
        t = scoped_tracker()
        before = len(t.history)
        t.record_targets(["#W2"])
        assert len(t.history) == before + 1
        assert t.history[-1].changed_state is False
        assert t.history[-1].facts.candidate_targets == frozenset({"#W2"})

    def test_updates_that_change_no_fact_are_not_recorded(self):
        t = SessionTracker()
        t.record_identity("u1")
        t.record_identity("u1")
        t.record_targets([])
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


class TestFactsInvariants:
    def test_approval_fields_must_move_together(self):
        with pytest.raises(StateError, match="set or cleared together"):
            SessionFacts(pending_approval_id="r1")
