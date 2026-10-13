"""Tests for harness/session.py.

The per-turn protocol driven directly, the way a caller that owns its own model
call (a LangGraph decision graph, say) would drive it. No tau2, no LLM, no
network: this file runs in CI's harness-only job, which is the evidence that the
protocol needs no agent framework.
"""

from datetime import datetime

from policygate.harness.approval import auto_approve, auto_reject
from policygate.harness.config import HarnessConfig
from policygate.harness.gating import ToolGate
from policygate.harness.session import ExtractedFacts, HarnessSession, all_allowed
from policygate.harness.state import RETAIL_PROFILE, SessionState
from policygate.harness.types import AuditEvent, ToolCall, ToolKind, ToolRole, ToolSpec
from policygate.harness.validation import DomainConstraints, Validator

CLOCK = lambda: datetime(2026, 1, 1, 12, 0, 0)  # noqa: E731

CATALOGUE = (
    ToolSpec(name="find_user_id_by_email", kind=ToolKind.READ, role=ToolRole.IDENTITY_LOOKUP),
    ToolSpec(name="get_order_details", kind=ToolKind.READ),
    ToolSpec(name="cancel_pending_order", kind=ToolKind.WRITE),
    ToolSpec(name="transfer_to_human_agents", kind=ToolKind.GENERIC, role=ToolRole.ESCALATION),
)


def session(approve=auto_approve, **overrides) -> HarnessSession:
    config = HarnessConfig.full(**overrides)
    return HarnessSession(
        gate=ToolGate(CATALOGUE, config),
        validator=Validator(DomainConstraints(), config),
        config=config,
        approve=approve,
        session_id="s1",
        clock=CLOCK,
        profile=RETAIL_PROFILE,
    )


def call(name: str, id: str = "c1", **args) -> ToolCall:
    return ToolCall(id=id, name=name, arguments=args)


def events(s: HarnessSession) -> list[AuditEvent]:
    return [r.event for r in s.chain.records]


class TestOffered:
    def test_pruning_keeps_only_what_fits_the_state_in_the_given_order(self):
        s = session(context_pruning_enabled=True)
        names = [spec.name for spec in reversed(CATALOGUE)]
        assert s.offered(names) == ["transfer_to_human_agents", "find_user_id_by_email"]

    def test_without_pruning_everything_is_offered(self):
        s = session(context_pruning_enabled=False)
        names = [spec.name for spec in CATALOGUE]
        assert s.offered(names) == names


class TestReview:
    def test_an_allowed_call_is_recorded_as_proposed(self):
        s = session()
        verdicts = s.review([call("find_user_id_by_email", email="a@b.c")])
        assert all_allowed(verdicts)
        assert events(s) == [AuditEvent.TOOL_PROPOSED]

    def test_a_refused_call_comes_back_with_text_for_the_model(self):
        s = session()
        verdicts = s.review([call("get_order_details", order_id="#W1")])
        assert not all_allowed(verdicts)
        [(call_id, text)] = HarnessSession.refusals(verdicts)
        assert call_id == "c1"
        assert text.startswith("Refused by the policy harness (gating.identity_first).")
        assert "Next: " in text
        assert events(s) == [AuditEvent.TOOL_DENIED]

    def test_the_innocent_call_in_a_refused_turn_is_told_why_it_did_not_run(self):
        s = session(parallel_call_block_enabled=False)
        verdicts = s.review([
            call("find_user_id_by_email", id="c1", email="a@b.c"),
            call("get_order_details", id="c2", order_id="#W1"),
        ])
        texts = dict(HarnessSession.refusals(verdicts))
        assert texts["c1"] == "Not run: another call in the same turn was refused."


class TestTheWritePath:
    def _scoped(self, s: HarnessSession) -> None:
        s.absorb_result("r1", error=False, facts=ExtractedFacts(user_id="u1"))
        s.absorb_result("r2", error=False, facts=ExtractedFacts(scope_targets=("#W1",)))
        assert s.state is SessionState.SCOPED

    def test_facts_from_results_move_the_state(self):
        s = session()
        assert s.state is SessionState.UNVERIFIED
        self._scoped(s)

    def test_an_approved_write_is_handed_off_and_its_result_spends_the_approval(self):
        s = session(approval_gate_enabled=True)
        self._scoped(s)
        write = call("cancel_pending_order", id="w1", order_id="#W1", reason="no longer needed")

        verdicts = s.review([write])
        assert all_allowed(verdicts)
        s.note_handoff([write])
        assert s.pending_execution == write
        assert s.tracker.facts.authorized_action == write.fingerprint()

        s.absorb_result("w1", error=False, facts=ExtractedFacts())
        assert s.pending_execution is None
        assert s.tracker.facts.authorized_action is None
        assert events(s) == [
            AuditEvent.APPROVAL_REQUESTED,
            AuditEvent.APPROVAL_RESOLVED,
            AuditEvent.TOOL_PROPOSED,
            AuditEvent.TOOL_EXECUTED,
        ]

    def test_a_rejected_write_is_refused_under_the_approval_rule(self):
        s = session(approve=auto_reject, approval_gate_enabled=True)
        self._scoped(s)
        [verdict] = s.review([call("cancel_pending_order", order_id="#W1", reason="x")])
        assert verdict.decision.rule == "approval.rejected"

    def test_a_failed_write_records_no_execution(self):
        s = session(approval_gate_enabled=True)
        self._scoped(s)
        write = call("cancel_pending_order", id="w1", order_id="#W1", reason="x")
        s.review([write])
        s.note_handoff([write])
        s.absorb_result("w1", error=True, facts=ExtractedFacts())
        assert AuditEvent.TOOL_EXECUTED not in events(s)

    def test_a_result_for_no_known_call_changes_nothing(self):
        s = session()
        s.absorb_result("unknown", error=False, facts=None)
        assert s.state is SessionState.UNVERIFIED
        assert s.chain.records == ()


class TestEscalate:
    def test_names_the_escalation_tool_and_records_the_budget_rule(self):
        s = session()
        tool, decision = s.escalate(3)
        assert tool == "transfer_to_human_agents"
        assert decision.rule == "gating.budget_exhausted"
        assert "after 3 attempts" in decision.reason
        assert events(s) == [AuditEvent.TOOL_DENIED]


class TestIdentityCommit:
    CATALOGUE = (
        ToolSpec(name="get_user_information_by_email", kind=ToolKind.READ,
                 role=ToolRole.IDENTITY_LOOKUP),
        ToolSpec(name="log_verification", kind=ToolKind.WRITE, role=ToolRole.IDENTITY_COMMIT),
        ToolSpec(name="transfer_to_human_agents", kind=ToolKind.GENERIC,
                 role=ToolRole.ESCALATION),
    )

    def _session(self, approve, **approval) -> HarnessSession:
        from policygate.harness.config import ApprovalConfig

        config = HarnessConfig.full(
            approval_gate_enabled=True, approval=ApprovalConfig(**approval)
        )
        return HarnessSession(
            gate=ToolGate(self.CATALOGUE, config),
            validator=Validator(DomainConstraints(), config),
            config=config,
            approve=approve,
            session_id="s1",
            clock=CLOCK,
            profile=RETAIL_PROFILE,
        )

    def test_the_commit_does_not_pause_for_approval(self):
        # Under auto_reject a gated commit would make verification impossible.
        s = self._session(auto_reject)
        commit = call("log_verification", id="v1", user_id="u1")
        assert all_allowed(s.review([commit]))
        assert AuditEvent.APPROVAL_REQUESTED not in events(s)

    def test_the_commit_result_settles_identity_and_its_execution_is_recorded(self):
        s = self._session(auto_reject)
        commit = call("log_verification", id="v1", user_id="u1")
        s.review([commit])
        s.note_handoff([commit])
        s.absorb_result("v1", error=False, facts=ExtractedFacts(user_id="u1"))
        assert s.state is SessionState.VERIFIED
        assert events(s)[-1] is AuditEvent.TOOL_EXECUTED

    def test_the_exemption_can_be_switched_off(self):
        s = self._session(auto_reject, exempt_roles=())
        [verdict] = s.review([call("log_verification", user_id="u1")])
        assert verdict.decision.rule == "approval.rejected"


class TestLoopGuard:
    def _session(self, enabled=True, limit=3):
        from policygate.harness.config import LoopGuardConfig

        return session(
            gating_enabled=False,
            loop_guards_enabled=enabled,
            loop_guards=LoopGuardConfig(max_repeated_actions=limit),
        )

    def _run(self, s, c):
        """Propose ``c``; hand it off if allowed. Returns the verdict."""
        [verdict] = s.review([c])
        if verdict.allowed:
            s.note_handoff([c])
        return verdict

    def test_the_call_after_the_limit_is_refused_with_a_remedy(self):
        s = self._session(limit=3)
        lookup = call("get_order_details", order_id="#W1")
        assert all(self._run(s, lookup).allowed for _ in range(3))
        refused = self._run(s, lookup)
        assert refused.decision.rule == "loop_guard.repeated_action"
        assert refused.decision.remedy
        assert events(s)[-1] is AuditEvent.TOOL_DENIED

    def test_a_different_call_resets_the_count(self):
        s = self._session(limit=2)
        a, b = call("get_order_details", order_id="#W1"), call("get_order_details", order_id="#W2")
        for c in (a, a, b, a, a):
            assert self._run(s, c).allowed
        assert not self._run(s, a).allowed

    def test_different_arguments_are_a_different_call(self):
        s = self._session(limit=1)
        assert self._run(s, call("get_order_details", order_id="#W1")).allowed
        assert self._run(s, call("get_order_details", order_id="#W2")).allowed

    def test_a_refused_repeat_does_not_extend_the_count(self):
        s = self._session(limit=1)
        lookup = call("get_order_details", order_id="#W1")
        assert self._run(s, lookup).allowed
        assert not self._run(s, lookup).allowed
        assert self._run(s, call("get_order_details", order_id="#W2")).allowed
        assert self._run(s, lookup).allowed

    def test_switched_off_it_never_refuses(self):
        s = self._session(enabled=False, limit=1)
        lookup = call("get_order_details", order_id="#W1")
        assert all(self._run(s, lookup).allowed for _ in range(10))
