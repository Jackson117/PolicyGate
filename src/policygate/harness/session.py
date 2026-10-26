"""One conversation under the harness: the per-turn review protocol, framework-free.

The tau2 adapter used to own this protocol -- record what the world reported,
gate and validate what the model proposes, pause writes for approval, phrase
refusals, escalate when the budget runs out. None of that depends on tau2 or on
how the model is called, and a second caller (a LangGraph decision graph that
calls the model in its own node) needs exactly the same protocol. Two copies of
it would mean two harnesses, and a result measured under one would say nothing
about the other. So it lives here, once, and adapters only translate.

What stays with the caller:

* calling the model, and deciding when to ask it again;
* turning tool results into :class:`ExtractedFacts` -- domain knowledge, which
  the harness does not have;
* building whatever message type its framework speaks from the
  ``(call_id, text)`` refusals and the escalation tool name returned here.

Pydantic and the standard library only. See docs/constraints.md #2.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from policygate.harness.approval import ApprovalDecider
from policygate.harness.audit import AuditChain
from policygate.harness.config import HarnessConfig
from policygate.harness.gating import GateRule, ToolGate
from policygate.harness.state import DomainStateProfile, SessionState, SessionTracker, StateError
from policygate.harness.types import (
    ApprovalRequest,
    ApprovalStatus,
    AuditEvent,
    CheckResult,
    Decision,
    DecisionOutcome,
    ToolCall,
    ToolKind,
    ToolRole,
)
from policygate.harness.validation import KnownObject, Validator, observe


class ExtractedFacts(BaseModel):
    """What one tool result established.

    ``scope_targets`` is ``None`` when the result says nothing about what the
    session is acting on, which is different from saying it acts on nothing.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str | None = None
    scope_targets: tuple[str, ...] | None = None
    objects: tuple[KnownObject, ...] = ()


class Verdict(BaseModel):
    """The harness's answer to one proposed call."""

    model_config = ConfigDict(frozen=True)

    call: ToolCall
    decision: Decision
    checks: CheckResult

    @property
    def allowed(self) -> bool:
        return self.decision.outcome is DecisionOutcome.ALLOW


def all_allowed(verdicts: Sequence[Verdict]) -> bool:
    """Whether the proposal may go to the environment as it stands."""
    return all(v.allowed for v in verdicts)


class HarnessSession:
    """Everything the harness has established about one conversation, and the
    operations a turn performs on it.

    ``gate``, ``validator`` and ``approve`` are shared across sessions; the
    tracker, the audit chain, the known world and the pending execution are this
    conversation's own.
    """

    def __init__(
        self,
        *,
        gate: ToolGate,
        validator: Validator,
        config: HarnessConfig,
        approve: ApprovalDecider,
        session_id: str,
        clock: Callable[[], datetime],
        profile: DomainStateProfile,
        tracker: SessionTracker | None = None,
        chain: AuditChain | None = None,
    ) -> None:
        self.gate = gate
        self.validator = validator
        self.config = config
        self.approve = approve
        self.tracker = tracker if tracker is not None else SessionTracker(profile=profile)
        self.chain = chain if chain is not None else AuditChain(session_id, clock, config)
        self.world: dict[str, KnownObject] = {}
        self.pending_execution: ToolCall | None = None
        """The write handed to the environment, awaiting its result."""
        self._last_handed_off: str | None = None
        self._streak = 0
        """How many times in a row the call fingerprinted ``_last_handed_off`` was
        handed to the environment. Tracked whatever the switch says; only the
        loop guard reads it."""

    @property
    def state(self) -> SessionState:
        return self.tracker.state

    # --- what the model is shown ---------------------------------------------

    def offered(self, names: Iterable[str]) -> list[str]:
        """The subset of ``names`` whose schemas the model is shown, in order.

        Prevention, as opposed to enforcement. Without context pruning the model
        sees everything and the gate only refuses afterwards, which is a
        different mechanism with a different cost and is switched separately.
        With pruning but no gating, the model is shown only what fits the state,
        yet a call to a hidden tool is not refused -- prevention with no
        enforcement behind it.
        """
        names = list(names)
        if not self.config.context_pruning_enabled:
            return names
        visible = self.gate.state_tools(self.tracker.state)
        return [name for name in names if name in visible]

    # --- absorbing what the world reported -----------------------------------

    def absorb_result(
        self, result_id: str, *, error: bool, facts: ExtractedFacts | None
    ) -> None:
        """Record one tool result, and only then update what is known.

        ``facts`` is ``None`` when the result answers no call this session knows
        of. Facts come from tool results, never from the user's text -- the same
        rule the state machine applies to identity, and the reason an order id
        the user merely mentioned is not a known referent.
        """
        executed = self.pending_execution
        if executed is not None and executed.id == result_id:
            self.pending_execution = None
            if not error:
                self._record_execution(executed)

        if facts is None:
            return

        observe(self.world, facts.objects)
        try:
            if facts.user_id:
                self.tracker.record_identity(facts.user_id)
            if facts.scope_targets is not None:
                self.tracker.record_targets(facts.scope_targets)
        except StateError:
            # The session is bound to another subject, or already transferred.
            # Refusing to update is the correct outcome; the gate will refuse
            # whatever is proposed next for the same reason.
            pass

    def _record_execution(self, call: ToolCall) -> None:
        if self.config.approval_gate_enabled:
            try:
                self.tracker.record_execution(call)
            except StateError:
                pass  # nothing was authorised; the chain still records what ran
        self.chain.append(AuditEvent.TOOL_EXECUTED, tool_call=call, decision=None)

    # --- reviewing ---------------------------------------------------------

    def review(self, calls: Sequence[ToolCall]) -> tuple[Verdict, ...]:
        """Gate, validate and, where required, seek approval for one proposal."""
        turn = self.gate.review_turn(calls, self.tracker.state)

        verdicts: list[Verdict] = []
        for call, decision in zip(calls, turn.decisions, strict=True):
            checks = CheckResult.ok()
            if decision.outcome is DecisionOutcome.ALLOW:
                checks = self.validator.review(
                    call,
                    subject=self.tracker.facts.user_id,
                    world=self.world,
                )
                if not checks.passed:
                    decision = Decision(
                        outcome=DecisionOutcome.DENY,
                        rule="validation.refused",
                        reason="; ".join(f.message for f in checks.failures),
                        # On the decision too, so the audit chain records what the
                        # model was told to do, not only what it was refused for.
                        remedy="; ".join(
                            dict.fromkeys(f.remedy for f in checks.failures if f.remedy)
                        ) or None,
                    )
                elif self._repeats_too_often(call):
                    limit = self.config.loop_guards.max_repeated_actions
                    decision = Decision(
                        outcome=DecisionOutcome.DENY,
                        rule="loop_guard.repeated_action",
                        reason=(
                            f"{call.name!r} with these exact arguments has run {limit} times "
                            "in a row; nothing has happened since that could change its result"
                        ),
                        remedy=(
                            "use the result you already have, or take a different step: "
                            "ask the user, call another tool, or answer"
                        ),
                    )
                elif self._needs_approval(call):
                    decision = self._seek_approval(call, decision)

            self.chain.append(
                AuditEvent.TOOL_PROPOSED if decision.outcome is DecisionOutcome.ALLOW
                else AuditEvent.TOOL_DENIED,
                tool_call=call,
                decision=decision,
            )
            verdicts.append(Verdict(call=call, decision=decision, checks=checks))
        return tuple(verdicts)

    def _repeats_too_often(self, call: ToolCall) -> bool:
        if not self.config.loop_guards_enabled:
            return False
        return (
            call.fingerprint() == self._last_handed_off
            and self._streak >= self.config.loop_guards.max_repeated_actions
        )

    def _needs_approval(self, call: ToolCall) -> bool:
        if not self.config.approval_gate_enabled:
            return False
        spec = self.gate.catalogue.get(call.name)
        return (
            spec is not None
            and spec.kind in self.config.approval.require_for_kinds
            and spec.role not in self.config.approval.exempt_roles
        )

    def _seek_approval(self, call: ToolCall, allowed: Decision) -> Decision:
        """Pause the write, ask, and record the two numbers separately.

        Whether the gate fired is the harness's contribution; whether the answer
        was yes belongs to the approval policy. Reporting only the end-to-end
        score would credit the harness with the decider's work, so both events
        go into the chain.
        """
        request = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            tool_call=call,
            reason=f"{call.name} changes stored state",
        )
        self.tracker.open_approval(request.request_id, call)
        self.chain.append(AuditEvent.APPROVAL_REQUESTED, tool_call=call, decision=allowed)

        answer = self.approve(request)
        self.tracker.resolve_approval(answer.status is ApprovalStatus.APPROVED)
        self.chain.append(AuditEvent.APPROVAL_RESOLVED, tool_call=call, approval=answer)

        if answer.status is ApprovalStatus.APPROVED:
            return allowed
        return Decision(
            outcome=DecisionOutcome.DENY,
            rule="approval.rejected",
            reason=answer.reason or "the approval gate did not authorise this action",
        )

    # --- refusals and handoff ----------------------------------------------

    @staticmethod
    def refusals(verdicts: Sequence[Verdict]) -> list[tuple[str, str]]:
        """``(call_id, text)`` for every call in a proposal that did not go through.

        The caller returns each to the model as an errored tool result -- a shape
        it already handles. The remedy is a separate sentence rather than part of
        the reason: W4 v3 refusals said only why, and were followed by a
        reference action 30% of the time against 48% for errors the environment
        itself returned.
        """
        out: list[tuple[str, str]] = []
        for verdict in verdicts:
            decision, checks = verdict.decision, verdict.checks
            if decision.outcome is DecisionOutcome.ALLOW:
                text = "Not run: another call in the same turn was refused."
            else:
                remedies = [f.remedy for f in checks.failures if f.remedy]
                if decision.remedy:
                    remedies.insert(0, decision.remedy)
                text = (
                    f"Refused by the policy harness ({decision.rule}). {decision.reason} "
                    "This action did not run."
                )
                if remedies:
                    text += " Next: " + "; ".join(dict.fromkeys(remedies)) + "."
            out.append((verdict.call.id, text))
        return out

    def note_handoff(self, calls: Sequence[ToolCall]) -> None:
        """Remember what is about to be handed to the environment.

        The write, for the approval bookkeeping when its result arrives; and
        every call, for the loop guard's count of consecutive repeats.
        """
        for call in calls:
            fingerprint = call.fingerprint()
            if fingerprint == self._last_handed_off:
                self._streak += 1
            else:
                self._last_handed_off, self._streak = fingerprint, 1
            spec = self.gate.catalogue.get(call.name)
            if spec is not None and spec.kind is ToolKind.WRITE:
                self.pending_execution = call

    def escalate(self, attempts: int) -> tuple[str | None, Decision]:
        """Record that no acceptable action appeared; name the tool to hand off with.

        Under its own rule, never mistaken for the model choosing to transfer --
        that distinction is the difference between a failure the model avoided
        and one the harness caught. Returns the escalation tool -- ``None`` when
        the domain has none, and the caller must then end the turn in text --
        and the recorded decision, whose reason is the handoff summary.
        """
        escalation = next(
            (
                name
                for name, spec in self.gate.catalogue.items()
                if spec.role is ToolRole.ESCALATION
            ),
            None,
        )
        decision = Decision(
            outcome=DecisionOutcome.DENY,
            rule=GateRule.ONE_CALL_PER_TURN.value.rsplit(".", 1)[0] + ".budget_exhausted",
            reason=(
                f"no acceptable action after {attempts} attempts; "
                "escalating to a human agent"
            ),
        )
        self.chain.append(AuditEvent.TOOL_DENIED, decision=decision)
        return escalation, decision
