"""A tau2 agent that proposes through the harness instead of straight to the world.

tau2 owns the outer loop and the environment executes tools, so the only place
the harness can act is between the model emitting a call and this adapter
returning an ``AssistantMessage``. Everything happens in that gap.

**Refusals never reach the user.** tau2 requires an assistant message to carry
either text or tool calls, never both, so a refusal expressed as text would be
delivered to the user simulator -- exposing harness internals and spending the
turn. Instead the refusal is appended to the model's own history as a
``ToolMessage`` with ``error=True`` and the model is asked again, inside this
method, invisibly to the orchestrator. The shape is one the model already
handles: W1 recorded 52 real tool errors and the model adjusted to them.

A refusal is not a tool error, though, and the audit chain keeps them apart:
``TOOL_DENIED`` means the call never ran, ``TOOL_EXECUTED`` means the
environment carried it out. Conflating them would make the harness look like the
cause of failures it actually prevented.

The internal loop is bounded. If no acceptable action appears within the budget
the agent escalates, which is what retail's policy prescribes for a request that
"cannot be handled within the scope of your actions" -- a proposal the harness
cannot accept after repeated attempts is exactly that. The escalation is
recorded under its own rule so it is never mistaken for the model choosing to
transfer.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
)
from tau2.data_model.message import ToolCall as Tau2ToolCall
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

from policygate.adapters.tool_mapping import ExtractedFacts, to_harness_call
from policygate.harness.approval import ApprovalDecider, auto_approve
from policygate.harness.audit import AuditChain
from policygate.harness.config import HarnessConfig
from policygate.harness.gating import GateRule, ToolGate
from policygate.harness.state import (
    RETAIL_PROFILE,
    DomainStateProfile,
    SessionTracker,
    StateError,
)
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
from policygate.harness.validation import DomainConstraints, KnownObject, Validator

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Some proposals will come back refused, with a reason. A refusal means the action
did not happen. Read the reason, fix what it names, and try again.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

FactExtractor = Callable[..., ExtractedFacts]


class HarnessAgentState:
    """Conversation plus everything the harness has established.

    A plain object rather than a pydantic model: it holds the tracker and the
    audit chain, which are stateful and are not meant to round-trip through
    serialisation.
    """

    def __init__(
        self,
        system_messages: list[SystemMessage],
        messages: list[APICompatibleMessage],
        tracker: SessionTracker,
        chain: AuditChain,
    ) -> None:
        self.system_messages = system_messages
        self.messages = messages
        self.tracker = tracker
        self.chain = chain
        self.world: dict[str, KnownObject] = {}
        self.pending_execution: ToolCall | None = None
        """The call handed to the environment, awaiting its result."""


class HarnessAgent(HalfDuplexAgent[HarnessAgentState]):
    """tau2's agent contract, with the harness in the proposal path."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        catalogue: Sequence[Any],
        constraints: DomainConstraints,
        extract_facts: FactExtractor,
        config: HarnessConfig,
        *,
        llm_args: dict | None = None,
        profile: DomainStateProfile = RETAIL_PROFILE,
        approve: ApprovalDecider = auto_approve,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        session_id: str | None = None,
        max_internal_attempts: int = 3,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = llm_args or {}
        self.config = config
        self.profile = profile
        self.approve = approve
        self.clock = clock
        self.session_id = session_id or str(uuid.uuid4())
        self.max_internal_attempts = max_internal_attempts
        self.extract_facts = extract_facts
        self.gate = ToolGate(catalogue, config)
        self.validator = Validator(constraints, config)
        self._tools_by_name = {t.name: t for t in tools}

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )

    def get_init_state(self, message_history: list[Message] | None = None) -> HarnessAgentState:
        return HarnessAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history or []),
            tracker=SessionTracker(profile=self.profile),
            chain=AuditChain(self.session_id, self.clock, self.config),
        )

    # --- tau2 entry point --------------------------------------------------

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: HarnessAgentState
    ) -> tuple[AssistantMessage, HarnessAgentState]:
        self._absorb(message, state)

        for _ in range(self.max_internal_attempts):
            response = self._propose(state)
            state.messages.append(response)

            if not response.tool_calls:
                return response, state  # a message to the user: nothing to gate

            verdicts = self._review(response.tool_calls, state)
            if all(d.outcome is DecisionOutcome.ALLOW for _, d, _ in verdicts):
                self._note_handoff(response, state)
                return response, state

            self._feed_back_refusals(verdicts, state)

        return self._escalate(state)

    # --- absorbing what the world reported ---------------------------------

    def _absorb(self, message: ValidAgentInputMessage, state: HarnessAgentState) -> None:
        """Record results, and only then update what is known.

        Facts come from tool results, never from the user's text -- the same rule
        the state machine applies to identity, and the reason an order id the
        user merely mentioned is not a known referent.
        """
        tool_messages: list[ToolMessage] = []
        if isinstance(message, MultiToolMessage):
            tool_messages = list(message.tool_messages)
            state.messages.extend(tool_messages)
        else:
            state.messages.append(message)
            if isinstance(message, ToolMessage):
                tool_messages = [message]

        for tool_message in tool_messages:
            self._absorb_result(tool_message, state)

    def _absorb_result(self, result: ToolMessage, state: HarnessAgentState) -> None:
        executed = state.pending_execution
        if executed is not None and executed.id == result.id:
            state.pending_execution = None
            if not result.error:
                self._record_execution(executed, state)

        name = self._tool_name_for(result, state)
        if name is None:
            return
        facts = self.extract_facts(name, result.content, error=result.error)

        if facts.objects:
            for obj in facts.objects:
                existing = state.world.get(obj.id)
                # Later observations win: a cancelled order really is cancelled
                # now, and the earlier reading is history, not a contradiction.
                state.world[obj.id] = obj if existing is None else obj
        try:
            if facts.user_id:
                state.tracker.record_identity(facts.user_id)
            if facts.scope_targets is not None:
                state.tracker.record_targets(facts.scope_targets)
        except StateError:
            # The session is bound to another subject, or already transferred.
            # Refusing to update is the correct outcome; the gate will refuse
            # whatever is proposed next for the same reason.
            pass

    def _tool_name_for(self, result: ToolMessage, state: HarnessAgentState) -> str | None:
        for message in reversed(state.messages):
            for call in getattr(message, "tool_calls", None) or []:
                if call.id == result.id:
                    return call.name
        return None

    def _record_execution(self, call: ToolCall, state: HarnessAgentState) -> None:
        if self.config.approval_gate_enabled:
            try:
                state.tracker.record_execution(call)
            except StateError:
                pass  # nothing was authorised; the chain still records what ran
        state.chain.append(AuditEvent.TOOL_EXECUTED, tool_call=call, decision=None)

    # --- proposing ---------------------------------------------------------

    def _propose(self, state: HarnessAgentState) -> AssistantMessage:
        return generate(
            model=self.llm,
            tools=self._offered_tools(state),
            messages=state.system_messages + state.messages,
            call_name="agent_response",
            **self.llm_args,
        )

    def _offered_tools(self, state: HarnessAgentState) -> list[Tool]:
        """The schemas the model is shown. Prevention, as opposed to enforcement.

        Without context pruning the model sees everything and the gate only
        refuses afterwards, which is a different mechanism with a different cost
        and is switched separately.
        """
        if not self.config.context_pruning_enabled:
            return list(self._tools_by_name.values())
        visible = self.gate.visible_tools(state.tracker.state)
        return [t for name, t in self._tools_by_name.items() if name in visible]

    # --- reviewing ---------------------------------------------------------

    def _review(
        self, calls: Sequence[Tau2ToolCall], state: HarnessAgentState
    ) -> list[tuple[Tau2ToolCall, Decision, CheckResult]]:
        harness_calls = [to_harness_call(c) for c in calls]
        turn = self.gate.review_turn(harness_calls, state.tracker.state)

        verdicts: list[tuple[Tau2ToolCall, Decision, CheckResult]] = []
        for original, call, decision in zip(calls, harness_calls, turn.decisions, strict=True):
            checks = CheckResult.ok()
            if decision.outcome is DecisionOutcome.ALLOW:
                checks = self.validator.review(
                    call,
                    scoped_target=self._scoped_target(state),
                    world=state.world,
                )
                if not checks.passed:
                    decision = Decision(
                        outcome=DecisionOutcome.DENY,
                        rule="validation.refused",
                        reason="; ".join(f.message for f in checks.failures),
                    )
                elif self._needs_approval(call, state):
                    decision = self._seek_approval(call, decision, state)

            state.chain.append(
                AuditEvent.TOOL_PROPOSED if decision.outcome is DecisionOutcome.ALLOW
                else AuditEvent.TOOL_DENIED,
                tool_call=call,
                decision=decision,
            )
            verdicts.append((original, decision, checks))
        return verdicts

    def _scoped_target(self, state: HarnessAgentState) -> str | None:
        targets = state.tracker.facts.candidate_targets
        return next(iter(targets)) if len(targets) == 1 else None

    def _needs_approval(self, call: ToolCall, state: HarnessAgentState) -> bool:
        if not self.config.approval_gate_enabled:
            return False
        spec = self.gate.catalogue.get(call.name)
        return spec is not None and spec.kind in self.config.approval.require_for_kinds

    def _seek_approval(
        self, call: ToolCall, allowed: Decision, state: HarnessAgentState
    ) -> Decision:
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
        state.tracker.open_approval(request.request_id, call)
        state.chain.append(AuditEvent.APPROVAL_REQUESTED, tool_call=call, decision=allowed)

        answer = self.approve(request)
        state.tracker.resolve_approval(answer.status is ApprovalStatus.APPROVED)
        state.chain.append(AuditEvent.APPROVAL_RESOLVED, tool_call=call, approval=answer)

        if answer.status is ApprovalStatus.APPROVED:
            return allowed
        return Decision(
            outcome=DecisionOutcome.DENY,
            rule="approval.rejected",
            reason=answer.reason or "the approval gate did not authorise this action",
        )

    # --- refusals and handoff ----------------------------------------------

    def _feed_back_refusals(
        self,
        verdicts: Sequence[tuple[Tau2ToolCall, Decision, CheckResult]],
        state: HarnessAgentState,
    ) -> None:
        """Return each refusal to the model in the shape it already understands."""
        for call, decision, _ in verdicts:
            text = (
                f"Refused by the policy harness ({decision.rule}). {decision.reason} "
                "This action did not run."
                if decision.outcome is not DecisionOutcome.ALLOW
                else "Not run: another call in the same turn was refused."
            )
            state.messages.append(
                ToolMessage(
                    id=call.id,
                    role="tool",
                    content=text,
                    requestor="assistant",
                    error=True,
                )
            )

    def _note_handoff(self, response: AssistantMessage, state: HarnessAgentState) -> None:
        """Remember the write we are about to hand to the environment."""
        for call in response.tool_calls or []:
            harness_call = to_harness_call(call)
            spec = self.gate.catalogue.get(harness_call.name)
            if spec is not None and spec.kind is ToolKind.WRITE:
                state.pending_execution = harness_call

    def _escalate(self, state: HarnessAgentState) -> tuple[AssistantMessage, HarnessAgentState]:
        """Hand to a human after the budget is spent.

        Under its own rule, never mistaken for the model choosing to transfer --
        that distinction is the difference between a failure the model avoided
        and one the harness caught.
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
                f"no acceptable action after {self.max_internal_attempts} attempts; "
                "escalating to a human agent"
            ),
        )
        state.chain.append(AuditEvent.TOOL_DENIED, decision=decision)

        if escalation is None:
            return AssistantMessage(
                role="assistant",
                content="I am unable to complete this request. Please hold for a human agent.",
            ), state

        call = Tau2ToolCall(
            id=str(uuid.uuid4()),
            name=escalation,
            arguments={"summary": decision.reason},
        )
        message = AssistantMessage(role="assistant", content=None, tool_calls=[call])
        state.messages.append(message)
        return message, state
