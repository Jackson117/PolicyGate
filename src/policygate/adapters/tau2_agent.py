"""A tau2 agent that proposes through the harness instead of straight to the world.

tau2 owns the outer loop and the environment executes tools, so the only place
the harness can act is between the model emitting a call and this adapter
returning an ``AssistantMessage``. Everything happens in that gap.

**Refusals never reach the user.** A refusal returned as a text-only message
would be routed to the user simulator -- exposing harness internals and spending
the turn. (tau2's "text or tool calls, never both" rule is only checked when
``enforce_communication_protocol`` is set, which it is not by default; see
``eval/violations.py``.) Instead the refusal is appended to the model's own
history as a ``ToolMessage`` with ``error=True`` and the model is asked again,
inside this method, invisibly to the orchestrator. The shape is one the model
already handles: W1 recorded 52 real tool errors and the model adjusted to them.

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
from policygate.harness.gating import ToolGate
from policygate.harness.session import HarnessSession, Verdict, all_allowed
from policygate.harness.state import (
    RETAIL_PROFILE,
    DomainStateProfile,
    SessionTracker,
)
from policygate.harness.types import ToolCall
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

    A plain object rather than a pydantic model: the harness session holds the
    tracker and the audit chain, which are stateful and are not meant to
    round-trip through serialisation. ``tracker``, ``chain``, ``world`` and
    ``pending_execution`` read through to it.
    """

    def __init__(
        self,
        system_messages: list[SystemMessage],
        messages: list[APICompatibleMessage],
        session: HarnessSession,
    ) -> None:
        self.system_messages = system_messages
        self.messages = messages
        self.session = session

    @property
    def tracker(self) -> SessionTracker:
        return self.session.tracker

    @property
    def chain(self) -> AuditChain:
        return self.session.chain

    @property
    def world(self) -> dict[str, KnownObject]:
        return self.session.world

    @property
    def pending_execution(self) -> ToolCall | None:
        """The call handed to the environment, awaiting its result."""
        return self.session.pending_execution


class HarnessAgent(HalfDuplexAgent[HarnessAgentState]):
    """tau2's agent contract, with the harness in the proposal path.

    The protocol itself lives in :class:`~policygate.harness.session.HarnessSession`;
    this class calls the model and translates between tau2's messages and the
    harness's types.
    """

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
            session=HarnessSession(
                gate=self.gate,
                validator=self.validator,
                config=self.config,
                approve=self.approve,
                session_id=self.session_id,
                clock=self.clock,
                profile=self.profile,
            ),
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

            calls = [to_harness_call(c) for c in response.tool_calls]
            verdicts = state.session.review(calls)
            if all_allowed(verdicts):
                state.session.note_handoff(calls)
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

        for result in tool_messages:
            name = self._tool_name_for(result, state)
            facts = (
                None if name is None
                else self.extract_facts(name, result.content, error=result.error)
            )
            state.session.absorb_result(result.id, error=result.error, facts=facts)

    def _tool_name_for(self, result: ToolMessage, state: HarnessAgentState) -> str | None:
        for message in reversed(state.messages):
            for call in getattr(message, "tool_calls", None) or []:
                if call.id == result.id:
                    return call.name
        return None

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
        """The schemas the model is shown; see :meth:`HarnessSession.offered`."""
        return [self._tools_by_name[n] for n in state.session.offered(self._tools_by_name)]

    # --- refusals and handoff ----------------------------------------------

    def _feed_back_refusals(self, verdicts: Sequence[Verdict], state: HarnessAgentState) -> None:
        """Return each refusal to the model in the shape it already understands.

        A ``ToolMessage`` with ``error=True``, appended to the model's own history
        and never routed to the user; see the module docstring.
        """
        for call_id, text in HarnessSession.refusals(verdicts):
            state.messages.append(
                ToolMessage(
                    id=call_id,
                    role="tool",
                    content=text,
                    requestor="assistant",
                    error=True,
                )
            )

    def _escalate(self, state: HarnessAgentState) -> tuple[AssistantMessage, HarnessAgentState]:
        """Hand to a human after the budget is spent; see :meth:`HarnessSession.escalate`."""
        escalation, decision = state.session.escalate(self.max_internal_attempts)

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
