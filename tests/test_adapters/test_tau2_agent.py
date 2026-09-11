"""Tests for adapters/tau2_agent.py.

The LLM is replaced by a scripted stub, so these assert on the harness's
behaviour rather than on a model's: what the agent shows the model, what it
refuses, what it feeds back, and what it finally hands to tau2. No network, no
tokens.

The tau2 side is real -- the retail toolkit, the tool types, the message
classes -- because this adapter exists precisely to meet it.
"""

import json
from datetime import datetime

import pytest
from tau2.data_model.message import (
    AssistantMessage,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.message import ToolCall as Tau2ToolCall
from tau2.registry import registry
from tau2.utils.utils import DATA_DIR

from policygate.adapters import tau2_agent as agent_module
from policygate.adapters.tau2_agent import HarnessAgent
from policygate.adapters.tool_mapping import (
    RETAIL_ESCALATION,
    RETAIL_IDENTITY_LOOKUPS,
    retail_constraints,
    retail_facts,
    tool_specs,
)
from policygate.harness.config import HarnessConfig
from policygate.harness.state import SessionState
from policygate.harness.types import ApprovalDecision, ApprovalStatus, AuditEvent

CLOCK = lambda: datetime(2026, 1, 1, 12, 0, 0)  # noqa: E731


@pytest.fixture(scope="module")
def env():
    return registry.get_env_constructor("retail")()


@pytest.fixture(scope="module")
def retail_db():
    return json.loads((DATA_DIR / "tau2" / "domains" / "retail" / "db.json").read_text())


class ScriptedLLM:
    """Returns prepared assistant messages, one per call, and records what it saw."""

    def __init__(self, *responses: AssistantMessage) -> None:
        self.responses = list(responses)
        self.offered_tool_names: list[set[str]] = []
        self.seen_messages: list[list] = []

    def __call__(self, *, model, tools, messages, **kwargs) -> AssistantMessage:
        self.offered_tool_names.append({t.name for t in tools})
        self.seen_messages.append(list(messages))
        if not self.responses:
            raise AssertionError("the agent asked for more responses than were scripted")
        return self.responses.pop(0)


def says(text: str) -> AssistantMessage:
    return AssistantMessage(role="assistant", content=text)


def calls(name: str, call_id: str = "c1", **arguments) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[Tau2ToolCall(id=call_id, name=name, arguments=arguments)],
    )


def build(env, retail_db, llm: ScriptedLLM, monkeypatch, **overrides) -> HarnessAgent:
    monkeypatch.setattr(agent_module, "generate", llm)
    return HarnessAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm="scripted",
        catalogue=tool_specs(
            env.tools, identity_lookups=RETAIL_IDENTITY_LOOKUPS, escalation=RETAIL_ESCALATION
        ),
        constraints=retail_constraints(retail_db),
        extract_facts=retail_facts,
        config=HarnessConfig.full(**overrides),
        clock=CLOCK,
        session_id="s1",
    )


def tool_result(call_id: str, content: str, error: bool = False) -> ToolMessage:
    return ToolMessage(
        id=call_id, role="tool", content=content, requestor="assistant", error=error
    )


ORDER_DELIVERED = json.dumps({"order_id": "#W5490111", "status": "delivered", "user_id": "u1"})
ORDER_PENDING = json.dumps({"order_id": "#W7387996", "status": "pending", "user_id": "u1"})


def verified_scoped(agent, llm, order_json: str = ORDER_DELIVERED):
    """Drive the agent to VERIFIED then SCOPED using real tool results.

    Every turn asks the model for a proposal, so each one needs its own scripted
    response: look up the user, fetch the order, then speak.
    """
    order_id = json.loads(order_json)["order_id"]
    state = agent.get_init_state()

    llm.responses.append(calls("find_user_id_by_email", "id1", email="u@example.com"))
    agent.generate_next_message(UserMessage(role="user", content="hello"), state)

    llm.responses.append(calls("get_order_details", "o1", order_id=order_id))
    agent.generate_next_message(tool_result("id1", "u1"), state)

    llm.responses.append(says("What would you like to do with that order?"))
    agent.generate_next_message(tool_result("o1", order_json), state)
    return state


class TestToolVisibility:
    def test_an_unverified_agent_is_only_offered_lookups_and_escalation(
        self, env, retail_db, monkeypatch
    ):
        llm = ScriptedLLM(says("Could I take your email?"))
        agent = build(env, retail_db, llm, monkeypatch)
        agent.generate_next_message(
            UserMessage(role="user", content="hi"), agent.get_init_state()
        )
        assert llm.offered_tool_names[0] == {
            "find_user_id_by_email",
            "find_user_id_by_name_zip",
            "transfer_to_human_agents",
        }

    def test_pruning_can_be_switched_off_leaving_only_enforcement(
        self, env, retail_db, monkeypatch
    ):
        llm = ScriptedLLM(says("hello"))
        agent = build(env, retail_db, llm, monkeypatch, context_pruning_enabled=False)
        agent.generate_next_message(
            UserMessage(role="user", content="hi"), agent.get_init_state()
        )
        assert len(llm.offered_tool_names[0]) == len(env.get_tools())


class TestFactsComeFromToolResults:
    def test_an_identity_lookup_result_verifies_the_session(self, env, retail_db, monkeypatch):
        llm = ScriptedLLM(calls("find_user_id_by_email", "id1", email="u@example.com"), says("ok"))
        agent = build(env, retail_db, llm, monkeypatch)
        state = agent.get_init_state()
        agent.generate_next_message(UserMessage(role="user", content="hi"), state)
        assert state.tracker.state is SessionState.UNVERIFIED
        agent.generate_next_message(tool_result("id1", "u1"), state)
        assert state.tracker.state is SessionState.VERIFIED

    def test_a_failed_lookup_verifies_nothing(self, env, retail_db, monkeypatch):
        llm = ScriptedLLM(calls("find_user_id_by_email", "id1", email="x@example.com"), says("ok"))
        agent = build(env, retail_db, llm, monkeypatch)
        state = agent.get_init_state()
        agent.generate_next_message(UserMessage(role="user", content="hi"), state)
        agent.generate_next_message(tool_result("id1", "Error: user not found", error=True), state)
        assert state.tracker.state is SessionState.UNVERIFIED

    def test_fetching_an_order_scopes_the_session_and_records_its_status(
        self, env, retail_db, monkeypatch
    ):
        llm = ScriptedLLM()
        agent = build(env, retail_db, llm, monkeypatch)
        state = verified_scoped(agent, llm)
        assert state.tracker.state is SessionState.SCOPED
        assert state.world["#W5490111"].attributes["status"] == "delivered"


class TestRefusalsStayInsideTheAgent:
    def test_a_refused_call_is_retried_and_never_shown_to_the_user(
        self, env, retail_db, monkeypatch
    ):
        # First proposal reads a profile before anyone is authenticated; the
        # second asks for the email instead.
        llm = ScriptedLLM(
            calls("get_user_details", "bad", user_id="u1"),
            says("Could I take your email address?"),
        )
        agent = build(env, retail_db, llm, monkeypatch)
        message, state = agent.generate_next_message(
            UserMessage(role="user", content="hi"), agent.get_init_state()
        )
        assert message.content == "Could I take your email address?"
        assert message.tool_calls is None

    def test_the_refusal_reaches_the_model_as_a_tool_error(self, env, retail_db, monkeypatch):
        llm = ScriptedLLM(calls("get_user_details", "bad", user_id="u1"), says("sorry"))
        agent = build(env, retail_db, llm, monkeypatch)
        agent.generate_next_message(
            UserMessage(role="user", content="hi"), agent.get_init_state()
        )
        second_attempt = llm.seen_messages[1]
        refusal = second_attempt[-1]
        assert isinstance(refusal, ToolMessage)
        assert refusal.error is True
        assert "gating.identity_first" in refusal.content

    def test_a_refusal_is_recorded_as_denied_not_executed(self, env, retail_db, monkeypatch):
        llm = ScriptedLLM(calls("get_user_details", "bad", user_id="u1"), says("sorry"))
        agent = build(env, retail_db, llm, monkeypatch)
        state = agent.get_init_state()
        agent.generate_next_message(UserMessage(role="user", content="hi"), state)
        events = [r.event for r in state.chain.records]
        assert AuditEvent.TOOL_DENIED in events
        assert AuditEvent.TOOL_EXECUTED not in events


class TestValidationRefusals:
    def test_a_write_against_an_unknown_order_is_refused(self, env, retail_db, monkeypatch):
        llm = ScriptedLLM()
        agent = build(env, retail_db, llm, monkeypatch)
        state = verified_scoped(agent, llm)
        llm.responses.extend(
            [
                calls("cancel_pending_order", "w1", order_id="#12345"),
                says("Which order did you mean?"),
            ]
        )
        message, _ = agent.generate_next_message(
            UserMessage(role="user", content="cancel it"), state
        )
        assert message.content == "Which order did you mean?"
        denied = [r for r in state.chain.records if r.event is AuditEvent.TOOL_DENIED]
        assert denied[-1].decision.rule == "validation.refused"
        assert "#12345" in denied[-1].decision.reason

    def test_a_precondition_failure_is_refused(self, env, retail_db, monkeypatch):
        llm = ScriptedLLM()
        agent = build(env, retail_db, llm, monkeypatch)
        state = verified_scoped(agent, llm)  # #W5490111 is delivered
        llm.responses.extend(
            [
                calls("cancel_pending_order", "w1", order_id="#W5490111"),
                says("That order has already been delivered."),
            ]
        )
        agent.generate_next_message(UserMessage(role="user", content="cancel it"), state)
        denied = [r for r in state.chain.records if r.event is AuditEvent.TOOL_DENIED]
        assert "status" in denied[-1].decision.reason


class TestApprovalGate:
    def test_an_approved_write_is_handed_to_the_environment(self, env, retail_db, monkeypatch):
        llm = ScriptedLLM()
        agent = build(env, retail_db, llm, monkeypatch)
        state = verified_scoped(agent, llm, ORDER_PENDING)
        llm.responses.append(calls("cancel_pending_order", "w1", order_id="#W7387996"))
        message, _ = agent.generate_next_message(
            UserMessage(role="user", content="cancel it"), state
        )
        assert message.tool_calls[0].name == "cancel_pending_order"
        events = [r.event for r in state.chain.records]
        assert AuditEvent.APPROVAL_REQUESTED in events
        assert AuditEvent.APPROVAL_RESOLVED in events

    def test_a_rejected_write_is_refused_and_retried(self, env, retail_db, monkeypatch):
        def always_reject(request):
            return ApprovalDecision(
                request_id=request.request_id,
                status=ApprovalStatus.REJECTED,
                policy="auto_reject",
                reason="not authorised",
            )

        llm = ScriptedLLM()
        agent = build(env, retail_db, llm, monkeypatch)
        agent.approve = always_reject
        state = verified_scoped(agent, llm, ORDER_PENDING)
        llm.responses.extend(
            [
                calls("cancel_pending_order", "w1", order_id="#W7387996"),
                says("I cannot cancel that for you."),
            ]
        )
        message, _ = agent.generate_next_message(
            UserMessage(role="user", content="cancel it"), state
        )
        assert message.content == "I cannot cancel that for you."

    def test_the_gate_can_be_switched_off(self, env, retail_db, monkeypatch):
        llm = ScriptedLLM()
        agent = build(env, retail_db, llm, monkeypatch, approval_gate_enabled=False)
        state = verified_scoped(agent, llm, ORDER_PENDING)
        llm.responses.append(calls("cancel_pending_order", "w1", order_id="#W7387996"))
        agent.generate_next_message(UserMessage(role="user", content="cancel it"), state)
        assert AuditEvent.APPROVAL_REQUESTED not in [r.event for r in state.chain.records]


class TestBudgetExhaustion:
    def test_the_agent_escalates_rather_than_looping(self, env, retail_db, monkeypatch):
        llm = ScriptedLLM(*[calls("get_user_details", f"bad{i}", user_id="u1") for i in range(3)])
        agent = build(env, retail_db, llm, monkeypatch)
        message, state = agent.generate_next_message(
            UserMessage(role="user", content="hi"), agent.get_init_state()
        )
        assert message.tool_calls[0].name == "transfer_to_human_agents"

    def test_the_forced_escalation_is_recorded_under_its_own_rule(
        self, env, retail_db, monkeypatch
    ):
        # It must never be mistaken for the model choosing to transfer: that is
        # the difference between a failure avoided and one the harness caught.
        llm = ScriptedLLM(*[calls("get_user_details", f"bad{i}", user_id="u1") for i in range(3)])
        agent = build(env, retail_db, llm, monkeypatch)
        _, state = agent.generate_next_message(
            UserMessage(role="user", content="hi"), agent.get_init_state()
        )
        assert state.chain.records[-1].decision.rule.endswith("budget_exhausted")

    def test_the_attempt_budget_is_respected(self, env, retail_db, monkeypatch):
        llm = ScriptedLLM(*[calls("get_user_details", f"bad{i}", user_id="u1") for i in range(3)])
        agent = build(env, retail_db, llm, monkeypatch)
        agent.generate_next_message(
            UserMessage(role="user", content="hi"), agent.get_init_state()
        )
        assert len(llm.seen_messages) == agent.max_internal_attempts


class TestAuditChain:
    def test_the_chain_stays_intact_across_a_conversation(self, env, retail_db, monkeypatch):
        llm = ScriptedLLM()
        agent = build(env, retail_db, llm, monkeypatch)
        state = verified_scoped(agent, llm, ORDER_PENDING)
        llm.responses.append(calls("cancel_pending_order", "w1", order_id="#W7387996"))
        agent.generate_next_message(UserMessage(role="user", content="cancel it"), state)
        assert state.chain.verify().intact is True

    def test_an_execution_is_recorded_when_the_result_comes_back(
        self, env, retail_db, monkeypatch
    ):
        llm = ScriptedLLM()
        agent = build(env, retail_db, llm, monkeypatch)
        state = verified_scoped(agent, llm, ORDER_PENDING)
        llm.responses.extend(
            [calls("cancel_pending_order", "w1", order_id="#W7387996"), says("done")]
        )
        agent.generate_next_message(UserMessage(role="user", content="cancel it"), state)
        agent.generate_next_message(
            tool_result("w1", json.dumps({"order_id": "#W7387996", "status": "cancelled"})), state
        )
        assert AuditEvent.TOOL_EXECUTED in [r.event for r in state.chain.records]

    def test_a_failed_execution_is_not_recorded_as_executed(self, env, retail_db, monkeypatch):
        llm = ScriptedLLM()
        agent = build(env, retail_db, llm, monkeypatch)
        state = verified_scoped(agent, llm, ORDER_PENDING)
        llm.responses.extend(
            [calls("cancel_pending_order", "w1", order_id="#W7387996"), says("hm")]
        )
        agent.generate_next_message(UserMessage(role="user", content="cancel it"), state)
        agent.generate_next_message(tool_result("w1", "Error: Order not found", error=True), state)
        assert AuditEvent.TOOL_EXECUTED not in [r.event for r in state.chain.records]

    def test_the_baseline_arm_records_nothing(self, env, retail_db, monkeypatch):
        llm = ScriptedLLM(calls("get_user_details", "x", user_id="u1"))
        monkeypatch.setattr(agent_module, "generate", llm)
        agent = HarnessAgent(
            tools=env.get_tools(),
            domain_policy=env.get_policy(),
            llm="scripted",
            catalogue=tool_specs(
                env.tools, identity_lookups=RETAIL_IDENTITY_LOOKUPS, escalation=RETAIL_ESCALATION
            ),
            constraints=retail_constraints(retail_db),
            extract_facts=retail_facts,
            config=HarnessConfig.baseline(),
            clock=CLOCK,
        )
        message, state = agent.generate_next_message(
            UserMessage(role="user", content="hi"), agent.get_init_state()
        )
        # Nothing gated, nothing validated, nothing recorded.
        assert message.tool_calls[0].name == "get_user_details"
        assert state.chain.records == ()
