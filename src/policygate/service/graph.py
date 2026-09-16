"""The deployment as a LangGraph state machine, with a human in the approval path.

The benchmark adapter (adapters/tau2_agent.py) runs propose -> review -> approve
-> hand over inside one synchronous method, because tau2 owns its loop and an
approval there is a function call. A deployment cannot do that: a reviewer
answers minutes or days later, from another request, possibly on another
process. So the same harness steps become graph nodes, and the approval becomes
an ``interrupt`` that a checkpointer persists until someone resumes it.

The harness logic is the same modules W4 measured -- ToolGate, Validator,
SessionTracker, AuditChain -- rebuilt each step from the checkpointed facts.
Nothing here re-implements a rule.

Two properties the graph is built around:

* **An interrupted node re-runs from its first line on resume.** So
  ``await_approval`` does nothing before calling ``interrupt``; every side effect
  of the decision happens after it returns, exactly once.
* **State is plain JSON.** Messages, facts, known objects, audit records and the
  executed-call log are all dumps, so any checkpointer can store them and
  nothing depends on pickling library objects.
"""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage

from policygate.adapters.tool_mapping import to_harness_call
from policygate.harness.audit import AuditChain
from policygate.harness.state import SessionFacts, SessionTracker, StateError
from policygate.harness.types import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    AuditEvent,
    AuditRecord,
    Decision,
    DecisionOutcome,
    ToolRole,
)
from policygate.harness.validation import KnownObject, observe
from policygate.service.runtime import RetailRuntime, dump, to_message


class ServiceState(TypedDict, total=False):
    session_id: str
    messages: Annotated[list[dict[str, Any]], operator.add]
    audit: Annotated[list[dict[str, Any]], operator.add]
    executed: Annotated[list[dict[str, Any]], operator.add]
    facts: dict[str, Any]
    world: dict[str, dict[str, Any]]
    proposal: dict[str, Any] | None
    """The assistant message whose tool calls passed review, awaiting approval or execution."""

    pending_approval: dict[str, Any] | None
    attempts: int
    reply: str | None


ESCALATION_TEXT = (
    "I'm not able to complete this request within what I'm allowed to do. "
    "I'm handing you to a human agent."
)


class _Step:
    """The harness objects for one node, rebuilt from checkpointed state."""

    def __init__(self, runtime: RetailRuntime, state: ServiceState) -> None:
        self.runtime = runtime
        self.session_id = state["session_id"]
        self.tracker = SessionTracker(
            SessionFacts.model_validate(state.get("facts") or {}),
        )
        records = [AuditRecord.model_validate(r) for r in state.get("audit", [])]
        self.chain = AuditChain.resume(self.session_id, runtime.clock, runtime.config, records)
        self._audit_start = len(records)
        self.world = {
            k: KnownObject.model_validate(v) for k, v in (state.get("world") or {}).items()
        }

    def new_audit(self) -> list[dict[str, Any]]:
        return [r.model_dump(mode="json") for r in self.chain.records[self._audit_start :]]

    def snapshot(self) -> dict[str, Any]:
        return {
            "facts": self.tracker.facts.model_dump(mode="json"),
            "world": {k: v.model_dump(mode="json") for k, v in self.world.items()},
            "audit": self.new_audit(),
        }


def _refusal(call_id: str, decision: Decision) -> dict[str, Any]:
    content = (f"Refused by the policy harness ({decision.rule}). {decision.reason} "
               "This action did not run.")
    if decision.remedy:
        content += f" Next: {decision.remedy}."
    return dump(ToolMessage(
        id=call_id, role="tool", requestor="assistant", error=True, content=content,
    ))


def build_graph(runtime: RetailRuntime, checkpointer: Any):
    config = runtime.config

    def propose(state: ServiceState) -> dict[str, Any]:
        step = _Step(runtime, state)
        offered = runtime.tools
        if config.context_pruning_enabled:
            visible = runtime.gate.state_tools(step.tracker.state)
            offered = [t for t in runtime.tools if t.name in visible]
        messages = [SystemMessage(role="system", content=runtime.system_prompt)]
        messages += [to_message(m) for m in state.get("messages", [])]
        response = runtime.generate(
            model=runtime.llm, tools=offered, messages=messages,
            call_name="service_agent_response", **runtime.llm_args,
        )
        update: dict[str, Any] = {"messages": [dump(response)]}
        if response.tool_calls:
            update["proposal"] = dump(response)
            update["reply"] = None
        else:
            update["proposal"] = None
            update["reply"] = response.content
            update["attempts"] = 0
        return update

    def review(state: ServiceState) -> dict[str, Any]:
        step = _Step(runtime, state)
        proposal = AssistantMessage.model_validate(state["proposal"])
        calls = [to_harness_call(c) for c in proposal.tool_calls]
        turn = runtime.gate.review_turn(calls, step.tracker.state)

        refusals: list[dict[str, Any]] = []
        verdicts: list[Decision] = []
        reviewed = zip(proposal.tool_calls, calls, turn.decisions, strict=True)
        for original, call, decision in reviewed:
            if decision.allowed:
                checks = runtime.validator.review(
                    call, subject=step.tracker.facts.user_id, world=step.world
                )
                if not checks.passed:
                    decision = Decision(
                        outcome=DecisionOutcome.DENY, rule="validation.refused",
                        reason="; ".join(f.message for f in checks.failures),
                        remedy="; ".join(
                            dict.fromkeys(f.remedy for f in checks.failures if f.remedy)
                        ) or None,
                    )
            step.chain.append(
                AuditEvent.TOOL_PROPOSED if decision.allowed else AuditEvent.TOOL_DENIED,
                tool_call=call, decision=decision,
            )
            verdicts.append(decision)
            if not decision.allowed:
                refusals.append(_refusal(original.id, decision))

        if refusals:
            # The whole turn is refused: executing part of what was proposed would
            # leave the world half-changed against an intent the model never had.
            for original, decision in zip(proposal.tool_calls, verdicts, strict=True):
                if decision.allowed:
                    refusals.append(dump(ToolMessage(
                        id=original.id, role="tool", requestor="assistant", error=True,
                        content="Not run: another call in the same turn was refused.",
                    )))
            attempts = state.get("attempts", 0) + 1
            update = {**step.snapshot(), "messages": refusals, "proposal": None,
                      "attempts": attempts}
            if attempts >= runtime.max_internal_attempts:
                return _escalate(step, update)
            return update

        call = calls[0]
        spec = runtime.spec(call.name)
        needs_approval = (
            config.approval_gate_enabled
            and spec is not None
            and spec.kind in config.approval.require_for_kinds
        )
        if needs_approval:
            request = ApprovalRequest(
                request_id=str(uuid.uuid4()), tool_call=call,
                reason=f"{call.name} changes stored state",
            )
            step.tracker.open_approval(request.request_id, call)
            step.chain.append(AuditEvent.APPROVAL_REQUESTED, tool_call=call, decision=verdicts[0])
            return {**step.snapshot(), "pending_approval": request.model_dump(mode="json")}
        return step.snapshot()

    def await_approval(state: ServiceState) -> dict[str, Any]:
        request = ApprovalRequest.model_validate(state["pending_approval"])
        answer = interrupt({"approval": request.model_dump(mode="json")})
        # Everything below runs once, after a reviewer has answered.
        step = _Step(runtime, state)
        approved = bool(answer.get("approved"))
        decision = ApprovalDecision(
            request_id=request.request_id,
            status=ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED,
            policy=f"human:{answer.get('reviewer', 'unknown')}",
            reason=answer.get("note"),
        )
        step.tracker.resolve_approval(approved)
        step.chain.append(AuditEvent.APPROVAL_RESOLVED, tool_call=request.tool_call,
                          approval=decision)
        update: dict[str, Any] = {**step.snapshot(), "pending_approval": None}
        if not approved:
            proposal = AssistantMessage.model_validate(state["proposal"])
            refused = Decision(outcome=DecisionOutcome.DENY, rule="approval.rejected",
                               reason=decision.reason or "the reviewer did not authorise it")
            update["messages"] = [_refusal(proposal.tool_calls[0].id, refused)]
            update["proposal"] = None
        return update

    def execute(state: ServiceState) -> dict[str, Any]:
        step = _Step(runtime, state)
        proposal = AssistantMessage.model_validate(state["proposal"])
        executed = list(state.get("executed", []))
        messages, new_calls = [], []
        for original in proposal.tool_calls:
            call = to_harness_call(original)
            result = runtime.execute(step.session_id, executed + new_calls, original)
            new_calls.append(original.model_dump(mode="json"))
            messages.append(dump(result))
            spec = runtime.spec(call.name)
            write = runtime.is_write(call.name)
            if not result.error:
                if write and config.approval_gate_enabled:
                    try:
                        step.tracker.record_execution(call)
                    except StateError:
                        pass  # nothing was authorised; the chain still records what ran
                step.chain.append(
                    AuditEvent.TOOL_EXECUTED,
                    tool_call=call,
                    decision=Decision(
                        outcome=DecisionOutcome.ALLOW,
                        rule="service.executed",
                        reason="passed review and approval" if write else "passed review",
                    ),
                )
                if spec is not None and spec.role is ToolRole.ESCALATION:
                    step.tracker.record_transfer()
            facts = runtime.extract_facts(call.name, result.content, error=result.error)
            observe(step.world, facts.objects)
            try:
                if facts.user_id:
                    step.tracker.record_identity(facts.user_id)
                if facts.scope_targets is not None:
                    step.tracker.record_targets(facts.scope_targets)
            except StateError:
                pass
        return {**step.snapshot(), "messages": messages, "executed": new_calls,
                "proposal": None, "attempts": 0}

    def _escalate(step: _Step, update: dict[str, Any]) -> dict[str, Any]:
        step.chain.append(
            AuditEvent.TOOL_DENIED,
            decision=Decision(outcome=DecisionOutcome.DENY, rule="gating.budget_exhausted",
                              reason="no acceptable action within the attempt budget"),
        )
        step.tracker.record_transfer()
        reply = dump(AssistantMessage(role="assistant", content=ESCALATION_TEXT))
        return {**update, **step.snapshot(), "messages": update["messages"] + [reply],
                "reply": ESCALATION_TEXT, "attempts": 0}

    def after_propose(state: ServiceState) -> Literal["review", "__end__"]:
        return "review" if state.get("proposal") else END

    def after_review(
        state: ServiceState,
    ) -> Literal["await_approval", "execute", "propose", "__end__"]:
        if SessionFacts.model_validate(state.get("facts") or {}).transferred:
            return END
        if state.get("pending_approval"):
            return "await_approval"
        if state.get("proposal"):
            return "execute"
        return "propose"

    def after_approval(state: ServiceState) -> Literal["execute", "propose"]:
        return "execute" if state.get("proposal") else "propose"

    def after_execute(state: ServiceState) -> Literal["propose", "__end__"]:
        if SessionFacts.model_validate(state.get("facts") or {}).transferred:
            return END
        return "propose"

    graph = StateGraph(ServiceState)
    graph.add_node("propose", propose)
    graph.add_node("review", review)
    graph.add_node("await_approval", await_approval)
    graph.add_node("execute", execute)
    graph.add_edge(START, "propose")
    graph.add_conditional_edges("propose", after_propose)
    graph.add_conditional_edges("review", after_review)
    graph.add_conditional_edges("await_approval", after_approval)
    graph.add_conditional_edges("execute", after_execute)
    return graph.compile(checkpointer=checkpointer)
