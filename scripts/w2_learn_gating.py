from ast import Tuple
from enum import StrEnum
from os import name
from typing import Iterable, Mapping
from attr import frozen
from litellm import ConfigDict, ModelConfig
from openai import BaseModel
from tau2.data_model.message import ToolCall
from policygate.harness.config import HarnessConfig
from policygate.harness.state import SessionState
from policygate.harness.types import (
    ToolKind,
    ToolCall,
    ToolRole,
    ToolSpec,
    ClauseRef,
    Decision,
    DecisionOutcome,
)

class GateRule(StrEnum):
    ALLOWED = "gating.allowed"
    DISABLED = "gating.disabled"

    UNKNOWN_TOOL = "gating.unknwon_tool"
    IDENTITY_FIRST = "gating.identity_first"
    WRITE_REQUIRES_SCOPE = "gating.write_requires_scope"
    FROZEN_PENDING_APPROVAL = "gating.frozen_pending_approval"
    SESSION_ENDED = "gating.session_ended"
    ONE_CALL_PER_TURN = "gating.one_call_per_turn"

class TurnReview(BaseModel):
    model_config = ConfigDict(frozen=True)

    decisions: tuple[Decision, ...]
    allowed_calls: tuple[ToolCall, ...]

    @property
    def refused(self) -> bool:
        return any(d.outcome is not DecisionOutcome.ALLOW for d in self.decisions)
    
class ToolGate:
    def __init__(
        self,
        catalogue: Iterable[ToolSpec],
        config: HarnessConfig,
        clauses: Mapping[GateRule, tuple[ClauseRef, ...]] | None = None
    ) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for spec in catalogue:
            if self._tools.get(spec.name):
                raise ValueError(f"duplicate tool in catalogue: {spec.name!r}")
            self._tools[spec.name] = spec
        self._config = config
        self._clauses = dict(clauses or {})
    
    @property
    def catalogue(self) -> Mapping[str, ToolSpec]:
        return dict(self._tools)
    
    def visible_tools(self, state: SessionState) -> frozenset[str]:
        if not self._config.gating_enabled:
            return frozenset(self._tools.keys())
        
        if state in (SessionState.PENDING_APPROVAL, SessionState.TRANSFERRED):
            return frozenset()
        
        if state is SessionState.UNVERIFIED:
            return frozenset(
                name 
                for name, spec in self._tools.items() 
                if spec.role in (ToolRole.IDENTITY_LOOKUP, ToolRole.ESCALATION)
            )
        
        if state is SessionState.SCOPED:
            return frozenset(self._tools)
        
        if state is SessionState.VERIFIED:
            return frozenset(
                name 
                for name, spec in self._tools.items()
                if spec.kind is not ToolKind.WRITE
            )
        
    def review_call(self, call: ToolCall, state: SessionState) -> Decision:
        if not self._config.gating_enabled:
            return self._allow(GateRule.DISABLED, "gating is disabled")
        
        # assert if call is in self._tools
        spec = self._tools.get(call.name)
        if spec is None:
            return self._deny(GateRule.UNKNOWN_TOOL, f"{call.name!r} is not a tool in this domain.")
        
        if call.name in self.visible_tools(state):
            return self._allow(GateRule.ALLOWED, f"{call.name!r} is visible in state {state}")
        
        return self._deny(*self._why_not(spec, state))
    
    def review_turn(self, calls: Iterable[ToolCall], state: SessionState) -> TurnReview:
        calls = tuple(calls)

        if self._config.parallel_call_block_enabled and len(calls) > 1:
            decision = self._deny(GateRule.ONE_CALL_PER_TURN, f"{len(calls)} tool calls proposed in one turn; policy allows at most one")
            return TurnReview(decisions=(decision, ), allowed_calls=())
        
        decisions = tuple(self.review_call(call, state) for call in calls)
        allowed_calls = tuple(c for c, d in zip(calls, decisions, strict=True) if d.allowed)

        return TurnReview(decisions=decisions, allowed_calls=allowed_calls)

    def _why_not(self, spec: ToolSpec, state: SessionState) -> tuple[GateRule, str]:
        pass

    def _allow(self, rule: GateRule, reason: str) -> Decision:
        return Decision(
            outcome=DecisionOutcome.ALLOW,
            rule = rule,
            reason=reason,
            clauses=self._clauses.get(rule, ())
        )

    def _deny(self, rule: GateRule, reason: str) -> Decision:
        return Decision(
            outcome=DecisionOutcome.DENY,
            rule=rule,
            reason=reason,
            clauses=self._clauses.get(rule, ())
        )