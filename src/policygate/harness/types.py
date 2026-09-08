"""Shared data models for the harness.

Hard constraint (CLAUDE.md #2): this package imports Pydantic and the standard
library, and nothing else. No LangGraph, no LiteLLM, no tau2. Everything here
must be constructible and testable without an LLM, a benchmark, or a network.

Two conventions run through this module:

*Structured refusal, not exceptions.* A check that fails returns a
:class:`CheckResult` describing what was wrong in terms the agent can act on.
Raising would collapse "the model proposed something invalid" into the same
channel as "the harness has a bug", and the first is normal operation.

*Explicit determinism.* Anything that feeds the audit hash chain or an
idempotency key is serialised through :func:`canonical_json`, and timestamps
are supplied by the caller rather than read from the clock. Hash chains that
embed ``datetime.now()`` cannot be reproduced, and a chain nobody can recompute
proves nothing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace.

    The same logical payload must produce the same string on every machine and
    every run, because this string is what gets hashed into the audit chain and
    what idempotency keys are derived from.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def sha256_hex(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


class ToolKind(StrEnum):
    """What a tool does to the world.

    Mirrors tau2's ``ToolType`` by value so an adapter can translate without a
    lookup table, but is declared here so that harness/ never imports tau2. The
    adapter is responsible for the mapping; the harness only ever sees this
    enum.
    """

    READ = "read"
    WRITE = "write"
    THINK = "think"
    GENERIC = "generic"


class ToolCall(BaseModel):

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: Mapping[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        return sha256_hex(canonical_json({"name": self.name, "arguments": dict(self.arguments)}))
    

class ClauseRef(BaseModel):
    """A pointer to the policy text that justifies a decision.

    Audit records are only useful if a human can check them against the rule
    that was supposedly applied, so every deny or approval request should carry
    at least one of these.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    clause: str
    quote: str | None = None


class DecisionOutcome(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class Decision(BaseModel):
    """The harness's verdict on one proposed tool call.

    ``rule`` names the specific mechanism that fired. Ablation tables are built
    by counting decisions per rule, so a decision that cannot say which module
    produced it is not measurable and therefore not evidence.
    """

    model_config = ConfigDict(frozen=True)

    outcome: DecisionOutcome
    rule: str
    reason: str
    clauses: tuple[ClauseRef, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.outcome is DecisionOutcome.ALLOW


class FailureCode(StrEnum):
    """Why a proposed call was rejected.

    These are the classes W1 actually observed in retail trajectories, not a
    speculative taxonomy. Loop and timeout categories are deliberately absent:
    across 90 simulations there were zero non-normal terminations, so inventing
    codes for them would be inventing evidence.
    """

    UNKNOWN_TOOL = "unknown_tool"           # name not in the domain's toolset
    TOOL_NOT_AVAILABLE = "tool_not_available"  # exists, but gated out in this state
    SCHEMA_INVALID = "schema_invalid"       # arguments do not match the signature
    VALUE_NOT_CANONICAL = "value_not_canonical"  # task 22: "United States" vs "USA"
    REFERENT_AMBIGUOUS = "referent_ambiguous"    # task 14: several candidate orders
    REFERENT_UNKNOWN = "referent_unknown"        # the 31x "Order not found"
    PRECONDITION_UNMET = "precondition_unmet"    # e.g. exchanging a non-delivered order
    PARALLEL_TOOL_CALLS = "parallel_tool_calls"  # retail policy.md: one call at a time


class CheckFailure(BaseModel):
    """One specific thing wrong with a proposed call.

    ``observed`` and ``expected`` exist so the refusal handed back to the model
    is actionable ("state must be NY, you sent New York") rather than a bare
    rejection it can only retry blindly.
    """

    model_config = ConfigDict(frozen=True)

    code: FailureCode
    message: str
    field: str | None = None
    observed: Any = None
    expected: Any = None


class CheckResult(BaseModel):
    """Outcome of running validators over a proposed call."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    failures: tuple[CheckFailure, ...] = ()

    @classmethod
    def ok(cls) -> CheckResult:
        return cls(passed=True)

    @classmethod
    def failed(cls, *failures: CheckFailure) -> CheckResult:
        if not failures:
            raise ValueError("a failed CheckResult must carry at least one failure")
        return cls(passed=False, failures=failures)

    def merged_with(self, other: CheckResult) -> CheckResult:
        """Combine two results, keeping every failure.

        Validators run independently and we want the model to see all the
        problems at once; fixing one error per turn burns turns and, worse,
        makes the harness look like it caused the extra steps.
        """
        failures = self.failures + other.failures
        return CheckResult(passed=not failures, failures=failures)


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalRequest(BaseModel):
    """A write the harness paused before executing.

    Distinct from tau2's ``transfer_to_human_agents``: that is a *tool* the
    model may choose to call and may equally fail to call, whereas this is
    imposed by the harness and is not in the model's action space at all.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    tool_call: ToolCall
    reason: str
    clauses: tuple[ClauseRef, ...] = ()


class ApprovalDecision(BaseModel):
    """The approval simulator's answer.

    ``status`` and the policy that produced it are recorded separately on
    purpose: gate *trigger* rate is the harness's contribution, while the
    approve/reject split is exogenous. Reporting only the final success rate
    would credit the harness with the oracle's work.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    status: ApprovalStatus
    policy: str
    reason: str | None = None


class AuditEvent(StrEnum):
    TOOL_PROPOSED = "tool_proposed"
    TOOL_DENIED = "tool_denied"
    TOOL_EXECUTED = "tool_executed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    STATE_TRANSITION = "state_transition"


class AuditRecord(BaseModel):
    """One link in the audit chain.

    A log says what happened; this is meant to *prove* it. Three properties are
    required of the chain as a whole -- completeness (every world-changing
    action appears), linkage (each record carries the evidence and decision
    behind it), and verifiable order (``prev_hash`` makes tampering break the
    chain). ``record_hash`` covers everything above it including ``prev_hash``.

    ``timestamp`` is supplied by the caller. A chain that embeds wall-clock time
    it read itself cannot be recomputed by anyone checking it later.
    """

    model_config = ConfigDict(frozen=True)

    seq: int
    timestamp: datetime
    event: AuditEvent
    session_id: str
    tool_call: ToolCall | None = None
    decision: Decision | None = None
    approval: ApprovalDecision | None = None
    world_diff: Mapping[str, Any] | None = None
    prev_hash: str
    record_hash: str

    def payload(self) -> dict[str, Any]:
        """Exactly the fields the hash covers, in canonical form."""
        return {
            "seq": self.seq,
            "timestamp": self.timestamp.isoformat(),
            "event": self.event.value,
            "session_id": self.session_id,
            "tool_call": self.tool_call.model_dump(mode="json") if self.tool_call else None,
            "decision": self.decision.model_dump(mode="json") if self.decision else None,
            "approval": self.approval.model_dump(mode="json") if self.approval else None,
            "world_diff": dict(self.world_diff) if self.world_diff is not None else None,
            "prev_hash": self.prev_hash,
        }

    def compute_hash(self) -> str:
        return sha256_hex(canonical_json(self.payload()))

    def is_self_consistent(self) -> bool:
        """Whether ``record_hash`` still matches the content it claims to cover."""
        return self.record_hash == self.compute_hash()


GENESIS_HASH = "0" * 64
"""``prev_hash`` of the first record in a chain."""

