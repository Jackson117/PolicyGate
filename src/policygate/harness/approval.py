"""The approval gate's deciders, and the numbers that must never be merged.

The gate itself lives in the agent: it pauses a write and asks. This module is
what answers, and what counts the answers afterwards.

Two things are being measured and they belong to different owners.
**Whether the gate fired** is the harness's contribution -- it is a property of
the trigger rule and of nothing else. **Whether the answer was yes** is
exogenous: it comes from whichever policy is installed, and under
:data:`ApprovalPolicy.ORACLE` it comes from the task's own reference
trajectory. An end-to-end score under an oracle is therefore partly a
measurement of the oracle, which is why :class:`ApprovalReport` keeps the two
rates apart and offers no combined figure.

The same applies to the two ways a trigger rule can be wrong. A write that
should have been paused and was not (``missed``) and a write that was paused
for nothing (``over_interruption``) have different denominators and different
costs -- the first is a compliance incident, the second is friction. Summing
them, or folding them into an F1, would let one hide the other, so this module
does not provide the means.

Pydantic and the standard library only. See docs/constraints.md #2.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from policygate.harness.config import ApprovalPolicy, HarnessConfig
from policygate.harness.types import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    AuditEvent,
    AuditRecord,
    ToolCall,
    canonical_json,
    sha256_hex,
)

ApprovalDecider = Callable[[ApprovalRequest], ApprovalDecision]
"""What the agent calls when the gate fires. One request in, one answer out."""


# --- ground truth ---------------------------------------------------------


class ExpectedAction(BaseModel):
    """One action from a task's reference trajectory.

    Mirrors tau2's ``Action`` without importing it; the adapter converts. The
    matching rule below is copied from tau2's own
    ``Action.compare_with_tool_call`` deliberately, so that "the oracle
    sanctions this call" means the same thing as "tau2's action evaluator would
    count this call", rather than meaning whatever I thought it should mean.
    """

    model_config = ConfigDict(frozen=True)

    action_id: str
    name: str
    arguments: Mapping[str, Any] = Field(default_factory=dict)
    compare_args: tuple[str, ...] | None = None
    """Which arguments to compare. None means "those the call actually carries"."""

    def matches(self, call: ToolCall) -> bool:
        if self.name != call.name:
            return False
        keys = call.arguments.keys() if self.compare_args is None else self.compare_args
        if not keys:
            # tau2's behaviour, faithfully: with nothing to compare, the name
            # alone matches. Reachable only for an argument-less call, which the
            # schema check refuses before approval is ever sought.
            return True
        return {k: v for k, v in call.arguments.items() if k in keys} == {
            k: v for k, v in self.arguments.items() if k in keys
        }


class GroundTruth(BaseModel):
    """The reference trajectory for one task -- and not more than that.

    ``evaluation_criteria.actions`` is one path that produces the target end
    state, not a requirement on the agent: tau2 scores retail on the database
    hash, so a different sequence of writes reaching an equivalent state passes.
    An oracle built on this list therefore rejects some genuinely correct
    alternatives. That bias is one-directional and worth stating in any table
    that reports oracle results: it makes the oracle stricter than the scorer,
    not more lenient.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    actions: tuple[ExpectedAction, ...] = ()

    def sanctions(self, call: ToolCall) -> bool:
        """Whether the reference trajectory contains this call.

        Stateless: a matching expectation is not consumed. Two identical writes
        both count as sanctioned, because catching the second one is
        idempotency's job (its own switch, deferred past W3), and folding it in
        here would let a duplicate-write defence be scored as an approval
        result.
        """
        return any(action.matches(call) for action in self.actions)


# --- the four policies ----------------------------------------------------


def auto_approve(request: ApprovalRequest) -> ApprovalDecision:
    """Always yes. The floor: measures the gate's cost with none of its benefit."""
    return ApprovalDecision(
        request_id=request.request_id,
        status=ApprovalStatus.APPROVED,
        policy=ApprovalPolicy.AUTO_APPROVE.value,
        reason="approved without review",
    )


def auto_reject(request: ApprovalRequest) -> ApprovalDecision:
    """Always no. The ceiling on damage and the floor on usefulness.

    Not a serious operating mode -- it is the arm that shows how much of the
    task actually depends on the gated writes, which is the denominator any
    approval result has to be read against.
    """
    return ApprovalDecision(
        request_id=request.request_id,
        status=ApprovalStatus.REJECTED,
        policy=ApprovalPolicy.AUTO_REJECT.value,
        reason="rejected without review",
    )


class Oracle(BaseModel):
    """Approves exactly the writes the task's reference trajectory contains.

    This hands the agent a free correctness checker, so any end-to-end score
    under it is inflated and is only interpretable beside the gate trigger rate.
    It is here as an upper bound, not as a proposed deployment.
    """

    model_config = ConfigDict(frozen=True)

    truth: GroundTruth

    @property
    def policy(self) -> str:
        return ApprovalPolicy.ORACLE.value

    def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        sanctioned = self.truth.sanctions(request.tool_call)
        return ApprovalDecision(
            request_id=request.request_id,
            status=ApprovalStatus.APPROVED if sanctioned else ApprovalStatus.REJECTED,
            policy=self.policy,
            reason=(
                "matches the reference trajectory"
                if sanctioned
                else "no action in the reference trajectory matches this call"
            ),
        )


class NoisyOracle(Oracle):
    """The oracle, wrong a fixed fraction of the time.

    The noise is drawn from the action's fingerprint and the run seed rather
    than from a stream, for two reasons. It reproduces: the same call in the
    same arm gets the same answer no matter what order things happened in. And
    it cannot be farmed: re-proposing an identical call after a rejection
    returns the same rejection, so the agent cannot retry until the coin lands,
    which would turn this into a measurement of retry persistence rather than of
    tolerance to an unreliable reviewer.

    The cost is that the noise is per-action, not per-request. A *different*
    proposal -- a corrected one -- gets an independent draw, which is the
    behaviour that matters.
    """

    noise_rate: float = Field(ge=0.0, le=1.0)
    seed: int = 0

    @property
    def policy(self) -> str:
        return ApprovalPolicy.NOISY_ORACLE.value

    def _draw(self, call: ToolCall) -> float:
        digest = sha256_hex(
            canonical_json({"seed": self.seed, "action": call.fingerprint()})
        )
        return int(digest[:16], 16) / 2**64

    def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        honest = super().__call__(request)
        if self._draw(request.tool_call) >= self.noise_rate:
            return honest.model_copy(update={"policy": self.policy})
        flipped = (
            ApprovalStatus.REJECTED
            if honest.status is ApprovalStatus.APPROVED
            else ApprovalStatus.APPROVED
        )
        return ApprovalDecision(
            request_id=request.request_id,
            status=flipped,
            policy=self.policy,
            reason=f"reviewer error (noise_rate={self.noise_rate})",
        )


def make_decider(config: HarnessConfig, *, truth: GroundTruth | None = None) -> ApprovalDecider:
    """Build the decider an arm calls for.

    Raises rather than falling back when an oracle is asked for without ground
    truth: silently degrading to ``auto_approve`` would produce a run labelled
    ``oracle`` that was nothing of the kind, and the label is what ends up in
    the results table.
    """
    policy = config.approval.policy
    if policy is ApprovalPolicy.AUTO_APPROVE:
        return auto_approve
    if policy is ApprovalPolicy.AUTO_REJECT:
        return auto_reject
    if truth is None:
        raise ValueError(f"approval policy {policy.value} needs ground truth")
    if policy is ApprovalPolicy.ORACLE:
        return Oracle(truth=truth)
    if policy is ApprovalPolicy.NOISY_ORACLE:
        return NoisyOracle(
            truth=truth, noise_rate=config.approval.noise_rate, seed=config.seed
        )
    raise AssertionError(f"unhandled approval policy: {policy}")


# --- reading the chain back -----------------------------------------------


class _CallTrace(BaseModel):
    """Everything the chain recorded about one proposed call."""

    model_config = ConfigDict(frozen=True)

    call: ToolCall
    gate_fired: bool = False
    resolution: ApprovalStatus | None = None
    proposed: bool = False
    denied: bool = False
    executed: bool = False

    @property
    def reached_gate(self) -> bool:
        """Whether the call got as far as the approval stage.

        A call refused by gating or validation never did, and counting it in the
        trigger-rate denominator would understate the rate by charging the
        approval gate for work another module had already done.
        """
        return self.gate_fired or self.proposed


def _trace_calls(records: Sequence[AuditRecord]) -> list[_CallTrace]:
    order: list[str] = []
    seen: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.tool_call is None:
            continue
        key = record.tool_call.id
        if key not in seen:
            order.append(key)
            seen[key] = {"call": record.tool_call}
        trace = seen[key]
        if record.event is AuditEvent.APPROVAL_REQUESTED:
            trace["gate_fired"] = True
        elif record.event is AuditEvent.APPROVAL_RESOLVED and record.approval is not None:
            trace["resolution"] = record.approval.status
        elif record.event is AuditEvent.TOOL_PROPOSED:
            trace["proposed"] = True
        elif record.event is AuditEvent.TOOL_DENIED:
            trace["denied"] = True
        elif record.event is AuditEvent.TOOL_EXECUTED:
            trace["executed"] = True
    return [_CallTrace(**seen[key]) for key in order]


class GateStats(BaseModel):
    """How often the gate fired, and how often the answer was yes.

    Two rates over two different denominators, reported separately because they
    have different owners: the trigger rate is the harness's, the pass rate is
    the installed policy's.
    """

    model_config = ConfigDict(frozen=True)

    writes_seen: int = 0
    """Every write the model proposed, whatever became of it."""

    stopped_upstream: int = 0
    """Writes gating or validation refused before approval was ever sought."""

    reached_gate: int = 0
    gate_fired: int = 0
    approved: int = 0
    rejected: int = 0
    unresolved: int = 0
    """Gate fired, no answer recorded. Should be zero; a run where it is not is broken."""

    executed: int = 0

    @property
    def gate_trigger_rate(self) -> float | None:
        """Share of writes reaching the gate that were paused. None if none did."""
        return self.gate_fired / self.reached_gate if self.reached_gate else None

    @property
    def approval_pass_rate(self) -> float | None:
        """Share of paused writes that were approved. None if the gate never fired."""
        return self.approved / self.gate_fired if self.gate_fired else None


class TriggerQuality(BaseModel):
    """Where the trigger rule was wrong, kept as two separate errors.

    Labels come from the reference trajectory, so "sanctioned" means "on the
    reference path", not "correct" -- see :class:`GroundTruth`.

    There is deliberately no combined score. ``missed`` is a compliance
    incident and ``over_interruption`` is friction; a single number would let a
    rule trade one for the other and still look unchanged.
    """

    model_config = ConfigDict(frozen=True)

    caught: int = 0
    """Paused, and off the reference path. What the gate is for."""

    over_interruption: int = 0
    """Paused, but on the reference path. Pure friction."""

    missed: int = 0
    """Not paused, and off the reference path. The failure that costs money."""

    clean: int = 0
    """Not paused, and on the reference path. Nothing to do."""

    @property
    def off_path(self) -> int:
        return self.caught + self.missed

    @property
    def on_path(self) -> int:
        return self.over_interruption + self.clean

    @property
    def missed_rate(self) -> float | None:
        """Of the off-path writes, the share that reached the environment ungated."""
        return self.missed / self.off_path if self.off_path else None

    @property
    def over_interruption_rate(self) -> float | None:
        """Of the on-path writes, the share paused for nothing."""
        return self.over_interruption / self.on_path if self.on_path else None


class ApprovalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    gate: GateStats
    quality: TriggerQuality | None = None
    """Absent without ground truth. There is no way to label a trigger without it."""


def summarise_approvals(
    records: Sequence[AuditRecord],
    *,
    write_tools: Collection[str],
    truth: GroundTruth | None = None,
) -> ApprovalReport:
    """Read one session's chain into the approval numbers.

    ``write_tools`` is passed in rather than looked up because the chain stores
    calls, not kinds, and the harness may not ask tau2 what kind a tool is.
    """
    traces = [t for t in _trace_calls(records) if t.call.name in write_tools]

    gate = GateStats(
        writes_seen=len(traces),
        stopped_upstream=sum(1 for t in traces if not t.reached_gate and t.denied),
        reached_gate=sum(1 for t in traces if t.reached_gate),
        gate_fired=sum(1 for t in traces if t.gate_fired),
        approved=sum(1 for t in traces if t.resolution is ApprovalStatus.APPROVED),
        rejected=sum(1 for t in traces if t.resolution is ApprovalStatus.REJECTED),
        unresolved=sum(1 for t in traces if t.gate_fired and t.resolution is None),
        executed=sum(1 for t in traces if t.executed),
    )
    if truth is None:
        return ApprovalReport(gate=gate)

    judged = [t for t in traces if t.reached_gate]
    labelled = [(t, truth.sanctions(t.call)) for t in judged]
    quality = TriggerQuality(
        caught=sum(1 for t, ok in labelled if t.gate_fired and not ok),
        over_interruption=sum(1 for t, ok in labelled if t.gate_fired and ok),
        missed=sum(1 for t, ok in labelled if not t.gate_fired and not ok),
        clean=sum(1 for t, ok in labelled if not t.gate_fired and ok),
    )
    return ApprovalReport(gate=gate, quality=quality)
