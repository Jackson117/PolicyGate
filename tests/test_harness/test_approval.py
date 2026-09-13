"""Tests for harness/approval.py.

Two things are checked here. The deciders behave as their names claim -- in
particular the noisy one is reproducible and cannot be farmed by retrying. And
the report keeps the numbers separate: gate trigger rate and approval pass rate
have different denominators, and so do the two ways a trigger rule is wrong.

No LLM, no benchmark, no network.
"""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from policygate.harness.approval import (
    ExpectedAction,
    GroundTruth,
    NoisyOracle,
    Oracle,
    auto_approve,
    auto_reject,
    make_decider,
    summarise_approvals,
)
from policygate.harness.audit import AuditChain
from policygate.harness.config import ApprovalConfig, ApprovalPolicy, HarnessConfig
from policygate.harness.types import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    AuditEvent,
    Decision,
    DecisionOutcome,
    ToolCall,
)

START = datetime(2026, 1, 1, 12, 0, 0)

CANCEL = ToolCall(
    id="c1",
    name="cancel_pending_order",
    arguments={"order_id": "#W1", "reason": "no longer needed"},
)
OTHER = ToolCall(
    id="c2",
    name="cancel_pending_order",
    arguments={"order_id": "#W9", "reason": "no longer needed"},
)
READ = ToolCall(id="r1", name="get_order_details", arguments={"order_id": "#W1"})

WRITE_TOOLS = frozenset({"cancel_pending_order", "modify_pending_order_items"})

TRUTH = GroundTruth(
    task_id="t1",
    actions=(
        ExpectedAction(
            action_id="a1",
            name="cancel_pending_order",
            arguments={"order_id": "#W1", "reason": "no longer needed"},
            compare_args=("order_id",),
        ),
    ),
)

ALLOW = Decision(outcome=DecisionOutcome.ALLOW, rule="gating.allowed", reason="scoped")
DENY = Decision(outcome=DecisionOutcome.DENY, rule="validation.refused", reason="bad value")


def request_for(call: ToolCall, request_id: str = "req") -> ApprovalRequest:
    return ApprovalRequest(request_id=request_id, tool_call=call, reason="write")


def ticking_clock(step_seconds: int = 1):
    state = {"n": 0}

    def clock() -> datetime:
        state["n"] += 1
        return START + timedelta(seconds=step_seconds * state["n"])

    return clock


# --- matching -------------------------------------------------------------


def test_name_must_match():
    action = ExpectedAction(action_id="a", name="cancel_pending_order", arguments={})
    assert not action.matches(READ)


def test_compare_args_narrows_the_comparison():
    """The reason a reason-text difference does not make the call a different one."""
    call = ToolCall(
        id="x",
        name="cancel_pending_order",
        arguments={"order_id": "#W1", "reason": "ordered by mistake"},
    )
    assert TRUTH.sanctions(call)


def test_without_compare_args_every_argument_the_call_carries_is_compared():
    action = ExpectedAction(
        action_id="a",
        name="cancel_pending_order",
        arguments={"order_id": "#W1", "reason": "no longer needed"},
    )
    assert action.matches(CANCEL)
    assert not action.matches(OTHER)


def test_a_different_referent_is_not_sanctioned():
    assert not TRUTH.sanctions(OTHER)


def test_sanctioning_does_not_consume_the_expectation():
    """Duplicate writes are idempotency's problem, not the oracle's."""
    assert TRUTH.sanctions(CANCEL)
    assert TRUTH.sanctions(CANCEL)


def test_an_empty_trajectory_sanctions_nothing():
    assert not GroundTruth(task_id="t").sanctions(CANCEL)


# --- the four policies ----------------------------------------------------


def test_auto_approve_and_auto_reject_report_their_own_policy():
    approved = auto_approve(request_for(CANCEL))
    rejected = auto_reject(request_for(CANCEL))
    assert approved.status is ApprovalStatus.APPROVED
    assert rejected.status is ApprovalStatus.REJECTED
    assert approved.policy == ApprovalPolicy.AUTO_APPROVE.value
    assert rejected.policy == ApprovalPolicy.AUTO_REJECT.value


def test_oracle_follows_the_reference_trajectory():
    oracle = Oracle(truth=TRUTH)
    assert oracle(request_for(CANCEL)).status is ApprovalStatus.APPROVED
    assert oracle(request_for(OTHER)).status is ApprovalStatus.REJECTED


def test_noiseless_noisy_oracle_is_the_oracle():
    noisy = NoisyOracle(truth=TRUTH, noise_rate=0.0, seed=7)
    honest = Oracle(truth=TRUTH)
    for call in (CANCEL, OTHER):
        assert noisy(request_for(call)).status is honest(request_for(call)).status


def test_full_noise_inverts_every_answer():
    noisy = NoisyOracle(truth=TRUTH, noise_rate=1.0, seed=7)
    assert noisy(request_for(CANCEL)).status is ApprovalStatus.REJECTED
    assert noisy(request_for(OTHER)).status is ApprovalStatus.APPROVED


def test_the_noisy_oracle_cannot_be_farmed_by_retrying():
    """An identical proposal gets an identical answer, under a fresh request id."""
    noisy = NoisyOracle(truth=TRUTH, noise_rate=0.5, seed=3)
    first = noisy(request_for(OTHER, "req-1"))
    second = noisy(request_for(OTHER, "req-2"))
    assert first.status is second.status
    assert second.request_id == "req-2"


def test_a_corrected_proposal_gets_an_independent_draw():
    noisy = NoisyOracle(truth=TRUTH, noise_rate=0.5, seed=3)
    calls = [
        ToolCall(id=f"c{i}", name="cancel_pending_order", arguments={"order_id": f"#W{i}"})
        for i in range(500)
    ]
    honest = Oracle(truth=TRUTH)
    flipped = sum(
        1
        for c in calls
        if noisy(request_for(c)).status is not honest(request_for(c)).status
    )
    assert 0.4 <= flipped / len(calls) <= 0.6


def test_the_seed_changes_which_answers_flip():
    a = NoisyOracle(truth=TRUTH, noise_rate=0.5, seed=1)
    b = NoisyOracle(truth=TRUTH, noise_rate=0.5, seed=2)
    calls = [
        ToolCall(id=f"c{i}", name="cancel_pending_order", arguments={"order_id": f"#W{i}"})
        for i in range(50)
    ]
    left = [a(request_for(c)).status for c in calls]
    right = [b(request_for(c)).status for c in calls]
    assert left != right


# --- building a decider ---------------------------------------------------


def test_make_decider_returns_the_policy_the_config_names():
    for policy, expected in (
        (ApprovalPolicy.AUTO_APPROVE, ApprovalPolicy.AUTO_APPROVE.value),
        (ApprovalPolicy.AUTO_REJECT, ApprovalPolicy.AUTO_REJECT.value),
        (ApprovalPolicy.ORACLE, ApprovalPolicy.ORACLE.value),
    ):
        config = HarnessConfig(approval=ApprovalConfig(policy=policy))
        decider = make_decider(config, truth=TRUTH)
        assert decider(request_for(CANCEL)).policy == expected


def test_a_noisy_oracle_takes_its_seed_from_the_run():
    config = HarnessConfig(
        seed=11,
        approval=ApprovalConfig(policy=ApprovalPolicy.NOISY_ORACLE, noise_rate=0.5),
    )
    decider = make_decider(config, truth=TRUTH)
    assert isinstance(decider, NoisyOracle)
    assert (decider.seed, decider.noise_rate) == (11, 0.5)


def test_an_oracle_without_ground_truth_raises_rather_than_degrading():
    """A run labelled `oracle` that quietly ran auto_approve would poison the table."""
    config = HarnessConfig(approval=ApprovalConfig(policy=ApprovalPolicy.ORACLE))
    with pytest.raises(ValueError, match="ground truth"):
        make_decider(config)


def test_noise_on_a_policy_that_ignores_it_is_refused():
    with pytest.raises(ValidationError):
        ApprovalConfig(policy=ApprovalPolicy.ORACLE, noise_rate=0.2)


# --- reading the chain back -----------------------------------------------


def gated_chain(config: HarnessConfig) -> AuditChain:
    """One session: a read, a gated-and-approved write, an off-path write rejected."""
    chain = AuditChain("s1", ticking_clock(), config)
    chain.append(AuditEvent.TOOL_PROPOSED, tool_call=READ, decision=ALLOW)

    chain.append(AuditEvent.APPROVAL_REQUESTED, tool_call=CANCEL, decision=ALLOW)
    chain.append(
        AuditEvent.APPROVAL_RESOLVED,
        tool_call=CANCEL,
        approval=ApprovalDecision(
            request_id="r1", status=ApprovalStatus.APPROVED, policy="oracle"
        ),
    )
    chain.append(AuditEvent.TOOL_PROPOSED, tool_call=CANCEL, decision=ALLOW)
    chain.append(AuditEvent.TOOL_EXECUTED, tool_call=CANCEL, decision=ALLOW)

    chain.append(AuditEvent.APPROVAL_REQUESTED, tool_call=OTHER, decision=ALLOW)
    chain.append(
        AuditEvent.APPROVAL_RESOLVED,
        tool_call=OTHER,
        approval=ApprovalDecision(
            request_id="r2", status=ApprovalStatus.REJECTED, policy="oracle"
        ),
    )
    chain.append(AuditEvent.TOOL_DENIED, tool_call=OTHER, decision=DENY)
    return chain


def test_reads_are_not_counted_as_writes():
    report = summarise_approvals(
        gated_chain(HarnessConfig.full()).records, write_tools=WRITE_TOOLS
    )
    assert report.gate.writes_seen == 2


def test_the_two_rates_have_their_own_denominators():
    report = summarise_approvals(
        gated_chain(HarnessConfig.full()).records, write_tools=WRITE_TOOLS
    )
    assert report.gate.reached_gate == 2
    assert report.gate.gate_trigger_rate == 1.0
    assert (report.gate.approved, report.gate.rejected) == (1, 1)
    assert report.gate.approval_pass_rate == 0.5
    assert report.gate.unresolved == 0
    assert report.gate.executed == 1


def test_a_write_stopped_upstream_is_not_charged_to_the_gate():
    """Gating and validation refuse before approval is sought; the rate excludes those."""
    config = HarnessConfig.full()
    chain = AuditChain("s2", ticking_clock(), config)
    chain.append(AuditEvent.TOOL_DENIED, tool_call=CANCEL, decision=DENY)
    chain.append(AuditEvent.APPROVAL_REQUESTED, tool_call=OTHER, decision=ALLOW)
    chain.append(
        AuditEvent.APPROVAL_RESOLVED,
        tool_call=OTHER,
        approval=ApprovalDecision(
            request_id="r", status=ApprovalStatus.APPROVED, policy="auto_approve"
        ),
    )
    chain.append(AuditEvent.TOOL_PROPOSED, tool_call=OTHER, decision=ALLOW)

    gate = summarise_approvals(chain.records, write_tools=WRITE_TOOLS).gate
    assert (gate.writes_seen, gate.stopped_upstream, gate.reached_gate) == (2, 1, 1)
    assert gate.gate_trigger_rate == 1.0


def test_without_ground_truth_there_is_no_quality_section():
    report = summarise_approvals(
        gated_chain(HarnessConfig.full()).records, write_tools=WRITE_TOOLS
    )
    assert report.quality is None


def test_over_interruption_and_misses_are_counted_apart():
    """A rule that pauses every write catches the off-path one and annoys the rest."""
    report = summarise_approvals(
        gated_chain(HarnessConfig.full()).records, write_tools=WRITE_TOOLS, truth=TRUTH
    )
    quality = report.quality
    assert quality is not None
    assert (quality.caught, quality.over_interruption) == (1, 1)
    assert (quality.missed, quality.clean) == (0, 0)
    assert quality.missed_rate == 0.0
    assert quality.over_interruption_rate == 1.0


def test_an_ungated_off_path_write_is_a_miss_not_a_catch():
    """The gate off entirely: the write the oracle would have stopped goes through."""
    config = HarnessConfig.full(approval_gate_enabled=False)
    chain = AuditChain("s3", ticking_clock(), config)
    chain.append(AuditEvent.TOOL_PROPOSED, tool_call=OTHER, decision=ALLOW)
    chain.append(AuditEvent.TOOL_EXECUTED, tool_call=OTHER, decision=ALLOW)
    chain.append(AuditEvent.TOOL_PROPOSED, tool_call=CANCEL, decision=ALLOW)

    report = summarise_approvals(chain.records, write_tools=WRITE_TOOLS, truth=TRUTH)
    quality = report.quality
    assert quality is not None
    assert (quality.caught, quality.over_interruption) == (0, 0)
    assert (quality.missed, quality.clean) == (1, 1)
    assert report.gate.gate_trigger_rate == 0.0
    assert report.gate.approval_pass_rate is None


def test_rates_are_none_rather_than_zero_when_nothing_happened():
    """A rate with an empty denominator is not 0.0, and averaging it in as 0 would lie."""
    config = HarnessConfig.full()
    chain = AuditChain("s4", ticking_clock(), config)
    chain.append(AuditEvent.TOOL_PROPOSED, tool_call=READ, decision=ALLOW)

    report = summarise_approvals(chain.records, write_tools=WRITE_TOOLS, truth=TRUTH)
    assert report.gate.gate_trigger_rate is None
    assert report.gate.approval_pass_rate is None
    assert report.quality is not None
    assert report.quality.missed_rate is None
    assert report.quality.over_interruption_rate is None


def test_an_unresolved_request_is_surfaced_not_hidden():
    config = HarnessConfig.full()
    chain = AuditChain("s5", ticking_clock(), config)
    chain.append(AuditEvent.APPROVAL_REQUESTED, tool_call=CANCEL, decision=ALLOW)

    gate = summarise_approvals(chain.records, write_tools=WRITE_TOOLS).gate
    assert gate.unresolved == 1
    assert gate.approval_pass_rate == 0.0


def test_disabled_auditing_leaves_nothing_to_summarise():
    """The report is built from the chain, so an arm without it reports no numbers."""
    config = HarnessConfig.full(audit_enabled=False)
    report = summarise_approvals(gated_chain(config).records, write_tools=WRITE_TOOLS)
    assert report.gate.writes_seen == 0
    assert report.gate.gate_trigger_rate is None
