"""Tests for viz/data.py: the numbers the dashboard shows, without the dashboard.

Built from real tau2 result types and real audit records, because the module's
job is to read them. The properties that matter: exclusions stay visible, costs
include what tau2 never saw, and a single trace always comes with what it
represents.
"""

from datetime import datetime

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.data_model.tasks import RewardType

from policygate.eval.runner import SessionRecord
from policygate.harness.audit import AuditChain
from policygate.harness.config import HarnessConfig
from policygate.harness.types import AuditEvent, Decision, DecisionOutcome
from policygate.harness.types import ToolCall as HarnessCall
from policygate.viz.data import (
    ArmData,
    compare_arms,
    first_divergence,
    harness_events_by_call,
    representativeness,
    simulation_rows,
    summarise_arm,
    task_shifts,
)

SEEDS = (11, 22, 33)


def sim(task, trial, reward, *, parallel=False, reason=TerminationReason.USER_STOP,
        agent_cost=0.01, user_cost=0.002, db=None):
    calls = [
        ToolCall(id=f"{task}-{trial}-a", name="get_order_details", arguments={"order_id": "#W1"})
    ]
    if parallel:
        calls.append(ToolCall(id=f"{task}-{trial}-b", name="get_order_details",
                              arguments={"order_id": "#W2"}))
    messages = [
        AssistantMessage(role="assistant", content="Hi"),
        UserMessage(role="user", content="help"),
        AssistantMessage(role="assistant", content=None, tool_calls=calls),
    ]
    info = None
    if reason is not TerminationReason.INFRASTRUCTURE_ERROR:
        breakdown = {RewardType.DB: reward if db is None else db, RewardType.NL_ASSERTION: 1.0}
        info = RewardInfo(reward=reward, reward_breakdown=breakdown)
    return SimulationRun(
        id=f"{task}-{trial}", task_id=task, start_time="t", end_time="t", duration=1.0,
        termination_reason=reason, trial=trial, seed=SEEDS[trial], reward_info=info,
        messages=messages if info else [], agent_cost=agent_cost, user_cost=user_cost,
    )


def session(task, trial, *, llm_cost=0.012, denied_rule=None):
    chain = AuditChain("s", lambda: datetime(2026, 1, 1), HarnessConfig.full())
    if denied_rule:
        chain.append(
            AuditEvent.TOOL_DENIED,
            tool_call=HarnessCall(id="refused-1", name="cancel_pending_order", arguments={}),
            decision=Decision(outcome=DecisionOutcome.DENY, rule=denied_rule, reason="no"),
        )
    return SessionRecord(
        task_id=task, seed=SEEDS[trial], session_id="s", written_at="t", proposals=3,
        llm_cost=llm_cost, chain_intact=True, records=chain.records,
    )


def arm(name, outcomes, *, trials=3, parallel=False, denied_rule=None, infra=()):
    sims, sessions = [], {}
    for task, passes in outcomes.items():
        for trial in range(trials):
            if (task, trial) in infra:
                sims.append(sim(task, trial, 0.0, reason=TerminationReason.INFRASTRUCTURE_ERROR))
                continue
            sims.append(sim(task, trial, float(passes[trial]), parallel=parallel))
            sessions[(task, SEEDS[trial])] = session(task, trial, denied_rule=denied_rule)
    return ArmData(name=name, phase="headline", trials=trials, simulations=tuple(sims),
                   sessions=sessions)


BASE = {"1": [1, 1, 1], "2": [0, 0, 0], "3": [1, 0, 1], "4": [0, 0, 0]}
FULL = {"1": [1, 1, 1], "2": [1, 1, 0], "3": [1, 1, 1], "4": [0, 0, 0]}


def test_rows_join_the_agent_s_records_to_tau2_s_simulations():
    rows = simulation_rows(arm("full", FULL, denied_rule="gating.identity_first"))
    assert len(rows) == 12
    first = rows[0]
    assert first["has_session"] and first["chain_intact"]
    assert first["denials_by_rule"] == {"gating.identity_first": 1}
    assert first["agent_llm_cost"] == pytest.approx(0.012)


def test_an_infrastructure_failure_is_a_row_with_no_outcome_not_a_failure():
    rows = simulation_rows(arm("full", FULL, infra={("2", 2)}))
    lost = next(r for r in rows if r["task_id"] == "2" and r["trial"] == 2)
    assert lost["passed"] is None and lost["termination"] == "infrastructure_error"


def test_the_arm_summary_carries_intervals_exclusions_and_hidden_cost():
    summary = summarise_arm(arm("full", FULL, infra={("2", 2)}), resamples=300)
    assert [e.statistic for e in summary.pass_k] == ["pass^1", "pass^2", "pass^3"]
    assert summary.pass_k[2].skipped_tasks == ("2",)  # lost a trial, so not scored at k=3
    assert summary.excluded == {"infrastructure_error:unknown": 1}
    # Agent spent 0.012 per session; tau2 saw 0.010 of it.
    assert summary.hidden_retry_share == pytest.approx(0.2)


def test_violation_rates_are_labelled_as_violations():
    summary = summarise_arm(arm("baseline", BASE, parallel=True), resamples=300)
    assert summary.parallel_violation_rate.statistic == "parallel_call_violations"
    assert summary.parallel_violation_rate.value == 1.0


def test_the_headline_comparison_is_paired_and_named():
    result = compare_arms(arm("full", FULL), arm("baseline", BASE), resamples=300)
    assert result.statistic == "reward pass^1"
    assert (result.treatment_higher, result.control_higher, result.ties) == (2, 0, 2)


def test_task_shifts_classify_migration():
    shifts = {s.task_id: s for s in task_shifts(arm("full", FULL), arm("baseline", BASE))}
    assert shifts["1"].category == "always → always"
    assert shifts["2"].category == "never → sometimes"
    assert shifts["3"].category == "sometimes → always"
    assert shifts["2"].delta == pytest.approx(2 / 3)


def test_a_trace_is_never_shown_without_what_it_stands_for():
    shifts = task_shifts(arm("full", FULL), arm("baseline", BASE))
    rep = representativeness("2", shifts)
    assert rep.task.task_id == "2"
    assert (rep.tasks_moving_the_same_way, rep.tasks_moving_the_other_way,
            rep.tasks_unchanged) == (1, 0, 2)


def test_first_divergence_finds_where_trajectories_part():
    a = [UserMessage(role="user", content="x"), AssistantMessage(role="assistant", content="y")]
    b = [UserMessage(role="user", content="x"), AssistantMessage(role="assistant", content="z")]
    assert first_divergence(a, b) == 1
    assert first_divergence(a, list(a)) is None
    assert first_divergence(a, a[:1]) == 1


def test_refused_proposals_are_exposed_by_call_id():
    """They never reach the trajectory; this is the part a generic viewer cannot show."""
    events = harness_events_by_call(session("1", 0, denied_rule="validation.refused"))
    assert events["refused-1"][0]["rule"] == "validation.refused"
    assert harness_events_by_call(None) == {}


def test_a_live_run_shows_only_chunks_the_ledger_has_closed(tmp_path, monkeypatch):
    """runner.load_arm globs every results file, including the one tau2 is still
    writing; the dashboard watches live runs, so it must not."""
    import json

    from tau2.data_model.simulation import Results

    from policygate.viz.data import load_run

    arms = [{"phase": "headline", "trials": 5, "config": {"name": n}} for n in ("baseline", "full")]
    (tmp_path / "plan.json").write_text(json.dumps(
        {"fingerprint": "f", "provenance": {}, "plan": {"arms": arms}}))
    (tmp_path / "ledger.json").write_text(json.dumps({
        "plan_fingerprint": "f",
        "complete": {"baseline/chunk_000": {}, "full/chunk_000": {}, "baseline/chunk_001": {}},
        "incomplete": {},
    }))
    for arm, chunks in (("baseline", 2), ("full", 2)):
        (tmp_path / arm).mkdir()
        for i in range(chunks):
            (tmp_path / arm / f"results_chunk_{i:03d}.json").write_text("{ half-writ")

    loaded = []

    class Fake:
        def __init__(self, path):
            self.simulations = [sim(path.name, 0, 1.0)]

    monkeypatch.setattr(Results, "load", classmethod(lambda cls, path: loaded.append(
        f"{path.parent.name}/{path.name}") or Fake(path)))

    run = load_run(tmp_path)
    assert sorted(loaded) == ["baseline/results_chunk_000.json",
                              "baseline/results_chunk_001.json",
                              "full/results_chunk_000.json"]
    assert len(run.arm("full").simulations) == 1
