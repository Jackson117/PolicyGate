"""Tests for eval/runner.py.

Two layers. The scheduling -- chunk order, budget, infrastructure stops, resume,
the plan guard -- is driven with a fake executor, because those decisions are
what a real run cannot afford to get wrong and cannot be tested by spending
money. The join -- that the agent's records land on disk keyed so they match
tau2's SimulationRun -- is driven through tau2's real orchestrator with both
LLMs scripted, because it depends on hooks tau2 calls, and only running them
shows they are called.
"""

import json

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.utils.utils import DATA_DIR

from policygate.eval import runner
from policygate.eval.runner import (
    ChunkReport,
    ChunkSpec,
    Ledger,
    PlanMismatchError,
    RunPlan,
    RunStatus,
    SessionRecord,
    Tau2ChunkExecutor,
    kept_sessions,
    measure_chunk,
    read_sessions,
    run_plan,
)
from policygate.harness.config import HarnessConfig

TASKS = tuple(str(i) for i in range(10))
REAL_PROVENANCE = runner._provenance


def clean(code="code-v1"):
    return lambda: {"started_at": "t", "commit": "c", "tau2_commit": "t2",
                    "code": code, "code_dirty": False}


@pytest.fixture(autouse=True)
def committed_code(monkeypatch):
    """Scheduling tests must not depend on whether this checkout has uncommitted work."""
    monkeypatch.setattr(runner, "_provenance", clean())


def plan(**overrides) -> RunPlan:
    defaults = dict(
        agent_llm="openai/gpt-4.1-mini",
        user_llm="openai/gpt-4.1-mini",
        judge_llm="openai/gpt-4.1",
        chunk_size=4,
        name="test",
    )
    return RunPlan.w4(TASKS, **(defaults | overrides))


class FakeExecutor:
    """Returns a scripted report per chunk and records what was asked of it."""

    def __init__(self, cost_per_simulation=0.01, failures=None):
        self.cost = cost_per_simulation
        self.failures = failures or {}
        self.calls: list[str] = []

    def __call__(self, spec: ChunkSpec) -> ChunkReport:
        self.calls.append(spec.key)
        remaining = self.failures.get(spec.key, [])
        failed = remaining.pop(0) if remaining else 0
        return ChunkReport(
            key=spec.key,
            simulations=spec.simulations,
            infrastructure_errors=failed,
            agent_llm_cost=spec.simulations * self.cost * 0.8,
            user_cost=spec.simulations * self.cost * 0.2,
        )


# --- the plan -------------------------------------------------------------


def test_w4_is_baseline_full_and_one_arm_per_mechanism():
    p = plan()
    names = [a.name for a in p.arms]
    switches = HarnessConfig.full().enabled_ablation_switches()
    assert names == ["baseline", "full", *(f"only_{s}" for s in switches)]
    assert [a.trials for a in p.arms] == [5, 5] + [1] * len(switches)
    assert all(a.config.audit_enabled for a in p.arms if a.phase == "attribution")
    assert p.simulations == len(TASKS) * (5 + 5 + len(switches))


def test_no_arm_turns_on_the_approval_gate():
    assert not any(a.config.approval_gate_enabled for a in plan().arms)


def test_tasks_are_shuffled_reproducibly_and_none_are_lost():
    a, b = plan(), plan()
    assert a.task_ids == b.task_ids
    assert sorted(a.task_ids) == sorted(TASKS)
    assert a.task_ids != TASKS
    assert plan(shuffle_seed=1).task_ids != a.task_ids


def test_the_fingerprint_tracks_what_the_run_measures():
    assert plan().fingerprint() == plan().fingerprint()
    assert plan(headline_trials=3).fingerprint() != plan().fingerprint()
    assert plan(agent_llm="openai/gpt-4.1").fingerprint() != plan().fingerprint()


def test_headline_chunks_come_first_and_alternate_between_arms():
    schedule = plan().schedule()
    phases = [plan().arm(c.arm).phase for c in schedule]
    assert phases == sorted(phases, key=lambda ph: ph != "headline")
    headline = [c.key for c in schedule if plan().arm(c.arm).phase == "headline"]
    assert headline[:4] == [
        "baseline/chunk_000",
        "full/chunk_000",
        "baseline/chunk_001",
        "full/chunk_001",
    ]


def test_every_arm_covers_every_task_exactly_once():
    p = plan()
    for arm in p.arms:
        covered = [t for c in p.schedule() if c.arm == arm.name for t in c.task_ids]
        assert sorted(covered) == sorted(TASKS)


def test_a_malformed_plan_is_refused():
    with pytest.raises(ValueError, match="unique"):
        RunPlan.w4(("1", "1"), agent_llm="a", user_llm="b", judge_llm="c")


# --- running --------------------------------------------------------------


def test_a_run_within_budget_completes_and_records_every_chunk(tmp_path):
    executor = FakeExecutor()
    outcome = run_plan(plan(), tmp_path, budget_usd=100, executor=executor)
    assert outcome.status is RunStatus.COMPLETE
    assert len(executor.calls) == len(plan().schedule())
    ledger = Ledger.load(tmp_path / "ledger.json", plan().fingerprint())
    assert set(ledger.complete) == {c.key for c in plan().schedule()}
    assert outcome.spend == pytest.approx(plan().simulations * 0.01)
    assert "nl_assertion_judge" in outcome.unaccounted


def test_a_second_invocation_resumes_without_re_running(tmp_path):
    run_plan(plan(), tmp_path, budget_usd=100, executor=FakeExecutor())
    again = FakeExecutor()
    outcome = run_plan(plan(), tmp_path, budget_usd=100, executor=again)
    assert again.calls == []
    assert len(outcome.already_complete) == len(plan().schedule())


def test_the_budget_stops_the_run_before_a_chunk_that_would_exceed_it(tmp_path):
    executor = FakeExecutor(cost_per_simulation=0.01)
    # baseline chunk 0 costs 4 x 5 x 0.01 = 0.20; so does full chunk 0.
    outcome = run_plan(
        plan(), tmp_path, budget_usd=0.45, executor=executor, prior_cost_per_simulation=0.01
    )
    assert outcome.status is RunStatus.BUDGET_EXHAUSTED
    assert executor.calls == ["baseline/chunk_000", "full/chunk_000"]
    assert outcome.stopped_before == "baseline/chunk_001"
    assert outcome.spend <= 0.45


def test_raising_the_budget_continues_where_it_stopped(tmp_path):
    run_plan(plan(), tmp_path, budget_usd=0.45, executor=FakeExecutor(),
             prior_cost_per_simulation=0.01)
    executor = FakeExecutor()
    outcome = run_plan(plan(), tmp_path, budget_usd=100, executor=executor)
    assert outcome.status is RunStatus.COMPLETE
    assert executor.calls[0] == "baseline/chunk_001"


def test_the_estimate_switches_to_measured_cost_once_there_is_some(tmp_path):
    """A prior that is ten times too optimistic must not carry the run past its budget."""
    executor = FakeExecutor(cost_per_simulation=0.1)
    outcome = run_plan(
        plan(), tmp_path, budget_usd=3.0, executor=executor, prior_cost_per_simulation=0.01
    )
    assert outcome.status is RunStatus.BUDGET_EXHAUSTED
    assert executor.calls == ["baseline/chunk_000"]  # $2 spent; next estimated at $2


def test_a_chunk_with_a_few_infrastructure_failures_is_repaired(tmp_path):
    executor = FakeExecutor(failures={"baseline/chunk_000": [3, 0]})
    outcome = run_plan(plan(), tmp_path, budget_usd=100, executor=executor)
    assert outcome.status is RunStatus.COMPLETE
    assert executor.calls[:3] == ["baseline/chunk_000", "baseline/chunk_000", "full/chunk_000"]


def test_persistent_infrastructure_failure_stops_the_run_and_keeps_its_spend(tmp_path):
    executor = FakeExecutor(failures={"full/chunk_000": [15, 12]})
    outcome = run_plan(plan(), tmp_path, budget_usd=100, executor=executor)
    assert outcome.status is RunStatus.INFRASTRUCTURE_FAILURE
    assert outcome.stopped_before == "full/chunk_000"
    ledger = Ledger.load(tmp_path / "ledger.json", plan().fingerprint())
    assert "full/chunk_000" in ledger.incomplete
    assert "full/chunk_000" not in ledger.complete
    assert ledger.spend() == pytest.approx(0.4)  # both chunks' spend counts


def test_a_failed_chunk_is_retried_on_the_next_invocation(tmp_path):
    run_plan(plan(), tmp_path, budget_usd=100,
             executor=FakeExecutor(failures={"full/chunk_000": [15, 12]}))
    executor = FakeExecutor()
    outcome = run_plan(plan(), tmp_path, budget_usd=100, executor=executor)
    assert outcome.status is RunStatus.COMPLETE
    assert executor.calls[0] == "full/chunk_000"
    ledger = Ledger.load(tmp_path / "ledger.json", plan().fingerprint())
    assert ledger.incomplete == {}


def test_a_different_plan_is_refused_in_the_same_directory(tmp_path):
    """tau2's auto_resume would only warn and append; the runner refuses."""
    run_plan(plan(), tmp_path, budget_usd=0.1, executor=FakeExecutor())
    with pytest.raises(PlanMismatchError):
        run_plan(plan(headline_trials=3), tmp_path, budget_usd=100, executor=FakeExecutor())


def test_the_plan_file_records_provenance(tmp_path):
    run_plan(plan(), tmp_path, budget_usd=0.1, executor=FakeExecutor())
    stored = json.loads((tmp_path / "plan.json").read_text())
    assert stored["fingerprint"] == plan().fingerprint()
    assert {"commit", "tau2_commit", "code", "code_dirty", "started_at"} <= set(
        stored["provenance"]
    )


def test_resuming_with_different_code_is_refused(tmp_path):
    """W4 v1: the plan fingerprint does not cover code, so a harness fix followed
    by a resume in the same directory would have mixed two versions' chunks."""
    run_plan(plan(), tmp_path, budget_usd=0.02, executor=FakeExecutor(), provenance=clean())
    executor = FakeExecutor()
    with pytest.raises(PlanMismatchError, match="different code"):
        run_plan(plan(), tmp_path, budget_usd=100, executor=executor,
                 provenance=clean("code-v2"))
    assert executor.calls == []


def test_resuming_with_the_same_code_proceeds(tmp_path):
    run_plan(plan(), tmp_path, budget_usd=0.02, executor=FakeExecutor(), provenance=clean())
    outcome = run_plan(plan(), tmp_path, budget_usd=100, executor=FakeExecutor(),
                       provenance=clean())
    assert outcome.status is RunStatus.COMPLETE


@pytest.mark.parametrize("identity", [
    {"code": "code-v1", "code_dirty": True},
    {"code": None, "code_dirty": False},
])
def test_uncommitted_or_unidentifiable_code_does_not_start(tmp_path, identity):
    executor = FakeExecutor()
    with pytest.raises(PlanMismatchError, match="uncommitted or unidentifiable"):
        run_plan(plan(), tmp_path, budget_usd=100, executor=executor,
                 provenance=lambda: {**clean()(), **identity})
    assert executor.calls == [] and not (tmp_path / "plan.json").exists()


def test_untracked_files_do_not_make_the_code_dirty(tmp_path, monkeypatch):
    """v1 was recorded dirty because its own run.log was untracked."""
    seen = []

    def git(*args, cwd=None):
        seen.append(args)
        return "" if args[0] == "status" else "abc"

    monkeypatch.setattr(runner, "_git", git)
    identity = REAL_PROVENANCE()
    assert identity["code_dirty"] is False and identity["code"]
    statuses = [a for a in seen if a[0] == "status"]
    assert statuses and all("--untracked-files=no" in a for a in statuses)


# --- measuring a chunk ----------------------------------------------------


def sim(task_id, trial, seed, *, reason=TerminationReason.USER_STOP, user_cost=0.001):
    return SimulationRun(
        id=f"{task_id}-{trial}", task_id=task_id, start_time="t", end_time="t", duration=1.0,
        termination_reason=reason, trial=trial, seed=seed, user_cost=user_cost,
    )


def session(task_id, seed, cost, proposals=1):
    return SessionRecord(
        task_id=task_id, seed=seed, session_id="s", written_at="t", proposals=proposals,
        llm_cost=cost, chain_intact=True, records=(),
    )


def test_a_chunk_is_measured_from_its_own_tasks_only():
    spec = ChunkSpec(arm="full", index=0, task_ids=("1", "2"), trials=1)
    report = measure_chunk(
        spec,
        [sim("1", 0, 7), sim("2", 0, 7, reason=TerminationReason.INFRASTRUCTURE_ERROR),
         sim("9", 0, 7)],
        [session("1", 7, 0.01), session("1", 7, 0.02), session("2", 7, 0.03),
         session("9", 7, 5.0)],
    )
    assert (report.simulations, report.infrastructure_errors) == (2, 1)
    # Both attempts at task 1 cost money; task 9 belongs to another chunk.
    assert report.agent_llm_cost == pytest.approx(0.06)
    assert report.user_cost == pytest.approx(0.002)


def test_the_last_attempt_is_the_one_kept():
    kept = kept_sessions([session("1", 7, 0.01, proposals=3), session("1", 7, 0.02, proposals=5)])
    assert kept[("1", 7)].proposals == 5


# --- through tau2's real orchestrator, offline ----------------------------


def test_records_join_to_the_simulation_tau2_produced(tmp_path, monkeypatch):
    """set_seed and stop are tau2's hooks; only running the orchestrator proves they fire."""
    import tau2.user.user_simulator as user_module
    from tau2.registry import registry
    from tau2.runner.build import build_text_orchestrator

    import policygate.adapters.tau2_agent as agent_module

    user = next(iter(json.loads(
        (DATA_DIR / "tau2" / "domains" / "retail" / "db.json").read_text()
    )["users"].values()))
    agent_replies = iter([
        AssistantMessage(
            role="assistant", content=None, cost=0.002,
            tool_calls=[ToolCall(id="t1", name="find_user_id_by_email",
                                 arguments={"email": user["email"]})],
        ),
        *[AssistantMessage(role="assistant", content="How can I help?", cost=0.002)] * 5,
    ])
    user_replies = iter(["I need help with an order.", "###STOP###"])
    monkeypatch.setattr(agent_module, "generate", lambda **kw: next(agent_replies))
    monkeypatch.setattr(
        user_module,
        "generate",
        lambda **kw: AssistantMessage(role="assistant", content=next(user_replies), cost=0.001),
    )

    p = plan(chunk_size=10)
    executor = Tau2ChunkExecutor(p, tmp_path, max_concurrency=1)
    spec = next(c for c in p.schedule() if c.arm == "full")
    config = executor.run_config(spec)
    assert config.auto_resume is True
    assert config.hallucination_retries == 0
    assert config.enforce_communication_protocol is False

    task = next(t for t in registry.get_tasks_loader("retail")(None) if t.id == spec.task_ids[0])
    simulation = build_text_orchestrator(config, task, seed=12345).run()

    records = read_sessions(executor.sessions_path("full"))
    assert len(records) == 1  # stop() may be called twice; one line per attempt
    kept = kept_sessions(records)
    assert (simulation.task_id, simulation.seed) in kept
    record = kept[(simulation.task_id, simulation.seed)]
    assert record.proposals == 2
    assert record.llm_cost == pytest.approx(0.004)
    assert record.chain_intact
    proposed = [r.tool_call.name for r in record.records if r.tool_call is not None]
    assert proposed == ["find_user_id_by_email"]
    # The environment really ran the lookup, and the harness learned from it.
    assert any(
        getattr(m, "role", None) == "tool" and user["user_id"] in (m.content or "")
        for m in simulation.messages
    )


def test_an_attempt_is_written_once_however_often_stop_is_called(tmp_path, monkeypatch):
    """tau2's error path can reach stop() from both finaliser and cleanup."""
    from tau2.registry import registry

    p = plan(chunk_size=10)
    executor = Tau2ChunkExecutor(p, tmp_path, max_concurrency=1)
    factory = registry.get_agent_factory(executor.run_config(p.schedule()[0]).agent)
    env = registry.get_env_constructor("retail")()
    task = registry.get_tasks_loader("retail")(None)[0]
    agent = factory(tools=env.get_tools(), domain_policy=env.get_policy(), llm="x", task=task)
    agent.set_seed(3)
    agent.get_init_state()
    agent.stop()
    agent.stop()
    assert len(read_sessions(executor.sessions_path("baseline"))) == 1


def test_a_missing_seed_is_written_visibly_rather_than_raised(tmp_path):
    from tau2.registry import registry

    from policygate.eval.runner import MISSING_SEED

    p = plan(chunk_size=10)
    executor = Tau2ChunkExecutor(p, tmp_path, max_concurrency=1)
    factory = registry.get_agent_factory(executor.run_config(p.schedule()[0]).agent)
    env = registry.get_env_constructor("retail")()
    task = registry.get_tasks_loader("retail")(None)[0]
    agent = factory(tools=env.get_tools(), domain_policy=env.get_policy(), llm="x", task=task)
    agent.stop()
    assert read_sessions(executor.sessions_path("baseline"))[0].seed == MISSING_SEED
