"""Everything the dashboard shows, computed without Streamlit.

The pages only render. Every number they display comes from a function here,
so it can be tested, and so the same figure cannot be computed two different
ways on two different pages.

Two rules carried over from the analysis modules:

* **Nothing is shown without the data behind it.** Every rate comes from
  :mod:`policygate.eval.passk` with its interval and its exclusions; every
  paired comparison names the tasks it could not compare.
* **No trajectory is shown without its representativeness.** A single trace
  proves only that something happened once. Each trace carries how often its
  task passed in each arm and how many other tasks moved the same way, so the
  viewer cannot quietly become a cherry-picking machine (docs/architecture.md).

Reads the output of :mod:`policygate.eval.runner`: completed chunks only, so a
page reloaded mid-run shows a consistent subset rather than a half-written one.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from policygate.eval.passk import (
    Estimate,
    ExclusionPolicy,
    PairedComparison,
    compare_paired,
    observations_from_simulations,
    pass_k,
)
from policygate.eval.runner import SessionRecord
from policygate.eval.violations import Clause, count_turns, violation_table
from policygate.harness.types import AuditEvent, DecisionOutcome

HEADLINE = ("baseline", "full")


def _reason(simulation: Any) -> str:
    return str(getattr(simulation.termination_reason, "value", simulation.termination_reason))


class ArmData(BaseModel):
    """One arm's completed simulations and the agent's records for them."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    phase: str
    trials: int
    simulations: tuple[Any, ...]
    sessions: Mapping[tuple[str, int], SessionRecord]

    def session_for(self, simulation: Any) -> SessionRecord | None:
        return self.sessions.get((simulation.task_id, simulation.seed))


class RunView(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    fingerprint: str
    provenance: Mapping[str, Any]
    arms: tuple[ArmData, ...]
    ledger: Mapping[str, Any]

    def arm(self, name: str) -> ArmData | None:
        return next((a for a in self.arms if a.name == name), None)

    @property
    def spend(self) -> float:
        reports = [*self.ledger.get("complete", {}).values(),
                   *self.ledger.get("incomplete", {}).values()]
        return sum(r["agent_llm_cost"] + r["user_cost"] for r in reports)


def load_run(out_dir: Path) -> RunView:
    """Read a runner output directory. Chunks still in flight are not included."""
    from tau2.data_model.simulation import Results

    from policygate.eval.runner import kept_sessions, read_sessions

    stored = json.loads((out_dir / "plan.json").read_text())
    ledger_path = out_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    # Only chunks the ledger has closed. eval.runner.load_arm globs every results
    # file, including one tau2 is still writing: during a live run that shows a
    # partial chunk (biasing pass^k for the arm in flight) or a half-written file.
    closed = {*ledger.get("complete", {}), *ledger.get("incomplete", {})}
    arms = []
    for arm in stored["plan"]["arms"]:
        name = arm["config"]["name"]
        simulations = [
            sim
            for key in sorted(k for k in closed if k.startswith(f"{name}/"))
            for sim in Results.load(out_dir / name / f"results_{key.split('/', 1)[1]}.json")
            .simulations
        ]
        sessions = kept_sessions(read_sessions(out_dir / name / "sessions.jsonl"))
        arms.append(
            ArmData(
                name=name,
                phase=arm["phase"],
                trials=arm["trials"],
                simulations=tuple(simulations),
                sessions=sessions,
            )
        )
    return RunView(
        fingerprint=stored["fingerprint"],
        provenance=stored["provenance"],
        arms=tuple(arms),
        ledger=ledger,
    )


# --- one row per simulation -----------------------------------------------


def simulation_rows(arm: ArmData) -> list[dict[str, Any]]:
    """The flat table every other view is derived from."""
    rows = []
    for sim in arm.simulations:
        session = arm.session_for(sim)
        reward = sim.reward_info.reward if sim.reward_info is not None else None
        breakdown = {
            str(getattr(k, "value", k)): v
            for k, v in ((sim.reward_info.reward_breakdown or {}).items()
                         if sim.reward_info is not None else ())
        }
        turns = count_turns(sim) if sim.messages else None
        denials = Counter()
        if session is not None:
            denials = Counter(
                r.decision.rule
                for r in session.records
                if r.event is AuditEvent.TOOL_DENIED
                and r.decision is not None
                and r.decision.outcome is not DecisionOutcome.ALLOW
            )
        rows.append(
            {
                "arm": arm.name,
                "task_id": sim.task_id,
                "trial": sim.trial,
                "seed": sim.seed,
                "termination": _reason(sim),
                "reward": reward,
                "passed": None if reward is None else reward >= 1 - 1e-9,
                "db": breakdown.get("DB"),
                "nl_assertion": breakdown.get("NL_ASSERTION"),
                "assistant_turns": None if turns is None else turns.assistant_turns,
                "parallel_turns": None if turns is None else turns.parallel_turns,
                "mixed_turns": None if turns is None else turns.mixed_turns,
                "tau2_agent_cost": sim.agent_cost or 0.0,
                "user_cost": sim.user_cost or 0.0,
                "agent_llm_cost": None if session is None else session.llm_cost,
                "proposals": None if session is None else session.proposals,
                "denials": sum(denials.values()),
                "denials_by_rule": dict(denials),
                "chain_intact": None if session is None else session.chain_intact,
                "has_session": session is not None,
            }
        )
    return rows


# --- headline -------------------------------------------------------------


class ArmSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    arm: str
    simulations: int
    excluded: Mapping[str, int]
    pass_k: tuple[Estimate, ...]
    """pass^1 .. pass^trials, each with its interval."""

    db_pass_1: Estimate | None
    parallel_violation_rate: Estimate | None
    mixed_violation_rate: Estimate | None
    cost_per_simulation: float | None
    """Agent spend including refused proposals, plus the user simulator."""

    hidden_retry_share: float | None
    """Agent spend tau2 never saw, as a share of what it did see."""


def summarise_arm(arm: ArmData, *, resamples: int = 2000, seed: int = 0) -> ArmSummary:
    table = observations_from_simulations(arm.simulations)
    by_task = table.by_task()
    # Up to the most trials any task kept: pass_k names the tasks too short for
    # a given k rather than letting one task's lost trial cap every estimate.
    max_k = max((len(v) for v in by_task.values()), default=0)
    estimates = tuple(pass_k(table, k, resamples=resamples, seed=seed) for k in range(1, max_k + 1))

    db_table = observations_from_simulations(arm.simulations, component="DB")
    db = pass_k(db_table, 1, resamples=resamples, seed=seed) if db_table.observations else None

    def violation(clause: Clause, name: str) -> Estimate | None:
        vt = violation_table(arm.simulations, clause)
        if not vt.observations:
            return None
        return pass_k(vt, 1, resamples=resamples, seed=seed).model_copy(update={"statistic": name})

    rows = simulation_rows(arm)
    with_sessions = [r for r in rows if r["has_session"]]
    cost = None
    hidden = None
    if with_sessions:
        agent = sum(r["agent_llm_cost"] for r in with_sessions)
        tau2 = sum(r["tau2_agent_cost"] for r in with_sessions)
        user = sum(r["user_cost"] for r in with_sessions)
        cost = (agent + user) / len(with_sessions)
        hidden = (agent - tau2) / tau2 if tau2 > 0 else None

    return ArmSummary(
        arm=arm.name,
        simulations=len(arm.simulations),
        excluded=dict(table.excluded),
        pass_k=estimates,
        db_pass_1=db,
        parallel_violation_rate=violation(Clause.PARALLEL_TOOL_CALLS, "parallel_call_violations"),
        mixed_violation_rate=violation(Clause.TEXT_WITH_TOOL_CALL, "text_with_tool_call"),
        cost_per_simulation=cost,
        hidden_retry_share=hidden,
    )


def compare_arms(
    treatment: ArmData,
    control: ArmData,
    *,
    k: int = 1,
    component: str | None = None,
    policy: ExclusionPolicy = ExclusionPolicy.EXCLUDE,
    resamples: int = 2000,
    seed: int = 0,
) -> PairedComparison:
    return compare_paired(
        observations_from_simulations(treatment.simulations, component=component, policy=policy),
        observations_from_simulations(control.simulations, component=component, policy=policy),
        k=k,
        name=f"{component or 'reward'} pass^{k}",
        resamples=resamples,
        seed=seed,
    )


# --- per task: migration and representativeness ---------------------------


class TaskShift(BaseModel):
    """How one task moved from control to treatment."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    control_passed: int
    control_trials: int
    treatment_passed: int
    treatment_trials: int

    @property
    def control_rate(self) -> float:
        return self.control_passed / self.control_trials

    @property
    def treatment_rate(self) -> float:
        return self.treatment_passed / self.treatment_trials

    @property
    def delta(self) -> float:
        return self.treatment_rate - self.control_rate

    @property
    def category(self) -> str:
        """Coarse migration class. Stability needs repeated trials to mean anything."""
        def label(passed: int, trials: int) -> str:
            if passed == trials:
                return "always"
            if passed == 0:
                return "never"
            return "sometimes"

        return (
            f"{label(self.control_passed, self.control_trials)}"
            f" → {label(self.treatment_passed, self.treatment_trials)}"
        )


def task_shifts(treatment: ArmData, control: ArmData) -> list[TaskShift]:
    """Per-task pass counts in both arms, on tasks both arms scored."""
    t = observations_from_simulations(treatment.simulations).by_task()
    c = observations_from_simulations(control.simulations).by_task()
    return [
        TaskShift(
            task_id=task,
            control_passed=sum(c[task]),
            control_trials=len(c[task]),
            treatment_passed=sum(t[task]),
            treatment_trials=len(t[task]),
        )
        for task in sorted(set(t) & set(c), key=_task_order)
    ]


def _task_order(task_id: str) -> tuple[int, str]:
    return (int(task_id), task_id) if task_id.isdigit() else (10**9, task_id)


class Representativeness(BaseModel):
    """What a single paired trace stands for."""

    model_config = ConfigDict(frozen=True)

    task: TaskShift
    tasks_moving_the_same_way: int
    tasks_moving_the_other_way: int
    tasks_unchanged: int


def representativeness(task_id: str, shifts: Sequence[TaskShift]) -> Representativeness:
    shift = next(s for s in shifts if s.task_id == task_id)
    sign = (shift.delta > 0) - (shift.delta < 0)
    others = [s for s in shifts if s.task_id != task_id]

    def direction(s: TaskShift) -> int:
        return (s.delta > 1e-12) - (s.delta < -1e-12)

    return Representativeness(
        task=shift,
        tasks_moving_the_same_way=sum(1 for s in others if sign and direction(s) == sign),
        tasks_moving_the_other_way=sum(1 for s in others if sign and direction(s) == -sign),
        tasks_unchanged=sum(1 for s in others if direction(s) == 0),
    )


# --- one trajectory against another ---------------------------------------


def _message_signature(message: Any) -> tuple:
    calls = tuple(
        (c.name, json.dumps(c.arguments, sort_keys=True))
        for c in (getattr(message, "tool_calls", None) or [])
    )
    content = getattr(message, "content", None)
    return (getattr(message, "role", None), (content or "").strip(), calls)


def first_divergence(a: Sequence[Any], b: Sequence[Any]) -> int | None:
    """Index of the first message whose role, text or tool calls differ. None if identical."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if _message_signature(x) != _message_signature(y):
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def harness_events_by_call(session: SessionRecord | None) -> dict[str, list[dict[str, Any]]]:
    """Audit decisions grouped by tool-call id, for inlining next to the trajectory.

    Refused proposals never reach the trajectory, so events whose call id does
    not appear in it are the harness's invisible interventions -- the part a
    generic trace viewer cannot show.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    if session is None:
        return grouped
    for record in session.records:
        key = record.tool_call.id if record.tool_call is not None else "(no call)"
        grouped.setdefault(key, []).append(
            {
                "seq": record.seq,
                "event": record.event.value,
                "tool": record.tool_call.name if record.tool_call is not None else None,
                "arguments": dict(record.tool_call.arguments) if record.tool_call else None,
                "rule": record.decision.rule if record.decision is not None else None,
                "reason": record.decision.reason if record.decision is not None else None,
            }
        )
    return grouped
