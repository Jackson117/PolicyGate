"""Run an ablation plan through tau2 in resumable, budgeted chunks.

W2's smoke script ran twelve simulations in one call. W4 runs thousands, and the
smoke already showed the three ways a run that size goes wrong: the provider
starts failing after the conversation has been paid for, a crash loses what the
agent recorded, and nothing stops the spend. This module is the answer to each.

**Chunks, because tau2 cannot be stopped from outside.** Its batch loop owns the
executor, keeps its shutdown event private, and on interrupt calls
``os._exit``. So each arm is split into chunks, each chunk is its own tau2 run
with ``auto_resume``, and every decision -- continue, stop for budget, stop for
failing infrastructure -- is taken between chunks. Trial seeds depend only on
the run seed and trial count, so a task's trial has the same seed in every chunk
and every arm.

**A plan fingerprint, because tau2's resume does not check the config.** With
``auto_resume`` on, a changed config only logs a warning and the new results are
appended to the old file. The plan is written once, and a later invocation with a
different plan is refused rather than mixed in.

**The ledger is recomputed from files, not accumulated in memory.** A chunk's
report is measured from what is on disk -- tau2's results and the agent's own
session records -- so re-running a chunk after a crash replaces its entry
instead of double-counting it, and spend already incurred by a failed chunk
still counts against the budget.

**The agent writes its own records.** The audit chain and the cost of refused
proposals live in the agent's state, which tau2 does not persist. The recording
agent learns its seed from ``set_seed`` and writes one line in ``stop``, keyed by
``(task_id, seed)`` -- unique within an arm, and the join key back to
``SimulationRun``. A retried simulation leaves several lines; the last is the
attempt tau2 kept, and every attempt's cost counts.

**Order is chosen for early stopping.** Headline arms run before attribution
arms and chunks alternate between arms, so a run stopped by the budget leaves
paired data on the same tasks. Tasks are shuffled with a fixed seed so that the
tasks finished first are not a biased prefix of retail's own ordering.

Unaccounted, and said so in every report: the NL-assertion judge's spend, and
the user simulator's spend on attempts tau2 retried and discarded.
"""

from __future__ import annotations

import json
import random
import subprocess
import threading
import uuid
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from policygate.harness.audit import verify_records
from policygate.harness.config import HarnessConfig
from policygate.harness.types import AuditRecord, canonical_json, sha256_hex

PRIOR_COST_PER_SIMULATION = 0.0137
"""W2 smoke, full arm, OpenAI list rates, judge excluded. Replaced by measurement."""

MISSING_SEED = -1
"""Written when tau2 never called ``set_seed``; such a record joins to nothing."""

_SINKS: dict[str, SessionSink] = {}
"""Registered agent name -> where its records go. Looked up per call, not captured."""

UNACCOUNTED = (
    "nl_assertion_judge",
    "user_simulator_on_discarded_retries",
)


# --- the plan -------------------------------------------------------------


class ArmPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    config: HarnessConfig
    trials: int = Field(gt=0)
    phase: Literal["headline", "attribution"]

    @property
    def name(self) -> str:
        return self.config.name


class ChunkSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    arm: str
    index: int
    task_ids: tuple[str, ...]
    trials: int

    @property
    def key(self) -> str:
        return f"{self.arm}/chunk_{self.index:03d}"

    @property
    def simulations(self) -> int:
        return len(self.task_ids) * self.trials


class RunPlan(BaseModel):
    """Everything that determines what a run measures. Its fingerprint guards resume.

    Operational knobs -- budget, concurrency -- are deliberately not fields:
    raising the budget between invocations must not look like a different
    experiment.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    domain: Literal["retail"] = "retail"
    task_ids: tuple[str, ...]
    arms: tuple[ArmPlan, ...]
    agent_llm: str
    user_llm: str
    judge_llm: str
    llm_args_agent: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.0})
    llm_args_user: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.0})
    seed: int = 42
    max_steps: int = 100
    max_internal_attempts: int = 3
    chunk_size: int = Field(default=20, gt=0)
    hallucination_retries: int = 0
    enforce_communication_protocol: bool = False

    @model_validator(mode="after")
    def _well_formed(self) -> RunPlan:
        if not self.task_ids:
            raise ValueError("a plan needs at least one task")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task ids must be unique")
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError(f"arm names must be unique: {names}")
        return self

    @classmethod
    def w4(
        cls,
        task_ids: Sequence[str],
        *,
        agent_llm: str,
        user_llm: str,
        judge_llm: str,
        headline_trials: int = 5,
        attribution_trials: int = 1,
        shuffle_seed: int = 0,
        name: str = "w4",
        **overrides: Any,
    ) -> RunPlan:
        """The design in docs/w4_design.md.

        Baseline and full at ``headline_trials``; one arm per mechanism switched
        on alone at ``attribution_trials``. Attribution arms keep the audit chain
        on: they are too small for significance, so which rule fired on which
        task is the evidence they exist to produce.
        """
        order = list(task_ids)
        random.Random(shuffle_seed).shuffle(order)
        arms = [
            ArmPlan(config=HarnessConfig.baseline(), trials=headline_trials, phase="headline"),
            ArmPlan(config=HarnessConfig.full(), trials=headline_trials, phase="headline"),
            *(
                ArmPlan(
                    config=HarnessConfig.only(switch, audit_enabled=True),
                    trials=attribution_trials,
                    phase="attribution",
                )
                for switch in HarnessConfig.full().enabled_ablation_switches()
            ),
        ]
        return cls(
            name=name,
            task_ids=tuple(order),
            arms=tuple(arms),
            agent_llm=agent_llm,
            user_llm=user_llm,
            judge_llm=judge_llm,
            **overrides,
        )

    def fingerprint(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))

    def arm(self, name: str) -> ArmPlan:
        return next(a for a in self.arms if a.name == name)

    def task_chunks(self) -> list[tuple[str, ...]]:
        size = self.chunk_size
        return [self.task_ids[i : i + size] for i in range(0, len(self.task_ids), size)]

    def schedule(self) -> list[ChunkSpec]:
        """Headline before attribution; within a phase, chunk by chunk across arms."""
        order: list[ChunkSpec] = []
        for phase in ("headline", "attribution"):
            arms = [a for a in self.arms if a.phase == phase]
            for index, tasks in enumerate(self.task_chunks()):
                order.extend(
                    ChunkSpec(arm=a.name, index=index, task_ids=tasks, trials=a.trials)
                    for a in arms
                )
        return order

    @property
    def simulations(self) -> int:
        return sum(len(self.task_ids) * a.trials for a in self.arms)


# --- measurement ----------------------------------------------------------


class ChunkReport(BaseModel):
    """What one chunk cost and how much of it failed, measured from disk."""

    model_config = ConfigDict(frozen=True)

    key: str
    simulations: int
    """Results tau2 holds for the chunk, infrastructure failures included."""

    infrastructure_errors: int
    agent_llm_cost: float
    """Every proposal the agent made, refused ones and discarded retries included."""

    user_cost: float
    """As tau2 records it on the simulations it kept."""

    @property
    def spend(self) -> float:
        return self.agent_llm_cost + self.user_cost

    @property
    def infrastructure_share(self) -> float:
        return self.infrastructure_errors / self.simulations if self.simulations else 1.0


ChunkExecutor = Callable[[ChunkSpec], ChunkReport]


class Ledger(BaseModel):
    """Per-chunk reports. Rewritten atomically after every chunk."""

    plan_fingerprint: str
    complete: dict[str, ChunkReport] = Field(default_factory=dict)
    incomplete: dict[str, ChunkReport] = Field(default_factory=dict)
    """Chunks stopped by failing infrastructure. Their spend still counts."""

    @classmethod
    def load(cls, path: Path, fingerprint: str) -> Ledger:
        if not path.exists():
            return cls(plan_fingerprint=fingerprint)
        ledger = cls.model_validate_json(path.read_text())
        if ledger.plan_fingerprint != fingerprint:
            raise PlanMismatchError(f"{path} belongs to a different plan")
        return ledger

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2))
        tmp.replace(path)

    def spend(self) -> float:
        return sum(r.spend for r in (*self.complete.values(), *self.incomplete.values()))

    def cost_per_simulation(self) -> float | None:
        sims = sum(r.simulations for r in self.complete.values())
        if not sims:
            return None
        return sum(r.spend for r in self.complete.values()) / sims


class PlanMismatchError(RuntimeError):
    """An output directory already holds a run of a different plan."""


class RunStatus(StrEnum):
    COMPLETE = "complete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


class RunOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: RunStatus
    executed: tuple[str, ...]
    already_complete: tuple[str, ...]
    spend: float
    stopped_before: str | None = None
    detail: str | None = None
    unaccounted: tuple[str, ...] = UNACCOUNTED


RESULT_PATHS = ("src/policygate/harness", "src/policygate/adapters", "src/policygate/eval",
                "uv.lock")
"""What decides a simulation's outcome. viz, service and docs can change mid-run."""

TAU2_CHECKOUT = "third_party/tau2-bench"


def _git(*args: str, cwd: str | None = None) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10, cwd=cwd
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _provenance() -> dict[str, Any]:
    """Where the run's code came from.

    ``code`` identifies only what decides outcomes: the committed trees of
    RESULT_PATHS and the tau2 checkout's commit. The plan fingerprint does not
    cover code, so W4 v1 could have been resumed after a harness fix with old
    and new chunks silently mixed; ``code`` is what the resume check compares.

    ``code_dirty`` ignores untracked files -- v1 was marked dirty by its own
    run.log -- and anything outside RESULT_PATHS.
    """
    root = _git("rev-parse", "--show-toplevel") or "."
    checkout = str(Path(root) / TAU2_CHECKOUT)
    trees = [_git("rev-parse", f"HEAD:{path}", cwd=root) for path in RESULT_PATHS]
    tau2 = _git("rev-parse", "HEAD", cwd=checkout)
    local = _git("status", "--porcelain", "--untracked-files=no", "--", *RESULT_PATHS, cwd=root)
    # tau2's own uv.lock is rewritten by `uv sync` (a version-string bump on this
    # checkout) and resolves nothing here: the environment comes from the root
    # uv.lock, which is in RESULT_PATHS.
    vendored = _git("status", "--porcelain", "--untracked-files=no", "--", ".",
                    ":(exclude)uv.lock", cwd=checkout)
    code = None
    if all(trees) and tau2 and local is not None and vendored is not None:
        code = sha256_hex("\n".join([*trees, tau2]))
    return {
        "started_at": datetime.now(UTC).isoformat(),
        "commit": _git("rev-parse", "HEAD", cwd=root),
        "tau2_commit": tau2,
        "code": code,
        "code_dirty": bool(local) or bool(vendored),
    }


def _claim_directory(
    plan: RunPlan, out_dir: Path, provenance: Callable[[], dict[str, Any]] = _provenance
) -> None:
    current = provenance()
    if current.get("code") is None or current.get("code_dirty"):
        raise PlanMismatchError(
            "the code that decides outcomes is uncommitted or unidentifiable "
            f"({', '.join(RESULT_PATHS)}, {TAU2_CHECKOUT}); commit it before running, "
            "or the results cannot be traced to a version"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "plan.json"
    if plan_path.exists():
        stored = json.loads(plan_path.read_text())
        if stored.get("fingerprint") != plan.fingerprint():
            raise PlanMismatchError(
                f"{out_dir} holds a different plan; refusing to mix results. "
                "Use a new output directory."
            )
        if stored.get("provenance", {}).get("code") != current["code"]:
            raise PlanMismatchError(
                f"{out_dir} was started with different code "
                f"(commit {stored.get('provenance', {}).get('commit')}); resuming would mix "
                "results from two versions. Use a new output directory."
            )
        return
    plan_path.write_text(
        json.dumps(
            {
                "fingerprint": plan.fingerprint(),
                "provenance": current,
                "plan": plan.model_dump(mode="json"),
            },
            indent=2,
        )
    )


def run_plan(
    plan: RunPlan,
    out_dir: Path,
    *,
    budget_usd: float,
    executor: ChunkExecutor,
    prior_cost_per_simulation: float = PRIOR_COST_PER_SIMULATION,
    max_infrastructure_share: float = 0.2,
    repair_passes: int = 1,
    provenance: Callable[[], dict[str, Any]] = lambda: _provenance(),
) -> RunOutcome:
    """Execute the plan's chunks in order until done, out of budget, or failing.

    A chunk is started only if its estimated cost fits in what is left. The
    estimate uses the measured cost per simulation once any chunk has finished,
    and the prior before that. A chunk that ends with infrastructure failures is
    re-invoked ``repair_passes`` times -- tau2's resume re-runs only the failed
    simulations -- and if the failed share is still above
    ``max_infrastructure_share`` the run stops, because a provider failing at
    that rate fails *after* each conversation has been paid for.
    """
    _claim_directory(plan, out_dir, provenance)
    ledger_path = out_dir / "ledger.json"
    ledger = Ledger.load(ledger_path, plan.fingerprint())

    executed: list[str] = []
    already: list[str] = []
    for spec in plan.schedule():
        if spec.key in ledger.complete:
            already.append(spec.key)
            continue

        per_simulation = ledger.cost_per_simulation() or prior_cost_per_simulation
        estimate = spec.simulations * per_simulation
        if ledger.spend() + estimate > budget_usd:
            return RunOutcome(
                status=RunStatus.BUDGET_EXHAUSTED,
                executed=tuple(executed),
                already_complete=tuple(already),
                spend=ledger.spend(),
                stopped_before=spec.key,
                detail=(
                    f"spent ${ledger.spend():.2f}; next chunk estimated ${estimate:.2f} "
                    f"({spec.simulations} x ${per_simulation:.4f}); budget ${budget_usd:.2f}"
                ),
            )

        report = executor(spec)
        for _ in range(repair_passes):
            if report.infrastructure_errors == 0:
                break
            report = executor(spec)
        executed.append(spec.key)

        if report.infrastructure_share > max_infrastructure_share:
            ledger.incomplete[spec.key] = report
            ledger.save(ledger_path)
            return RunOutcome(
                status=RunStatus.INFRASTRUCTURE_FAILURE,
                executed=tuple(executed),
                already_complete=tuple(already),
                spend=ledger.spend(),
                stopped_before=spec.key,
                detail=(
                    f"{report.infrastructure_errors}/{report.simulations} simulations in "
                    f"{spec.key} failed on infrastructure after {repair_passes} repair pass(es)"
                ),
            )

        ledger.incomplete.pop(spec.key, None)
        ledger.complete[spec.key] = report
        ledger.save(ledger_path)

    return RunOutcome(
        status=RunStatus.COMPLETE,
        executed=tuple(executed),
        already_complete=tuple(already),
        spend=ledger.spend(),
    )


# --- session records ------------------------------------------------------


class SessionRecord(BaseModel):
    """One agent attempt: what it proposed, what it cost, and its audit chain."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    seed: int
    session_id: str
    written_at: str
    proposals: int
    llm_cost: float
    chain_intact: bool
    records: tuple[AuditRecord, ...]


class SessionSink:
    """Appends session records to a JSONL file, one line per agent attempt."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: SessionRecord) -> None:
        line = record.model_dump_json() + "\n"
        with self._lock, self.path.open("a") as fh:
            fh.write(line)
            fh.flush()


def read_sessions(path: Path) -> list[SessionRecord]:
    """Every attempt, in the order written. Use :func:`kept_sessions` for the join."""
    if not path.exists():
        return []
    return [
        SessionRecord.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def kept_sessions(records: Iterable[SessionRecord]) -> dict[tuple[str, int], SessionRecord]:
    """The last attempt per ``(task_id, seed)`` -- the one tau2 kept."""
    kept: dict[tuple[str, int], SessionRecord] = {}
    for record in records:
        kept[(record.task_id, record.seed)] = record
    return kept


def recording_agent_class():
    """The harness agent plus the bookkeeping tau2 has no place for.

    Built on demand so this module imports without tau2 for the planning and
    ledger logic.
    """
    from policygate.adapters.tau2_agent import HarnessAgent

    class RecordingHarnessAgent(HarnessAgent):
        def __init__(self, *args: Any, task_id: str, sink: SessionSink, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.task_id = task_id
            self.sink = sink
            self.seed: int | None = None
            self.proposals = 0
            self.llm_cost = 0.0
            self._last_state = None
            self._written = False

        def set_seed(self, seed: int) -> None:
            self.seed = seed

        def get_init_state(self, message_history=None):
            self._last_state = super().get_init_state(message_history)
            return self._last_state

        def _propose(self, state):
            response = super()._propose(state)
            self.proposals += 1
            self.llm_cost += response.cost or 0.0
            return response

        def stop(self, message=None, state=None) -> None:
            # One agent instance is one attempt; tau2 may call stop from both
            # its finaliser and its cleanup path, and a second line would count
            # the attempt's cost twice.
            if self._written:
                return
            # Never raise here: tau2 wraps stop() and turns an exception into a
            # warning, so raising would lose the record silently. A missing seed
            # is written as MISSING_SEED and surfaces when the join finds no match.
            state = state if state is not None else self._last_state
            records = tuple(state.chain.records) if state is not None else ()
            self.sink.write(
                SessionRecord(
                    task_id=self.task_id,
                    seed=self.seed if self.seed is not None else MISSING_SEED,
                    session_id=self.session_id,
                    written_at=datetime.now(UTC).isoformat(),
                    proposals=self.proposals,
                    llm_cost=self.llm_cost,
                    chain_intact=verify_records(records).intact,
                    records=records,
                )
            )
            self._written = True

    return RecordingHarnessAgent


# --- the tau2 executor ----------------------------------------------------


class Tau2ChunkExecutor:
    """Runs one chunk as its own resumable tau2 batch and measures it from disk."""

    def __init__(self, plan: RunPlan, out_dir: Path, *, max_concurrency: int = 4) -> None:
        import tau2.evaluator.evaluator_nl_assertions as nl_assertions
        from tau2.registry import registry
        from tau2.utils.utils import DATA_DIR

        from policygate.adapters.tool_mapping import (
            RETAIL_ESCALATION,
            RETAIL_IDENTITY_LOOKUPS,
            retail_constraints,
            retail_facts,
            tool_specs,
        )

        self.plan = plan
        self.out_dir = out_dir
        self.max_concurrency = max_concurrency
        # Rebound in the module that uses it: the evaluator binds tau2's default
        # by value at import, and that default is unroutable on AIHubMix.
        nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = plan.judge_llm

        environment = registry.get_env_constructor(plan.domain)()
        db_path = DATA_DIR / "tau2" / "domains" / plan.domain / "db.json"
        constraints = retail_constraints(
            json.loads(db_path.read_text()), tools=environment.get_tools()
        )
        catalogue = tool_specs(
            environment.tools,
            identity_lookups=RETAIL_IDENTITY_LOOKUPS,
            escalation=RETAIL_ESCALATION,
        )
        agent_class = recording_agent_class()
        prefix = f"policygate_{plan.fingerprint()[:10]}"

        self._agent_names: dict[str, str] = {}
        for arm in plan.arms:
            name = f"{prefix}_{arm.name}"
            self._agent_names[arm.name] = name
            # tau2's registry is process-global and refuses re-registration, so
            # the factory looks its sink up at call time: a second executor for
            # the same plan in another directory redirects records instead of
            # silently writing into the first one's.
            _SINKS[name] = SessionSink(self.sessions_path(arm.name))
            if registry.get_agent_factory(name) is not None:
                continue

            def factory(
                tools, domain_policy, llm=None, llm_args=None, task=None, _arm=arm, _name=name,
                **_ignored,
            ):
                return agent_class(
                    tools=tools,
                    domain_policy=domain_policy,
                    llm=llm,
                    catalogue=catalogue,
                    constraints=constraints,
                    extract_facts=retail_facts,
                    config=_arm.config,
                    llm_args=llm_args or {},
                    session_id=str(uuid.uuid4()),
                    max_internal_attempts=plan.max_internal_attempts,
                    task_id=task.id,
                    sink=_SINKS[_name],
                )

            registry.register_agent_factory(factory, name)

    def sessions_path(self, arm: str) -> Path:
        return self.out_dir / arm / "sessions.jsonl"

    def results_path(self, spec: ChunkSpec) -> Path:
        return self.out_dir / spec.arm / f"results_chunk_{spec.index:03d}.json"

    def run_config(self, spec: ChunkSpec):
        from tau2.data_model.simulation import TextRunConfig

        plan = self.plan
        return TextRunConfig(
            domain=plan.domain,
            task_ids=list(spec.task_ids),
            agent=self._agent_names[spec.arm],
            llm_agent=plan.agent_llm,
            llm_args_agent=dict(plan.llm_args_agent),
            user="user_simulator",
            llm_user=plan.user_llm,
            llm_args_user=dict(plan.llm_args_user),
            num_trials=spec.trials,
            seed=plan.seed,
            max_steps=plan.max_steps,
            max_concurrency=self.max_concurrency,
            hallucination_retries=plan.hallucination_retries,
            enforce_communication_protocol=plan.enforce_communication_protocol,
            # Deterministic, so a re-invocation resumes the same checkpoint.
            save_to=f"{plan.name}_{plan.fingerprint()[:10]}/{spec.key}",
            auto_resume=True,
        )

    def __call__(self, spec: ChunkSpec) -> ChunkReport:
        from tau2.runner import run_domain

        results = run_domain(self.run_config(spec))
        path = self.results_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        results.save(path)
        return measure_chunk(spec, results.simulations, read_sessions(self.sessions_path(spec.arm)))


def measure_chunk(
    spec: ChunkSpec, simulations: Sequence[Any], sessions: Sequence[SessionRecord]
) -> ChunkReport:
    """A chunk's report from its simulations and its arm's session records."""
    tasks = set(spec.task_ids)
    ours = [s for s in simulations if s.task_id in tasks]
    infrastructure = sum(
        1
        for s in ours
        if str(getattr(s.termination_reason, "value", s.termination_reason))
        == "infrastructure_error"
    )
    return ChunkReport(
        key=spec.key,
        simulations=len(ours),
        infrastructure_errors=infrastructure,
        agent_llm_cost=sum(r.llm_cost for r in sessions if r.task_id in tasks),
        user_cost=sum(s.user_cost or 0.0 for s in ours),
    )


def load_arm(out_dir: Path, arm: str) -> tuple[list[Any], dict[tuple[str, int], SessionRecord]]:
    """An arm's simulations across its chunks, and its kept session records."""
    from tau2.data_model.simulation import Results

    simulations: list[Any] = []
    for path in sorted((out_dir / arm).glob("results_chunk_*.json")):
        simulations.extend(Results.load(path).simulations)
    return simulations, kept_sessions(read_sessions(out_dir / arm / "sessions.jsonl"))
