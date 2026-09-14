"""pass^k with intervals, and the paired comparison an ablation actually needs.

Everything here takes a table of per-simulation binary observations and returns
estimates that carry their own uncertainty. Nothing returns a bare point.

**The unit of resampling is the task, not the simulation.** Trials of one task
share its difficulty, so treating 114 tasks x 5 trials as 570 independent draws
would shrink every interval by roughly sqrt(5) for no reason. Each task is
reduced to its own unbiased estimate first; the bootstrap then resamples tasks.
That single-stage cluster bootstrap still carries within-task noise, because each
per-task estimate is itself noisy -- the variance of their mean includes it.

**Arms are compared on the same tasks, jointly.** Two independent intervals that
overlap say little: task difficulty is shared between arms and cancels in a
paired design. So the difference is estimated per task and resampled as pairs,
and the test is a sign-flip randomisation test on per-task differences. For one
binary trial per task that test is exactly McNemar's; with several trials it
uses the whole per-task estimate rather than pretending trial i of one arm is
paired with trial i of the other -- W1 showed the user simulator diverges at
temperature 0, so no such pairing exists.

**Exclusions are reported, never absorbed.** tau2's ``run_with_retry`` catches
*any* exception, reruns the simulation, and after three failures records it as
``INFRASTRUCTURE_ERROR`` -- which tau2's own metrics then drop. An exception
thrown by the harness itself lands in the same bucket. If it only fires on hard
tasks, those tasks vanish from the harness arm alone and its score rises. So the
loader records what was excluded and why, the paired comparison reports
exclusions per arm, and :class:`ExclusionPolicy` lets the same data be scored
with exclusions counted as failures, as a bound.

Only the standard library and pydantic, so this runs and is tested without the
benchmark installed.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

SUCCESS_THRESHOLD = 1.0 - 1e-9
"""tau2 counts a simulation as passed when its reward is 1."""

INFRASTRUCTURE_ERROR = "infrastructure_error"
DEFAULT_LEVEL = 0.95
DEFAULT_RESAMPLES = 10_000


def pass_hat_k(num_trials: int, successes: int, k: int) -> float:
    """Unbiased estimate of p^k from ``successes`` out of ``num_trials``.

    C(c, k) / C(n, k): the probability that k trials drawn without replacement
    from the n observed were all successes. Unbiased for p^k for any n >= k,
    which is what lets tasks with different trial counts be averaged.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if num_trials < k:
        raise ValueError(f"need at least k={k} trials, got {num_trials}")
    if not 0 <= successes <= num_trials:
        raise ValueError(f"successes={successes} outside [0, {num_trials}]")
    return math.comb(successes, k) / math.comb(num_trials, k)


# --- observations ---------------------------------------------------------


class Observation(BaseModel):
    """One simulation reduced to one binary value.

    Usually "passed", but deliberately not named so: a policy violation, a
    forced escalation or a DB-component match is the same shape, and gets the
    same clustering and the same intervals.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    trial: int = Field(ge=0)
    value: bool


class ObservationTable(BaseModel):
    """Observations for one arm, plus an account of what did not make it in."""

    model_config = ConfigDict(frozen=True)

    observations: tuple[Observation, ...]
    excluded: Mapping[str, int] = Field(default_factory=dict)
    """Reason -> count. Infrastructure errors keep their exception type."""

    excluded_tasks: Mapping[str, int] = Field(default_factory=dict)
    """Task id -> simulations excluded for it. Needed to see *where* losses fell."""

    @model_validator(mode="after")
    def _one_observation_per_trial(self) -> ObservationTable:
        seen = Counter((o.task_id, o.trial) for o in self.observations)
        duplicated = sorted(key for key, n in seen.items() if n > 1)
        if duplicated:
            # Two results for one (task, trial) usually means two runs were
            # concatenated. Keeping both would count one trial twice.
            raise ValueError(f"duplicate (task_id, trial): {duplicated[:5]}")
        return self

    def by_task(self) -> dict[str, tuple[bool, ...]]:
        """Values per task, in a fixed order so resampling is reproducible."""
        grouped: dict[str, list[tuple[int, bool]]] = defaultdict(list)
        for o in self.observations:
            grouped[o.task_id].append((o.trial, o.value))
        return {
            task: tuple(v for _, v in sorted(rows)) for task, rows in sorted(grouped.items())
        }

    @property
    def n_excluded(self) -> int:
        return sum(self.excluded.values())


class ExclusionPolicy(StrEnum):
    EXCLUDE = "exclude"
    """Drop simulations that produced no outcome. The usual reading."""

    COUNT_AS_FAILURE = "count_as_failure"
    """Score them as a 0. A worst case for whichever arm lost them."""


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def observations_from_simulations(
    simulations: Iterable[Any],
    *,
    component: str | None = None,
    policy: ExclusionPolicy = ExclusionPolicy.EXCLUDE,
    threshold: float = SUCCESS_THRESHOLD,
) -> ObservationTable:
    """Read tau2 ``SimulationRun`` objects into an observation table.

    Duck-typed on the fields it reads, so this module does not import tau2.

    ``component`` scores one entry of ``reward_breakdown`` (``"DB"``,
    ``"NL_ASSERTION"``) instead of the combined reward. Retail combines both on
    112 of 114 tasks; the harness acts on world state, and the judge adds noise
    it cannot affect, so the DB component is the one attribution rests on.

    Only ``INFRASTRUCTURE_ERROR`` and a missing reward are treated as "no
    outcome". ``AGENT_ERROR``, ``MAX_STEPS`` and the rest carry a reward and are
    the agent's failures to own.
    """
    observations: list[Observation] = []
    excluded: Counter[str] = Counter()
    excluded_tasks: Counter[str] = Counter()

    for sim in simulations:
        if sim.trial is None:
            raise ValueError(f"simulation {sim.id} for task {sim.task_id} has no trial number")

        reason = _enum_value(sim.termination_reason)
        cause: str | None = None
        value: bool | None = None

        if reason == INFRASTRUCTURE_ERROR:
            error_type = (getattr(sim, "info", None) or {}).get("error_type", "unknown")
            cause = f"{INFRASTRUCTURE_ERROR}:{error_type}"
        elif sim.reward_info is None:
            cause = f"no_reward:{reason}"
        elif component is None:
            value = sim.reward_info.reward >= threshold
        else:
            breakdown = {
                _enum_value(key): val
                for key, val in (sim.reward_info.reward_breakdown or {}).items()
            }
            if component in breakdown:
                value = breakdown[component] >= threshold
            else:
                # Structural, not a failure: two retail tasks have no NL
                # assertion at all.
                cause = f"component_absent:{component}"

        if cause is not None:
            excluded[cause] += 1
            excluded_tasks[sim.task_id] += 1
            if policy is ExclusionPolicy.COUNT_AS_FAILURE and not cause.startswith(
                "component_absent"
            ):
                value = False
            else:
                continue

        observations.append(Observation(task_id=sim.task_id, trial=sim.trial, value=value))

    return ObservationTable(
        observations=tuple(observations),
        excluded=dict(excluded),
        excluded_tasks=dict(excluded_tasks),
    )


# --- resampling -----------------------------------------------------------


def _index_draws(n: int, resamples: int, seed: int) -> Iterator[list[int]]:
    """Task indices for each bootstrap replicate.

    Every statistic computed from the same seed and the same number of tasks
    sees the same replicates, so pass^1 and pass^3 intervals from one table are
    consistent with each other rather than independently jittered.
    """
    rng = random.Random(seed)
    population = range(n)
    for _ in range(resamples):
        yield rng.choices(population, k=n)


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default, type 7)."""
    h = (len(ordered) - 1) * q
    lo = math.floor(h)
    hi = math.ceil(h)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def _percentile_interval(
    values: Sequence[float], level: float, resamples: int, seed: int
) -> tuple[float, float]:
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if resamples < 1:
        raise ValueError(f"resamples must be positive, got {resamples}")
    n = len(values)
    means = sorted(
        math.fsum(values[i] for i in draw) / n for draw in _index_draws(n, resamples, seed)
    )
    alpha = 1 - level
    return _quantile(means, alpha / 2), _quantile(means, 1 - alpha / 2)


class Estimate(BaseModel):
    """A point estimate and its percentile cluster-bootstrap interval.

    Percentile intervals under-cover when there are few tasks and the statistic
    sits near 0 or 1. ``n_tasks`` is on the record so that can be judged; at the
    30-task W1 scale it is a real concern, at 114 much less so.
    """

    model_config = ConfigDict(frozen=True)

    statistic: str
    value: float
    lo: float
    hi: float
    level: float
    n_tasks: int
    resamples: int
    skipped_tasks: tuple[str, ...] = ()
    """Tasks with too few trials for this statistic. Not silently averaged over."""

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2


def per_task_pass_hat_k(
    table: ObservationTable, k: int
) -> tuple[dict[str, float], tuple[str, ...]]:
    """Each task's pass^k estimate, and the tasks with fewer than k trials."""
    estimates: dict[str, float] = {}
    skipped: list[str] = []
    for task, values in table.by_task().items():
        if len(values) < k:
            skipped.append(task)
        else:
            estimates[task] = pass_hat_k(len(values), sum(values), k)
    return estimates, tuple(skipped)


def pass_k(
    table: ObservationTable,
    k: int = 1,
    *,
    level: float = DEFAULT_LEVEL,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Estimate:
    """pass^k averaged over tasks, with a task-clustered interval.

    Tasks are weighted equally. For pass^1 this is the mean of per-task success
    rates, which equals the pooled rate only when every task has the same number
    of trials -- after infrastructure losses it does not, and pooling would let a
    task that happened to keep all its trials count for more.
    """
    estimates, skipped = per_task_pass_hat_k(table, k)
    if not estimates:
        raise ValueError(f"no task has at least k={k} trials")
    values = list(estimates.values())
    lo, hi = _percentile_interval(values, level, resamples, seed)
    return Estimate(
        statistic=f"pass^{k}",
        value=math.fsum(values) / len(values),
        lo=lo,
        hi=hi,
        level=level,
        n_tasks=len(values),
        resamples=resamples,
        skipped_tasks=skipped,
    )


def rate(
    table: ObservationTable,
    *,
    name: str = "rate",
    level: float = DEFAULT_LEVEL,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Estimate:
    """Mean per-task rate of any binary observation, with the same clustering.

    Numerically pass^1; named separately because a policy-violation rate reported
    under the label "pass^1" would be read the wrong way round.
    """
    estimate = pass_k(table, 1, level=level, resamples=resamples, seed=seed)
    return estimate.model_copy(update={"statistic": name})


class Stability(BaseModel):
    """How tasks behaved across repeated trials.

    ``never`` is the headroom a treatment can act on; ``always`` can only be
    broken. A comparison whose headroom is a handful of tasks cannot detect an
    effect however good the treatment is.
    """

    model_config = ConfigDict(frozen=True)

    always: int
    never: int
    unstable: int
    single_trial: int
    """Tasks observed once. Their stability is unknown, not stable."""

    @property
    def flip_rate(self) -> float | None:
        classified = self.always + self.never + self.unstable
        return self.unstable / classified if classified else None


def stability(table: ObservationTable) -> Stability:
    counts: Counter[str] = Counter()
    for values in table.by_task().values():
        if len(values) < 2:
            counts["single_trial"] += 1
        elif all(values):
            counts["always"] += 1
        elif not any(values):
            counts["never"] += 1
        else:
            counts["unstable"] += 1
    return Stability(
        always=counts["always"],
        never=counts["never"],
        unstable=counts["unstable"],
        single_trial=counts["single_trial"],
    )


# --- paired comparison ----------------------------------------------------


def mcnemar_exact(treatment_only: int, control_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant-pair counts.

    Under the null each discordant task is equally likely to favour either arm,
    so the smaller count is Binomial(b + c, 1/2). Computed in integers.
    """
    if treatment_only < 0 or control_only < 0:
        raise ValueError("discordant counts must be non-negative")
    n = treatment_only + control_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(treatment_only, control_only) + 1))
    return min(1.0, 2 * tail / 2**n)


class PairedTest(StrEnum):
    MCNEMAR_EXACT = "mcnemar_exact"
    """Every non-zero difference is +/-1: one binary trial per task."""

    SIGN_FLIP_EXACT = "sign_flip_exact"
    """Non-zero differences share one magnitude, so the sign count is binomial."""

    SIGN_FLIP_MONTE_CARLO = "sign_flip_monte_carlo"
    """Mixed magnitudes. Resolution is limited by the number of draws."""


class PairedComparison(BaseModel):
    """Treatment against control on the tasks both arms can be scored on."""

    model_config = ConfigDict(frozen=True)

    statistic: str
    treatment: float
    control: float
    difference: Estimate
    """Treatment minus control, per task, averaged; tasks resampled as pairs."""

    treatment_better: int
    control_better: int
    ties: int
    p_value: float
    test: PairedTest

    balanced: bool
    """Both arms have the same trial count on every compared task.

    The sign-flip null needs each per-task difference to be symmetric about
    zero, which holds exactly only when both estimates come from the same
    number of trials. Unequal losses to exclusion weaken it slightly.
    """

    tasks_only_in_treatment: tuple[str, ...] = ()
    tasks_only_in_control: tuple[str, ...] = ()
    tasks_below_k: tuple[str, ...] = ()
    treatment_excluded: int = 0
    control_excluded: int = 0

    @property
    def differential_exclusion(self) -> bool:
        """The arms lost different numbers of simulations. Read the result with care."""
        return self.treatment_excluded != self.control_excluded


def _sign_flip_p(differences: Sequence[float], draws: int, seed: int) -> tuple[float, PairedTest]:
    nonzero = [d for d in differences if not math.isclose(d, 0.0, abs_tol=1e-12)]
    if not nonzero:
        return 1.0, PairedTest.SIGN_FLIP_EXACT

    magnitude = abs(nonzero[0])
    if all(math.isclose(abs(d), magnitude, rel_tol=1e-9) for d in nonzero):
        positive = sum(1 for d in nonzero if d > 0)
        p = mcnemar_exact(positive, len(nonzero) - positive)
        if math.isclose(magnitude, 1.0):
            return p, PairedTest.MCNEMAR_EXACT
        return p, PairedTest.SIGN_FLIP_EXACT

    observed = abs(math.fsum(nonzero))
    rng = random.Random(seed)
    n = len(nonzero)
    at_least_as_extreme = 0
    for _ in range(draws):
        bits = rng.getrandbits(n)
        total = math.fsum(d if (bits >> i) & 1 else -d for i, d in enumerate(nonzero))
        if abs(total) >= observed - 1e-12:
            at_least_as_extreme += 1
    # The +1 counts the observed assignment itself, so p is never reported as 0.
    return (at_least_as_extreme + 1) / (draws + 1), PairedTest.SIGN_FLIP_MONTE_CARLO


def compare_paired(
    treatment: ObservationTable,
    control: ObservationTable,
    *,
    k: int = 1,
    level: float = DEFAULT_LEVEL,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> PairedComparison:
    """Paired pass^k comparison over the tasks both arms can score.

    A task is compared only if both arms have at least k trials for it. Tasks
    dropped for any reason are named in the result, per arm, because an
    exclusion that falls on one arm is exactly how a broken treatment passes for
    a good one.
    """
    treated, treated_short = per_task_pass_hat_k(treatment, k)
    controlled, control_short = per_task_pass_hat_k(control, k)
    t_tasks, c_tasks = treatment.by_task(), control.by_task()

    common = sorted(set(treated) & set(controlled))
    if not common:
        raise ValueError(f"no task has at least k={k} trials in both arms")

    differences = [treated[t] - controlled[t] for t in common]
    lo, hi = _percentile_interval(differences, level, resamples, seed)
    p_value, test = _sign_flip_p(differences, resamples, seed)

    in_both = set(t_tasks) & set(c_tasks)
    return PairedComparison(
        statistic=f"pass^{k}",
        treatment=math.fsum(treated[t] for t in common) / len(common),
        control=math.fsum(controlled[t] for t in common) / len(common),
        difference=Estimate(
            statistic=f"pass^{k} difference",
            value=math.fsum(differences) / len(differences),
            lo=lo,
            hi=hi,
            level=level,
            n_tasks=len(common),
            resamples=resamples,
        ),
        treatment_better=sum(1 for d in differences if d > 1e-12),
        control_better=sum(1 for d in differences if d < -1e-12),
        ties=sum(1 for d in differences if abs(d) <= 1e-12),
        p_value=p_value,
        test=test,
        balanced=all(len(t_tasks[t]) == len(c_tasks[t]) for t in common),
        tasks_only_in_treatment=tuple(sorted(set(t_tasks) - set(c_tasks))),
        tasks_only_in_control=tuple(sorted(set(c_tasks) - set(t_tasks))),
        tasks_below_k=tuple(sorted((set(treated_short) | set(control_short)) & in_both)),
        treatment_excluded=treatment.n_excluded,
        control_excluded=control.n_excluded,
    )
