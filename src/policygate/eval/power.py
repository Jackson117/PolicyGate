"""How large an effect the W4 design can detect, before paying to run it.

Simulation-based power for the paired, task-clustered comparison in
:mod:`policygate.eval.passk`, with task difficulty taken from real runs.

**Task difficulty is a fitted Beta, not W1's per-task rates.** With three trials
a task's observed rate is one of 0, 1/3, 2/3 or 1, and 13 of W1's 30 tasks sit
at an endpoint. Plugging those in as true probabilities would make them
deterministic, remove their headroom, and understate power for a reason that is
an artefact of three trials. A beta-binomial fitted by moments keeps the
between-task spread W1 showed while letting every task be uncertain;
:meth:`BetaBinomialFit.stability_check` compares the always / never / unstable
split it implies with the one observed, so the fit is checkable rather than
assumed.

**The effect is a constant shift on the logit scale.** That is an assumption,
chosen because it makes headroom behave realistically: a task solved 98% of the
time can barely improve, one at 50% can move a lot. Effects are reported on the
scale people read -- the expected change in pass^k -- not in logits.

**Inside the loop the test is the normal approximation to the sign-flip test.**
The exact test in passk is too slow to run thousands of times; the approximation
(sum of per-task differences over the root of their summed squares, with a
continuity correction for the lattice those differences live on) is the standard
large-sample form of the same randomisation test. Unit tests hold it to the
Monte Carlo version on the decision that matters -- reject or not -- and hold
its size at alpha.

**Multiplicity is handled conservatively.** The seven single-mechanism arms are
compared with one shared baseline under Holm. Holm's first step is Bonferroni,
so computing their power at alpha / 7 gives the power for the *least* favoured
comparison -- a floor, not an estimate of the typical one. The pre-specified
headline, full against baseline, is a single comparison at alpha.

Standard library and pydantic only.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from policygate.eval.passk import pass_hat_k

_EPS = 1e-9


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def _expit(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# --- the difficulty model -------------------------------------------------


class StabilityCheck(BaseModel):
    """Observed always / never / unstable task counts against the fit's expectation."""

    model_config = ConfigDict(frozen=True)

    observed_always: int
    observed_never: int
    observed_unstable: int
    expected_always: float
    expected_never: float
    expected_unstable: float


class BetaBinomialFit(BaseModel):
    """Per-task success probability ~ Beta(alpha, beta), trials ~ Binomial."""

    model_config = ConfigDict(frozen=True)

    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)
    n_tasks: int
    trials: int
    intraclass_correlation: float
    """How much of the outcome variance is between tasks rather than within one."""

    observed_counts: tuple[int, ...]

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def probability_of_count(self, successes: int, trials: int) -> float:
        """Beta-binomial pmf."""
        return math.exp(
            math.log(math.comb(trials, successes))
            + _log_beta(self.alpha + successes, self.beta + trials - successes)
            - _log_beta(self.alpha, self.beta)
        )

    def stability_check(self) -> StabilityCheck:
        n = self.trials
        p_always = self.probability_of_count(n, n)
        p_never = self.probability_of_count(0, n)
        counts = self.observed_counts
        return StabilityCheck(
            observed_always=sum(1 for c in counts if c == n),
            observed_never=sum(1 for c in counts if c == 0),
            observed_unstable=sum(1 for c in counts if 0 < c < n),
            expected_always=self.n_tasks * p_always,
            expected_never=self.n_tasks * p_never,
            expected_unstable=self.n_tasks * (1 - p_always - p_never),
        )


def fit_beta_binomial(counts: Sequence[int], trials: int) -> BetaBinomialFit:
    """Method-of-moments fit from per-task success counts out of ``trials``.

    Requires a balanced table: every task with the same number of trials, which
    is what W1 has and what the estimator assumes. The intraclass correlation is
    clamped away from 0 and 1 -- at 0 every task is identical, at 1 every task is
    deterministic -- because both make the Beta degenerate.
    """
    if trials < 2:
        raise ValueError("between-task spread cannot be separated from noise with one trial")
    if len(counts) < 2:
        raise ValueError("need at least two tasks to fit a spread")
    if any(not 0 <= c <= trials for c in counts):
        raise ValueError(f"counts must lie in [0, {trials}]")

    n = trials
    m = len(counts)
    mean_count = math.fsum(counts) / m
    p = mean_count / n
    if p <= 0 or p >= 1:
        raise ValueError("every task failed, or every task passed: there is no spread to fit")
    variance = math.fsum((c - mean_count) ** 2 for c in counts) / (m - 1)
    rho = (variance / (n * p * (1 - p)) - 1) / (n - 1)
    rho = min(max(rho, 1e-4), 1 - 1e-4)
    total = 1 / rho - 1
    return BetaBinomialFit(
        alpha=p * total,
        beta=(1 - p) * total,
        n_tasks=m,
        trials=n,
        intraclass_correlation=rho,
        observed_counts=tuple(counts),
    )


# --- the design and the simulation ----------------------------------------


class Design(BaseModel):
    """One candidate W4 configuration for a single two-arm comparison."""

    model_config = ConfigDict(frozen=True)

    n_tasks: int = Field(gt=1)
    n_trials: int = Field(gt=0)
    k: int = Field(default=1, gt=0)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    comparisons: int = Field(default=1, gt=0)
    """Family size. Power is computed at alpha / comparisons (Holm's first step)."""

    @property
    def threshold(self) -> float:
        return self.alpha / self.comparisons


def sign_flip_normal_p(differences: Sequence[float], lattice_unit: float = 0.0) -> float:
    """Two-sided p-value of the sign-flip test by its normal approximation.

    Under the null every per-task difference is equally likely to carry either
    sign, so their sum has mean 0 and variance equal to the sum of squares.

    ``lattice_unit`` is the continuity correction. Per-task pass^k estimates are
    multiples of 1 / C(n, k), so the signed sum moves in steps of twice that and
    the exact test counts ties at the observed value as extreme. Without the
    correction the approximation is liberal near the boundary -- measured at W4's
    scale, it rejected where the exact test did not in 3% of tables. With one
    trial per task the corrected form is the classical continuity-corrected
    McNemar statistic, (|b - c| - 1) / sqrt(b + c).
    """
    total_square = math.fsum(d * d for d in differences)
    if total_square == 0:
        return 1.0
    excess = max(abs(math.fsum(differences)) - lattice_unit, 0.0)
    return math.erfc(excess / math.sqrt(total_square) / math.sqrt(2))


def _binomial_quantile(u: float, n: int, p: float) -> int:
    """Inverse binomial CDF. One uniform per task-arm keeps draws comparable across effects."""
    cumulative = 0.0
    for c in range(n + 1):
        cumulative += math.comb(n, c) * p**c * (1 - p) ** (n - c)
        if u <= cumulative:
            return c
    return n


class PowerPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    design: Design
    logit_shift: float
    expected_difference: float
    """Expected change in pass^k across the task population, treatment minus control."""

    power: float
    replicates: int

    @property
    def standard_error(self) -> float:
        return math.sqrt(self.power * (1 - self.power) / self.replicates)


def _replicate_draws(fit: BetaBinomialFit, design: Design, replicates: int, seed: int):
    """Task probabilities and uniforms, fixed per seed.

    Common random numbers: every effect size is evaluated on the same tasks and
    the same uniforms, so the power curve moves only because the effect moved,
    and a bisection over it is not chasing simulation noise.
    """
    rng = random.Random(seed)
    for _ in range(replicates):
        yield [
            (rng.betavariate(fit.alpha, fit.beta), rng.random(), rng.random())
            for _ in range(design.n_tasks)
        ]


def expected_difference(
    fit: BetaBinomialFit, logit_shift: float, k: int = 1, draws: int = 200_000, seed: int = 1
) -> float:
    """E[p_T^k - p_C^k] over the fitted population, without trial noise."""
    rng = random.Random(seed)
    total = 0.0
    for _ in range(draws):
        p = rng.betavariate(fit.alpha, fit.beta)
        total += _expit(_logit(p) + logit_shift) ** k - p**k
    return total / draws


def _rejection_rate(
    fit: BetaBinomialFit, design: Design, logit_shift: float, replicates: int, seed: int
) -> float:
    n = design.n_trials
    if design.k > n:
        raise ValueError(f"pass^{design.k} needs at least {design.k} trials, design has {n}")
    lattice_unit = 1 / math.comb(n, design.k)
    rejected = 0
    for tasks in _replicate_draws(fit, design, replicates, seed):
        differences = []
        for p_control, u_control, u_treatment in tasks:
            p_treatment = _expit(_logit(p_control) + logit_shift)
            c_control = _binomial_quantile(u_control, n, p_control)
            c_treatment = _binomial_quantile(u_treatment, n, p_treatment)
            differences.append(
                pass_hat_k(n, c_treatment, design.k) - pass_hat_k(n, c_control, design.k)
            )
        if sign_flip_normal_p(differences, lattice_unit) < design.threshold:
            rejected += 1
    return rejected / replicates


def simulate_power(
    fit: BetaBinomialFit,
    design: Design,
    logit_shift: float,
    *,
    replicates: int = 1000,
    seed: int = 0,
) -> PowerPoint:
    """Share of simulated experiments in which the paired test rejects."""
    return PowerPoint(
        design=design,
        logit_shift=logit_shift,
        expected_difference=expected_difference(fit, logit_shift, design.k),
        power=_rejection_rate(fit, design, logit_shift, replicates, seed),
        replicates=replicates,
    )


class MinimumDetectableEffect(BaseModel):
    model_config = ConfigDict(frozen=True)

    design: Design
    target_power: float
    achievable: bool
    """False if even the largest shift searched does not reach the target."""

    logit_shift: float
    expected_difference: float
    power: float
    replicates: int


def minimum_detectable_effect(
    fit: BetaBinomialFit,
    design: Design,
    *,
    target_power: float = 0.8,
    max_logit_shift: float = 4.0,
    iterations: int = 10,
    replicates: int = 1000,
    seed: int = 0,
) -> MinimumDetectableEffect:
    """Smallest positive logit shift reaching ``target_power``, by bisection.

    Searched for an improvement only. Under a logit shift a harm of the same
    size is not exactly symmetric in pass^k, because the population sits above
    or below one half; the report states this rather than mirroring the number.
    """

    def power_at(shift: float) -> float:
        return _rejection_rate(fit, design, shift, replicates, seed)

    ceiling = power_at(max_logit_shift)
    if ceiling < target_power:
        return MinimumDetectableEffect(
            design=design,
            target_power=target_power,
            achievable=False,
            logit_shift=max_logit_shift,
            expected_difference=expected_difference(fit, max_logit_shift, design.k),
            power=ceiling,
            replicates=replicates,
        )

    lo, hi, hi_power = 0.0, max_logit_shift, ceiling
    for _ in range(iterations):
        mid = (lo + hi) / 2
        power = power_at(mid)
        if power >= target_power:
            hi, hi_power = mid, power
        else:
            lo = mid
    return MinimumDetectableEffect(
        design=design,
        target_power=target_power,
        achievable=True,
        logit_shift=hi,
        expected_difference=expected_difference(fit, hi, design.k),
        power=hi_power,
        replicates=replicates,
    )
