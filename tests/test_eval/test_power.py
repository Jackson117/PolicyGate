"""Tests for eval/power.py.

The simulation is only worth trusting if three things hold, and those are what
is tested: the difficulty model recovers what generated the data, the fast test
inside the loop agrees with the exact one it stands in for and holds its size,
and power moves the right way with effect, trials and multiplicity.

No LLM, no benchmark run, no network.
"""

import math
import random

import pytest

from policygate.eval.passk import Observation, ObservationTable, compare_paired
from policygate.eval.power import (
    Design,
    fit_beta_binomial,
    minimum_detectable_effect,
    sign_flip_normal_p,
    simulate_power,
)

REPS = 400


def beta_binomial_counts(alpha, beta, trials, tasks, seed):
    rng = random.Random(seed)
    counts = []
    for _ in range(tasks):
        p = rng.betavariate(alpha, beta)
        counts.append(sum(rng.random() < p for _ in range(trials)))
    return counts


@pytest.fixture(scope="module")
def w1_like():
    """A fit shaped like W1's reward component: mean near 0.5, strong task spread."""
    return fit_beta_binomial([3] * 7 + [0] * 6 + [1] * 8 + [2] * 9, 3)


# --- the difficulty model -------------------------------------------------


def test_the_fit_recovers_the_generating_distribution():
    counts = beta_binomial_counts(2.0, 3.0, trials=5, tasks=6000, seed=3)
    fit = fit_beta_binomial(counts, 5)
    assert fit.mean == pytest.approx(0.4, abs=0.02)
    assert fit.intraclass_correlation == pytest.approx(1 / 6, abs=0.03)


def test_the_stability_check_accounts_for_every_task(w1_like):
    check = w1_like.stability_check()
    assert check.observed_always + check.observed_never + check.observed_unstable == 30
    expected = check.expected_always + check.expected_never + check.expected_unstable
    assert expected == pytest.approx(30)


def test_the_fit_reproduces_its_own_stability_split_on_simulated_data():
    counts = beta_binomial_counts(1.4, 1.3, trials=3, tasks=5000, seed=11)
    check = fit_beta_binomial(counts, 3).stability_check()
    assert check.expected_always == pytest.approx(check.observed_always, rel=0.06)
    assert check.expected_never == pytest.approx(check.observed_never, rel=0.06)


@pytest.mark.parametrize(
    "counts,trials,match",
    [
        ([1, 0, 1], 1, "one trial"),
        ([2], 3, "two tasks"),
        ([3, 3, 3], 3, "no spread"),
        ([0, 0, 0], 3, "no spread"),
        ([4, 1], 3, r"\[0, 3\]"),
    ],
)
def test_the_fit_refuses_what_it_cannot_estimate(counts, trials, match):
    with pytest.raises(ValueError, match=match):
        fit_beta_binomial(counts, trials)


# --- the test inside the loop ---------------------------------------------


def test_the_approximation_makes_the_same_decisions_as_the_exact_test():
    """What the power loop uses is reject-or-not, so that is what is compared.

    Tables straddle the rejection boundary: effects from none to clear, one to
    five trials, 114 tasks. Mid-range p-values are allowed to differ -- the
    per-task differences live on a lattice and no smooth approximation matches
    the exact tail there -- but decisions at alpha must almost always agree.
    """
    rng = random.Random(0)
    disagreements = 0
    tables = 60
    for rep in range(tables):
        shift = rng.choice([0.0, 0.06, 0.12])
        trials = rng.choice([1, 3, 5])
        treatment, control, differences = [], [], []
        for task in range(114):
            p = rng.betavariate(1.4, 1.3)
            t = [rng.random() < min(p + shift, 1) for _ in range(trials)]
            c = [rng.random() < p for _ in range(trials)]
            treatment += [Observation(task_id=str(task), trial=i, value=v) for i, v in enumerate(t)]
            control += [Observation(task_id=str(task), trial=i, value=v) for i, v in enumerate(c)]
            differences.append((sum(t) - sum(c)) / trials)
        exact = compare_paired(
            ObservationTable(observations=tuple(treatment)),
            ObservationTable(observations=tuple(control)),
            resamples=4000,
            seed=rep,
        ).p_value
        approximate = sign_flip_normal_p(differences, lattice_unit=1 / trials)
        disagreements += (exact < 0.05) != (approximate < 0.05)
    assert disagreements <= 3


def test_with_one_trial_the_correction_is_continuity_corrected_mcnemar():
    differences = [1.0] * 9 + [-1.0] * 2 + [0.0] * 20
    b, c = 9, 2
    z = (abs(b - c) - 1) / math.sqrt(b + c)
    assert sign_flip_normal_p(differences, lattice_unit=1.0) == pytest.approx(
        math.erfc(z / math.sqrt(2))
    )


def test_no_difference_is_p_one():
    assert sign_flip_normal_p([0.0, 0.0, 0.0]) == 1.0


def test_the_test_holds_its_size_under_no_effect(w1_like):
    """Continuity-corrected, so at or a little under alpha -- never materially over."""
    result = simulate_power(w1_like, Design(n_tasks=114, n_trials=3), 0.0, replicates=3000)
    assert 0.025 <= result.power <= 0.06


# --- how power moves ------------------------------------------------------


def test_power_rises_with_the_effect(w1_like):
    design = Design(n_tasks=60, n_trials=3)
    powers = [simulate_power(w1_like, design, s, replicates=REPS).power for s in (0.0, 0.4, 0.8)]
    assert powers == sorted(powers)
    assert powers[-1] > powers[0] + 0.3


def test_power_rises_with_trials(w1_like):
    few = simulate_power(w1_like, Design(n_tasks=60, n_trials=1), 0.5, replicates=REPS)
    many = simulate_power(w1_like, Design(n_tasks=60, n_trials=5), 0.5, replicates=REPS)
    assert many.power > few.power


def test_a_larger_family_costs_power(w1_like):
    single = simulate_power(w1_like, Design(n_tasks=60, n_trials=3), 0.5, replicates=REPS)
    family = simulate_power(
        w1_like, Design(n_tasks=60, n_trials=3, comparisons=7), 0.5, replicates=REPS
    )
    assert family.design.threshold == pytest.approx(0.05 / 7)
    assert family.power < single.power


def test_a_positive_shift_is_reported_as_a_positive_change_in_pass_rate(w1_like):
    point = simulate_power(w1_like, Design(n_tasks=30, n_trials=3), 0.5, replicates=50)
    assert 0 < point.expected_difference < 0.5


def test_pass_k_beyond_the_trials_is_refused(w1_like):
    with pytest.raises(ValueError, match="pass\\^3"):
        simulate_power(w1_like, Design(n_tasks=30, n_trials=2, k=3), 0.5, replicates=10)


# --- minimum detectable effect --------------------------------------------


def test_more_trials_detect_a_smaller_effect(w1_like):
    few = minimum_detectable_effect(
        w1_like, Design(n_tasks=114, n_trials=1), replicates=REPS, iterations=7
    )
    many = minimum_detectable_effect(
        w1_like, Design(n_tasks=114, n_trials=5), replicates=REPS, iterations=7
    )
    assert few.achievable and many.achievable
    assert many.expected_difference < few.expected_difference
    assert many.power >= 0.8


def test_an_underpowered_design_says_so_instead_of_returning_a_number(w1_like):
    result = minimum_detectable_effect(
        w1_like,
        Design(n_tasks=5, n_trials=1, comparisons=7),
        max_logit_shift=0.5,
        replicates=REPS,
        iterations=3,
    )
    assert result.achievable is False
