"""Tests for eval/passk.py.

Most of these check a statistical property rather than an output value: the
estimator is unbiased, the bootstrap resamples tasks rather than trials, the
paired design is tighter than two independent intervals, and an exclusion that
falls on one arm is visible in the result. A test that only pinned numbers
would pass for a wrong method as long as it was consistently wrong.

No LLM, no benchmark run, no network. The loader is exercised against real tau2
result types, since matching their shape is its whole job.
"""

from fractions import Fraction
from math import comb

import pytest
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.data_model.tasks import RewardType

from policygate.eval.passk import (
    ExclusionPolicy,
    Observation,
    ObservationTable,
    PairedTest,
    compare_paired,
    mcnemar_exact,
    observations_from_simulations,
    pass_hat_k,
    pass_k,
    rate,
    stability,
)

FAST = 2_000  # resamples: enough for stable assertions, quick enough for CI


def table(rows: dict[str, list[bool]]) -> ObservationTable:
    return ObservationTable(
        observations=tuple(
            Observation(task_id=task, trial=i, value=v)
            for task, values in rows.items()
            for i, v in enumerate(values)
        )
    )


# --- the estimator --------------------------------------------------------


def test_pass_hat_k_matches_the_combinatorial_definition():
    assert pass_hat_k(5, 3, 2) == comb(3, 2) / comb(5, 2)
    assert pass_hat_k(5, 1, 2) == 0.0
    assert pass_hat_k(5, 5, 5) == 1.0


@pytest.mark.parametrize("n,k", [(3, 1), (3, 3), (5, 2), (8, 4)])
@pytest.mark.parametrize("p", [Fraction(1, 5), Fraction(1, 2), Fraction(9, 10)])
def test_pass_hat_k_is_exactly_unbiased_for_p_to_the_k(n, k, p):
    """E[C(c,k)/C(n,k)] over c ~ Binomial(n, p) equals p^k, in exact arithmetic.

    This is the property that makes averaging tasks with different trial
    counts legitimate.
    """
    expectation = sum(
        comb(n, c) * p**c * (1 - p) ** (n - c) * Fraction(comb(c, k), comb(n, k))
        for c in range(n + 1)
    )
    assert expectation == p**k


def test_pass_hat_k_refuses_what_it_cannot_estimate():
    with pytest.raises(ValueError):
        pass_hat_k(2, 1, 3)
    with pytest.raises(ValueError):
        pass_hat_k(3, 4, 1)
    with pytest.raises(ValueError):
        pass_hat_k(3, 1, 0)


# --- the table ------------------------------------------------------------


def test_a_trial_recorded_twice_is_refused():
    with pytest.raises(ValueError, match="duplicate"):
        ObservationTable(
            observations=(
                Observation(task_id="1", trial=0, value=True),
                Observation(task_id="1", trial=0, value=False),
            )
        )


def test_tasks_are_weighted_equally_not_by_surviving_trials():
    """After losses the pooled rate and the task mean differ; the task mean is used."""
    ragged = table({"a": [True] * 5, "b": [False]})
    assert pass_k(ragged, resamples=FAST).value == 0.5  # pooled would be 5/6


def test_tasks_too_short_for_k_are_named_not_averaged_over():
    estimate = pass_k(table({"a": [True, True, True], "b": [True]}), 3, resamples=FAST)
    assert estimate.skipped_tasks == ("b",)
    assert estimate.n_tasks == 1


def test_no_scorable_task_raises():
    with pytest.raises(ValueError):
        pass_k(table({"a": [True]}), 2, resamples=FAST)


# --- the interval ---------------------------------------------------------


def test_the_bootstrap_is_reproducible_from_its_seed():
    t = table({str(i): [i % 3 == 0, i % 2 == 0] for i in range(20)})
    assert pass_k(t, resamples=FAST, seed=4) == pass_k(t, resamples=FAST, seed=4)
    assert pass_k(t, resamples=FAST, seed=4) != pass_k(t, resamples=FAST, seed=5)


def test_the_interval_contains_the_estimate():
    t = table({str(i): [i % 3 == 0, i % 2 == 0, i % 5 == 0] for i in range(30)})
    e = pass_k(t, resamples=FAST)
    assert e.lo <= e.value <= e.hi


def test_the_bootstrap_resamples_tasks_not_trials():
    """Ten tasks, each internally unanimous: the information is ten draws, not fifty.

    A trial-level bootstrap would give roughly [0.36, 0.64] here; clustering by
    task has to give an interval about as wide as n=10 implies.
    """
    t = table({str(i): [i < 5] * 5 for i in range(10)})
    e = pass_k(t, resamples=FAST)
    assert e.value == 0.5
    assert e.lo <= 0.3
    assert e.hi >= 0.7


def test_a_unanimous_table_has_a_degenerate_interval():
    e = pass_k(table({str(i): [True, True] for i in range(5)}), resamples=FAST)
    assert (e.lo, e.value, e.hi) == (1.0, 1.0, 1.0)


def test_a_rate_is_labelled_as_what_it_measures():
    violations = table({"a": [True, False], "b": [True, True]})
    e = rate(violations, name="parallel_call_violation_rate", resamples=FAST)
    assert e.statistic == "parallel_call_violation_rate"
    assert e.value == 0.75


# --- stability ------------------------------------------------------------


def test_stability_classifies_tasks_and_does_not_guess_about_single_trials():
    s = stability(
        table(
            {
                "always": [True, True],
                "never": [False, False, False],
                "flip": [True, False],
                "once": [True],
            }
        )
    )
    assert (s.always, s.never, s.unstable, s.single_trial) == (1, 1, 1, 1)
    assert s.flip_rate == pytest.approx(1 / 3)


# --- McNemar --------------------------------------------------------------


def test_mcnemar_on_the_w2_smoke_counts_detects_nothing():
    """One discordant task (baseline passed, full failed) is p = 1."""
    assert mcnemar_exact(0, 1) == 1.0


@pytest.mark.parametrize(
    "b,c,expected",
    [(0, 0, 1.0), (10, 0, 2 / 2**10), (6, 0, 2 / 2**6), (3, 3, 1.0)],
)
def test_mcnemar_exact_values(b, c, expected):
    assert mcnemar_exact(b, c) == pytest.approx(expected)
    assert mcnemar_exact(c, b) == pytest.approx(expected)


# --- paired comparison ----------------------------------------------------


def paired_arms(n_tasks: int, both: int, treatment_extra: int, control_extra: int = 0):
    control_rows, treatment_rows = {}, {}
    for i in range(n_tasks):
        base = i < both
        treatment_rows[str(i)] = [base or both <= i < both + treatment_extra]
        control_rows[str(i)] = [
            base or both + treatment_extra <= i < both + treatment_extra + control_extra
        ]
    return table(treatment_rows), table(control_rows)


def test_single_binary_trials_are_tested_with_mcnemar():
    treatment, control = paired_arms(40, both=20, treatment_extra=6, control_extra=1)
    result = compare_paired(treatment, control, resamples=FAST)
    assert result.test is PairedTest.MCNEMAR_EXACT
    assert (result.treatment_higher, result.control_higher) == (6, 1)
    assert result.p_value == pytest.approx(mcnemar_exact(6, 1))


def test_pairing_detects_what_two_independent_intervals_would_not():
    """0.65 against 0.50 on 40 tasks: overlapping marginals, clear paired effect."""
    treatment, control = paired_arms(40, both=20, treatment_extra=6)
    result = compare_paired(treatment, control, resamples=FAST)

    t_alone = pass_k(treatment, resamples=FAST)
    c_alone = pass_k(control, resamples=FAST)
    assert t_alone.lo < c_alone.hi  # the marginal intervals overlap

    assert result.difference.value == pytest.approx(0.15)
    assert result.difference.lo > 0
    assert result.p_value < 0.05


def test_identical_arms_show_no_difference():
    treatment, control = paired_arms(20, both=10, treatment_extra=0)
    result = compare_paired(treatment, control, resamples=FAST)
    assert result.difference.value == 0.0
    assert result.p_value == 1.0
    assert result.ties == 20


def test_multi_trial_differences_use_the_whole_per_task_estimate():
    treatment = table({str(i): [True, True, i % 2 == 0] for i in range(12)})
    control = table({str(i): [True, i % 3 == 0, False] for i in range(12)})
    result = compare_paired(treatment, control, k=1, resamples=FAST)
    assert result.test is PairedTest.SIGN_FLIP_MONTE_CARLO
    assert 0 < result.p_value <= 1
    assert result.balanced


def test_the_monte_carlo_p_value_is_never_zero():
    treatment = table({str(i): [True, True, True] for i in range(15)})
    control = table({str(i): [i % 2 == 0, False, i % 3 == 0] for i in range(15)})
    result = compare_paired(treatment, control, resamples=FAST)
    assert result.p_value >= 1 / (FAST + 1)


def test_tasks_missing_from_one_arm_are_named_and_not_compared():
    treatment = table({"a": [True], "b": [True], "c": [False]})
    control = table({"a": [False], "b": [True]})
    result = compare_paired(treatment, control, resamples=FAST)
    assert result.tasks_only_in_treatment == ("c",)
    assert result.difference.n_tasks == 2


def test_tasks_below_k_in_either_arm_are_named():
    treatment = table({"a": [True, True], "b": [True, True]})
    control = table({"a": [True, False], "b": [True]})
    result = compare_paired(treatment, control, k=2, resamples=FAST)
    assert result.tasks_below_k == ("b",)
    assert result.difference.n_tasks == 1


# --- loading tau2 results -------------------------------------------------


def sim(task_id, trial, *, reward=None, reason=TerminationReason.USER_STOP, breakdown=None,
        error_type=None):
    reward_info = (
        None
        if reward is None
        else RewardInfo(reward=reward, reward_breakdown=breakdown)
    )
    return SimulationRun(
        id=f"{task_id}-{trial}",
        task_id=task_id,
        start_time="t0",
        end_time="t1",
        duration=1.0,
        termination_reason=reason,
        reward_info=reward_info,
        trial=trial,
        info={"error_type": error_type} if error_type else None,
    )


def test_rewards_become_observations():
    t = observations_from_simulations([sim("1", 0, reward=1.0), sim("1", 1, reward=0.0)])
    assert t.by_task() == {"1": (True, False)}
    assert t.n_excluded == 0


def test_agent_failures_are_the_agent_s_to_own():
    """MAX_STEPS and AGENT_ERROR carry a reward: they are failures, not exclusions."""
    t = observations_from_simulations(
        [sim("1", 0, reward=0.0, reason=TerminationReason.MAX_STEPS)]
    )
    assert t.by_task() == {"1": (False,)}


def test_infrastructure_errors_are_excluded_with_their_cause():
    t = observations_from_simulations(
        [
            sim("1", 0, reward=1.0),
            sim("2", 0, reason=TerminationReason.INFRASTRUCTURE_ERROR, error_type="APIError"),
            sim("3", 0, reason=TerminationReason.INFRASTRUCTURE_ERROR, error_type="StateError"),
        ]
    )
    assert t.by_task() == {"1": (True,)}
    assert t.excluded == {
        "infrastructure_error:APIError": 1,
        "infrastructure_error:StateError": 1,
    }
    assert t.excluded_tasks == {"2": 1, "3": 1}


def test_exclusions_can_be_scored_as_failures_for_a_bound():
    sims = [
        sim("1", 0, reward=1.0),
        sim("2", 0, reason=TerminationReason.INFRASTRUCTURE_ERROR, error_type="RuntimeError"),
    ]
    t = observations_from_simulations(sims, policy=ExclusionPolicy.COUNT_AS_FAILURE)
    assert t.by_task() == {"1": (True,), "2": (False,)}
    assert t.n_excluded == 1  # still on the record


def test_a_single_reward_component_can_be_scored():
    sims = [
        sim("1", 0, reward=0.0, breakdown={RewardType.DB: 1.0, RewardType.NL_ASSERTION: 0.0}),
        sim("2", 0, reward=1.0, breakdown={RewardType.DB: 1.0}),
    ]
    db = observations_from_simulations(sims, component="DB")
    nl = observations_from_simulations(sims, component="NL_ASSERTION")
    assert db.by_task() == {"1": (True,), "2": (True,)}
    assert nl.by_task() == {"1": (False,)}
    assert nl.excluded == {"component_absent:NL_ASSERTION": 1}


def test_an_absent_component_is_not_turned_into_a_failure():
    sims = [sim("2", 0, reward=1.0, breakdown={RewardType.DB: 1.0})]
    nl = observations_from_simulations(
        sims, component="NL_ASSERTION", policy=ExclusionPolicy.COUNT_AS_FAILURE
    )
    assert nl.by_task() == {}


def test_differential_exclusion_is_flagged_in_the_comparison():
    """The failure mode the loader exists to expose: one arm quietly loses its hard tasks."""
    control = observations_from_simulations(
        [sim(str(i), 0, reward=float(i < 4)) for i in range(10)]
    )
    treatment = observations_from_simulations(
        [sim(str(i), 0, reward=float(i < 4)) for i in range(6)]
        + [
            sim(str(i), 0, reason=TerminationReason.INFRASTRUCTURE_ERROR, error_type="StateError")
            for i in range(6, 10)
        ]
    )
    result = compare_paired(treatment, control, resamples=FAST)
    assert result.differential_exclusion
    assert (result.treatment_excluded, result.control_excluded) == (4, 0)
    assert result.tasks_only_in_control == ("6", "7", "8", "9")
    # Paired on the survivors the arms are identical, which is correct as far as
    # it goes -- but an unpaired read would show 0.67 against 0.40, and only the
    # exclusion accounting shows that the four lost tasks were all failures.
    assert result.difference.value == 0.0
    assert result.treatment > pass_k(control, resamples=FAST).value
