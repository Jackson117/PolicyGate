#!/usr/bin/env python
"""W1 signal-quality analysis over a tau2 Results file.

Answers the W1 stop-loss question. Reports, with intervals, never point estimates
alone:

  pass^1 / pass^k    -- level, with a task-clustered bootstrap CI
  per-trial spread   -- how much the aggregate moves between identical trials
  flip rate          -- fraction of tasks that did NOT give the same outcome in
                        every trial. This, not the aggregate spread, is the real
                        instrument: on n tasks the aggregate moves in steps of
                        1/n, so with n=30 a "+/-2% spread" band sits below the
                        resolution of the measurement and can be satisfied by
                        flips that happen to cancel.
  cost / latency     -- per simulation and extrapolated to a full ablation matrix

Usage:
    .venv/bin/python scripts/w1_analyze.py experiments/<run>/results.json
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

SUCCESS_THRESHOLD = 1.0 - 1e-9  # tau2 treats reward == 1 as success


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ablation-cells",
        type=int,
        default=10,
        help="Number of config cells you expect to run in W4, for the cost extrapolation.",
    )
    return p.parse_args()


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """Unbiased pass^k: P(all of k trials drawn without replacement succeeded)."""
    if k > num_trials:
        return float("nan")
    if success_count < k:
        return 0.0
    return math.comb(success_count, k) / math.comb(num_trials, k)


def ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def main() -> int:
    args = parse_args()

    from tau2.data_model.simulation import Results

    results = Results.load(args.results)
    sims = results.simulations

    # outcomes[task_id][trial] = 1/0
    outcomes: dict[str, dict[int, int]] = defaultdict(dict)
    costs: list[float] = []
    durations: list[float] = []
    terminations: dict[str, int] = defaultdict(int)
    retries_used = 0
    no_reward = 0

    for s in sims:
        terminations[str(getattr(s.termination_reason, "value", s.termination_reason))] += 1
        retries_used += s.hallucination_retries_used or 0
        durations.append(s.duration)
        costs.append((s.agent_cost or 0.0) + (s.user_cost or 0.0))
        if s.reward_info is None or s.reward_info.reward is None:
            no_reward += 1
            continue
        outcomes[s.task_id][s.trial if s.trial is not None else 0] = int(
            s.reward_info.reward >= SUCCESS_THRESHOLD
        )

    task_ids = sorted(outcomes)
    trials_per_task = {t: len(outcomes[t]) for t in task_ids}
    n_trials = max(trials_per_task.values()) if trials_per_task else 0
    n_tasks = len(task_ids)

    print(f"file            {args.results}")
    print(f"simulations     {len(sims)}  ({n_tasks} tasks x up to {n_trials} trials)")
    if no_reward:
        print(f"  !! {no_reward} simulations carried no reward and were dropped from rates")
    ragged = [t for t, c in trials_per_task.items() if c != n_trials]
    if ragged:
        print(f"  !! {len(ragged)} tasks have fewer than {n_trials} trials: {ragged[:5]}")
    if retries_used:
        print(f"  !! {retries_used} hallucination retries fired -- run-to-run variance is understated")
    print()

    # ---- level ----
    per_task_succ = np.array([sum(outcomes[t].values()) for t in task_ids], dtype=float)
    per_task_n = np.array([len(outcomes[t]) for t in task_ids], dtype=float)
    pass1 = float(per_task_succ.sum() / per_task_n.sum())

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n_tasks, size=(args.bootstrap, n_tasks))
    boot_pass1 = per_task_succ[idx].sum(axis=1) / per_task_n[idx].sum(axis=1)
    lo, hi = ci(boot_pass1)
    print("--- level (task-clustered bootstrap, 95% CI) ---")
    print(f"pass^1          {pass1:.3f}  [{lo:.3f}, {hi:.3f}]   (+/-{(hi - lo) / 2 * 100:.1f}pp half-width)")

    for k in range(2, n_trials + 1):
        pk_per_task = np.array(
            [pass_hat_k(len(outcomes[t]), sum(outcomes[t].values()), k) for t in task_ids]
        )
        boot_pk = pk_per_task[idx].mean(axis=1)
        klo, khi = ci(boot_pk)
        print(f"pass^{k}          {pk_per_task.mean():.3f}  [{klo:.3f}, {khi:.3f}]")
    print()

    # ---- run-to-run noise ----
    print("--- run-to-run noise (same 30 tasks, different execution seed) ---")
    trial_rates = []
    for tr in range(n_trials):
        vals = [outcomes[t][tr] for t in task_ids if tr in outcomes[t]]
        if vals:
            trial_rates.append(sum(vals) / len(vals))
            print(f"  trial {tr}       {trial_rates[-1]:.3f}  ({sum(vals)}/{len(vals)})")
    if len(trial_rates) > 1:
        spread = max(trial_rates) - min(trial_rates)
        print(f"  spread        {spread * 100:.1f}pp   (resolution floor: 1 task = {100 / n_tasks:.1f}pp)")

    flipped = [t for t in task_ids if len(set(outcomes[t].values())) > 1]
    flip_rate = len(flipped) / n_tasks if n_tasks else float("nan")
    boot_flip = np.array([len(set(outcomes[t].values())) > 1 for t in task_ids], dtype=float)[idx].mean(axis=1)
    flo, fhi = ci(boot_flip)
    print(f"  flip rate     {flip_rate:.3f}  [{flo:.3f}, {fhi:.3f}]   ({len(flipped)}/{n_tasks} tasks not unanimous)")
    always = sum(1 for t in task_ids if all(v == 1 for v in outcomes[t].values()))
    never = sum(1 for t in task_ids if all(v == 0 for v in outcomes[t].values()))
    print(f"  always pass   {always}   always fail {never}   unstable {len(flipped)}")
    print()

    # ---- what this subset can detect ----
    print("--- detectable effect (paired design, harness vs baseline on these same tasks) ---")
    print(f"  'always fail' tasks are the headroom the harness can act on: {never}/{n_tasks}")
    print(f"  'always pass' tasks can only be broken, not improved:        {always}/{n_tasks}")
    print(f"  unstable tasks contribute noise to both arms:                {len(flipped)}/{n_tasks}")
    print("  A paired test (McNemar over discordant tasks) is the right comparison and is")
    print("  far more powerful than comparing two independent aggregate rates -- task")
    print("  difficulty cancels. Report discordant-pair counts, not just two rates.")
    print()

    # ---- reward component decomposition ----
    # retail scores tasks on DB AND an LLM-judged NL assertion (112/114 tasks).
    # Splitting them answers the question the aggregate cannot: is the run-to-run
    # noise the agent taking different world-state actions (signal the harness is
    # meant to fix), or the judge grading the same trajectory differently
    # (measurement error that just dilutes any effect)?
    comp: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for s_ in sims:
        if s_.reward_info is None:
            continue
        for rt, val in (s_.reward_info.reward_breakdown or {}).items():
            comp[str(getattr(rt, "value", rt))][s_.task_id][s_.trial] = val

    if comp:
        print("--- reward components (retail = DB AND NL_ASSERTION) ---")
        for name in sorted(comp):
            d = comp[name]
            ct = sorted(d)
            mean = sum(sum(d[t].values()) for t in ct) / sum(len(d[t]) for t in ct)
            unstable = [t for t in ct if len(set(d[t].values())) > 1]
            never = sum(1 for t in ct if all(v < SUCCESS_THRESHOLD for v in d[t].values()))
            pk = np.array([pass_hat_k(len(d[t]), sum(v >= SUCCESS_THRESHOLD for v in d[t].values()), n_trials) for t in ct])
            print(f"  {name:14s} pass^1={mean:.3f}  pass^{n_trials}={pk.mean():.3f}  "
                  f"unstable={len(unstable):2d}/{len(ct)} ({len(unstable) / len(ct):.0%})  never={never}")
        print("  Noise concentrated in DB = the agent really does act differently run to run.")
        print("  That is the target of the harness, not measurement error.")
        print()

    # ---- cost ----
    total_cost = sum(costs)
    have_cost = any(c > 0 for c in costs)
    print("--- cost and latency ---")
    if have_cost:
        per_sim = total_cost / len(sims)
        per_trial = total_cost / n_trials if n_trials else float("nan")
        print(f"  total         ${total_cost:.2f}   per simulation ${per_sim:.4f}   per trial ${per_trial:.2f}")
        print(f"  extrapolation: {args.ablation_cells} ablation cells x {n_trials} trials x {n_tasks} tasks"
              f" = ${per_sim * args.ablation_cells * n_trials * n_tasks:.2f}")
        print(f"                 same, but on all 114 retail tasks"
              f" = ${per_sim * args.ablation_cells * n_trials * 114:.2f}")
    else:
        print("  no cost recorded by litellm for this model -- price it manually from token counts")
    d = np.array(durations)
    print(f"  duration/sim  median {np.median(d):.1f}s   p90 {np.quantile(d, 0.9):.1f}s   max {d.max():.1f}s")
    print()

    print("--- termination reasons ---")
    for reason, count in sorted(terminations.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:28s} {count}")
    infra = terminations.get("infrastructure_error", 0) + terminations.get("unexpected_error", 0)
    if infra:
        print(f"  !! {infra} simulations died on infrastructure, not on agent behaviour."
              " Those are not evidence about the agent; exclude or re-run them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
