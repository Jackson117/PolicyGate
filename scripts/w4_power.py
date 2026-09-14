#!/usr/bin/env python
"""W4 design sizing: minimum detectable effect for candidate designs, from W1 data.

Reads W1's baseline results, fits the between-task difficulty distribution for
the combined reward and for the DB component, and reports -- per design -- the
smallest improvement the paired comparison detects with 80% power.

Two families are reported for every design:
  headline     full vs baseline, a single pre-specified comparison at alpha
  attribution  each single-mechanism arm vs baseline, Holm over 7; power is
               computed at alpha/7, Holm's first step, so it is a floor

Caveats that travel with every number printed here:
  * W1 ran tau2's own llm_agent on retail's first 30 tasks. W4 runs this
    project's adapter on all 114. The difficulty distribution is extrapolated.
  * The effect is modelled as a constant logit shift on every task.
  * Cost per simulation is the W2 smoke's full-arm figure at OpenAI list rates,
    excluding the NL judge -- an extrapolation from 12 simulations.

Usage:
    uv run python scripts/w4_power.py experiments/2026-09-02_w1_baseline/results.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("results", type=Path)
    p.add_argument("--n-tasks", type=int, default=114)
    p.add_argument("--trials", type=int, nargs="+", default=[1, 2, 3, 5])
    p.add_argument("--arms", type=int, default=9, help="baseline + full + 7 single-mechanism arms")
    p.add_argument("--attribution-comparisons", type=int, default=7)
    p.add_argument("--cost-per-sim", type=float, default=0.0137)
    p.add_argument("--replicates", type=int, default=1000)
    p.add_argument("--target-power", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from tau2.data_model.simulation import Results

    from policygate.eval.passk import observations_from_simulations
    from policygate.eval.power import Design, fit_beta_binomial, minimum_detectable_effect

    simulations = Results.load(args.results).simulations
    started = time.time()
    report: dict = {
        "source": str(args.results),
        "n_tasks": args.n_tasks,
        "arms": args.arms,
        "cost_per_sim": args.cost_per_sim,
        "replicates": args.replicates,
        "target_power": args.target_power,
        "metrics": {},
    }

    for metric, component in (("reward", None), ("DB", "DB")):
        table = observations_from_simulations(simulations, component=component)
        by_task = table.by_task()
        trials_seen = {len(v) for v in by_task.values()}
        if len(trials_seen) != 1:
            raise SystemExit(f"{metric}: unbalanced table {sorted(trials_seen)}; the fit needs one")
        fit = fit_beta_binomial([sum(v) for v in by_task.values()], trials_seen.pop())
        check = fit.stability_check()
        print(
            f"\n=== {metric}: {fit.n_tasks} W1 tasks x {fit.trials} trials; "
            f"mean {fit.mean:.3f}, ICC {fit.intraclass_correlation:.3f}, "
            f"Beta({fit.alpha:.2f}, {fit.beta:.2f})"
        )
        print(
            f"    fit check  always/never/unstable  observed "
            f"{check.observed_always}/{check.observed_never}/{check.observed_unstable}  "
            f"expected {check.expected_always:.1f}/{check.expected_never:.1f}/"
            f"{check.expected_unstable:.1f}"
        )
        print(
            f"    {'trials':>6}  {'k':>2}  {'cost':>7}  "
            f"{'headline MDE':>13}  {'attribution MDE (floor)':>24}"
        )

        rows = []
        for trials in args.trials:
            for k in sorted({1, min(3, trials)} if trials >= 3 else {1}):
                cost = args.arms * args.n_tasks * trials * args.cost_per_sim
                cells = {}
                families = (("headline", 1), ("attribution", args.attribution_comparisons))
                for family, comparisons in families:
                    mde = minimum_detectable_effect(
                        fit,
                        Design(n_tasks=args.n_tasks, n_trials=trials, k=k, comparisons=comparisons),
                        target_power=args.target_power,
                        replicates=args.replicates,
                        seed=args.seed,
                    )
                    cells[family] = mde.model_dump(mode="json")
                rows.append({"trials": trials, "k": k, "cost": cost, **cells})

                def show(cell):
                    if not cell["achievable"]:
                        return "not reached"
                    return f"+{cell['expected_difference'] * 100:.1f}pp"

                print(
                    f"    {trials:>6}  {k:>2}  ${cost:>6.0f}  "
                    f"{show(cells['headline']):>13}  {show(cells['attribution']):>24}"
                )

        report["metrics"][metric] = {
            "fit": fit.model_dump(mode="json"),
            "stability_check": check.model_dump(mode="json"),
            "rows": rows,
        }

    out = args.out or REPO / "experiments" / f"{date.today():%Y-%m-%d}_w4_power" / "power.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n{(time.time() - started) / 60:.1f} min -> {out}")
    print("MDE = expected change in pass^k across tasks, at the stated power, for an improvement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
