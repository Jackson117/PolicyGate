#!/usr/bin/env python
"""Run the W4 ablation plan (docs/w4_design.md) with a hard budget.

Resumable: re-running the same command with the same output directory continues
where it stopped, and a different plan in that directory is refused. Raise
--budget to continue past a budget stop.

Usage:
    uv run python scripts/w4_run.py --budget 30 --dry-run
    uv run python scripts/w4_run.py --budget 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--budget", type=float, required=True, help="USD, OpenAI list-rate equivalent")
    p.add_argument("--out", type=Path, default=REPO / "experiments" / "w4_ablation")
    p.add_argument("--agent-llm", default="openai/gpt-4.1-mini")
    p.add_argument("--user-llm", default="openai/gpt-4.1-mini")
    p.add_argument("--judge-llm", default="openai/gpt-4.1")
    p.add_argument("--headline-trials", type=int, default=5)
    p.add_argument("--attribution-trials", type=int, default=1)
    p.add_argument("--num-tasks", type=int, default=None, help="default: all tasks in the domain")
    p.add_argument("--chunk-size", type=int, default=20)
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from tau2.runner import get_tasks

    from policygate.eval.runner import (
        PRIOR_COST_PER_SIMULATION,
        UNACCOUNTED,
        RunPlan,
        Tau2ChunkExecutor,
        run_plan,
    )

    task_ids = [t.id for t in get_tasks("retail")]
    if args.num_tasks is not None:
        task_ids = task_ids[: args.num_tasks]
    plan = RunPlan.w4(
        task_ids,
        agent_llm=args.agent_llm,
        user_llm=args.user_llm,
        judge_llm=args.judge_llm,
        headline_trials=args.headline_trials,
        attribution_trials=args.attribution_trials,
        chunk_size=args.chunk_size,
    )

    print(f"plan {plan.fingerprint()[:12]}  ->  {args.out}")
    chunks = len(plan.schedule())
    print(f"{len(plan.task_ids)} tasks, {chunks} chunks, {plan.simulations} simulations")
    for arm in plan.arms:
        print(f"  {arm.phase:<11} {arm.name:<40} x{arm.trials}")
    estimate = plan.simulations * PRIOR_COST_PER_SIMULATION
    print(f"prior estimate ${estimate:.2f} at ${PRIOR_COST_PER_SIMULATION}/simulation; "
          f"budget ${args.budget:.2f}")
    print(f"not counted against the budget: {', '.join(UNACCOUNTED)}")
    if args.dry_run:
        return 0

    executor = Tau2ChunkExecutor(plan, args.out, max_concurrency=args.max_concurrency)
    outcome = run_plan(plan, args.out, budget_usd=args.budget, executor=executor)
    print(json.dumps(outcome.model_dump(mode="json"), indent=2))
    return 0 if outcome.status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
