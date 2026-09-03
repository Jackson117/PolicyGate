#!/usr/bin/env python
"""W1 feasibility baseline: run an unmodified tau2 LLM agent on a task subset.

This deliberately uses tau2's own agent with zero policygate harness involvement.
Its only job is to answer the W1 go/no-go question: is the trial-to-trial signal
clean enough, and cheap enough, to run ablations against?

Usage:
    .venv/bin/python scripts/w1_baseline.py \
        --agent-llm anthropic/claude-haiku-4-5-20251001 \
        --user-llm  anthropic/claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default="retail")
    p.add_argument("--num-tasks", type=int, default=30)
    p.add_argument("--num-trials", type=int, default=3)
    p.add_argument("--agent-llm", required=True)
    p.add_argument("--user-llm", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--judge-llm",
        default="openai/gpt-4.1",
        help=(
            "Model for the NL-assertion LLM judge. tau2 hardcodes "
            "DEFAULT_LLM_NL_ASSERTIONS='gpt-4.1-2025-04-14' -- a bare id with no "
            "provider prefix, which litellm resolves against OPENAI_BASE_URL. That "
            "exact id does not exist on the AIHubMix catalogue, so every task with "
            "non-empty nl_assertions fails deterministically with "
            "'no_available_channel' AFTER the conversation has already run and been "
            "paid for, and tau2 discards the whole simulation as an "
            "INFRASTRUCTURE_ERROR. Pass a routable id instead."
        ),
    )
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument(
        "--hallucination-retries",
        type=int,
        default=0,
        help=(
            "tau2 defaults this to 3: on a detected user-simulator hallucination it "
            "silently re-runs the simulation under a different seed. That is a hidden "
            "resampler which suppresses exactly the run-to-run variance W1 exists to "
            "measure, and it costs extra tokens. Default it off here; set it to 3 only "
            "when reproducing published tau2 numbers."
        ),
    )
    p.add_argument("--out", default=None, help="Output directory (default: experiments/<date>_w1_baseline)")
    p.add_argument("--dry-run", action="store_true", help="Print the resolved config and exit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Imported late so --dry-run/--help stay fast and key-free.
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import get_tasks, run_domain

    # The judge model is a module-level constant, not a config field, and the
    # evaluator binds it by value at import time -- so patching tau2.config is not
    # enough, the name has to be rebound in the module that uses it.
    from tau2.evaluator import evaluator_nl_assertions

    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = args.judge_llm

    all_tasks = get_tasks(args.domain)
    # Deterministic subset: the first N task ids in the domain's own order.
    # Not a random sample -- W1 measures noise, not a population estimate, and a
    # fixed subset keeps every later ablation comparable to this run.
    task_ids = [t.id for t in all_tasks[: args.num_tasks]]

    out_dir = Path(args.out) if args.out else REPO / "experiments" / f"{date.today():%Y-%m-%d}_w1_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    # tau2 treats save_to as a RUN NAME, not a path: it always writes to
    # DATA_DIR/simulations/<save_to>/results.json. Passing a path here creates a
    # directory literally named "....json" inside the pinned tau2 checkout. Give
    # it a name, then copy the results next to the config that produced them.
    run_name = out_dir.name

    config = TextRunConfig(
        domain=args.domain,
        task_ids=task_ids,
        agent="llm_agent",
        llm_agent=args.agent_llm,
        llm_args_agent={"temperature": 0.0},
        user="user_simulator",
        llm_user=args.user_llm,
        llm_args_user={"temperature": 0.0},
        num_trials=args.num_trials,
        seed=args.seed,
        max_steps=args.max_steps,
        max_concurrency=args.max_concurrency,
        hallucination_retries=args.hallucination_retries,
        save_to=run_name,
    )

    # The config that produced the results lives next to the results. Minimum bar
    # for reproducibility, and the source material for docs/methodology.md.
    (out_dir / "config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=2))
    print(f"config -> {out_dir / 'config.json'}")
    print(f"{args.domain}: {len(all_tasks)} tasks total, running {len(task_ids)} x {args.num_trials} trials")

    if args.dry_run:
        print("dry run, not executing")
        return 0

    started = time.time()
    results = run_domain(config)
    elapsed = time.time() - started

    # Results and the config that produced them must live together.
    results.save(results_path)

    (out_dir / "wallclock.json").write_text(
        json.dumps({"total_seconds": elapsed, "num_simulations": len(task_ids) * args.num_trials}, indent=2)
    )
    print(f"\ndone in {elapsed / 60:.1f} min -> {results_path}")
    print(f"analyze with: .venv/bin/python scripts/w1_analyze.py {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
