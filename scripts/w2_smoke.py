#!/usr/bin/env python
"""W2 smoke run: put the harness agent in front of a real model, once.

Not an experiment. Its job is to answer questions the unit tests cannot, because
the tests script the model:

* does the adapter satisfy tau2's agent contract end to end;
* does the harness refuse anything a real model actually proposes, and under
  which rules;
* what does the approval gate cost in interruptions.

Two things this run deliberately measures that tau2's own accounting cannot.
Refused proposals never enter the returned trajectory, so ``agent_cost`` on a
SimulationRun *undercounts* the harness arm -- every internal retry is invisible
to it. And the audit chain lives in the agent's state, which tau2 does not
persist. Both are collected here instead.

Usage:
    uv run python scripts/w2_smoke.py --num-tasks 3 --agent-llm openai/gpt-4.1-mini \
        --user-llm openai/gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

AGENT_NAME = "policygate_agent"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default="retail")
    p.add_argument("--num-tasks", type=int, default=3)
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--agent-llm", default="openai/gpt-4.1-mini")
    p.add_argument("--user-llm", default="openai/gpt-4.1-mini")
    p.add_argument(
        "--judge-llm",
        default="openai/gpt-4.1",
        help="tau2 hardcodes an id AIHubMix cannot route; see scripts/w1_baseline.py.",
    )
    p.add_argument(
        "--arm",
        default="full",
        choices=["full", "baseline"],
        help="baseline switches every harness mechanism off, through the same adapter.",
    )
    p.add_argument(
        "--only-write-tasks",
        action="store_true",
        help=(
            "Keep only tasks whose reference trajectory contains a write. The "
            "approval gate cannot fire on a task that never writes, and the first "
            "three retail tasks are read-only. This is a selection on the task, "
            "not on the outcome, but it still makes the reward here "
            "non-comparable to a run over the unselected prefix."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-concurrency", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-internal-attempts", type=int, default=3)
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_factory(args, sessions: list) -> Any:
    """Register the harness agent with tau2 and collect what each one did.

    tau2 builds a fresh agent per simulation and hands the factory the task, so
    every agent here belongs to exactly one simulation and can be read back
    afterwards.
    """
    from tau2.registry import registry
    from tau2.utils.utils import DATA_DIR

    from policygate.adapters.tau2_agent import HarnessAgent
    from policygate.adapters.tool_mapping import (
        RETAIL_ESCALATION,
        RETAIL_IDENTITY_LOOKUPS,
        retail_constraints,
        retail_facts,
        tool_specs,
    )
    from policygate.harness.config import HarnessConfig

    db = json.loads((DATA_DIR / "tau2" / "domains" / args.domain / "db.json").read_text())
    environment = registry.get_env_constructor(args.domain)()
    constraints = retail_constraints(db, tools=environment.get_tools())
    toolkit = environment.tools
    catalogue = tool_specs(
        toolkit, identity_lookups=RETAIL_IDENTITY_LOOKUPS, escalation=RETAIL_ESCALATION
    )
    config = HarnessConfig.full() if args.arm == "full" else HarnessConfig.baseline()

    class RecordingHarnessAgent(HarnessAgent):
        """The agent under test, plus the bookkeeping tau2 has no place for."""

        def __init__(self, *a: Any, task_id: str, **kw: Any) -> None:
            super().__init__(*a, **kw)
            self.task_id = task_id
            self.proposals = 0
            self.llm_cost = 0.0
            self.last_state = None

        def _propose(self, state):
            response = super()._propose(state)
            self.proposals += 1
            self.llm_cost += response.cost or 0.0
            return response

        def generate_next_message(self, message, state):
            self.last_state = state
            return super().generate_next_message(message, state)

    def factory(tools, domain_policy, llm=None, llm_args=None, task=None, **_ignored):
        agent = RecordingHarnessAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            catalogue=catalogue,
            constraints=constraints,
            extract_facts=retail_facts,
            config=config,
            llm_args=llm_args or {},
            session_id=str(uuid.uuid4()),
            max_internal_attempts=args.max_internal_attempts,
            task_id=task.id if task is not None else "unknown",
        )
        sessions.append(agent)
        return agent

    registry.register_agent_factory(factory, AGENT_NAME)
    _CATALOGUE.extend(catalogue)
    return config


def chain_of(agent) -> list:
    state = agent.last_state
    return list(state.chain.records) if state is not None else []


def summarise(agent, write_tools) -> dict[str, Any]:
    from policygate.harness.approval import summarise_approvals
    from policygate.harness.types import AuditEvent, DecisionOutcome

    records = chain_of(agent)
    report = summarise_approvals(records, write_tools=write_tools)
    denials = Counter(
        r.decision.rule
        for r in records
        if r.decision is not None and r.decision.outcome is not DecisionOutcome.ALLOW
    )
    verification = agent.last_state.chain.verify() if agent.last_state else None
    return {
        "task_id": agent.task_id,
        "session_id": agent.session_id,
        "proposals": agent.proposals,
        "llm_cost_including_refused": agent.llm_cost,
        "records": len(records),
        "chain_intact": None if verification is None else verification.intact,
        "denials_by_rule": dict(denials),
        "executed": sum(1 for r in records if r.event is AuditEvent.TOOL_EXECUTED),
        "gate": report.gate.model_dump(),
        "gate_trigger_rate": report.gate.gate_trigger_rate,
        "approval_pass_rate": report.gate.approval_pass_rate,
    }


def main() -> int:
    args = parse_args()

    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator import evaluator_nl_assertions
    from tau2.runner import get_tasks, run_domain

    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = args.judge_llm

    sessions: list = []
    config = build_factory(args, sessions)

    all_tasks = get_tasks(args.domain)
    if args.only_write_tasks:
        from policygate.adapters.tool_mapping import ground_truth
        from policygate.harness.types import ToolKind

        write_names = {s.name for s in catalogue_of_config() if s.kind is ToolKind.WRITE}
        all_tasks = [
            t
            for t in all_tasks
            if any(a.name in write_names for a in ground_truth(t).actions)
        ]
    # A deterministic prefix, as in W1: the subset is fixed, not sampled.
    task_ids = [t.id for t in all_tasks[: args.num_tasks]]

    out_dir = (
        Path(args.out)
        if args.out
        else REPO / "experiments" / f"{date.today():%Y-%m-%d}_w2_smoke_{args.arm}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = TextRunConfig(
        domain=args.domain,
        task_ids=task_ids,
        agent=AGENT_NAME,
        llm_agent=args.agent_llm,
        llm_args_agent={"temperature": 0.0},
        user="user_simulator",
        llm_user=args.user_llm,
        llm_args_user={"temperature": 0.0},
        num_trials=args.num_trials,
        seed=args.seed,
        max_steps=args.max_steps,
        max_concurrency=args.max_concurrency,
        hallucination_retries=0,
        # Unique per invocation. tau2 writes to DATA_DIR/simulations/<save_to>/
        # and, finding a file there, *prompts* to resume -- which in a
        # non-interactive shell is an EOFError that kills the run after the
        # config has been printed but before anything has run.
        save_to=f"{out_dir.name}_{int(time.time())}",
    )
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "run": run_config.model_dump(mode="json"),
                "harness": config.model_dump(mode="json"),
                "max_internal_attempts": args.max_internal_attempts,
            },
            indent=2,
        )
    )
    n = len(task_ids) * args.num_trials
    print(f"arm={args.arm} {args.domain}: {len(task_ids)} tasks x {args.num_trials} = {n} sims")
    print(f"config -> {out_dir / 'config.json'}")
    if args.dry_run:
        print("dry run, not executing")
        return 0

    started = time.time()
    results = run_domain(run_config)
    elapsed = time.time() - started
    results.save(out_dir / "results.json")

    from policygate.harness.types import ToolKind

    write_tools = frozenset(s.name for s in catalogue_of(sessions) if s.kind is ToolKind.WRITE)
    # tau2 builds an agent per attempt, not per simulation: a retried simulation
    # leaves behind agents that were constructed and never asked for anything.
    # Counting those would dilute every rate and, worse, make an empty chain look
    # like a broken one.
    used = [a for a in sessions if a.proposals > 0]
    unused = len(sessions) - len(used)
    per_session = [summarise(a, write_tools) for a in used]

    rewards = [s.reward_info.reward for s in results.simulations if s.reward_info]
    tau2_agent_cost = sum(s.agent_cost or 0.0 for s in results.simulations)
    tau2_user_cost = sum(s.user_cost or 0.0 for s in results.simulations)
    harness_llm_cost = sum(a.llm_cost for a in sessions)

    report = {
        "arm": args.arm,
        "simulations": len(results.simulations),
        "agents_built": len(sessions),
        "agents_never_used": unused,
        "wallclock_seconds": elapsed,
        "reward_mean": (sum(rewards) / len(rewards)) if rewards else None,
        "rewards": rewards,
        "termination_reasons": dict(
            Counter(str(s.termination_reason) for s in results.simulations)
        ),
        "cost": {
            # tau2 only sees the trajectory it was handed, and refused proposals
            # never enter it -- so its agent_cost is a floor for this arm, not the
            # bill. The difference is what the harness spent on retries.
            "tau2_agent_cost": tau2_agent_cost,
            "tau2_user_cost": tau2_user_cost,
            "agent_llm_cost_including_refused": harness_llm_cost,
            "hidden_retry_cost": harness_llm_cost - tau2_agent_cost,
            "note": "litellm prices at OpenAI list rates, not the AIHubMix bill",
        },
        "denials_by_rule": dict(
            sum((Counter(s["denials_by_rule"]) for s in per_session), Counter())
        ),
        "chains_intact": all(s["chain_intact"] for s in per_session),
        "infrastructure_errors": [
            str((sim.info or {}).get("error"))[:300]
            for sim in results.simulations
            if sim.info and sim.info.get("error")
        ],
        "sessions": per_session,
    }
    (out_dir / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))

    # The chain is contribution 3 and lives only in the agent's state, which tau2
    # does not persist. Without this the evidence the run produced would be
    # summarised and then thrown away.
    with (out_dir / "chains.jsonl").open("w") as fh:
        for agent in used:
            fh.write(
                json.dumps(
                    {
                        "task_id": agent.task_id,
                        "session_id": agent.session_id,
                        "records": [r.model_dump(mode="json") for r in chain_of(agent)],
                    },
                    default=str,
                )
                + "\n"
            )

    print(f"\ndone in {elapsed / 60:.1f} min -> {out_dir}")
    print(json.dumps({k: v for k, v in report.items() if k != "sessions"}, indent=2, default=str))
    return 0


def catalogue_of(sessions: list):
    return sessions[0].gate.catalogue.values() if sessions else ()


_CATALOGUE: list = []


def catalogue_of_config():
    """The tool catalogue, available before any agent has been built."""
    return _CATALOGUE


if __name__ == "__main__":
    raise SystemExit(main())
