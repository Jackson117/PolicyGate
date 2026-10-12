#!/usr/bin/env python
"""Prove a harness change leaves retail behaviour unchanged, byte for byte.

The reported results were produced by the code at ``7bb16c1``. A change to
``harness/`` or ``adapters/`` -- a refactor, a new role for another domain -- must
not move them. Unit tests check what someone thought to assert; this checks
everything at once, by running the reference code and the working tree through
the same sessions and diffing the transcripts.

Each side runs in its own process with its own ``policygate`` on ``sys.path``,
and the same tau2 checkout. Everything that would vary is fixed:

* the model is a scripted stub, seeded per session, proposing tool calls drawn
  from the proposals recorded in the W4 audit chains (refused ones included),
  the task's reference actions, bare calls to every tool, and unknown tools;
* the real retail environment executes whatever the harness lets through;
* ``uuid4`` is a counter, the harness clock is fixed, and tau2's wall-clock
  message timestamps are removed before comparison.

Configs cover every arm shape the ablation ran (baseline, full, each ``only_*``)
plus the approval gate under all four deciders and with gating off.

What is compared: every model call (which tool schemas were offered, the last
message seen), every message returned to tau2, the agent's full message
history, every audit record, every state transition, and the final facts, known
world and pending execution.

Limits. The worker drives ``HarnessAgent`` and reads ``state.tracker``,
``state.chain``, ``state.world`` and ``state.pending_execution``; a change to that
surface needs the worker updated on both sides. The proposal pool comes from
``experiments/**/sessions.jsonl``, which is git-ignored, so this runs on a
machine that holds the W4 runs, not in CI. When it was first used (v0.2.0), two
runs of the same code were identical and a one-word change to the refusal text
was detected.

Usage, from the repo root:

    python scripts/harness_diff.py                    # working tree vs 7bb16c1
    python scripts/harness_diff.py --ref v0.2.0 --sessions 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REFERENCE = "7bb16c1"
"""The commit that produced the reported W4 results."""


# --- orchestration ---------------------------------------------------------


def _first_difference(old: list[dict], new: list[dict]) -> str:
    if len(old) != len(new):
        return f"session count differs: {len(old)} vs {len(new)}"
    for i, (a, b) in enumerate(zip(old, new, strict=True)):
        if a == b:
            continue
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                left, right = a.get(key), b.get(key)
                if isinstance(left, list) and isinstance(right, list):
                    for j, (x, y) in enumerate(zip(left, right, strict=False)):
                        if x != y:
                            return (
                                f"session {i} ({a['config']}, task {a['task']}), {key}[{j}]:\n"
                                f"  ref:  {json.dumps(x)[:600]}\n  tree: {json.dumps(y)[:600]}"
                            )
                    return f"session {i}, {key}: lengths {len(left)} vs {len(right)}"
                return (
                    f"session {i} ({a['config']}, task {a['task']}), {key}:\n"
                    f"  ref:  {json.dumps(left)[:600]}\n  tree: {json.dumps(right)[:600]}"
                )
    return "transcripts differ outside the sessions"


def compare(ref: str, sessions: int) -> int:
    with tempfile.TemporaryDirectory(prefix="harness_diff_") as tmp:
        tmp_path = Path(tmp)
        worktree = tmp_path / "ref"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), ref],
            cwd=REPO, check=True, capture_output=True,
        )
        try:
            sides = {"ref": worktree / "src", "tree": REPO / "src"}
            procs = {
                name: subprocess.Popen(
                    [sys.executable, __file__, "--worker", str(src),
                     str(tmp_path / f"{name}.json"), str(sessions)],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                )
                for name, src in sides.items()
            }
            for name, proc in procs.items():
                out, _ = proc.communicate()
                if proc.returncode != 0:
                    print(f"{name} worker failed (exit {proc.returncode})", file=sys.stderr)
                    return 2
                print(f"{name:4s} {out.strip()}")

            old = (tmp_path / "ref.json").read_bytes()
            new = (tmp_path / "tree.json").read_bytes()
            if old == new:
                print(f"IDENTICAL: working tree reproduces {ref} byte for byte")
                return 0
            print(f"DIFFERENT from {ref}. First difference:", file=sys.stderr)
            print(_first_difference(json.loads(old), json.loads(new)), file=sys.stderr)
            return 1
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=REPO, check=False, capture_output=True,
            )


# --- the worker: one side of the comparison --------------------------------


def worker(src: str, out_path: str, sessions: int) -> None:
    sys.path.insert(0, src)

    import glob
    import random
    import uuid
    from datetime import datetime

    import policygate

    loaded = Path(policygate.__file__).resolve()
    if not loaded.is_relative_to(Path(src).resolve()):
        raise SystemExit(f"loaded policygate from {loaded}, not from {src}")

    from tau2.data_model.message import AssistantMessage, MultiToolMessage, UserMessage
    from tau2.data_model.message import ToolCall as Tau2ToolCall
    from tau2.registry import registry
    from tau2.utils.utils import DATA_DIR

    from policygate.adapters import tau2_agent as agent_module
    from policygate.adapters.tau2_agent import HarnessAgent
    from policygate.adapters.tool_mapping import (
        RETAIL_ESCALATION,
        RETAIL_IDENTITY_LOOKUPS,
        ground_truth,
        retail_constraints,
        retail_facts,
        tool_specs,
    )
    from policygate.harness.approval import make_decider
    from policygate.harness.config import ApprovalConfig, ApprovalPolicy, HarnessConfig

    def clock() -> datetime:
        return datetime(2026, 1, 1, 12, 0, 0)

    counter = [0]

    def fake_uuid4() -> uuid.UUID:
        counter[0] += 1
        return uuid.UUID(int=counter[0])

    uuid.uuid4 = fake_uuid4

    # Proposals recorded in the W4 audit chains, deduplicated and ordered.
    seen: set[str] = set()
    pool: list[tuple[str, dict]] = []
    for path in sorted(glob.glob(str(REPO / "experiments/**/sessions.jsonl"), recursive=True)):
        with open(path) as f:
            for line in f:
                for record in json.loads(line).get("records", []):
                    tc = record.get("tool_call")
                    if not tc:
                        continue
                    key = json.dumps([tc["name"], tc["arguments"]], sort_keys=True)
                    if key not in seen:
                        seen.add(key)
                        pool.append((tc["name"], tc["arguments"]))
    if not pool:
        raise SystemExit("no recorded proposals under experiments/; this needs the W4 runs")
    pool.sort(key=lambda p: json.dumps(p, sort_keys=True))

    db = json.loads((DATA_DIR / "tau2/domains/retail/db.json").read_text())
    probe = registry.get_env_constructor("retail")()
    constraints = retail_constraints(db, tools=probe.get_tools())
    catalogue = tool_specs(
        probe.tools, identity_lookups=RETAIL_IDENTITY_LOOKUPS, escalation=RETAIL_ESCALATION
    )
    tool_names = sorted(s.name for s in catalogue)
    tasks = sorted(registry.get_tasks_loader("retail")(), key=lambda t: t.id)

    gated = {"approval_gate_enabled": True, "audit_enabled": True}
    full, baseline = HarnessConfig.full(), HarnessConfig.baseline()
    configs = [baseline, full]
    configs += [
        HarnessConfig.only(s, audit_enabled=True) for s in full.enabled_ablation_switches()
    ]
    configs += [
        full.model_copy(update=gated | {"name": f"full_gate_{p.value}",
                                        "approval": ApprovalConfig(policy=p)})
        for p in (ApprovalPolicy.AUTO_APPROVE, ApprovalPolicy.AUTO_REJECT, ApprovalPolicy.ORACLE)
    ]
    configs += [
        full.model_copy(update=gated | {
            "name": "full_gate_noisy",
            "approval": ApprovalConfig(policy=ApprovalPolicy.NOISY_ORACLE, noise_rate=0.3),
        }),
        # The gate without gating: writes reach it from any state.
        baseline.model_copy(update=gated | {
            "name": "baseline_gate_reject",
            "approval": ApprovalConfig(policy=ApprovalPolicy.AUTO_REJECT),
        }),
        baseline.model_copy(update=gated | {"name": "baseline_gate_approve"}),
    ]

    def strip_wallclock(x):
        # tau2 stamps messages with datetime.now(); the harness's records use clock().
        if isinstance(x, dict):
            return {
                k: strip_wallclock(v) for k, v in x.items()
                if not (k == "timestamp" and isinstance(v, str) and not v.startswith("2026-01-01"))
            }
        if isinstance(x, list):
            return [strip_wallclock(v) for v in x]
        return x

    def dump(obj):
        if hasattr(obj, "model_dump"):
            return strip_wallclock(obj.model_dump(mode="json"))
        return repr(obj)

    def run_session(cfg, seed: int) -> dict:
        rng = random.Random(seed)
        task = tasks[seed % len(tasks)]
        truth = ground_truth(task)
        task_pool = [(a.name, dict(a.arguments)) for a in truth.actions]
        env = registry.get_env_constructor("retail")()
        if task.initial_state is not None:
            try:
                env.set_state(
                    initialization_data=task.initial_state.initialization_data,
                    initialization_actions=task.initial_state.initialization_actions,
                    message_history=[],
                )
            except Exception:
                pass  # identical on both sides; the session runs from the default state
        log: list = []
        call_ids = [0]

        def scripted(model, tools, messages, call_name, **kwargs):
            log.append({"generate": {
                "offered": sorted(t.name for t in tools),
                "n_messages": len(messages),
                "last": dump(messages[-1]),
            }})
            if rng.random() < 0.15:
                return AssistantMessage(
                    role="assistant", content=f"text {rng.random():.6f}", cost=0.001
                )
            calls = []
            for _ in range(1 if rng.random() < 0.8 else 2):
                call_ids[0] += 1
                q = rng.random()
                if q < 0.35 and task_pool:
                    name, args = rng.choice(task_pool)
                elif q < 0.9:
                    name, args = rng.choice(pool)
                elif q < 0.97:
                    name, args = rng.choice(tool_names), {}
                else:
                    name, args = "nonexistent_tool", {"x": 1}
                calls.append(Tau2ToolCall(id=f"c{call_ids[0]}", name=name, arguments=dict(args)))
            return AssistantMessage(role="assistant", content=None, tool_calls=calls, cost=0.001)

        agent_module.generate = scripted
        agent = HarnessAgent(
            tools=env.get_tools(), domain_policy=env.get_policy(), llm="scripted",
            catalogue=catalogue, constraints=constraints, extract_facts=retail_facts,
            config=cfg, approve=make_decider(cfg, truth=truth), clock=clock,
            session_id=f"s{seed}", max_internal_attempts=3,
        )
        state = agent.get_init_state()
        msg = UserMessage(role="user", content="hello")
        for step in range(30):
            try:
                out, state = agent.generate_next_message(msg, state)
            except Exception as e:  # recorded: both sides must fail identically
                log.append({"exception": f"{type(e).__name__}: {e}"})
                break
            log.append({"out": dump(out)})
            if not out.tool_calls:
                msg = UserMessage(role="user", content=f"user turn {step}")
                continue
            try:
                results = [env.get_response(tc) for tc in out.tool_calls]
            except Exception as e:
                log.append({"env_exception": f"{type(e).__name__}: {e}"})
                break
            msg = (
                results[0] if len(results) == 1
                else MultiToolMessage(role="tool", tool_messages=results)
            )
        return {
            "config": cfg.name,
            "task": task.id,
            "log": log,
            "messages": [dump(m) for m in state.messages],
            "system": [dump(m) for m in state.system_messages],
            "records": [dump(r) for r in state.chain.records],
            "history": [dump(h) for h in state.tracker.history],
            "facts": dump(state.tracker.facts),
            "state": str(state.tracker.state),
            "world": {k: dump(v) for k, v in sorted(state.world.items())},
            "pending": dump(state.pending_execution) if state.pending_execution else None,
        }

    transcripts = [
        run_session(cfg, seed=1000 * i + k)
        for i, cfg in enumerate(configs)
        for k in range(sessions)
    ]
    Path(out_path).write_text(json.dumps(transcripts, indent=1, sort_keys=True))
    records = sum(len(t["records"]) for t in transcripts)
    print(
        f"{len(configs)} configs x {sessions} sessions, {records} audit records, "
        f"{len(pool)} recorded proposals"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", default=REFERENCE, help=f"reference commit (default {REFERENCE})")
    parser.add_argument("--sessions", type=int, default=10, help="sessions per config")
    parser.add_argument("--worker", nargs=3, metavar=("SRC", "OUT", "N"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        src, out, n = args.worker
        worker(src, out, int(n))
        return 0
    return compare(args.ref, args.sessions)


if __name__ == "__main__":
    sys.exit(main())
