#!/usr/bin/env python
"""Where do a task's succeeding and failing runs stop agreeing?

For every unstable task (one that neither always passes nor always fails), this
pairs its first failing trial against its first succeeding trial and draws their
tool-call sequences as one shared trunk that forks at the first step where the
two disagree.

Reading the plot: a long grey trunk means both runs did the same thing for a
while and one wrong turn late decided the outcome. A fork at step 1 means the
runs diverged almost immediately.

Usage:
    .venv/bin/python scripts/w1_plot_divergence.py experiments/<run>/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TRUNK = "#8d939b"   # steps both runs agree on
OK = "#2a9d8f"      # the trial that passed
FAIL = "#e05c4a"    # the trial that failed
FORK_OFFSET = 0.21  # vertical gap between the two branches

# Marker at the fork, by what kind of disagreement it was.
FORK_MARKER = {
    "diff-tool": "X",             # called a different tool
    "same-tool-diff-args": "D",   # same tool, different arguments
    "ran-out": "s",               # one run simply stopped here
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--out", type=Path, default=None, help="Default: <run dir>/figures/divergence.png")
    p.add_argument("--label-chars", type=int, default=22, help="Truncate tool names in labels.")
    return p.parse_args()


def tool_calls(sim) -> list[tuple[str, str]]:
    """(tool_name, canonical_args) in call order. Arguments are sorted so that
    two calls compare equal only when they really are the same call."""
    out = []
    for m in sim.messages or []:
        if getattr(m, "role", None) == "assistant":
            for tc in m.tool_calls or []:
                out.append((tc.name, json.dumps(tc.arguments, sort_keys=True)))
    return out


def build_pairs(results):
    """One (task, failing sim, passing sim) per unstable task, lowest trial first
    on each side so the figure is reproducible."""
    by_task: dict[str, dict[int, object]] = {}
    for s in results.simulations:
        if s.reward_info is None or s.reward_info.reward is None:
            continue
        by_task.setdefault(s.task_id, {})[s.trial] = s

    pairs = []
    for task_id, trials in by_task.items():
        passed = [s for _, s in sorted(trials.items()) if s.reward_info.reward >= 1.0 - 1e-9]
        failed = [s for _, s in sorted(trials.items()) if s.reward_info.reward < 1.0 - 1e-9]
        if not passed or not failed:
            continue  # always-pass and always-fail tasks have nothing to compare
        pairs.append((task_id, failed[0], passed[0]))
    return pairs


def describe(fail_calls, ok_calls) -> tuple[int, str, str]:
    """(fork index, kind, label) for one pair."""
    n = min(len(fail_calls), len(ok_calls))
    i = next((j for j in range(n) if fail_calls[j] != ok_calls[j]), n)
    if i == n:
        # No disagreement inside the common length: one run just kept going.
        return i, "ran-out", ""
    if fail_calls[i][0] == ok_calls[i][0]:
        return i, "same-tool-diff-args", fail_calls[i][0]
    return i, "diff-tool", fail_calls[i][0]


def main() -> int:
    args = parse_args()

    from tau2.data_model.simulation import Results

    results = Results.load(args.results)
    pairs = build_pairs(results)
    if not pairs:
        print("no unstable tasks in this run -- nothing to plot")
        return 1

    rows = []
    for task_id, fail_sim, ok_sim in pairs:
        f, o = tool_calls(fail_sim), tool_calls(ok_sim)
        fork, kind, label = describe(f, o)
        rows.append({"task": task_id, "fail": f, "ok": o, "fork": fork, "kind": kind, "label": label})
    # Shortest agreement first: the figure reads as a staircase.
    rows.sort(key=lambda r: (r["fork"], -max(len(r["fail"]), len(r["ok"]))))

    max_len = max(max(len(r["fail"]), len(r["ok"])) for r in rows)
    fig, (ax, hx) = plt.subplots(
        2, 1, figsize=(11.5, 0.52 * len(rows) + 2.6), sharex=True,
        gridspec_kw={"height_ratios": [len(rows), 2.4], "hspace": 0.06},
    )

    for y, r in enumerate(rows):
        fork = r["fork"]
        # Shared trunk: the steps both runs took identically.
        if fork > 0:
            ax.plot(range(fork), [y] * fork, "-", color=TRUNK, lw=2.2, zorder=2)
            ax.plot(range(fork), [y] * fork, "o", color=TRUNK, ms=4.5, zorder=3)
        stem = fork - 1 if fork > 0 else -0.55

        for calls, sign, color in ((r["ok"], +1, OK), (r["fail"], -1, FAIL)):
            if fork >= len(calls):
                # This run ended before the other one diverged.
                ax.plot([stem, stem + 0.45], [y, y + sign * FORK_OFFSET], "-", color=color, lw=1.4, alpha=0.55, zorder=2)
                ax.plot([stem + 0.45], [y + sign * FORK_OFFSET], "|", color=color, ms=9, mew=2, zorder=4)
                continue
            yb = y + sign * FORK_OFFSET
            xs = list(range(fork, len(calls)))
            ax.plot([stem, xs[0]], [y, yb], "-", color=color, lw=1.8, zorder=2)   # the fork itself
            ax.plot(xs, [yb] * len(xs), "-", color=color, lw=1.8, zorder=2)
            ax.plot(xs, [yb] * len(xs), "o", color=color, ms=4.5, zorder=3)
            ax.plot([xs[-1]], [yb], "|", color=color, ms=9, mew=2, zorder=4)      # end of run

        ax.plot([fork], [y], FORK_MARKER[r["kind"]], color="#1c1c1c", ms=6.5 if r["kind"] == "ran-out" else 7.5,
                mew=1.6, fillstyle="none", zorder=5)
        if r["label"]:
            ax.annotate(r["label"][: args.label_chars], (fork, y), xytext=(0, 9.5),
                        textcoords="offset points", ha="center", fontsize=6.6, color="#4a4a4a")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["task"] for r in rows], fontsize=8)
    ax.set_ylabel("task", fontsize=9)
    ax.set_ylim(-0.9, len(rows) - 0.1)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e6e6e6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    counts = [0] * (max_len + 1)
    for r in rows:
        counts[r["fork"]] += 1
    hx.bar(range(max_len + 1), counts, color="#c3c8ce", width=0.55, zorder=2)
    hx.set_xlabel("tool call step", fontsize=9)
    hx.set_ylabel("tasks\nforking here", fontsize=8)
    hx.set_xticks(range(max_len + 1))
    hx.tick_params(labelsize=8)
    hx.set_yticks(range(0, max(counts) + 1, max(1, max(counts) // 3)))
    hx.grid(axis="x", color="#e6e6e6", lw=0.8, zorder=0)
    hx.set_axisbelow(True)
    for side in ("top", "right", "left"):
        hx.spines[side].set_visible(False)

    fig.legend(
        handles=[
            Line2D([], [], color=TRUNK, lw=2.2, marker="o", ms=4.5, label="both runs agree"),
            Line2D([], [], color=OK, lw=1.8, marker="o", ms=4.5, label="passing trial"),
            Line2D([], [], color=FAIL, lw=1.8, marker="o", ms=4.5, label="failing trial"),
            Line2D([], [], color="#1c1c1c", lw=0, marker="X", ms=7.5, mew=1.6, fillstyle="none", label="forked on a different tool"),
            Line2D([], [], color="#1c1c1c", lw=0, marker="D", ms=7.5, mew=1.6, fillstyle="none", label="same tool, different arguments"),
            Line2D([], [], color="#1c1c1c", lw=0, marker="s", ms=6.5, mew=1.6, fillstyle="none", label="one run stopped early"),
            Line2D([], [], color="#1c1c1c", lw=0, marker="|", ms=9, mew=2, label="end of run"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.012), fontsize=8, frameon=False, ncol=3,
    )
    ax.set_title(
        f"{len(rows)} unstable tasks: how far the passing and failing runs agree before forking\n"
        f"label = the tool the failing run reached for at the fork",
        fontsize=10, loc="left", pad=10,
    )

    out = args.out or args.results.parent / "figures" / "divergence.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"{len(rows)} unstable tasks -> {out}")
    for r in rows:
        print(f"  task {r['task']:>3}  fork@{r['fork']:<2} {r['kind']:<20} {r['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
