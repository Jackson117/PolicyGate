"""The banking retrieval study (policygate-rag), read from its published results.

policygate-rag runs this harness on tau2's banking_knowledge domain and varies only
retrieval. It publishes every reported number as JSON in its ``results/``
directory; this module reads those files and nothing else -- no runs, no labels,
no import of that project's code -- so the dashboard shows exactly what its
README reports. No Streamlit here: loading and figures are testable without it.

Colour: the study's measures, not this project's arms, so slot 1 (``Tokens.full``)
marks the measure a chart is about and slot 2 its comparison. Partial arms are
drawn hollow and labelled, never by colour alone.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from policygate.viz.charts import _layout
from policygate.viz.theme import Tokens, rgba

SCHEMA_VERSION = 1
FILES = ("index.json", "stage2_arms.json", "stage1_judges.json", "replay_a3.json",
         "spend.json")
ARM_ORDER = ("A0", "A1", "A2", "A3", "A5", "A4")
"""Floor first, ceiling last: the order of the decomposition."""

DEFAULT_RESULTS = Path(os.environ.get("POLICYGATE_RAG_RESULTS", "../policygate-rag/results"))


class ResultsError(ValueError):
    """The directory is not a policygate-rag results directory this page can read."""


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    n_tasks: int

    @classmethod
    def of(cls, raw: Mapping[str, Any]) -> Interval:
        return cls(raw["estimate"], raw["low"], raw["high"], raw["n_tasks"])


@dataclass(frozen=True)
class RagResults:
    index: dict
    arms: dict
    stage1: dict
    replay: dict | None
    spend: dict

    def arm_ids(self, *, with_pass1: bool = True) -> list[str]:
        ids = [a for a in ARM_ORDER if a in self.arms["arms"]]
        ids += sorted(a for a in self.arms["arms"] if a not in ARM_ORDER)
        if with_pass1:
            ids = [a for a in ids if "pass^1" in self.arms["arms"][a].get("pass_k", {})]
        return ids

    def arm(self, arm_id: str) -> dict:
        return self.arms["arms"][arm_id]

    def pass1(self, arm_id: str) -> Interval:
        return Interval.of(self.arm(arm_id)["pass_k"]["pass^1"])

    def partial(self, arm_id: str) -> bool:
        return self.arm(arm_id)["status"] != "complete"

    def paired(self, a: str, b: str) -> tuple[str, dict] | None:
        """The pass^k differences for a pair, oriented ``a - b`` (flipped if stored ``b - a``)."""
        stored = self.arms.get("paired", {})
        if f"{a} - {b}" in stored:
            return f"{a} - {b}", stored[f"{a} - {b}"]
        if f"{b} - {a}" in stored:
            flipped = {k: {"estimate": -v["estimate"], "low": -v["high"], "high": -v["low"],
                           "n_tasks": v["n_tasks"]} for k, v in stored[f"{b} - {a}"].items()}
            return f"{a} - {b}", flipped
        return None


def load(directory: Path = DEFAULT_RESULTS) -> RagResults:
    """Read and check the study's results. Refuses a schema it does not know."""
    if not (directory / "index.json").exists():
        raise ResultsError(f"no policygate-rag results at {directory}")
    raw = {}
    for name in FILES:
        path = directory / name
        if not path.exists():
            if name == "replay_a3.json":
                raw[name] = None
                continue
            raise ResultsError(f"{path} is missing")
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ResultsError(f"{path}: schema {payload.get('schema_version')}, "
                               f"this page reads {SCHEMA_VERSION}")
        raw[name] = payload
    return RagResults(index=raw["index.json"], arms=raw["stage2_arms.json"],
                      stage1=raw["stage1_judges.json"], replay=raw["replay_a3.json"],
                      spend=raw["spend.json"])


def arm_rows(r: RagResults) -> list[dict]:
    """One row per arm for the summary table; numbers as strings with their interval."""
    def fmt(raw):
        i = Interval.of(raw)
        return f"{i.estimate:.3f} [{i.low:.3f}, {i.high:.3f}]"

    rows = []
    for a in r.arm_ids(with_pass1=False):
        e = r.arm(a)
        pk, ret, cost = e.get("pass_k", {}), e.get("retrieval") or {}, e.get("cost") or {}
        rows.append({
            "arm": a, "label": e["label"], "status": e["status"],
            "episodes": e["episodes"],
            "pass^1": fmt(pk["pass^1"]) if "pass^1" in pk else "–",
            "pass^3": fmt(pk["pass^3"]) if "pass^3" in pk else "–",
            "coverage (episode)": fmt(ret["coverage"]) if "coverage" in ret else "–",
            "coverage (at tool call)": (fmt(ret["action_coverage"])
                                        if "action_coverage" in ret else "–"),
            "USD / episode": f"{cost['usd_mean']:.4f}" if cost else "–",
            "judge USD (own account)": f"{cost.get('external_usd', 0.0):.3f}" if cost else "–",
        })
    return rows


def billed_elsewhere(spend: Mapping[str, Any]) -> float:
    """Spend outside the model provider's account (judges), summed over every entry."""
    def total(value: Any) -> float:
        if isinstance(value, Mapping):
            return sum(total(v) for v in value.values())
        return float(value)
    return total(spend.get("billed_elsewhere_usd", {}))


# --- figures ---------------------------------------------------------------------

def _intervals(fig: go.Figure, labels: Sequence[str], intervals: Sequence[Interval], *,
               color: str, t: Tokens, name: str, hollow: Sequence[bool] | None = None,
               offset: float = 0.0, showlegend: bool = False) -> None:
    hollow = hollow or [False] * len(labels)
    fig.add_trace(go.Scatter(
        x=[i.estimate for i in intervals],
        y=[k + offset for k in range(len(labels))],
        mode="markers", name=name, showlegend=showlegend,
        marker=dict(size=10, color=[t.surface if h else color for h in hollow],
                    line=dict(color=color, width=2)),
        error_x=dict(type="data", symmetric=False,
                     array=[i.high - i.estimate for i in intervals],
                     arrayminus=[i.estimate - i.low for i in intervals],
                     color=rgba(color, 0.75), thickness=2, width=0),
        customdata=[[lab, i.low, i.high, i.n_tasks] for lab, i in
                    zip(labels, intervals, strict=True)],
        hovertemplate=(f"<b>%{{customdata[0]}}</b><br>{name}: %{{x:.3f}}<br>95% CI "
                       "[%{customdata[1]:.3f}, %{customdata[2]:.3f}] · %{customdata[3]} tasks"
                       "<extra></extra>"),
    ))


def _rows_axis(fig: go.Figure, labels: Sequence[str]) -> None:
    fig.update_yaxes(tickvals=list(range(len(labels))), ticktext=list(labels),
                     autorange="reversed", showgrid=False)


def pass1_forest(r: RagResults, t: Tokens) -> go.Figure:
    ids = r.arm_ids()
    labels = [f"{a} · {r.arm(a)['label']}" + (" (partial)" if r.partial(a) else "")
              for a in ids]
    fig = go.Figure()
    _intervals(fig, labels, [r.pass1(a) for a in ids], color=t.full, t=t, name="pass^1",
               hollow=[r.partial(a) for a in ids])
    fig = _layout(fig, t, title="Task success by arm (pass^1, 95% CI)",
                  height=max(280, 46 * len(ids) + 110))
    _rows_axis(fig, labels)
    fig.update_xaxes(title_text="pass^1", range=[0, max(0.5, max(
        r.pass1(a).high for a in ids) + 0.05)])
    return fig


def coverage_dumbbell(r: RagResults, t: Tokens) -> go.Figure:
    """Episode-union coverage against coverage at each proposed tool call, per arm."""
    ids = [a for a in r.arm_ids(with_pass1=False)
           if "action_coverage" in (r.arm(a).get("retrieval") or {})]
    labels = [f"{a} · {r.arm(a)['label']}" + (" (partial)" if r.partial(a) else "")
              for a in ids]
    fig = go.Figure()
    _intervals(fig, labels, [Interval.of(r.arm(a)["retrieval"]["action_coverage"])
                             for a in ids],
               color=t.full, t=t, name="at each proposed tool call", offset=-0.12,
               showlegend=True)
    _intervals(fig, labels, [Interval.of(r.arm(a)["retrieval"]["coverage"]) for a in ids],
               color=t.baseline, t=t, name="any time in the episode", offset=0.12,
               showlegend=True)
    fig = _layout(fig, t, title="Required documents in the model's context",
                  height=max(260, 70 * len(ids) + 110))
    _rows_axis(fig, labels)
    fig.update_xaxes(title_text="share of the task's required documents", range=[0, 1])
    return fig


def replay_forest(r: RagResults, t: Tokens) -> go.Figure | None:
    if r.replay is None:
        return None
    names = list(r.replay["policies"])
    fig = go.Figure()
    _intervals(fig, names, [Interval.of(r.replay["policies"][n]["mean_action_coverage"])
                            for n in names],
               color=t.full, t=t, name="coverage at tool calls")
    fig = _layout(fig, t, title="Replay of A3's conversations under other context policies",
                  height=max(300, 34 * len(names) + 110))
    _rows_axis(fig, names)
    fig.update_xaxes(title_text="required documents in context at each proposed tool call",
                     range=[0, 0.7])
    return fig


JUDGE_METRICS = {"mrr": "MRR", "coverage_at_k": "Coverage@5", "precision_at_k": "Precision@5"}


def judge_forest(r: RagResults, t: Tokens, metric: str = "mrr") -> go.Figure:
    common = r.stage1["common"]
    names = sorted(common["judges"], key=lambda j: -common["judges"][j][metric]["estimate"])
    labels = [r.stage1["judges"][j]["label"] for j in names]
    fig = go.Figure()
    _intervals(fig, labels, [Interval.of(common["judges"][j][metric]) for j in names],
               color=t.full, t=t, name=JUDGE_METRICS[metric])
    fig = _layout(fig, t, title=f"Stage 1 judges: {JUDGE_METRICS[metric]} on "
                                f"{common['queries']} queries ({common['tasks']} tasks)",
                  height=max(240, 50 * len(names) + 110))
    _rows_axis(fig, labels)
    fig.update_xaxes(title_text=JUDGE_METRICS[metric], range=[0, 1])
    return fig
