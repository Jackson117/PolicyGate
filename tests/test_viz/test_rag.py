"""viz/rag.py: reading policygate-rag's published results, and the page's figures.

A small results directory written here stands in for the study's; the real one
lives in another repository. What matters: an unknown schema is refused, pairs
read the same in either order, partial arms are visible as partial, and the
chart rules hold (one y-axis, intervals drawn, text never in a series colour).
"""

import json

import pytest

from policygate.viz import rag
from policygate.viz.theme import LIGHT


def iv(est, lo, hi, n=30):
    return {"estimate": est, "low": lo, "high": hi, "n_tasks": n}


def arm(label, p1, *, status="complete", retrieval=None, p3=None):
    pk = {"pass^1": p1} | ({"pass^3": p3} if p3 else {})
    out = {"label": label, "runs": [], "episodes": 90 if status == "complete" else 72,
           "status": status, "pass_k": pk,
           "cost": {"usd_mean": 0.04, "external_usd": 0.5}}
    if retrieval:
        out["retrieval"] = retrieval
    if status != "complete":
        out["covers_all_tasks"] = True
    return out


@pytest.fixture
def results(tmp_path):
    ret = {"coverage": iv(0.43, 0.36, 0.50), "action_coverage": iv(0.14, 0.10, 0.18)}
    files = {
        "index.json": {"generated_at": "2026-10-21T00:00:00+00:00", "code_commit": "abc123",
                       "intervals": "95% bootstrap", "arms": {}},
        "stage2_arms.json": {
            "arms": {"A4": arm("ceiling", iv(0.22, 0.09, 0.38), p3=iv(0.2, 0.07, 0.37)),
                     "A3": arm("agentic", iv(0.02, 0.0, 0.06), retrieval=ret),
                     "A5": arm("working set", iv(0.06, 0.0, 0.13), status="partial",
                               retrieval=ret),
                     "A0": arm("floor", iv(0.02, 0.0, 0.07)),
                     "A9": {"label": "never run", "runs": [], "episodes": 0,
                            "status": "not run"}},
            "paired": {"A4 - A0": {"pass^1": iv(0.2, 0.056, 0.367)}},
        },
        "stage1_judges.json": {
            "judges": {"jev": {"label": "Jev"}, "none": {"label": "No judge"}},
            "common": {"queries": 203, "tasks": 30, "judges": {
                "jev": {"mrr": iv(0.55, 0.48, 0.63), "coverage_at_k": iv(0.14, 0.11, 0.17),
                        "precision_at_k": iv(0.26, 0.21, 0.31),
                        "calibration": {"brier": 0.1, "ece": 0.05, "pairs": 100}},
                "none": {"mrr": iv(0.37, 0.29, 0.44), "coverage_at_k": iv(0.09, 0.07, 0.11),
                         "precision_at_k": iv(0.16, 0.13, 0.21)}}},
        },
        "replay_a3.json": {"metric": "coverage at tool calls", "policies": {
            "replace (A3)": {"mean_action_coverage": iv(0.144, 0.105, 0.184)},
            "ws20 kept": {"mean_action_coverage": iv(0.333, 0.274, 0.396)}}},
        "spend.json": {"ceiling_usd": 33.8, "total_spent": 28.88,
                       "lines": {"stage2": {"allocated": 29.02, "spent": 27.89}},
                       "billed_elsewhere_usd": {"stage1_judges": {"jev": 0.43},
                                                "stage2_a3/A3": 0.75}},
    }
    for name, payload in files.items():
        (tmp_path / name).write_text(json.dumps({"schema_version": 1} | payload))
    return tmp_path


def test_a_directory_without_results_or_with_another_schema_is_refused(results, tmp_path):
    with pytest.raises(rag.ResultsError):
        rag.load(tmp_path / "nowhere")
    raw = json.loads((results / "spend.json").read_text())
    (results / "spend.json").write_text(json.dumps(raw | {"schema_version": 2}))
    with pytest.raises(rag.ResultsError, match="schema 2"):
        rag.load(results)


def test_arms_follow_the_decomposition_and_only_measured_ones_are_plotted(results):
    r = rag.load(results)
    assert r.arm_ids() == ["A0", "A3", "A5", "A4"]
    assert r.arm_ids(with_pass1=False) == ["A0", "A3", "A5", "A4", "A9"]
    assert r.partial("A5") and not r.partial("A3")


def test_a_pair_reads_the_same_in_either_order(results):
    r = rag.load(results)
    name, d = r.paired("A0", "A4")
    assert name == "A0 - A4"
    assert d["pass^1"] == {"estimate": -0.2, "low": -0.367, "high": -0.056, "n_tasks": 30}
    assert r.paired("A3", "A5") is None


def test_rows_mark_what_was_not_measured(results):
    rows = {row["arm"]: row for row in rag.arm_rows(rag.load(results))}
    assert rows["A9"]["pass^1"] == "–" and rows["A9"]["status"] == "not run"
    assert rows["A5"]["pass^3"] == "–"
    assert rows["A4"]["pass^3"] == "0.200 [0.070, 0.370]"


def test_judge_spend_sums_every_account(results):
    assert rag.billed_elsewhere(rag.load(results).spend) == pytest.approx(1.18)


@pytest.mark.parametrize("build", [
    lambda r: rag.pass1_forest(r, LIGHT),
    lambda r: rag.coverage_dumbbell(r, LIGHT),
    lambda r: rag.replay_forest(r, LIGHT),
    lambda r: rag.judge_forest(r, LIGHT, "coverage_at_k"),
])
def test_figures_have_one_axis_and_draw_every_interval(results, build):
    fig = build(rag.load(results))
    assert [k for k in fig.layout.to_plotly_json() if k.startswith("yaxis")] == ["yaxis"]
    for trace in fig.data:
        assert len(trace.error_x.array) == len(trace.x)
        assert all(v >= 0 for v in trace.error_x.array)


def test_a_partial_arm_is_hollow_and_labelled(results):
    fig = rag.pass1_forest(rag.load(results), LIGHT)
    [trace] = fig.data
    labels = list(fig.layout.yaxis.ticktext)
    i = next(k for k, lab in enumerate(labels) if lab.startswith("A5"))
    assert labels[i].endswith("(partial)")
    assert trace.marker.color[i] == LIGHT.surface
    assert trace.marker.line.color == LIGHT.full


def test_two_measures_get_two_slots_and_a_legend(results):
    fig = rag.coverage_dumbbell(rag.load(results), LIGHT)
    assert {tr.marker.line.color for tr in fig.data} == {LIGHT.full, LIGHT.baseline}
    assert all(tr.showlegend for tr in fig.data)


def test_without_a_replay_file_the_replay_figure_is_absent(results):
    (results / "replay_a3.json").unlink()
    assert rag.replay_forest(rag.load(results), LIGHT) is None
