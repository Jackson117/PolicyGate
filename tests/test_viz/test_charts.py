"""The chart rules, as assertions: one y-axis, colour follows the entity,
intervals are drawn, the headline leads, and text never wears a series colour."""

import pytest

from policygate.eval.passk import Estimate, Observation, ObservationTable, compare_paired
from policygate.viz import charts
from policygate.viz.theme import DARK, LIGHT, ink_on, tokens


def est(value, lo=None, hi=None, n=10):
    return Estimate(statistic="pass^1", value=value, lo=value - 0.1 if lo is None else lo,
                    hi=value + 0.1 if hi is None else hi, level=0.95, n_tasks=n, resamples=10)


def comparison(treatment_wins: int, n: int = 20):
    def table(passes):
        return ObservationTable(observations=tuple(
            Observation(task_id=str(i), trial=0, value=v) for i, v in enumerate(passes)))
    control = [False] * n
    treatment = [i < treatment_wins for i in range(n)]
    return compare_paired(table(treatment), table(control), resamples=200)


def y_axes(fig):
    return [k for k in fig.layout.to_plotly_json() if k.startswith("yaxis")]


def test_tokens_default_to_light_when_the_theme_is_unknown():
    assert tokens(None) is LIGHT
    assert tokens("dark") is DARK


def test_colour_follows_the_entity_not_the_position():
    alone = charts.pass_k_curve({"full": [est(0.5)]}, LIGHT)
    both = charts.pass_k_curve({"baseline": [est(0.4)], "full": [est(0.5)]}, LIGHT)
    colour = {tr.name: tr.line.color for tr in both.data if tr.mode == "lines+markers"}
    assert colour == {"baseline": LIGHT.baseline, "full": LIGHT.full}
    assert next(tr for tr in alone.data if tr.name == "full").line.color == LIGHT.full


def test_pass_k_curve_has_one_axis_a_legend_and_drawn_intervals():
    fig = charts.pass_k_curve({"baseline": [est(0.5), est(0.3)], "full": [est(0.6), est(0.4)]},
                              DARK)
    assert y_axes(fig) == ["yaxis"]
    assert sum(1 for tr in fig.data if tr.showlegend is not False) == 2
    washes = [tr for tr in fig.data if tr.fill == "toself"]
    assert len(washes) == 2 and all("0.1)" in tr.fillcolor for tr in washes)
    assert fig.layout.plot_bgcolor == DARK.surface


def test_labels_wear_text_ink_not_series_colour():
    fig = charts.pass_k_curve({"full": [est(0.5)]}, LIGHT)
    assert all(a.font.color == LIGHT.text_secondary for a in fig.layout.annotations)


def test_the_forest_leads_with_the_headline_and_draws_asymmetric_intervals():
    rows = [("only_schema_check_enabled", comparison(1)), ("full", comparison(6))]
    fig = charts.attribution_forest(rows, LIGHT)
    trace = fig.data[0]
    assert list(trace.y)[-1] == "full"  # plotted bottom-up, so last is on top
    assert y_axes(fig) == ["yaxis"]
    full = rows[1][1].difference
    assert trace.error_x.array[-1] == pytest.approx(full.hi - full.value)
    assert trace.error_x.arrayminus[-1] == pytest.approx(full.value - full.lo)
    assert any(s.x0 == 0 for s in fig.layout.shapes)


def test_the_migration_matrix_labels_every_cell_with_legible_ink():
    counts = {("never", "always"): 3, ("always", "always"): 20, ("sometimes", "never"): 1}
    fig = charts.migration_matrix(counts, LIGHT)
    assert fig.data[0].z[0][0] == 20  # always → always
    assert fig.data[0].z[2][0] == 3   # never → always
    assert len(fig.layout.annotations) == 9
    assert ink_on("#0d366b", LIGHT) == "#ffffff"
    assert ink_on("#cde2fb", LIGHT) == "#0b0b0b"


def test_the_frontier_names_only_the_headline_arms():
    points = [("baseline", 0.012, est(0.5)), ("full", 0.015, est(0.6)),
              ("only_gating_enabled", 0.013, est(0.55))]
    fig = charts.cost_frontier(points, LIGHT)
    assert {tr.name for tr in fig.data if tr.showlegend} == {"baseline", "full"}
    muted = next(tr for tr in fig.data if tr.name == "only_gating_enabled")
    assert muted.marker.color == LIGHT.muted
    assert y_axes(fig) == ["yaxis"]


def test_rule_bars_put_the_longest_on_top():
    fig = charts.rule_bars({"a": 1, "b": 9, "c": 4}, LIGHT, title="x")
    assert list(fig.data[0].y) == ["a", "c", "b"]
