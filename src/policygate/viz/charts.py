"""The dashboard's four figures, as plain plotly objects.

No Streamlit here: each function takes computed data and tokens and returns a
figure, so the chart rules are testable -- one y-axis, colour by entity, a
legend whenever there are two series, intervals drawn rather than implied.

The four figures:

* **pass^k curve** -- baseline and full across k, with bootstrap intervals.
* **attribution forest** -- each single-mechanism arm's paired difference from
  baseline, with its interval and zero marked. A heatmap in the original plan;
  with one domain and one model its second axis is empty, so the honest form is
  one row per arm.
* **migration matrix** -- how tasks moved between always / sometimes / never.
* **cost-reliability frontier** -- pass^1 against cost per simulation, one point
  per arm, cost including the refused proposals tau2 never saw.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import plotly.graph_objects as go

from policygate.eval.passk import Estimate, PairedComparison
from policygate.viz.theme import Tokens, ink_on, rgba

CATEGORIES = ("always", "sometimes", "never")


def _layout(fig: go.Figure, t: Tokens, *, title: str, height: int = 360) -> go.Figure:
    axis = dict(
        showgrid=True, gridcolor=t.grid, gridwidth=1, zeroline=False,
        linecolor=t.axis, linewidth=1, showline=True, ticks="outside", tickcolor=t.axis,
        tickfont=dict(color=t.muted), title_font=dict(color=t.text_secondary),
        # Category labels (rule ids, "sometimes") are wider than the fixed margin;
        # without automargin they are clipped or run into the axis title.
        automargin=True, title_standoff=12,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color=t.text_primary, size=15), x=0, xanchor="left"),
        font=dict(family=t.font, color=t.text_secondary, size=12),
        paper_bgcolor=t.surface,
        plot_bgcolor=t.surface,
        height=height,
        margin=dict(l=56, r=96, t=48, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    font=dict(color=t.text_secondary)),
        hoverlabel=dict(font=dict(family=t.font)),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def pass_k_curve(curves: Mapping[str, Sequence[Estimate]], t: Tokens) -> go.Figure:
    """pass^k by k for each arm, interval as a wash, value labelled at the line end."""
    fig = go.Figure()
    for arm, estimates in curves.items():
        if not estimates:
            continue
        color = t.arm_color(arm)
        ks = list(range(1, len(estimates) + 1))
        fig.add_trace(go.Scatter(
            x=ks + ks[::-1],
            y=[e.hi for e in estimates] + [e.lo for e in estimates][::-1],
            # mode must be explicit: plotly draws markers on short traces by default,
            # which paints the interval's corners in the colorway's next hues.
            mode="lines", fill="toself", fillcolor=rgba(color, 0.10), line=dict(width=0),
            hoverinfo="skip", showlegend=False, name=f"{arm} interval",
        ))
        fig.add_trace(go.Scatter(
            x=ks, y=[e.value for e in estimates], mode="lines+markers", name=arm,
            line=dict(color=color, width=2, shape="linear"),
            marker=dict(size=8, color=color, line=dict(color=t.surface, width=2)),
            customdata=[[e.lo, e.hi, e.n_tasks] for e in estimates],
            hovertemplate=(f"<b>{arm}</b><br>pass^%{{x}} = %{{y:.3f}}<br>"
                           "95% CI [%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                           "<br>%{customdata[2]} tasks<extra></extra>"),
        ))
        last = estimates[-1]
        fig.add_annotation(
            x=ks[-1], y=last.value, text=f"{arm} {last.value:.2f}", showarrow=False,
            xanchor="left", xshift=10, font=dict(color=t.text_secondary),
        )
    fig = _layout(fig, t, title="pass^k by arm")
    # Pin the range to the measured k: autorange pads to an empty k+1 tick.
    top = max((len(e) for e in curves.values()), default=1)
    fig.update_xaxes(title_text="k (trials that must all succeed)", dtick=1,
                     range=[0.7, top + 0.3])
    fig.update_yaxes(title_text="pass^k", range=[0, 1])
    return fig


def attribution_forest(
    rows: Sequence[tuple[str, PairedComparison]], t: Tokens, *, headline: str | None = "full"
) -> go.Figure:
    """Paired difference from baseline per arm. The headline arm, if present, leads."""
    ordered = sorted(rows, key=lambda r: (r[0] != headline, r[0]))
    labels = [name for name, _ in ordered][::-1]
    values = [c.difference.value for _, c in ordered][::-1]
    lo = [c.difference.lo for _, c in ordered][::-1]
    hi = [c.difference.hi for _, c in ordered][::-1]
    colors = [t.arm_color(name) if name == headline else t.full for name in labels]

    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color=t.axis, width=1))
    fig.add_trace(go.Scatter(
        x=values, y=labels, mode="markers", name="difference from baseline",
        marker=dict(size=10, color=colors, line=dict(color=t.surface, width=2)),
        error_x=dict(type="data", symmetric=False,
                     array=[h - v for h, v in zip(hi, values, strict=True)],
                     arrayminus=[v - lo_ for v, lo_ in zip(values, lo, strict=True)],
                     color=t.text_secondary, thickness=2, width=0),
        customdata=[[c.p_value, c.treatment_higher, c.control_higher, c.difference.n_tasks,
                     c.test.value]
                    for _, c in ordered][::-1],
        hovertemplate=("<b>%{y}</b><br>Δ = %{x:+.3f}<br>p = %{customdata[0]:.3g} "
                       "(%{customdata[4]})<br>higher: %{customdata[1]} tasks vs "
                       "%{customdata[2]}<br>%{customdata[3]} tasks compared<extra></extra>"),
        showlegend=False,
    ))
    fig = _layout(fig, t, title="Paired difference from baseline (pass^1, 95% CI)",
                  height=max(260, 44 * len(labels) + 100))
    fig.update_xaxes(title_text="treatment − baseline", zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


def migration_matrix(counts: Mapping[tuple[str, str], int], t: Tokens) -> go.Figure:
    """Tasks by (baseline class, treatment class). Counts labelled in their cells."""
    z = [[counts.get((row, col), 0) for col in CATEGORIES] for row in CATEGORIES]
    peak = max((v for r in z for v in r), default=0) or 1
    steps = t.sequential
    cell_colors = [[steps[min(len(steps) - 1, round(v / peak * (len(steps) - 1)))]
                    for v in r] for r in z]
    colorscale = [[i / (len(steps) - 1), c] for i, c in enumerate(steps)]

    fig = go.Figure(go.Heatmap(
        z=z, x=list(CATEGORIES), y=list(CATEGORIES), colorscale=colorscale,
        zmin=0, zmax=peak, xgap=2, ygap=2, showscale=False,
        hovertemplate="baseline %{y} → full %{x}<br>%{z} tasks<extra></extra>",
    ))
    for i, row in enumerate(CATEGORIES):
        for j, col in enumerate(CATEGORIES):
            fig.add_annotation(x=col, y=row, text=str(z[i][j]), showarrow=False,
                               font=dict(color=ink_on(cell_colors[i][j], t), size=14))
    fig = _layout(fig, t, title="Task migration, baseline → full", height=340)
    fig.update_xaxes(title_text="full", showgrid=False, side="bottom")
    fig.update_yaxes(title_text="baseline", showgrid=False, autorange="reversed")
    return fig


def cost_frontier(
    points: Sequence[tuple[str, float, Estimate]],
    t: Tokens,
    *,
    label: Sequence[str] = ("full", "baseline"),
) -> go.Figure:
    """pass^1 against cost per simulation. Headline arms coloured and labelled; the rest muted."""
    fig = go.Figure()
    for name, cost, estimate in points:
        color = t.arm_color(name)
        fig.add_trace(go.Scatter(
            x=[cost], y=[estimate.value], mode="markers", name=name,
            showlegend=name in label,
            marker=dict(size=10, color=color, line=dict(color=t.surface, width=2)),
            error_y=dict(type="data", symmetric=False, array=[estimate.hi - estimate.value],
                         arrayminus=[estimate.value - estimate.lo], color=rgba(color, 0.6),
                         thickness=2, width=0),
            hovertemplate=(
                f"<b>{name}</b><br>$" + "%{x:.4f} per simulation<br>pass^1 %{y:.3f}"
                + f" [{estimate.lo:.3f}, {estimate.hi:.3f}]<extra></extra>"
            ),
        ))
        if name in label:
            fig.add_annotation(x=cost, y=estimate.value, text=name, showarrow=False,
                               xanchor="left", xshift=10, font=dict(color=t.text_secondary))
    fig = _layout(fig, t, title="Cost and reliability per arm")
    fig.update_xaxes(title_text="cost per simulation (USD, list rates, judge excluded)",
                     tickprefix="$")
    fig.update_yaxes(title_text="pass^1", range=[0, 1])
    return fig


def rule_bars(counts: Mapping[str, int], t: Tokens, *, arm: str, title: str) -> go.Figure:
    """Refusals by rule for one arm, longest first; value at the bar tip, arm's colour."""
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
        marker=dict(color=t.arm_color(arm), cornerradius=4), width=0.6,
        text=[str(v) for _, v in items], textposition="outside",
        textfont=dict(color=t.text_secondary), cliponaxis=False,
        hovertemplate="%{y}<br>%{x} refusals<extra></extra>",
    ))
    fig = _layout(fig, t, title=title, height=max(220, 40 * len(items) + 100))
    fig.update_xaxes(title_text="refusals")
    fig.update_yaxes(showgrid=False)
    return fig
