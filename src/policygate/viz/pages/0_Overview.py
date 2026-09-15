"""Overview: the headline comparison, its exclusions, and two of the four figures."""

from __future__ import annotations

import streamlit as st

from policygate.viz import charts
from policygate.viz.data import compare_arms, summarise_arm
from policygate.viz.session import partial_run_notice, run_view, theme

st.title("Overview")

run = run_view()
if run is None:
    st.stop()
partial_run_notice(run)
t = theme()

provenance = run.provenance
st.caption(
    f"plan `{run.fingerprint[:12]}` · commit `{(provenance.get('commit') or '?')[:10]}`"
    + (" · **working tree dirty at start**" if provenance.get("dirty") else "")
)

baseline, full = run.arm("baseline"), run.arm("full")
if not (baseline and full and baseline.simulations and full.simulations):
    st.info("The headline arms have no completed chunks yet.")
    st.stop()

metric = st.radio("Reward", ["reward", "DB"], horizontal=True,
                  help="DB is the world-state component the harness acts on; "
                       "reward also includes the LLM-judged NL assertion.")
component = None if metric == "reward" else "DB"
comparison = compare_arms(full, baseline, component=component)
d = comparison.difference

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"full {metric} pass^1", f"{comparison.treatment:.3f}")
c2.metric(f"baseline {metric} pass^1", f"{comparison.control:.3f}")
c3.metric("Paired difference", f"{d.value:+.3f}", help=f"95% CI [{d.lo:+.3f}, {d.hi:+.3f}]")
c4.metric("p", f"{comparison.p_value:.3g}", help=comparison.test.value)
st.caption(
    f"95% CI [{d.lo:+.3f}, {d.hi:+.3f}] over {d.n_tasks} tasks · "
    f"full higher on {comparison.treatment_higher}, baseline higher on "
    f"{comparison.control_higher}, tied on {comparison.ties}"
)
if comparison.differential_exclusion:
    st.warning(
        f"The arms lost different numbers of simulations to exclusion "
        f"(full {comparison.treatment_excluded}, baseline {comparison.control_excluded}). "
        "A harness exception is recorded by tau2 as an infrastructure error; check the "
        "exception types before reading the difference."
    )

summaries = {a.name: summarise_arm(a) for a in run.arms if a.simulations}
left, right = st.columns(2)
with left:
    st.plotly_chart(
        charts.pass_k_curve({n: summaries[n].pass_k for n in ("baseline", "full")}, t),
        width="stretch", theme=None,
    )
with right:
    points = [
        (name, s.cost_per_simulation, s.pass_k[0])
        for name, s in summaries.items()
        if s.cost_per_simulation is not None and s.pass_k
    ]
    st.plotly_chart(charts.cost_frontier(points, t), width="stretch", theme=None)

with st.expander("Table view"):
    st.dataframe(
        [
            {
                "arm": name,
                "simulations": s.simulations,
                "excluded": sum(s.excluded.values()),
                **{e.statistic: f"{e.value:.3f} [{e.lo:.3f}, {e.hi:.3f}]" for e in s.pass_k},
                "DB pass^1": None if s.db_pass_1 is None else f"{s.db_pass_1.value:.3f}",
                "parallel-call violations": None if s.parallel_violation_rate is None
                else f"{s.parallel_violation_rate.value:.3f}",
                "text+call violations": None if s.mixed_violation_rate is None
                else f"{s.mixed_violation_rate.value:.3f}",
                "cost/sim": None if s.cost_per_simulation is None
                else f"${s.cost_per_simulation:.4f}",
                "hidden retry cost": None if s.hidden_retry_share is None
                else f"{s.hidden_retry_share:+.1%}",
            }
            for name, s in summaries.items()
        ],
        hide_index=True,
    )
st.caption(
    "The parallel-call violation rate is zero by construction in any arm that enforces "
    "the rule; it shows the mechanism is wired, not what it is worth. Text+call "
    "violations are not enforced by the harness and serve as a side-effect check."
)
