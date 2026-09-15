"""Attribution: each mechanism on its own against the shared baseline.

These arms were sized for description, not significance (docs/w4_design.md):
one trial each, and a mechanism that changes two tasks moves pass^1 by under
two points. The intervals are shown so that is visible rather than stated.
"""

from __future__ import annotations

import streamlit as st

from policygate.eval.passk import ExclusionPolicy
from policygate.viz import charts
from policygate.viz.data import compare_arms
from policygate.viz.session import partial_run_notice, run_view, theme

st.title("Ablation")

run = run_view()
if run is None:
    st.stop()
partial_run_notice(run)
t = theme()

baseline = run.arm("baseline")
if baseline is None or not baseline.simulations:
    st.info("The baseline has no completed chunks yet.")
    st.stop()

f1, f2 = st.columns(2)
component = {"reward": None, "DB": "DB"}[f1.radio("Reward", ["reward", "DB"], horizontal=True)]
policy = ExclusionPolicy(f2.radio(
    "Exclusions", [p.value for p in ExclusionPolicy], horizontal=True,
    help="count_as_failure scores lost simulations as failures: a bound for an arm "
         "that lost more than the baseline.",
))

rows = []
for arm in run.arms:
    if arm.name == "baseline" or not arm.simulations:
        continue
    try:
        rows.append((arm.name, compare_arms(arm, baseline, component=component, policy=policy)))
    except ValueError:
        continue  # no task scored in both arms yet
if not rows:
    st.info("No arm shares a completed task with the baseline yet.")
    st.stop()

st.plotly_chart(charts.attribution_forest(rows, t), width="stretch", theme=None)
st.caption(
    "Single-mechanism arms are compared with the baseline under Holm correction across the "
    "family; p-values below are unadjusted. Effects need not add up to full − baseline: "
    "the gap is interaction between mechanisms."
)

with st.expander("Table view", expanded=True):
    st.dataframe(
        [
            {
                "arm": name,
                "tasks compared": c.difference.n_tasks,
                "difference": round(c.difference.value, 4),
                "95% CI": f"[{c.difference.lo:+.3f}, {c.difference.hi:+.3f}]",
                "p (unadjusted)": c.p_value,
                "test": c.test.value,
                "treatment higher": c.treatment_higher,
                "baseline higher": c.control_higher,
                "balanced": c.balanced,
                "excluded (arm / baseline)": f"{c.treatment_excluded} / {c.control_excluded}",
            }
            for name, c in rows
        ],
        hide_index=True,
    )
