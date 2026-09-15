"""Where tasks moved, and which rule was involved.

No failure classifier: migration and rule attribution come straight from
recorded outcomes and audit decisions, so nothing here is a heuristic's opinion.
"""

from __future__ import annotations

from collections import Counter

import streamlit as st

from policygate.viz import charts
from policygate.viz.data import simulation_rows, task_shifts
from policygate.viz.session import partial_run_notice, run_view, theme

st.title("Failures and migration")

run = run_view()
if run is None:
    st.stop()
partial_run_notice(run)
t = theme()

baseline, full = run.arm("baseline"), run.arm("full")
if not (baseline and full and baseline.simulations and full.simulations):
    st.info("The headline arms have no completed chunks yet.")
    st.stop()

shifts = task_shifts(full, baseline)
counts = Counter(tuple(s.category.split(" → ")) for s in shifts)
left, right = st.columns([1, 1])
with left:
    st.plotly_chart(charts.migration_matrix(counts, t), width="stretch", theme=None)
    st.caption(
        f"{len(shifts)} tasks scored in both arms. Rows: baseline; columns: full. "
        "“sometimes” needs repeated trials to mean anything."
    )

arms_with_chains = [a.name for a in run.arms if a.simulations and a.name != "baseline"]
with right:
    chosen = st.selectbox("Refusals in arm", arms_with_chains,
                          index=arms_with_chains.index("full") if "full" in arms_with_chains else 0)
    rows = simulation_rows(run.arm(chosen))
    by_rule = Counter()
    for row in rows:
        by_rule.update(row["denials_by_rule"])
    if by_rule:
        st.plotly_chart(charts.rule_bars(by_rule, t, title=f"Refusals by rule · {chosen}"),
                        width="stretch", theme=None)
    else:
        st.info("No refusals recorded in this arm.")

category = st.selectbox(
    "Tasks in migration class",
    sorted({s.category for s in shifts}),
    index=0,
)
full_rows = {(r["task_id"], r["trial"]): r for r in simulation_rows(full)}
st.dataframe(
    [
        {
            "task": s.task_id,
            "baseline": f"{s.control_passed}/{s.control_trials}",
            "full": f"{s.treatment_passed}/{s.treatment_trials}",
            "Δ": round(s.delta, 2),
            "refusals in full": sum(
                r["denials"] for (task, _), r in full_rows.items() if task == s.task_id
            ),
            "rules": ", ".join(sorted({
                rule for (task, _), r in full_rows.items() if task == s.task_id
                for rule in r["denials_by_rule"]
            })),
        }
        for s in shifts
        if s.category == category
    ],
    hide_index=True,
)
