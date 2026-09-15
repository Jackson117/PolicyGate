"""Single trace: one simulation step by step. The fallback to the paired view.

It answers "what happened here", not "what did the harness change" -- so it
still shows how often this task passed in this arm, to keep one run from
standing in for all of them.
"""

from __future__ import annotations

import streamlit as st

from policygate.eval.passk import observations_from_simulations
from policygate.viz.data import harness_events_by_call
from policygate.viz.session import money, partial_run_notice, run_view
from policygate.viz.trace import render_trajectory

st.title("Single trace")

run = run_view()
if run is None:
    st.stop()
partial_run_notice(run)

arms = [a.name for a in run.arms if a.simulations]
if not arms:
    st.info("No arm has completed chunks yet.")
    st.stop()
c1, c2, c3 = st.columns(3)
arm = run.arm(c1.selectbox("Arm", arms))
tasks = sorted({s.task_id for s in arm.simulations}, key=lambda x: (len(x), x))
task_id = c2.selectbox("Task", tasks)
trials = sorted((s for s in arm.simulations if s.task_id == task_id), key=lambda s: s.trial)
sim = c3.selectbox("Trial", trials, format_func=lambda s: f"trial {s.trial}")

outcomes = observations_from_simulations(arm.simulations).by_task().get(task_id, ())
st.info(
    f"Task {task_id} passed {sum(outcomes)}/{len(outcomes)} scored trial(s) in `{arm.name}`."
    + ("" if len(outcomes) > 1 else " With one trial, stability is unknown.")
)
session = arm.session_for(sim)
if session is not None:
    st.caption(
        f"{session.proposals} proposal(s) · agent spend {money(session.llm_cost, 4)} "
        f"(tau2 recorded {money(sim.agent_cost or 0, 4)}) · audit chain "
        + ("intact" if session.chain_intact else "**BROKEN**")
    )
render_trajectory(sim, harness_events_by_call(session))
