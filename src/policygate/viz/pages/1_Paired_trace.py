"""Paired trace: the same task under baseline and full, aligned where they part.

The main view. A single trajectory shows what happened once; a pair shows what
the harness changed -- and every pair carries what it stands for, so a striking
example cannot be mistaken for a typical one.
"""

from __future__ import annotations

import streamlit as st

from policygate.viz.data import (
    first_divergence,
    harness_events_by_call,
    representativeness,
    task_shifts,
)
from policygate.viz.session import partial_run_notice, run_view
from policygate.viz.trace import render_trajectory

st.title("Paired trace")

run = run_view()
if run is None:
    st.stop()
partial_run_notice(run)

arms = [a.name for a in run.arms if a.simulations]


def default_index(name: str) -> int:
    return arms.index(name) if name in arms else 0


c1, c2, c3 = st.columns(3)
control_name = c1.selectbox("Control", arms, index=default_index("baseline"))
treatment_name = c2.selectbox("Treatment", arms, index=default_index("full"))
control, treatment = run.arm(control_name), run.arm(treatment_name)
shifts = task_shifts(treatment, control)
if not shifts:
    st.info("No task has completed in both arms yet.")
    st.stop()

ordered = sorted(shifts, key=lambda s: -abs(s.delta))
choice = c3.selectbox(
    "Task (largest change first)",
    [s.task_id for s in ordered],
    format_func=lambda tid: next(
        f"{tid} · {s.category} · Δ {s.delta:+.2f}" for s in ordered if s.task_id == tid
    ),
)

rep = representativeness(choice, shifts)
s = rep.task
st.info(
    f"**What this pair stands for.** Task {s.task_id} passed "
    f"{s.control_passed}/{s.control_trials} under {control_name} and "
    f"{s.treatment_passed}/{s.treatment_trials} under {treatment_name}. "
    f"Of the other {len(shifts) - 1} tasks, {rep.tasks_moving_the_same_way} moved the same "
    f"way, {rep.tasks_moving_the_other_way} the other way, {rep.tasks_unchanged} did not move."
)

def trials_of(arm, task_id):
    return sorted((x for x in arm.simulations if x.task_id == task_id), key=lambda x: x.trial)

left_trials, right_trials = trials_of(control, choice), trials_of(treatment, choice)

def trial_label(sim):
    reward = "no outcome" if sim.reward_info is None else f"reward {sim.reward_info.reward:.0f}"
    return f"trial {sim.trial} · {reward}"

p1, p2 = st.columns(2)
left = p1.selectbox(f"{control_name} trial", left_trials, format_func=trial_label)
right = p2.selectbox(f"{treatment_name} trial", right_trials, format_func=trial_label)

divergence = first_divergence(left.messages or [], right.messages or [])
st.caption(
    "Identical trajectories." if divergence is None
    else f"Trajectories first differ at message {divergence}. Earlier messages are collapsed."
)

for column, arm, sim in ((p1, control, left), (p2, treatment, right)):
    with column:
        events = harness_events_by_call(arm.session_for(sim))
        render_trajectory(sim, events, collapse_before=divergence)
