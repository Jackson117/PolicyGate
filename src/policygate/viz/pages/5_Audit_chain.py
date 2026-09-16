"""The audit chain for one simulation: verify it, and say where to look.

The chain is the harness's third contribution, and W4 is the evidence that it
pays: both defects the runs uncovered were found by comparing refusals against
each task's reference actions, not by looking at scores. This page is that
comparison, per simulation.

What it does not do is guess. tau2 scores a run by hashing the world state at
the end, so no record is "the failure" and none can be marked as one. Every
anchor is an observable -- a reference action that never ran, a write the
reference does not contain, the first refusal, the handover, a break in the
chain -- and the trace below starts at whichever one is selected.
"""

from __future__ import annotations

import json

import streamlit as st

from policygate.viz.data import (
    anchors,
    chain_integrity,
    chain_rows,
    harness_events_by_call,
    message_index_for,
    representativeness,
    simulation_rows,
    task_shifts,
)
from policygate.viz.session import money, partial_run_notice, run_view
from policygate.viz.trace import render_trajectory

st.title("Audit chain")

run = run_view()
if run is None:
    st.stop()
partial_run_notice(run)

arms = [a for a in run.arms if a.simulations]
if not arms:
    st.info("No completed chunks in this run yet.")
    st.stop()

names = [a.name for a in arms]
arm = run.arm(st.sidebar.selectbox("Arm", names, index=names.index("full") if "full" in names
                                   else 0))
rows = simulation_rows(arm)
only_failed = st.sidebar.checkbox("Failed simulations only", value=True)
choices = [r for r in rows if not only_failed or r["passed"] is False]
if not choices:
    st.info("Every simulation in this arm passed.")
    st.stop()

# Most refusals first: the sessions where the harness intervened most are the
# ones whose chain has something to say.
choices.sort(key=lambda r: (-r["denials"], r["task_id"], r["trial"]))
label = {f"task {r['task_id']} · trial {r['trial']} · {r['denials']} refusal(s)"
         f"{'' if r['passed'] else ' · failed'}": r for r in choices}
picked = label[st.selectbox("Simulation", list(label))]

simulation = next(s for s in arm.simulations
                  if s.task_id == picked["task_id"] and s.trial == picked["trial"])
session = arm.session_for(simulation)

reference = []
try:  # the tasks are the benchmark's, and the dashboard may run without it
    from tau2.registry import registry

    task = next(t for t in registry.get_tasks_loader(run.ledger.get("domain", "retail"))(None)
                if t.id == simulation.task_id)
    criteria = task.evaluation_criteria
    reference = [a for a in (criteria.actions or ()) if a.requestor == "assistant"]
except Exception as exc:  # noqa: BLE001 - the page still works without it
    st.caption(f"Reference actions unavailable ({type(exc).__name__}); anchors will be partial.")

intact, problems = chain_integrity(session)
found = anchors(session, reference)

left, right = st.columns([2, 3])
with left:
    if intact:
        st.success(f"Chain verified · {len(chain_rows(session))} records")
    else:
        st.error("Chain broken")
        for problem in problems:
            st.caption(problem)
with right:
    control = run.arm("baseline")
    if control is not None and control.simulations and arm.name != "baseline":
        shifts = task_shifts(arm, control)
        rep = representativeness(simulation.task_id, shifts)
        st.caption(
            f"Task {simulation.task_id} passed {rep.task.treatment_passed}/"
            f"{rep.task.treatment_trials} under {arm.name} and "
            f"{rep.task.control_passed}/{rep.task.control_trials} under baseline · "
            f"{rep.tasks_moving_the_same_way} other task(s) moved the same way, "
            f"{rep.tasks_moving_the_other_way} the other way, "
            f"{rep.tasks_unchanged} did not move"
        )
    st.caption(f"{picked['proposals']} proposal(s) · {picked['denials']} refused · "
               f"{money(picked['agent_llm_cost'] or 0.0, 4)} of agent spend")

st.subheader("Where to look")
if not found:
    st.info(
        "Nothing to flag: every reference action ran, nothing else was written, "
        "no refusal, no handover, chain intact."
    )
# Kept in session state and keyed by the simulation: a button's True lasts only
# for the rerun it triggered, so without this the trace would spring back open
# the moment anything else on the page was touched.
key = f"{arm.name}/{simulation.task_id}/{simulation.trial}"
if st.session_state.get("audit_jump_for") != key:
    st.session_state["audit_jump_for"] = key
    st.session_state["audit_jump"] = 0

for index, anchor in enumerate(found):
    columns = st.columns([5, 1])
    with columns[0]:
        st.markdown(f"**{anchor.label}**")
        st.caption(anchor.detail + ("" if anchor.seq is None else f" · record #{anchor.seq}"))
    if anchor.seq is not None:
        with columns[1]:
            if st.button("Go to", key=f"jump-{index}"):
                st.session_state["audit_jump"] = anchor.seq
jump = st.session_state.get("audit_jump", 0)

records = chain_rows(session)
with st.expander(f"The chain · {len(records)} records", expanded=False):
    st.dataframe(
        [
            {
                "seq": r.seq,
                "linked": "yes" if r.linked else "BROKEN",
                "event": r.event,
                "tool": r.tool,
                "rule": r.rule,
                "arguments": json.dumps(r.arguments) if r.arguments else None,
                "world diff": json.dumps(r.world_diff)[:200] if r.world_diff else None,
                "prev": r.prev_hash[:10],
                "hash": r.record_hash[:10],
            }
            for r in records
        ],
        hide_index=True,
    )
st.caption(
    "`linked` is recomputed here from the records themselves: each one's `prev` must equal "
    "the previous record's `hash`. Editing any field of a stored record breaks the link "
    "after it, which is what makes the chain evidence rather than a log."
)

st.subheader("Trajectory")
if jump:
    columns = st.columns([5, 1])
    columns[0].caption(f"Collapsed everything before record #{jump}.")
    if columns[1].button("Show all"):
        st.session_state["audit_jump"] = 0
        st.rerun()
# A record's seq is not a message index -- a refused proposal has a record and no
# message -- so the two are joined on the tool call's id.
start = message_index_for(simulation, session, jump) if jump else 0
render_trajectory(simulation, harness_events_by_call(session), collapse_before=start)
