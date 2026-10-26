"""A live conversation: you type, the model proposes, the harness decides.

The rest of this dashboard reads runs that finished. This page is the same
harness working now, against the running service -- the deployment path, not
the benchmark one. What W4 measured over 4158 simulations happens here one turn
at a time, and the three things it measured are the three things on screen: the
rules that fired, the approval a person has to give before a write, and the
chain verifying itself.

The retail backend is tau2's environment with a fresh database per session, so
the orders are real objects with real statuses and nothing here touches a real
shop. The person, the model and the harness are real.
"""

from __future__ import annotations

import json

import streamlit as st

from policygate.viz.data import anchors, chain_integrity, chain_rows
from policygate.viz.live import (
    DEFAULT_URL,
    ServiceClient,
    ServiceUnavailable,
    SessionNotFound,
    records_as_audit,
)
from policygate.viz.session import theme

st.title("Live session")

url = st.sidebar.text_input("Service URL", DEFAULT_URL)
client = ServiceClient(url)
theme()

st.caption(
    "A real conversation through the deployed path: your words, the model, and the harness "
    "deciding on every tool call. Each turn calls the model, so each turn costs money."
)

waiting = client.inbox()
if waiting is None:
    st.warning(
        f"No service answering at `{url}`.\n\n"
        "Start it locally with `make service` (it needs a provider key in `.env` and a "
        "database — SQLite by default), or point the sidebar at a running one."
    )
    st.stop()

# --- the session ----------------------------------------------------------

if "live_session" not in st.session_state:
    st.session_state["live_session"] = None

columns = st.columns([3, 1])
with columns[0]:
    session_id = st.session_state["live_session"]
    st.caption(f"session `{session_id}`" if session_id else "no session yet")
with columns[1]:
    if st.button("New session", width="stretch"):
        try:
            st.session_state["live_session"] = client.open_session()
            # POST /sessions mints an id; the session exists from its first
            # message. Remembering that we minted this one is what separates
            # "nothing said yet" from "that id does not exist".
            st.session_state["live_session_is_new"] = True
            st.session_state["live_error"] = None
            st.rerun()
        except ServiceUnavailable as exc:
            st.session_state["live_error"] = str(exc)

if st.session_state.get("live_error"):
    st.error(st.session_state["live_error"])

# The reviewer's way in. A person approving a write did not start the
# conversation -- they arrive because something is waiting -- so every session
# with a pending approval is reachable here, whoever opened it.
if waiting:
    st.subheader(f"Waiting for a decision · {len(waiting)}")
    for item in waiting:
        columns = st.columns([5, 1])
        columns[0].markdown(
            f"**{item.tool}** · `{item.session_id[:8]}` · "
            f"{json.dumps(item.arguments)[:120]}".replace("$", "\\$")
        )
        if columns[1].button("Open", key=f"open-{item.request_id}", width="stretch"):
            st.session_state["live_session"] = item.session_id
            st.session_state["live_session_is_new"] = False
            st.rerun()

if session_id is None:
    with st.expander("Open a session by id"):
        typed = st.text_input("Session id", placeholder="paste a session id")
        if typed:
            st.session_state["live_session"] = typed.strip()
            st.session_state["live_session_is_new"] = False
            st.rerun()
    st.info(
        "Start a session, then ask for something a retail agent would have to act on — "
        "cancelling an order, returning an item. Identify yourself the way a customer would: "
        "the environment knows users by email, or by name and zip."
    )
    st.stop()

try:
    view = client.view(session_id)
except SessionNotFound:
    # Either a session that has not been spoken to yet, or an id that was
    # never one. Only the first is worth a chat box.
    view = None
    if not st.session_state.get("live_session_is_new"):
        st.error(f"The service holds no session `{session_id}`.")
        if st.button("Forget it"):
            st.session_state["live_session"] = None
            st.rerun()
        st.stop()

# --- what was said --------------------------------------------------------

st.subheader("Conversation")
if view is None:
    st.caption("Nothing said yet. The session starts with your first message.")
for message in (view.transcript if view else []):
    role = message.get("role")
    if role == "system":
        continue
    if role == "tool":
        with st.chat_message("ai"):
            st.caption("tool result · refused by the harness" if message.get("error")
                       else "tool result")
            st.code((message.get("content") or "")[:800], language="json")
        continue
    with st.chat_message("user" if role == "user" else "assistant"):
        if message.get("content"):
            st.markdown(str(message["content"]).replace("$", "\\$"))
        for name in message.get("tool_calls") or []:
            st.caption(f"→ `{name}`")

# --- the approval gate ----------------------------------------------------

pending = view.pending_approval if view else None
if pending is not None:
    st.subheader("Waiting for you")
    st.warning(
        f"**{pending.tool}** — {pending.reason}\n\n"
        f"```json\n{json.dumps(pending.arguments, indent=2)}\n```\n"
        "The harness stopped here. The call has not run, and will not run unless you say so."
    )
    reviewer = st.text_input("Your name", value="reviewer")
    note = st.text_input("Note (optional)", placeholder="why you are approving or refusing")
    approve, reject = st.columns(2)
    decision = None
    if approve.button("Approve", type="primary", width="stretch"):
        decision = True
    if reject.button("Reject", width="stretch"):
        decision = False
    if decision is not None:
        with st.spinner("Resuming the session…"):
            try:
                client.answer(pending.request_id, approved=decision, reviewer=reviewer or "anon",
                              note=note or None)
            except ServiceUnavailable as exc:
                st.session_state["live_error"] = str(exc)
        st.rerun()

# --- say something --------------------------------------------------------

prompt = st.chat_input("Say something to the agent", disabled=pending is not None)
if prompt:
    with st.spinner("The model is proposing, the harness is deciding…"):
        try:
            st.session_state["live_turn_start"] = len(client.audit(session_id)[2])
            client.send(session_id, prompt)
            st.session_state["live_error"] = None
        except ServiceUnavailable as exc:
            st.session_state["live_error"] = str(exc)
    st.rerun()

# --- what the harness did -------------------------------------------------

try:
    claimed_intact, claimed_breaks, raw = client.audit(session_id)
except SessionNotFound:
    claimed_intact, claimed_breaks, raw = True, [], []
records = records_as_audit(raw)
rows = chain_rows(records)
# Verified here, from the records themselves, rather than taking the service's
# word for it: a chain whose holder vouches for it proves nothing.
intact, breaks = chain_integrity(records)

st.subheader("What the harness did")
start = st.session_state.get("live_turn_start") or 0
scope = st.radio("Show", ["This turn", "Whole session"], horizontal=True,
                 label_visibility="collapsed")
shown = rows[start:] if scope == "This turn" and start < len(rows) else rows
if not shown:
    st.caption("Nothing yet — the harness records a decision for every proposed call.")
for row in shown:
    mark = {"tool_executed": "✅", "tool_denied": "⛔️"}.get(row.event, "·")
    head = f"{mark} `{row.event}`" + (f" · **{row.tool}**" if row.tool else "")
    st.markdown(head + (f" · `{row.rule}`" if row.rule else ""))
    if row.arguments:
        st.caption(json.dumps(row.arguments)[:300].replace("$", "\\$"))
    if row.event == "tool_denied" and row.reason:
        st.caption(f"↳ {row.reason}".replace("$", "\\$"))
        if row.remedy:
            st.caption(f"↳ **Next:** {row.remedy}".replace("$", "\\$"))

# --- the chain ------------------------------------------------------------

st.subheader("Audit chain")
left, right = st.columns([2, 3])
with left:
    if intact:
        st.success(f"Verified here · {len(rows)} records")
    else:
        st.error("Broken")
        for problem in breaks:
            st.caption(str(problem))
    if claimed_intact != intact:
        st.warning(
            f"The service reports intact={claimed_intact}, this page recomputes {intact}. "
            "Trust neither until the difference is explained."
        )
        for problem in claimed_breaks:
            st.caption(f"service: {problem}")
with right:
    found = anchors(records)
    if found:
        st.caption("Worth a look: " + "; ".join(a.label for a in found[:3]))
    else:
        st.caption("Nothing flagged: no refusal stopped a write, nothing ran unasked.")

with st.expander("The records", expanded=False):
    st.dataframe(
        [
            {
                "seq": r.seq,
                "linked": "yes" if r.linked else "BROKEN",
                "event": r.event,
                "tool": r.tool,
                "rule": r.rule,
                "prev": r.prev_hash[:10],
                "hash": r.record_hash[:10],
            }
            for r in rows
        ],
        hide_index=True,
    )
st.caption(
    "`linked` is recomputed from the records the service returned: each one's `prev` must "
    "equal the previous record's `hash`. This is the same verification the benchmark pages "
    "run, against a session that is still going."
)
