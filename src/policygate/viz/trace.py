"""Rendering one trajectory with the harness's decisions inline."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st


def _refused(events: dict[str, list[dict[str, Any]]], trajectory_ids: set[str]) -> list[dict]:
    return [
        e
        for call_id, group in events.items()
        if call_id not in trajectory_ids
        for e in group
        if e["event"] == "tool_denied"
    ]


def _plain(text: str) -> str:
    """Stop Streamlit reading a pair of dollar signs as LaTeX."""
    return text.replace("$", "\\$")


def render_trajectory(
    sim: Any, events: dict[str, list[dict[str, Any]]], *, collapse_before: int | None = None
) -> None:
    messages = sim.messages or []
    breakdown = {}
    if sim.reward_info is not None:
        breakdown = {str(getattr(k, "value", k)): v
                     for k, v in (sim.reward_info.reward_breakdown or {}).items()}
    st.markdown(
        f"**{str(getattr(sim.termination_reason, 'value', sim.termination_reason))}** · "
        + (" · ".join(f"{k} {v:.0f}" for k, v in breakdown.items()) or "no reward")
    )

    trajectory_ids = {
        c.id for m in messages for c in (getattr(m, "tool_calls", None) or [])
    }
    refused = _refused(events, trajectory_ids)
    if refused:
        with st.expander(f"{len(refused)} refused proposal(s) — never reached tau2", expanded=True):
            # Rows, not a table: in a half-width column a table hides the reason
            # behind a horizontal scroll, and the reason is what the reader came for.
            for e in refused:
                st.markdown(f"`{e['seq']}` **{e['tool']}** · `{e['rule']}`")
                st.caption(_plain(f"{e['reason']} — {json.dumps(e['arguments'])}"))
                if e.get("remedy"):
                    st.caption(_plain(f"**Next:** {e['remedy']}"))

    start = collapse_before or 0
    if start:
        with st.expander(f"{start} identical earlier message(s)"):
            for m in messages[:start]:
                _message(m, events)
    for m in messages[start:]:
        _message(m, events)


def _message(m: Any, events: dict[str, list[dict[str, Any]]]) -> None:
    role = getattr(m, "role", "?")
    calls = getattr(m, "tool_calls", None) or []
    with st.chat_message({"assistant": "assistant", "user": "user"}.get(role, "ai")):
        if role == "tool":
            flag = " · error" if getattr(m, "error", False) else ""
            st.caption(f"tool result{flag}")
            st.code((m.content or "")[:1500], language="json")
            return
        if getattr(m, "content", None):
            st.markdown(_plain(m.content))
        for c in calls:
            decisions = [e["rule"] for e in events.get(c.id, []) if e["rule"]]
            st.caption(f"→ `{c.name}`" + (f" · {', '.join(decisions)}" if decisions else ""))
            st.code(json.dumps(c.arguments, indent=1), language="json")
