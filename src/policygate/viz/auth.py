"""One shared password in front of the dashboard, for the public deployment.

The Live session page calls a model on every turn, so a public URL is a button
a stranger can press to spend this project's budget. The service's per-session
cap bounds the damage; this bounds who gets to cause it.

What this is and is not. It is one shared secret, compared in constant time,
held in Streamlit's per-session state -- enough to keep a crawler and a passer-by
out of a demo whose link is handed out in an application. It is not an
authentication system: there are no accounts, nothing is logged, and anyone
with the password is every other person with the password. A deployment that
needs more than that should put a real identity proxy in front of this.

Unset means no gate. That is deliberate: on your own machine the dashboard
reads local files and talks to a service on localhost, and a password there
would protect nothing from anyone.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

ENV_VAR = "POLICYGATE_DASHBOARD_PASSWORD"

_UNLOCKED = "_auth_unlocked"
_WRONG = "_auth_wrong"


def require_password() -> None:
    """Show a password prompt and stop the script until the right one arrives.

    Called before the pages are registered, so no page body runs -- and no
    result, trace or service call is reachable -- until it returns.
    """
    expected = os.environ.get(ENV_VAR)
    if not expected:
        return
    if st.session_state.get(_UNLOCKED):
        return

    st.title("policygate")
    st.caption(
        "This dashboard runs a real agent against a real model, so it is not left open. "
        "The password is in the link you were sent."
    )

    with st.form("unlock"):
        typed = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Open")

    if submitted:
        # compare_digest, not ==, so a wrong guess takes the same time as a
        # right one whatever its first character is.
        if hmac.compare_digest(typed, expected):
            st.session_state[_UNLOCKED] = True
            st.session_state.pop(_WRONG, None)
            st.rerun()
        st.session_state[_WRONG] = True

    if st.session_state.get(_WRONG):
        st.error("That is not the password.")

    st.stop()
