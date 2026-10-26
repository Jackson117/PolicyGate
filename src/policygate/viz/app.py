"""Dashboard entry point.

Run with:
    streamlit run src/policygate/viz/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from policygate.environment import apply_project_env
from policygate.viz.auth import require_password

# Before any page calls a provider. The dashboard embeds a question and talks to
# the service, and both go wrong quietly when a shell's stray key shadows this
# project's own (see policygate.environment).
apply_project_env()

PAGES = Path(__file__).parent / "pages"

st.set_page_config(page_title="policygate", layout="wide")

# Before the pages are registered, so nothing a page does -- read a run, call
# the service, embed a query -- is reachable without the password. No-op unless
# POLICYGATE_DASHBOARD_PASSWORD is set, which is how it stays out of the way
# locally.
require_password()
st.navigation(
    [
        st.Page(PAGES / "0_Overview.py", title="Overview", default=True),
        st.Page(PAGES / "1_Paired_trace.py", title="Paired trace"),
        st.Page(PAGES / "2_Single_trace.py", title="Single trace"),
        st.Page(PAGES / "3_Ablation.py", title="Ablation"),
        st.Page(PAGES / "4_Failures.py", title="Failures"),
        st.Page(PAGES / "5_Audit_chain.py", title="Audit chain"),
        st.Page(PAGES / "6_Banking_RAG.py", title="Banking RAG"),
        st.Page(PAGES / "7_Live_session.py", title="Live session"),
        st.Page(PAGES / "8_Retrieval.py", title="Retrieval"),
    ]
).run()
