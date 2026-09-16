"""Dashboard entry point.

Run with:
    streamlit run src/policygate/viz/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PAGES = Path(__file__).parent / "pages"

st.set_page_config(page_title="policygate", layout="wide")
st.navigation(
    [
        st.Page(PAGES / "0_Overview.py", title="Overview", default=True),
        st.Page(PAGES / "1_Paired_trace.py", title="Paired trace"),
        st.Page(PAGES / "2_Single_trace.py", title="Single trace"),
        st.Page(PAGES / "3_Ablation.py", title="Ablation"),
        st.Page(PAGES / "4_Failures.py", title="Failures"),
        st.Page(PAGES / "5_Audit_chain.py", title="Audit chain"),
    ]
).run()
