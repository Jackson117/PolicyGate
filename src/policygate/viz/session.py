"""Streamlit glue shared by every page: which run, cached, and which colours."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from policygate.viz.data import RunView, load_run
from policygate.viz.theme import Tokens, tokens

DEFAULT_RUN = Path(os.environ.get("POLICYGATE_RUN_DIR", "experiments/w4_v3"))


@st.cache_resource(show_spinner="Loading run…")
def _load(path: str, version: float) -> RunView:
    # `version` is the ledger's mtime: a finished chunk invalidates the cache, a
    # page reload in between does not re-read thousands of simulations.
    return load_run(Path(path))


def run_view() -> RunView | None:
    path = Path(st.sidebar.text_input("Run directory", str(DEFAULT_RUN)))
    if not (path / "plan.json").exists():
        st.warning(f"No runner output at `{path}`.")
        return None
    ledger = path / "ledger.json"
    version = ledger.stat().st_mtime if ledger.exists() else 0.0
    if st.sidebar.button("Reload"):
        _load.clear()
    return _load(str(path), version)


def money(value: float, digits: int = 2) -> str:
    """A dollar amount safe inside Streamlit markdown, where a pair of $ starts LaTeX."""
    return f"\\${value:.{digits}f}"


def theme() -> Tokens:
    return tokens(st.context.theme.type)


def partial_run_notice(run: RunView) -> None:
    complete = len(run.ledger.get("complete", {}))
    incomplete = len(run.ledger.get("incomplete", {}))
    st.sidebar.caption(
        f"{complete} chunk(s) complete, {incomplete} stopped · spend {money(run.spend)} "
        "(list rates; judge excluded)"
    )
