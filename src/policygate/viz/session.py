"""Streamlit glue shared by every page: which run, cached, and which colours."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from policygate.viz.data import RunView, load_run
from policygate.viz.theme import Tokens, tokens

DEFAULT_RUN = Path(os.environ.get("POLICYGATE_RUN_DIR", "experiments/w4_v4"))
RUN_ROOT = Path(os.environ.get("POLICYGATE_RUN_ROOT", "experiments"))


def discover_runs(root: Path = RUN_ROOT, *, default: Path = DEFAULT_RUN) -> list[Path]:
    """Runner output directories, newest first, with ``default`` leading if present.

    W4 left four of them (v1 through v4) and comparing runs is most of the
    analysis, so the picker lists what is on disk instead of making the reader
    remember directory names.
    """
    found = [p.parent for p in sorted(root.glob("*/plan.json"))] if root.exists() else []

    def freshness(path: Path) -> float:
        ledger = path / "ledger.json"
        return ledger.stat().st_mtime if ledger.exists() else 0.0

    found.sort(key=freshness, reverse=True)
    if default in found:
        found.remove(default)
        found.insert(0, default)
    return found


RUN_CACHE = int(os.environ.get("POLICYGATE_RUN_CACHE", "2"))
"""How many runs may be held in memory at once.

The cache is one per server process, shared by everyone connected, and a run is
hundreds of megabytes of parsed simulations -- w4_v4 and its ablation together
measure about 1.1 GB. Without a limit, a visitor who clicks through the run
selector loads every run the deployment ships and the container dies. Two is
enough to compare a run with its ablation without re-reading; set it to 1 on a
box with a gigabyte."""


@st.cache_resource(show_spinner="Loading run…", max_entries=RUN_CACHE)
def _load(path: str, version: float) -> RunView:
    # `version` is the ledger's mtime: a finished chunk invalidates the cache, a
    # page reload in between does not re-read thousands of simulations.
    return load_run(Path(path))


def run_view() -> RunView | None:
    runs = discover_runs()
    options = [str(p) for p in runs] + ["other…"]
    choice = st.sidebar.selectbox("Run", options, index=0 if runs else len(options) - 1)
    path = Path(
        st.sidebar.text_input("Run directory", str(DEFAULT_RUN))
        if choice == "other…"
        else choice
    )
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
