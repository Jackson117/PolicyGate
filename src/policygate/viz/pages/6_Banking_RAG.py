"""Banking RAG: the retrieval study run on this harness (policygate-rag)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from policygate.viz import rag
from policygate.viz.session import money, theme

st.title("Banking RAG")
st.caption(
    "policygate-rag: tau2 banking_knowledge with this harness on (gating, audit, loop guard) "
    "in every arm; only retrieval varies. One model (gpt-4.1-mini) for agent and user "
    "simulator, 30 tasks × 3 trials. Read from the study's published results."
)

directory = Path(st.sidebar.text_input("Results directory", str(rag.DEFAULT_RESULTS)))
try:
    r = rag.load(directory)
except rag.ResultsError as e:
    st.info(f"{e}. Point the sidebar at policygate-rag's `results/` directory "
            "(or set POLICYGATE_RAG_RESULTS).")
    st.stop()

t = theme()
idx = r.index
st.sidebar.caption(f"results from commit `{idx['code_commit'][:10]}` · "
                   f"{idx['generated_at'][:10]}")
st.caption(idx["intervals"])

# --- headline -----------------------------------------------------------------
ceiling = r.paired("A4", "A0")
if ceiling:
    d = ceiling[1]["pass^1"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Perfect documents add (A4 − A0)", f"{d['estimate']:+.3f}",
              help=f"95% CI [{d['low']:+.3f}, {d['high']:+.3f}]")
    c2.metric("Ceiling pass^1 (A4)", f"{r.pass1('A4').estimate:.3f}")
    c3.metric("Fails even with perfect documents", f"{1 - r.pass1('A4').estimate:.3f}")
    st.caption(f"A4 − A0: 95% CI [{d['low']:+.3f}, {d['high']:+.3f}] over {d['n_tasks']} tasks")

partial = [a for a in r.arm_ids(with_pass1=False) if r.partial(a)]
for a in partial:
    e = r.arm(a)
    st.warning(f"{a} is partial: {e['episodes']} episodes. "
               + ("Every task has at least one trial, so pass^1 (the per-task mean of the "
                  "trials run) is shown, drawn hollow; pass^k for k > 1 is not."
                  if e.get("covers_all_tasks") else "No numbers are shown for it."))

st.plotly_chart(rag.pass1_forest(r, t), width="stretch")

# --- any pair -------------------------------------------------------------------
st.subheader("Paired differences")
ids = r.arm_ids()
left, right = st.columns(2)
a = left.selectbox("Arm", ids, index=ids.index("A4") if "A4" in ids else 0)
b = right.selectbox("minus", [x for x in ids if x != a], index=0)
pair = r.paired(a, b)
if pair:
    name, diffs = pair
    st.dataframe([{"pair": name, "statistic": k,
                   "difference": f"{v['estimate']:+.3f}",
                   "95% CI": f"[{v['low']:+.3f}, {v['high']:+.3f}]", "tasks": v["n_tasks"]}
                  for k, v in diffs.items()], hide_index=True)

st.subheader("Arms")
st.dataframe(rag.arm_rows(r), hide_index=True)

# --- retrieval --------------------------------------------------------------------
st.subheader("What reached the model")
st.plotly_chart(rag.coverage_dumbbell(r, t), width="stretch")
st.caption("Coverage over the episode is the union of every document shown; at a tool call it "
           "is what the model had when it proposed the call.")
fig = rag.replay_forest(r, t)
if fig is not None:
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{r.replay['metric']}. A counterfactual on fixed conversations: it bounds "
               "what a policy supplies, it does not predict pass^1.")

# --- stage 1 ------------------------------------------------------------------------
st.subheader("Stage 1: re-ranking judges")
metric = st.radio("Metric", list(rag.JUDGE_METRICS), horizontal=True,
                  format_func=rag.JUDGE_METRICS.get)
st.plotly_chart(rag.judge_forest(r, t, metric), width="stretch")
calibrated = {j: m["calibration"] for j, m in r.stage1["common"]["judges"].items()
              if m.get("calibration")}
if calibrated:
    st.dataframe([{"judge": r.stage1["judges"][j]["label"], "Brier": f"{c['brier']:.3f}",
                   "ECE": f"{c['ece']:.3f}", "scored pairs": c["pairs"]}
                  for j, c in calibrated.items()], hide_index=True)
    st.caption("Calibration of the judges that output probabilities, against required "
               "documents.")

# --- spend ---------------------------------------------------------------------------
st.subheader("Spend")
sp = r.spend
st.write(f"{money(sp['total_spent'])} of a {money(sp['ceiling_usd'])} ceiling with the model "
         f"provider (the study's ledger); judges billed to their own accounts: "
         f"{money(rag.billed_elsewhere(sp))}.")
st.dataframe([{"line": k, "allocated": f"{v['allocated']:.2f}", "spent": f"{v['spent']:.2f}"}
              for k, v in sp["lines"].items()], hide_index=True)
