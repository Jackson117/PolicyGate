"""Ask the banking corpus something, and watch retrieval answer.

The Banking RAG page reports what retrieval was worth across the study's tasks.
This page shows the step that table is made of: for one question, which index
found what, what fusion did with the two lists, and which documents survived
into the prompt.

Nothing is rebuilt here. The embeddings are the study's cache; a query embeds
one short string, and the page reports what that cost rather than assuming it.
"""

from __future__ import annotations

import streamlit as st

from policygate.viz.retrieval import ResourcesUnavailable, inspect, resources
from policygate.viz.session import money, theme

st.title("Retrieval")
theme()

st.caption(
    "One hybrid retrieval over the banking knowledge corpus, the way the study's naive arm "
    "runs it: BM25 and dense search, fused with reciprocal rank fusion, top documents into "
    "the prompt. The study's measured effect is on the Banking RAG page."
)


@st.cache_resource(show_spinner="Loading the study's index…")
def _resources():
    return resources()


try:
    index, docs, embedder, config = _resources()
except ResourcesUnavailable as exc:
    st.info(str(exc))
    st.stop()

st.sidebar.caption(f"{len(docs)} documents · embeddings `{config.embedding_model}`")
k_bm25 = st.sidebar.slider("BM25 candidates", 5, 50, config.k_bm25)
k_dense = st.sidebar.slider("Dense candidates", 5, 50, config.k_dense)
k_final = st.sidebar.slider("Kept for the prompt", 1, 20, config.k_final)
rrf_k = st.sidebar.slider("RRF constant", 1, 120, config.rrf_k,
                          help="score = sum of 1 / (constant + rank). Larger flattens the "
                               "advantage of being first in either list.")

query = st.text_input(
    "Question",
    placeholder="e.g. how long do I have to dispute a card transaction?",
)
if not query.strip():
    st.stop()

with st.spinner("Embedding the question and searching…"):
    try:
        result = inspect(query, index=index, docs=docs, embedder=embedder, k_bm25=k_bm25,
                         k_dense=k_dense, rrf_k=rrf_k, k_final=k_final)
    except Exception as exc:  # noqa: BLE001 - the page explains rather than crashes
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Candidates fused", len(result.candidates))
c2.metric("Found by both", len(result.agreed),
          help="Documents BM25 and the dense index both returned.")
# Six places, because five round this to $0.00000 and a cost that reads as
# zero is a cost nobody checks.
c3.metric("This query cost", money(result.embedding_usd, 6),
          help=f"{result.embedding_tokens} tokens embedded. The corpus embeddings are the "
               "study's cache and are never rebuilt here.")

st.subheader(f"Into the prompt · top {result.k_final}")
for position, candidate in enumerate(result.kept, start=1):
    with st.container(border=True):
        st.markdown(f"**{position}. {candidate.title}** · `{candidate.doc_id}`")
        st.caption(f"{candidate.sources} · fused {candidate.score:.4f}")
        st.caption(candidate.snippet.replace("$", "\\$") + "…")

rest = [c for c in result.candidates if not c.kept]
if rest:
    with st.expander(f"Considered and dropped · {len(rest)}"):
        st.dataframe(
            [
                {
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "BM25": c.bm25_rank,
                    "dense": c.dense_rank,
                    "fused": round(c.score, 5),
                }
                for c in rest
            ],
            hide_index=True,
        )

st.caption(
    "A document found by only one index is not a mistake: BM25 matches words, the dense "
    "index matches meaning, and fusion exists because neither is reliable alone. What the "
    "study measures is whether any of this changes the answer — on its tasks, the honest "
    "summary is on the Banking RAG page."
)
