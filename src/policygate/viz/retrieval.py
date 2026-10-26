"""Run one retrieval the way the banking study's naive arm runs it.

The Banking RAG page reports what retrieval was worth across 30 tasks. This
module answers the question that table raises and cannot show: for *this*
question, what came back, from which index, and what survived into the prompt.

Two deliberate shapes:

* **The sibling is imported in one function.** ``resources()`` is the only
  thing here that knows ``policygate_rag`` exists, so the dashboard still runs
  without that checkout and everything below is testable with fakes.
* **Nothing is rebuilt.** The embeddings come from the study's cache; building
  them costs real money and is the study's job, not a dashboard's. A query
  embeds one short string, which is a fraction of a cent, and the cost comes
  back with the result rather than being assumed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict


class ResourcesUnavailable(RuntimeError):
    """The study's index or its package could not be loaded."""


class Candidate(BaseModel):
    """One document the fusion considered, and where it came from."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    snippet: str
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    kept: bool
    """Whether it survived into the prompt -- the top ``k_final`` after fusion."""

    @property
    def sources(self) -> str:
        found = []
        if self.bm25_rank is not None:
            found.append(f"BM25 #{self.bm25_rank}")
        if self.dense_rank is not None:
            found.append(f"dense #{self.dense_rank}")
        return " · ".join(found)


class Inspection(BaseModel):
    """What one query returned, and what it cost."""

    model_config = ConfigDict(frozen=True)

    query: str
    candidates: tuple[Candidate, ...]
    k_final: int
    embedding_usd: float
    embedding_tokens: int

    @property
    def kept(self) -> tuple[Candidate, ...]:
        return tuple(c for c in self.candidates if c.kept)

    @property
    def agreed(self) -> tuple[Candidate, ...]:
        """Found by both indexes -- the ones fusion is least likely to be wrong about."""
        return tuple(
            c for c in self.candidates if c.bm25_rank is not None and c.dense_rank is not None
        )


class _Index(Protocol):
    def search(
        self, query: str, query_vector: Sequence[float], *, k_bm25: int, k_dense: int, rrf_k: int
    ) -> list[Any]: ...


class _Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Any: ...


def resources(results_dir: Any = None) -> tuple[Any, Mapping[str, Any], Any, Any]:
    """``(index, docs, embedder, config)`` from the study's own loader.

    Raises ResourcesUnavailable with a sentence the page can show, rather than
    an ImportError: a dashboard without the sibling checkout is a normal state,
    not a crash.
    """
    try:
        from policygate_rag.config import RagConfig, RetrievalMode
        from policygate_rag.runner import retrieval_resources
    except ImportError as exc:
        raise ResourcesUnavailable(
            "policygate-rag is not importable. Install the sibling checkout "
            "(`uv pip install -e ../policygate-rag`) to inspect retrieval."
        ) from exc

    config = RagConfig(name="inspector", retrieval=RetrievalMode.NAIVE)
    try:
        built = retrieval_resources(config)
    except FileNotFoundError as exc:
        raise ResourcesUnavailable(
            f"The study's embedding cache is missing ({exc}). The dashboard never builds it: "
            "embedding the corpus costs money and belongs to the study."
        ) from exc
    if built is None:
        raise ResourcesUnavailable("retrieval is disabled in this configuration")
    return built.index, built.docs, built.embedder, config


def inspect(
    query: str,
    *,
    index: _Index,
    docs: Mapping[str, Any],
    embedder: _Embedder,
    k_bm25: int = 20,
    k_dense: int = 20,
    rrf_k: int = 60,
    k_final: int = 5,
    snippet: int = 320,
) -> Inspection:
    """One hybrid retrieval, with every candidate's rank in each source."""
    if not query.strip():
        raise ValueError("a query is needed")

    embedding = embedder.embed([query])
    fused = index.search(
        query, embedding.vectors[0], k_bm25=k_bm25, k_dense=k_dense, rrf_k=rrf_k
    )

    candidates = []
    for position, item in enumerate(fused):
        document = docs.get(item.doc_id)
        candidates.append(
            Candidate(
                doc_id=item.doc_id,
                title=getattr(document, "title", "(not in the corpus)"),
                snippet=(getattr(document, "content", "") or "")[:snippet],
                score=item.score,
                bm25_rank=item.ranks.get("bm25"),
                dense_rank=item.ranks.get("dense"),
                kept=position < k_final,
            )
        )
    return Inspection(
        query=query,
        candidates=tuple(candidates),
        k_final=k_final,
        embedding_usd=getattr(embedding, "usd", 0.0),
        embedding_tokens=getattr(embedding, "tokens", 0),
    )
