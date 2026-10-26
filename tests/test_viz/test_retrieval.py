"""Tests for the retrieval inspector's shaping.

Fakes throughout: no sibling checkout, no embedding cache, no API call. What
is pinned is what the page claims on screen -- which index found a document,
what fusion did with it, what survived into the prompt, and what the query
cost.
"""

import pytest

from policygate.viz.retrieval import Inspection, ResourcesUnavailable, inspect, resources


class FakeFused:
    def __init__(self, doc_id, score, ranks):
        self.doc_id, self.score, self.ranks = doc_id, score, ranks


class FakeDoc:
    def __init__(self, title, content):
        self.title, self.content = title, content


class FakeIndex:
    """Returns a fixed fusion, and records the knobs it was called with."""

    def __init__(self, fused):
        self.fused = fused
        self.seen = {}

    def search(self, query, query_vector, *, k_bm25, k_dense, rrf_k):
        self.seen = {"query": query, "vector": list(query_vector),
                     "k_bm25": k_bm25, "k_dense": k_dense, "rrf_k": rrf_k}
        return self.fused


class FakeEmbedding:
    vectors = [[0.1, 0.2, 0.3]]
    tokens = 7
    usd = 0.000013


class FakeEmbedder:
    def __init__(self):
        self.texts = None

    def embed(self, texts):
        self.texts = list(texts)
        return FakeEmbedding()


DOCS = {
    "d1": FakeDoc("Wire transfer limits", "Daily limits are set per account tier. " * 20),
    "d2": FakeDoc("Dispute a card charge", "File within 60 days."),
    "d3": FakeDoc("Statement credits", "Applied within one cycle."),
}


def run(fused, **kwargs) -> tuple[Inspection, FakeIndex, FakeEmbedder]:
    index, embedder = FakeIndex(fused), FakeEmbedder()
    result = inspect("what is my wire limit?", index=index, docs=DOCS, embedder=embedder,
                     **kwargs)
    return result, index, embedder


class TestWhatCameBack:
    def test_each_candidate_says_which_index_found_it_and_at_what_rank(self):
        result, _, _ = run([
            FakeFused("d1", 0.032, {"bm25": 1, "dense": 2}),
            FakeFused("d2", 0.016, {"dense": 5}),
            FakeFused("d3", 0.015, {"bm25": 9}),
        ])
        by_id = {c.doc_id: c for c in result.candidates}
        assert (by_id["d1"].bm25_rank, by_id["d1"].dense_rank) == (1, 2)
        assert (by_id["d2"].bm25_rank, by_id["d2"].dense_rank) == (None, 5)
        assert by_id["d1"].sources == "BM25 #1 · dense #2"
        assert by_id["d2"].sources == "dense #5"

    def test_the_documents_that_both_indexes_found_are_singled_out(self):
        result, _, _ = run([
            FakeFused("d1", 0.03, {"bm25": 1, "dense": 2}),
            FakeFused("d2", 0.01, {"dense": 5}),
        ])
        assert [c.doc_id for c in result.agreed] == ["d1"]

    def test_only_the_top_k_final_are_marked_as_reaching_the_prompt(self):
        result, _, _ = run(
            [FakeFused(f"d{i}", 1 / i, {"bm25": i}) for i in range(1, 4)], k_final=2
        )
        assert [c.doc_id for c in result.kept] == ["d1", "d2"]
        assert [c.kept for c in result.candidates] == [True, True, False]

    def test_titles_and_snippets_come_from_the_corpus(self):
        result, _, _ = run([FakeFused("d1", 0.03, {"bm25": 1})], snippet=30)
        candidate = result.candidates[0]
        assert candidate.title == "Wire transfer limits"
        assert len(candidate.snippet) == 30

    def test_a_fused_id_the_corpus_does_not_have_is_shown_rather_than_crashing(self):
        result, _, _ = run([FakeFused("ghost", 0.01, {"dense": 3})])
        assert result.candidates[0].title == "(not in the corpus)"


class TestTheQueryAndItsCost:
    def test_the_query_is_embedded_once_and_its_vector_is_what_searches(self):
        _, index, embedder = run([FakeFused("d1", 0.03, {"bm25": 1})])
        assert embedder.texts == ["what is my wire limit?"]
        assert index.seen["vector"] == [0.1, 0.2, 0.3]
        assert index.seen["query"] == "what is my wire limit?"

    def test_the_knobs_reach_the_index(self):
        _, index, _ = run([], k_bm25=7, k_dense=9, rrf_k=11)
        assert (index.seen["k_bm25"], index.seen["k_dense"], index.seen["rrf_k"]) == (7, 9, 11)

    def test_what_the_embedding_cost_is_reported_not_assumed(self):
        result, _, _ = run([FakeFused("d1", 0.03, {"bm25": 1})])
        assert (result.embedding_usd, result.embedding_tokens) == (0.000013, 7)

    def test_an_empty_query_is_refused_before_anything_is_spent(self):
        embedder = FakeEmbedder()
        with pytest.raises(ValueError, match="a query is needed"):
            inspect("   ", index=FakeIndex([]), docs=DOCS, embedder=embedder)
        assert embedder.texts is None


def test_a_missing_sibling_checkout_is_a_sentence_not_an_import_error(monkeypatch):
    """The dashboard runs without policygate-rag; this page just says so."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("policygate_rag"):
            raise ImportError("No module named 'policygate_rag'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ResourcesUnavailable, match="not importable"):
        resources()
