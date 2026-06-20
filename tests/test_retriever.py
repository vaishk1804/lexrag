"""
Tests for the hybrid retriever.

These require a built index (ChromaDB + BM25) to exist at the paths
defined in .env. Run the ingestion pipeline first if these fail with FileNotFoundError.

Covers:
- RRF fusion logic in insolation (no index needed)
- HybridRetriever against the read indexed corpus
- Edge cases - empty query, very short query, very long query, query with no relevant answer in the corpus
"""

import pytest

from src.retrieval.retriever import HybridRetriever, reciprocal_rank_fusion

# --- RRF fusion logic ( no index required) -----------------------

class TestReciprocalRankFusion:
  def test_documents_in_both_lists_rank_higher(self):
    """A doc appearing in both lists should outrank a doc in only one."""
    dense = [("A","text-a",{}),("B","text-b",{}),("C","text-c",{})]
    bm25 =[("B","text-b",{}),("A","text-a",{}),("D","text-d",{})]

    fused = reciprocal_rank_fusion(dense, bm25, top_k=4)
    fused_ids = [r[0] for r in fused]

    assert fused_ids[0] in {"A","B"}
    assert fused_ids[1] in {"A","B"}
    assert fused_ids[2] in {"C","D"}
    assert fused_ids[3] in {"C","D"}
  
  def test_top_k_respected(self):
    dense = [(f"id{i}","t",{}) for i in range(5)]
    bm25 = [(f"id{i}","t",{}) for i in range(4,-1,-1)]

    fused = reciprocal_rank_fusion(dense,bm25,top_k=3)
    assert len(fused) == 3

  def test_doc_only_in_dense_still_included(self):
    """A doc that only appears in dense results should still be retrievable."""
    dense = [("A","text-a",{})]
    bm25 = [("B","text-b",{})]

    fused = reciprocal_rank_fusion(dense, bm25, top_k=2)
    fused_ids = [r[0] for r in fused]
    assert "A" in fused_ids
    assert "B" in fused_ids

  def test_rrf_score_matches_hand_calculation(self):
    """
    Verify against a known hand-calculated example.
    """
    dense = [("A","t",{}),("B","t",{})]
    bm25 = [("B","t",{}),("A","t",{})]

    fused = reciprocal_rank_fusion(dense,bm25, k=60, top_k=2)
    scores = {doc_id: score for doc_id, _, _, score in fused}

    expected = 1 /61 + 1/62
    assert abs(scores["A"] - expected) < 1e-9
    assert abs(scores["B"] - expected) < 1e-9

  def test_empty_lists_return_empty(self):
    fused = reciprocal_rank_fusion([],[], top_k=5)
    assert fused == []

# --- HybridRetriever against the real index

@pytest.fixture(scope="module")
def retriever():
  """
  Build one retriever instance for all tests in this module.
  Requires the ingestion pipeline to have been run already.
  """
  try:
    return HybridRetriever(retrieval_k=10, top_k=5)
  except FileNotFoundError:
    pytest.skip("Index not found. Run: python -m src.ingestion.pipeline")
  
class TestHybridRetrieverBasic:

  def test_dense_mode_returns_results(self, retriever):
    chunks = retriever.retrieve("What are the termination conditions?", mode="dense")
    assert len(chunks) > 0
    assert len(chunks) <= retriever.top_k
  
  def test_bm25_mode_returns_results(self, retriever):
    chunks = retriever.retrieve("What are the termination conditions?", mode="bm25")
    assert len(chunks) > 0
  
  def test_hybrid_mode_returns_results(self, retriever):
    chunks = retriever.retrieve("What are the termination conditions?", mode="hybrid")
    assert len(chunks) > 0

  def test_invalid_mode_raises(self, retriever):
    with pytest.raises(ValueError):
      retriever.retrieve("test query",mode="not_a_real_mode")

  def test_results_have_required_fields(self, retriever):
    chunks = retriever.retrieve("What is the governing law?", mode="hybrid")
    for chunk in chunks:
      assert hasattr(chunk, "text")
      assert hasattr(chunk, "metadata")
      assert hasattr(chunk, "score")
      assert hasattr(chunk, "rank")
      assert len(chunk.text) > 0

class TestRetrieverEdgeCases:

  def test_empty_query_raises(self, retriever):
    with pytest.raises(ValueError):
      retriever.retrieve("",mode="hybrid")
  
  def test_whitespace_only_query_raises(self, retriever):
    with pytest.raises(ValueError):
      retriever.retrieve("       ",mode="hybrid")

  def test_very_short_query_does_not_crash(self, retriever):
    """A single word should still return something, even if low quality."""
    chunks = retriever.retrieve("terminate",mode="hybrid")
    assert isinstance(chunks, list)

  def test_very_long_query_does_not_crash(self, retriever):
    """A long, rambling query should not break retrieval."""
    long_query = (
    "I would like to know in great detail what happens regarding "
    "termination of this agreement including notice periods, "
    "cure periods, material breach definitions, and what each "
    "party is required to do upon termination including any "
    "surviving obligations that continue after the agreement ends"
    )
    chunks = retriever.retrieve(long_query, mode="hybrid")
    assert isinstance(chunks, list)
    assert len(chunks) > 0
  
  def test_nonsense_query_does_not_crash(self, retriever):
    """A query with no real answer in the corpus should not error out."""
    chunks = retriever.retrieve("purple elephant spaceship banana", mode="hybrid")
    assert isinstance(chunks, list)
    # doesn't assert relevance, just that it doesn't crash
    # and returns the requested number of results (or fewer)
    assert len(chunks) <= retriever.top_k

  def test_query_with_special_characters_does_not_crash(self,retriever):
    chunks = retriever.retrieve("What about §10.5(b) — termination?", mode="hybrid")
    assert isinstance(chunks, list)
  
  def test_dense_and_bm25_can_return_different_results(self,retriever):
    """
    Sanity check that dense and BM25 aren't just returning identical results.
    """
    dense_chunks = retriever.retrieve("What is the governing law?", mode="dense")
    bm25_chunks = retriever.retrieve("What is the governing law?", mode="bm25")

    dense_sources = {c.metadata.get("source") for c in dense_chunks}
    bm25_sources = {c.metadata.get("source") for c in bm25_chunks}

    assert dense_sources != bm25_sources

class TestRetrieverConfig:

  def test_top_k_limits_results(self):
    try:
      r = HybridRetriever(retrieval_k=10, top_k=2)
    except FileNotFoundError:
       pytest.skip("Index not found.")
    chunks = r.retrieve("What are the termination conditions?", mode="hybrid")
    assert len(chunks) <= 2

  def test_retrieval_k_must_be_at_least_top_k(self):
    """
    retrieval_k should be >= top_k for fusion to make sense —
    not strictly enforced in code, but worth checking it doesn't crash.
    """
    try:
      r = HybridRetriever(retrieval_k=3, top_k=5)
    except FileNotFoundError:
      pytest.skip("Index not found.")
    chunks = r.retrieve("What are the termination conditions?", mode="hybrid")
    assert isinstance(chunks, list)