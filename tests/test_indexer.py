"""
Tests for embedder and indexer.

Smoke tests that verify:
- Embedder produces correct shape and normalized vectors
- BM25 tokenizer handles punctuation correctly
- DualIndexer builds both stores from a small chunk set
- Chunk IDs are unique (no collisions)
- BM25 returns relevant results for a legal query
"""

import numpy as np
import pytest

from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import embed_texts, embed_query
from src.ingestion.indexer import DualIndexer, tokenize_for_bm25


# Small realistic contract — enough to produce a few chunks
SAMPLE_CONTRACT = """
SERVICES AGREEMENT

This Services Agreement is entered into as of January 1, 2024,
between Acme Corp ("Client") and Beta LLC ("Provider").

ARTICLE 1 - DEFINITIONS
1.1 Services means the professional services described herein.
1.2 Confidential Information means any non-public information.

ARTICLE 2 - TERMINATION
2.1 Either party may terminate this Agreement with 30 days written notice.
2.2 Client may terminate immediately upon Provider material breach.
2.3 Upon termination, all outstanding fees become immediately due and payable.

ARTICLE 3 - INDEMNIFICATION
3.1 Each party shall indemnify and hold harmless the other party from claims
arising from its own negligence or willful misconduct.

ARTICLE 4 - GOVERNING LAW
4.1 This Agreement shall be governed by the laws of the State of Delaware.
4.2 Disputes shall be resolved through binding arbitration in New York.

ARTICLE 5 - LIMITATION OF LIABILITY
5.1 In no event shall either party be liable for indirect or consequential damages.
5.2 Total liability shall not exceed fees paid in the preceding 12 months.
""" * 3


# ── Embedder tests ────────────────────────────────────────────────────────────

class TestEmbedder:

    def test_embed_query_shape(self):
        """Single query should produce a 384-dimensional vector."""
        vec = embed_query("What are the termination conditions?")
        assert vec.shape == (384,), f"Expected (384,) got {vec.shape}"

    def test_embed_query_normalized(self):
        """Vector should be L2-normalized — norm should be ~1.0."""
        vec = embed_query("What are the termination conditions?")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-4, f"Expected norm ~1.0, got {norm}"

    def test_embed_texts_shape(self):
        """Batch embedding should return one vector per text."""
        texts = [
            "What are the termination conditions?",
            "Who is responsible for indemnification?",
            "What is the governing law?",
        ]
        embeddings = embed_texts(texts, show_progress=False)
        assert embeddings.shape == (3, 384)

    def test_semantic_similarity_ordering(self):
        """
        Semantically similar queries should have higher cosine similarity
        than unrelated ones.
        'termination' and 'contract ended' should be closer than
        'termination' and 'payment schedule'.
        """
        v1 = embed_query("What are the termination conditions?")
        v2 = embed_query("How can this contract be ended?")
        v3 = embed_query("What is the payment schedule?")

        sim_related = np.dot(v1, v2)      # should be high
        sim_unrelated = np.dot(v1, v3)    # should be lower

        assert sim_related > sim_unrelated, (
            f"Expected termination/ended ({sim_related:.3f}) > "
            f"termination/payment ({sim_unrelated:.3f})"
        )

    def test_embed_texts_rejects_empty_list(self):
        with pytest.raises(ValueError):
            embed_texts([])

    def test_embed_texts_rejects_whitespace_only(self):
        with pytest.raises(ValueError):
            embed_texts(["     "])

    def test_embed_query_rejects_empty_string(self):
        with pytest.raises(ValueError):
            embed_query("")

    def test_model_cache_returns_same_object(self):
        """Model should be loaded once and cached — same object on second call."""
        from src.ingestion.embedder import get_embedding_model
        m1 = get_embedding_model()
        m2 = get_embedding_model()
        assert m1 is m2


# ── BM25 tokenizer tests ──────────────────────────────────────────────────────

class TestTokenizer:

    def test_punctuation_stripped(self):
        """Trailing punctuation should be removed."""
        tokens = tokenize_for_bm25("The licensee.")
        assert "licensee" in tokens
        assert "licensee." not in tokens

    def test_lowercased(self):
        """All tokens should be lowercase."""
        tokens = tokenize_for_bm25("ARTICLE 1 DEFINITIONS")
        assert all(t == t.lower() for t in tokens)

    def test_legal_terms_kept_intact(self):
        """Legal terms should not be split into subwords."""
        tokens = tokenize_for_bm25("indemnification clause force majeure")
        assert "indemnification" in tokens
        assert "force" in tokens
        assert "majeure" in tokens

    def test_empty_tokens_filtered(self):
        """Tokens that become empty after stripping should be removed."""
        tokens = tokenize_for_bm25("hello . world")
        assert "" not in tokens
        assert "." not in tokens

    def test_comma_stripped(self):
        tokens = tokenize_for_bm25("force majeure, beyond control.")
        assert "majeure" in tokens
        assert "majeure," not in tokens
        assert "control" in tokens
        assert "control." not in tokens


# ── DualIndexer tests ─────────────────────────────────────────────────────────

class TestDualIndexer:

    @pytest.fixture
    def indexed(self, tmp_path, monkeypatch):
        """
        Build a small index in a temp directory.
        Uses monkeypatch to redirect ChromaDB and BM25 paths
        so tests don't pollute the real data/ index.
        """
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path / "bm25.pkl"))

        # Reload indexer module so it picks up monkeypatched env vars
        import importlib
        import src.ingestion.indexer as idx_module
        importlib.reload(idx_module)

        chunks = chunk_document(
            SAMPLE_CONTRACT,
            metadata={"source": "test.txt", "source_path": "data/cuad/test.txt"},
            chunk_size=512,
        )

        indexer = idx_module.DualIndexer()
        indexer.index_chunks(chunks)
        return indexer, chunks

    def test_chromadb_count_matches_chunks(self, indexed):
        indexer, chunks = indexed
        assert indexer.get_collection_count() == len(chunks)

    def test_bm25_count_matches_chunks(self, indexed):
        indexer, chunks = indexed
        assert len(indexer.bm25_chunks) == len(chunks)

    def test_chunk_ids_are_unique(self, indexed):
        """No two chunks should produce the same ID."""
        from src.ingestion.indexer import _make_chunk_id
        indexer, chunks = indexed
        ids = [_make_chunk_id(c, i) for i, c in enumerate(chunks)]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected"

    def test_bm25_returns_termination_chunks(self, indexed):
        """
        BM25 query for 'termination' should return chunks that
        actually mention termination — not unrelated clauses.
        """
        indexer, chunks = indexed
        tokens = tokenize_for_bm25("termination conditions")
        scores = indexer.bm25.get_scores(tokens)
        top_idx = scores.argsort()[-1]  # single best result
        top_text = indexer.bm25_chunks[top_idx].text.lower()
        assert "terminat" in top_text, (
            f"Top BM25 result for 'termination' doesn't mention termination:\n{top_text}"
        )

    def test_index_chunks_rejects_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path / "bm25.pkl"))

        import importlib
        import src.ingestion.indexer as idx_module
        importlib.reload(idx_module)

        indexer = idx_module.DualIndexer()
        with pytest.raises(ValueError):
            indexer.index_chunks([])