"""
Dual idexing: ChromaDB (dense vectors) + BM25 (sparse keyword).

Both indexes are built from the same list of Chunk objects and kept in sync.
At query time, both are queried independently and their results are fused using Reciprocal Rank Fusion

ChromDB - persistent vector store
    - Stored embeddings + original text + metadata
    - Queried by cosine similarity (HNSW approximate nearest neighbour)
    - Persists to disk at CHROMA_PERSIST_DIR

BM25 - sparse keyword index
    - Stores tokenized text ( whitespace split, lowercased, punctuation stripped)
    - Queried by TF-IDF-style scoring (BM25Okapi formula)
    - Serialized to disk as a pickle file (full rebuild on each run)
    - exact term matching for domain vocabulary which dense embeddings may generalize over

Two indexes
    - Dense retrieval captures semantic similarity but missed exact terms.
    - BM25 captures exact terms but missed paraphrasing.
    - Hybrid fusion captures both - this is the core retrieval insight.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.ingestion.embedder import embed_texts

# Storage paths - read from .env, with sensible defaults
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH","./data/bm25_index.pkl")
COLLECTION_NAME = "legal_contracts"

def tokenize_for_bm25(text: str) -> List[str]:
  """
  Whitespace tokenization with punctuation stripping for BM25.

  Strips leading/trailing punctuation from each token. Keeps legal terms intact as single tokens - no subword splitting.
  """
  tokens = text.lower().split()
  stripped = [
    token.strip(".,;:!?\"'()[]{}")
    for token in tokens
  ]
  # Filter out tokens that became empty after stripping
  return [t for t in stripped if t]

def _make_chunk_id(chunk: Chunk, fallback_index: int) -> str:
  """
  Build a unique chunk ID for ChromaDB.

  Uses source filename + chunk_index + a hash of the full source path.
  The path hash handles cases where two contracts share the same filename but come from different directories - without it they would collide in ChromaDB and silently overwrite each other.

  Args:
      chunk: Chunk object with metadata
      fallback_index: Used if chunk_index is missing from metadata

  Returns:
      String ID unique to this chunk across the entire corpus
  """
  source = chunk.metadata.get("source","doc")
  chunk_index = chunk.metadata.get("chunk_index", fallback_index)
  # Hash the full path to disambiguate same-name files in different dirs
  source_path = chunk.metadata.get("source_path",source)
  path_hash = hash(source_path) % 100000
  return f"{source}_{chunk_index}_{path_hash}"

class DualIndexer:
  """
  Manages ChromaDB (dense) and BM25 (sparse) indexes in parallel.
  Both are built from the same list of Chunk objects.
  """

  def __init__(
      self,
      embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
  ):
    """
    Args:
        embedding_model: HuggingFace model name or path to fine-tuned model.
    """
    self.embedding_model = embedding_model

    # Initialize ChromaDB persistent client
    self.chroma_client = chromadb.PersistentClient(
      path= CHROMA_PERSIST_DIR,
      settings= Settings(anonymized_telemetry=False),
    )

    # Get or create the collection
    # hnsw:space=cosine means similarity is measured by cosine distance
    self.collection = self.chroma_client.get_or_create_collection(
      name=COLLECTION_NAME,
      metadata={"hnsw:space":"cosine"},
    )

    # BM25 state - loaded from disk if available
    self.bm25: Optional[BM25Okapi] = None
    self.bm25_chunks: List[Chunk] = []
    self._load_bm25_if_exists()

  def _load_bm25_if_exists(self):
    """Load BM25 index from disk if it exists."""
    if Path(BM25_INDEX_PATH).exists():
      with open(BM25_INDEX_PATH,"rb") as f:
        data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.bm25_chunks = data["chunks"]
      print(f"Loaded BM25 index: {len(self.bm25_chunks)} chunks")
  
  def _save_bm25(self):
    """Serialize BM25 index to disk."""
    Path(BM25_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH,"wb") as f:
      pickle.dump(
        {"bm25": self.bm25, "chunks": self.bm25_chunks}, f
      )
    print(f"Saved BM25 index to {BM25_INDEX_PATH}")

  def index_chunks(self, chunks: List[Chunk], batch_size: int = 256):
    """
    Index a list of chunks into both ChromaDB amd BM25.

    Args:
        chunks: List of Chunk objects from the chunker
        batch_size: ChromaDB upsert batch size.

    Note: Full rebuild on every run.
      BM25 does not support incremental updates natively.
      ChromaDB upsert is safe to re-run with the same IDs.
    """
    if not chunks:
      raise ValueError("chunks list must not be empty")
    
    texts = [c.text for c in chunks]
    chunk_ids = [_make_chunk_id(c,i) for i,c in enumerate(chunks)]

    # --- ChromaDB (dense) ---
    print(f"Generating embeddings for {len(chunks)} chunks....")
    embeddings = embed_texts(
      texts,
      model_name=self.embedding_model,
      show_progress=True,
    )

    print(f"Upserting into ChromaDB in batches of {batch_size}...")
    for i in range(0,len(chunks),batch_size):
      batch_end = min(i+batch_size,len(chunks))
      self.collection.upsert(
        documents=texts[i:batch_end],
        ids=chunk_ids[i:batch_end],
        embeddings=embeddings[i:batch_end].tolist(),
        metadatas=[c.metadata for c in chunks[i:batch_end]],
      )
    print(f"ChromaDB: {self.collection.count()} vectors indexed")

    # --- BM25 (sparse) ---
    print("Building BM25 index...")
    tokenized = [tokenize_for_bm25(t) for t in texts]
    self.bm25 = BM25Okapi(tokenized)
    self.bm25_chunks=chunks
    self._save_bm25()
    print(f"BM25: {len(chunks)} documents indexed")

  def get_collection_count(self) -> int:
    """Return number of vectors currently in ChromaDB."""
    return self.collection.count()