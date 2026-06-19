"""
Hybrid retrievar: BM25 (sparse) + ChromaDB dense + Reciprocal Rank Fusion

RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each retrieval list i where k=60 is a constant that dampens the impact of ver high ranks.

"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from src.ingestion.embedder import embed_query
from src.ingestion.indexer import (
  BM25_INDEX_PATH,
  CHROMA_PERSIST_DIR,
  COLLECTION_NAME,
  tokenize_for_bm25,
)

class RetrievedChunk:
  """A single retrieved chunk with its fused score and provenance"""

  def __init__(self, text:str, metadata: dict,score:float , rank:int , source:str):
    self.text = text
    self.metadata = metadata
    self.score = score
    self.rank = rank
    self.source = source    # "bm25", "dense" or "hybrid"

  def __repr__(self):
    return (
      f"RetrievedChunk(rank={self.rank}, score={self.score:.4f}, "
      f"source={self.source}, text={self.text[:60]}...)"
    )
  
def reciprocal_rank_fusion(
    dense_results: List[Tuple[str,str,dict]],
    bm25_results: List[Tuple[str,str,dict]],
    k:int = 60,
    top_k: int = 5,
) -> List[Tuple[str,str,dict,float]]:
  """
  Fuse two ranked lists using Reciprocal Rank Fusion.

  Each result list is [(id,text,metadata),...] ordered best-first.
  A document's RRF score is the sum of 1/(k+rank+1) across ever list it appears in. Rank is 0-indexed so +1 avoids division asymmetry.

  Args:
      dense_results: Ranked list from dense retrieval
      bm25_results: Ranked list from BM25 retrieval
      k: RRF constant - higher k flatterns the score curve reducing the dominance of top-ranked results. 60 is standard value from original RRF paper.
      top_k: Number of fused results to return
  """
  scores: Dict[str,float]={}
  docs: Dict[str,Tuple[str,dict]] = {}

  for rank, (doc_id, text, meta) in enumerate(dense_results):
    scores[doc_id]=scores.get(doc_id,0.0)+1.0/(k+rank+1)
    docs[doc_id] = (text,meta)

  for rank, (doc_id,text,meta) in enumerate(bm25_results):
    scores[doc_id] = scores.get(doc_id,0.0)+1.0/(k+rank+1)
    docs[doc_id]=(text,meta)

  sorted_ids = sorted(scores.keys(), key=lambda d:scores[d], reverse=True)[:top_k]
  return [(doc_id,docs[doc_id][0],docs[doc_id][1], scores[doc_id]) for doc_id in sorted_ids]

class HybridRetriever:
  """
  Retrieves relevant contract chunks using hybrid BM25 + dense search
  fused with Reciprocal Rank Fusion.
  """

  def __init__(self,retrieval_k:int=10,top_k:int=5,rrf_k:int=60):
    """
    Args:
        retrieval_k: How many  results to fetch from each retriever.
        top_k: Final number of chunks returned after fusion
        rrf_k: RRF dampening constant
    """
    self.retrieval_k=retrieval_k
    self.top_k=top_k
    self.rrf_k=rrf_k

    self.chroma_client = chromadb.PersistentClient(
      path=CHROMA_PERSIST_DIR,
      settings=Settings(anonymized_telemetry=False)
    )

    self.collection = self.chroma_client.get_collection(COLLECTION_NAME)

    if not Path(BM25_INDEX_PATH).exists():
      raise FileNotFoundError(
        f"BM25 index not found at {BM25_INDEX_PATH}. "
        "Run ingestion pipeline first."
      )
    
    with open(BM25_INDEX_PATH,"rb") as f:
      data=pickle.load(f)
      self.bm25: BM25Okapi = data["bm25"]
      self.bm25_chunks = data["chunks"]

    print(
      f"HybridRetriever ready: {self.collection.count():,} dense vectors, "
      f"{len(self.bm25_chunks):,}BM25 docs"
      ) 
    
    if self.collection.count() != len(self.bm25_chunks):
      print(
        f"\n Warning index mismatch detected!\n"
        f"   ChromaDB has {self.collection.count():,} vectors but "
                f"BM25 has {len(self.bm25_chunks):,} documents.\n"
                f"   Re-run the ingestion pipeline to fix:\n"
                f"   python -m src.ingestion.pipeline --chunk-size 512\n"
      )
    
  def _dense_search(self,query:str) -> List[Tuple[str,str,dict]]:
    query_embedding = embed_query(query).tolist()
    results = self.collection.query(
      query_embeddings=[query_embedding],
      n_results=self.retrieval_k,
      include=["documents","metadatas","distances"],
    )
    output=[]
    for i in range(len(results["ids"][0])):
      doc_id = results["ids"][0][i]
      text=results["documents"][0][i]
      meta=results["metadatas"][0][i]
      output.append((doc_id,text,meta))
    return output
  def _bm25_search(self, query: str) -> List[Tuple[str, str, dict]]:
        tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.retrieval_k]

        output = []
        for idx in top_indices:
            chunk = self.bm25_chunks[idx]
            chunk_id = f"{chunk.metadata.get('source', 'doc')}_{chunk.metadata.get('chunk_index', idx)}"
            output.append((chunk_id, chunk.text, chunk.metadata))
        return output
  
  def retrieve(self,query:str,mode: str = "hybrid") -> List[RetrievedChunk]:
    """
    Retrieve relevant chunks for a query.

    Args:
        query: Natural language question about a contract
        mode: "hybrid" (default) | "dense" | "bm25"
    Returns:
        List of RetrievedChunk, best first.
    """
    if not query or not query.strip():
      raise ValueError("query must be a non-empty string")
    
    if mode == "dense":
      results = self._dense_search(query)
      fused = [
        (doc_id,text,meta,1.0/(i+1))
        for i, (doc_id,text,meta) in enumerate(results)
      ][:self.top_k]
    
    elif mode=="bm25":
      results = self._bm25_search(query)
      fused = [
        (doc_id,text,meta,1.0/(i+1))
        for i, (doc_id,text,meta) in enumerate(results)
      ][:self.top_k]

    elif mode == "hybrid":
      dense_results = self._dense_search(query)
      bm25_results = self._bm25_search(query)
      fused = reciprocal_rank_fusion(
        dense_results, bm25_results,k=self.rrf_k, top_k=self.top_k
      )

    else:
      raise ValueError(f"Unknown mode: {mode}. Use 'hybrid','dense', or 'bm25'.")
    
    return [
      RetrievedChunk(text=text, metadata=meta,score=score, rank=rank,source=mode)
      for rank, (doc_id,text, meta, score) in enumerate(fused)
    ]
