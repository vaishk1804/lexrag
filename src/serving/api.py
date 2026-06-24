"""
FastAPI application for LexRAG.

Endpoints: 
  GET /health - liveness check + index/model info
  POST /query - main RAG query endpoint

Uses whichever embedding model and LLM provider are configured via
.env (EMBEDDING_MODEL, LLM_PROVIDER, LLM_MODEL) - this makes it
agnostic to whether the index was built with the base or fine-tuned
embedding model, no code change needed when swapping.

Run locally:
  uvicorn src.serving.api:app --reload --port 8000

API docs:
  https://localhost:8000/docs
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from src.evaluation.llm_client import get_llm_client, get_model_name, get_provider
from src.retrieval.retriever import HybridRetriever
from src.serving.schemas import ContextChunk, HealthResponse, QueryRequest, QueryResponse

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lexrag")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

retriever: Optional[HybridRetriever] = None
llm_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
  """Initialize the retriever and LLM client once at startup"""
  global retriever, llm_client
  logger.info(f"Starting LexRAG API - embedding model: {EMBEDDING_MODEL}")
  retriever = HybridRetriever(
    retrieval_k=int(os.getenv("RETRIEVAL_K","10")),
    top_k= int(os.getenv("TOP_K","5")),
    embedding_model=EMBEDDING_MODEL,
  )
  llm_client = get_llm_client()
  logger.info("LexRAG API ready")
  yield
  logger.info("Shutting down LexRAG API")

app = FastAPI(
  title="LexRAG - Legal Contract Q&A",
  description="Hybrid retrieval RAG system for commercial contracts (CUAD dataset)",
  version="1.0.0",
  lifespan=lifespan,
)

def generate_answer(question: str, contexts: list[str]) -> str:
  """Generate an answer grounded in retrieved context, via whichever LLM provider is active."""
  context_text = "\n\n---\n\n".join(contexts)
  prompt = f"""You are a legal document analyst. Answer the question based ONLY on the provided contract excerpts. If the answer is not found in the excerpts, say "This information is not found in the provided contract sections."
Be specific and reference relevant contract language when possible.

CONTRACT EXCERPTS:
{context_text}

QUESTION: {question}

ANSWER:"""
  
  response = llm_client.chat.completions.create(
    model=get_model_name(),
    messages=[{"role":"user","content": prompt}],
    temperature=0,
    max_tokens=512,
  )
  return response.choices[0].message.content.strip()

@app.get("/health", response_model=HealthResponse)
async def health():
  """Liveness and readiness check."""
  if retriever is None:
    raise HTTPException(status_code=503, detail="Retriever not initialized")
  return HealthResponse(
    status="ok",
    index_size=retriever.collection.count(),
    embedding_model=EMBEDDING_MODEL,
    llm_provider=get_provider(),
    llm_model=get_model_name(),
  )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
  """
  Answer a natural language question about legal contracts.

  Retrieves relevant contract chunks using hybrid BM25 + dense search,
  then generates an answer grounded in those chunks.
  """
  if retriever is None or llm_client is None:
    raise HTTPException(status_Code=503, detail="Service not ready")
  
  t0 = time.time()

  chunks = retriever.retrieve(request.question, mode= request.retrieval_mode)
  context_texts = [c.text for c in chunks]
  answer = generate_answer(request.question, context_texts)

  latency_ms = (time.time() - t0) * 1000

  return QueryResponse(
    question=request.question,
    answer=answer,
    retrieved_chunks=[
      ContextChunk(
        text=c.text,
        source=c.metadata.get("source","unknown"),
        chunk_index=c.metadata.get("chunk_index",-1),
        relevance_score=round(c.score,4),
      )
      for c in chunks
    ],
    retrieval_mode=request.retrieval_mode,
    latency_ms=round(latency_ms,1),
  )