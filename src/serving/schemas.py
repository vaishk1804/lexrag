"""Pydantic request/response schemas for the LexRAG FastAPI service."""

from typing import List, Optional, Literal

from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
  question: str = Field(..., min_length=5, max_length=500,
                        description="Natural language question about a contract")
  retrieval_mode:Literal["hybrid","dense","bm25"] = Field(default="hybrid", description="hybrid | dense | bm25")

  class Config:
    json_schema_extra = {
      "example": {
        "question": "What are the termination conditions in this contract?",
        "retrieval_mode": "hybrid",
      }
    }

class ContextChunk(BaseModel):
  text: str
  source: str
  chunk_index: int
  relevance_score: float

class QueryResponse(BaseModel):
  question: str
  answer: str
  retrieved_chunks: List[ContextChunk]
  retrieval_mode: str
  latency_ms: float

class HealthResponse(BaseModel):
  status: str
  index_size: int
  embedding_model: str
  llm_provider: str
  llm_model: str