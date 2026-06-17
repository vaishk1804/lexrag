"""
Embedding generation for legal contract chunks.

Uses sentence-transformers locally - no API cost, runs on CPU.
Model: all-MiniLM-L6-v2 (384-dimensional embeddings)

Why this model:
 - Fast on CPU (import for indexing 500+ contracts)
 - Strong semantic similarity performance on legal text
 - Free, no API key needed for indexing

The model cache (_model_cache ensures we load the model once per process.
Loading a transformer model takes 2-3 seconds - reloading it for every batch would make ingestion 10X slower.)

For fine-tuning, the fine-tuned model is loaded via the same interface by passing model_name="./models/lexrag-embeddings".
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Cache loaded models by name - avoids reloading on every call
_model_cache = {}

def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SentenceTransformer:
  """
  Load and cache a SentenceTransformer model.
  
  Args:
      model_name: HuggingFace model name or local path to fine-tuned model

  Returns:
      Loaded SentenceTransformer model
  """
  if model_name not in _model_cache:
    print(f"Loading embedding mode : {model_name}")
    _model_cache[model_name]=SentenceTransformer(model_name)
  return _model_cache[model_name]

def embed_texts(
    texts: List[str],
    model_name:str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress:bool=True,
) -> np.ndarray:
  """
  Generate embeddings for a list of texts.

  Args:
      texts: List of strings to embed
      model_name: HuggingFace model name or local path
      batch_size: Number of texts to embed at once
          Larger = faster but more RAM
      show_progress: Show tqdm progress bar during encoding
  
  Returns:
      numpy array of shape(len(texts),384)
      L2-normalized so dot product == cosine similarity
  """
  if not texts:
    raise ValueError("texts list must not be empty")
  if not all(isinstance(t,str) and t.strip() for t in texts):
    raise ValueError("all items in texts must be non-empt, non-whitespace strings")
  
  model=get_embedding_model(model_name)
  embeddings = model.encode(texts,
                            batch_size=batch_size,
                            show_progress_bar=show_progress,
                            normalize_embeddings=True,    # L2 normalize - dot product becomes cosine similarity
                            convert_to_numpy=True,)
  return embeddings

def embed_query(
    query:str,
    model_name:str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
  """
  Embed a single query string.

  Seperate from embed_texts because queries are always single strings and dont need batching or progress bars.

  Returns:
      1D numpy array of shape (384,)
  """
  if not isinstance(query,str) or not query.strip():
    raise ValueError("query must be a non-empty string")
  
  model=get_embedding_model(model_name)
  return model.encode(
    [query],
    normalize_embeddings=True,
    convert_to_numpy=True,
  )[0]
