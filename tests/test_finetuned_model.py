"""
Tests verifying the fine-tuned embedding model saved and loads correctly.
Skips automatically if the model hasn't been trained yet.
"""

from pathlib import Path
import pytest

FINETUNED_MODEL_PATH = "./models/lexrag-embeddings"

@pytest.fixture(scope="module")
def finetuned_model():
  if not Path(FINETUNED_MODEL_PATH).exists():
    pytest.skip(
      f"Fine-tuned model not found at {FINETUNED_MODEL_PATH}. "
      "Run: python -m src.training.finetune_embeddings"
    )
  from sentence_transformers import SentenceTransformer
  return SentenceTransformer(FINETUNED_MODEL_PATH)

def test_finetuned_model_laods(finetuned_model):
  assert finetuned_model is not None

def test_finetuned_model_produces_correct_embedding_shape(finetuned_model):
  vec = finetuned_model.encode("What are the termination conditions?")
  assert vec.shape == (384,)

def test_finetuned_model_embeddings_are_normalized_when_requested(finetuned_model):
  import numpy as np
  vec = finetuned_model.encode("test query", normalize_embeddings=True)
  norm = np.linalg.norm(vec)
  assert abs(norm -1.0) < 1e-4

def test_finetuned_model_distinguishes_different_legal_concepts(finetuned_model):
  """
  Sanity check: the fine-tuned model should still place semantically
  different legal concepts at meaningfully different distances - 
  fine-tuning shouldn't collapse all embeddings toward a single point.
  """
  import numpy as np
  v1 = finetuned_model.encode("termination conditions", normalize_embeddings=True)
  v2 = finetuned_model.encode("payment schedule", normalize_embeddings=True)
  similarity = float(np.dot(v1,v2))
  assert similarity < 0.99, "Embeddings are nearly identical - model may have collapsed"