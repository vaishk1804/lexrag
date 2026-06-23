"""
Final held-out evaluation of the fine-tuned embedding model against
CUAD's TEST split (test_qa.json) - data the model has never seen during
fine-tuning, not eveb as a validation slice.

Compare base model vs fine-tuned model on a simple retieval metric:
for each test question, does the model's top-1 nearest neighbor (by
cosine similarity) match the correct ground-truth context, out of a pool of candidate contexts built from the same test set.

Usage:
  python -m src.evaluation.eval_finetuned_embeddings
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

SEED = 42
random.seed(SEED)

TEST_QA_PATH = "./data/cuad_eval/test_qa.json"
CONTRACTS_DIR="./data/cuad"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINETUNED_MODEL_PATH = "./models/lexrag-embeddings"

def load_test_pairs(qa_path: str, contracts_dir: str, max_pairs: int = 200) -> List[Tuple[str,str]]:
  """
  Load (question, ground_truth_context) pairs from the held-out test set.
  Capped at max-pairs for evaluation speed - this is a held-out
  correctness check, not exhaustive benchmarking.
  """

  with open(qa_path, encoding="utf-8") as f:
    qa_pairs = json.load(f)

  contracts_dir_path = Path(contracts_dir)
  pairs=[]

  for item in qa_pairs:
    question = item.get("question","").strip()
    ground_truth = item.get("ground_truth","").strip()
    contract_title = item.get("contract_title","")

    if not question or not ground_truth:
      continue

    safe_title = "".join(
      c if c.isalnum() or c in "._- " else "_" for c in contract_title
    )
    matches = list(contracts_dir_path.glob(f"{safe_title[:80]}*.txt"))

    if matches:
      contract_text = matches[0].read_text(encoding="utf-8", errors="ignore")
      idx = contract_text.find(ground_truth[:50])
      if idx >= 0:
        start = max(0,idx-128)
        end = min(len(contract_text),idx + len(ground_truth)+384)
        context = contract_text[start:end].strip()
      else:
        context = ground_truth
    
    else:
      context = ground_truth
    
    if len(context) > 20:
      pairs.append((question,context))
  
  random.shuffle(pairs)
  return pairs[:max_pairs]

def evaluate_top1_accuracy(model_path: str, pairs: List[Tuple[str,str]]) -> float:
  """
  For each question, embed it and find which context (among all
  contexts in the pair list) is closest by cosine similarity.
  Return the fraction where correct context was ranked #1
  """
  model = SentenceTransformer(model_path)

  questions =[p[0] for p in pairs]
  contexts = [p[1] for p in pairs]

  q_embeddings = model.encode(questions, normalize_embeddings=True, show_progress_bar=True)
  c_embeddings = model.encode(contexts, normalize_embeddings=True, show_progress_bar=True)

  correct = 0
  for i, q_emb in enumerate(q_embeddings):
    similarities = c_embeddings @ q_emb   # dot product == cosine sim (normalized)
    top1_idx = int(np.argmax(similarities))
    if top1_idx == i:
      correct+=1
  
  return correct / len(pairs)

def main():
  print("Loading held-out test pairs (never used in training)...")
  pairs = load_test_pairs(TEST_QA_PATH, CONTRACTS_DIR, max_pairs=200)
  print(f"Evaluating on {len(pairs)} held-out test pairs\n")
  print("--- Base model (all-MiniLM-L6-v2) ---")
  base_acc = evaluate_top1_accuracy(BASE_MODEL, pairs)
  print(f"Top-1 accuracy: {base_acc:.4f}\n")
 
  print("--- Fine-tuned model ---")
  finetuned_acc = evaluate_top1_accuracy(FINETUNED_MODEL_PATH, pairs)
  print(f"Top-1 accuracy: {finetuned_acc:.4f}\n")
 
  delta = finetuned_acc - base_acc
  print(f"=== Held-out test set comparison ===")
  print(f"Base model      : {base_acc:.4f}")
  print(f"Fine-tuned model: {finetuned_acc:.4f}")
  print(f"Delta           : {delta:+.4f}")
 
 
if __name__ == "__main__":
  main()