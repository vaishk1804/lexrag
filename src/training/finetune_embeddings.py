"""Fine-tune sentence-transformers on CUAD contract Q&A pairs using PyTorch.

The default all-MiniLM-L6-v2 model was trained on general web text.
Legal contracts use domain-specific language ("indemnification","licensor", "force-majeure) that general-purpose embeddings represent 
imprecisely. Fine-tuning on real CUAD question-context pairs teaches the model that a question about "termination conditions" should be close in
embedding space to a contract chunk containing "Either party may 
terminate...", using actual legal phrasing rather than generic web text 
as the training signal.

APPROACH: Multiple Negatives Ranking (MNR) Loss
For each (question, positive_context) pair in a training batch, every 
OTHER context in that same batch is treated as a negative example - 
no explicit negative mining step is needed. The model is trained to push the embeddings of a question and its true positive context closer together (higher cosine similarity) while pushing the question's embeddings away from every other context in the batch.

Concretely, for a batch of size 4 with pairs (q1,c1), (q2,c2), (q3,c3), (q4,c4):
    - q1 should be close to c1, and far from c2, c3, c4
    - q2 should be close to c2, and far from c1, c3, c4
    - ... and so on for every pair in the batch

this is why it's called "in-batch negatives" - the negatives come free from whatever else happens to be in the same batch, rather than being deliberately selected. Larger batch sizes give more negatives per positive, which generally improves training signal (more contrast to learn from) at the cost of more memory.

Mathematically a InfoNCE-style contrastive loss - the same
family of loss function used in contrastive learning more broadly
(e.g. SimCLR for images), applied here to text pairs.

TRAINING SETUP:
  - Base model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
  - Loss: MultipleNegativesRankingLoss
  - Optimizer: AdamW with linear warmup (handled internally by
  SentenceTranformer.fit(), which wraps the standard PyTorch training loop)
  - Epochs: 3 - sufficient for domain adaptation on a corpus this size
  without overfitting to CUAD's specific phrasing
  - Evaluator: InformationRetrievalEvaluator tracks MRR@10 on a held-out
  slice of CUAD Q&A pairs after each epoch, giving a real learning curve

OUTPUT: saved to ./models/lexrag-embeddings/
"""

import json
import random
from pathlib import Path
from typing import List

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

# Reproducibiilty - same seed every run so training is comparible

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR="./models/lexrag-rmbeddings"

FINETUNE_QA_PATH="./data/cuad_eval/finetune_qa.json"
CONTRACTS_DIR="./data/cuad"

BATCH_SIZE=16
EPOCHS=3
WARMUP_RATIO=0.1
LEARNING_RATE = 2e-15
VALIDATION_FRACTION = 0.1  # 10% of fine-tuning pairs held out as validation

def load_training_pairs(qa_path: str, contracts_dir: str) -> List[InputExample]:
  """
  Build (question, positive_context) pairs from CUAD TRAIN-split
  annotations only (finetune_qa.json). Never reads from test_qa.json.

  For each Q&A pair, finds the matching contract text and extracts a 
  window of text centered on where the ground-truth answer actually appears 
  - this gives the model real contract language as the 
  positive example, not just the bare ground-truth answer string
  in isolation.
  """
  if not Path(qa_path).exists():
    raise FileNotFoundError(
      f"Fine-tuning data file not found at {qa_path}. "
      "Run scripts/download_data.py first."
    )
  
  with open(qa_path, encoding="utf-8") as f:
    qa_pairs = json.load(f)

  contracts_dir_path=Path(contracts_dir)
  examples=[]

  for item in qa_pairs:
    question = item.get("question","").strip()
    ground_truth = item.get("ground_truth","").strip()
    contract_title = item.get("contract_title","")

    if not question or not ground_truth:
      continue

    safe_title = "".join(
      c if c.isalnum() or c in ":._- " else "_" for c in contract_title
    )

    # Filenames have an index suffix from the download script,
    # so this does a prefix match rather than an exact filename match.
    matches = list(contracts_dir_path.glob(f"{safe_title[:80]}*.txt"))

    if matches:
      contract_text = matches[0].read_text(encoding="utf-8",errors="ignore")
      idx = contract_text.find(ground_truth[:50])
      if idx>=0:
        start = max(0,idx-128)
        end = min(len(contract_text),idx+len(ground_truth)+384)
        positive_context = contract_text[start:end].strip()
      else:
        positive_context = ground_truth
    else:
      positive_context = ground_truth

    if len(positive_context)>20:
      examples.append(InputExample(texts=[question,positive_context]))
  
  print(f"Build {len(examples)} fine-tuning pairs from CUAD train-split annotations")
  return examples

def build_evaluator(validation_examples: List[InputExample]) -> InformationRetrievalEvaluator:
  """
  Build an Information Retrieval evaluator from a random validation
  split of fine-tuning pairs
  """
  queries ={}
  corpus = {}
  relevant_docs = {}

  for i,example in enumerate(validation_examples):
    qid=f"q_{i}"
    did=f"d_{i}"
    queries[qid] = example.texts[0]
    corpus[did]=example.texts[1]
    relevant_docs[qid]={did}
  
  return InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="cuad-validation",
    show_progress_bar=False,
  )

def finetune(
    base_model: str = BASE_MODEL,
    output_dir:str = OUTPUT_DIR,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
):
  """
  Fine-tune a sentence-transformer on CUAD TRAIN-split Q&A pairs using
  MultipleNegativesRankingLoss. Uses a random validation split for
  tracking training progress - test_qa.json remains fully untouched
  and is reserved for the final, independent evaluation.
  """
  print(f"=== LexRAG Embedding Fine-Tuning ===")
  print(f"Base model : {base_model}")
  print(f"Output dir : {output_dir}")
  print(f"Batch size : {batch_size}")
  print(f"Epochs     : {epochs}")
  print(f"Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

  model = SentenceTransformer(base_model)

  all_examples = load_training_pairs(FINETUNE_QA_PATH, CONTRACTS_DIR)

  random.shuffle(all_examples)
  val_size = int(len(all_examples) * VALIDATION_FRACTION)
  validation_examples = all_examples[:val_size]
  train_examples = all_examples[val_size:]

  print(f"Training on {len(train_examples)} pairs")
  print(f"Validating on {len(validation_examples)} pairs (random_split, never trained on)\n")

  train_dataloader = DataLoader(train_examples, shuffle = True, batch_size=batch_size)

  train_loss = losses.MultipleNegativesRankingLoss(model=model)

  evaluator = build_evaluator(validation_examples)

  total_steps = len(train_dataloader) * epochs
  warmup_Steps = int(total_steps*WARMUP_RATIO)

  model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=epochs,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": LEARNING_RATE},
    output_path=output_dir,
    show_progress_bar=True,
    evaluation_steps=len(train_dataloader),  # evaluate once per epoch
    save_best_model=True,  # keeps the checkpoint with the best MRR@10, not just the last one
  )
  print(f"\nFine-tuning complete. Best model saved to: {output_dir}")
  print("\nTo use the fine-tuned model in the pipeline:")
  print(f"  python -m src.ingestion.pipeline --chunk-size 512 --embedding-model {output_dir}")
  print("\nFinal evaluation should run against data/cuad_eval/test_qa.json,")
  print("which was never seen during this training run:")
  print("  python -m src.evaluation.eval_finetuned_embeddings")

if __name__ == "__main__":
  import argparse
 
  parser = argparse.ArgumentParser(description="Fine-tune embedding model on CUAD")
  parser.add_argument("--base-model", default=BASE_MODEL)
  parser.add_argument("--output-dir", default=OUTPUT_DIR)
  parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
  parser.add_argument("--epochs", type=int, default=EPOCHS)
  args = parser.parse_args()
 
  finetune(
    base_model=args.base_model,
    output_dir=args.output_dir,
    batch_size=args.batch_size,
    epochs=args.epochs,
  )






