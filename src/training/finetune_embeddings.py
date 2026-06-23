"""
Fine-tune sentence-transformers on CUAD contract Q&A pairs using PyTorch.

WHY THIS EXISTS:
The default all-MiniLM-L6-v2 model was trained on general web text.
Legal contracts use domain-specific language ("indemnification",
"licensor", "force majeure") that general-purpose embeddings represent
imprecisely. Fine-tuning on real CUAD question-context pairs teaches the
model that a question about "termination conditions" should be close in
embedding space to a contract chunk containing "Either party may
terminate...", using actual legal phrasing as the training signal.

APPROACH: Multiple Negatives Ranking (MNR) Loss
For each (anchor, positive) pair in a training batch, every OTHER
positive in that same batch is treated as a negative example for every
other anchor — no explicit negative mining step is needed. The model is
trained to push the embeddings of a question (anchor) and its true
positive context closer together (higher cosine similarity) while
pushing the question's embedding away from every other context in the
batch.

Concretely, for a batch of size 4 with pairs (q1,c1), (q2,c2), (q3,c3), (q4,c4):
  - q1 should be close to c1, and far from c2, c3, c4
  - q2 should be close to c2, and far from c1, c3, c4
  - ...and so on for every pair in the batch
This is why it's called "in-batch negatives" — the negatives come free
from whatever else happens to be in the same batch, rather than being
deliberately retrieved through a separate search step.

This is mathematically an InfoNCE-style contrastive loss — the same
family of loss function used in contrastive learning more broadly
(e.g. SimCLR for images), applied here to text pairs.

TRAINING API NOTE:
sentence-transformers 3.x replaced the older SentenceTransformer.fit()
wrapper with SentenceTransformerTrainer, built directly on Hugging
Face's Trainer. This script uses the modern Trainer-based API:
  - Training data is a datasets.Dataset with "anchor"/"positive" columns
    (MultipleNegativesRankingLoss's expected column names)
  - SentenceTransformerTrainingArguments configures epochs, batch size,
    learning rate, warmup, and eval/save strategy
  - The base Hugging Face Trainer requires an eval_dataset whenever
    eval_strategy != "no" — this is separate from the
    InformationRetrievalEvaluator below, which computes the actual
    MRR@10 metric used for model selection. eval_dataset just satisfies
    the Trainer's generic loss-based eval loop using the same
    anchor/positive format as training data.
  - SentenceTransformerTrainer.train() runs the actual training loop

TRAIN/TEST SEPARATION (important):
Training data comes ONLY from data/cuad_eval/finetune_qa.json, which is
built from CUAD's TRAIN split (see scripts/download_data.py). The random
validation split below is carved out of THIS file only, for tracking
training progress (MRR@10 during training). It is NOT a substitute for
a genuinely independent test set.

data/cuad_eval/test_qa.json (CUAD's TEST split) is never read by this
script. It is reserved entirely for src/evaluation/eval_finetuned_embeddings.py,
which runs the final, truly held-out comparison between the base and
fine-tuned models after training completes.

OUTPUT: saved to ./models/lexrag-embeddings/
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss

# Reproducibility — same seed every run so training is comparable
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "./models/lexrag-embeddings"

# Sourced from CUAD's TRAIN split only — never overlaps with test_qa.json,
# which is reserved as a genuinely held-out evaluation set, untouched by
# any part of the fine-tuning process (training or internal validation).
FINETUNE_QA_PATH = "./data/cuad_eval/finetune_qa.json"
CONTRACTS_DIR = "./data/cuad"

BATCH_SIZE = 16
EPOCHS = 3
WARMUP_RATIO = 0.1
LEARNING_RATE = 2e-5
VALIDATION_FRACTION = 0.1  # 10% of fine-tuning pairs held out as validation


def load_training_pairs(qa_path: str, contracts_dir: str) -> List[Tuple[str, str]]:
    """
    Build (question, positive_context) pairs from CUAD TRAIN-split
    annotations only (finetune_qa.json). Never reads from test_qa.json.

    For each Q&A pair, finds the matching contract text and extracts a
    window of text centered on where the ground-truth answer actually
    appears — this gives the model real contract language as the
    positive example, not just the bare ground-truth answer string
    in isolation.

    Returns:
        List of (question, positive_context) tuples.
    """
    if not Path(qa_path).exists():
        raise FileNotFoundError(
            f"Fine-tuning data file not found at {qa_path}. "
            "Run scripts/download_data.py first."
        )

    with open(qa_path, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    contracts_dir_path = Path(contracts_dir)
    examples = []

    for item in qa_pairs:
        question = item.get("question", "").strip()
        ground_truth = item.get("ground_truth", "").strip()
        contract_title = item.get("contract_title", "")

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
                start = max(0, idx - 128)
                end = min(len(contract_text), idx + len(ground_truth) + 384)
                positive_context = contract_text[start:end].strip()
            else:
                positive_context = ground_truth
        else:
            positive_context = ground_truth

        if len(positive_context) > 20:
            examples.append((question, positive_context))

    print(f"Built {len(examples)} fine-tuning pairs from CUAD train-split annotations")
    return examples


def build_evaluator(validation_pairs: List[Tuple[str, str]]) -> InformationRetrievalEvaluator:
    """
    Build an Information Retrieval evaluator from a random validation
    split of fine-tuning pairs (NOT from test_qa.json — this evaluator
    tracks training progress only; it is not the final reported metric).
    """
    queries = {}
    corpus = {}
    relevant_docs = {}

    for i, (question, context) in enumerate(validation_pairs):
        qid = f"q_{i}"
        did = f"d_{i}"
        queries[qid] = question
        corpus[did] = context
        relevant_docs[qid] = {did}

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="cuad-validation",
        show_progress_bar=False,
    )


def finetune(
    base_model: str = BASE_MODEL,
    output_dir: str = OUTPUT_DIR,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
):
    """
    Fine-tune a sentence-transformer on CUAD TRAIN-split Q&A pairs using
    MultipleNegativesRankingLoss, via the modern SentenceTransformerTrainer
    API. Uses a random validation split for tracking training progress —
    test_qa.json remains fully untouched and is reserved for the final,
    genuinely independent evaluation.
    """
    print(f"=== LexRAG Embedding Fine-Tuning ===")
    print(f"Base model : {base_model}")
    print(f"Output dir : {output_dir}")
    print(f"Batch size : {batch_size}")
    print(f"Epochs     : {epochs}")
    print(f"Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

    model = SentenceTransformer(base_model)

    all_pairs = load_training_pairs(FINETUNE_QA_PATH, CONTRACTS_DIR)

    # Random split into train/validation — NOT a tail-slice, so the
    # validation set isn't biased toward whatever happens to be last
    # in the source file's ordering.
    random.shuffle(all_pairs)
    val_size = int(len(all_pairs) * VALIDATION_FRACTION)
    validation_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]

    print(f"Training on {len(train_pairs)} pairs")
    print(f"Validating on {len(validation_pairs)} pairs (random split, never trained on)\n")

    # MultipleNegativesRankingLoss expects a Dataset with "anchor" and
    # "positive" columns by default — questions are the anchor, contract
    # context windows are the positive.
    train_dataset = Dataset.from_dict({
        "anchor": [q for q, _ in train_pairs],
        "positive": [c for _, c in train_pairs],
    })

    # The base Hugging Face Trainer requires an eval_dataset whenever
    # eval_strategy is not "no" — this is separate from the
    # InformationRetrievalEvaluator below, which computes MRR@10
    # specifically. eval_dataset here just satisfies the Trainer's
    # generic loss-based eval loop using the same anchor/positive format.
    eval_dataset = Dataset.from_dict({
        "anchor": [q for q, _ in validation_pairs],
        "positive": [c for _, c in validation_pairs],
    })

    train_loss = MultipleNegativesRankingLoss(model=model)
    evaluator = build_evaluator(validation_pairs)

    total_steps = (len(train_dataset) // batch_size) * epochs
    eval_save_steps = max(1, len(train_dataset) // batch_size)  # roughly once per epoch

    print(f"Total training steps (approx) : {total_steps}")
    print(f"Eval/save interval (steps)    : {eval_save_steps}\n")

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,   # equivalent to the old save_best_model=True —
                                        # keeps the checkpoint with the best eval score,
                                        # not necessarily the last one trained
        metric_for_best_model="eval_cuad-validation_cosine_mrr@10",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",  # disable wandb/etc auto-logging — keep this self-contained
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )

    trainer.train()

    # Save the final (best, due to load_best_model_at_end=True) model
    # explicitly — this is the checkpoint the rest of the project loads.
    model.save_pretrained(output_dir)

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