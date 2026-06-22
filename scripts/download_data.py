"""
Downloads the CUAD dataset from HuggingFace Hub.

CUAD (Contract Understanding Atticus Dataset):
- 510 commercial contracts
- 41 clause categories, 13,000+ expert annotations
- License: CC BY 4.0 (free to use)

After running:
  - Contracts saved to: ./data/cuad/   (~500 .txt files, from both splits)
  - Fine-tuning pairs (TRAIN split only) saved to: ./data/cuad_eval/finetune_qa.json
  - Held-out test pairs (TEST split only) saved to: ./data/cuad_eval/test_qa.json

IMPORTANT — train/test separation:
finetune_qa.json is built ONLY from dataset["train"] and is the source
used for embedding fine-tuning (src/training/finetune_embeddings.py).
test_qa.json is built ONLY from dataset["test"] and must NEVER be used
for training — it is reserved as the genuinely held-out evaluation set
(src/evaluation/eval_finetuned_embeddings.py). Keeping these in separate
files makes data leakage between train and test structurally harder to
introduce by accident.
"""

import json
from pathlib import Path


def download_cuad():
    from datasets import load_dataset

    print("Downloading CUAD dataset from HuggingFace (~700MB, a few minutes)...")
    dataset = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)

    contracts_dir = Path("./data/cuad")
    contracts_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path("./data/cuad_eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset splits: {list(dataset.keys())}")

    # --- Save contracts (from BOTH splits, deduplicated by title) ---
    # Contracts are saved once regardless of split, since the retrieval
    # corpus itself isn't split — only the Q&A pairs used for
    # fine-tuning vs evaluation need to stay separated.
    seen_titles = set()
    contract_count = 0
    failed_count = 0

    for split in ["train", "test"]:
        print(f"Processing {split} split contracts...")
        for item in dataset[split]:
            title = item.get("title", f"contract_{contract_count}")
            ctx = item.get("context", "")

            if not ctx or title in seen_titles:
                continue

            seen_titles.add(title)

            safe = "".join(
                c if c.isalnum() or c in "._- " else "_"
                for c in title
            )
            filename = f"{safe[:80]}_{contract_count:04d}.txt"
            filepath = contracts_dir / filename

            try:
                filepath.write_text(ctx, encoding="utf-8")
                contract_count += 1
            except Exception as e:
                print(f"Warning: failed to write {filename}: {e}")
                failed_count += 1

    print(f"Saved {contract_count} contracts to {contracts_dir}/")
    if failed_count:
        print(f"Warning: {failed_count} contracts failed to save")

    # --- Fine-tuning pairs: TRAIN split ONLY ---
    # This file is the source for embedding fine-tuning. It must never
    # be confused with or merged into the test set below.
    finetune_pairs = []
    for item in dataset["train"]:
        answers = item.get("answers", {}).get("text", [])
        question = item.get("question", "").strip()
        if answers and question:
            finetune_pairs.append({
                "question": question,
                "ground_truth": answers[0],
                "all_answers": answers,
                "contract_title": item.get("title", ""),
            })

    try:
        with open(eval_dir / "finetune_qa.json", "w", encoding="utf-8") as f:
            json.dump(finetune_pairs, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(finetune_pairs)} fine-tuning pairs (TRAIN split) to {eval_dir}/finetune_qa.json")
    except Exception as e:
        print(f"Error saving fine-tuning pairs: {e}")

    # --- Test pairs: TEST split ONLY ---
    # Genuinely held-out evaluation set. NEVER read by
    # src/training/finetune_embeddings.py. Only read by
    # src/evaluation/eval_finetuned_embeddings.py for the final,
    # independent comparison between base and fine-tuned models.
    test_pairs = []
    for item in dataset["test"]:
        answers = item.get("answers", {}).get("text", [])
        question = item.get("question", "").strip()
        if answers and question:
            test_pairs.append({
                "question": question,
                "ground_truth": answers[0],
                "all_answers": answers,
                "contract_title": item.get("title", ""),
            })

    try:
        with open(eval_dir / "test_qa.json", "w", encoding="utf-8") as f:
            json.dump(test_pairs, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(test_pairs)} held-out test pairs (TEST split) to {eval_dir}/test_qa.json")
    except Exception as e:
        print(f"Error saving test pairs: {e}")

    print("\nDownload complete.")
    print(f"  Contracts (retrieval corpus) : {contract_count}")
    print(f"  Fine-tuning pairs (train)    : {len(finetune_pairs)}")
    print(f"  Held-out test pairs (test)   : {len(test_pairs)}")
    print("\nNext: python -m src.ingestion.pipeline --chunk-size 512")


if __name__ == "__main__":
    download_cuad()