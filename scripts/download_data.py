import json
from pathlib import Path

def download_cuad():
  from datasets import load_dataset

  print("Downloading CUAD dataset from HuggingFace (~700MB)...")

  dataset = load_dataset("theatticusproject/cuad-qa",trust_remote_code=True)

  contracts_dir = Path("./data/cuad")
  contracts_dir.mkdir(parents=True,exist_ok=True)
  eval_dir = Path("./data/cuad_eval")
  eval_dir.mkdir(parents=True,exist_ok=True)

  print(f"Dataset splits: {list(dataset.keys())}")

  # Save contracts from BOTH train and test splits so retriever can find contracts references in eval questions
  seen_titles=set()
  contract_count=0
  failed_count = 0

  for split in ["train","test"]:
    print(f"Processing {split} split ...")
    for item in dataset[split]:
      title=item.get("title",f"contract_{contract_count}")
      ctx = item.get("context","")

      if not ctx or title in seen_titles:
        continue

      seen_titles.add(title)

      # Sanitizing title for use as a filename
      safe = "".join(
        c if c.isalnum() or c in "._-" else "_"
        for c in title
      )

      # Always append index to prevent truncation collisions
      filename = f"{safe[:80]}_{contract_count:04d}.txt"
      filepath = contracts_dir / filename

      try:
        filepath.write_text(ctx, encoding="utf-8")
        contract_count+=1
      except Exception as e:
        print(f"Warning failed to write {filename}: {e}")
        failed_count+=1

  print(f"Saved {contract_count} contracts to {contracts_dir}/")
  if failed_count:
    print(f"Warning: {failed_count} contracts failed to save")

  # Save eval Q&A pairs from test split
  eval_pairs=[]
  for item in dataset["test"]:
    try:
      answers=item.get("answers",{}).get("text",[])
      question = item.get("question","").strip()
      if answers and question:
        eval_pairs.append({

          "question": question,
          "ground_truth": answers[0],
          "all_answers": answers, # keeping all for reference
          "contract_title": item.get("title",""),
        })
    except Exception as e:
      print(f"Warning: failed to process eval item: {e}")
  
  try:
    with open(eval_dir / "test_qa.json","w",encoding="utf-8") as f:
      json.dump(eval_pairs,f,indent=2, ensure_ascii=False)
    print(f"Saved {len(eval_pairs)} eval Q&A pairs to {eval_dir}/test_qa.json")
  except Exception as e:
    print(f"Error saving eval pairs: {e}")
  
  print("\n Download complete.")

if __name__ == "__main__":
  download_cuad()




