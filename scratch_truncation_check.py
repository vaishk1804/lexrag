from src.evaluation.ragas_eval import build_ragas_dataset

ds = build_ragas_dataset(
    questions=["q1"],
    answers=["a1"],
    contexts=[["x" * 5000]],
    ground_truths=["gt1"],
)
print(f"Context length after truncation: {len(ds[0]['contexts'][0])}")
assert len(ds[0]["contexts"][0]) == 3000, "truncation not applied"
print("ok - truncation working")