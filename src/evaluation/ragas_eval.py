"""
RAGAS evaluation wrapper for Lexrag.

RAGAS measures four dimensions of RAG quality , each catching a distinct failure mode:

1. FAITHFULNESS - does the answer containe only claims supported by the retrieved context? A faithfulness failure means the LLM is hallucinating facts not present in what was retrieved.

2. ANSWER RELEVANCY - does the answer actually address the question asked? An answer can be faithful to context but still miss the point if the wrong context is retrieved.

3. CONTEXT PRECISION - of the retrieved chunks, how many were actually relevatn? Low precision means retrieval is noisy.

4. CONTEXT RECALL - of the information needed to answer , how much was actually retrieved? Low recall means the right chunk exists is the corpus but wasn't surfaced/

These 4 metrics form a diagnostic grid: low precision + low recall means retrieval is fundamentally broken; high precision + recall but low faithfulness means the problem is in generation, not retrieval.
"""
from typing import Dict, List, Tuple

from ragas.run_config import RunConfig

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
  answer_relevancy,
  context_precision,
  context_recall,
  faithfulness,
)

from src.evaluation.llm_client import get_chat_model, get_provider

def build_ragas_dataset(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
) -> Dataset:
  """Build a HuggingFace Dataset in the format RAGAS expects."""
  assert len(questions) == len(answers) == len(contexts) ==len(ground_truths), (
    "Questions, answers, contexts, and ground_truths must all have equal length."
  )
  return Dataset.from_dict({
    "question":questions,
    "answer": answers,
    "contexts":contexts,
    "ground_truth":ground_truths,
  })

def run_ragas_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
) -> Tuple[Dict[str,float], Dict[str,float]]:
  """
  Run RAGAS evaluation using whichever LLM provider is active.
    Note: answer_relevancy is excluded when using Groq — RAGAS implements
    this metric using OpenAI's `n` parameter (multiple completions per
    call) for efficiency, which Groq's API does not support (n must be 1).
    This is a known compatibility gap between providers, not a bug.
    The final OpenAI-based evaluation pass includes all four metrics.

  """
  dataset= build_ragas_dataset(questions,answers,contexts,ground_truths)
  llm=get_chat_model()
  metrics = [faithfulness, context_precision, context_recall]
  
  if get_provider() == "openai":
    metrics.append(answer_relevancy)
  # Limit concurrency to avoid tripping Groq's free-tier rate limit.
  # Default RAGAS concurrency fires ~30 requests at once; Groq's free
  # tier allows far fewer requests per minute. max_workers=2 with a
  # longer timeout trades speed for reliability on the free tier.
  
  run_config = RunConfig(max_workers = 1, timeout=120)

  result = evaluate(
    dataset=dataset,
    metrics = metrics,
    llm=llm,
    raise_exceptions=False,
    run_config=run_config,
  )

  # RAGAS silently drops failed jobs from the average — surface this
  # so partial-data runs are never mistaken for complete ones.
  result_df = result.to_pandas()
  total_rows = len(result_df)

  scores={}
  completeness={}
  metric_names = ["faithfulness","context_precision","context_recall"]
  if "answer_relevancy" in result_df.columns:
    metric_names.append("abswer_relevancy")
  for metric_name in metric_names:
    if metric_name in result_df.columns:
      valid_count = result_df[metric_name].notna().sum()
      fraction = valid_count/total_rows if total_rows > 0 else 0.0
      completeness[f"{metric_name}_completeness"] = round(fraction, 4)

      if valid_count < total_rows:
        print(
          f"⚠️  {metric_name}: only {valid_count}/{total_rows} "
          f"questions scored successfully (rest failed/timed out)"
        )

  scores[metric_name] = round(float(result_df[metric_name].mean()), 4)

  return scores, completeness