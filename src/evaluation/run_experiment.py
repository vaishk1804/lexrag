"""
Experiment runner - ties together ingestion, retrieval, generation, and
RAGAS evaluation, with all params, metrics, and completeness fractions
logged to MLflow.

Usage:
    mlflow ui --port 5000
    python -m src.evaluation.run_experiment --config experiments/baseline.yaml

Completeness metrics (e.g. faithfulness_completeness: 0.7) are logged
alongside the RAGAS scores so model_registry.py can exclude partial runs
from champion consideration.
"""

import argparse
import time
from typing import List

import mlflow
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from src.evaluation.llm_client import get_llm_client, get_model_name, get_provider
from src.evaluation.ragas_eval import run_ragas_evaluation
from src.ingestion.pipeline import run_pipeline
from src.retrieval.retriever import HybridRetriever

load_dotenv()


EVAL_QUESTIONS = [
    "What is the duration of the agreement?",
    "Who is responsible for indemnification in case of third-party claims?",
    "What are the termination conditions?",
    "Is there a limitation of liability clause? What does it cap?",
    "What governing law applies to this contract?",
    "What are the payment terms?",
    "Are there any non-compete or non-solicitation clauses?",
    "What intellectual property rights does each party retain?",
    "What are the conditions for contract renewal?",
    "Is there a confidentiality or non-disclosure obligation?",
]

EVAL_GROUND_TRUTHS = [
    "The agreement term and duration should be specified in the contract.",
    "The indemnifying party is responsible for defending and holding harmless the other party.",
    "Either party may terminate for material breach with a notice period.",
    "Liability is typically capped at fees paid in the preceding 12 months.",
    "The governing law clause specifies the jurisdiction for dispute resolution.",
    "Payment terms specify the amount, frequency, and method of payment.",
    "Non-compete clauses restrict a party from engaging in competitive activities.",
    "Each party retains ownership of its pre-existing intellectual property.",
    "Renewal conditions specify automatic renewal or notice requirements.",
    "Confidential information must not be disclosed to third parties.",
]


def generate_answer(question: str, contexts: List[str], client, model: str) -> str:
    """
    Generate an answer grounded in retrieved context. Explicitly instructed
    to say when an answer isn't found.
    """
    context_text = "\n\n---\n\n".join(contexts)

    prompt = f"""You are a legal document analyst. Answer the question based ONLY on the provided contract excerpts.
If the answer is not found in the excerpts, say "This information is not found in the provided contract sections."
Be specific and reference relevant contract language when possible.

CONTRACT EXCERPTS:
{context_text}

QUESTION: {question}

ANSWER:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def run_experiment(config: dict):
    """Run one full experiment: ingest, retrieve, generate, evaluate, log to MLflow."""
    experiment_name = config.get("experiment_name", "lexrag_experiments")
    mlflow.set_experiment(experiment_name)

    model_name = get_model_name()
    provider = get_provider()

    with mlflow.start_run(run_name=config.get("run_name", "unnamed_run")):
        mlflow.log_params({
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "retrieval_mode": config["retrieval_mode"],
            "retrieval_k": config["retrieval_k"],
            "top_k": config["top_k"],
            "llm_model": model_name,
            "llm_provider": provider,
            "rrf_k": config.get("rrf_k", 60),
        })

        print(f"\n[1/3] Building index with chunk_size={config['chunk_size']}...")
        run_pipeline(
            data_dir=config.get("data_dir", "./data/cuad"),
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            embedding_model=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        )

        print(f"\n[2/3] Running retrieval ({config['retrieval_mode']}) + generation "
              f"using {provider}/{model_name}...")
        retriever = HybridRetriever(
            retrieval_k=config["retrieval_k"],
            top_k=config["top_k"],
            rrf_k=config.get("rrf_k", 60),
            embedding_model=config.get("embedding_model","sentence-transformers/all-MiniLM-L6-v2")
        )
        client = get_llm_client()

        all_answers = []
        all_contexts = []
        latencies = []

        for question in tqdm(EVAL_QUESTIONS, desc="Evaluating"):
            t0 = time.time()
            chunks = retriever.retrieve(question, mode=config["retrieval_mode"])
            context_texts = [c.text for c in chunks]
            answer = generate_answer(question, context_texts, client, model_name)
            latencies.append((time.time() - t0) * 1000)
            all_answers.append(answer)
            all_contexts.append(context_texts)

        print(f"\n[3/3] Running RAGAS evaluation...")
        scores, completeness = run_ragas_evaluation(
            questions=EVAL_QUESTIONS,
            answers=all_answers,
            contexts=all_contexts,
            ground_truths=EVAL_GROUND_TRUTHS,
        )

        mlflow.log_metrics(scores)
        mlflow.log_metrics(completeness)
        mlflow.log_metrics({
            "latency_p50_ms": sorted(latencies)[len(latencies) // 2],
            "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "latency_mean_ms": sum(latencies) / len(latencies),
        })

        print(f"\n=== Results: {config.get('run_name')} ({provider}/{model_name}) ===")
        for k, v in scores.items():
            print(f"  {k:<24}: {v:.4f}")
        print(f"  --- completeness ---")
        for k, v in completeness.items():
            flag = "  WARNING: PARTIAL" if v < 1.0 else ""
            print(f"  {k:<24}: {v:.4f}{flag}")
        print(f"  latency_p95_ms          : {sorted(latencies)[int(len(latencies)*0.95)]:.0f}")
        print(f"\nLogged to MLflow experiment '{experiment_name}'. View at http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_experiment(config)