# LexRAG — Findings Log

## Sprint 1 — Ingestion

**BM25 noise on common legal terms**
Query "indemnification clause" — top BM25 result was an unrelated penal clause; correct language ranked #2. Cause: "indemnif" appears in 1,306/14,492 chunks (9%), common enough that IDF weighting doesn't strongly differentiate it; "clause" added further noise.

**Dense semantic drift on lexically similar terms**
Query "governing law" — dense returned zero relevant results in top 5, confusing it with "Government Official"/"Governmental Authorities." BM25 correctly ranked a section titled "GOVERNING LAW" at #1 on the same topic (keyword-style phrasing).

## Sprint 2 — Retrieval

**Index desync bug**
ChromaDB's upsert only overwrites matching IDs. Re-running ingestion with a different chunk_size produces different chunk_index sequences, so old vectors never get overwritten — they silently accumulate (58,828 vectors from 3 mixed runs vs 14,492 in BM25). Fix: force a full collection reset before every ingestion run.

**Query phrasing changes BM25 performance significantly**
"governing law jurisdiction" (keyword-style) → BM25 ranks correctly at #1.
"What is the governing law?" (natural language) → BM25 fails entirely.
Cause: natural-language questions are full of stopwords ("what", "is", "the") that dilute BM25's signal — tokenizer has no stopword filtering. Concrete, fixable gap.

**Hybrid matched-or-beat the better individual method on 5/5 test questions**
Across termination, indemnification, governing law, non-compete, payment terms — hybrid never underperformed both dense and BM25 individually.

**Retrieval has no concept of "no good answer exists"**
Nonsense query ("purple elephant spaceship banana") still returned 3 results, including a chunk from a company literally named "Elephant Talk Communications" — pure lexical overlap, zero semantic relevance. Retrieval always returns top-k regardless of relevance. Motivates: (1) explicit "not found in context" instruction in the generation prompt, (2) RAGAS faithfulness scoring as a downstream check.

## Sprint 3 — Evaluation

**Groq free tier viable for development, not final reporting**
Used Groq (llama-3.1-8b-instant) for generation and RAGAS judging during dev to avoid OpenAI cost. Two limitations surfaced:

1. `answer_relevancy` uses OpenAI's `n` parameter internally — Groq rejects n>1, so this metric is excluded on Groq (re-enabled automatically when LLM_PROVIDER=openai).
2. Daily token quota (500K/day) exhausted after ~2 full evaluation runs — RAGAS's internal calls are token-heavy.

Decision: Groq for free dev-phase iteration (proved the full pipeline works end-to-end), one clean final pass on OpenAI for resume numbers.

**RAGAS silently averages over partial data**
`evaluate()` drops failed/timed-out jobs without flagging it — a run with 7/10 successful judgments reports the same way as 10/10, just with a different number. Added explicit completion tracking (row count vs expected total, per metric) so partial runs are never mistaken for complete ones. This caught real data loss: a baseline run that looked clean in the summary was actually built from 70–90% completion per metric.

**Concurrency: speed vs reliability trade-off**
Default RAGAS concurrency (~30 parallel requests) hit timeouts immediately on Groq's rate limits. `max_workers=1` (fully serialized) eliminated timeouts on a clean-quota run (hybrid_512: 30/30 succeeded) but takes ~25–30 min for the RAGAS step alone. Real throughput-vs-reliability trade-off when working with rate-limited external APIs, not a code bug.

**Preliminary numbers (Groq, partial data — directional only)**

| Config                | Faithfulness | Context Precision | Context Recall | Composite   | Completeness     |
| --------------------- | ------------ | ----------------- | -------------- | ----------- | ---------------- |
| baseline (dense, 512) | 0.84–0.88\*  | 0.91–0.94\*       | 0.73–0.74\*    | 0.83–0.85\* | 70–90% (partial) |
| hybrid_512            | 0.8418       | 0.9489            | 0.7137         | 0.8348      | 100% (30/30)     |

\*Two baseline runs attempted, both partial due to quota exhaustion mid-run; ranges shown for reference only.

No confident dense-vs-hybrid comparison yet — baseline data is incomplete. Final numbers will come from one clean OpenAI pass across all 4 configs.

## Open questions

- Run a complete OpenAI evaluation pass across baseline, hybrid_512, hybrid_256, hybrid_1024 — these are the resume numbers.
- Does context_precision actually improve with hybrid once measured on complete data, consistent with the Sprint 2 qualitative finding (5/5 questions)?
- Does adding stopword filtering to BM25 measurably move precision/recall on natural-language queries like "What is the governing law?"

### Model registry: aliases over deprecated stages

MLflow deprecated stage-based model lifecycle management (None/Staging/
Production/Archived) starting in MLflow 2.8/2.9 in favor of model version
aliases — mutable named pointers (e.g. "champion") that can be reassigned
to any version without requiring other versions to be archived first.
Implemented using set_registered_model_alias() instead of the deprecated
transition_model_version_stage().

### Champion selection must exclude incomplete runs, not just deprioritize them

Initial model registry design selected the best run by composite score
alone. This is a real bug: a run with 70% data completion can produce a
misleadingly high (or low) composite average — it is not on equal footing
with a 100%-complete run, so it cannot be ranked against it at all, not
just ranked lower. Fixed by logging a completeness fraction per RAGAS
metric (rows scored successfully / total rows) and excluding any run
below 100% completeness from champion consideration entirely, with a
direct regression test confirming a partial run's higher composite score
does not let it beat a complete run's lower one.

### Fine-tuning result: held-out test set comparison

Fine-tuned all-MiniLM-L6-v2 via PyTorch (MultipleNegativesRankingLoss,
3 epochs, 10,050 train pairs from CUAD's train split). Evaluated on 200
pairs from CUAD's test split — never seen during training or internal
validation. Top-1 retrieval accuracy: base model 0.0550, fine-tuned
0.1450 (+0.0900 absolute, ~2.6x relative improvement). This is the
genuinely independent metric; internal validation MRR@10 during training
(0.0721 -> 0.0767) moved less, likely due to a harder 1,116-way ranking
task with many near-duplicate CUAD question templates acting as
distractors.
