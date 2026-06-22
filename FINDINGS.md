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
