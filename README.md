# LexRAG — Legal Contract Q&A with Retrieval Evaluation & Fine-Tuned Embeddings

A production-grade RAG system for querying commercial contracts, built on the CUAD benchmark dataset. Features hybrid BM25 + dense retrieval, a fine-tuned PyTorch embedding model, RAGAS-based evaluation with MLflow experiment tracking and model registry, and FastAPI/Streamlit deployment.

## Why this project exists

Naive RAG fails on legal text. Contract language is long-form, terminology-dense, and clause-dependent — a keyword match that works on general text misses the nuance of indemnification clauses spread across subsections, and a general-purpose embedding model conflates "governing law" with unrelated terms like "Government Official." This project treats retrieval quality as an engineering problem: measure it, fine-tune for the domain, experiment systematically, and evaluate honestly — including documenting the bugs found and fixed along the way.

## Architecture

```
CUAD Contracts (510, .txt)
        │
        ▼
  Ingestion Pipeline
  ├── Chunking (recursive, legal-aware separators, tiktoken-based sizing)
  ├── Embedding (base or fine-tuned sentence-transformer, configurable)
  └── ChromaDB (dense) + BM25 (sparse) — full rebuild each run, kept in sync
        │
        ▼
  PyTorch Fine-Tuning (embedding model)
  ├── MultipleNegativesRankingLoss on 10,050 CUAD train-split Q&A pairs
  ├── Held-out evaluation on CUAD test-split (200 pairs, never trained on)
  └── Output: domain-adapted embedding model
        │
        ▼
  Hybrid Retriever
  ├── BM25 sparse retrieval
  ├── Dense semantic retrieval (base or fine-tuned embeddings)
  └── Reciprocal Rank Fusion
        │
        ▼
  LLM Generation (Groq for dev, OpenAI for final reported numbers)
        │
        ▼
  RAGAS Evaluation
  ├── Faithfulness, Context Precision, Context Recall (+ Answer Relevancy on OpenAI)
  └── Completeness tracking — flags partial/failed evaluation jobs explicitly
        │
        ▼
  MLflow Tracking + Model Registry (alias-based, completeness-aware champion selection)
        │
        ▼
  FastAPI + Streamlit + Docker
```

## Stack

| Layer | Tool | Why |
|---|---|---|
| Chunking | LangChain RecursiveCharacterTextSplitter + tiktoken | Legal-aware separators, accurate token counting |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (base) + fine-tuned variant | Free, fast; fine-tuned version adapts to legal terminology |
| Fine-tuning | PyTorch, MultipleNegativesRankingLoss, AdamW | In-batch contrastive learning, no negative mining needed |
| Vector store | ChromaDB (local, persistent) | Cosine similarity, HNSW approximate search |
| Sparse retrieval | rank_bm25 | Exact legal terminology matching |
| Fusion | Reciprocal Rank Fusion | No learned weights, robust across score scales |
| LLM | Groq (dev, free) / OpenAI (final numbers) | Swappable via one .env variable |
| Evaluation | RAGAS | Faithfulness, relevancy, precision, recall |
| Experiment tracking | MLflow | Params, metrics, completeness fractions, model registry |
| Serving | FastAPI + Uvicorn | Async, production-style endpoint |
| UI | Streamlit | Interactive demo |
| Containerization | Docker + docker-compose | API + UI services |

## Dataset

**CUAD (Contract Understanding Atticus Dataset)** — 510 commercial contracts, 13,000+ expert annotations, CC BY 4.0 license.

Two separate files are derived from CUAD's own train/test split, kept strictly apart to avoid data leakage:
- `data/cuad_eval/finetune_qa.json` — from CUAD's **train** split, used only for embedding fine-tuning
- `data/cuad_eval/test_qa.json` — from CUAD's **test** split, never touched during training; reserved for the genuinely held-out comparison

## Quickstart

```bash
git clone https://github.com/<your-username>/lexrag
cd lexrag
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

cp .env.example .env           # add your API keys (see Configuration below)

python scripts/download_data.py
python -m src.ingestion.pipeline --chunk-size 512

mlflow ui --port 5000           # separate terminal

uvicorn src.serving.api:app --reload --port 8000     # FastAPI
streamlit run src/ui/app.py                          # Streamlit demo
```

## Configuration (.env)

```
OPENAI_API_KEY=your-openai-key
GROQ_API_KEY=your-groq-key
LLM_PROVIDER=groq                          # groq (free dev) | openai (final numbers)
LLM_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=./models/lexrag-embeddings # or sentence-transformers/all-MiniLM-L6-v2 for base
CHROMA_PERSIST_DIR=./data/chroma_db
BM25_INDEX_PATH=./data/bm25_index.pkl
RETRIEVAL_K=10
TOP_K=5
```

`top_k` is fixed at deployment time via this `.env` value — neither the FastAPI nor Streamlit interface allows a caller to override it per-request, by design.

## PyTorch Fine-Tuning

```bash
python -m src.training.finetune_embeddings        # ~4 hours on CPU, 3 epochs
python -m src.evaluation.eval_finetuned_embeddings # held-out base vs fine-tuned comparison
python -m src.ingestion.pipeline --chunk-size 512 --embedding-model ./models/lexrag-embeddings
```

**Held-out test set result** (200 pairs from CUAD's test split, never seen during training):

| Model | Top-1 Retrieval Accuracy |
|---|---|
| Base (all-MiniLM-L6-v2) | 0.0550 |
| Fine-tuned | 0.1450 |
| **Delta** | **+0.0900 absolute (~2.6x relative)** |

This result was reproduced twice (identical seed, identical numbers both times) after discovering and fixing a contract-corpus duplication bug — see `FINDINGS.md` for the full investigation.

## Running Experiments

```bash
python -m src.evaluation.run_experiment --config experiments/baseline.yaml
python -m src.evaluation.run_experiment --config experiments/hybrid_512.yaml
python -m src.evaluation.run_experiment --config experiments/finetuned_baseline.yaml
python -m src.evaluation.run_experiment --config experiments/finetuned_hybrid_512.yaml

python -m src.evaluation.model_registry   # registers best complete-data run as @champion
```

## Experiment Results

*Final numbers below are from a clean, complete evaluation pass. Earlier Groq free-tier runs hit daily token quota limits and per-minute rate limits causing partial data on some metrics — the model registry explicitly excludes any run with incomplete data from champion consideration (see Model Registry section). Numbers below reflect only fully-complete runs.*

| Config | Embedding | Retrieval | Faithfulness | Context Precision | Context Recall | Composite |
|---|---|---|---|---|---|---|
| baseline | base | dense | TBD | TBD | TBD | TBD |
| hybrid-512 | base | hybrid | TBD | TBD | TBD | TBD |
| finetuned-dense | fine-tuned | dense | TBD | TBD | TBD | TBD |
| finetuned-hybrid | fine-tuned | hybrid | TBD | TBD | TBD | TBD |

*Populated from the final OpenAI-based evaluation pass — see `FINDINGS.md` for the full experiment log including Groq-based directional results.*

## Model Registry

Uses MLflow model version **aliases**, not the deprecated stage-based lifecycle (`None`/`Staging`/`Production`/`Archived`). The best-performing retriever *configuration* (chunk size, retrieval mode, embedding model path, RRF constant) — not trained weights — is registered and pointed to by a `champion` alias:

```python
model = mlflow.pyfunc.load_model("models:/lexrag-retriever@champion")
config = model.unwrap_python_model().config
```

Champion selection is **completeness-aware**: a run with partial evaluation data (e.g., 9/10 questions scored due to an API timeout) is excluded from consideration entirely, even if its raw composite score looks higher than a complete run's — averaging over fewer samples produces a less trustworthy number, not a directly comparable one.

## Key Design Decisions

**Why hybrid over dense-only?**
Legal text contains precise terminology where exact lexical matching (BM25) outperforms semantic similarity, and vice versa for paraphrased or conceptually-similar queries. A documented case: querying "governing law" with dense-only retrieval returned zero relevant results in the top 5 — it confused the query with lexically similar but unrelated terms like "Government Official." BM25, on the same corpus, correctly surfaced a section titled "GOVERNING LAW" at rank 1. Hybrid retrieval with RRF fusion matched or beat the better of the two individual methods on every one of 5 manually-reviewed test questions.

**Why fine-tune the embedding model?**
General-purpose embeddings represent legal terminology ("indemnification", "force majeure") imprecisely. Fine-tuning on 10,050 real CUAD question-context pairs with PyTorch's `MultipleNegativesRankingLoss` improved held-out top-1 retrieval accuracy from 5.5% to 14.5% — measured on a test set the model never saw during training or validation.

**Why does retrieval need a faithfulness check downstream?**
Retrieval has no concept of "no good answer exists." A nonsense query ("purple elephant spaceship banana") still returned 3 results, including a chunk from a contract with a company literally named "Elephant Talk Communications" — pure lexical overlap, zero semantic relevance. This is why the generation prompt explicitly instructs the LLM to say when an answer isn't found in context, and why RAGAS faithfulness scoring exists as an independent downstream check.

**Why exclude incomplete runs from model registry consideration, not just deprioritize them?**
The composite score averages across all RAGAS metrics. If even one metric has partial data, the composite is contaminated by that partial number — there's no way to "partially trust" an average with one untrustworthy ingredient. The only safe response is exclusion, not deprioritization.

Full investigation log — including a critical bug where experiment scores were only partially logged for several days before being caught and fixed — is in `FINDINGS.md`.

## Project Structure

```
lexrag/
├── data/                         # CUAD contracts + eval splits (gitignored)
├── models/                       # Fine-tuned embedding checkpoint (gitignored)
├── src/
│   ├── ingestion/                # Chunking, embedding, indexing, pipeline
│   ├── retrieval/                # Hybrid retriever (BM25 + dense + RRF)
│   ├── training/                 # PyTorch fine-tuning (MultipleNegativesRankingLoss)
│   ├── evaluation/                # RAGAS, experiment runner, model registry, LLM client abstraction
│   ├── serving/                  # FastAPI app + schemas
│   └── ui/                       # Streamlit demo
├── experiments/                  # YAML configs per experiment run
├── scripts/                      # CUAD download (train/test split separation)
├── tests/                        # Unit + integration tests
├── Dockerfile / docker-compose.yml
├── FINDINGS.md                   # Full investigation log, every bug + fix documented
├── RESUME_BULLETS.md
└── README.md
```

## Deployment

```bash
docker-compose up --build
# API: http://localhost:8000/docs
# UI:  http://localhost:8501
```

## Testing

```bash
pytest tests/ -v
```

Covers: chunking edge cases, embedder validation, BM25 tokenization, RRF fusion logic (including a hand-verified numerical example), retriever edge cases (empty/nonsense/special-character queries), RAGAS completeness calculation, model registry champion-selection logic (including a regression test for the partial-data exclusion bug), and fine-tuned model integrity checks.
