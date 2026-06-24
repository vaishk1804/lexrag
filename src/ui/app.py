"""
Streamlit demo UI for LexRAG.

Run with: streamlit run src/ui/app.py
"""
import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LexRAG - Legal Contract Q&A", page_icon="⚖️", layout="wide")

st.title("⚖️ LexRAG - Legal Contract Q&A")
st.caption("Hybrid retrieval RAG system over CUAD commerical contracts · BM25 + Dense + RRF")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")

with st.sidebar:
  st.header("Settings")
  retrieval_mode = st.selectbox("Retrieval Mode", ["hybrid","dense","bm25"],index=0)
  st.divider()
  st.markdown(f"**Embedding model** `{EMBEDDING_MODEL.split('/')[-1]}`")
  st.markdown(f"**Chunks per query:** `{os.getenv('TOP_K', '5')}` (fixed at startup, matches API config)")
  st.markdown("Built on [CUAD]((https://www.atticusprojectai.org/cuad) — 510 commercial contracts.")

@st.cache_resource
def load_retriever(_embedding_model:str):
  from src.retrieval.retriever import HybridRetriever
  fixed_top_k = int(os.getenv("TOP_K","5"))
  return HybridRetriever(retrieval_k=fixed_top_k*2, top_k = fixed_top_k, embedding_model=_embedding_model)

@st.cache_resource
def load_llm_client():
  from src.evaluation.llm_client import get_llm_client
  return get_llm_client()

try:
  retriever = load_retriever( EMBEDDING_MODEL)
  client = load_llm_client()
  st.success(f"Index loaded - {retriever.collection.count():,} contract chunks ready")
except Exception as e:
  st.error("Failed to load index: {e}\n\nRun ingestion first: `python -m src.ingestion.pipeline`")
  st.stop()

st.subheader("Ask a question about the contracts")
question = st.text_input(
  "Question",
  placeholed="e.g. What are the termination conditions in this contract?",
  label_visibility="collapsed",
)

search_clicked = st.button("Search", type="primary")

st.caption("Try: *What are the termination conditions?* · *Who handles indemnification?* · *What is the governing law?*")

if search_clicked and question:
  from src.evaluation.llm_client import get_model_name

  with st.spinner("Retrieving and generating answer ..."):
    t0 = time.time()
    chunks = retriever.retrieve(question, mode=retrieval_mode)
    context_texts = [c.text for c in chunks]
    context_block = "\n\n---\n\n".join(context_texts)

    prompt = f"""Answer based ONLY on the contract excerpts below.
  If not found, say "This information is not found in the provided contract sections."

  CONTRACT EXCERPTS:
  {context_block}

  QUESTION: {question}
  ANSWER:"""
    
    response = client.chat.completions.create(
      model=get_model_name(),
      messages=[{"role":"user","content": prompt}],
      temperature=0,
      max_tokens= 512,
    )
    answer = response.choices[0].message.content.strip()
    latency_ms = (time.time() - t0) * 1000
  
  st.divider()
  col_ans, col_meta = st.columns([2,1])
  with col_ans:
    st.subheader("Answer")
    st.markdown(answer)
  with col_meta:
    st.metric("Latency",f"{latency_ms:.0f}ms")
    st.metric("Mode",retrieval_mode)
    st.metric("Chunks", len(chunks))

  st.divider()
  st.subheader(f"Retrieved Context ({len(chunks)} chunks)")
  for i, chunk in enumerate(chunks):
    with st.expander(f"Chunk {i+1} · {chunk.metadata.get('source','?')} · score={chunk.score:.3f}"):
      st.text(chunk.text)