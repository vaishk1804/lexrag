"""
Main ingestion pipeline - orchestrates loading, chunking, and indexing
of all CUAD contracts into ChromaDB (dense) and BM25 (sparse).

Usage:
    # Default - 512 token chunks, base embedding model
    python src/ingestion/pipeline.py

    # Custom chunk size
    python src/ingestion/pipeline.py --chunk-size 256
    python src/ingestion/pipeline.py --chunk-size 1024

    # Fine-tuned embedding model
    python src/ingestion/pipeline.py --embedding-model ./models/lexrag-embeddings

Re-running with the same chunk size is safe:
    ChromaDB uses upsert - existing vectors are overwritten not duplicated.
    BM25 is always fully rebuilt from scratch.
"""

import argparse
import os
from pathlib import Path
from typing import List

import pdfplumber
from dotenv import load_dotenv
from tqdm import tqdm

from src.ingestion.chunker import chunk_documents
from src.ingestion.indexer import DualIndexer

load_dotenv()

def extract_text_from_pdf(pdf_path: Path) -> str:
  """
  Extract text from a PDF using pdfplumber.
  Joins all pages with double newline to preserve paragraph structure.
  """
  text_parts =[]
  with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
      page_text = page.extract_text()
      if page_text:
        text_parts.append(page_text)
  return "\n\n".join(text_parts)

def load_documents(data_dir: str) -> List[dict]:
  """
  Load all contracts from data_dir.
  Supports .txt and .pdf files.

  Returns:
      List of dicts, each with 'text' and 'metadata' keys.
      Skips files that fail to load with a warning.
  """
  data_path = Path(data_dir)
  if not data_path.exists():
    raise FileNotFoundError(
      f"Data dictionary not found: {data_dir}\n"
      "Run scripts/download_data.py first."
    )
  
  contract_files = (
    list(data_path.glob("**/*.txt")) +
    list(data_path.glob("**/*.pdf"))
  )

  if not contract_files:
    raise FileNotFoundError(
      f"No .txt or .pdf files found in {data_dir}\n"
      "Run scripts/download_data.py first."
    )
  
  print(f"Found {len(contract_files)} contract files in {data_dir}")

  docs = []
  failed = 0
  
  for file_path in tqdm(contract_files, desc = "Loading contracts"):
    try:
      if file_path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(file_path)
      else:
        text = file_path.read_text(encoding="utf-8", errors="ignore")

      # Skip near-empty files - likely headers or corrupt files
      if len(text.strip()) < 100:
        continue

      docs.append({
        "text": text,
        "metadata": {
          "source": file_path.name,
          "source_path": str(file_path),
          "file_type": file_path.suffix,
          "char_count": len(text),
        },
      })
    
    except Exception as e:
      print(f"\n Warning: failed to load {file_path.name}: {e}")
      failed+=1
  
  print(f"Loaded {len(docs)} documents ({failed}) failed")
  return docs

def run_pipeline(
    data_dir:str = "/data/cuad",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
  """
  Full ingestion pipeline - load, chunk, index.
  """

  print(f"\n{'='*50}")
  print(f"LexRAG Ingestion Pipeline")
  print(f"{'='*50}")
  print(f"Data dir    : {data_dir}")
  print(f"Chunk size    : {chunk_size} tokens")
  print(f"Chunk overlap  : {chunk_overlap} tokens")
  print(f"Embedding model : {embedding_model}")
  print(f"{'='*50}")

  # Step 1 - Load documents
  docs = load_documents(data_dir)

  # Step 2 - chunk documents
  print(f"\nChunking {len(docs)} documents...")
  chunks = chunk_documents(
    docs,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
  )

  if not chunks:
    raise RuntimeError("No chunks produced. Check data directory contents.")
  
  avg_tokens = sum(c.metadata["token_length"] for c in chunks) // len(chunks)
  print(f"Produced {len(chunks):,} chunks")
  print(f"Average chunk length: {avg_tokens} tokens")

  # Step 3 - Index chunks
  print(f"\nIndexing {len(chunks)} chunks...")
  indexer = DualIndexer(embedding_model=embedding_model)
  indexer.index_chunks(chunks)

  # Summary
  print(f"\n{'='*50}")
  print(f"Ingestion complete")
  print(f"{'='*50}")
  print(f"ChromaDB vectors : {indexer.get_collection_count():,}")
  print(f"BM25 documents   : {len(indexer.bm25_chunks):,}")
  print(f"Chunk size used  : {chunk_size} tokens")
  print(f"{'='*50}\n")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="LexRAG ingestion pipeline")
  parser.add_argument(
    "--data-dir",
    default="./data/cuad",
    help="Directory containing contract files (default: ./data/cuad)"
  )
  parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in tokens (default: 512)",
    )
  parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in tokens (default: 50)",
    )
  parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model name or path to fine-tuned model",
    )
  
  args = parser.parse_args()

  run_pipeline(
    data_dir=args.data_dir,
    chunk_size=args.chunk_size,
    chunk_overlap=args.chunk_overlap,
    embedding_model=args.embedding_model,
  )
