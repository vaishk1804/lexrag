"""
Chunking strategies for legal contract text.

Three strategies are compared in experiments:
  - 256 tokens: fine-grained, risks splitting clauses mid-sentence
  - 512 tokens: empirically the best on CUAD
  - 1024 tokens: preserves more context but increases LLM noise

Uses LangChain's RecursiveCharacterTextSplitter with legal-aware separators.

Design decisions documented:
  - Separator order: paragraph breaks first because not all contracts use ARTICLE/Section headers.
  - Tokenizer: uses tiktoken (cl100k_base) for accurate token counting instead of 1-token=4-chars approximation
"""

import re
from dataclasses import dataclass
from typing import List
import copy

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Tokenizer for accurate token counting
# cl100k_base is the encoding used by GPT-3.5/GPT-4 and is a reasonable proxy
# proxy for sentence-transformers token counts (which use WordPiece).
_tokenizer = tiktoken.get_encoding("cl100k_base")

def token_length(text:str)->int:
  """Count tokens using tiktoken instead of character approximation"""
  return len(_tokenizer.encode(text))

# Separators ordered by preference for legal text.
# Paragraph breaks come first as they are the most universal separator
# across contract formats - not all contracts use ARTICLE or Section headers
# Regex mode enable to support case-insensitive section/article matching.
LEGAL_SEPARATORS = [
  "\n\n",       # paragraph breaks
  r"\n(?:ARTICLE|Article|article) ",   # article headers
  r"\n(?:SECTION|Section|section) ",    # section headers
  r"\n(?:WHEREAS|Whereas|whereas) ",    # recital clauses
  "\n",   # line breaks
  r"\. ",   # sentence breaks
  " ",      # word boundaries (last resort)
]

# Short causes which are legally significant and should not be filtered even under the length threshold.
IMPORTANT_SHORT_PATTERNS = [
  r"governing law",
  r"jurisdiction",
  r"force majeure",
  r"indemnif",
  r"terminat",
  r"confidential",
  r"warrant",
]
_important_re = re.compile("|".join(IMPORTANT_SHORT_PATTERNS),re.IGNORECASE)

def _is_worth_keeping(text:str)->bool:
  """
  Decide whether a chunk is worth keeping.
  Filters near-empty chunks but preserves short legally singnificant clauses.
  """
  stripped=text.strip()
  if len(stripped) >80:
    return True
  # Keep short chunks if they contain important legal terms
  if _important_re.search(stripped):
    return True
  return False

@dataclass
class Chunk:
  text:str
  metadata: dict

def chunk_document(
    text:str,
    metadata:dict,
    chunk_size:int = 512,
    chunk_overlap: int=50,
) -> List[Chunk]:
  """
  Split a single document into chunks

  Args:
      text: Full contract text
      metadata: Source info dict - must contain at least 'source' key.
          Values must be primitives (str , int, float) - no nested dicts
      chunk_size: Target chunk size in tokens
      chunk_overlap: Overlap between adjacent chunks in tokens.
          Preserves context at chunk boundaries so a clause split across two chunks retains its full meaning in each.
  Returns:
    List of Chunk objects with text and metadata
  
  Raises:
    ValueError: if text or metadata are missing/wrong type
  """
  if not isinstance(text,str) or not text.strip():
    raise ValueError("text must be a non-empty string")
  if not isinstance(metadata,dict):
    raise ValueError("metadata must be a dict")
  
  splitter = RecursiveCharacterTextSplitter(
    separators = LEGAL_SEPARATORS,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function = token_length,     # accurate token counting via tiktoken
    is_separator_regex=True,     # enables regex separators for case-insensitivity
  )

  raw_chunks = splitter.split_text(text)

  chunks=[]
  for i, chunk_text in enumerate(raw_chunks):
    chunk_meta=copy.deepcopy(metadata)
    chunk_meta.update({
      "chunk_index": i,
      "chunk_size_config": chunk_size,
      "chunk_overlap_config": chunk_overlap,
      "char_length": len(chunk_text),
      "token_length": token_length(chunk_text), 
    })
    chunks.append(Chunk(text=chunk_text.strip(),metadata=chunk_meta))
  
  return [c for c in chunks if _is_worth_keeping(c.text)]

def chunk_documents(
    docs: List[dict],
    chunk_size: int=512,
    chunk_overlap: int = 50,
) -> List[Chunk]:
  """
  Chunk a list of documents.

  Each doc must be a dict with 'text' (str) and 'metadata' (dict) keys.
  Invalid docs are skipped with a warning rather than crashing the pipeline.
  """
  all_chunks = []
  for i,doc in enumerate(docs):
    if not isinstance(doc,dict):
      print(f"Warning: doc at index {i} is not a dict, skipping")
      continue
    if "text" not in doc or "metadata" not in doc:
      print(f"Warning: doc at index {i} missing 'text' or 'metadata' key, skipping")
      continue
  
    try:
      chunks = chunk_document(
      text=doc["text"],
      metadata=doc["metadata"],
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,

    )
      all_chunks.extend(chunks)
    except ValueError as e:
      print(f"Warning: failed to chunk doc at index {i}: {e}")
  
  return all_chunks