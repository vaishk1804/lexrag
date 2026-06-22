"""
LLM client factory - abstracts away the difference between OpenAI and Groq.

Both are OpenAI-API-compatible, so the only difference is base_url and api_key. This lets the rest of the codebase call get_ll_client() or get_chat_model() without caring which provider is active - swapping providers is a single .env change, not a code change.

Development workflow:
    LLM_PROVIDER=groq     while building and debugging = free, fast iteration.
    LLM_PROVIDER=openai   for final experiment runs
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI

load_dotenv()

GROQ_BASE_URL="https://api.groq.com/openai/v1"

def get_provider() -> str:
  return os.getenv("LLM_PROVIDER","groq").lower()

def get_model_name() -> str:
  return os.getenv("LLM_MODEL","llama-3.1-8b-instant")

def get_llm_client() -> OpenAI:
  """Raw OpenAI-SDK client pointed at whichever provider is active."""
  provider=get_provider()

  if provider=="groq":
    return OpenAI(
      api_key=os.getenv("GROQ_API_KEY"),
      base_url=GROQ_BASE_URL,
    )
  elif provider =="openai":
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  else:
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}, Use'groq' or 'openai'.")
  
def get_chat_model() -> ChatOpenAI:
  """LangChain-style chat model, used by RAGAS internally."""
  provider= get_provider()
  model_name=get_model_name()

  if provider == "groq":
    return ChatOpenAI(
      model=model_name,
      api_key=os.getenv("GROQ_API_KEY"),
      base_url=GROQ_BASE_URL,
      temperature=0,
    )
  elif provider=="openai":
    return ChatOpenAI(
      model=model_name,
      api_key=os.getenv("OPENAI_API_KEY"),
      temperature=0,
    )
  else:
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}, Use 'groq' or 'openai'.")