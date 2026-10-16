"""
config.py - the single place for every setting in the project.

Change a model, a path or a tuning knob here and every other module picks it
up. Other modules import this file; it imports nothing from the project, so it
can never cause a circular import.
"""
import os
import json

# Optional: load API keys from a .env file if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------- storage paths ----------
STORE_DIR = "store"                          # everything generated lives here
JSON_PATH = f"{STORE_DIR}/chunks.json"       # source of truth: text + metadata
NPY_PATH = f"{STORE_DIR}/vecs.npy"           # source of truth: normalized vectors
CHROMA_PATH = f"{STORE_DIR}/chroma"          # rebuildable Chroma index
TRACE_PATH = f"{STORE_DIR}/traces.jsonl"     # one JSON line per traced run
EVAL_SET_PATH = "eval_set.json"              # your hand-written test questions
EVAL_RESULTS_PATH = f"{STORE_DIR}/evals.jsonl"  # one summary line per eval run
COLLECTION = "docs"                          # Chroma collection name

# ---------- models (fallback defaults; prefer config/llms.json) ----------
# Kept for backward compatibility; active_llm() is the source of truth.
LLM_MODEL = "claude-sonnet-4-5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # local reranker

# ---------- chunking ----------
CHUNK_SIZE = 1500                            # characters per chunk (~375 tokens)
CHUNK_OVERLAP = 200                          # characters shared by neighbours

# ---------- retrieval ----------
TOP_K = 5                                    # chunks sent to the LLM
HYBRID_POOL = 20                             # candidates per search before fusing
RRF_C = 60                                   # standard Reciprocal Rank Fusion constant
RERANK_POOL = 20                             # candidates fetched before reranking to k

# ---------- observability ----------
SHOW_TRACES = True                           # print a trace table after each run
# USD per 1M tokens — used when a profile omits price_in / price_out
PRICE_IN = 0.0
PRICE_OUT = 0.0

os.makedirs(STORE_DIR, exist_ok=True)

# ---------- embedding profiles (managed by the UI) ----------
EMBED_CONFIG_PATH = "config/embeddings.json"

# ---------- LLM profiles (managed by the UI / config/llms.json) ----------
LLM_CONFIG_PATH = "config/llms.json"


def active_embedding():
    """Return the active embedding profile dict (env EMBED_PROFILE overrides the file)."""
    cfg = json.load(open(EMBED_CONFIG_PATH))
    pid = os.getenv("EMBED_PROFILE") or cfg["active"]
    return next(p for p in cfg["profiles"] if p["id"] == pid)


def active_llm():
    """Return the active LLM profile dict (env LLM_PROFILE overrides the file)."""
    if not os.path.exists(LLM_CONFIG_PATH):
        # Fallback so older checkouts still work
        return {
            "id": "fallback",
            "provider": "anthropic",
            "model": LLM_MODEL,
            "api_key_env": "ANTHROPIC_API_KEY",
            "max_tokens": 800,
            "price_in": PRICE_IN,
            "price_out": PRICE_OUT,
        }
    cfg = json.load(open(LLM_CONFIG_PATH))
    pid = os.getenv("LLM_PROFILE") or cfg["active"]
    return next(p for p in cfg["profiles"] if p["id"] == pid)
