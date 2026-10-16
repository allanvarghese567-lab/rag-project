"""
config.py - the single place for every setting in the project.

Change a model, a path or a tuning knob here and every other module picks it
up. Other modules import this file; it imports nothing from the project, so it
can never cause a circular import.
"""
import os

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

# ---------- models ----------
LLM_MODEL = "claude-sonnet-5-5"              # answers + eval judge
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
# USD per 1M tokens. Fill in from the provider's pricing page (0 = cost shows 0).
PRICE_IN = 0.0
PRICE_OUT = 0.0

os.makedirs(STORE_DIR, exist_ok=True)

# ---------- embedding profiles (managed by the UI) ----------
import json
EMBED_CONFIG_PATH = "config/embeddings.json"


def active_embedding():
    """Return the active embedding profile dict (env EMBED_PROFILE overrides the file)."""
    cfg = json.load(open(EMBED_CONFIG_PATH))
    pid = os.getenv("EMBED_PROFILE") or cfg["active"]
    return next(p for p in cfg["profiles"] if p["id"] == pid)
