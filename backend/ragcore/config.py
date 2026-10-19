"""
config.py - single place for every setting.

Paths are resolved relative to the backend/ directory so the app works
regardless of process cwd (uvicorn, pytest, Actions).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# backend/ (parent of ragcore/)
BACKEND_ROOT = Path(__file__).resolve().parent.parent

# ---------- storage paths ----------
STORE_DIR = str(BACKEND_ROOT / "store")
JSON_PATH = str(BACKEND_ROOT / "store" / "chunks.json")
NPY_PATH = str(BACKEND_ROOT / "store" / "vecs.npy")
CHROMA_PATH = str(BACKEND_ROOT / "store" / "chroma")
TRACE_PATH = str(BACKEND_ROOT / "store" / "traces.jsonl")
EVAL_SET_PATH = str(BACKEND_ROOT / "eval_set.json")
EVAL_RESULTS_PATH = str(BACKEND_ROOT / "store" / "evals.jsonl")
COLLECTION = "docs"

# ---------- models ----------
LLM_MODEL = "claude-sonnet-4-5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------- chunking ----------
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# ---------- retrieval ----------
TOP_K = 5
HYBRID_POOL = 20
RRF_C = 60
RERANK_POOL = 20

# ---------- observability ----------
SHOW_TRACES = True
PRICE_IN = 0.0
PRICE_OUT = 0.0

os.makedirs(STORE_DIR, exist_ok=True)

EMBED_CONFIG_PATH = str(BACKEND_ROOT / "config" / "embeddings.json")
LLM_CONFIG_PATH = str(BACKEND_ROOT / "config" / "llms.json")

# Repo-level docs/ for reindex job (sibling of backend/)
DOCS_DIR = str(BACKEND_ROOT.parent / "docs")


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def active_embedding() -> dict[str, Any]:
    cfg = load_json(EMBED_CONFIG_PATH)
    pid = os.getenv("EMBED_PROFILE") or cfg["active"]
    try:
        return next(p for p in cfg["profiles"] if p["id"] == pid)
    except StopIteration as e:
        raise ValueError(
            f"Embedding profile {pid!r} not found in {EMBED_CONFIG_PATH}"
        ) from e


def active_llm() -> dict[str, Any]:
    if not os.path.exists(LLM_CONFIG_PATH):
        return {
            "id": "fallback",
            "provider": "anthropic",
            "model": LLM_MODEL,
            "api_key_env": "ANTHROPIC_API_KEY",
            "max_tokens": 800,
            "price_in": PRICE_IN,
            "price_out": PRICE_OUT,
        }
    cfg = load_json(LLM_CONFIG_PATH)
    pid = os.getenv("LLM_PROFILE") or cfg["active"]
    try:
        return next(p for p in cfg["profiles"] if p["id"] == pid)
    except StopIteration as e:
        raise ValueError(f"LLM profile {pid!r} not found in {LLM_CONFIG_PATH}") from e


def resolve_llm_profile(profile: str | dict[str, Any] | None = None) -> dict[str, Any]:
    if profile is None:
        return active_llm()
    if isinstance(profile, dict):
        return profile
    cfg = load_json(LLM_CONFIG_PATH)
    try:
        return next(p for p in cfg["profiles"] if p["id"] == profile)
    except StopIteration as e:
        raise ValueError(f"LLM profile {profile!r} not found in {LLM_CONFIG_PATH}") from e
