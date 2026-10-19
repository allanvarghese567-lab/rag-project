"""
store.py - the storage layer. The only module that touches files or Chroma.

Design: JSON (text + metadata) and NumPy (vectors) are the SOURCE OF TRUTH.
Chroma is a fast index that can always be rebuilt from them with sync_chroma().
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from ragcore import config

_client = None


def _chroma():
    global _client
    if _client is None:
        import chromadb

        _client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return _client


def load_store() -> tuple[list[dict[str, Any]], np.ndarray]:
    if not os.path.exists(config.JSON_PATH):
        return [], np.empty((0, 0), dtype="float32")
    with open(config.JSON_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    vecs = np.load(config.NPY_PATH)
    return rows, vecs


def save_store(rows: list[dict[str, Any]], vecs: np.ndarray) -> None:
    with open(config.JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    np.save(config.NPY_PATH, vecs)


def get_collection():
    return _chroma().get_or_create_collection(
        config.COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def sync_chroma() -> None:
    rows, vecs = load_store()
    try:
        _chroma().delete_collection(config.COLLECTION)
    except Exception:
        pass
    col = get_collection()
    for i in range(0, len(rows), 500):
        batch = rows[i : i + 500]
        col.add(
            ids=[r["id"] for r in batch],
            documents=[r["content"] for r in batch],
            metadatas=[{"source": r["source"]} for r in batch],
            embeddings=vecs[i : i + 500].tolist(),
        )
