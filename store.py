"""
store.py - the storage layer. The only module that touches files or Chroma.

Design: JSON (text + metadata) and NumPy (vectors) are the SOURCE OF TRUTH.
Chroma is a fast index that can always be rebuilt from them with sync_chroma().
Swapping storage later (Pinecone, Qdrant, ...) means editing only this file.
"""
import os
import json
import numpy as np

import config

_client = None  # Chroma client, created lazily


def _chroma():
    """Return a shared persistent Chroma client (created on first use)."""
    global _client
    if _client is None:
        import chromadb
        _client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return _client


def load_store():
    """Return (rows, vecs) from disk, or ([], empty array) if nothing is stored."""
    if not os.path.exists(config.JSON_PATH):
        return [], np.empty((0, 0), dtype="float32")
    return json.load(open(config.JSON_PATH)), np.load(config.NPY_PATH)


def save_store(rows, vecs):
    """Write the chunk rows to JSON and the vectors to a .npy file."""
    json.dump(rows, open(config.JSON_PATH, "w"), indent=2)
    np.save(config.NPY_PATH, vecs)


def get_collection():
    """Get (or create) the Chroma collection, configured for cosine distance."""
    return _chroma().get_or_create_collection(
        config.COLLECTION, metadata={"hnsw:space": "cosine"})


def sync_chroma():
    """Rebuild the Chroma index from JSON + NumPy (drops and recreates it)."""
    rows, vecs = load_store()
    try:
        _chroma().delete_collection(config.COLLECTION)
    except Exception:
        pass  # collection did not exist yet
    col = get_collection()
    for i in range(0, len(rows), 500):          # add in batches of 500
        b = rows[i:i + 500]
        col.add(ids=[r["id"] for r in b],
                documents=[r["content"] for r in b],
                metadatas=[{"source": r["source"]} for r in b],
                embeddings=vecs[i:i + 500].tolist())
