"""
helpers.py - basic building blocks: load, chunk, embed.
Depends only on config, embeddings and tracing.
"""
from __future__ import annotations

import numpy as np
from pypdf import PdfReader

import config
import embeddings
from tracing import span


def load(path: str) -> str:
    """Read a PDF and return all of its text as one string."""
    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)


def chunk(
    text: str,
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping fixed-size pieces (sizes in characters)."""
    if not text or size <= 0:
        return []
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts with the ACTIVE profile in config/embeddings.json (unit-length vectors)."""
    with span("embed", n_texts=len(texts), profile=config.active_embedding()["id"]):
        return embeddings.embed_texts(texts)
