"""helpers.py - load, chunk, embed."""
from __future__ import annotations

import numpy as np
from pypdf import PdfReader

from ragcore import config, embeddings
from ragcore.tracing import span


def load(path: str) -> str:
    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)


def chunk(
    text: str,
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    if not text or size <= 0:
        return []
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]


def embed(texts: list[str]) -> np.ndarray:
    with span("embed", n_texts=len(texts), profile=config.active_embedding()["id"]):
        return embeddings.embed_texts(texts)
