"""rerank.py - cross-encoder reranker."""
from __future__ import annotations

from typing import Any

from ragcore import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(config.RERANK_MODEL)
    return _model


def rerank(question: str, hits: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if not hits:
        return []
    model = _get_model()
    pairs = [[question, h["content"]] for h in hits]
    scores = model.predict(pairs)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**hits[i], "score": float(scores[i])} for i in order]
