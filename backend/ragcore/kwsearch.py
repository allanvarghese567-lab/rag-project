"""kwsearch.py - BM25 keyword search."""
from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

from ragcore import store


def retrieve_keyword(question: str, k: int = 20) -> list[dict[str, Any]]:
    rows, _ = store.load_store()
    if not rows:
        return []
    bm25 = BM25Okapi([r["content"].lower().split() for r in rows])
    scores = bm25.get_scores(question.lower().split())
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {
            "content": rows[i]["content"],
            "source": rows[i]["source"],
            "score": float(scores[i]),
        }
        for i in top
    ]
