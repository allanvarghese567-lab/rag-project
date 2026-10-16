"""
kwsearch.py - BM25 keyword search over the stored chunks.
"""
from rank_bm25 import BM25Okapi
import store


def retrieve_keyword(question, k=20):
    """Return top-k chunks by BM25 score."""
    rows, _ = store.load_store()
    if not rows:
        return []
    bm25 = BM25Okapi([r["content"].lower().split() for r in rows])
    scores = bm25.get_scores(question.lower().split())
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{"content": rows[i]["content"], "source": rows[i]["source"],
             "score": float(scores[i])} for i in top]
