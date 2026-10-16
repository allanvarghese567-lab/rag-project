"""
rerank.py - cross-encoder reranker (local sentence-transformers model).
"""
import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(config.RERANK_MODEL)
    return _model


def rerank(question, hits, k):
    """Score (question, chunk) pairs and keep the top-k."""
    if not hits:
        return []
    model = _get_model()
    pairs = [[question, h["content"]] for h in hits]
    scores = model.predict(pairs)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**hits[i], "score": float(scores[i])} for i in order]
