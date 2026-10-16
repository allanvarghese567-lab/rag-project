"""
retrieval.py - every way of finding relevant chunks, behind one entry point.

Base backends : numpy, chroma, hybrid
Add "+rerank" : e.g. "chroma+rerank", "hybrid+rerank"

Scores are NOT comparable across backends: cosine (0-1), RRF (max ~0.033) and
cross-encoder logits (can be negative) are different scales.
"""
import config
import helpers
import store
import kwsearch
import rerank
from tracing import span


def retrieve_numpy(qvec, k=config.TOP_K):
    """Exact search: dot product against every stored vector (= cosine)."""
    rows, vecs = store.load_store()
    if not rows:
        return []
    sims = vecs @ qvec                       # vectors are normalized
    top = sims.argsort()[::-1][:k]
    return [{"content": rows[i]["content"], "source": rows[i]["source"],
             "score": float(sims[i])} for i in top]


def retrieve_chroma(qvec, k=config.TOP_K, source=None):
    """Approximate (HNSW) search via Chroma; optionally filter by source file."""
    r = store.get_collection().query(
        query_embeddings=[qvec.tolist()], n_results=k,
        where={"source": source} if source else None)
    return [{"content": d, "source": m["source"], "score": 1 - dist}
            for d, m, dist in zip(r["documents"][0], r["metadatas"][0],
                                  r["distances"][0])]


def rrf(rank_lists, k=config.TOP_K, c=config.RRF_C):
    """
    Reciprocal Rank Fusion: merge several ranked lists using rank positions
    only. A chunk earns 1/(c + rank) from each list it appears in.
    """
    scores, items = {}, {}
    for hits in rank_lists:
        for rank, h in enumerate(hits):
            key = h["content"]
            scores[key] = scores.get(key, 0) + 1 / (c + rank + 1)
            items.setdefault(key, h)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{**items[key], "score": round(scores[key], 4)} for key in best]


def retrieve_hybrid(question, qvec, k=config.TOP_K):
    """Run vector search and BM25 keyword search, then fuse them with RRF."""
    pool = config.HYBRID_POOL
    with span("vector_search", pool=pool):
        vec = retrieve_chroma(qvec, pool)
    with span("keyword_search", pool=pool):
        kw = kwsearch.retrieve_keyword(question, pool)
    with span("fuse"):
        return rrf([vec, kw], k)


def retrieve(question, k=config.TOP_K, backend="chroma", **kw):
    """
    Main entry point. Embeds the question once, searches with the chosen
    backend, and optionally reranks ("<backend>+rerank").
    """
    base, _, rr = backend.partition("+")
    pool = config.RERANK_POOL if rr == "rerank" else k
    qvec = helpers.embed([question])[0]

    with span(f"search_{base}", k=pool) as s:
        if base == "numpy":
            hits = retrieve_numpy(qvec, pool)
        elif base == "hybrid":
            hits = retrieve_hybrid(question, qvec, pool)
        else:
            hits = retrieve_chroma(qvec, pool, **kw)
        s["data"]["top_score"] = round(hits[0]["score"], 4) if hits else None

    if rr == "rerank":
        with span("rerank", pool=len(hits)) as s:
            hits = rerank.rerank(question, hits, k)
            s["data"]["top_score"] = hits[0]["score"] if hits else None
    return hits
