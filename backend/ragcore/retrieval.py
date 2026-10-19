"""retrieval.py - numpy / chroma / hybrid / +rerank."""
from __future__ import annotations

from typing import Any

from ragcore import config, helpers, kwsearch, rerank, store
from ragcore.tracing import span


def retrieve_numpy(qvec, k: int = config.TOP_K) -> list[dict[str, Any]]:
    rows, vecs = store.load_store()
    if not rows:
        return []
    sims = vecs @ qvec
    top = sims.argsort()[::-1][:k]
    return [
        {
            "content": rows[i]["content"],
            "source": rows[i]["source"],
            "score": float(sims[i]),
        }
        for i in top
    ]


def retrieve_chroma(
    qvec, k: int = config.TOP_K, source: str | None = None
) -> list[dict[str, Any]]:
    r = store.get_collection().query(
        query_embeddings=[qvec.tolist()],
        n_results=k,
        where={"source": source} if source else None,
    )
    docs = r.get("documents") or [[]]
    metas = r.get("metadatas") or [[]]
    dists = r.get("distances") or [[]]
    if not docs[0]:
        return []
    return [
        {"content": d, "source": m["source"], "score": 1 - dist}
        for d, m, dist in zip(docs[0], metas[0], dists[0])
    ]


def rrf(
    rank_lists: list[list[dict[str, Any]]],
    k: int = config.TOP_K,
    c: int = config.RRF_C,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}
    for hits in rank_lists:
        for rank, h in enumerate(hits):
            key = h["content"]
            scores[key] = scores.get(key, 0) + 1 / (c + rank + 1)
            items.setdefault(key, h)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{**items[key], "score": round(scores[key], 4)} for key in best]


def retrieve_hybrid(question: str, qvec, k: int = config.TOP_K) -> list[dict[str, Any]]:
    pool = config.HYBRID_POOL
    with span("vector_search", pool=pool):
        vec = retrieve_chroma(qvec, pool)
    with span("keyword_search", pool=pool):
        kw = kwsearch.retrieve_keyword(question, pool)
    with span("fuse"):
        return rrf([vec, kw], k)


def retrieve(
    question: str, k: int = config.TOP_K, backend: str = "chroma", **kw
) -> list[dict[str, Any]]:
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
