"""
rag.py - entry point. Wires retrieval and the LLM together.

Usage:
    python rag.py                                   # ingest notes.pdf, ask a demo question
    python rag.py my.pdf "What is the refund window?"
    python rag.py my.pdf "..." hybrid+rerank        # pick a backend
"""
from __future__ import annotations

import sys
from typing import Any

import config
import llm
import retrieval
from compare import compare
from ingest import ingest
from tracing import span, trace

SYSTEM = "Answer ONLY from the context. If it's not there, say you don't know."


def ask(
    question: str,
    backend: str = "chroma",
    k: int = config.TOP_K,
    llm_profile: str | dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Answer a question from the stored documents.
    Returns (answer_text, hits). The whole run is traced: retrieval spans,
    plus an "llm" span with token counts and cost.
    llm_profile: optional profile dict or id string; default = active_llm().
    """
    prof = config.resolve_llm_profile(llm_profile)

    with trace("ask", question=question[:80], backend=backend):
        hits = retrieval.retrieve(question, k, backend)
        ctx = "\n\n---\n\n".join(h["content"] for h in hits)
        with span("llm", model=prof["model"], profile=prof["id"]) as s:
            text, usage = llm.complete(
                SYSTEM,
                f"Context:\n{ctx}\n\nQuestion: {question}",
                profile=prof,
            )
            s["data"].update(usage)
    return text, hits


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "notes.pdf"
    q = sys.argv[2] if len(sys.argv) > 2 else "What does the document say about X?"
    backend = sys.argv[3] if len(sys.argv) > 3 else "chroma"

    print("Ingested", ingest(pdf), "chunks")
    answer, hits = ask(q, backend)
    print("\n" + answer + "\n")
    for h in hits:
        print(round(h["score"], 3), h["source"])
    compare(q)
