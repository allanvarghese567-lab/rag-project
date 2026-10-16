"""
rag.py - entry point. Wires retrieval and the LLM together.

Usage:
    python rag.py                                   # ingest notes.pdf, ask a demo question
    python rag.py my.pdf "What is the refund window?"
    python rag.py my.pdf "..." hybrid+rerank        # pick a backend
"""
import sys

import config
import retrieval
import llm
from ingest import ingest
from compare import compare
from tracing import trace, span

SYSTEM = "Answer ONLY from the context. If it's not there, say you don't know."


def ask(question, backend="chroma", k=config.TOP_K, llm_profile=None):
    """
    Answer a question from the stored documents.
    Returns (answer_text, hits). The whole run is traced: retrieval spans,
    plus an "llm" span with token counts and cost.
    llm_profile: optional profile dict or id string; default = active_llm().
    """
    if isinstance(llm_profile, str):
        cfg = __import__("json").load(open(config.LLM_CONFIG_PATH))
        llm_profile = next(p for p in cfg["profiles"] if p["id"] == llm_profile)

    with trace("ask", question=question[:80], backend=backend):
        hits = retrieval.retrieve(question, k, backend)
        ctx = "\n\n---\n\n".join(h["content"] for h in hits)
        prof = llm_profile or config.active_llm()
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
