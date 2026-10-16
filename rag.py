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
from ingest import ingest
from compare import compare
from tracing import trace, span

SYSTEM = "Answer ONLY from the context. If it's not there, say you don't know."
_claude = None  # Anthropic client, created on first use


def _anthropic():
    """Return a shared Anthropic client (created lazily)."""
    global _claude
    if _claude is None:
        from anthropic import Anthropic
        _claude = Anthropic()
    return _claude


def ask(question, backend="chroma", k=config.TOP_K):
    """
    Answer a question from the stored documents.
    Returns (answer_text, hits). The whole run is traced: retrieval spans,
    plus an "llm" span with token counts and cost.
    """
    with trace("ask", question=question[:80], backend=backend):
        hits = retrieval.retrieve(question, k, backend)
        ctx = "\n\n---\n\n".join(h["content"] for h in hits)
        with span("llm", model=config.LLM_MODEL) as s:
            msg = _anthropic().messages.create(
                model=config.LLM_MODEL, max_tokens=800, system=SYSTEM,
                messages=[{"role": "user",
                           "content": f"Context:\n{ctx}\n\nQuestion: {question}"}])
            u = msg.usage
            s["data"].update(
                tokens_in=u.input_tokens, tokens_out=u.output_tokens,
                cost_usd=round((u.input_tokens * config.PRICE_IN +
                                u.output_tokens * config.PRICE_OUT) / 1e6, 5))
    return msg.content[0].text, hits


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
