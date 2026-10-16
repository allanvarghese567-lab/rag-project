"""
tracing.py - tiny, dependency-free observability.

Vocabulary:
  trace = one full run (one `ask` or one `ingest`)
  span  = one timed stage inside a trace (embed, search, llm, ...)

Every finished trace is appended to store/traces.jsonl and optionally printed.
Spans used outside a trace are harmless: they time themselves and record nothing.
"""
import time
import json
import uuid
import contextvars
from contextlib import contextmanager

import config

# Holds the trace that is currently running (None when no trace is active).
_trace = contextvars.ContextVar("trace", default=None)


@contextmanager
def trace(name, **meta):
    """Open a trace for one full run; on exit save it to disk and print it."""
    t = {"id": uuid.uuid4().hex[:8], "name": name, "meta": meta,
         "ts": time.time(), "spans": []}
    token, start = _trace.set(t), time.perf_counter()
    try:
        yield t
    finally:
        t["total_ms"] = round((time.perf_counter() - start) * 1000, 1)
        _trace.reset(token)
        with open(config.TRACE_PATH, "a") as f:
            f.write(json.dumps(t) + "\n")
        show(t)


@contextmanager
def span(name, **data):
    """
    Time one stage. Yields a dict so the caller can attach details after the
    work is done, e.g. `s["data"]["top_score"] = 0.81`.
    """
    s, start = {"name": name, "data": dict(data)}, time.perf_counter()
    try:
        yield s
    finally:
        s["ms"] = round((time.perf_counter() - start) * 1000, 1)
        t = _trace.get()
        if t is not None:           # only record if a trace is active
            t["spans"].append(s)


def show(t):
    """Print one trace as a readable table (skipped when SHOW_TRACES is False)."""
    if not config.SHOW_TRACES:
        return
    print(f"\n-- trace {t['id']} | {t['name']} | {t['total_ms']} ms --")
    for s in t["spans"]:
        extra = " ".join(f"{k}={v}" for k, v in s["data"].items())
        print(f"  {s['name']:<16}{s['ms']:>8} ms  {extra}")
