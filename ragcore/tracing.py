"""tracing.py - dependency-free observability (trace + span)."""
from __future__ import annotations

import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from ragcore import config

_trace: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "trace", default=None
)


@contextmanager
def trace(name: str, **meta) -> Iterator[dict[str, Any]]:
    t: dict[str, Any] = {
        "id": uuid.uuid4().hex[:8],
        "name": name,
        "meta": meta,
        "ts": time.time(),
        "spans": [],
    }
    token, start = _trace.set(t), time.perf_counter()
    try:
        yield t
    finally:
        t["total_ms"] = round((time.perf_counter() - start) * 1000, 1)
        _trace.reset(token)
        with open(config.TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(t) + "\n")
        show(t)


@contextmanager
def span(name: str, **data) -> Iterator[dict[str, Any]]:
    s: dict[str, Any] = {"name": name, "data": dict(data)}
    start = time.perf_counter()
    try:
        yield s
    finally:
        s["ms"] = round((time.perf_counter() - start) * 1000, 1)
        t = _trace.get()
        if t is not None:
            t["spans"].append(s)


def show(t: dict[str, Any]) -> None:
    if not config.SHOW_TRACES:
        return
    print(f"\n-- trace {t['id']} | {t['name']} | {t['total_ms']} ms --")
    for s in t["spans"]:
        extra = " ".join(f"{k}={v}" for k, v in s["data"].items())
        print(f"  {s['name']:<16}{s['ms']:>8} ms  {extra}")
