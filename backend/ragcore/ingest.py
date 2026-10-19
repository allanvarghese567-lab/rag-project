"""ingest.py - PDF → chunks → store → Chroma."""
from __future__ import annotations

import os

import numpy as np

from ragcore import helpers, store
from ragcore.tracing import span, trace


def ingest(path: str) -> int:
    name = os.path.basename(path)
    with trace("ingest", file=name):
        with span("load"):
            text = helpers.load(path)
        with span("chunk") as s:
            texts = helpers.chunk(text)
            s["data"]["chunks"] = len(texts)

        if not texts:
            return 0

        new_vecs = helpers.embed(texts)
        new_rows = [
            {"id": f"{name}-{i}", "source": name, "content": t}
            for i, t in enumerate(texts)
        ]

        rows, vecs = store.load_store()
        if rows and vecs.shape[1] != new_vecs.shape[1]:
            raise ValueError(
                "Embedding size changed (%d -> %d). Delete store/ and re-ingest all docs."
                % (vecs.shape[1], new_vecs.shape[1])
            )
        if rows:
            keep = [i for i, r in enumerate(rows) if r["source"] != name]
            rows = [rows[i] for i in keep]
            vecs = (
                vecs[keep]
                if keep
                else np.empty((0, new_vecs.shape[1]), dtype="float32")
            )
            vecs = np.vstack([vecs, new_vecs]) if len(rows) else new_vecs
        else:
            vecs = new_vecs
        rows += new_rows

        with span("save"):
            store.save_store(rows, vecs)
        with span("sync_chroma"):
            store.sync_chroma()
    return len(new_rows)
