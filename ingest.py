"""
ingest.py - turn a PDF into stored, searchable chunks.

Pipeline: load -> chunk -> embed -> merge into store -> save -> sync Chroma.
Every stage is timed with a span inside one "ingest" trace.
"""
import os
import numpy as np

import helpers
import store
from tracing import trace, span


def ingest(path: str) -> int:
    """
    Add one PDF to the store and return how many chunks it produced.
    Re-ingesting a file with the same name replaces its old chunks.
    """
    name = os.path.basename(path)
    with trace("ingest", file=name):
        with span("load"):
            text = helpers.load(path)
        with span("chunk") as s:
            texts = helpers.chunk(text)
            s["data"]["chunks"] = len(texts)

        new_vecs = helpers.embed(texts)  # helpers adds its own "embed" span
        new_rows = [{"id": f"{name}-{i}", "source": name, "content": t}
                    for i, t in enumerate(texts)]

        rows, vecs = store.load_store()
        if rows and vecs.shape[1] != new_vecs.shape[1]:
            raise ValueError("Embedding size changed (%d -> %d). Delete store/ and re-ingest all docs."
                             % (vecs.shape[1], new_vecs.shape[1]))
        if rows:  # drop any old chunks from this same file, then append
            keep = [i for i, r in enumerate(rows) if r["source"] != name]
            rows, vecs = [rows[i] for i in keep], vecs[keep]
            vecs = np.vstack([vecs, new_vecs]) if len(rows) else new_vecs
        else:
            vecs = new_vecs
        rows += new_rows

        with span("save"):
            store.save_store(rows, vecs)
        with span("sync_chroma"):
            store.sync_chroma()
    return len(new_rows)
