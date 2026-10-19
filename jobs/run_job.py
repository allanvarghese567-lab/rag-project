"""
run_job.py - what the GitHub Actions job executes.

Indexes every PDF in docs/ with the active embedding profile when no store exists.
Optionally runs evals (RUN_EVALS=true). Writes a summary to the Actions UI.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragcore import config  # noqa: E402


def main() -> None:
    prof, t0, counts = config.active_embedding(), time.time(), {}
    if not os.path.exists(config.JSON_PATH):
        from ragcore.ingest import ingest

        for pdf in sorted(glob.glob("docs/*.pdf")):
            counts[os.path.basename(pdf)] = ingest(pdf)
    if os.getenv("RUN_EVALS") == "true":
        from ragcore import evals

        evals.run(f"job-{prof['id']}", "chroma")
    summary = {
        "profile": prof["id"],
        "model": prof["model"],
        "chunks": counts,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(config.STORE_DIR, exist_ok=True)
    with open(f"{config.STORE_DIR}/run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    md = (
        f"### Reindex\n- profile: `{prof['id']}` ({prof['provider']} / {prof['model']})\n"
        f"- chunks: {counts or 'index already present, nothing to embed'}\n"
        f"- took {summary['seconds']}s\n"
    )
    print(md)
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as f:
            f.write(md)


if __name__ == "__main__":
    main()
