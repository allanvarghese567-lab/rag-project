"""
run_job.py - what the GitHub Actions job executes.

Indexes every PDF in docs/ with the active embedding profile when no store exists
(Actions restores store/ from cache when nothing changed, so unchanged runs skip
this). Optionally runs evals (RUN_EVALS=true). Writes a summary to the Actions UI.
"""
import os
import sys
import glob
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def main():
    """Index docs, optionally run evals, and write store/run_summary.json + job summary."""
    prof, t0, counts = config.active_embedding(), time.time(), {}
    if not os.path.exists(config.JSON_PATH):
        from ingest import ingest
        for pdf in sorted(glob.glob("docs/*.pdf")):
            counts[os.path.basename(pdf)] = ingest(pdf)
    if os.getenv("RUN_EVALS") == "true":
        import evals
        evals.run(f"job-{prof['id']}", "chroma")
    summary = {"profile": prof["id"], "model": prof["model"], "chunks": counts,
               "seconds": round(time.time() - t0, 1)}
    json.dump(summary, open(f"{config.STORE_DIR}/run_summary.json", "w"), indent=2)
    md = (f"### Reindex\n- profile: `{prof['id']}` ({prof['provider']} / {prof['model']})\n"
          f"- chunks: {counts or 'index already present, nothing to embed'}\n- took {summary['seconds']}s\n")
    print(md)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(md)


if __name__ == "__main__":
    main()
