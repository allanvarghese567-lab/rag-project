"""evals.py - offline eval set scoring."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

from ragcore import config, rag

JUDGE_SYS = (
    "You grade answers. Given a question, a reference answer and a candidate "
    "answer, reply with ONLY JSON: "
    '{"correct": true or false, "reason": "<one sentence>"}. '
    "Correct means the candidate agrees with the reference on the key facts. "
    "'I don't know' is incorrect unless the reference says the info is absent."
)


def judge(q: str, ref: str, cand: str) -> dict[str, Any]:
    from ragcore import llm

    text, _ = llm.complete(
        JUDGE_SYS,
        f"Question: {q}\nReference: {ref}\nCandidate: {cand}",
        max_tokens=200,
    )
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        return {"correct": False, "reason": "judge parse error"}


def run(label: str, backend: str = "chroma", k: int = config.TOP_K) -> None:
    with open(config.EVAL_SET_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    rows: list[dict[str, Any]] = []
    config.SHOW_TRACES = False
    try:
        for c in cases:
            t = time.perf_counter()
            answer, hits = rag.ask(c["question"], backend, k)
            ms = (time.perf_counter() - t) * 1000
            blob = " ".join(h["content"] for h in hits).lower()
            hit = any(kw.lower() in blob for kw in c["keywords"])
            v = judge(c["question"], c["expected_answer"], answer)
            rows.append(
                {
                    "q": c["question"],
                    "hit": hit,
                    "correct": bool(v.get("correct")),
                    "reason": v.get("reason", ""),
                    "ms": ms,
                }
            )
    finally:
        config.SHOW_TRACES = True

    n = len(rows)
    if n == 0:
        print("No eval cases found.")
        return
    summary = {
        "label": label,
        "backend": backend,
        "k": k,
        "chunk_size": config.CHUNK_SIZE,
        "overlap": config.CHUNK_OVERLAP,
        "n": n,
        "retrieval_hit": round(sum(r["hit"] for r in rows) / n, 3),
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "avg_ms": round(statistics.mean(r["ms"] for r in rows), 1),
    }
    with open(config.EVAL_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    diagnose(rows)
    print("\n", summary)


def diagnose(rows: list[dict[str, Any]]) -> None:
    both = sum(r["hit"] and r["correct"] for r in rows)
    gen_fail = [r for r in rows if r["hit"] and not r["correct"]]
    ret_fail = [r for r in rows if not r["hit"] and not r["correct"]]
    lucky = sum((not r["hit"]) and r["correct"] for r in rows)
    print(f"\nWorking:                  {both}")
    print(f"Retrieved but wrong:      {len(gen_fail)}  (prompt / LLM problem)")
    print(f"Not retrieved, wrong:     {len(ret_fail)}  (chunking / k / embedding problem)")
    print(f"Wrong keywords or lucky:  {lucky}  (check your keywords)")
    for r in gen_fail + ret_fail:
        print(f" x {r['q']}\n   {r['reason']}")


def report() -> None:
    print(f"{'label':<18}{'backend':<16}{'k':<3}{'chunk':<7}{'hit':<7}{'acc':<7}{'ms':<8}")
    try:
        with open(config.EVAL_RESULTS_PATH, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("(no eval runs yet)")
        return
    for line in lines:
        s = json.loads(line)
        print(
            f"{s['label']:<18}{s['backend']:<16}{s['k']:<3}{s['chunk_size']:<7}"
            f"{s['retrieval_hit']:<7}{s['accuracy']:<7}{s['avg_ms']:<8}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="baseline")
    p.add_argument("--backend", default="chroma")
    p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    report() if a.report else run(a.label, a.backend, a.k)
