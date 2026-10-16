"""
evals.py - score the app against a hand-written question set.

Per question it records:
  retrieval hit : did any retrieved chunk contain an expected keyword?
  correct       : does an LLM judge agree with your reference answer?
Comparing the two tells you WHERE a failure happened (retrieval vs generation).

Usage:
    python evals.py --label baseline
    python evals.py --label hybrid --backend hybrid --k 5
    python evals.py --report
"""
import json
import time
import argparse
import statistics

import config
import rag

_judge_client = None

JUDGE_SYS = (
    "You grade answers. Given a question, a reference answer and a candidate "
    "answer, reply with ONLY JSON: "
    '{"correct": true or false, "reason": "<one sentence>"}. '
    "Correct means the candidate agrees with the reference on the key facts. "
    "'I don't know' is incorrect unless the reference says the info is absent."
)


def _anthropic():
    """Return a shared Anthropic client for the judge (created lazily)."""
    global _judge_client
    if _judge_client is None:
        from anthropic import Anthropic
        _judge_client = Anthropic()
    return _judge_client


def judge(q, ref, cand):
    """Ask the LLM to grade one answer. Returns {"correct": bool, "reason": str}."""
    msg = _anthropic().messages.create(
        model=config.LLM_MODEL, max_tokens=200, system=JUDGE_SYS,
        messages=[{"role": "user",
                   "content": f"Question: {q}\nReference: {ref}\nCandidate: {cand}"}])
    text = msg.content[0].text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        return {"correct": False, "reason": "judge parse error"}


def run(label, backend="chroma", k=config.TOP_K):
    """Run every eval question, print a diagnosis, append a summary line."""
    cases = json.load(open(config.EVAL_SET_PATH))
    rows = []
    config.SHOW_TRACES = False               # keep the eval output readable
    try:
        for c in cases:
            t = time.perf_counter()
            answer, hits = rag.ask(c["question"], backend, k)
            ms = (time.perf_counter() - t) * 1000
            blob = " ".join(h["content"] for h in hits).lower()
            hit = any(kw.lower() in blob for kw in c["keywords"])
            v = judge(c["question"], c["expected_answer"], answer)
            rows.append({"q": c["question"], "hit": hit,
                         "correct": bool(v.get("correct")),
                         "reason": v.get("reason", ""), "ms": ms})
    finally:
        config.SHOW_TRACES = True

    n = len(rows)
    summary = {
        "label": label, "backend": backend, "k": k,
        "chunk_size": config.CHUNK_SIZE, "overlap": config.CHUNK_OVERLAP,
        "n": n,
        "retrieval_hit": round(sum(r["hit"] for r in rows) / n, 3),
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "avg_ms": round(statistics.mean(r["ms"] for r in rows), 1),
    }
    with open(config.EVAL_RESULTS_PATH, "a") as f:
        f.write(json.dumps(summary) + "\n")
    diagnose(rows)
    print("\n", summary)


def diagnose(rows):
    """Bucket the results to show whether failures are retrieval or generation."""
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


def report():
    """Print a comparison table of every past eval run."""
    print(f"{'label':<18}{'backend':<16}{'k':<3}{'chunk':<7}{'hit':<7}{'acc':<7}{'ms':<8}")
    try:
        lines = open(config.EVAL_RESULTS_PATH).readlines()
    except FileNotFoundError:
        print("(no eval runs yet)")
        return
    for line in lines:
        s = json.loads(line)
        print(f"{s['label']:<18}{s['backend']:<16}{s['k']:<3}{s['chunk_size']:<7}"
              f"{s['retrieval_hit']:<7}{s['accuracy']:<7}{s['avg_ms']:<8}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="baseline")
    p.add_argument("--backend", default="chroma",
                   help="numpy | chroma | hybrid, optionally +rerank")
    p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    report() if a.report else run(a.label, a.backend, a.k)
