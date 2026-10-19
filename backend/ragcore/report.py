"""report.py - latency stats from traces.jsonl."""
from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict

from ragcore import config


def main() -> None:
    if not os.path.exists(config.TRACE_PATH):
        print("No traces yet.")
        return
    with open(config.TRACE_PATH, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        print("No traces yet.")
        return
    stages: dict[str, list[float]] = defaultdict(list)
    for line in content.splitlines():
        t = json.loads(line)
        for s in t["spans"]:
            stages[s["name"]].append(s["ms"])
    print(f"{'stage':<20} {'n':>5} {'avg':>8} {'p95':>8}")
    for name, ms in sorted(stages.items()):
        avg = statistics.mean(ms)
        p95 = statistics.quantiles(ms, n=20)[18] if len(ms) >= 20 else max(ms)
        print(f"{name:<20} {len(ms):>5} {avg:>8.1f} {p95:>8.1f}")


if __name__ == "__main__":
    main()
