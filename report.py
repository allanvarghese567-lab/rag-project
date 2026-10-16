"""
report.py - average / p95 latency per stage from traces.jsonl.
"""
import json
import statistics
from collections import defaultdict
import config


def main():
    if not open(config.TRACE_PATH).read().strip():
        print("No traces yet.")
        return
    stages = defaultdict(list)
    for line in open(config.TRACE_PATH):
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
