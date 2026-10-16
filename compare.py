"""
compare.py - side-by-side NumPy vs Chroma agreement and timing.
"""
import time
import retrieval
from tracing import span


def compare(question, k=5):
    """Print overlap between exact (numpy) and approximate (chroma) results."""
    t0 = time.perf_counter()
    a = retrieval.retrieve(question, k, "numpy")
    t1 = time.perf_counter()
    b = retrieval.retrieve(question, k, "chroma")
    t2 = time.perf_counter()
    set_a = {h["content"][:80] for h in a}
    set_b = {h["content"][:80] for h in b}
    overlap = len(set_a & set_b)
    print(f"\nNumPy vs Chroma (k={k}): overlap {overlap}/{k}")
    print(f"  numpy  {(t1-t0)*1000:.1f} ms")
    print(f"  chroma {(t2-t1)*1000:.1f} ms")
