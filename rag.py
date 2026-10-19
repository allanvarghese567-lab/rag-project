"""Compatibility shim — prefer `from ragcore.rag import ask`."""
from ragcore.rag import SYSTEM, ask  # noqa: F401

if __name__ == "__main__":
    from ragcore.rag import __name__ as _  # keep CLI via module
    import runpy

    runpy.run_module("ragcore.rag", run_name="__main__")
