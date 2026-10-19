"""Compatibility shim — prefer `from ragcore import evals`."""
from ragcore.evals import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("ragcore.evals", run_name="__main__")
