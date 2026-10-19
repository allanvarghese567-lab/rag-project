"""Compatibility shim — prefer `python -m ragcore.report`."""
from ragcore.report import main  # noqa: F401

if __name__ == "__main__":
    main()
