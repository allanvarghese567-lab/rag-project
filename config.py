"""Compatibility shim — prefer `from ragcore import config`."""
from ragcore.config import *  # noqa: F401,F403
from ragcore.config import (  # noqa: F401
    active_embedding,
    active_llm,
    load_json,
    resolve_llm_profile,
)
