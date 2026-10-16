"""
llm.py - pluggable chat/completion providers for answers and the eval judge.

The active profile comes from config/llms.json (same pattern as embeddings).
Providers:
  anthropic          - Anthropic Messages API
  openai             - OpenAI Chat Completions
  openai_compatible  - any OpenAI-compatible gateway (set base_url)

Keys are read from the env var named in the profile (never stored in the file).
"""
from __future__ import annotations

import os
from typing import Any

import config

_clients: dict[str, Any] = {}


def _anthropic(p: dict, system: str, user: str, max_tokens: int) -> tuple[str, dict]:
    from anthropic import Anthropic
    if p["id"] not in _clients:
        _clients[p["id"]] = Anthropic(api_key=os.getenv(p.get("api_key_env", "ANTHROPIC_API_KEY")))
    msg = _clients[p["id"]].messages.create(
        model=p["model"],
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = msg.content[0].text
    u = msg.usage
    return text, {
        "tokens_in": u.input_tokens,
        "tokens_out": u.output_tokens,
    }


def _openai_like(p: dict, system: str, user: str, max_tokens: int) -> tuple[str, dict]:
    from openai import OpenAI
    if p["id"] not in _clients:
        _clients[p["id"]] = OpenAI(
            api_key=os.getenv(p.get("api_key_env", "OPENAI_API_KEY")),
            base_url=p.get("base_url") or None,
        )
    resp = _clients[p["id"]].chat.completions.create(
        model=p["model"],
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content or ""
    u = resp.usage
    return text, {
        "tokens_in": getattr(u, "prompt_tokens", 0) or 0,
        "tokens_out": getattr(u, "completion_tokens", 0) or 0,
    }


PROVIDERS = {
    "anthropic": _anthropic,
    "openai": _openai_like,
    "openai_compatible": _openai_like,
}


def complete(system: str, user: str, profile: dict | None = None, max_tokens: int | None = None) -> tuple[str, dict]:
    """
    Run one chat completion with the given (or active) profile.
    Returns (text, usage) where usage has tokens_in, tokens_out, cost_usd, model, profile_id.
    """
    p = profile or config.active_llm()
    fn = PROVIDERS.get(p["provider"])
    if fn is None:
        raise ValueError(f"Unknown LLM provider: {p.get('provider')!r}. "
                         f"Known: {', '.join(PROVIDERS)}")
    mt = max_tokens if max_tokens is not None else int(p.get("max_tokens") or 800)
    text, usage = fn(p, system, user, mt)
    pin = float(p.get("price_in") or config.PRICE_IN or 0)
    pout = float(p.get("price_out") or config.PRICE_OUT or 0)
    usage["cost_usd"] = round(
        (usage["tokens_in"] * pin + usage["tokens_out"] * pout) / 1e6, 5)
    usage["model"] = p["model"]
    usage["profile_id"] = p["id"]
    return text, usage
