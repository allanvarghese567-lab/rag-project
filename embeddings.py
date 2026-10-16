"""
embeddings.py - pluggable embedding providers.

The active profile comes from config/embeddings.json (editable from the UI).
To add a provider: write one function f(profile, texts) -> list of vectors and
register it in PROVIDERS. Keys are read from the env var named in the profile
(never stored in the file).
"""
import os
import numpy as np

import config

_clients = {}  # cached clients / models, one per profile id


def _openai_like(p, texts):
    """OpenAI or any OpenAI-compatible endpoint (set base_url for the latter)."""
    from openai import OpenAI
    if p["id"] not in _clients:
        _clients[p["id"]] = OpenAI(api_key=os.getenv(p.get("api_key_env", "OPENAI_API_KEY")),
                                   base_url=p.get("base_url") or None)
    kw = {"model": p["model"], "input": texts}
    if p.get("dimensions"):
        kw["dimensions"] = int(p["dimensions"])   # text-embedding-3 models only
    return [d.embedding for d in _clients[p["id"]].embeddings.create(**kw).data]


def _sentence_transformers(p, texts):
    """Local model via sentence-transformers (downloaded on first use)."""
    from sentence_transformers import SentenceTransformer
    if p["id"] not in _clients:
        _clients[p["id"]] = SentenceTransformer(p["model"])
    return _clients[p["id"]].encode(texts, normalize_embeddings=True).tolist()


def _ollama(p, texts):
    """Local Ollama server (not reachable from GitHub Actions runners)."""
    import requests
    url = p.get("base_url") or "http://localhost:11434"
    r = requests.post(f"{url}/api/embed", json={"model": p["model"], "input": texts}, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"]


PROVIDERS = {"openai": _openai_like, "openai_compatible": _openai_like,
             "sentence_transformers": _sentence_transformers, "ollama": _ollama}


def embed_texts(texts, profile=None):
    """Embed a list of strings with the given (or active) profile; return unit vectors as ndarray."""
    p = profile or config.active_embedding()
    fn = PROVIDERS[p["provider"]]
    batch = int(p.get("batch_size") or 64)
    out = []
    for i in range(0, len(texts), batch):
        out.extend(fn(p, texts[i:i + batch]))
    arr = np.array(out, dtype="float32")
    # ensure unit length (OpenAI already is; local models may not be)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms
