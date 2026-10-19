"""Offline unit tests (no API keys)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ragcore import config, helpers, retrieval, store
from ragcore.tracing import span, trace


def test_chunk_overlap():
    text = "a" * 3000
    chunks = helpers.chunk(text, size=1000, overlap=100)
    assert len(chunks) >= 3
    assert chunks[0][-100:] == chunks[1][:100]


def test_chunk_empty():
    assert helpers.chunk("") == []
    assert helpers.chunk("hello", size=0) == []


def test_config_paths():
    assert config.STORE_DIR.endswith("store") or "store" in config.STORE_DIR
    assert config.JSON_PATH.endswith("chunks.json")
    assert os_path_exists_parent(config.EMBED_CONFIG_PATH)


def os_path_exists_parent(path: str) -> bool:
    import os

    return os.path.exists(path)


def test_active_embedding_exists():
    p = config.active_embedding()
    assert "id" in p and "provider" in p


def test_active_llm_exists():
    p = config.active_llm()
    assert "id" in p and "provider" in p


def test_resolve_llm_profile_none():
    p = config.resolve_llm_profile(None)
    assert p["id"] == config.active_llm()["id"]


def test_resolve_llm_profile_unknown():
    with pytest.raises(ValueError, match="not found"):
        config.resolve_llm_profile("does-not-exist-xyz")


def test_load_store_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JSON_PATH", str(tmp_path / "chunks.json"))
    monkeypatch.setattr(config, "NPY_PATH", str(tmp_path / "vecs.npy"))
    rows, vecs = store.load_store()
    assert rows == []
    assert vecs.size == 0


def test_save_and_load_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JSON_PATH", str(tmp_path / "chunks.json"))
    monkeypatch.setattr(config, "NPY_PATH", str(tmp_path / "vecs.npy"))
    rows = [{"id": "a-0", "source": "a.pdf", "content": "hello world"}]
    vecs = np.array([[0.0, 1.0]], dtype="float32")
    store.save_store(rows, vecs)
    loaded_rows, loaded_vecs = store.load_store()
    assert loaded_rows == rows
    np.testing.assert_allclose(loaded_vecs, vecs)


def test_rrf_merges_lists():
    a = [
        {"content": "x", "source": "s", "score": 0.9},
        {"content": "y", "source": "s", "score": 0.8},
    ]
    b = [
        {"content": "y", "source": "s", "score": 5.0},
        {"content": "z", "source": "s", "score": 4.0},
    ]
    fused = retrieval.rrf([a, b], k=3)
    assert len(fused) == 3
    assert "y" in [h["content"] for h in fused]


def test_retrieve_numpy_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JSON_PATH", str(tmp_path / "chunks.json"))
    monkeypatch.setattr(config, "NPY_PATH", str(tmp_path / "vecs.npy"))
    qvec = np.array([1.0, 0.0], dtype="float32")
    assert retrieval.retrieve_numpy(qvec, k=3) == []


def test_trace_writes_file(tmp_path, monkeypatch):
    path = tmp_path / "traces.jsonl"
    monkeypatch.setattr(config, "TRACE_PATH", str(path))
    monkeypatch.setattr(config, "SHOW_TRACES", False)
    with trace("unit", foo=1):
        with span("step") as s:
            s["data"]["x"] = 2
    data = json.loads(path.read_text(encoding="utf-8").strip())
    assert data["name"] == "unit"
    assert data["spans"][0]["data"]["x"] == 2


def test_embeddings_json_valid():
    cfg = config.load_json(config.EMBED_CONFIG_PATH)
    assert cfg["active"] in {p["id"] for p in cfg["profiles"]}


def test_llms_json_valid():
    cfg = config.load_json(config.LLM_CONFIG_PATH)
    assert cfg["active"] in {p["id"] for p in cfg["profiles"]}
