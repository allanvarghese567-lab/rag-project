"""Offline unit tests that do not need API keys."""
from __future__ import annotations

import json
import os
from pathlib import Path

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


def test_chunk_step_one_when_overlap_ge_size():
    # size - overlap would be <= 0 → step forced to 1
    chunks = helpers.chunk("abcdef", size=3, overlap=3)
    assert len(chunks) >= 1
    assert all(len(c) <= 3 for c in chunks)


def test_config_paths():
    assert config.STORE_DIR
    assert config.JSON_PATH.endswith("chunks.json")
    assert config.NPY_PATH.endswith("vecs.npy")


def test_active_embedding_exists():
    p = config.active_embedding()
    assert "id" in p and "provider" in p and "model" in p


def test_active_llm_exists():
    p = config.active_llm()
    assert "id" in p and "provider" in p and "model" in p


def test_resolve_llm_profile_none():
    p = config.resolve_llm_profile(None)
    assert p["id"] == config.active_llm()["id"]


def test_resolve_llm_profile_dict():
    d = {"id": "custom", "provider": "openai", "model": "x"}
    assert config.resolve_llm_profile(d) is d


def test_resolve_llm_profile_by_id():
    active = config.active_llm()
    resolved = config.resolve_llm_profile(active["id"])
    assert resolved["id"] == active["id"]


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
    assert loaded_vecs.shape == (1, 2)
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
    # y appears in both → should rank high
    contents = [h["content"] for h in fused]
    assert "y" in contents
    assert fused[0]["content"] in ("y", "x")  # y often first


def test_rrf_empty():
    assert retrieval.rrf([], k=5) == []
    assert retrieval.rrf([[]], k=5) == []


def test_retrieve_numpy_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JSON_PATH", str(tmp_path / "chunks.json"))
    monkeypatch.setattr(config, "NPY_PATH", str(tmp_path / "vecs.npy"))
    qvec = np.array([1.0, 0.0], dtype="float32")
    assert retrieval.retrieve_numpy(qvec, k=3) == []


def test_span_outside_trace_is_harmless():
    with span("solo", n=1) as s:
        s["data"]["ok"] = True
    assert s["ms"] >= 0


def test_trace_writes_file(tmp_path, monkeypatch):
    path = tmp_path / "traces.jsonl"
    monkeypatch.setattr(config, "TRACE_PATH", str(path))
    monkeypatch.setattr(config, "SHOW_TRACES", False)
    with trace("unit", foo=1):
        with span("step") as s:
            s["data"]["x"] = 2
    assert path.exists()
    line = path.read_text(encoding="utf-8").strip()
    data = json.loads(line)
    assert data["name"] == "unit"
    assert data["meta"]["foo"] == 1
    assert data["spans"][0]["name"] == "step"
    assert data["spans"][0]["data"]["x"] == 2


def test_embeddings_json_valid():
    cfg = config.load_json(config.EMBED_CONFIG_PATH)
    assert "active" in cfg and "profiles" in cfg
    ids = {p["id"] for p in cfg["profiles"]}
    assert cfg["active"] in ids


def test_llms_json_valid():
    cfg = config.load_json(config.LLM_CONFIG_PATH)
    assert "active" in cfg and "profiles" in cfg
    ids = {p["id"] for p in cfg["profiles"]}
    assert cfg["active"] in ids


def test_compat_shims_importable():
    import config as root_config
    import helpers as root_helpers

    assert root_config.TOP_K == config.TOP_K
    assert callable(root_helpers.chunk)
