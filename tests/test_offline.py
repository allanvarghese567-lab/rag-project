"""Offline unit tests that do not need API keys."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import helpers
import config


def test_chunk_overlap():
    text = "a" * 3000
    chunks = helpers.chunk(text, size=1000, overlap=100)
    assert len(chunks) >= 3
    assert chunks[0][-100:] == chunks[1][:100]


def test_config_paths():
    assert config.STORE_DIR
    assert config.JSON_PATH.endswith("chunks.json")
