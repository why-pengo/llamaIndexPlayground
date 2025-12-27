import sys
import asyncio
import pathlib
import pickle

import pytest

HERE = pathlib.Path(__file__).resolve().parents[1]
SRC = HERE / "src"
sys.path.insert(0, str(SRC))

import persistence
import summary


class FakeIndex:
    def __init__(self, data):
        self.data = data


def test_pickle_cache_roundtrip(tmp_path, monkeypatch):
    # Force persistence helper to only use pickle fallback
    monkeypatch.setattr(persistence, "HAS_STORAGE", False)
    monkeypatch.setattr(persistence, "HAS_INDEX_SAVE", False)
    monkeypatch.setattr(persistence, "HAS_INDEX_LOAD", False)

    fake_index = FakeIndex({"k": "v"})
    cache_dir = tmp_path / "cache"
    saved = persistence.save_index(fake_index, str(cache_dir))
    assert saved

    # Validate file exists and can be loaded
    pkl = cache_dir / "index.pkl"
    assert pkl.exists()

    loaded = persistence.load_index(str(cache_dir))
    assert isinstance(loaded, FakeIndex)
    assert loaded.data["k"] == "v"

    # Now ensure summary.main dry-run sees the cache (don't initialize models)
    class FakeReader:
        def __init__(self, *a, **k):
            pass

        def load_data(self):
            class Doc:
                def get_content(self):
                    return "doc"

            return [Doc()]

    monkeypatch.setattr(summary, "SimpleDirectoryReader", FakeReader)
    asyncio.run(summary.main("Q", verbose=False, dry_run=True, embed_model_name="x", llm_model_name="y", cache_dir=str(cache_dir), use_cache=True, rebuild=False))

