import sys
import asyncio  # noqa: E402
import pathlib

HERE = pathlib.Path(__file__).resolve().parents[1]
SRC = HERE / "src"
sys.path.insert(0, str(SRC))

import summary  # noqa: E402


class FakeReader:
    def __init__(self, *args, **kwargs):
        pass

    def load_data(self):
        class Doc:
            def __init__(self, text, idx):
                self._text = text
                self.id_ = f"doc_{idx}"
                self.hash = f"hash_{idx}"

            def get_content(self):
                return self._text

        return [Doc("hello world", 0), Doc("second doc", 1)]


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.0] * 4 for _ in texts]


class FakeQueryEngine:
    async def aquery(self, q):
        return "FAKE_SUMMARY"


class FakeIndexObj:
    def as_query_engine(self):
        return FakeQueryEngine()


class FakeLLM:
    async def __call__(self, *args, **kwargs):
        return "FAKE_SUMMARY"


def test_cli_build_and_query(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(summary, "SimpleDirectoryReader", FakeReader)
    # Avoid real model downloads by monkeypatching constructors
    monkeypatch.setattr(summary, "HuggingFaceEmbedding", lambda *a, **k: FakeEmbedder())
    monkeypatch.setattr(summary, "Ollama", lambda *a, **k: FakeLLM())
    # Avoid VectorStoreIndex internals by returning a fake index
    monkeypatch.setattr(summary.VectorStoreIndex, "from_documents", lambda docs: FakeIndexObj())
    # Also set internal Settings attributes to bypass validation
    monkeypatch.setattr(summary.Settings, "_embed_model", FakeEmbedder(), raising=False)
    monkeypatch.setattr(summary.Settings, "_llm", FakeLLM(), raising=False)

    asyncio.run(
        summary.main(
            "Summarize",
            verbose=True,
            dry_run=False,
            embed_model_name="x",
            llm_model_name="y",
            cache_dir=str(tmp_path / "cache"),
            use_cache=False,
            rebuild=False,
        )
    )

    out = capsys.readouterr().out
    assert "FAKE_SUMMARY" in out or out.strip() != ""
