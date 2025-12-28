import asyncio

import summary


class FakeReader:
    def __init__(self, *args, **kwargs):
        pass

    def load_data(self):
        class Doc:
            def get_content(self):
                return "hello world"

        return [Doc()]


def test_dry_run_no_models(monkeypatch, capsys):
    # Patch the SimpleDirectoryReader to avoid filesystem dependency
    monkeypatch.setattr(summary, "SimpleDirectoryReader", FakeReader)

    # Prevent model initialization if accidentally reached
    monkeypatch.setattr(
        summary,
        "HuggingFaceEmbedding",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Should not init embedder")),
        raising=False,
    )
    monkeypatch.setattr(
        summary, "Ollama", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Should not init LLM")), raising=False
    )

    asyncio.run(
        summary.main(
            "What is this?",
            verbose=False,
            dry_run=True,
            embed_model_name="x",
            llm_model_name="y",
            cache_dir=".tmp_cache",
            use_cache=False,
            rebuild=False,
        )
    )

    out = capsys.readouterr().out
    assert "First document snippet" in out
    assert "Document count: 1" in out
