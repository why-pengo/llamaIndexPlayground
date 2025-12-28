# llamaIndex example

A small example repository showing how to use llama_index with Ollama (local LLM) and HuggingFace embeddings, plus a robust persistence helper and development tooling.

This repo includes:
- `src/summary.py` — example script that builds a VectorStoreIndex from `src/data/` and queries it using Ollama and HuggingFace embeddings.
- `src/persistence.py` — helper that prefers llama_index storage persistence APIs (StorageContext/load_index_from_storage) and falls back to a pickle cache.
- `tests/` — pytest tests that mock external services so you can run tests offline.

Requirements
------------
- Python 3.10+ (3.11 recommended)
- A working Python environment with required packages. Runtime dependencies vary with your configuration (Ollama, HuggingFace, etc.).
- (Ollama)[https://ollama.com/] installed and see https://developers.llamaindex.ai/python/framework/getting_started/starter_example_local/
- (HuggingFace)[https://huggingface.co/] account and token if you plan to use private models.
- Tested on Linux Bazzite OS with Nvidia RTX 40?? GPU 8GB VRAM, Intel Core 7 32GB RAM.

Developer setup
---------------
Create and activate a virtual environment, then install development dependencies. This example is for Linux; adapt as needed for Windows or MacOS.:

```sh
python3 -m venv venv
. venv/bin/activate
pip install -r requirements-dev.txt
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
tar -xvzf ollama-linux-amd64.tgz
wget https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt -O data/paul_graham_essay.txt
```

If you plan to run the example in non-dry-run mode you will also need the runtime dependencies required by `llama_index`, `transformers`, and your embedding/LLM backends; install those as appropriate.

Running the example
-------------------
- Dry-run (no external model downloads or Ollama calls):

```sh
python3 src/summary.py --dry-run
```

- Run a real query (this will initialize the embedding model and contact the Ollama server if configured):

```sh
python3 src/summary.py --query "Summarize the essay in one sentence."
```

CLI options
-----------
- `--query/-q` : natural language question to run against the index (default: "What is this document about?")
- `--verbose/-v` : enable verbose logging
- `--dry-run` : only load documents and show a snippet; don't initialize models or run queries
- `--embed-model` : HuggingFace embedding model name (default: `BAAI/bge-base-en-v1.5`)
- `--llm-model` : Ollama model name (default: `llama3.1`)
- `--cache-dir` : directory to store/load a cached index (default: `.index_cache`)
- `--use-cache` : attempt to load index from cache if available
- `--rebuild` : force rebuilding the index and overwrite cache

Index persistence behavior
--------------------------
The `src/persistence.py` helper will:
1. Try to use llama_index's StorageContext + `load_index_from_storage` if available.
2. Fall back to older `VectorStoreIndex` save/load APIs if found.
3. Finally fall back to a pickle-based cache at `<cache-dir>/index.pkl`.

This arrangement is "best-effort" and aims to be compatible across llama_index versions; if you have a pinned llama_index version with a preferred persistence API we can simplify the helper accordingly.

Testing
-------
Run tests with pytest:

```sh
pytest -q
```

The tests mock HuggingFace and Ollama constructors and the index builder where appropriate so they run quickly and offline.

Development tools (linters, pre-commit)
---------------------------------------
Install dev dependencies (already listed in `requirements-dev.txt`) and register pre-commit hooks:

```sh
pip install -r requirements-dev.txt
pip install pre-commit
pre-commit install
```

Run pre-commit on all files:

```sh
pre-commit run --all-files
```

Run linters and type checks:

```sh
flake8
mypy
```

Troubleshooting
---------------
- If the script fails to initialize embeddings or LLMs, make sure:
  - Ollama is installed and the Ollama daemon is running locally if you plan to use a local Ollama server.
  - The Hugging Face model you specify is available or you have network access / valid HF token if using private models.
- If persistence save/load fails, the helper will log errors and fall back to rebuilding the index.

License
-------
This example repo contains no license file; add one if you plan to publish or share widely.
