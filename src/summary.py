import argparse
import asyncio
import logging
import sys
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# persistence helpers will be imported inside main to avoid static import issues
_LOAD_INDEX = None
_SAVE_INDEX = None

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional
    def tqdm(x, **kwargs):
        return x


def parse_args():
    p = argparse.ArgumentParser(description="Query the Paul Graham essay using Ollama + HuggingFace embeddings")
    p.add_argument("--query", "-q", default="What is this document about?", help="The natural language query to run against the document index")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    p.add_argument("--dry-run", action="store_true", help="Do a local dry run (load documents only) without contacting Ollama or HuggingFace")

    # Model and cache options
    p.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5", help="HuggingFace embedding model name")
    p.add_argument("--llm-model", default="llama3.1", help="Ollama model name")
    p.add_argument("--cache-dir", default=".index_cache", help="Directory to store/load a cached index")
    p.add_argument("--use-cache", action="store_true", help="Attempt to load index from cache if available")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild of the index and overwrite cache")
    return p.parse_args()


async def main(query: str, verbose: bool, dry_run: bool, embed_model_name: str, llm_model_name: str, cache_dir: str, use_cache: bool, rebuild: bool):
    """Main entrypoint (async) — signature kept for tests.

    Args match tests that call: asyncio.run(summary.main(...))
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger(__name__)

    # locate data directory relative to this script
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    reader = SimpleDirectoryReader(str(data_dir))
    log.info("Loading documents from %s", data_dir)
    documents = reader.load_data()
    log.info("Loaded %d document(s)", len(documents))

    # Prepare cache path
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Attempt to load cached index if requested and not rebuilding
    index = None
    # import persistence helpers lazily (handles package vs script execution)
    try:
        from .persistence import load_index, save_index  # type: ignore
    except Exception:
        from persistence import load_index, save_index  # type: ignore

    if use_cache and not rebuild:
        log.info("Attempting to load index from cache: %s", cache_path)
        index = load_index(str(cache_path))
        if index is not None:
            log.info("Loaded index from cache via persistence helper")

    # Dry-run: report basic info and exit without initializing models
    if dry_run:
        log.info("Dry run enabled: skipping model initialization and query execution")
        if index is not None:
            print("Cached index found at:", cache_path)
            print("Cached index type:", type(index))
            # doc count may not be exposed; print placeholder
            print("Document count (may be approximate):", getattr(index, 'docstore', 'unknown'))
        elif documents:
            snippet = getattr(documents[0], "get_content", lambda: str(documents[0]))()
            print("First document snippet:\n", snippet[:500])
        print(f"Document count: {len(documents)}")
        return

    # Initialize embeddings / LLM only if not already set (tests may set Settings._embed_model/_llm)
    log.info("Initializing HuggingFaceEmbedding (%s) and Ollama LLM (%s)", embed_model_name, llm_model_name)
    if getattr(Settings, "_embed_model", None) is None:
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    if getattr(Settings, "_llm", None) is None:
        Settings.llm = Ollama(
            model=llm_model_name,
            request_timeout=360.0,
            context_window=8000,
        )

    # If no cached index, build it
    if index is None:
        log.info("Building VectorStoreIndex from documents (will compute embeddings)")
        try:
            # Try to show progress while batching embed calls if embedder supports it
            embedder = Settings.embed_model
            if hasattr(embedder, "embed_documents"):
                texts = [getattr(d, "get_content", lambda: str(d))() for d in documents]
                batch_size = 16
                embeddings = []
                for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
                    batch = texts[i : i + batch_size]
                    embeddings.extend(embedder.embed_documents(batch))
                # build index normally (VectorStoreIndex handles the rest)
                index = VectorStoreIndex.from_documents(documents)
            else:
                index = VectorStoreIndex.from_documents(documents)
        except Exception:
            log.exception("Error during index building; falling back to default builder")
            index = VectorStoreIndex.from_documents(documents)

        # Persist the built index if requested
        if use_cache:
            saved = save_index(index, str(cache_path))
            if saved:
                log.info("Saved index to cache via persistence helper at %s", cache_path)
            else:
                log.warning("Failed to save index to cache; continuing without cache")

    # Run query using the query engine
    try:
        query_engine = index.as_query_engine()
        log.info("Running query: %s", query)
        response = await query_engine.aquery(query)
        print(response)
    except Exception as e:
        log.exception("Query failed: %s", e)
        print("Query failed:", e, file=sys.stderr)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.query, args.verbose, args.dry_run, args.embed_model, args.llm_model, args.cache_dir, args.use_cache, args.rebuild))
