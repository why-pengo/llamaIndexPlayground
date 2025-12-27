"""
Persistence helpers for saving/loading llama_index indices.

This module attempts to use llama_index's native persistence APIs when available
(StorageContext + load_index_from_storage, or index.save_to_disk / load_from_disk).
If those aren't available it falls back to a best-effort pickle-based cache.
"""
import logging
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# Feature detection
HAS_STORAGE = False
HAS_LOAD_FROM_STORAGE = False
HAS_INDEX_SAVE = False
HAS_INDEX_LOAD = False

_storage_module = None
_load_index_from_storage = None

try:
    # try common modern locations
    try:
        from llama_index.storage import StorageContext  # type: ignore
    except Exception:
        from llama_index.storage.storage_context import StorageContext  # type: ignore
    try:
        from llama_index import load_index_from_storage  # type: ignore
        _load_index_from_storage = load_index_from_storage
        HAS_LOAD_FROM_STORAGE = True
    except Exception:
        HAS_LOAD_FROM_STORAGE = False
    HAS_STORAGE = True
    _storage_module = StorageContext
    log.debug("Detected StorageContext in llama_index")
except Exception:
    HAS_STORAGE = False

# Detect older index save/load methods on VectorStoreIndex
try:
    import llama_index as _ll
    if hasattr(_ll, "VectorStoreIndex"):
        v = getattr(_ll, "VectorStoreIndex")
        HAS_INDEX_SAVE = hasattr(v, "save_to_disk") or hasattr(v, "save")
        HAS_INDEX_LOAD = hasattr(v, "load_from_disk") or hasattr(v, "load")
except Exception:
    pass


def _atomic_replace_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        os.replace(str(src), str(dst))
    except Exception:
        shutil.move(str(src), str(dst))


def load_index(cache_dir: str) -> Optional[Any]:
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None

    if HAS_STORAGE and HAS_LOAD_FROM_STORAGE:
        try:
            log.info("Loading index via llama_index StorageContext from %s", cache_dir)
            StorageContext = _storage_module
            sc = StorageContext.from_defaults(persist_dir=str(cache_path))
            return _load_index_from_storage(sc)
        except Exception:
            log.exception("Failed to load index via StorageContext; falling back")

    if HAS_INDEX_LOAD:
        try:
            import llama_index as _ll
            v = getattr(_ll, "VectorStoreIndex")
            if hasattr(v, "load_from_disk"):
                log.info("Loading index via VectorStoreIndex.load_from_disk from %s", cache_dir)
                return v.load_from_disk(str(cache_path))
            elif hasattr(v, "load"):
                log.info("Loading index via VectorStoreIndex.load from %s", cache_dir)
                return v.load(str(cache_path))
        except Exception:
            log.exception("Failed to load index via VectorStoreIndex.load_from_disk; falling back")

    pkl = cache_path / "index.pkl"
    if pkl.exists():
        try:
            log.info("Loading index via pickle fallback from %s", pkl)
            with open(pkl, "rb") as f:
                return pickle.load(f)
        except Exception:
            log.exception("Failed to unpickle index at %s", pkl)
    return None


def save_index(index: Any, cache_dir: str) -> bool:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if HAS_STORAGE:
        try:
            log.info("Saving index via llama_index StorageContext into %s", cache_dir)
            StorageContext = _storage_module
            tmp_dir = Path(tempfile.mkdtemp(dir=str(cache_path)))
            if hasattr(index, "save_to_disk"):
                index.save_to_disk(str(tmp_dir))
            elif hasattr(index, "save"):
                index.save(str(tmp_dir))
            elif hasattr(index, "persist"):
                index.persist(str(tmp_dir))
            else:
                sc = StorageContext.from_defaults(persist_dir=str(tmp_dir))
                if hasattr(sc, "persist"):
                    sc.persist(index)
                else:
                    raise RuntimeError("No known persistence method on index or StorageContext")
            _atomic_replace_dir(tmp_dir, cache_path)
            return True
        except Exception:
            log.exception("Failed to save index via llama_index StorageContext; falling back to pickle")

    if HAS_INDEX_SAVE:
        try:
            import llama_index as _ll
            v = getattr(_ll, "VectorStoreIndex")
            tmp_dir = Path(tempfile.mkdtemp(dir=str(cache_path)))
            if hasattr(index, "save_to_disk"):
                index.save_to_disk(str(tmp_dir))
            elif hasattr(index, "save"):
                index.save(str(tmp_dir))
            else:
                raise RuntimeError("No known save method on index")
            _atomic_replace_dir(tmp_dir, cache_path)
            return True
        except Exception:
            log.exception("Failed to save index via VectorStoreIndex.save_to_disk; falling back to pickle")

    try:
        pkl = cache_path / "index.pkl"
        with tempfile.NamedTemporaryFile(dir=str(cache_path), delete=False) as tf:
            tmp_name = tf.name
        with open(tmp_name, "wb") as f:
            f.write(pickle.dumps(index))
        Path(tmp_name).replace(pkl)
        log.info("Saved index via pickle fallback to %s", pkl)
        return True
    except Exception:
        log.exception("Failed to pickle index into %s", cache_path)
    return False

