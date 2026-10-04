"""Text embedding.

``FastEmbedder`` runs BAAI/bge-small-en-v1.5 on ONNX Runtime (fastembed): 384-d,
~5 ms per query on CPU, no torch dependency. Query and passage encodings are
kept separate because bge uses an instruction prefix for queries.

``HashingEmbedder`` is a deterministic, dependency-free stand-in used by unit
tests so they run without downloading a model. It is never used in production.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str
    dim: int

    def count_tokens(self, text: str) -> int: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


class _LRU:
    """Thread-safe LRU cache for query vectors. Repeated/refined voice queries hit it often."""

    def __init__(self, size: int):
        self.size = size
        self._data: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def put(self, key: str, value: list[float]) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.size:
                self._data.popitem(last=False)


class PriorityLock:
    """Serialises access to one shared model; interactive callers go ahead of bulk callers.

    A plain Lock is not fair: an ingestion loop that releases and immediately re-acquires it
    between batches can starve a waiting query for the whole upload. Here a query announces
    itself, and bulk work waits before taking its next batch while any query is waiting, so a
    query waits for at most one in-flight batch.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cv = threading.Condition()
        self._waiting_high = 0

    @contextmanager
    def high(self):
        with self._cv:
            self._waiting_high += 1
        try:
            self._lock.acquire()
        finally:
            with self._cv:
                self._waiting_high -= 1
                self._cv.notify_all()
        try:
            yield
        finally:
            self._lock.release()

    @contextmanager
    def low(self):
        with self._cv:
            while self._waiting_high:
                self._cv.wait(timeout=0.05)
        with self._lock:
            yield


BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def default_query_prefix(model_name: str) -> str:
    """bge retrieval models are trained with an instruction on the *query* side (passages get none)."""
    return BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() and "-en" in model_name.lower() else ""


class FastEmbedder:
    passage_batch_size = 8

    def __init__(self, model_name: str, model_path: str = "", cache_dir: str | None = None, cache_size: int = 2048,
                 query_prefix: str | None = None):
        from fastembed import TextEmbedding

        kwargs: dict = {"cache_dir": cache_dir} if cache_dir else {}
        if model_path:
            kwargs["specific_model_path"] = model_path
        self._model = TextEmbedding(model_name, **kwargs)
        self.model_name = model_name
        self.query_prefix = default_query_prefix(model_name) if query_prefix is None else query_prefix
        self._tokenizer = self._load_tokenizer(model_path)
        self.dim = len(next(iter(self._model.passage_embed(["dimension probe"]))))
        self._cache = _LRU(cache_size)
        self._lock = PriorityLock()  # one shared ONNX session; queries take priority over ingestion

    def _load_tokenizer(self, model_path: str):
        """A separate copy of the model's tokenizer with truncation off, for exact token counts."""
        from pathlib import Path

        from tokenizers import Tokenizer

        model_dir = Path(model_path) if model_path else Path(getattr(self._model.model, "_model_dir", ""))
        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tok.no_truncation()
        tok.no_padding()
        return tok

    def count_tokens(self, text: str) -> int:
        """Tokens the model would see for `text`, excluding [CLS]/[SEP] (the 512 limit includes those 2)."""
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def embed_query(self, text: str) -> list[float]:
        key = " ".join(text.lower().split())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        with self._lock.high():
            vec = next(iter(self._model.query_embed([self.query_prefix + text])))
        out = np.asarray(vec, dtype=np.float32).tolist()
        self._cache.put(key, out)
        return out

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        # Small batches with the lock released in between, so a query arriving during
        # an upload waits for at most one batch (~tens of ms), not the whole document.
        # We don't rely on concurrent calls being safe: ONNX Runtime's session.run is,
        # but the shared HF tokenizer object is not documented as such.
        out: list[list[float]] = []
        for i in range(0, len(texts), self.passage_batch_size):
            with self._lock.low():
                vecs = list(self._model.passage_embed(texts[i : i + self.passage_batch_size],
                                                      batch_size=self.passage_batch_size))
            out.extend(np.asarray(v, dtype=np.float32).tolist() for v in vecs)
        return out

    def clear_cache(self) -> None:
        """Benchmarks: measure embedding cost, not cache hits."""
        self._cache.clear()

    @property
    def cache_stats(self) -> dict[str, int]:
        return {"hits": self._cache.hits, "misses": self._cache.misses}


class HashingEmbedder:
    """Bag-of-words feature hashing → L2-normalised vector. Test double only."""

    def __init__(self, dim: int = 256):
        self.model_name = "hashing-test-embedder"
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def count_tokens(self, text: str) -> int:
        return len(re.findall(r"[a-z0-9]+|[^\sa-z0-9]", text.lower()))

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]
