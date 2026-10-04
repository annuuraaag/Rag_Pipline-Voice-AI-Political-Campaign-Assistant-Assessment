"""Lexical (BM25) retrieval over the indexed chunks.

Why BM25 next to dense vectors: embeddings capture meaning but blur exact tokens.
Scheme names ("Udayam Breakfast"), place names ("Gollapudi") and numbers are where
lexical matching is strongest. The index covers `embed_text` (chunk + contextual
header), so district and section names are searchable too.

The corpus here is small (hundreds of chunks), so the index lives in memory and is
rebuilt lazily after any upload/delete. Rebuilding ~100 chunks takes a few ms; at
millions of chunks this would move to a search engine with an inverted index
(OpenSearch, Qdrant sparse vectors).
"""
from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable

from rank_bm25 import BM25Okapi

from app.domain import Chunk, MetadataFilter, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was what when where "
    "which who will with about any can do does there their they we you your me my our".split()
)


def _stem(tok: str) -> str:
    # Light plural folding only ("schemes"→"scheme", "hospitals"→"hospital"); aggressive
    # stemmers mangle the proper nouns this corpus depends on.
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


logger = logging.getLogger(__name__)


class BM25Index:
    def __init__(self, load_chunks: Callable[[], list[Chunk]]):
        self._load = load_chunks
        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._dirty = True

    def invalidate(self) -> None:
        """Called by IngestionService on every corpus change; rebuild happens on next search."""
        self._dirty = True

    def warm_async(self) -> None:
        """Rebuild in the background right after an upload, so the next question doesn't pay for it
        (~100 ms per 150 chunks). A query arriving mid-rebuild just waits on the lock."""
        def run() -> None:
            try:
                self._ensure()
            except Exception as exc:  # store closed (tests, shutdown): the next search rebuilds anyway
                logger.debug("BM25 background rebuild skipped: %s", exc)
                self._dirty = True

        threading.Thread(target=run, name="bm25-rebuild", daemon=True).start()

    def _ensure(self) -> tuple[list[Chunk], BM25Okapi | None]:
        with self._lock:
            if self._dirty:
                chunks = self._load()
                corpus = [tokenize(c.embed_text) for c in chunks]
                # BM25Okapi divides by corpus size; an empty corpus means "no lexical side".
                self._bm25 = BM25Okapi(corpus) if any(corpus) else None
                self._chunks = chunks
                self._dirty = False
            return self._chunks, self._bm25

    def search(self, query: str, limit: int, flt: MetadataFilter | None = None) -> list[ScoredChunk]:
        chunks, bm25 = self._ensure()
        tokens = tokenize(query)
        if bm25 is None or not tokens:
            return []
        scores = bm25.get_scores(tokens)
        ranked = sorted(
            (i for i, s in enumerate(scores) if s > 0 and (flt is None or flt.matches(chunks[i].metadata))),
            key=lambda i: scores[i],
            reverse=True,
        )[:limit]
        return [
            ScoredChunk(chunk=chunks[i], score=float(scores[i]), rank=r + 1, retriever="bm25",
                        scores={"bm25": round(float(scores[i]), 3)})
            for r, i in enumerate(ranked)
        ]

    @property
    def size(self) -> int:
        return len(self._chunks)
