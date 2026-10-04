"""Retrieval pipeline.

    query ─┬─ dense (bge-small → Qdrant, filtered) ─┐
           └─ BM25 (in-memory, same filter) ─────────┴→ RRF fusion → top-N
           → cross-encoder rerank (if available) → relevance gate → multi-document context

Stage 1 is for recall (20 candidates each side), stage 2 for precision. Dense and
BM25 run concurrently. If a district filter leaves too few candidates, the search
is retried without it rather than answering from almost nothing.

The answerability gate uses the most comparable score available:
* with the reranker: per-chunk cross-encoder probability ≥ RERANK_THRESHOLD;
* without it: the best dense cosine similarity ≥ SIMILARITY_THRESHOLD (query-level),
  because RRF scores are ranks, not relevance, and BM25 scores are unbounded.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from app.domain import MetadataFilter, ScoredChunk
from app.observability.timing import StageTimer
from app.retrieval.bm25 import BM25Index
from app.retrieval.context_builder import build_context
from app.retrieval.embedder import Embedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import Reranker, rerank
from app.retrieval.vector_store import QdrantStore

Mode = Literal["dense", "bm25", "hybrid"]
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval")


class RetrievalResult(BaseModel):
    query: str
    filters: dict = Field(default_factory=dict)
    candidates: list[ScoredChunk] = Field(default_factory=list)  # fused (and reranked) pool, for the trace
    results: list[ScoredChunk] = Field(default_factory=list)     # final context chunks
    threshold: float
    top_score: float = 0.0
    answerable: bool = False
    has_documents: bool = False
    strategy: str = "dense"
    gate: str = "dense"            # which score the gate used: rerank | dense | none
    filter_relaxed: bool = False   # district filter dropped because it left too few candidates
    counts: dict[str, int] = Field(default_factory=dict)


@dataclass
class Stage1:
    """Fused stage-1 candidates for one (query, filters) pair."""
    query: str
    filters: MetadataFilter
    fused: list[ScoredChunk]
    counts: dict[str, int]
    relaxed: bool = False
    empty: bool = False   # the corpus had no chunks


class Retriever:
    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
        *,
        mode: Mode = "hybrid",
        candidate_k: int = 20,
        rerank_k: int = 12,
        top_k: int = 4,
        threshold: float = 0.62,
        rerank_threshold: float = 0.15,
        rrf_k: int = 60,
        max_per_doc: int = 2,
        min_filtered: int = 3,
    ):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25 if bm25 is not None else BM25Index(store.iter_chunks)
        self.reranker = reranker
        self.mode: Mode = mode
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k
        self.top_k = top_k
        self.threshold = threshold
        self.rerank_threshold = rerank_threshold
        self.rrf_k = rrf_k
        self.max_per_doc = max_per_doc
        self.min_filtered = min_filtered
        # Cached so the query path never asks the store (which may be locked by an upload).
        self.chunk_count = 0

    @property
    def strategy(self) -> str:
        s = {"dense": "dense", "bm25": "bm25", "hybrid": "hybrid(dense+bm25, rrf)"}[self.mode]
        return s + (" + rerank" if self.reranker else "")

    def refresh_count(self) -> None:
        self.chunk_count = self.store.count()

    def on_corpus_change(self) -> None:
        self.bm25.invalidate()
        self.refresh_count()
        self.bm25.warm_async()

    # ── stage 1 ─────────────────────────────────────────────────────────
    def _dense(self, query: str, flt: MetadataFilter | None, timer: StageTimer) -> list[ScoredChunk]:
        with timer.stage("embed"):
            vec = self.embedder.embed_query(query)
        with timer.stage("dense"):
            return self.store.search(vec, limit=self.candidate_k, flt=flt)

    def _bm25(self, query: str, flt: MetadataFilter | None, timer: StageTimer) -> list[ScoredChunk]:
        with timer.stage("bm25"):
            return self.bm25.search(query, limit=self.candidate_k, flt=flt)

    def _stage1(self, query: str, flt: MetadataFilter | None, timer: StageTimer) -> tuple[list[ScoredChunk], dict]:
        dense: list[ScoredChunk] = []
        lexical: list[ScoredChunk] = []
        if self.mode == "hybrid":
            fut = _POOL.submit(self._dense, query, flt, timer)  # dense in a worker, BM25 here, concurrently
            lexical = self._bm25(query, flt, timer)
            dense = fut.result()
            with timer.stage("fusion"):
                fused = reciprocal_rank_fusion({"dense": dense, "bm25": lexical}, k=self.rrf_k)
        elif self.mode == "dense":
            dense = fused = self._dense(query, flt, timer)
        else:
            lexical = fused = self._bm25(query, flt, timer)
        return fused, {"dense": len(dense), "bm25": len(lexical), "fused": len(fused)}

    # ── full pipeline ───────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        filters: MetadataFilter | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        timer: StageTimer | None = None,
    ) -> RetrievalResult:
        timer = timer or StageTimer()
        return self.finish(self.candidates(query, filters, timer), top_k, threshold, timer)

    def candidates(self, query: str, filters: MetadataFilter | None, timer: StageTimer | None = None) -> Stage1:
        """Stage 1 (recall): dense ‖ BM25 → RRF. Cheap (~10–30 ms), so the voice path runs it
        speculatively on partial transcripts and reuses it when the final transcript matches."""
        timer = timer or StageTimer()
        if filters is None or not filters.campaign_id:
            # Tenant isolation is enforced here, not only in the API: no code path can search
            # across campaigns by forgetting a filter.
            raise ValueError("campaign_id is required for every search")
        flt = filters
        if self.chunk_count == 0:
            return Stage1(query=query, filters=flt, fused=[], counts={}, empty=True)
        fused, counts = self._stage1(query, flt, timer)
        relaxed = False
        if flt.district and len(fused) < self.min_filtered:
            fused, counts = self._stage1(query, flt.model_copy(update={"district": None}), timer)  # campaign stays
            relaxed = True
        return Stage1(query=query, filters=flt, fused=fused, counts=counts, relaxed=relaxed)

    def finish(self, s1: Stage1, top_k: int | None = None, threshold: float | None = None,
               timer: StageTimer | None = None) -> RetrievalResult:
        """Stage 2 (precision): cross-encoder rerank → relevance gate → multi-document context."""
        timer = timer or StageTimer()
        k = top_k or self.top_k
        query, fused, counts = s1.query, s1.fused, dict(s1.counts)
        base = dict(query=query, filters=s1.filters.as_dict(), strategy=self.strategy)
        if s1.empty:
            return RetrievalResult(**base, threshold=threshold or self.threshold)

        best_dense = max((c.scores.get("dense", 0.0) for c in fused), default=0.0)
        if self.reranker is not None and fused:
            tau = self.rerank_threshold if threshold is None else threshold
            with timer.stage("rerank"):
                pool = rerank(self.reranker, query, fused[: self.rerank_k])
            with timer.stage("threshold"):
                passing = [c for c in pool if c.score >= tau]
            candidates, top, gate = pool + fused[self.rerank_k:], (pool[0].score if pool else 0.0), "rerank"
        else:
            tau = self.threshold if threshold is None else threshold
            with timer.stage("threshold"):
                if self.mode == "bm25":
                    passing, gate = list(fused), "none"   # BM25 scores are unbounded; ranking only
                else:
                    passing, gate = (list(fused) if best_dense >= tau else []), "dense"
            candidates, top = fused, best_dense

        with timer.stage("context"):
            context = build_context(passing, k, max_per_doc=self.max_per_doc)
        counts.update(reranked=min(len(fused), self.rerank_k) if gate == "rerank" else 0,
                      passed=len(passing), final=len(context))
        return RetrievalResult(
            **base, candidates=candidates, results=context, threshold=tau, top_score=round(top, 4),
            answerable=bool(context), has_documents=True, gate=gate, filter_relaxed=s1.relaxed, counts=counts,
        )
