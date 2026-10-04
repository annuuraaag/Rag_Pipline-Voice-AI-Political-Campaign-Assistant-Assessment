"""Dependency wiring. One place builds every service; tests swap in fakes here."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.conversation.rewriter import LLMRewriter
from app.conversation.state import SessionStore
from app.generation.llm.base import LLMProvider
from app.generation.llm.extractive import ExtractiveLLM
from app.generation.llm.openai_compat import OpenAICompatibleLLM
from app.ingestion.chunker import make_chunker
from app.ingestion.ocr import OcrEngine
from app.ingestion.registry import IN_PROGRESS, DocumentRegistry
from app.ingestion.service import IngestionService, UploadRejected
from app.observability.metrics import LatencyMetrics
from app.rag.service import RAGService
from app.retrieval.bm25 import BM25Index
from app.retrieval.embedder import Embedder, FastEmbedder
from app.retrieval.pipeline import Retriever
from app.retrieval.reranker import Reranker, load_reranker
from app.retrieval.vector_store import QdrantStore
from app.voice.transcriber import Transcriber, WhisperTranscriber

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    embedder: Embedder
    store: QdrantStore
    registry: DocumentRegistry
    ingestion: IngestionService
    retriever: Retriever
    llm: LLMProvider
    rag: RAGService
    metrics: LatencyMetrics
    sessions: SessionStore = field(default_factory=SessionStore)
    reranker_status: str = "disabled"
    ocr: OcrEngine | None = None
    startup_report: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    startup_timings_ms: dict[str, float] = field(default_factory=dict)
    transcriber: Transcriber | None = None


def build_transcriber(s: Settings) -> Transcriber | None:
    """Whisper via the LLM provider's OpenAI-compatible API, when it has one and a key."""
    if s.llm_provider not in ("groq", "openai") or not s.llm_api_key:
        return None
    model = s.stt_model or ("whisper-large-v3-turbo" if s.llm_provider == "groq" else "whisper-1")
    return WhisperTranscriber(s.llm_provider, s.resolved_llm_base_url, s.llm_api_key, model=model)


def build_llm(s: Settings) -> LLMProvider:
    if s.llm_provider == "extractive":
        return ExtractiveLLM()
    if s.llm_provider in ("groq", "openai") and not s.llm_api_key:
        logger.warning("No API key for LLM provider '%s' — using the offline extractive fallback.", s.llm_provider)
        return ExtractiveLLM()
    return OpenAICompatibleLLM(
        name=s.llm_provider, base_url=s.resolved_llm_base_url, model=s.llm_model, api_key=s.llm_api_key,
        timeout_s=s.llm_timeout_s, max_tokens=s.llm_max_tokens, temperature=s.llm_temperature,
    )


def build_container(
    s: Settings,
    *,
    embedder: Embedder | None = None,
    store: QdrantStore | None = None,
    llm: LLMProvider | None = None,
    registry: DocumentRegistry | None = None,
    reranker: Reranker | None | str = "auto",
) -> Container:
    timings: dict[str, float] = {}
    t = time.perf_counter()
    if embedder is None:
        embedder = FastEmbedder(
            s.embedding_model, model_path=s.resolved_embedding_model_path,
            cache_dir=str(s.model_cache_dir), cache_size=s.query_cache_size,
            query_prefix=None if s.embedding_query_prefix == "auto" else s.embedding_query_prefix,
        )
    timings["load_embedder"] = round((time.perf_counter() - t) * 1000, 1)

    store = store or QdrantStore(s.qdrant_collection, url=s.qdrant_url, path=s.qdrant_path)
    _wait_for(store)
    store.ensure_collection(embedder.dim)
    registry = registry if registry is not None else DocumentRegistry(s.registry_path)
    migrated = store.backfill_campaign(s.default_campaign_id)
    if migrated:
        logger.warning("Assigned %d pre-existing chunk(s) to campaign '%s'", migrated, s.default_campaign_id)
    report = _reconcile(store, registry)
    report["campaign_backfill"] = migrated

    chunker = make_chunker(s.chunk_strategy, max_tokens=s.chunk_max_tokens, min_tokens=s.chunk_min_tokens,
                           max_words=s.chunk_max_words, overlap_words=s.chunk_overlap_words,
                           measure=embedder.count_tokens)
    ocr = OcrEngine(max_pages=s.ocr_max_pages) if s.ocr_enabled else None
    ingestion = IngestionService(store, embedder, registry, chunker, max_upload_bytes=s.max_upload_mb * 1024 * 1024,
                                 boilerplate_patterns=s.boilerplate_patterns, ocr=ocr)
    t = time.perf_counter()
    if reranker == "auto":
        reranker, reranker_status = load_reranker(s.reranker_model, str(s.model_cache_dir / "hf"))
    else:
        reranker_status = f"{reranker.model_name} (injected)" if reranker else "disabled"
    timings["load_reranker"] = round((time.perf_counter() - t) * 1000, 1)

    retriever = Retriever(
        store, embedder, BM25Index(store.iter_chunks), reranker,
        mode=s.retrieval_mode, candidate_k=s.candidate_k, rerank_k=s.rerank_k, top_k=s.top_k,
        threshold=s.similarity_threshold, rerank_threshold=s.rerank_threshold, rrf_k=s.rrf_k,
        max_per_doc=s.max_chunks_per_doc,
    )
    retriever.refresh_count()
    ingestion.on_change(retriever.on_corpus_change)
    metrics = LatencyMetrics(window=s.metrics_window)
    llm = llm or build_llm(s)
    sessions = SessionStore(ttl_s=s.session_ttl_s)
    rewriter = LLMRewriter(llm, timeout_s=s.rewrite_timeout_s) if s.query_rewriter == "llm" else None
    rag = RAGService(retriever, llm, metrics, sessions=sessions, llm_rewriter=rewriter,
                     log_query_text=s.log_query_text)
    rag.default_campaign = s.default_campaign_id
    return Container(s, embedder, store, registry, ingestion, retriever, llm, rag, metrics, sessions=sessions,
                     reranker_status=reranker_status, ocr=ocr, startup_report=report, startup_timings_ms=timings,
                     transcriber=build_transcriber(s))


def _wait_for(store: QdrantStore, attempts: int = 15, delay_s: float = 2.0) -> None:
    """A Qdrant server may still be starting (or DNS not yet ready) when the API boots."""
    for i in range(attempts):
        if store.ping():
            return
        logger.warning("Qdrant not reachable (attempt %d/%d); retrying in %.0fs", i + 1, attempts, delay_s)
        time.sleep(delay_s)
    raise RuntimeError("Qdrant is unreachable; check QDRANT_URL")


def _reconcile(store: QdrantStore, registry: DocumentRegistry) -> dict[str, list[str]]:
    """Registry and vector store can drift (volume wiped, crash mid-upload, failed rollback).

    * Chunks of documents that are not indexed in the registry are deleted: nothing can list,
      cite consistently or delete them, yet they would still be retrieved.
    * `indexed` rows with no chunks are dropped: they can no longer be searched.
    * Rows left mid-pipeline by a restart are marked failed so the user knows to retry.
    * `failed` rows are kept: they explain why a document is missing.
    """
    in_store = store.document_ids()
    records = registry.all()
    indexed = {r.document_id for r in records if r.status == "indexed"}
    orphan_chunks = sorted(in_store - indexed)
    empty_rows = sorted(indexed - in_store)
    interrupted = sorted(r.document_id for r in records if r.status in IN_PROGRESS)
    for doc_id in orphan_chunks:
        store.delete_document(doc_id)
    for doc_id in empty_rows:
        registry.remove(doc_id)
    for doc_id in interrupted:
        registry.update(doc_id, status="failed", error="Interrupted by a server restart; please upload again.")
    if orphan_chunks:
        logger.warning("Reconcile: deleted chunks of %d unregistered document(s): %s", len(orphan_chunks), orphan_chunks)
    if empty_rows:
        logger.warning("Reconcile: removed %d registry row(s) with no chunks: %s", len(empty_rows), empty_rows)
    if interrupted:
        logger.warning("Reconcile: marked %d interrupted upload(s) as failed: %s", len(interrupted), interrupted)
    return {"deleted_orphan_documents": orphan_chunks, "removed_registry_rows": empty_rows,
            "interrupted_uploads": interrupted}


def seed_sample_data(c: Container, sample_dir: Path) -> list[dict]:
    """Ingest the bundled sample corpus using its manifest (idempotent: duplicates are skipped)."""
    manifest_path = sample_dir / "manifest.json"
    if not manifest_path.exists():
        logger.warning("No sample manifest at %s", manifest_path)
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for entry in manifest["documents"]:
        path = sample_dir / entry["path"]
        try:
            res = c.ingestion.ingest(path.read_bytes(), path.name, manifest_entry=entry,
                                     campaign_id=c.settings.default_campaign_id)
            results.append({"file": entry["path"], "status": res.status, "chunks": res.document.chunk_count})
        except (OSError, UploadRejected) as exc:
            logger.error("Seeding %s failed: %s", path, exc)
            results.append({"file": entry["path"], "status": "error", "error": str(exc)})
    return results
