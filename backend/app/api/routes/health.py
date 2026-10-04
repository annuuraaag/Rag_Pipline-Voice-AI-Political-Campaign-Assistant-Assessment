"""GET /health — component-level diagnostics, GET /metrics — latency percentiles."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from app.api.deps import get_container
from app.container import Container
from app.voice.endpointing import endpointing_info

router = APIRouter(tags=["system"])


@router.get("/health", summary="Component health and configuration diagnostics")
def health(c: Container = Depends(get_container)) -> dict:
    s = c.settings
    store_ok = c.store.ping()
    chunks = c.retriever.chunk_count if store_ok else 0  # cached: never waits on an upload's store lock
    # Extractive is "configured" only if explicitly chosen; otherwise it means a key is missing.
    llm_configured = (not c.llm.is_fallback) or s.llm_provider == "extractive"
    status = "ok"
    if not store_ok:
        status = "down"
    elif not llm_configured or chunks == 0:
        status = "degraded"
    return {
        "status": status,
        "uptime_s": round(time.time() - c.started_at, 1),
        "components": {
            "vector_store": {"ok": store_ok, "mode": c.store.mode, "collection": c.store.collection, "chunks": chunks},
            "embedder": {"ok": True, "model": c.embedder.model_name, "dim": c.embedder.dim,
                         "query_cache": getattr(c.embedder, "cache_stats", None)},
            "llm": {
                "ok": llm_configured,
                "provider": c.llm.name,
                "model": c.llm.model,
                "fallback_active": c.llm.is_fallback,
                "model_check": getattr(c.llm, "model_check", None),
                "note": None if llm_configured else f"No API key for '{s.llm_provider}'; using offline extractive fallback.",
            },
            "reranker": {"ok": c.retriever.reranker is not None, "status": c.reranker_status},
            "bm25": {"ok": True, "indexed_chunks": c.retriever.bm25.size},
            "documents": {"count": sum(c.registry.campaigns().values()), "campaigns": c.registry.campaigns(),
                          "failed": sum(1 for d in c.registry.all() if d.status == "failed"),
                          "corpus_revision": c.ingestion.revision},
            "ocr": {"ok": bool(c.ocr and c.ocr.available), "enabled": s.ocr_enabled,
                    "engine": "RapidOCR (ONNX)" if c.ocr else None,
                    "status": c.ocr.status if c.ocr else "disabled"},
            "voice": {"ok": True, "websocket": "/ws/voice",
                      "server_transcription": f"{c.transcriber.name}/{c.transcriber.model}" if c.transcriber else None,
                      "server_speech": c.speech_status,
                      "speculation": {"debounce_ms": s.voice_debounce_ms, "stable_ms": s.voice_stable_ms,
                                      "min_words": s.voice_min_words, "word_step": s.voice_word_step},
                      "endpointing": endpointing_info(s)},
            "index_consistency": {
                "chunks_from_other_embedding_models": c.store.count_stale(c.embedder.model_name) if store_ok else None,
                "startup": c.startup_report,
            },
        },
        "config": {
            "chunk_strategy": s.chunk_strategy,
            "chunk_max_tokens": s.chunk_max_tokens,
            "default_campaign_id": s.default_campaign_id,
            "candidate_k": s.candidate_k,
            "top_k": s.top_k,
            "similarity_threshold": s.similarity_threshold,
            "rerank_k": s.rerank_k,
            "rerank_threshold": s.rerank_threshold,
            "gate": "rerank" if c.retriever.reranker is not None else "dense",
            "retrieval_strategy": c.retriever.strategy,
            "query_rewriter": s.query_rewriter,
            "max_chunks_per_doc": s.max_chunks_per_doc,
        },
        "startup_ms": c.startup_timings_ms,
    }


@router.get("/metrics", summary="Rolling latency percentiles per endpoint and pipeline stage")
def metrics(c: Container = Depends(get_container)) -> dict:
    return c.metrics.summary()
