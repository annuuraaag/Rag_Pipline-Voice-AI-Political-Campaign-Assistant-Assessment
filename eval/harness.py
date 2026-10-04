"""Shared setup for the answer-level evaluation and the latency benchmark.

Builds the real pipeline in-process (bge-small embeddings, in-memory Qdrant, BM25, optional
cross-encoder) over the bundled sample corpus. The LLM comes from the environment exactly as
in the server (GROQ_API_KEY → Groq); without a key the offline extractive fallback is used and
every report says so.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend") if (ROOT / "backend").exists() else str(ROOT))

from app.config import Settings  # noqa: E402
from app.container import Container, build_container, seed_sample_data  # noqa: E402
from app.ingestion.registry import DocumentRegistry  # noqa: E402
from app.retrieval.embedder import FastEmbedder, HashingEmbedder  # noqa: E402
from app.retrieval.vector_store import QdrantStore  # noqa: E402

MODEL = ROOT / "models" / "bge-small-en-v1.5"
CAMPAIGN = "default"


def build_eval_container(tmp: Path, rerank: bool, *, test_embedder: bool = False, **settings) -> Container:
    """`test_embedder`: a bag-of-words stand-in for bge-small, for running the speech and latency
    scripts where the model can't be downloaded. Retrieval quality is then NOT representative
    and every report says so; only use it for numbers that don't depend on retrieval."""
    if not test_embedder and not (MODEL / "tokenizer.json").exists():
        raise SystemExit(f"Embedding model missing: {MODEL}. Run: python scripts/download_models.py")
    os.environ.pop("QDRANT_URL", None)  # always an isolated in-memory index, even inside Docker
    if test_embedder:
        settings.setdefault("similarity_threshold", 0.15)   # hashing-embedder cosine scale
    s = Settings(data_dir=tmp, seed_sample_data=False, log_level="WARNING", _env_file=None, **settings)
    c = build_container(
        s, embedder=HashingEmbedder() if test_embedder else FastEmbedder("BAAI/bge-small-en-v1.5", model_path=str(MODEL)),
        store=QdrantStore("eval", in_memory=True), registry=DocumentRegistry(None),
        reranker="auto" if rerank else None,
    )
    if rerank and c.retriever.reranker is None:
        raise SystemExit(f"--rerank requested but the reranker is unavailable: {c.reranker_status}")
    seed_sample_data(c, ROOT / "sample_data")
    return c


def describe(c: Container) -> dict:
    return {
        "llm": f"{c.llm.name}/{c.llm.model}" + (" (offline fallback: no API key)" if c.llm.is_fallback else ""),
        "retrieval": c.retriever.strategy,
        "gate": "rerank" if c.retriever.reranker else "dense",
        "embedder": c.embedder.model_name + (" (TEST STAND-IN: retrieval not representative)"
                                             if isinstance(c.embedder, HashingEmbedder) else ""),
    }
