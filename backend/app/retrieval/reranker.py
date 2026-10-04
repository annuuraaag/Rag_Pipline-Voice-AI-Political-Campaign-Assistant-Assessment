"""Second-stage reranking with a cross-encoder.

Stage 1 (dense + BM25) is built for recall: cheap, independent query/chunk
encodings. A cross-encoder reads query and chunk *together*, which is far more
precise but costs one model pass per pair, so it only sees the top N fused
candidates. Its sigmoid score (0–1) is also the one per-chunk relevance score
that is comparable across queries, so the answerability gate uses it.

Model: ms-marco-MiniLM-L-6-v2 (ONNX via fastembed, ~80 MB, CPU-friendly).
If the model cannot be loaded (e.g. offline and not pre-downloaded) the pipeline
runs without reranking and says so in /health and in every trace.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Protocol

from app.domain import ScoredChunk

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2", cache_dir: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name, cache_dir=cache_dir) if cache_dir else TextCrossEncoder(model_name)
        self.model_name = model_name
        self._lock = threading.Lock()

    def score(self, query: str, texts: list[str]) -> list[float]:
        with self._lock:
            logits = list(self._model.rerank(query, texts, batch_size=16))
        return [_sigmoid(float(x)) for x in logits]


def load_reranker(model_name: str, cache_dir: str | None) -> tuple[Reranker | None, str]:
    """Returns (reranker or None, status message)."""
    if not model_name:
        return None, "disabled (RERANKER_MODEL empty)"
    try:
        return CrossEncoderReranker(model_name, cache_dir), f"{model_name} loaded"
    except Exception as exc:  # network policy, missing files, onnx error
        logger.warning("Reranker '%s' unavailable, continuing without it: %s", model_name, exc)
        return None, f"unavailable: {exc.__class__.__name__}"


def rerank(reranker: Reranker, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
    """Re-score candidates (keeps stage-1 scores in the trace) and sort by cross-encoder score."""
    if not candidates:
        return []
    # embed_text carries the document/section header, which helps the cross-encoder too.
    scores = reranker.score(query, [c.chunk.embed_text for c in candidates])
    out = []
    for c, s in zip(candidates, scores, strict=True):
        out.append(c.model_copy(update={"score": s, "scores": {**c.scores, "rerank": round(s, 4)}}))
    out.sort(key=lambda c: -c.score)
    for i, c in enumerate(out, start=1):
        c.rank = i
    return out
