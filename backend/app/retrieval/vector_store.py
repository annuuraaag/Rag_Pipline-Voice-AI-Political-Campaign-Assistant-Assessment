"""Qdrant vector store wrapper.

Why Qdrant: metadata (payload) filters are applied *inside* the ANN search, so a
district filter does not shrink the result set after the fact; the same client
runs embedded (tests, local dev) or against a server (Docker).
"""
from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.domain import Chunk, ChunkMetadata, MetadataFilter, ScoredChunk

logger = logging.getLogger(__name__)

_NS = uuid.UUID("6f1c2a9e-3b7d-4c55-9a4e-2d8f0b1c7e11")
INDEXED_FIELDS = ("district", "districts_mentioned", "category", "topic", "document_id", "content_type",
                  "embedding_model")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


def build_filter(f: MetadataFilter | None) -> qm.Filter | None:
    """Translate a MetadataFilter into a Qdrant filter.

    District matching includes statewide content and chunks that mention the
    district: a Vijayawada resident should also see statewide manifesto promises.
    """
    if f is None or f.is_empty():
        return None
    must: list[Any] = []
    if f.campaign_id:
        must.append(qm.FieldCondition(key="campaign_id", match=qm.MatchValue(value=f.campaign_id)))
    if f.content_type:
        must.append(qm.FieldCondition(key="content_type", match=qm.MatchValue(value=f.content_type)))
    if f.district:
        must.append(
            qm.Filter(
                should=[
                    qm.FieldCondition(key="district", match=qm.MatchAny(any=[f.district, "statewide"])),
                    qm.FieldCondition(key="districts_mentioned", match=qm.MatchValue(value=f.district)),
                ]
            )
        )
    if f.category:
        must.append(qm.FieldCondition(key="category", match=qm.MatchValue(value=f.category)))
    if f.topic:
        must.append(qm.FieldCondition(key="topic", match=qm.MatchValue(value=f.topic)))
    if f.document_ids:
        must.append(qm.FieldCondition(key="document_id", match=qm.MatchAny(any=f.document_ids)))
    return qm.Filter(must=must)


def _payload(chunk: Chunk) -> dict[str, Any]:
    return {"text": chunk.text, "embed_text": chunk.embed_text, **chunk.metadata.model_dump()}


def _chunk_from_payload(p: dict[str, Any]) -> Chunk:
    meta_fields = set(ChunkMetadata.model_fields)
    return Chunk(
        text=p.get("text", ""),
        embed_text=p.get("embed_text", p.get("text", "")),
        metadata=ChunkMetadata(**{k: v for k, v in p.items() if k in meta_fields}),
    )


class QdrantStore:
    def __init__(self, collection: str, url: str = "", path: Path | None = None, in_memory: bool = False):
        self.collection = collection
        if in_memory:
            self.client = QdrantClient(":memory:")
            self.mode = "memory"
        elif url:
            self.client = QdrantClient(url=url, timeout=10)
            self.mode = "server"
        else:
            assert path is not None
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
            self.mode = "embedded"
        # The embedded client is not safe for concurrent writers; a lock is cheap at this scale.
        self._lock = threading.RLock()
        self._dim: int | None = None

    def ensure_collection(self, dim: int) -> None:
        with self._lock:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    self.collection,
                    vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                )
                if self.mode == "server":  # payload indexes are a server feature
                    # campaign_id is the tenant key: Qdrant co-locates each tenant's points,
                    # so filtered searches stay fast as campaigns are added.
                    self.client.create_payload_index(
                        self.collection, "campaign_id",
                        qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD, is_tenant=True),
                    )
                    for field in INDEXED_FIELDS:
                        self.client.create_payload_index(self.collection, field, qm.PayloadSchemaType.KEYWORD)
            else:
                existing = self.client.get_collection(self.collection).config.params.vectors
                size = getattr(existing, "size", None)
                if size is not None and size != dim:
                    raise RuntimeError(
                        f"Collection '{self.collection}' has dim {size} but embedder produces {dim}. "
                        "Delete the collection (or data/ dir) and re-ingest after changing models."
                    )
            self._dim = dim

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            qm.PointStruct(id=point_id(c.metadata.chunk_id), vector=v, payload=_payload(c))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        # Lock per batch (not per document) so concurrent searches interleave with large uploads.
        for i in range(0, len(points), 64):
            with self._lock:
                self.client.upsert(self.collection, points=points[i : i + 64], wait=True)

    def search(self, vector: list[float], limit: int, flt: MetadataFilter | None = None) -> list[ScoredChunk]:
        with self._lock:
            res = self.client.query_points(
                self.collection,
                query=vector,
                limit=limit,
                query_filter=build_filter(flt),
                with_payload=True,
            )
        return [
            ScoredChunk(chunk=_chunk_from_payload(p.payload or {}), score=float(p.score), rank=i + 1,
                        retriever="dense", scores={"dense": round(float(p.score), 4)})
            for i, p in enumerate(res.points)
        ]

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self.client.delete(
                self.collection,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))])
                ),
                wait=True,
            )

    def iter_chunks(self, flt: MetadataFilter | None = None) -> list[Chunk]:
        """All chunks (optionally filtered), ordered by document then position. Used by BM25 and the chunk browser."""
        out: list[Chunk] = []
        offset = None
        with self._lock:
            while True:
                points, offset = self.client.scroll(
                    self.collection, scroll_filter=build_filter(flt), limit=256, offset=offset, with_payload=True
                )
                out.extend(_chunk_from_payload(p.payload or {}) for p in points)
                if offset is None:
                    break
        out.sort(key=lambda c: (c.metadata.document_id, c.metadata.chunk_index))
        return out

    def document_ids(self) -> set[str]:
        """Distinct document_ids present in the collection (payload-only scroll, no vectors or text)."""
        ids: set[str] = set()
        offset = None
        with self._lock:
            if not self.client.collection_exists(self.collection):
                return ids
            while True:
                points, offset = self.client.scroll(
                    self.collection, limit=1024, offset=offset, with_payload=["document_id"], with_vectors=False
                )
                ids.update(str((p.payload or {}).get("document_id")) for p in points)
                if offset is None:
                    return ids

    def delete_campaign(self, campaign_id: str) -> None:
        with self._lock:
            if self.client.collection_exists(self.collection):
                self.client.delete(
                    self.collection,
                    points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                        qm.FieldCondition(key="campaign_id", match=qm.MatchValue(value=campaign_id))])),
                    wait=True,
                )

    def backfill_campaign(self, campaign_id: str) -> int:
        """Points indexed before campaign isolation existed get the default campaign (one-time migration)."""
        missing = qm.Filter(must=[qm.IsEmptyCondition(is_empty=qm.PayloadField(key="campaign_id"))])
        with self._lock:
            if not self.client.collection_exists(self.collection):
                return 0
            n = self.client.count(self.collection, count_filter=missing, exact=True).count
            if n:
                self.client.set_payload(self.collection, payload={"campaign_id": campaign_id},
                                        points=qm.FilterSelector(filter=missing), wait=True)
            return n

    def count_stale(self, embedding_model: str) -> int:
        """Points embedded with a different (or unrecorded) model: their vectors don't match new queries."""
        stale = qm.Filter(must_not=[qm.FieldCondition(key="embedding_model", match=qm.MatchValue(value=embedding_model))])
        with self._lock:
            if not self.client.collection_exists(self.collection):
                return 0
            return self.client.count(self.collection, count_filter=stale, exact=True).count

    def count(self) -> int:
        with self._lock:
            if not self.client.collection_exists(self.collection):
                return 0
            return self.client.count(self.collection, exact=True).count

    def ping(self) -> bool:
        try:
            with self._lock:
                self.client.get_collections()
            return True
        except Exception as exc:  # pragma: no cover - depends on infra
            logger.warning("Qdrant ping failed: %s", exc)
            return False

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # pragma: no cover
            pass
