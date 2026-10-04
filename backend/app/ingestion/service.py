"""Ingestion orchestration: bytes → parse (OCR fallback) → metadata → chunk → embed → index.

Every upload belongs to one campaign and gets a document record whose status moves
uploaded → parsing → chunking → embedding → indexed, or failed (with the error) at any
step. Partial chunks of a failed upload are removed, so only indexed documents are
searchable.

Idempotency rules (per campaign):
* Same content hash already indexed → no-op, status ``duplicate``.
* Same filename + district, different content → new version indexed, then the old one
  removed; status ``reindexed``.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import PurePath

from pydantic import BaseModel, Field

from app.domain import DEFAULT_CAMPAIGN, Chunk
from app.ingestion.chunker import Chunker
from app.ingestion.cleaning import strip_boilerplate
from app.ingestion.metadata import content_hash, enrich_chunks, resolve_document_metadata, slugify
from app.ingestion.ocr import OcrEngine
from app.ingestion.parsers import SUPPORTED_EXTENSIONS, DocumentParseError, parse_document
from app.ingestion.registry import DocumentRecord, DocumentRegistry
from app.observability.logging import log_event
from app.observability.timing import StageTimer
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)
_CAMPAIGN_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def normalize_campaign(campaign_id: str | None) -> str:
    cid = (campaign_id or DEFAULT_CAMPAIGN).strip().lower()
    if not _CAMPAIGN_RE.match(cid):
        raise ValueError("campaign_id must be 1-64 chars: lowercase letters, digits, '-' or '_'.")
    return cid


class UploadRejected(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


class IngestResult(BaseModel):
    status: str  # indexed | duplicate | reindexed
    document: DocumentRecord
    replaced_document_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class IngestionService:
    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder,
        registry: DocumentRegistry,
        chunker: Chunker,
        max_upload_bytes: int = 20 * 1024 * 1024,
        boilerplate_patterns: list[str] | None = None,
        ocr: OcrEngine | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.registry = registry
        self.chunker = chunker
        self.max_upload_bytes = max_upload_bytes
        self._boilerplate = [re.compile(p, re.DOTALL) for p in (boilerplate_patterns or [])]
        self.ocr = ocr
        self.revision = 0  # bumped on every corpus change; lexical indexes key their cache on it
        self._listeners: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    def on_change(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def _changed(self) -> None:
        self.revision += 1
        for fn in self._listeners:
            fn()

    def ingest(
        self,
        data: bytes,
        filename: str,
        overrides: Mapping[str, str | None] | None = None,
        manifest_entry: Mapping[str, str | None] | None = None,
        replace_document_id: str | None = None,
        campaign_id: str | None = None,
    ) -> IngestResult:
        filename = PurePath(filename or "").name
        ext = PurePath(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UploadRejected(f"Unsupported file type '{ext or '?'}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}", 415)
        if len(data) > self.max_upload_bytes:
            raise UploadRejected(f"File exceeds {self.max_upload_bytes // (1024 * 1024)} MB limit.", 413)
        if not data:
            raise UploadRejected("File is empty.", 422)
        try:
            campaign = normalize_campaign(campaign_id)
        except ValueError as exc:
            raise UploadRejected(str(exc), 422) from exc

        timer = StageTimer()
        digest = content_hash(data)
        with self._lock:
            existing = self.registry.by_hash(digest, campaign)
            if existing:
                return IngestResult(status="duplicate", document=existing, timings_ms=timer.as_dict())
            if replace_document_id:
                target = self.registry.get(replace_document_id)
                if target is None or target.campaign_id != campaign:
                    raise UploadRejected(f"replace_document_id '{replace_document_id}' does not exist "
                                         f"in campaign '{campaign}'.", 404)

            # The campaign is part of the id, so the same file in two campaigns never collides.
            doc_key = hashlib.sha256(f"{campaign}:{digest}".encode()).hexdigest()[:8]
            document_id = f"{slugify(filename)}_{doc_key}"
            uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            record = DocumentRecord(
                campaign_id=campaign, document_id=document_id, filename=filename, title=PurePath(filename).stem,
                file_type=ext.lstrip("."), content_hash=digest, size_bytes=len(data), uploaded_at=uploaded_at,
                status="uploaded", chunk_strategy=self.chunker.name, embedding_model=self.embedder.model_name,
            )
            self.registry.add(record)   # visible (status: uploaded) from the first moment
            try:
                result = self._run(record, data, filename, overrides, manifest_entry, replace_document_id, timer)
            except UploadRejected as exc:
                self._fail(document_id, str(exc))
                raise
            except Exception as exc:
                logger.exception("Ingestion of %s failed", filename)
                self._fail(document_id, f"{exc.__class__.__name__}: {exc}")
                raise UploadRejected(f"Ingestion failed: {exc.__class__.__name__}", 500) from exc
            self._changed()

        log_event("ingest", campaign_id=campaign, document_id=document_id, filename=filename,
                  chunks=result.document.chunk_count, ocr_pages=result.document.ocr_pages,
                  status=result.status, timings_ms=result.timings_ms)
        return result

    def _run(self, record: DocumentRecord, data: bytes, filename: str, overrides, manifest_entry,
             replace_document_id: str | None, timer: StageTimer) -> IngestResult:
        doc_id = record.document_id
        self.registry.update(doc_id, status="parsing")
        with timer.stage("parse"):
            try:
                parsed = parse_document(data, filename, self.ocr)
            except DocumentParseError as exc:
                raise UploadRejected(str(exc), 422) from exc
            parsed.blocks = strip_boilerplate(parsed.blocks, self._boilerplate)
            if not parsed.blocks:
                raise UploadRejected("Document contains only boilerplate text.", 422)

        self.registry.update(doc_id, status="chunking", title=parsed.title, page_count=parsed.page_count,
                             ocr_pages=parsed.ocr_pages, file_type=parsed.file_type)
        with timer.stage("chunk"):
            doc_meta = resolve_document_metadata(parsed, filename, overrides, manifest_entry)
            replaced = (self.registry.get(replace_document_id) if replace_document_id
                        else self.registry.find(filename, doc_meta.district, record.campaign_id))
            raws = self.chunker.chunk(parsed)
            chunks: list[Chunk] = enrich_chunks(
                raws, parsed, doc_meta, document_id=doc_id, document_name=filename,
                strategy=self.chunker.name, uploaded_at=record.uploaded_at, campaign_id=record.campaign_id,
                embedding_model=self.embedder.model_name, count_tokens=self.embedder.count_tokens,
            )
        if not chunks:
            raise UploadRejected("Document produced no indexable chunks.", 422)

        self.registry.update(doc_id, status="embedding", metadata=doc_meta)
        with timer.stage("embed"):
            vectors = self.embedder.embed_passages([c.embed_text for c in chunks])

        warnings: list[str] = []
        oversized = [c.metadata.chunk_id for c in chunks if self.embedder.count_tokens(c.embed_text) > 510]
        if oversized:
            warnings.append(f"{len(oversized)} chunk(s) exceed the embedder's 512-token window and were truncated.")
        with timer.stage("index"):
            self._index(doc_id, chunks, vectors)
        content_types: dict[str, int] = {}
        for c in chunks:
            content_types[c.metadata.content_type] = content_types.get(c.metadata.content_type, 0) + 1
        final = self.registry.update(
            doc_id, status="indexed", chunk_count=len(chunks),
            section_count=len({c.metadata.section for c in chunks}), content_types=content_types,
            token_count=sum(c.metadata.token_count for c in chunks),
        )
        if replaced and replaced.document_id != doc_id and not self._retire(replaced):
            warnings.append(f"Old version {replaced.document_id} could not be removed; both versions are indexed.")
            replaced = None
        if parsed.ocr_pages:
            warnings.append(f"OCR used for {'the image' if parsed.ocr_pages == [0] else f'pages {parsed.ocr_pages}'}.")
        return IngestResult(
            status="reindexed" if replaced else "indexed",
            document=final,
            replaced_document_id=replaced.document_id if replaced else None,
            warnings=warnings,
            timings_ms=timer.as_dict(),
        )

    def _fail(self, document_id: str, error: str) -> None:
        """Keep the record (status failed, with the reason) so the user can see what happened and retry."""
        try:
            self.store.delete_document(document_id)  # remove any partially written chunks
        except Exception:
            logger.exception("Could not remove partial chunks of %s; startup reconciliation will", document_id)
        self.registry.update(document_id, status="failed", error=error[:500], chunk_count=0)

    def _index(self, document_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Write the new version's chunks. On failure they are rolled back by _fail()."""
        self.store.ensure_collection(self.embedder.dim)
        self.store.upsert(chunks, vectors)

    def _retire(self, old: DocumentRecord) -> bool:
        """Remove a replaced version only after the new one is fully indexed.

        Chunks go before the registry row: if deleting chunks fails, the old document
        stays complete and listed rather than becoming a registry entry with no chunks.
        """
        try:
            self.store.delete_document(old.document_id)
        except Exception:
            logger.exception("Could not delete chunks of replaced document %s", old.document_id)
            return False
        self.registry.remove(old.document_id)
        return True

    def delete_campaign(self, campaign_id: str) -> int:
        """Remove every chunk and record of one campaign. Returns the number of documents removed."""
        with self._lock:
            self.store.delete_campaign(campaign_id)
            recs = self.registry.all(campaign_id)
            for r in recs:
                self.registry.remove(r.document_id)
            self._changed()
            return len(recs)

    def delete(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            rec = self.registry.get(document_id)
            if rec is None:
                return None
            self.store.delete_document(document_id)
            self.registry.remove(document_id)
            self._changed()
            return rec
