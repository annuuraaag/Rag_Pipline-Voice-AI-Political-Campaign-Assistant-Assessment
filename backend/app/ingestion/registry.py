"""Document registry: one JSON file mapping document_id → DocumentRecord.

The vector store holds chunks; the registry holds document-level facts (hash,
size, chunk count) needed for dedup, re-indexing and the document browser.
A JSON file is enough for a single-node prototype; production would use a DB.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.domain import DEFAULT_CAMPAIGN, DocumentMetadata

# Lifecycle: uploaded → parsing → chunking → embedding → indexed, or failed (with error) at any step.
# Only `indexed` documents have chunks in the vector store, so only they are searchable.
Status = Literal["uploaded", "parsing", "chunking", "embedding", "indexed", "failed"]
IN_PROGRESS = ("uploaded", "parsing", "chunking", "embedding")


class DocumentRecord(BaseModel):
    campaign_id: str = DEFAULT_CAMPAIGN
    document_id: str
    filename: str
    title: str
    file_type: str
    content_hash: str
    size_bytes: int
    page_count: int | None = None
    chunk_count: int = 0
    section_count: int = 0
    chunk_strategy: str = ""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    uploaded_at: str
    status: Status = "indexed"   # records written before lifecycle tracking were all indexed
    error: str | None = None
    ocr_pages: list[int] = Field(default_factory=list)
    content_types: dict[str, int] = Field(default_factory=dict)  # chunks per content type
    token_count: int = 0
    embedding_model: str = ""


class DocumentRegistry:
    def __init__(self, path: Path | None):
        self.path = path
        self._docs: dict[str, DocumentRecord] = {}
        self._lock = threading.RLock()
        if path and path.exists():
            raw = json.loads(path.read_text(encoding="utf-8") or "{}")
            self._docs = {k: DocumentRecord(**v) for k, v in raw.items()}

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: v.model_dump() for k, v in self._docs.items()}, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic on POSIX

    def add(self, rec: DocumentRecord) -> None:
        with self._lock:
            self._docs[rec.document_id] = rec
            self._save()

    def remove(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            rec = self._docs.pop(document_id, None)
            self._save()
            return rec

    # Reads take the lock too: iterating the dict while another thread adds/removes
    # a document raises "dictionary changed size during iteration".
    def get(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._docs.get(document_id)

    def update(self, document_id: str, **changes) -> DocumentRecord | None:
        with self._lock:
            rec = self._docs.get(document_id)
            if rec is None:
                return None
            rec = rec.model_copy(update=changes)
            self._docs[document_id] = rec
            self._save()
            return rec

    def by_hash(self, content_hash: str, campaign_id: str = DEFAULT_CAMPAIGN) -> DocumentRecord | None:
        """Indexed duplicate in the same campaign (failed attempts don't block a retry)."""
        with self._lock:
            return next((d for d in self._docs.values() if d.content_hash == content_hash
                         and d.campaign_id == campaign_id and d.status == "indexed"), None)

    def find(self, filename: str, district: str, campaign_id: str = DEFAULT_CAMPAIGN) -> DocumentRecord | None:
        """Same filename alone is not identity: two districts may both upload `plan.md`."""
        with self._lock:
            return next(
                (d for d in self._docs.values() if d.filename == filename and d.metadata.district == district
                 and d.campaign_id == campaign_id and d.status == "indexed"),
                None,
            )

    def all(self, campaign_id: str | None = None) -> list[DocumentRecord]:
        with self._lock:
            docs = [d for d in self._docs.values() if campaign_id is None or d.campaign_id == campaign_id]
            return sorted(docs, key=lambda d: d.uploaded_at)

    def campaigns(self) -> dict[str, int]:
        """Indexed document count per campaign."""
        with self._lock:
            out: dict[str, int] = {}
            for d in self._docs.values():
                if d.status == "indexed":
                    out[d.campaign_id] = out.get(d.campaign_id, 0) + 1
            return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._docs)
