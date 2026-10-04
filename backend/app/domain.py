"""Core value types shared by ingestion, retrieval and generation.

Kept free of framework code so every layer (and the eval scripts) can use them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


# ── Ingestion ────────────────────────────────────────────────────────────
@dataclass
class ParsedBlock:
    """A paragraph-level unit of text with its structural position."""

    text: str
    page: int | None = None                         # 1-based; None for formats without pages
    section_path: list[str] = field(default_factory=list)  # heading breadcrumb, outermost first
    kind: str = "paragraph"                         # paragraph | list_item | table_row
    content_type: str = "text"                      # text | table | ocr  (tables/OCR are chunked separately)


@dataclass
class ParsedDocument:
    title: str
    file_type: str
    blocks: list[ParsedBlock]
    page_count: int | None = None
    front_matter: dict[str, str] = field(default_factory=dict)
    ocr_pages: list[int] = field(default_factory=list)  # pages whose text came from OCR (0 = whole image)

    @property
    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


class DocumentMetadata(BaseModel):
    """Document-level metadata. Assessment-required fields: district, category, source, topic."""

    district: str = "statewide"
    category: str = "other"
    topic: str = "general"
    source: str = ""
    candidate: str | None = None


DEFAULT_CAMPAIGN = "default"


class ChunkMetadata(BaseModel):
    campaign_id: str = DEFAULT_CAMPAIGN     # tenant key: every search is filtered on it
    document_id: str
    document_name: str
    document_title: str
    file_type: str
    chunk_id: str
    chunk_index: int
    page: int | None = None
    page_end: int | None = None
    section: str = ""
    district: str = "statewide"
    districts_mentioned: list[str] = Field(default_factory=list)
    category: str = "other"
    topic: str = "general"
    source: str = ""
    candidate: str | None = None
    content_hash: str = ""
    chunk_strategy: str = ""
    uploaded_at: str = ""
    word_count: int = 0
    token_count: int = 0                    # embedding-model tokens in `text`
    content_type: str = "text"              # text | table | ocr
    embedding_model: str = ""               # model that produced the vector (detects stale points)


class Chunk(BaseModel):
    text: str           # what users see and what gets cited
    embed_text: str     # text + contextual header; what gets embedded / BM25-indexed
    metadata: ChunkMetadata


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int = 0
    retriever: str = "dense"
    scores: dict[str, float] = Field(default_factory=dict)  # per-stage scores for the trace


# ── Retrieval filters ────────────────────────────────────────────────────
class MetadataFilter(BaseModel):
    campaign_id: str | None = None   # required by the retriever on every search (tenant isolation)
    district: str | None = None
    category: str | None = None
    topic: str | None = None
    document_ids: list[str] | None = None
    content_type: str | None = None

    def is_empty(self) -> bool:
        return not any([self.campaign_id, self.district, self.category, self.topic, self.document_ids,
                        self.content_type])

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def matches(self, m: ChunkMetadata) -> bool:
        """Python twin of vector_store.build_filter, used by the in-memory BM25 index.

        Keep the two in sync: a district matches its own chunks, statewide chunks,
        and chunks that mention the district.
        """
        if self.campaign_id and m.campaign_id != self.campaign_id:
            return False
        if self.content_type and m.content_type != self.content_type:
            return False
        if self.district and not (m.district in (self.district, "statewide") or self.district in m.districts_mentioned):
            return False
        if self.category and m.category != self.category:
            return False
        if self.topic and m.topic != self.topic:
            return False
        if self.document_ids and m.document_id not in self.document_ids:
            return False
        return True
