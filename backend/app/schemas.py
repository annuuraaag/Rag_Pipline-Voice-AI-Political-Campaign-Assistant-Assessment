"""HTTP request/response models (the public API contract)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.domain import MetadataFilter, ScoredChunk
from app.generation.citations import Citation
from app.generation.verify import Verification
from app.ingestion.registry import DocumentRecord
from app.lexicon import normalize_district


class RetrievedChunk(BaseModel):
    rank: int
    score: float
    scores: dict[str, float] = Field(default_factory=dict)
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None = None
    page_end: int | None = None
    section: str = ""
    district: str
    category: str
    topic: str
    source: str
    content_type: str = "text"
    text: str

    @classmethod
    def from_scored(cls, c: ScoredChunk, rank: int | None = None) -> "RetrievedChunk":
        m = c.chunk.metadata
        return cls(
            rank=rank or c.rank, score=round(c.score, 4), scores=c.scores, chunk_id=m.chunk_id,
            document_id=m.document_id, document_name=m.document_name, page=m.page, page_end=m.page_end,
            section=m.section, district=m.district, category=m.category, topic=m.topic, source=m.source,
            content_type=m.content_type, text=c.chunk.text,
        )


class FilterIn(BaseModel):
    district: str | None = Field(default=None, examples=["vijayawada"])
    category: str | None = Field(default=None, examples=["manifesto"])
    topic: str | None = Field(default=None, examples=["healthcare"])
    document_ids: list[str] | None = None
    content_type: Literal["text", "table", "ocr"] | None = None

    def to_domain(self) -> MetadataFilter:
        return MetadataFilter(
            district=normalize_district(self.district),
            category=self.category.lower() if self.category else None,
            topic=self.topic.lower() if self.topic else None,
            document_ids=self.document_ids or None,
            content_type=self.content_type,
        )


_CAMPAIGN_FIELD = dict(default=None, description="Campaign (tenant) to search. Defaults to the server's default campaign.",
                       examples=["default"])


class RetrieveRequest(BaseModel):
    campaign_id: str | None = Field(**_CAMPAIGN_FIELD)
    query: str = Field(min_length=1, max_length=1000, examples=["What healthcare schemes are available in Vijayawada?"])
    session_id: str | None = Field(default=None, description="Read-only: use this conversation's state for rewriting.")
    filters: FilterIn | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    include_candidates: bool = True


class RetrieveResponse(BaseModel):
    request_id: str
    campaign_id: str
    query: str
    rewritten_query: str
    rewrite_reasons: list[str] = Field(default_factory=list)
    filters: dict
    filter_relaxed: bool = False
    strategy: str
    gate: str
    threshold: float
    top_score: float
    answerable: bool
    counts: dict[str, int] = Field(default_factory=dict)
    results: list[RetrievedChunk]
    candidates: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: dict[str, float]


class QueryRequest(BaseModel):
    campaign_id: str | None = Field(**_CAMPAIGN_FIELD)
    query: str = Field(min_length=1, max_length=1000, examples=["What healthcare initiatives are proposed for Vijayawada?"])
    session_id: str | None = Field(
        default=None,
        description="Conversation id. Enables follow-ups (\"What about education?\") via conversation memory.",
    )
    filters: FilterIn | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    stream: bool = Field(default=False, description="If true, respond with Server-Sent Events.")


class LLMInfo(BaseModel):
    provider: str
    model: str
    fallback: bool = False
    error: str | None = None


class RetrievalTrace(BaseModel):
    original_query: str = ""
    rewritten_query: str
    rewrite_reasons: list[str] = Field(default_factory=list)
    rewrite_method: str = "rules"
    filters: dict
    filter_relaxed: bool = False
    strategy: str
    gate: str = "dense"
    counts: dict[str, int] = Field(default_factory=dict)
    threshold: float
    top_score: float
    candidate_count: int
    final_count: int
    results: list[RetrievedChunk]
    candidates: list[RetrievedChunk] = Field(default_factory=list)


class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    answerable: bool
    refusal_reason: str | None = None  # no_documents | below_threshold | model_refused | unverified
    citations: list[Citation]
    retrieval: RetrievalTrace
    llm: LLMInfo
    latency_ms: dict[str, float]
    conversation: dict | None = None
    verification: Verification | None = None  # claim-by-claim check of the answer against its sources


class UploadResponse(BaseModel):
    status: str
    document: DocumentRecord
    replaced_document_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float]


class DocumentChunk(BaseModel):
    campaign_id: str
    chunk_id: str
    chunk_index: int
    page: int | None
    page_end: int | None
    section: str
    district: str
    districts_mentioned: list[str]
    category: str
    topic: str
    word_count: int
    token_count: int
    content_type: str
    embedding_model: str
    text: str
    embed_text: str
