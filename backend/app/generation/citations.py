"""Citation construction from retrieval metadata.

The LLM only emits ``[S#]`` markers. Here they are validated against the sources
actually sent; unknown markers are stripped, and each citation's file/page/section
comes from chunk metadata, never from generated text.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from app.domain import ScoredChunk
from app.generation.prompt import REFUSAL_TEXT

_MARKER = re.compile(r"\[(S\d+(?:\s*[,;]\s*S?\d+)*)\]")


class Citation(BaseModel):
    source_id: str          # "S1"
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None
    page_end: int | None
    section: str
    district: str
    category: str
    topic: str
    score: float
    cited: bool             # True if the answer text references it
    snippet: str


def _marker_numbers(marker_body: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", marker_body)]


def is_refusal(answer: str) -> bool:
    return REFUSAL_TEXT.lower()[:60] in answer.lower() or "couldn't find sufficiently relevant" in answer.lower()


def finalize_answer(answer: str, sources: list[ScoredChunk]) -> tuple[str, list[Citation], list[int]]:
    """Return (clean_answer, citations, invalid_marker_numbers)."""
    n = len(sources)
    cited: set[int] = set()
    invalid: list[int] = []

    def _fix(m: re.Match[str]) -> str:
        nums = _marker_numbers(m.group(1))
        good = [x for x in nums if 1 <= x <= n]
        invalid.extend(x for x in nums if not 1 <= x <= n)
        cited.update(good)
        return "".join(f"[S{x}]" for x in good)

    clean = _MARKER.sub(_fix, answer).strip()
    clean = re.sub(r"\s+([.,;])", r"\1", clean)
    refused = is_refusal(clean)

    citations = []
    for i, c in enumerate(sources, start=1):
        m = c.chunk.metadata
        citations.append(
            Citation(
                source_id=f"S{i}", chunk_id=m.chunk_id, document_id=m.document_id, document_name=m.document_name,
                page=m.page, page_end=m.page_end, section=m.section, district=m.district, category=m.category,
                topic=m.topic, score=round(c.score, 4), cited=(i in cited) and not refused,
                snippet=c.chunk.text[:280] + ("…" if len(c.chunk.text) > 280 else ""),
            )
        )
    return clean, citations, invalid
