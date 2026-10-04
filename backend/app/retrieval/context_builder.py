"""Multi-document context assembly.

A question like "What healthcare initiatives does the candidate propose for
Vijayawada?" is answered by several documents (candidate profile, district plan,
manifesto). Taking the raw top-k often returns k neighbouring chunks of one
document. This builder walks the ranked list and caps chunks per document so the
LLM sees the different sources, while still preferring higher-ranked evidence.
"""
from __future__ import annotations

from app.domain import ScoredChunk


def build_context(ranked: list[ScoredChunk], top_k: int, max_per_doc: int = 2,
                  max_words: int = 900) -> list[ScoredChunk]:
    chosen: list[ScoredChunk] = []
    per_doc: dict[str, int] = {}
    seen_text: set[str] = set()
    words = 0
    overflow: list[ScoredChunk] = []
    for c in ranked:
        m = c.chunk.metadata
        if m.content_hash and m.content_hash in seen_text:  # identical text in two documents
            continue
        if per_doc.get(m.document_id, 0) >= max_per_doc:
            overflow.append(c)
            continue
        if words + m.word_count > max_words and chosen:
            break
        chosen.append(c)
        seen_text.add(m.content_hash)
        per_doc[m.document_id] = per_doc.get(m.document_id, 0) + 1
        words += m.word_count
        if len(chosen) == top_k:
            return chosen
    # Fewer distinct documents than slots: fill the rest with the best skipped chunks.
    for c in overflow:
        if len(chosen) == top_k or words + c.chunk.metadata.word_count > max_words:
            break
        chosen.append(c)
        words += c.chunk.metadata.word_count
    chosen.sort(key=lambda c: c.rank)
    return chosen
