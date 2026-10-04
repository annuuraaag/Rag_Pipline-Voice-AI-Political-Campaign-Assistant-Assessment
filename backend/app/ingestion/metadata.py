"""Metadata resolution and chunk enrichment.

Precedence for document-level fields (most explicit wins):
    upload form fields  >  document front matter  >  corpus manifest  >  auto-detection

Chunk-level ``topic`` and ``district`` are refined from the chunk's own section
heading and text, so a statewide manifesto's "Vijayawada" section is still
filterable by district.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from pathlib import PurePath

from app.domain import Chunk, ChunkMetadata, DocumentMetadata, ParsedDocument
from app.ingestion.chunker import RawChunk
from app.lexicon import DISTRICT_DISPLAY, detect_topic, district_mentions, normalize_district, topic_scores

CATEGORIES = {"manifesto", "district_profile", "candidate_profile", "scheme", "policy", "faq", "other"}

_CATEGORY_HINTS = [
    ("manifesto", "manifesto"),
    ("faq", "faq"),
    ("candidate", "candidate_profile"),
    ("profile", "candidate_profile"),
    ("scheme", "scheme"),
    ("policy", "policy"),
    ("plan", "policy"),
]


def _first(*values: str | None) -> str | None:
    for v in values:
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _auto_district(text: str) -> str:
    counts = district_mentions(text)
    total = sum(counts.values())
    if not total:
        return "statewide"
    top, n = counts.most_common(1)[0]
    return top if n >= 3 and n / total >= 0.6 else "statewide"


def _auto_category(filename: str, district: str) -> str:
    stem = PurePath(filename).stem.lower()
    # A district-specific "plan"/"district" file is a district profile, not a statewide policy;
    # checked before the generic hints, where "plan" would otherwise map to policy.
    if district != "statewide" and ("district" in stem or "plan" in stem):
        return "district_profile"
    for hint, cat in _CATEGORY_HINTS:
        if hint in stem:
            return cat
    if district != "statewide":
        return "district_profile"
    return "other"


def _auto_topic(text: str) -> str:
    scores = topic_scores(text)
    total = sum(scores.values())
    if not total:
        return "general"
    topic, n = scores.most_common(1)[0]
    return topic if n / total >= 0.4 else "general"


def resolve_document_metadata(
    parsed: ParsedDocument,
    filename: str,
    overrides: Mapping[str, str | None] | None = None,
    manifest_entry: Mapping[str, str | None] | None = None,
) -> DocumentMetadata:
    o = overrides or {}
    fm = parsed.front_matter
    mf = manifest_entry or {}
    text = parsed.full_text

    district = normalize_district(_first(o.get("district"), fm.get("district"), mf.get("district"))) or _auto_district(text)
    category = _first(o.get("category"), fm.get("category"), mf.get("category")) or _auto_category(filename, district)
    category = category.lower() if category.lower() in CATEGORIES else "other"
    topic = (_first(o.get("topic"), fm.get("topic"), mf.get("topic")) or _auto_topic(text)).lower()
    source = _first(o.get("source"), fm.get("source"), mf.get("source")) or PurePath(filename).name
    candidate = _first(o.get("candidate"), fm.get("candidate"), mf.get("candidate"))
    return DocumentMetadata(district=district, category=category, topic=topic, source=source, candidate=candidate)


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chunk_district(doc_district: str, raw: RawChunk) -> tuple[str, list[str]]:
    section_text = " ".join(raw.section_path)
    mentioned = district_mentions(section_text + " " + raw.text)
    mentioned_list = [d for d, _ in mentioned.most_common()]
    if doc_district != "statewide":
        return doc_district, sorted(set(mentioned_list) | {doc_district})
    in_heading = [d for d, _ in district_mentions(section_text).most_common()]
    if len(in_heading) == 1:
        return in_heading[0], mentioned_list
    if len(mentioned) == 1 and mentioned_list and mentioned[mentioned_list[0]] >= 2:
        return mentioned_list[0], mentioned_list
    return "statewide", mentioned_list


def _chunk_topic(doc_topic: str, raw: RawChunk) -> str:
    # Heading words are the strongest topic signal; fall back to body text, then the document.
    for heading in reversed(raw.section_path):
        t = detect_topic(heading)
        if t:
            return t
    return detect_topic(raw.text, min_hits=2) or doc_topic


def build_header(title: str, section: str, district: str, topic: str, category: str) -> str:
    """Contextual header prepended to the chunk *for embedding and BM25 only*.

    It restores context a chunk loses when cut from its document (which document,
    which section, which district) — e.g. a "Hospitals" paragraph from the
    Guntur profile becomes distinguishable from the Vijayawada one.
    """
    crumb = f"{title} › {section}" if section else title
    return f"{crumb}\n[district: {DISTRICT_DISPLAY.get(district, district.title())} | topic: {topic} | category: {category}]"


def enrich_chunks(
    raws: list[RawChunk],
    parsed: ParsedDocument,
    doc_meta: DocumentMetadata,
    *,
    document_id: str,
    document_name: str,
    strategy: str,
    uploaded_at: str,
    campaign_id: str = "default",
    embedding_model: str = "",
    count_tokens: Callable[[str], int] | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for raw in raws:
        text = raw.text.strip()
        if not text:
            continue
        district, mentioned = _chunk_district(doc_meta.district, raw)
        topic = _chunk_topic(doc_meta.topic, raw)
        section = raw.section
        meta = ChunkMetadata(
            campaign_id=campaign_id,
            document_id=document_id,
            document_name=document_name,
            document_title=parsed.title,
            file_type=parsed.file_type,
            chunk_id=f"{document_id}:{len(chunks):04d}",
            chunk_index=len(chunks),
            page=raw.page,
            page_end=raw.page_end,
            section=section,
            district=district,
            districts_mentioned=mentioned,
            category=doc_meta.category,
            topic=topic,
            source=doc_meta.source,
            candidate=doc_meta.candidate,
            content_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            chunk_strategy=strategy,
            uploaded_at=uploaded_at,
            word_count=len(text.split()),
            token_count=count_tokens(text) if count_tokens else len(text.split()),
            content_type=raw.content_type,
            embedding_model=embedding_model,
        )
        header = build_header(parsed.title, section, district, topic, doc_meta.category)
        chunks.append(Chunk(text=text, embed_text=f"{header}\n{text}", metadata=meta))
    return chunks


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", PurePath(name).stem.lower()).strip("_")[:40] or "doc"
