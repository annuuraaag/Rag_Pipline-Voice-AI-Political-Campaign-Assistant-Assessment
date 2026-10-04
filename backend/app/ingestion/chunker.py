"""Chunking strategies.

* ``SectionChunker`` (structure-aware): never crosses a heading boundary, packs whole
  sentences up to a token budget (counted with the embedding model's own tokenizer),
  keeps tables and OCR text in their own chunks, overlaps one sentence when a long section is split,
  and merges tiny sections into their neighbour. Each chunk keeps its heading path
  and page span, so it stays meaningful when retrieved alone and can be cited.
* ``FixedChunker`` (baseline): sliding word window with overlap, ignoring structure.
  Kept so the eval can measure what structure-awareness actually buys.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from app.domain import ParsedBlock, ParsedDocument

# Sentence boundary: terminal punctuation + space + an uppercase/digit/quote/currency start.
# Abbreviations common in this corpus are protected first.
_ABBREVIATIONS = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Rs.", "No.", "St.", "e.g.", "i.e.", "vs.", "approx.", "Govt."]
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(₹])")


def split_sentences(text: str) -> list[str]:
    protected = text
    for i, abbr in enumerate(_ABBREVIATIONS):
        protected = protected.replace(abbr, abbr.replace(".", f"<DOT{i}>"))
    parts = _SENT_BOUNDARY.split(protected)
    out = []
    for p in parts:
        for i, abbr in enumerate(_ABBREVIATIONS):
            p = p.replace(abbr.replace(".", f"<DOT{i}>"), abbr)
        p = p.strip()
        if p:
            out.append(p)
    return out


@dataclass
class RawChunk:
    text: str
    section_path: list[str]
    page: int | None = None
    page_end: int | None = None
    extra_sections: list[list[str]] = field(default_factory=list)
    content_type: str = "text"

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def section(self) -> str:
        label = " › ".join(self.section_path)
        if self.extra_sections:
            label += "; " + "; ".join(s[-1] for s in self.extra_sections if s)
        return label


class Chunker(Protocol):
    name: str

    def chunk(self, doc: ParsedDocument) -> list[RawChunk]: ...


Measure = Callable[[str], int]


def count_words(text: str) -> int:
    return len(text.split())


@dataclass
class _Unit:
    text: str
    page: int | None
    size: int            # in the chunker's unit (embedding-model tokens by default)
    content_type: str


def _units_from_blocks(blocks: list[ParsedBlock], measure: Measure) -> list[_Unit]:
    units: list[_Unit] = []
    for b in blocks:
        # List items and table rows are atomic; paragraphs are split into sentences.
        if b.kind == "list_item":
            pieces = [f"• {b.text}"]
        elif b.kind == "table_row":
            pieces = [b.text]
        else:
            pieces = split_sentences(b.text)
        for s in pieces:
            units.append(_Unit(text=s, page=b.page, size=measure(s), content_type=b.content_type))
    return units


def _join_units(units: list[_Unit]) -> str:
    out: list[str] = []
    for u in units:
        if (u.text.startswith("• ") or u.content_type == "table") and out:
            out.append("\n" + u.text)   # list items and table rows keep one per line
        else:
            out.append((" " if out else "") + u.text)
    return "".join(out).strip()


def _split_oversized(units: list[_Unit], max_size: int, measure: Measure) -> list[_Unit]:
    """Cut any single unit over the budget on word boundaries.

    Sentences, list items and table rows are normally atomic, but a run-on sentence, a
    giant table row or OCR text with no punctuation would otherwise become a chunk past
    the embedder's 512-token limit and be silently truncated when embedded.
    """
    out: list[_Unit] = []
    for u in units:
        if u.size <= max_size:
            out.append(u)
            continue
        piece: list[str] = []
        for word in u.text.split():
            if piece and measure(" ".join(piece + [word])) > max_size:
                text = " ".join(piece)
                out.append(_Unit(text, u.page, measure(text), u.content_type))
                piece = []
            piece.append(word)
        if piece:
            text = " ".join(piece)
            out.append(_Unit(text, u.page, measure(text), u.content_type))
    return out


def _pages(units: list[_Unit]) -> tuple[int | None, int | None]:
    pages = [u.page for u in units if u.page is not None]
    return (min(pages), max(pages)) if pages else (None, None)


def _runs(units: list[_Unit]) -> list[list[_Unit]]:
    """Consecutive units of one content type. Tables and OCR text never share a chunk with body text."""
    runs: list[list[_Unit]] = []
    for u in units:
        if runs and runs[-1][-1].content_type == u.content_type:
            runs[-1].append(u)
        else:
            runs.append([u])
    return runs


class SectionChunker:
    """Structure-aware chunking with a token budget measured by the embedding model's tokenizer."""

    name = "section"

    def __init__(self, max_size: int = 256, min_size: int = 32, measure: Measure = count_words):
        self.max_size = max_size
        self.min_size = min_size
        self.measure = measure

    def _sections(self, doc: ParsedDocument) -> list[tuple[list[str], list[ParsedBlock]]]:
        sections: list[tuple[list[str], list[ParsedBlock]]] = []
        for b in doc.blocks:
            if sections and sections[-1][0] == b.section_path:
                sections[-1][1].append(b)
            else:
                sections.append((list(b.section_path), [b]))
        return sections

    def _pack(self, path: list[str], units: list[_Unit]) -> list[RawChunk]:
        units = _split_oversized(units, self.max_size, self.measure)
        chunks: list[RawChunk] = []
        for run in _runs(units):
            ctype = run[0].content_type
            cur: list[_Unit] = []
            cur_size = 0
            for u in run:
                if cur and cur_size + u.size > self.max_size:
                    p0, p1 = _pages(cur)
                    chunks.append(RawChunk(_join_units(cur), path, p0, p1, content_type=ctype))
                    # One-sentence overlap preserves the thread across a split section
                    # (not for tables: a repeated row would be double-counted).
                    tail = cur[-1]
                    cur = [tail] if ctype != "table" and tail.size < self.max_size // 3 else []
                    cur_size = sum(x.size for x in cur)
                cur.append(u)
                cur_size += u.size
            if cur:
                p0, p1 = _pages(cur)
                chunks.append(RawChunk(_join_units(cur), path, p0, p1, content_type=ctype))
        return chunks

    def chunk(self, doc: ParsedDocument) -> list[RawChunk]:
        out: list[RawChunk] = []
        pending: RawChunk | None = None  # a tiny section waiting to merge forward
        for path, blocks in self._sections(doc):
            pieces = self._pack(path, _units_from_blocks(blocks, self.measure))
            if pending is not None and pieces:
                first = pieces[0]
                mergeable = (pending.section_path[:1] == first.section_path[:1]
                             and pending.content_type == first.content_type
                             and self.measure(pending.text) + self.measure(first.text) <= self.max_size)
                if mergeable:
                    pages = [p for p in (pending.page, first.page, pending.page_end, first.page_end) if p]
                    pieces[0] = RawChunk(
                        text=f"{pending.text}\n{first.text}",
                        section_path=pending.section_path,
                        page=min(pages) if pages else None,
                        page_end=max(pages) if pages else None,
                        extra_sections=pending.extra_sections + [first.section_path],
                        content_type=first.content_type,
                    )
                else:
                    out.append(pending)
                pending = None
            if len(pieces) == 1 and self.measure(pieces[0].text) < self.min_size:
                pending = pieces[0]
                continue
            out.extend(pieces)
        if pending is not None:
            out.append(pending)
        return out


class FixedChunker:
    name = "fixed"

    def __init__(self, max_words: int = 180, overlap_words: int = 30):
        if overlap_words >= max_words:
            raise ValueError("overlap_words must be smaller than max_words")
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk(self, doc: ParsedDocument) -> list[RawChunk]:
        words: list[tuple[str, int | None, list[str]]] = []
        for b in doc.blocks:
            for w in b.text.split():
                words.append((w, b.page, b.section_path))
        chunks: list[RawChunk] = []
        stride = self.max_words - self.overlap_words
        for start in range(0, len(words), stride):
            window = words[start : start + self.max_words]
            if not window:
                break
            pages = [p for _, p, _ in window if p is not None]
            chunks.append(
                RawChunk(
                    text=" ".join(w for w, _, _ in window),
                    section_path=list(window[0][2]),
                    page=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                )
            )
            if start + self.max_words >= len(words):
                break
        return chunks


def make_chunker(strategy: str, *, max_tokens: int = 256, min_tokens: int = 32, max_words: int = 180,
                 overlap_words: int = 30, measure: Measure = count_words) -> Chunker:
    """`section` uses a token budget (measured with the embedder's tokenizer); `fixed` stays a word window."""
    if strategy == "fixed":
        return FixedChunker(max_words=max_words, overlap_words=overlap_words)
    if strategy == "section":
        return SectionChunker(max_size=max_tokens, min_size=min_tokens, measure=measure)
    raise ValueError(f"Unknown chunk strategy: {strategy}")
