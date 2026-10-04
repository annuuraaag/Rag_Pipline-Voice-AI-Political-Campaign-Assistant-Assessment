"""Text normalisation applied to every parsed block before chunking."""
from __future__ import annotations

import re
import unicodedata

_DOT_LEADER = re.compile(r"(\.\s?){5,}")
_MULTI_SPACE = re.compile(r"[ \t ]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s+)?\d+(\s*(/|of)\s*\d+)?\s*$", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Unicode-normalise, re-join hyphenated line breaks, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("‐", "-").replace("‑", "-")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def is_noise_line(line: str) -> bool:
    """Lines that carry no retrievable meaning: bare page numbers, TOC dot leaders."""
    stripped = line.strip()
    if not stripped:
        return True
    if _PAGE_NUMBER_LINE.match(stripped):
        return True
    if _DOT_LEADER.search(stripped):
        return True
    return False


# Boilerplate repeated across a corpus (disclaimers, "published by" notices, legal footers)
# matches many generic questions but answers none, so it is stripped before chunking.
# Configurable via BOILERPLATE_PATTERNS (e.g. a "Published and printed by …" notice); patterns
# should be specific, since a loose one deletes real content. The default covers only this repo's
# fictional-data disclaimer.
DEFAULT_BOILERPLATE = [
    r"FICTIONAL SAMPLE DOCUMENT\..*?\b(?:invented|coincidental)\b[^.]*\.",
]


def strip_boilerplate(blocks: list, patterns: list[re.Pattern[str]]) -> list:
    """Remove boilerplate spans from parsed blocks; drop blocks left empty."""
    if not patterns:
        return blocks
    out = []
    for b in blocks:
        text = b.text
        for p in patterns:
            text = p.sub(" ", text)
        text = normalize_text(text)
        if text:
            b.text = text
            out.append(b)
    return out
