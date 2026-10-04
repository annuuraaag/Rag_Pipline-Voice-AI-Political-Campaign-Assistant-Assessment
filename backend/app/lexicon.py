"""Domain lexicons: district gazetteer and topic keywords.

Used at ingestion (metadata enrichment) and at query time (cheap, LLM-free query
understanding). Deterministic lookups cost microseconds, which matters for the
partial-transcript path where we cannot afford an LLM call per fragment.
"""
from __future__ import annotations

import re
from collections import Counter

# canonical slug → aliases (lowercase). Includes common spellings and old names
# because voice transcripts are noisy ("Vizag", "Bezawada").
DISTRICT_ALIASES: dict[str, list[str]] = {
    "vijayawada": ["vijayawada", "vijaywada", "bezawada", "ntr district"],
    "guntur": ["guntur"],
    "visakhapatnam": ["visakhapatnam", "vishakhapatnam", "visakhapatanam", "vizag", "visakha", "waltair"],
    "krishna": ["krishna district", "machilipatnam"],
    "tirupati": ["tirupati", "tirupathi"],
    "nellore": ["nellore"],
    "kurnool": ["kurnool"],
    "kakinada": ["kakinada"],
    "rajahmundry": ["rajahmundry", "rajamahendravaram"],
    "anantapur": ["anantapur", "anantapuramu"],
    "kadapa": ["kadapa", "cuddapah"],
    "srikakulam": ["srikakulam"],
    "vizianagaram": ["vizianagaram"],
    "prakasam": ["prakasam", "ongole"],
    "eluru": ["eluru"],
}

DISTRICT_DISPLAY = {slug: slug.title() for slug in DISTRICT_ALIASES} | {"statewide": "Statewide"}

# Spelling variants and old names of the *same* place (not nearby towns such as Machilipatnam).
# The rewriter adds the canonical name next to them: documents and the cross-encoder know
# "Vijayawada", not "bezawada".
NAME_VARIANTS: dict[str, str] = {
    "vijaywada": "vijayawada", "bezawada": "vijayawada",
    "vishakhapatnam": "visakhapatnam", "visakhapatanam": "visakhapatnam", "vizag": "visakhapatnam",
    "visakha": "visakhapatnam", "tirupathi": "tirupati", "rajamahendravaram": "rajahmundry",
    "anantapuramu": "anantapur", "cuddapah": "kadapa",
}
_VARIANT_RE = re.compile(r"\b(" + "|".join(sorted(NAME_VARIANTS, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def canonicalize_place_names(text: str) -> str:
    """'I live in bezawada' → 'I live in bezawada (Vijayawada)'. The original word is kept for BM25."""
    return _VARIANT_RE.sub(lambda m: f"{m.group(0)} ({DISTRICT_DISPLAY[NAME_VARIANTS[m.group(0).lower()]]})", text)

# Keywords may be multi-word phrases. Generic single words ("it", "cold", "power",
# "social", "office") are only used inside phrases: alone they appear in ordinary prose
# ("it is", "contact us") and would bias every chunk towards one topic.
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "healthcare": [
        "health", "healthcare", "hospital", "hospitals", "clinic", "clinics", "doctor", "doctors",
        "medical", "medicine", "medicines", "diagnostic", "diagnostics", "health insurance", "maternal",
        "nutrition", "phc", "phcs", "patients", "treatment", "ambulance", "dialysis", "cancer",
    ],
    "education": [
        "education", "school", "schools", "student", "students", "scholarship", "scholarships",
        "college", "colleges", "teacher", "teachers", "university", "classroom", "classrooms",
        "learning", "literacy", "exam", "exams", "fees",
    ],
    "employment": [
        "employment", "job", "jobs", "skill", "skills", "skilling", "youth", "startup", "startups",
        "msme", "msmes", "industry", "industries", "wage", "wages", "livelihood", "unemployment",
        "apprenticeship", "apprenticeships", "it sector", "it jobs", "it corridor", "it park",
        "it and electronics", "hiring", "placement",
    ],
    "agriculture": [
        "agriculture", "farmer", "farmers", "farming", "crop", "crops", "irrigation", "chilli",
        "cotton", "paddy", "msp", "harvest", "cold storage", "fishermen", "fisheries", "aqua",
        "tobacco", "rythu", "seeds", "fertiliser", "fertilizer",
    ],
    "infrastructure": [
        "infrastructure", "road", "roads", "water", "drinking water", "drainage", "flood", "floods",
        "metro", "transport", "bus", "buses", "electricity", "power supply", "power cuts", "canal", "port", "bridge",
        "sanitation", "sewage", "housing", "urban",
    ],
    "welfare": [
        "welfare", "women", "pension", "pensions", "elderly", "disability", "disabled", "widow",
        "self-help", "shg", "shgs", "social security", "social welfare", "ration", "girls", "safety",
    ],
    "governance": [
        "governance", "corruption", "transparency", "grievance", "grievances", "accountability",
        "e-governance", "citizen", "public services", "government services", "audit", "rti",
    ],
    "campaign_info": [
        "vote", "voting", "polling", "booth", "voter registration", "register to vote", "registered to vote",
        "volunteer", "volunteers", "rally", "campaign office", "contact number", "helpline", "election", "manifesto",
    ],
}


def _topic_pattern(keywords: list[str]) -> re.Pattern[str]:
    # Longest first so a phrase wins over any keyword it contains; spaces match any whitespace.
    alts = sorted(keywords, key=len, reverse=True)
    body = "|".join(re.escape(k).replace(r"\ ", r"\s+") for k in alts)
    return re.compile(rf"(?<![a-z0-9-])(?:{body})(?![a-z0-9-])")


_TOPIC_PATTERNS = {topic: _topic_pattern(kws) for topic, kws in TOPIC_KEYWORDS.items()}


def _alias_patterns() -> list[tuple[re.Pattern[str], str]]:
    pats = []
    for slug, aliases in DISTRICT_ALIASES.items():
        for alias in aliases:
            pats.append((re.compile(rf"\b{re.escape(alias)}\b"), slug))
    return pats


_DISTRICT_PATTERNS = _alias_patterns()


def district_mentions(text: str) -> Counter[str]:
    """Count canonical district mentions in `text`."""
    low = text.lower()
    counts: Counter[str] = Counter()
    for pat, slug in _DISTRICT_PATTERNS:
        n = len(pat.findall(low))
        if n:
            counts[slug] += n
    return counts


def detect_districts(text: str) -> list[str]:
    """Districts mentioned in `text`, most-mentioned first."""
    return [d for d, _ in district_mentions(text).most_common()]


def normalize_district(value: str | None) -> str | None:
    """Map user/form input ("Vizag", "NTR district") to a canonical slug."""
    if not value:
        return None
    v = value.strip().lower()
    if v in ("statewide", "state", "all", "andhra pradesh", "ap"):
        return "statewide"
    if v in DISTRICT_ALIASES:
        return v
    found = detect_districts(v)
    return found[0] if found else v


def topic_scores(text: str) -> Counter[str]:
    """Keyword/phrase hits per topic (whole-word, case-insensitive)."""
    low = text.lower()
    scores: Counter[str] = Counter()
    for topic, pat in _TOPIC_PATTERNS.items():
        n = len(pat.findall(low))
        if n:
            scores[topic] = n
    return scores


def detect_topic(text: str, min_hits: int = 1) -> str | None:
    """Dominant topic of `text`, or None if nothing matches."""
    scores = topic_scores(text)
    if not scores:
        return None
    topic, hits = scores.most_common(1)[0]
    return topic if hits >= min_hits else None
