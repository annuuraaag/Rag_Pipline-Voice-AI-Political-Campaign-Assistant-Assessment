"""Cheap, deterministic query understanding (no LLM, ~0.1 ms).

Extracts what retrieval needs from an utterance: district, topic, named entity
(e.g. a scheme name), whether it is a follow-up, and a cleaned version with
voice disfluencies removed. Being LLM-free matters: it runs on every partial
transcript and must never add network latency or fail.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.lexicon import DISTRICT_ALIASES, detect_districts, detect_topic

_FILLERS = re.compile(
    r"(^\s*so\b|\b(uh+|um+|erm|hmm+|you know|i mean|like,|basically|actually|okay so|ok so|guys)\b)[,]?\s*",
    re.IGNORECASE
)
_FOLLOW_UP_PREFIX = re.compile(r"^\s*(and|also|what about|how about|and what about|what of|same for)\b", re.IGNORECASE)
_PRONOUN = re.compile(r"\b(it|its|that|this|those|these|they|them|there|that scheme|this scheme)\b", re.IGNORECASE)
_SELF_INTRO = re.compile(r"\b(i'?m from|i am from|i live in|i stay in|my district is)\b", re.IGNORECASE)
# Capitalised multi-word names ("Pratibha Scholarship", "SCA Health Shield"), not sentence-initial words.
_ENTITY = re.compile(r"(?<!^)(?<![.?!]\s)\b((?:[A-Z][A-Za-z]+|SCA)(?:\s+(?:[A-Z][A-Za-z]+|of|and))*\s+[A-Z][A-Za-z]+)")
_ENTITY_STOP = {"Andhra Pradesh", "Sunrise Coast Alliance", "Vijayawada Central"}
# Whole-utterance small talk: no information need, so no retrieval and no LLM call.
_SMALLTALK = [
    ("greeting", re.compile(r"^(hi|hello|hey|namaste|namaskaram|good (morning|afternoon|evening))( there)?$")),
    ("affirm", re.compile(r"^(yes|yeah|yep|yup|ok|okay|sure|alright|fine|go ahead|please|yes please|tell me|hmm+)$")),
    ("thanks", re.compile(r"^(thanks|thank you|thank you so much|thanks a lot|great|nice|cool|got it)$")),
    ("bye", re.compile(r"^(bye|goodbye|see you|that'?s all|nothing else|no thanks|no)$")),
]


@dataclass
class Understanding:
    original: str
    cleaned: str
    districts: list[str] = field(default_factory=list)
    topic: str | None = None
    entity: str | None = None
    is_follow_up: bool = False
    is_self_intro: bool = False
    intro_only: bool = False   # "I'm from Vijayawada." – nothing to retrieve, just remember the district
    smalltalk: str | None = None  # greeting | affirm | thanks | bye
    content_words: int = 0


def clean_utterance(text: str) -> str:
    t = text
    for _ in range(3):  # fillers can be adjacent ("uh so ..."); repeat until stable
        t2 = _FILLERS.sub("", t)
        if t2 == t:
            break
        t = t2
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" ,")
    return t[:1].upper() + t[1:] if t else t


def _smalltalk(text: str) -> str | None:
    t = re.sub(r"[^a-z' ]+", " ", text.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    for kind, pat in _SMALLTALK:
        if pat.match(t):
            return kind
    return None


def _entity(text: str) -> str | None:
    for m in _ENTITY.finditer(text):
        name = m.group(1).strip()
        if name not in _ENTITY_STOP and not detect_districts(name):
            return name
    return None


_STOPWORDS = set("a an the of to in on for and or is are what which who how when where do does i me my you your it "
                 "about any there this that be will can tell please".split())


def understand(text: str) -> Understanding:
    cleaned = clean_utterance(text)
    words = re.findall(r"[a-zA-Z0-9']+", cleaned.lower())
    content = [w for w in words if w not in _STOPWORDS]
    districts = detect_districts(cleaned)
    topic = detect_topic(cleaned)
    follow = bool(_FOLLOW_UP_PREFIX.match(cleaned)) or (
        bool(_PRONOUN.search(cleaned)) and not districts and _entity(cleaned) is None
    )
    is_intro = bool(_SELF_INTRO.search(cleaned))
    rest = _SELF_INTRO.sub(" ", cleaned.lower())
    for d in districts:
        for alias in DISTRICT_ALIASES[d]:
            rest = rest.replace(alias, " ")
    leftover = [w for w in re.findall(r"[a-z0-9']+", rest) if w not in _STOPWORDS and w not in {"and", "so", "hi", "hello"}]
    return Understanding(
        original=text,
        cleaned=cleaned,
        districts=districts,
        topic=topic,
        entity=_entity(cleaned),
        is_follow_up=follow,
        is_self_intro=is_intro,
        intro_only=is_intro and bool(districts) and not leftover,
        smalltalk=_smalltalk(cleaned),
        content_words=len(content),
    )
