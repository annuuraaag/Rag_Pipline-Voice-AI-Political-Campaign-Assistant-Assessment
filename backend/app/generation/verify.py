"""Claim-level citation verification.

``finalize_answer`` guarantees that every ``[S#]`` marker points at a passage that was really
sent to the model. That says nothing about whether the passage *says* what the sentence claims.
This module checks every sentence of an answer against the passages it cites:

  1. numbers  every figure in the sentence appears in a cited passage ("15 lakh", "2,400", "2026")
  2. places   every district it names is one the passages cover: their district, a district they
              mention, or (only when every passage used is one) a statewide passage naming no district
  3. names    capitalised names (schemes, hospitals, programmes) appear in the passage or its header
  4. wording  most of the sentence's content words appear in the passage; when a cross-encoder is
              loaded, it accepts paraphrases the word check would miss

Outcome per sentence:
  supported    its cited passages pass
  corrected    they fail, but another passage in the context passes: the citation is moved there
               (also used for a factual sentence the model left uncited)
  unsupported  no passage in the context supports it
  no_claim     nothing to check ("The documents don't say when.", pure restatement of the question)

Everything here is deterministic and runs in well under a millisecond per sentence without the
cross-encoder, so it can also run per sentence while an answer streams (before it is spoken).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from app.domain import ScoredChunk
from app.generation.prompt import REFUSAL_TEXT
from app.lexicon import DISTRICT_ALIASES, DISTRICT_DISPLAY, detect_districts
from app.retrieval.reranker import Reranker

Verdict = Literal["supported", "corrected", "unsupported", "no_claim"]
Policy = Literal["off", "flag", "strict"]

_MARKERS = re.compile(r"\s*\[S(\d+)\]")
_MARKER_RUN = re.compile(r"((?:\s*\[S\d+\])+)")
# A sentence ends at . ! ? followed by space + a capital/digit/quote, or at a line break.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(₹])|\n+")
_ABBREV = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof|Rs|No|St|vs|approx|Govt|e\.g|i\.e)\.$", re.IGNORECASE)
_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)(?!\w)")
_NUM_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
              "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15", "twenty": "20",
              "thirty": "30", "forty": "40", "fifty": "50", "hundred": "100", "thousand": "1000"}
_WORD = re.compile(r"[a-z][a-z'-]+")
_CAPS = re.compile(r"\b([A-Z][a-zA-Z]{2,}|[A-Z]{2,})\b")
_META = re.compile(
    r"(couldn'?t find|could not find|no information|not (?:covered|mentioned|specified|stated|available)|"
    r"(?:do|does|did)(?: not|n'?t) (?:mention|say|specify|state|include|cover|provide|give|list|describe)|"
    r"there is no (?:mention|detail)|isn'?t (?:covered|mentioned)|not in the (?:campaign )?documents)",
    re.IGNORECASE)

# Words that frame an answer rather than state a fact ("The manifesto proposes ...").
_FRAMING = {
    "according", "campaign", "document", "documents", "source", "sources", "manifesto", "alliance", "sunrise",
    "coast", "sca", "candidate", "propose", "proposes", "proposed", "proposal", "plan", "plans", "planned",
    "promise", "promises", "promised", "commit", "commits", "committed", "mention", "mentions", "mentioned",
    "state", "states", "stated", "say", "says", "said", "describe", "describes", "outline", "outlines", "note",
    "notes", "also", "include", "includes", "including", "aim", "aims", "intend", "intends", "would", "will",
    "there", "these", "this", "that", "those", "which", "with", "from", "into", "their", "they", "them", "about",
    "under", "such", "other", "each", "every", "more", "most", "some", "than", "then", "been", "being", "have",
    "has", "had", "are", "was", "were", "for", "and", "the", "its", "can", "could", "should", "may", "might",
    "both", "what", "when", "where", "who", "how", "per", "any", "all", "not", "only", "upto", "up", "over",
    "well", "within", "among", "while", "after", "before", "here", "your", "you", "our", "his", "her",
    "provides", "provide", "provided", "offer", "offers", "offered", "receive", "receives", "get", "gets",
}
_GENERIC_NAMES = {"the", "this", "these", "those", "that", "according", "however", "also", "yes", "alliance",
                  "sunrise", "coast", "campaign", "manifesto", "candidate", "government", "state", "district",
                  "andhra", "pradesh", "india", "indian", "sca", "am", "pm", "rs", "for", "in", "it", "its",
                  "statewide", "and", "under", "each", "every", "all", "both", "there", "they", "their"}

_PLACE_WORDS = {w for aliases in DISTRICT_ALIASES.values() for a in aliases for w in a.split()}  # checked as places


# ── text helpers ─────────────────────────────────────────────────────────
def _stem(w: str) -> str:
    for suf in ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ied", "ed", "es", "s", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if len(w) >= 5 and w[-1] in "ey":
        w = w[:-1]
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]
    return w


def terms(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _FRAMING}


def numbers(text: str, words: bool = False) -> set[str]:
    out = {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}
    out |= {n.split(".")[0] for n in out if n.endswith(".0")}
    if words:
        out |= {v for k, v in _NUM_WORDS.items() if re.search(rf"\b{k}\b", text, re.IGNORECASE)}
    return out


def names(sentence: str) -> set[str]:
    """Capitalised words that are not the first word: likely names of schemes, places, institutions."""
    body = sentence.lstrip("\"“'( ")
    first = re.match(r"\S+", body)
    rest = body[first.end():] if first else body
    return {w.lower() for w in _CAPS.findall(rest) if w.lower() not in _GENERIC_NAMES and w.lower() not in _PLACE_WORDS}


def split_claims(answer: str) -> list[tuple[str, list[int]]]:
    """[(sentence without markers, [S#] numbers)]. Markers written after the full stop
    ("... planned. [S1] Next ...") belong to the sentence before them."""
    text = re.sub(r"([.!?])((?:\s*\[S\d+\])+)", lambda m: m.group(2) + m.group(1), answer)
    parts, buf = [], ""
    for piece in _BOUNDARY.split(text):
        buf = f"{buf} {piece}".strip() if buf else piece.strip()
        if buf and not _ABBREV.search(_MARKERS.sub("", buf)):
            parts.append(buf)
            buf = ""
    if buf:
        parts.append(buf)
    out = []
    for p in parts:
        cited = [int(n) for n in _MARKERS.findall(p)]
        sentence = re.sub(r"\s+([.,;:!?])", r"\1", _MARKERS.sub("", p)).strip(" -•*")
        if sentence:
            out.append((sentence, list(dict.fromkeys(cited))))
    return out


# ── results ──────────────────────────────────────────────────────────────
class ClaimCheck(BaseModel):
    index: int
    text: str                                         # the sentence, markers removed
    cited: list[int] = Field(default_factory=list)    # [S#] the model wrote
    sources: list[int] = Field(default_factory=list)  # [S#] that support it (after correction)
    verdict: Verdict
    support: float = 0.0                              # share of content words found (or cross-encoder score)
    issues: list[str] = Field(default_factory=list)


class Verification(BaseModel):
    policy: str
    method: str
    claims: list[ClaimCheck] = Field(default_factory=list)
    supported: int = 0
    corrected: int = 0
    unsupported: int = 0
    removed: int = 0                                  # strict policy: unsupported sentences dropped
    latency_ms: float = 0.0

    @property
    def checked(self) -> int:
        return self.supported + self.corrected + self.unsupported


@dataclass
class _Evidence:
    """One passage, pre-processed once per answer."""
    number: int
    text: str
    terms: set[str]
    numbers: set[str]
    words: set[str]
    districts: set[str]
    statewide: bool
    names: set[str] = field(default_factory=set)


def _evidence(i: int, c: ScoredChunk) -> _Evidence:
    m = c.chunk.metadata
    full = f"{c.chunk.embed_text}\n{m.document_title}\n{m.section}"
    low = full.lower()
    mentioned = set(m.districts_mentioned) | set(detect_districts(full))
    if m.district and m.district != "statewide":
        mentioned.add(m.district)
    words = {w.removesuffix("'s") for w in _WORD.findall(low)}   # "People's Manifesto" → people
    return _Evidence(i, c.chunk.text, terms(full), numbers(full, words=True), words, mentioned, m.district == "statewide")


# ── the verifier ─────────────────────────────────────────────────────────
class ClaimVerifier:
    def __init__(self, reranker: Reranker | None = None, min_coverage: float = 0.5,
                 semantic_threshold: float = 0.5, semantic_floor: float = 0.3):
        self.reranker = reranker
        self.min_coverage = min_coverage
        self.semantic_threshold = semantic_threshold
        self.semantic_floor = semantic_floor

    @property
    def method(self) -> str:
        return "lexical + cross-encoder" if self.reranker else "lexical"

    def _against(self, sentence: str, claim_terms: set[str], ev: list[_Evidence], question_terms: set[str]
                 ) -> tuple[bool, float, list[str]]:
        """Does the union of these passages support the sentence? (ok, support, issues)"""
        issues = []
        pool_numbers = set().union(*(e.numbers for e in ev))
        missing = sorted(numbers(sentence) - pool_numbers)
        if missing:
            issues.append(f"figure{'s' if len(missing) > 1 else ''} {', '.join(missing)} not in the source")
        places = set(detect_districts(sentence))
        covered = set().union(*(e.districts for e in ev))
        # A statewide passage that names no district applies everywhere, but only on its own: mixed
        # with a Vijayawada passage it must not lend "Guntur" to that passage's facts.
        blanket = all(e.statewide and not e.districts for e in ev)
        wrong = sorted(places - covered) if not blanket else []
        if wrong:
            issues.append(f"{', '.join(DISTRICT_DISPLAY.get(d, d) for d in wrong)} not covered by the source")
        pool_words = set().union(*(e.words for e in ev))
        unknown = sorted(n for n in names(sentence) if n not in pool_words and _stem(n) not in
                         set().union(*(e.terms for e in ev)))
        if unknown:
            issues.append(f"'{', '.join(unknown)}' not in the source")
        content = claim_terms - (question_terms - set().union(*(e.terms for e in ev)))
        if not content:
            return not issues, 1.0 if not issues else 0.0, issues
        pool_terms = set().union(*(e.terms for e in ev))
        coverage = len(content & pool_terms) / len(content)
        if issues:
            return False, round(coverage, 3), issues
        if coverage >= self.min_coverage:
            return True, round(coverage, 3), issues
        if self.reranker is not None and coverage >= self.semantic_floor:
            score = max(self.reranker.score(sentence, [e.text for e in ev]), default=0.0)
            if score >= self.semantic_threshold:
                return True, round(score, 3), issues
        issues.append(f"only {coverage:.0%} of its content words are in the source")
        return False, round(coverage, 3), issues

    def check(self, index: int, sentence: str, cited: list[int], evidence: list[_Evidence],
              question_terms: set[str]) -> ClaimCheck:
        claim_terms = terms(sentence)
        factual = bool(numbers(sentence) or detect_districts(sentence) or names(sentence) or
                       (claim_terms - question_terms))
        if _META.search(sentence) or sentence.strip() == REFUSAL_TEXT or not factual:
            return ClaimCheck(index=index, text=sentence, cited=cited, sources=cited, verdict="no_claim", support=1.0)
        by_num = {e.number: e for e in evidence}
        cited_ev = [by_num[n] for n in cited if n in by_num]
        issues: list[str] = []
        support = 0.0
        if cited_ev:
            ok, support, issues = self._against(sentence, claim_terms, cited_ev, question_terms)
            if ok:
                return ClaimCheck(index=index, text=sentence, cited=cited, sources=cited, verdict="supported",
                                  support=support)
        else:
            issues = ["no citation"]
        # Re-attribute: the best single passage, then the best pair.
        ranked = sorted(evidence, key=lambda e: -len(claim_terms & e.terms))
        groups = [[e] for e in ranked] + ([ranked[:2]] if len(ranked) > 1 else [])
        for group in groups:
            if [e.number for e in group] == cited:
                continue
            ok, score, _ = self._against(sentence, claim_terms, group, question_terms)
            if ok:
                return ClaimCheck(index=index, text=sentence, cited=cited, sources=[e.number for e in group],
                                  verdict="corrected", support=score, issues=issues)
        return ClaimCheck(index=index, text=sentence, cited=cited, sources=cited, verdict="unsupported",
                          support=support, issues=issues)

    def verify(self, answer: str, sources: list[ScoredChunk], question: str = "",
               policy: Policy = "flag") -> tuple[str, Verification]:
        """Check every sentence; returns (answer with corrected citations, report). With the strict
        policy, unsupported sentences are removed from the returned answer."""
        t = time.perf_counter()
        evidence = [_evidence(i, c) for i, c in enumerate(sources, start=1)]
        question_terms = terms(question)
        report = Verification(policy=policy, method=self.method)
        rebuilt, changed = [], False
        for i, (sentence, cited) in enumerate(split_claims(answer)):
            c = self.check(i, sentence, cited, evidence, question_terms)
            report.claims.append(c)
            if c.verdict in ("supported", "corrected", "unsupported"):
                setattr(report, c.verdict, getattr(report, c.verdict) + 1)
            if c.verdict == "unsupported" and policy == "strict":
                report.removed += 1
                changed = True
                continue
            if c.sources != c.cited:
                changed = True
            rebuilt.append(_with_markers(sentence, c.sources))
        report.latency_ms = round((time.perf_counter() - t) * 1000, 3)
        if not changed:
            return answer, report
        if policy == "strict" and report.removed and not (report.supported or report.corrected):
            return REFUSAL_TEXT, report
        return " ".join(rebuilt), report


def _with_markers(sentence: str, sources: list[int]) -> str:
    if not sources:
        return sentence
    marks = "".join(f"[S{n}]" for n in sources)
    if sentence[-1:] in ".!?":
        return f"{sentence[:-1].rstrip()} {marks}{sentence[-1]}"
    return f"{sentence} {marks}"


def citation_status(report: Verification | None, number: int) -> bool | None:
    """Per source: True if every claim citing it is supported by it, False if one is not."""
    if report is None:
        return None
    verdicts = [c.verdict for c in report.claims if number in c.sources and c.verdict != "no_claim"]
    if not verdicts:
        return None
    return all(v in ("supported", "corrected") for v in verdicts)
