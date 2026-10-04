"""Query rewriting for retrieval.

The user's literal words are often not a good search query: "What about
education?" or "Who is eligible for it?" only make sense with conversation
state. The rewriter produces a standalone *retrieval* query plus metadata
filters; the original question is still what the LLM answers.

Two tiers:
* ``rewrite_with_rules`` – deterministic, microseconds, always runs. Resolves
  follow-ups, pronouns and the user's district from structured state.
* ``LLMRewriter`` – optional (QUERY_REWRITER=llm). Gets the rule output as a
  hint, has a hard timeout, and any failure/timeout/odd output falls back to the
  rule result. Rewriting can therefore never block or break an answer.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from app.conversation.state import ConversationState
from app.conversation.understanding import Understanding, understand
from app.domain import MetadataFilter
from app.lexicon import DISTRICT_DISPLAY, canonicalize_place_names

logger = logging.getLogger(__name__)

_FOLLOW_UP_PREFIX = re.compile(r"^\s*(and also|and what about|what about|how about|and|also|what of|same for)\s+",
                               re.IGNORECASE)
_THERE = re.compile(r"\bthere\b", re.IGNORECASE)
_IT = re.compile(r"\b(it|that scheme|this scheme|that|this)\b(?!\s+(?:is|was)\s+the\b)", re.IGNORECASE)


@dataclass
class Rewrite:
    original: str
    query: str
    filters: MetadataFilter
    reasons: list[str] = field(default_factory=list)
    understanding: Understanding | None = None
    method: str = "rules"

    @property
    def changed(self) -> bool:
        return self.query.strip() != self.original.strip()


def rewrite_with_rules(text: str, state: ConversationState | None) -> Rewrite:
    u = understand(text)
    q = u.cleaned or text.strip()
    reasons: list[str] = []
    if q != text.strip():
        reasons.append("removed filler words")
    canonical = canonicalize_place_names(q)
    if canonical != q:
        q = canonical
        reasons.append("added canonical place name")

    # One district in the utterance → filter on it; several ("Guntur and Vizag") → no district
    # filter, so both sides can be retrieved; none → fall back to the remembered district.
    if len(u.districts) > 1:
        district = None
    else:
        district = u.districts[0] if u.districts else (state.district if state else None)

    if state and u.is_follow_up:
        m = _FOLLOW_UP_PREFIX.match(q)
        if m:
            subject = q[m.end():].rstrip(" ?.")
            # "What about education?" → reuse the previous intent with the new subject.
            q = f"What are the {subject} plans and schemes"
            reasons.append(f"follow-up: expanded '{m.group(0).strip()} {subject}'")
        if state.entity and _IT.search(q) and not u.entity:
            q = _IT.sub(state.entity, q, count=1)
            reasons.append(f"resolved pronoun → '{state.entity}'")
        if state.topic and not u.topic and not m and not state.entity:
            q = f"{q} ({state.topic})"
            reasons.append(f"carried topic '{state.topic}'")
    if state and state.district and _THERE.search(q) and not u.districts:
        q = _THERE.sub(f"in {DISTRICT_DISPLAY.get(state.district, state.district)}", q, count=1)
        reasons.append("resolved 'there' → district")

    # Add the remembered district to the *text* only for recognised campaign topics/entities.
    # Appending a place name to an off-topic question ("lunar mining in Vijayawada") makes it
    # look similar to district chunks and can sneak it past the relevance gate. The district
    # *filter* below still scopes retrieval either way.
    on_topic = bool(u.topic or u.entity or (state and u.is_follow_up and (state.topic or state.entity)))
    if (district and on_topic and district not in u.districts
            and DISTRICT_DISPLAY.get(district, district).lower() not in q.lower()):
        q = f"{q.rstrip(' ?.')} in {DISTRICT_DISPLAY.get(district, district)}?"
        reasons.append(f"added district from conversation: {district}")

    filters = MetadataFilter(district=district) if district else MetadataFilter()
    if district:
        reasons.append(f"district filter: {district} (+ statewide)")
    return Rewrite(original=text, query=q, filters=filters, reasons=reasons, understanding=u)


REWRITE_PROMPT = """Rewrite the user's latest message into ONE standalone search query for a campaign-document search engine.
Resolve pronouns and follow-ups using the conversation. Keep names, places and scheme names exactly. Do not answer.
Output only the query on one line.

Known context: district={district}; topic={topic}; last entity={entity}
Recent conversation:
{history}
Latest message: {message}
Rule-based draft (may be imperfect): {draft}
Standalone query:"""


class LLMRewriter:
    def __init__(self, llm, timeout_s: float = 1.2):
        self.llm = llm
        self.timeout_s = timeout_s

    async def rewrite(self, text: str, state: ConversationState | None) -> Rewrite:
        base = rewrite_with_rules(text, state)
        if not state or not state.turns or getattr(self.llm, "is_fallback", False):
            return base  # nothing to resolve, or no real LLM available
        history = "\n".join(f"{t.role}: {t.text[:200]}" for t in list(state.turns)[-4:])
        prompt = REWRITE_PROMPT.format(district=state.district, topic=state.topic, entity=state.entity,
                                       history=history, message=text, draft=base.query)
        try:
            out = await asyncio.wait_for(self.llm.complete([{"role": "user", "content": prompt}], max_tokens=60),
                                         timeout=self.timeout_s)
        except Exception as exc:  # timeout, provider error: keep the rule rewrite
            logger.info("LLM rewrite fell back to rules: %s", exc.__class__.__name__)
            base.reasons.append("llm rewrite failed → rules")
            return base
        line = (out or "").strip().splitlines()[0].strip().strip('"') if out and out.strip() else ""
        if not (3 <= len(line) <= 300):
            base.reasons.append("llm rewrite rejected → rules")
            return base
        return Rewrite(original=text, query=line, filters=base.filters, reasons=base.reasons + ["llm rewrite"],
                       understanding=base.understanding, method="llm")
