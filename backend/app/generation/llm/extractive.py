"""Offline extractive fallback "LLM".

Used when no API key is configured or the provider fails. It selects the source
sentences that best overlap the question and returns them verbatim with
citations: lower fluency, but grounded by construction and zero latency/cost.
Responses are flagged ``is_fallback`` so the UI and API never pass it off as an LLM.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

from app.generation.llm.base import Grounding, LLMError
from app.generation.prompt import REFUSAL_TEXT
from app.ingestion.chunker import split_sentences

_STOP = set(
    "a an the of to in on for and or is are was were be been what which who whom how when where why do does did "
    "i im i'm me my we our you your it its this that these those there their they them with about from by at as "
    "any all available tell please can could would should will shall give know want like more some much many".split()
)


def _terms(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9₹]+", text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        out.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)
    return out


class ExtractiveLLM:
    name = "extractive"
    model = "extractive-fallback"
    is_fallback = True

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def answer(self, grounding: Grounding | None) -> str:
        if grounding is None:
            raise LLMError("extractive fallback needs structured grounding (question + sources)")
        q_terms = _terms(grounding.question)
        scored: list[tuple[float, int, int, str]] = []
        for rank, src in enumerate(grounding.sources):
            num, text = rank + 1, src.chunk.text
            # Bullets are separate lines; split them before sentence segmentation.
            sentences = [x for line in text.splitlines() for x in split_sentences(line.lstrip("• "))]
            for pos, sent in enumerate(sentences):
                overlap = len(q_terms & _terms(sent))
                if overlap:
                    scored.append((overlap / (1 + 0.15 * rank), num, pos, sent.strip()))
        if not scored:
            return REFUSAL_TEXT
        # Prefer the best sentence from each source first (multi-document coverage), then fill by score.
        ranked = sorted(scored, key=lambda x: -x[0])
        best, seen = [], set()
        for item in ranked:
            if item[1] not in seen:
                best.append(item)
                seen.add(item[1])
        best = (best + [x for x in ranked if x not in best])[: self.max_sentences]
        best.sort(key=lambda x: (x[1], x[2]))  # keep document order for readability
        return "According to the campaign documents: " + " ".join(f"{s} [S{n}]" for _, n, _, s in best)

    async def stream(self, messages: list[dict[str, str]], max_tokens: int | None = None,
                     grounding: Grounding | None = None) -> AsyncIterator[str]:
        for word in re.split(r"(\s+)", self.answer(grounding)):
            if word:
                yield word
                await asyncio.sleep(0)

    async def complete(self, messages: list[dict[str, str]], max_tokens: int | None = None,
                       grounding: Grounding | None = None) -> str:
        return self.answer(grounding)
