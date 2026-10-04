"""LLM provider interface. Everything the RAG service needs: stream tokens, or complete."""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from app.domain import ScoredChunk


class LLMError(RuntimeError):
    """Provider failure (network, auth, rate limit, timeout). Callers fall back, never crash."""


@dataclass(frozen=True)
class Grounding:
    """The question and the numbered sources ([S1] = sources[0]) behind a prompt.

    Passed alongside the messages so providers that work on structure (the extractive
    fallback) never have to re-parse prompt text. Chat LLMs ignore it.
    """

    question: str
    sources: list[ScoredChunk]


class LLMProvider(Protocol):
    name: str
    model: str
    is_fallback: bool

    def stream(self, messages: list[dict[str, str]], max_tokens: int | None = None,
               grounding: Grounding | None = None) -> AsyncIterator[str]: ...

    async def complete(self, messages: list[dict[str, str]], max_tokens: int | None = None,
                       grounding: Grounding | None = None) -> str: ...
