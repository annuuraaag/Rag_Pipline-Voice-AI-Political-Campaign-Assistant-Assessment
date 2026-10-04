"""Server-side speech-to-text for browsers without built-in speech recognition (e.g. Firefox).

Chrome, Edge and Safari stream partial transcripts from the Web Speech API, which is what the
speculative-retrieval path needs. Other browsers record a short clip (MediaRecorder) and POST
it to /transcribe; the transcript then goes through the same WebSocket as a final utterance.

Provider: Whisper through any OpenAI-compatible `/audio/transcriptions` endpoint (Groq by
default, `whisper-large-v3-turbo`). A vocabulary prompt with the campaign's place and scheme
names noticeably helps Whisper with Indian proper nouns ("Vijayawada", "Pratibha").
"""
from __future__ import annotations

import logging
import time
from typing import Protocol

import httpx

from app.generation.llm.base import LLMError
from app.lexicon import DISTRICT_DISPLAY

logger = logging.getLogger(__name__)

BASE_VOCABULARY = ["Andhra Pradesh", "Sunrise Coast Alliance", "manifesto", "scheme", "district"]


class Transcriber(Protocol):
    name: str
    model: str

    async def transcribe(self, audio: bytes, filename: str, content_type: str,
                         language: str | None = None, vocabulary: list[str] | None = None) -> dict: ...


def vocabulary_prompt(extra: list[str] | None = None, limit_chars: int = 600) -> str:
    """Whisper's `prompt` biases spelling of rare words; it is capped at ~224 tokens."""
    words = BASE_VOCABULARY + [v for k, v in DISTRICT_DISPLAY.items() if k != "statewide"] + list(extra or [])
    seen, out = set(), []
    for w in words:
        if w and w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return (", ".join(out))[:limit_chars]


class WhisperTranscriber:
    def __init__(self, name: str, base_url: str, api_key: str, model: str = "whisper-large-v3-turbo",
                 timeout_s: float = 30.0, transport: httpx.AsyncBaseTransport | None = None):
        self.name = name
        self.model = model
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout_s, connect=5.0),
                                         headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                                         transport=transport)

    async def transcribe(self, audio: bytes, filename: str, content_type: str,
                         language: str | None = None, vocabulary: list[str] | None = None) -> dict:
        data = {"model": self.model, "response_format": "json", "temperature": "0",
                "prompt": vocabulary_prompt(vocabulary)}
        if language:
            data["language"] = language.split("-")[0]  # Whisper takes ISO-639-1 ("en"), browsers send "en-IN"
        t = time.perf_counter()
        try:
            r = await self._client.post("/audio/transcriptions", data=data,
                                        files={"file": (filename, audio, content_type or "application/octet-stream")})
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name} transcription failed: {exc.__class__.__name__}: {exc}") from exc
        if r.status_code >= 400:
            raise LLMError(f"{self.name} transcription HTTP {r.status_code}: {r.text[:300]}")
        try:
            text = (r.json().get("text") or "").strip()
        except ValueError as exc:
            raise LLMError(f"{self.name} returned an unexpected transcription payload") from exc
        return {"text": text, "provider": self.name, "model": self.model,
                "latency_ms": round((time.perf_counter() - t) * 1000, 2)}

    async def aclose(self) -> None:
        await self._client.aclose()
