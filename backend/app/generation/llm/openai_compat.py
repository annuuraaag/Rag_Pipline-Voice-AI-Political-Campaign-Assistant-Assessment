"""OpenAI-compatible chat-completions client (Groq, OpenAI, Ollama, vLLM, ...).

One adapter covers every provider that speaks `/chat/completions`, so switching
from Groq to a local Ollama model is a config change. Uses raw httpx (no SDK)
for full control over streaming and timeouts.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator

import httpx

from app.generation.llm.base import Grounding, LLMError

logger = logging.getLogger(__name__)

# Providers retire models (Groq does so regularly). If the configured model is gone,
# we pick the first available model matching these: fast non-reasoning models first, because
# reasoning models spend latency and token budget on hidden thinking before the answer.
PREFERRED_MODELS = ["llama-3.1-8b", "llama-4-scout", "llama-3.3-70b", "gpt-oss-20b", "qwen3-32b", "gpt-oss-120b"]
# Reasoning models count their hidden thinking against max_tokens; with a 300-token budget the
# answer gets cut mid-sentence. They get minimal reasoning effort plus extra headroom.
_REASONING = re.compile(r"gpt-oss|qwen3|deepseek-r1|(^|/)o[134](-|$)", re.IGNORECASE)
REASONING_HEADROOM = 1024


def is_reasoning_model(model: str) -> bool:
    return bool(_REASONING.search(model))
_NON_CHAT = ("whisper", "tts", "guard", "embed", "playai", "orpheus", "compound", "prompt-guard", "safeguard")


class ModelUnavailable(LLMError):
    """The configured model does not exist / was decommissioned for this account."""


class UnsupportedParameter(LLMError):
    """The provider rejected one of the optional reasoning parameters."""


def _is_model_error(status: int, body: str) -> bool:
    low = body.lower()
    return status in (400, 404) and ("model_not_found" in low or "decommissioned" in low or "does not exist" in low)


def choose_model(available: list[str]) -> str | None:
    chat = [m for m in available if not any(x in m.lower() for x in _NON_CHAT)]
    for pref in PREFERRED_MODELS:
        for m in sorted(chat):
            if pref in m.lower():
                return m
    return sorted(chat)[0] if chat else None


class OpenAICompatibleLLM:
    is_fallback = False

    def __init__(self, name: str, base_url: str, model: str, api_key: str = "", timeout_s: float = 20.0,
                 max_tokens: int = 300, temperature: float = 0.1, transport: httpx.AsyncBaseTransport | None = None):
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.send_reasoning_params = True  # turned off if the provider rejects them
        self.model_check: str | None = None  # result of the last model availability check (shown in /health)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        # A single pooled client keeps the TLS connection warm between requests (saves ~100-300 ms).
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=transport,
        )

    def _body(self, messages: list[dict[str, str]], max_tokens: int | None, stream: bool) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }
        if is_reasoning_model(self.model):
            body["max_tokens"] += REASONING_HEADROOM
            low = self.model.lower() if self.send_reasoning_params else ""
            if "gpt-oss" in low:
                body["reasoning_effort"] = "low"
                if self.name == "groq":
                    body["include_reasoning"] = False  # don't send the thinking over the wire
            elif "qwen3" in low and self.name == "groq":
                body["reasoning_effort"] = "none"
                body["reasoning_format"] = "hidden"
        return body

    def _raise_for(self, status: int, body: str) -> None:
        if _is_model_error(status, body):
            raise ModelUnavailable(f"{self.name} HTTP {status}: {body[:300]}")
        if status == 400 and self.send_reasoning_params and "reasoning" in body.lower():
            raise UnsupportedParameter(f"{self.name} HTTP {status}: {body[:300]}")
        raise LLMError(f"{self.name} HTTP {status}: {body[:300]}")

    async def _list_models(self) -> list[str]:
        r = await self._client.get("/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", []) if m.get("active", True)]

    async def _recover_model(self) -> bool:
        """Ask the provider which models exist and switch to a suitable one. Returns True if switched."""
        try:
            ids = await self._list_models()
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            self.model_check = f"'{self.model}' unavailable and the model list could not be fetched: {exc}"
            logger.error(self.model_check)
            return False
        new = choose_model(ids)
        if not new or new == self.model:
            self.model_check = f"'{self.model}' unavailable and no alternative chat model found in: {ids[:15]}"
            logger.error(self.model_check)
            return False
        self.model_check = f"switched from retired '{self.model}' to '{new}'"
        logger.warning("%s: %s. Set LLM_MODEL to pin a model.", self.name, self.model_check)
        self.model = new
        return True

    async def verify_model(self) -> str:
        """Startup check: make sure the configured model exists, switching before the first user query if not."""
        try:
            ids = await self._list_models()
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            self.model_check = f"could not verify model (will retry on first request): {exc.__class__.__name__}: {exc}"
            logger.warning("%s: %s", self.name, self.model_check)
            return self.model_check
        if self.model in ids:
            self.model_check = f"'{self.model}' available"
            logger.info("%s: %s", self.name, self.model_check)
            return self.model_check
        await self._recover_model()
        return self.model_check or ""

    async def stream(self, messages: list[dict[str, str]], max_tokens: int | None = None,
                     grounding: Grounding | None = None) -> AsyncIterator[str]:
        try:
            async for tok in self._stream_with_param_fallback(messages, max_tokens):
                yield tok
        except ModelUnavailable as exc:
            # Nothing was yielded yet (the error comes before the first token), so retrying is safe.
            if not await self._recover_model():
                raise ModelUnavailable(f"{exc} | auto-switch failed: {self.model_check}") from exc
            async for tok in self._stream_with_param_fallback(messages, max_tokens):
                yield tok

    def _drop_reasoning_params(self, exc: Exception) -> None:
        logger.warning("%s rejected reasoning parameters, retrying without them: %s", self.name, exc)
        self.send_reasoning_params = False

    async def _stream_with_param_fallback(self, messages, max_tokens) -> AsyncIterator[str]:
        try:
            async for tok in self._stream_once(messages, max_tokens):
                yield tok
        except UnsupportedParameter as exc:  # raised before the first token, so a retry is safe
            self._drop_reasoning_params(exc)
            async for tok in self._stream_once(messages, max_tokens):
                yield tok

    async def _stream_once(self, messages: list[dict[str, str]], max_tokens: int | None) -> AsyncIterator[str]:
        try:
            async with self._client.stream("POST", "/chat/completions", json=self._body(messages, max_tokens, True)) as r:
                if r.status_code >= 400:
                    self._raise_for(r.status_code, (await r.aread()).decode(errors="replace"))
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        choice = json.loads(payload)["choices"][0]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    delta = choice.get("delta", {}).get("content")
                    if delta:
                        yield delta
                    if choice.get("finish_reason") == "length":
                        logger.warning("%s/%s hit max_tokens; answer truncated (raise LLM_MAX_TOKENS)",
                                       self.name, self.model)
                        yield " …"
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name} request failed: {exc.__class__.__name__}: {exc}") from exc

    async def complete(self, messages: list[dict[str, str]], max_tokens: int | None = None,
                       grounding: Grounding | None = None) -> str:
        try:
            return await self._complete_with_param_fallback(messages, max_tokens)
        except ModelUnavailable as exc:
            if not await self._recover_model():
                raise ModelUnavailable(f"{exc} | auto-switch failed: {self.model_check}") from exc
            return await self._complete_with_param_fallback(messages, max_tokens)

    async def _complete_with_param_fallback(self, messages, max_tokens) -> str:
        try:
            return await self._complete_once(messages, max_tokens)
        except UnsupportedParameter as exc:
            self._drop_reasoning_params(exc)
            return await self._complete_once(messages, max_tokens)

    async def _complete_once(self, messages: list[dict[str, str]], max_tokens: int | None) -> str:
        try:
            r = await self._client.post("/chat/completions", json=self._body(messages, max_tokens, False))
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name} request failed: {exc.__class__.__name__}: {exc}") from exc
        if r.status_code >= 400:
            self._raise_for(r.status_code, r.text)
        try:
            choice = r.json()["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"{self.name} returned an unexpected payload") from exc
        if choice.get("finish_reason") == "length":
            logger.warning("%s/%s hit max_tokens; answer truncated (raise LLM_MAX_TOKENS)", self.name, self.model)
            text += " …"
        return text

    async def aclose(self) -> None:
        await self._client.aclose()
