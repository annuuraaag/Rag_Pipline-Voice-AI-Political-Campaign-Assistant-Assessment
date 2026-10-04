"""Text-to-speech on the server (TTS_PROVIDER): answers are spoken by a voice this deployment
chooses and streamed to the browser as audio, instead of each browser's own speechSynthesis
voice (which differs by browser and OS, and cannot be measured from the server).

    local       sherpa-onnx on CPU: a Piper (VITS) voice (default: en_US libritts_r, ~0.1x real
                time) or Kokoro (more natural, ~0.7x real time: needs a faster CPU)
    deepgram    Deepgram Aura, streamed (DEEPGRAM_API_KEY)
    openai      any OpenAI-compatible /audio/speech: OpenAI by default, or a self-hosted server
                (Kokoro-FastAPI, Speaches, ...) via TTS_BASE_URL
    elevenlabs  ElevenLabs, streamed (ELEVENLABS_API_KEY)

``stream(text)`` yields PCM16 mono at ``sample_rate``. The answer is synthesised one sentence at
a time while the LLM is still writing (app/voice/speech.py), so the first audio depends on the
first sentence only.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

from app.voice.audio import float_to_pcm16, frames_of, parse_wav

logger = logging.getLogger(__name__)

CHUNK_MS = 250   # audio is sent in pieces this long, so playback can start (and stop) promptly


class TTSProvider(Protocol):
    name: str
    voice: str
    sample_rate: int

    def stream(self, text: str, speed: float = 1.0) -> AsyncIterator[bytes]: ...


# ── local: sherpa-onnx (Piper / Kokoro) ─────────────────────────────────────
class SherpaTTS:
    name = "local"

    def __init__(self, model_dir: str, speaker: int = 0, num_threads: int = 2):
        import sherpa_onnx

        d = Path(model_dir)
        if not d.is_dir():
            raise FileNotFoundError(f"TTS model directory not found: {model_dir}")
        if (d / "voices.bin").exists():  # Kokoro
            cfg = sherpa_onnx.OfflineTtsModelConfig(kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(d / "model.onnx"), voices=str(d / "voices.bin"), tokens=str(d / "tokens.txt"),
                data_dir=str(d / "espeak-ng-data")), num_threads=num_threads)
        else:  # Piper / VITS
            onnx = sorted(d.glob("*.onnx"))
            if not onnx:
                raise FileNotFoundError(f"no .onnx voice in {model_dir}")
            cfg = sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(onnx[0]), lexicon="", tokens=str(d / "tokens.txt"), data_dir=str(d / "espeak-ng-data")),
                num_threads=num_threads)
        self._tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=cfg, max_num_sentences=1))
        self.speaker = min(max(speaker, 0), max(self._tts.num_speakers - 1, 0))
        self.voice = f"{d.name}" + (f" #{self.speaker}" if self._tts.num_speakers > 1 else "")
        self.sample_rate = int(self._tts.sample_rate)
        self._lock = threading.Lock()  # one synthesis at a time per loaded voice

    def synthesize(self, text: str, speed: float = 1.0) -> bytes:
        with self._lock:
            audio = self._tts.generate(text, sid=self.speaker, speed=speed)
        return float_to_pcm16(np.asarray(audio.samples, dtype=np.float32))

    async def stream(self, text: str, speed: float = 1.0) -> AsyncIterator[bytes]:
        pcm = await asyncio.to_thread(self.synthesize, text, speed)
        for chunk in frames_of(pcm, self.sample_rate, CHUNK_MS):
            yield chunk


# ── HTTP providers ───────────────────────────────────────────────────────
class _HTTPTTS:
    """Shared plumbing: one pooled client, and PCM re-chunking that never splits a sample."""

    def __init__(self, base_url: str, headers: dict[str, str], timeout_s: float = 20.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, transport=transport,
                                         timeout=httpx.Timeout(timeout_s, connect=5.0))

    async def _pcm_stream(self, url: str, body: dict, params: dict | None = None) -> AsyncIterator[bytes]:
        step = int(self.sample_rate * CHUNK_MS / 1000) * 2
        buf = bytearray()
        async with self._client.stream("POST", url, json=body, params=params) as r:
            if r.status_code >= 400:
                raise RuntimeError(f"{self.name} TTS HTTP {r.status_code}: {(await r.aread())[:300]!r}")
            async for data in r.aiter_bytes():
                buf += data
                while len(buf) >= step:
                    yield bytes(buf[:step])
                    del buf[:step]
        if len(buf) >= 2:
            yield bytes(buf[: len(buf) - len(buf) % 2])

    async def aclose(self) -> None:
        await self._client.aclose()


class DeepgramTTS(_HTTPTTS):
    name = "deepgram"

    def __init__(self, api_key: str, voice: str = "aura-2-thalia-en", sample_rate: int = 24000,
                 transport: httpx.AsyncBaseTransport | None = None):
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY is not set")
        super().__init__("https://api.deepgram.com/v1", {"Authorization": f"Token {api_key}"}, transport=transport)
        self.voice = voice
        self.sample_rate = sample_rate

    def stream(self, text: str, speed: float = 1.0) -> AsyncIterator[bytes]:
        params = {"model": self.voice, "encoding": "linear16", "sample_rate": self.sample_rate, "container": "none"}
        return self._pcm_stream("/speak", {"text": text}, params)


class ElevenLabsTTS(_HTTPTTS):
    name = "elevenlabs"

    def __init__(self, api_key: str, voice: str, model: str = "eleven_flash_v2_5", sample_rate: int = 24000,
                 transport: httpx.AsyncBaseTransport | None = None):
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is not set")
        if not voice:
            raise ValueError("TTS_VOICE must be an ElevenLabs voice id")
        super().__init__("https://api.elevenlabs.io/v1", {"xi-api-key": api_key}, transport=transport)
        self.voice = voice
        self.model = model
        self.sample_rate = sample_rate

    def stream(self, text: str, speed: float = 1.0) -> AsyncIterator[bytes]:
        body: dict[str, Any] = {"text": text, "model_id": self.model}
        if speed != 1.0:
            body["voice_settings"] = {"speed": min(max(speed, 0.7), 1.2)}
        return self._pcm_stream(f"/text-to-speech/{self.voice}/stream", body,
                                {"output_format": f"pcm_{self.sample_rate}"})


class OpenAITTS(_HTTPTTS):
    """OpenAI's /audio/speech, or a self-hosted server with the same API.

    ``response_format="pcm"`` streams raw 24 kHz PCM (OpenAI and most self-hosted servers);
    ``"wav"`` is for servers that only return WAV (read whole, then chunked).
    """

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o-mini-tts", voice: str = "alloy",
                 response_format: str = "pcm", sample_rate: int = 24000, transport: httpx.AsyncBaseTransport | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        super().__init__(base_url.rstrip("/"), headers, transport=transport)
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.sample_rate = sample_rate

    async def stream(self, text: str, speed: float = 1.0) -> AsyncIterator[bytes]:
        body: dict[str, Any] = {"model": self.model, "input": text, "voice": self.voice,
                                "response_format": self.response_format}
        if speed != 1.0:
            body["speed"] = min(max(speed, 0.5), 2.0)
        if self.response_format == "pcm":
            async for chunk in self._pcm_stream("/audio/speech", body):
                yield chunk
            return
        r = await self._client.post("/audio/speech", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"{self.name} TTS HTTP {r.status_code}: {r.text[:300]}")
        pcm, self.sample_rate = parse_wav(r.content)
        for chunk in frames_of(pcm, self.sample_rate, CHUNK_MS):
            yield chunk


# ── factory ──────────────────────────────────────────────────────────────
def build_tts(s: Any) -> tuple[TTSProvider | None, str]:
    """(provider or None, status for /health). A provider that fails to load never blocks startup."""
    kind = s.tts_provider
    if kind == "none":
        return None, "disabled (TTS_PROVIDER=none: browsers speak with their own voices)"
    try:
        t = time.perf_counter()
        if kind == "local":
            p: TTSProvider = SherpaTTS(s.resolved_tts_model_dir, speaker=int(s.tts_voice or 0))
        elif kind == "deepgram":
            p = DeepgramTTS(s.deepgram_api_key, voice=s.tts_voice or "aura-2-thalia-en")
        elif kind == "elevenlabs":
            p = ElevenLabsTTS(s.elevenlabs_api_key, voice=s.tts_voice, model=s.tts_model or "eleven_flash_v2_5")
        elif kind == "openai":
            base = s.tts_base_url or "https://api.openai.com/v1"
            key = s.tts_api_key or (s.openai_api_key if "api.openai.com" in base else "")
            if "api.openai.com" in base and not key:
                raise ValueError("needs OPENAI_API_KEY (or TTS_API_KEY), or TTS_BASE_URL for a self-hosted server")
            p = OpenAITTS(base, key, model=s.tts_model or "gpt-4o-mini-tts", voice=s.tts_voice or "alloy",
                          response_format=s.tts_response_format)
        else:
            raise ValueError(f"unknown TTS_PROVIDER {kind!r}")
        logger.info("Server text-to-speech: %s/%s (%.1f s to load)", p.name, p.voice, time.perf_counter() - t)
        return p, f"{p.name}/{p.voice}"
    except Exception as exc:
        logger.warning("Server text-to-speech '%s' unavailable: %s", kind, exc)
        return None, f"unavailable: {exc}"[:200]


def warm_up(p: TTSProvider | None) -> None:
    if isinstance(p, SherpaTTS):
        p.synthesize("Ready.")
