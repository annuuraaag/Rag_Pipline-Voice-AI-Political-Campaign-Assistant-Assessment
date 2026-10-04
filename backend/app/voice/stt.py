"""Streaming speech-to-text on the server (STT_PROVIDER), for microphone audio sent over /ws/voice.

The browser's Web Speech API sends audio to the browser vendor's cloud and can't be controlled
or measured. With a server provider the audio goes to a recogniser this deployment chooses:

    local     sherpa-onnx streaming transducer on CPU (default model: Kroko streaming zipformer,
              ~70 MB, ~0.05x real time). Audio never leaves the server.
    deepgram  Deepgram's streaming API over a WebSocket (DEEPGRAM_API_KEY), with key terms
              (district and scheme names) to bias recognition.
    whisper   Whisper through any OpenAI-compatible /audio/transcriptions endpoint: Groq by
              default (GROQ_API_KEY), or a self-hosted server via STT_BASE_URL. Not a streaming
              model: partial transcripts come from re-transcribing the audio every
              STT_PARTIAL_INTERVAL_MS.

Every provider opens one stream per utterance:

    send(pcm)   16 kHz PCM16 mono; partial transcripts arrive through the ``on_partial`` callback
    flush()     the best transcript of everything sent so far, as fast as possible. Called at
                pauses, so end-of-turn detection can judge the whole question, not a transcript
                that lags behind the audio.
    finish()    the final transcript; the stream is done
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode

import numpy as np

from app.voice.audio import STT_RATE, pcm16_to_float, wav_bytes
from app.voice.transcriber import BASE_VOCABULARY, WhisperTranscriber

logger = logging.getLogger(__name__)

OnPartial = Callable[[str], Awaitable[None]]


class STTStream(Protocol):
    async def send(self, pcm: bytes) -> None: ...

    async def flush(self) -> str: ...

    async def finish(self) -> str: ...

    async def close(self) -> None: ...


class STTProvider(Protocol):
    name: str
    model: str
    streaming: bool  # partial transcripts while the user is still talking

    async def open(self, on_partial: OnPartial, language: str = "en", vocabulary: list[str] | None = None
                   ) -> STTStream: ...


def tidy(text: str) -> str:
    """Recognisers trained on audiobooks shout ("WHAT IS THE PLAN"); keep everything else as is."""
    t = " ".join(text.split())
    return t.lower() if t.isupper() else t


# ── local: sherpa-onnx ────────────────────────────────────────────────────
class SherpaSTT:
    """Streaming transducer (zipformer / NeMo fast-conformer) from a sherpa-onnx model directory."""

    name = "local"
    streaming = True

    def __init__(self, model_dir: str, num_threads: int = 2, tail_s: float = 1.2):
        import sherpa_onnx

        d = Path(model_dir)
        if not d.is_dir():
            raise FileNotFoundError(f"STT model directory not found: {model_dir}")

        def pick(part: str) -> str:
            files = sorted(d.glob(f"{part}*.onnx"))
            plain = [f for f in files if ".int8." not in f.name]
            if not files:
                raise FileNotFoundError(f"no {part}*.onnx in {model_dir}")
            return str((plain or files)[0])
        self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=pick("encoder"), decoder=pick("decoder"), joiner=pick("joiner"),
            num_threads=num_threads, sample_rate=STT_RATE, feature_dim=80)
        self.model = d.name
        self.tail = np.zeros(int(tail_s * STT_RATE), dtype=np.float32)  # flushes the model's look-ahead
        self._lock = threading.Lock()

    async def open(self, on_partial: OnPartial, language: str = "en", vocabulary: list[str] | None = None
                   ) -> _SherpaStream:
        return _SherpaStream(self, on_partial)

    def decode(self, stream: Any, x: np.ndarray, final: bool = False) -> str:
        with self._lock:
            stream.accept_waveform(STT_RATE, x)
            if final:
                stream.accept_waveform(STT_RATE, self.tail)
                stream.input_finished()
            while self._rec.is_ready(stream):
                self._rec.decode_stream(stream)
            return tidy(self._rec.get_result(stream))

    def transcribe(self, audio: np.ndarray) -> str:
        """Decode a whole clip on a fresh stream, look-ahead included (a few % of its duration)."""
        return self.decode(self._rec.create_stream(), audio, final=True)


class _SherpaStream:
    def __init__(self, p: SherpaSTT, on_partial: OnPartial):
        self.p = p
        self.on_partial = on_partial
        self.stream = p._rec.create_stream()
        self.chunks: list[np.ndarray] = []
        self.samples = 0
        self.text = ""
        self._flushed: tuple[int, str] | None = None

    async def send(self, pcm: bytes) -> None:
        x = pcm16_to_float(pcm)
        if not len(x):
            return
        self.chunks.append(x)
        self.samples += len(x)
        text = await asyncio.to_thread(self.p.decode, self.stream, x)
        if text and text != self.text:
            self.text = text
            await self.on_partial(text)

    async def flush(self) -> str:
        # The live stream trails the audio by the model's look-ahead (~1 s for Kroko), so the last
        # words would be missing. Re-decode the utterance with silence appended instead.
        if self._flushed is None or self._flushed[0] != self.samples:
            n = self.samples
            audio = np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.float32)
            self._flushed = (n, await asyncio.to_thread(self.p.transcribe, audio))
        return self._flushed[1]

    async def finish(self) -> str:
        if self._flushed is not None and self._flushed[0] == self.samples:
            return self._flushed[1]
        return await asyncio.to_thread(self.p.decode, self.stream, np.zeros(0, dtype=np.float32), True)

    async def close(self) -> None:
        self.chunks = []


# ── Deepgram ─────────────────────────────────────────────────────────────
class DeepgramSTT:
    name = "deepgram"
    streaming = True

    def __init__(self, api_key: str, model: str = "nova-3", url: str = "wss://api.deepgram.com/v1/listen",
                 connect: Callable[..., Any] | None = None):
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.url = url
        self._connect = connect

    async def open(self, on_partial: OnPartial, language: str = "en", vocabulary: list[str] | None = None
                   ) -> _DeepgramStream:
        params: list[tuple[str, str]] = [
            ("model", self.model), ("language", language or "en"), ("encoding", "linear16"),
            ("sample_rate", str(STT_RATE)), ("channels", "1"), ("interim_results", "true"),
            ("punctuate", "true"), ("smart_format", "true"),
        ]
        terms = list(dict.fromkeys([*BASE_VOCABULARY, *(vocabulary or [])]))[:50]
        # Nova-3 takes "keyterm" prompts; older models take boosted "keywords".
        params += [("keyterm", t) for t in terms] if self.model.startswith("nova-3") else [("keywords", t) for t in terms]
        connect = self._connect
        if connect is None:
            from websockets.asyncio.client import connect
        ws = await connect(f"{self.url}?{urlencode(params)}", additional_headers={"Authorization": f"Token {self.api_key}"},
                           open_timeout=5)
        return _DeepgramStream(ws, on_partial)


class _DeepgramStream:
    def __init__(self, ws: Any, on_partial: OnPartial):
        self.ws = ws
        self.on_partial = on_partial
        self.finals: list[str] = []
        self.interim = ""
        self.text = ""
        self._finalized = asyncio.Event()
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            async for raw in self.ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                if msg.get("type") != "Results":
                    continue
                alt = ((msg.get("channel") or {}).get("alternatives") or [{}])[0].get("transcript", "").strip()
                if msg.get("is_final"):
                    if alt:
                        self.finals.append(alt)
                    self.interim = ""
                else:
                    self.interim = alt
                text = " ".join([*self.finals, self.interim]).strip()
                if text and text != self.text:
                    self.text = text
                    await self.on_partial(text)
                if msg.get("from_finalize"):
                    self._finalized.set()
        except Exception as exc:  # connection closed or provider error: the session falls back to what it has
            logger.info("deepgram stream ended: %s", exc.__class__.__name__)
        finally:
            self._finalized.set()

    async def send(self, pcm: bytes) -> None:
        await self.ws.send(pcm)

    async def flush(self) -> str:
        """Ask Deepgram to finalise what it has heard (no need to wait out its own look-ahead)."""
        self._finalized.clear()
        await self.ws.send(json.dumps({"type": "Finalize"}))
        try:
            await asyncio.wait_for(self._finalized.wait(), timeout=1.5)
        except TimeoutError:
            logger.info("deepgram: no finalize response within 1.5 s")
        return " ".join(self.finals).strip() or self.text

    async def finish(self) -> str:
        text = await self.flush()
        await self.close()
        return text

    async def close(self) -> None:
        try:
            await self.ws.send(json.dumps({"type": "CloseStream"}))
        except Exception:  # noqa: BLE001 - already closed
            pass
        await self.ws.close()
        self._reader.cancel()


# ── Whisper (OpenAI-compatible) ──────────────────────────────────────────
class WhisperSTT:
    streaming = False

    def __init__(self, transcriber: WhisperTranscriber, partial_interval_s: float = 1.5):
        self.t = transcriber
        self.name = f"whisper ({transcriber.name})"
        self.model = transcriber.model
        self.partial_interval_s = partial_interval_s

    async def open(self, on_partial: OnPartial, language: str = "en", vocabulary: list[str] | None = None
                   ) -> _WhisperStream:
        return _WhisperStream(self, on_partial, language, vocabulary)


class _WhisperStream:
    def __init__(self, p: WhisperSTT, on_partial: OnPartial, language: str, vocabulary: list[str] | None):
        self.p = p
        self.on_partial = on_partial
        self.language = language
        self.vocabulary = vocabulary
        self.pcm = bytearray()
        self._partial_at = 0          # bytes covered by the last partial request
        self._partial: asyncio.Task | None = None
        self._cache: tuple[int, str] | None = None

    async def _transcribe(self, n: int) -> str:
        if self._cache and self._cache[0] == n:
            return self._cache[1]
        r = await self.p.t.transcribe(wav_bytes(bytes(self.pcm[:n]), STT_RATE), "speech.wav", "audio/wav",
                                      language=self.language, vocabulary=self.vocabulary)
        self._cache = (n, tidy(r["text"]))
        return self._cache[1]

    async def send(self, pcm: bytes) -> None:
        self.pcm += pcm
        step = int(self.p.partial_interval_s * STT_RATE) * 2
        if step and len(self.pcm) - self._partial_at >= step and (self._partial is None or self._partial.done()):
            self._partial_at = len(self.pcm)
            self._partial = asyncio.create_task(self._emit_partial(len(self.pcm)))

    async def _emit_partial(self, n: int) -> None:
        try:
            text = await self._transcribe(n)
        except Exception as exc:  # noqa: BLE001 - a failed partial only costs speculation
            logger.info("whisper partial failed: %s", exc)
            return
        if text:
            await self.on_partial(text)

    async def flush(self) -> str:
        return await self._transcribe(len(self.pcm))

    async def finish(self) -> str:
        return await self.flush()

    async def close(self) -> None:
        if self._partial and not self._partial.done():
            self._partial.cancel()


# ── factory ──────────────────────────────────────────────────────────────
def build_stt(s: Any, transcriber: WhisperTranscriber | None) -> tuple[STTProvider | None, str]:
    """(provider or None, status for /health). A provider that fails to load never blocks startup."""
    kind = s.stt_provider
    if kind == "none":
        return None, "disabled (STT_PROVIDER=none: browsers recognise speech themselves)"
    try:
        t = time.perf_counter()
        if kind == "local":
            p: STTProvider = SherpaSTT(s.resolved_stt_model_dir)
        elif kind == "deepgram":
            p = DeepgramSTT(s.deepgram_api_key, model=s.stt_model or "nova-3")
        elif kind == "whisper":
            if transcriber is None:
                raise ValueError("needs an OpenAI-compatible key (GROQ_API_KEY / OPENAI_API_KEY) or STT_BASE_URL")
            p = WhisperSTT(transcriber, partial_interval_s=s.stt_partial_interval_ms / 1000)
        else:
            raise ValueError(f"unknown STT_PROVIDER {kind!r}")
        logger.info("Server speech-to-text: %s/%s (%.1f s to load)", p.name, p.model, time.perf_counter() - t)
        return p, f"{p.name}/{p.model}"
    except Exception as exc:
        logger.warning("Server speech-to-text '%s' unavailable: %s", kind, exc)
        return None, f"unavailable: {exc}"[:200]


def warm_up(p: STTProvider | None) -> None:
    """First ONNX inference allocates buffers; pay that at startup, not on the first question."""
    if isinstance(p, SherpaSTT):
        p.transcribe(np.zeros(STT_RATE // 2, dtype=np.float32))

