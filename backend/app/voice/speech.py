"""Server-side speech for one /ws/voice connection: microphone audio in, spoken answers out.

Input (``SpeechInput``): the browser streams 16 kHz PCM16 while the user talks.

    audio ─┬─ VAD (20 ms frames) ── silence timing, pauses, barge-in
           └─ streaming STT ──────── partial transcripts → speculative retrieval (S1/S2)
    pause ≥ PROBE_AFTER  → flush the recogniser: the whole question so far, look-ahead included
                         → end-of-turn verdict on it (app/voice/endpointing.py)
    silence ≥ verdict's wait → final transcript → the answer starts

  Transcripts pass through the campaign's vocabulary corrector (app/voice/vocabulary.py), so
  "Vid, awada" becomes "Vijayawada" before retrieval or end-of-turn rules see it.
  Silence is measured on the audio clock (samples received), so it is exact whatever the network
  jitter. While an answer is playing, audio is only watched for barge-in: speech that the
  recogniser turns into at least two words not taken from the answer itself (echo) stops the
  answer and becomes the next question, its first words included.

Output (``AnswerSpeaker``): answer tokens are cut into sentences as they stream (the first piece
may end at a comma) and synthesised one by one, in order, while the LLM keeps writing. Audio goes
to the browser in ~250 ms binary frames. With CITATION_VERIFICATION=strict a sentence is checked
against its sources before it is spoken, and skipped if unsupported.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.voice.audio import STT_RATE, SentenceSplitter, pcm16_to_float, speakable
from app.voice.endpointing import EndOfTurn
from app.voice.stt import STTProvider, STTStream
from app.voice.tts import TTSProvider
from app.voice.vad import VAD

logger = logging.getLogger(__name__)

FRAME = STT_RATE // 50          # 20 ms VAD frames
_WORDS = re.compile(r"[a-z0-9']+")

Emit = Callable[[dict[str, Any]], Awaitable[None]]


def words(text: str) -> list[str]:
    return _WORDS.findall(text.lower())


def is_echo(heard: str, spoken: str) -> bool:
    """True when `heard` is probably the assistant's own voice picked up by the microphone: its
    words follow what is being said, in the same order. Word pairs, not single words, so a
    question that reuses some of the answer's words ("who is eligible for ...") is still the user."""
    w = words(heard)
    if len(w) < 2:
        return True   # one word ("ok", "the") is never a deliberate interruption
    said = words(spoken)
    if len(said) < 2:
        return False
    pairs = set(zip(said, said[1:], strict=False))
    heard_pairs = list(zip(w, w[1:], strict=False))
    return sum(p in pairs for p in heard_pairs) / len(heard_pairs) >= 0.6


@dataclass
class SpeechHooks:
    """What the voice connection provides to the speech input."""
    emit: Emit                                                   # JSON event to the client
    assess: Callable[[str], EndOfTurn | None]                    # end-of-turn verdict for a transcript
    on_partial: Callable[[str], Awaitable[None]]                 # speculative retrieval + endpoint hint
    on_utterance: Callable[[str, dict[str, Any]], Awaitable[None]]  # final transcript → answer
    answering: Callable[[], bool]                                # an answer is being generated or spoken
    spoken_text: Callable[[], str]                               # what the assistant just said (echo check)
    on_barge_in: Callable[[], Awaitable[None]]                   # stop the current answer


class SpeechInput:
    def __init__(self, stt: STTProvider, vad: VAD, hooks: SpeechHooks, *, language: str = "en",
                 vocabulary: list[str] | None = None, correct: Callable[[str], str] | None = None,
                 unsure_wait_s: float = 0.9, probe_after_s: float = 0.12,
                 max_utterance_s: float = 30.0, no_speech_s: float = 8.0, barge_in_s: float = 0.25,
                 barge_in_words: int = 2):
        self.stt = stt
        self.vad = vad
        self.h = hooks
        self.language = language
        self.vocabulary = vocabulary
        self.fix = correct or (lambda text: text)   # misheard place and scheme names → the real ones
        self.unsure_wait_s = unsure_wait_s
        self.probe_after_s = probe_after_s
        self.max_utterance_s = max_utterance_s
        self.no_speech_s = no_speech_s
        self.barge_in_s = barge_in_s
        self.barge_in_words = barge_in_words
        self.stream: STTStream | None = None
        self.listening = False        # a user turn is open
        self.candidate = False        # speech during an answer, not yet judged as barge-in
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll: deque[bytes] = deque(maxlen=25)   # recent audio (~1 s), so a barge-in keeps its first words
        self._finalizing = False
        self._probe_task: asyncio.Task | None = None
        self._watchdog: asyncio.Task | None = None
        self.last_frame_wall = 0.0
        self._reset_clock()

    # ── turn lifecycle ─────────────────────────────────────────────────
    def _reset_clock(self) -> None:
        self.t = 0.0                  # audio clock: seconds received in this turn
        self.speech = False           # speech heard in this turn
        self.last_voice_t = 0.0
        self.voiced_run = 0.0
        self.probe: tuple[float, str, EndOfTurn | None] | None = None   # (last_voice_t, text, verdict)
        self.partial = ""

    async def start_turn(self) -> None:
        """The client opened the microphone for a question."""
        if self.listening:
            return
        await self._close_stream()
        self._reset_clock()
        self.listening, self.candidate, self._finalizing = True, False, False
        self.stream = await self.stt.open(self._on_stt_partial, self.language, self.vocabulary)
        self._ensure_watchdog()

    async def stop_turn(self) -> None:
        """Push-to-talk released or the stop button: what was said so far is the question."""
        if self.listening and not self._finalizing:
            await self._finalize("manual")

    async def cancel_turn(self) -> None:
        self.listening = self.candidate = False
        await self._close_stream()

    async def close(self) -> None:
        for task in (self._probe_task, self._watchdog):
            if task and not task.done():
                task.cancel()
        await self.cancel_turn()

    def _ensure_watchdog(self) -> None:
        self.last_frame_wall = time.perf_counter()
        if self._watchdog is None or self._watchdog.done():
            self._watchdog = asyncio.create_task(self._watch())

    async def _watch(self) -> None:
        """Audio normally keeps flowing while the microphone is open, and the audio clock decides.
        If it stops (network stall, a client that stops sending at silence), end the question on
        the wall clock instead of waiting forever."""
        while True:
            await asyncio.sleep(0.25)
            if not self.listening:
                if not self.h.answering():
                    return
                continue
            idle = time.perf_counter() - self.last_frame_wall
            if self.speech and not self._finalizing and idle > 1.5:
                await self._finalize("stalled")

    async def _close_stream(self) -> None:
        stream, self.stream = self.stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                await stream.close()

    # ── audio ──────────────────────────────────────────────────────────
    async def on_audio(self, pcm: bytes) -> None:
        if not (self.listening or self.h.answering()) or self._finalizing:
            self._pending = np.zeros(0, dtype=np.float32)
            return
        self._preroll.append(pcm)
        self.last_frame_wall = time.perf_counter()
        if self.stream is not None and (self.listening or self.candidate):
            await self.stream.send(pcm)
        x = np.concatenate([self._pending, pcm16_to_float(pcm)])
        n = len(x) // FRAME * FRAME
        self._pending = x[n:]
        for i in range(0, n, FRAME):
            self._frame(x[i:i + FRAME])
        if self.listening:
            await self._check_turn()
        elif self.h.answering():
            await self._check_barge_in()

    def _frame(self, frame: np.ndarray) -> None:
        self.t += FRAME / STT_RATE
        if self.vad.is_speech(frame):
            self.voiced_run += FRAME / STT_RATE
            if self.voiced_run >= 0.06:          # 3 frames: not a click
                self.speech = True
                self.last_voice_t = self.t
        else:
            self.voiced_run = 0.0

    @property
    def silence(self) -> float:
        return self.t - self.last_voice_t if self.speech else 0.0

    # ── end of turn ────────────────────────────────────────────────────
    async def _check_turn(self) -> None:
        if not self.speech:
            if self.t >= self.no_speech_s:
                self.listening = False
                await self._close_stream()
                await self.h.emit({"type": "utterance", "text": "", "reason": "no_speech"})
            return
        if self.t >= self.max_utterance_s:
            await self._finalize("max_length")
            return
        if self.silence < self.probe_after_s:
            return
        probe = self.probe
        if probe is None or probe[0] != self.last_voice_t:
            if self._probe_task is None or self._probe_task.done():
                self._probe_task = asyncio.create_task(self._run_probe(self.last_voice_t))
            return
        _, text, verdict = probe
        if not text:
            if self.silence >= 1.0:          # a cough or a door: no words, keep listening
                self.speech = False
            return
        wait = verdict.wait_ms / 1000 if verdict is not None else self.unsure_wait_s
        if self.silence >= wait:
            await self._finalize("silence")

    async def _run_probe(self, at: float) -> None:
        """At a pause: the full transcript so far, and how finished it sounds."""
        stream = self.stream
        if stream is None:
            return
        try:
            text = self.fix(await stream.flush())
        except Exception as exc:  # noqa: BLE001 - provider hiccup: the next pause probes again
            logger.info("speech: flush failed: %s", exc)
            return
        if at != self.last_voice_t or not (self.listening or self.candidate):
            return   # the user kept talking; this transcript is already stale
        verdict = self.h.assess(text) if text else None
        self.probe = (at, text, verdict)
        if text and self.listening:
            await self._show_partial(text)
            await self._check_turn()   # the silence may already be long enough

    async def _finalize(self, reason: str) -> None:
        if self._finalizing:
            return
        self._finalizing = True
        silence_ms = round(self.silence * 1000)
        t0 = time.perf_counter()   # the moment the turn was judged over
        stream, self.stream = self.stream, None
        text = ""
        try:
            probe = self.probe
            if probe is not None and probe[0] == self.last_voice_t and reason == "silence":
                text = probe[1]   # nothing was said since the probe: it is the final transcript
            elif stream is not None:
                text = self.fix(await stream.finish())
        except Exception as exc:  # noqa: BLE001
            logger.warning("speech: final transcript failed: %s", exc)
            text = self.partial or (probe[1] if probe else "")
        finally:
            if stream is not None:
                with contextlib.suppress(Exception):
                    await stream.close()
        self.listening = False
        info = {"reason": reason, "endpoint_ms": silence_ms, "stt_final_ms": round((time.perf_counter() - t0) * 1000, 1),
                "decided_wall": t0, "speech_s": round(self.last_voice_t, 2)}
        try:
            if text.strip():
                await self.h.on_utterance(text.strip(), info)
            else:
                await self.h.emit({"type": "utterance", "text": "", "reason": "no_words"})
        finally:
            self._finalizing = False

    # ── partial transcripts ────────────────────────────────────────────
    async def _on_stt_partial(self, text: str) -> None:
        text = self.fix(text)
        if self.listening:
            await self._show_partial(text)
        elif self.candidate:
            await self._judge_candidate(text)

    async def _show_partial(self, text: str) -> None:
        if text == self.partial:
            return
        self.partial = text
        await self.h.emit({"type": "transcript", "text": text, "final": False})
        await self.h.on_partial(text)

    # ── barge-in ───────────────────────────────────────────────────────
    async def _check_barge_in(self) -> None:
        if not self.candidate:
            if self.voiced_run >= self.barge_in_s:
                # Someone is talking over the answer: start recognising, first 0.5 s included.
                self.candidate = True
                self.t, self.speech, self.last_voice_t = 0.0, True, 0.0
                self.probe = None
                self.stream = await self.stt.open(self._on_stt_partial, self.language, self.vocabulary)
                for chunk in list(self._preroll):
                    await self.stream.send(chunk)
            return
        if self.silence >= 0.8:                  # they stopped without saying anything that counts
            self.candidate = False
            await self._close_stream()
            return
        # Recognisers trail the audio (look-ahead, or no streaming at all): probe the words so far
        # every ~0.3 s of speech (0.6 s for a batch recogniser) instead of waiting for them.
        every = 0.3 if self.stt.streaming else 0.6
        if self.t - (self.probe[0] if self.probe else 0.0) >= every:
            if self._probe_task is None or self._probe_task.done():
                self.probe = (self.t, "", None)
                self._probe_task = asyncio.create_task(self._probe_candidate())

    async def _probe_candidate(self) -> None:
        stream = self.stream
        if stream is None:
            return
        try:
            text = self.fix(await stream.flush())
        except Exception:  # noqa: BLE001
            return
        if self.candidate:
            await self._judge_candidate(text)

    async def _judge_candidate(self, text: str) -> None:
        if len(words(text)) < self.barge_in_words or is_echo(text, self.h.spoken_text()):
            return
        logger.info("speech: barge-in")
        self.candidate = False
        await self.h.on_barge_in()
        # The interruption is the next question: keep its audio, clock and transcript.
        self.listening = True
        self.probe = None
        self._ensure_watchdog()
        await self._show_partial(text)


class AnswerSpeaker:
    """Speaks one answer while it is being written: tokens in, ordered audio frames out."""

    def __init__(self, tts: TTSProvider, send_audio: Callable[[dict, bytes], Awaitable[None]], emit: Emit,
                 turn_id: str, *, speed: float = 1.0, first_clause: int = 24,
                 gate: Callable[[str], Awaitable[bool]] | None = None):
        self.tts = tts
        self.send_audio = send_audio
        self.emit = emit
        self.turn_id = turn_id
        self.speed = speed
        self.gate = gate
        self.splitter = SentenceSplitter(first_clause=0 if gate else first_clause)
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.spoken: list[str] = []
        self.started = time.perf_counter()
        self.first_audio_at: float | None = None
        self.first_synth_ms: float | None = None
        self.chunks = 0
        self.skipped = 0
        self.error: str | None = None
        self.task = asyncio.create_task(self._run())

    def push(self, delta: str) -> None:
        for sentence in self.splitter.push(delta):
            self.queue.put_nowait(sentence)

    def end(self) -> None:
        for sentence in self.splitter.flush():
            self.queue.put_nowait(sentence)
        self.queue.put_nowait(None)

    def recent_text(self) -> str:
        return " ".join(self.spoken[-3:])

    async def wait(self) -> None:
        await self.task

    async def cancel(self) -> None:
        if not self.task.done():
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.task

    async def _run(self) -> None:
        index = 0
        try:
            while (sentence := await self.queue.get()) is not None:
                if self.gate is not None and not await self.gate(sentence):
                    self.skipped += 1
                    continue
                text = speakable(sentence)
                if not text:
                    continue
                t = time.perf_counter()
                first = True
                async for pcm in self.tts.stream(text, self.speed):
                    header: dict[str, Any] = {"turn_id": self.turn_id, "seq": self.chunks, "rate": self.tts.sample_rate,
                                              "sentence": index}
                    if first:
                        header["text"] = text
                        first = False
                        if self.first_audio_at is None:
                            self.first_audio_at = time.perf_counter()
                            self.first_synth_ms = round((self.first_audio_at - t) * 1000, 1)
                    await self.send_audio(header, pcm)
                    self.chunks += 1
                self.spoken.append(text)
                index += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the text answer still arrives; say the voice failed
            logger.warning("speech: TTS failed: %s", exc)
            self.error = str(exc)[:200]
            await self.emit({"type": "audio_error", "turn_id": self.turn_id, "message": "The server voice failed."})
        await self.emit({"type": "audio_end", "turn_id": self.turn_id, "chunks": self.chunks, "sentences": index,
                         "skipped": self.skipped})
