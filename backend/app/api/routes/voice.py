"""Voice endpoints.

WS /ws/voice — real-time voice turns: speculative retrieval on partial transcripts, end-of-turn
hints, and, when the server has speech providers (STT_PROVIDER / TTS_PROVIDER), microphone audio
in and spoken answers out (app/voice/speech.py).

Client → server, JSON:
    {"type": "start", "campaign_id"?, "session_id"?, "filters"?,
     "speech"?: {"input": "server"|"browser", "output": "server"|"browser", "rate"?, "language"?}}
    {"type": "partial", "text"}                       interim transcript from the browser's recogniser
    {"type": "speech_end"}                            optional: the browser heard the user stop
    {"type": "final", "text", "turn_id"?, "speak"?}   final transcript or typed question → answer;
                                                      speak: also speak it with the server voice
    {"type": "audio_start"} | {"type": "audio_stop"} | {"type": "audio_cancel"}
                                                      server recognition: open / end now / drop a question
    {"type": "cancel"}                                stop the current answer and its audio
    {"type": "playback", "playing": bool}             the browser started / finished speaking an answer
                                                      (barge-in is watched for until it finishes)
    {"type": "ping"}
Client → server, binary: microphone audio, 16 kHz PCM16 mono: after audio_start, and while an
    answer plays in conversation mode, so the server can hear the user interrupt.

Server → client, JSON:
    ready · started · transcript (server recognition, partial) · utterance (final transcript; it
    opens a turn and the answer's events carry its turn_id) · endpoint · speculative · retrieval ·
    token · done · audio_end · audio_error · voice_metrics · barge_in · cancelled · pong · error.
    `endpoint` follows every partial: how finished the utterance sounds (complete | likely | unsure
    | incomplete) and the silence (`wait_ms`) that should end the turn. `done.voice` reports the
    cache outcome, the retrieval time saved and final-transcript → first-token latency;
    `voice_metrics` (server voice) adds first-audio and last-word → first-audio latency.
Server → client, binary: the spoken answer. Each frame: uint32 LE header length, a JSON header
    {"turn_id", "seq", "rate", "sentence", "text"?}, then PCM16 mono at `rate` Hz.

POST /transcribe — audio clip → text (Whisper), for browsers without the Web Speech API.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import struct
import time
import uuid
from collections import deque

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.api.deps import get_container, resolve_campaign
from app.container import Container
from app.domain import MetadataFilter, ScoredChunk
from app.generation.llm.base import LLMError
from app.ingestion.service import normalize_campaign
from app.schemas import FilterIn
from app.voice.controller import PartialTranscriptController
from app.voice.endpointing import build_endpointer, endpointing_info
from app.voice.speech import AnswerSpeaker, SpeechHooks, SpeechInput

router = APIRouter(tags=["voice"])
logger = logging.getLogger(__name__)

MAX_TEXT = 1000
MAX_AUDIO_FRAME = 64 * 1024   # bytes per binary message (2 s of 16 kHz PCM16)


def _origin_allowed(ws: WebSocket, c: Container) -> bool:
    """Browsers don't apply CORS to WebSockets, so check Origin here (same policy as HTTP)."""
    origins = c.settings.cors_allow_origins()
    origin = ws.headers.get("origin")
    return not origin or "*" in origins or origin in origins


def speech_info(c: Container) -> dict:
    stt, tts = c.stt, c.tts
    return {
        "stt": {"provider": stt.name, "model": stt.model, "streaming": stt.streaming} if stt else None,
        "tts": {"provider": tts.name, "voice": tts.voice, "sample_rate": tts.sample_rate} if tts else None,
        "vad": c.vad.name,
    }


class VoiceSession:
    def __init__(self, ws: WebSocket, c: Container):
        self.ws = ws
        self.c = c
        self._send_lock = asyncio.Lock()
        self.ctrl: PartialTranscriptController | None = None
        self.answer: asyncio.Task | None = None
        self.answer_turn: str | None = None
        self.answer_text = ""                 # what the current answer says so far (echo check)
        self.speech: SpeechInput | None = None
        self.speaker: AnswerSpeaker | None = None
        self.speak_server = False             # the client asked for the server voice
        self.playing_until = 0.0              # the client is still playing the answer until then (estimate)
        self.playback: deque[list] = deque(maxlen=24)   # [start, end, text] of each sentence as it is heard
        self.client_playing = False           # ... or says so (browser voice, or until it reports the end)
        self.rate = 1.0
        self.language = "en"
        self._audio_warned = False

    async def send(self, event: dict) -> None:
        async with self._send_lock:
            await self.ws.send_json(event)

    async def send_audio(self, header: dict, pcm: bytes) -> None:
        h = json.dumps(header, separators=(",", ":")).encode()
        if len(h) % 2:
            h += b" "   # keeps the PCM 2-byte aligned for the client's Int16Array view
        async with self._send_lock:
            await self.ws.send_bytes(struct.pack("<I", len(h)) + h + pcm)
        # Synthesis runs far ahead of playback: the answer is still being heard after the last frame
        # is sent, and that is when the user may talk over it. The client plays frames back to back.
        start = max(self.playing_until, time.perf_counter())
        self.playing_until = start + len(pcm) / 2 / header["rate"]
        if "text" in header:
            self.playback.append([start, self.playing_until, header["text"]])
        elif self.playback:
            self.playback[-1][1] = self.playing_until

    def spoken_text(self) -> str:
        """What the user is hearing about now: echo can only come from the last few seconds."""
        now = time.perf_counter()
        playing = [text for start, end, text in self.playback if start <= now + 0.5 and end >= now - 3.0]
        if playing:
            return " ".join(playing)
        return self.speaker.recent_text() if self.speaker else self.answer_text[-400:]

    def answering(self) -> bool:
        """An answer is being generated, or is still being heard."""
        busy = self.answer is not None and not self.answer.done()
        return busy or self.client_playing or time.perf_counter() < self.playing_until + 0.3

    async def configure(self, msg: dict) -> PartialTranscriptController:
        s = self.c.settings
        campaign = normalize_campaign(msg.get("campaign_id") or s.default_campaign_id)
        filters = FilterIn(**msg["filters"]).to_domain() if msg.get("filters") else None
        session_id = str(msg.get("session_id") or uuid.uuid4().hex)[:100]
        self.ctrl = PartialTranscriptController(
            self.c.rag, campaign, session_id, filters, revision=lambda: self.c.ingestion.revision, emit=self.send,
            debounce_s=s.voice_debounce_ms / 1000, stable_s=s.voice_stable_ms / 1000,
            min_words=s.voice_min_words, word_step=s.voice_word_step, endpointer=build_endpointer(s),
        )
        sp = msg.get("speech") if isinstance(msg.get("speech"), dict) else {}
        self.language = str(sp.get("language") or "en")[:12]
        try:
            self.rate = min(max(float(sp.get("rate") or 1.0), 0.5), 2.0)
        except (TypeError, ValueError):
            self.rate = 1.0
        self.speak_server = sp.get("output") == "server" and self.c.tts is not None
        if self.speech is not None:
            await self.speech.close()
            self.speech = None
        if sp.get("input") == "server" and self.c.stt is not None:
            corrector = self.c.vocabulary.get(campaign)
            titles = [d.title for d in self.c.registry.all(campaign) if d.title][:10]
            names = [n for n in corrector.names if len(n.split()) > 1][:30] + titles
            self.speech = SpeechInput(
                self.c.stt, self.c.vad(), SpeechHooks(
                    emit=self.send, assess=lambda t: self.ctrl.assess_turn(t) if self.ctrl else None,
                    on_partial=lambda t: self.ctrl.on_partial(t), on_utterance=self._utterance,
                    answering=self.answering,
                    spoken_text=self.spoken_text,
                    on_barge_in=self._barge_in),
                language=self.language, vocabulary=names, correct=corrector.correct,
                unsure_wait_s=s.voice_endpoint_unsure_ms / 1000)
        return self.ctrl

    def speech_state(self) -> dict:
        return {"input": "server" if self.speech else "browser", "output": "server" if self.speak_server else "browser"}

    # ── answers ────────────────────────────────────────────────────────
    async def start_answer(self, text: str, turn_id: str, *, speak: bool, voice_info: dict | None = None) -> None:
        await self.cancel_answer()   # a new utterance supersedes an unfinished answer
        assert self.ctrl is not None
        self.ctrl.reset_turn()       # next partials belong to the next utterance (cache is kept)
        self.answer_turn = turn_id
        self.answer = asyncio.create_task(self.run_answer(text, turn_id, speak=speak, voice_info=voice_info))

    async def cancel_answer(self) -> bool:
        """Stop the answer; True if it was still being generated or heard."""
        task, self.answer = self.answer, None
        heard = self.client_playing or time.perf_counter() < self.playing_until
        self.playing_until, self.client_playing = 0.0, False
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            return True
        return heard

    async def _utterance(self, text: str, info: dict) -> None:
        """Server recognition decided the question is over: it becomes a turn."""
        turn_id = uuid.uuid4().hex[:8]
        public = {k: v for k, v in info.items() if k != "decided_wall"}
        await self.send({"type": "utterance", "text": text, "turn_id": turn_id, **public})
        await self.start_answer(text, turn_id, speak=self.speak_server, voice_info=info)

    async def _barge_in(self) -> None:
        turn = self.answer_turn
        if await self.cancel_answer():
            await self.send({"type": "barge_in", "turn_id": turn})

    def _gate(self, question: str, sources: list[ScoredChunk]):
        """CITATION_VERIFICATION=strict: a sentence is spoken only if its sources support it."""
        verifier = self.c.rag.verifier
        if verifier is None or self.c.rag.verify_policy != "strict":
            return None

        async def gate(sentence: str) -> bool:
            check = verifier.check_sentence(sentence, sources, question)
            return check is None or check.verdict != "unsupported"
        return gate

    async def run_answer(self, text: str, turn_id: str, *, speak: bool = False, voice_info: dict | None = None) -> None:
        ctrl = self.ctrl
        assert ctrl is not None
        t_final = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        first_token_ms: float | None = None
        rewritten, filters = None, None
        sources: list[ScoredChunk] = []
        speaker = None
        if speak and self.c.tts is not None:
            speaker = AnswerSpeaker(self.c.tts, self.send_audio, self.send, turn_id, speed=self.rate,
                                    gate=self._gate(text, sources))
        self.speaker, self.answer_text = speaker, ""
        try:
            stream = self.c.rag.stream(text, ctrl.explicit, request_id=request_id, session_id=ctrl.session_id,
                                       campaign_id=ctrl.campaign_id, reuse=ctrl.reuse, sources_out=sources)
            async for event in stream:
                event["turn_id"] = turn_id
                if event["type"] == "retrieval":
                    rewritten = event["retrieval"]["rewritten_query"]
                    filters = event["retrieval"]["filters"]
                elif event["type"] == "token":
                    if first_token_ms is None:
                        first_token_ms = round((time.perf_counter() - t_final) * 1000, 2)
                    self.answer_text += event["text"]
                    if speaker is not None:
                        speaker.push(event["text"])
                elif event["type"] == "done":
                    saved = ctrl.saved_ms(rewritten, MetadataFilter(**filters), event.get("cache")) \
                        if rewritten is not None and filters is not None else 0.0
                    event["voice"] = {
                        "cache": event.get("cache"), "retrieval_saved_ms": saved,
                        "final_to_first_token_ms": first_token_ms,
                        "partials": ctrl.stats.partials, "ignored_partials": ctrl.stats.ignored,
                        "speculative_searches": ctrl.stats.stage1_runs, "speculative_refines": ctrl.stats.stage2_runs,
                        "server_voice": speaker is not None,
                        **({"endpoint_wait_ms": voice_info["endpoint_ms"], "stt_final_ms": voice_info["stt_final_ms"]}
                           if voice_info else {}),
                    }
                    self.c.metrics.record("voice", {"final_to_first_token": first_token_ms or 0.0,
                                                    "retrieval_saved": saved})
                    if speaker is not None:
                        speaker.end()
                await self.send(event)
            if speaker is not None:
                await speaker.wait()
                await self.send(self._voice_metrics(turn_id, t_final, speaker, voice_info))
        except asyncio.CancelledError:
            if speaker is not None:
                await speaker.cancel()
            raise
        except Exception:
            if speaker is not None:
                await speaker.cancel()
            logger.exception("Voice answer failed request_id=%s", request_id)
            await self.send({"type": "error", "turn_id": turn_id, "request_id": request_id,
                             "message": f"Something went wrong while answering. Reference: {request_id}"})

    def _voice_metrics(self, turn_id: str, t_final: float, speaker: AnswerSpeaker, info: dict | None) -> dict:
        def ms(a: float | None, b: float | None) -> float | None:
            return round((a - b) * 1000, 1) if a is not None and b else None
        first = speaker.first_audio_at
        m = {"final_to_first_audio_ms": ms(first, t_final), "first_synth_ms": speaker.first_synth_ms,
             "sentences": len(speaker.spoken), "skipped_unverified": speaker.skipped}
        if info:
            # Last voiced audio → first answer audio sent: the whole voice pipeline. The silence that
            # ended the turn is measured on the audio clock, the rest on the wall clock, so the
            # number holds however fast the audio arrived (network and playback buffering excluded).
            after = ms(first, info.get("decided_wall"))
            m.update(endpoint_wait_ms=info.get("endpoint_ms"), stt_final_ms=info.get("stt_final_ms"),
                     last_voice_to_first_audio_ms=round(info["endpoint_ms"] + after, 1) if after is not None else None)
        self.c.metrics.record("voice_audio", {k: v for k, v in m.items() if isinstance(v, int | float) and v is not None
                                              and k.endswith("_ms")})
        return {"type": "voice_metrics", "turn_id": turn_id, **m}

    # ── audio ──────────────────────────────────────────────────────────
    async def on_audio(self, pcm: bytes) -> None:
        if self.speech is None:
            if not self._audio_warned:
                self._audio_warned = True
                await self.send({"type": "error", "message": "Server speech recognition is off (STT_PROVIDER=none); "
                                                             "send transcripts as JSON instead."})
            return
        if len(pcm) > MAX_AUDIO_FRAME:
            return
        await self.speech.on_audio(pcm)

    async def close(self) -> None:
        await self.cancel_answer()
        if self.speech is not None:
            await self.speech.close()
        if self.ctrl:
            self.ctrl.cancel_timers()


async def _handle(vs: VoiceSession, msg: dict) -> None:
    kind = msg.get("type")
    text = str(msg.get("text") or "")[:MAX_TEXT]
    if kind == "start" or vs.ctrl is None and kind in ("partial", "final", "speech_end", "audio_start"):
        ctrl = await vs.configure(msg if kind == "start" else {})
        if kind == "start":
            await vs.send({"type": "started", "campaign_id": ctrl.campaign_id, "session_id": ctrl.session_id,
                           "speech": vs.speech_state()})
            return
    if kind == "partial":
        await vs.ctrl.on_partial(text)
    elif kind == "speech_end":
        asyncio.create_task(vs.ctrl.speech_end())
    elif kind == "final":
        if text.strip():
            turn_id = str(msg.get("turn_id") or uuid.uuid4().hex[:8])[:64]
            await vs.start_answer(text.strip(), turn_id, speak=bool(msg.get("speak")) and vs.c.tts is not None)
    elif kind in ("audio_start", "audio_stop", "audio_cancel"):
        if vs.speech is None:
            await vs.send({"type": "error", "message": "Server speech recognition is not enabled for this connection."})
        elif kind == "audio_start":
            await vs.speech.start_turn()
        elif kind == "audio_stop":
            await vs.speech.stop_turn()
        else:
            await vs.speech.cancel_turn()
    elif kind == "cancel":
        if await vs.cancel_answer():
            await vs.send({"type": "cancelled"})
    elif kind == "playback":
        vs.client_playing = bool(msg.get("playing"))
        if not vs.client_playing:
            vs.playing_until = 0.0
    elif kind == "ping":
        await vs.send({"type": "pong", "t": msg.get("t")})
    else:
        await vs.send({"type": "error", "message": f"Unknown message type: {kind!r}"})


@router.websocket("/ws/voice")
async def voice_ws(ws: WebSocket) -> None:
    c: Container = ws.app.state.container
    if not _origin_allowed(ws, c):
        await ws.close(code=1008)
        return
    await ws.accept()
    vs = VoiceSession(ws, c)
    await vs.send({
        "type": "ready", "llm": {"provider": c.llm.name, "model": c.llm.model},
        "reranker": c.retriever.reranker is not None, "transcribe": c.transcriber is not None,
        "speculation": {"debounce_ms": c.settings.voice_debounce_ms, "stable_ms": c.settings.voice_stable_ms,
                        "min_words": c.settings.voice_min_words},
        "endpointing": endpointing_info(c.settings),
        "speech": speech_info(c),
    })
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await vs.on_audio(message["bytes"])
                continue
            try:
                msg = json.loads(message.get("text") or "")
            except ValueError:
                await vs.send({"type": "error", "message": "Messages must be JSON objects."})
                continue
            if not isinstance(msg, dict):
                continue
            try:
                await _handle(vs, msg)
            except (ValueError, ValidationError) as exc:  # bad campaign id / filters
                await vs.send({"type": "error", "message": str(exc)[:300]})
    except WebSocketDisconnect:
        pass
    finally:
        await vs.close()


@router.post("/transcribe", summary="Speech-to-text for browsers without built-in speech recognition")
async def transcribe(
    file: UploadFile = File(..., description="Audio clip (webm/ogg/wav/mp3/m4a), up to MAX_AUDIO_MB."),
    language: str | None = Form(default="en", description="BCP-47 or ISO-639-1, e.g. en-IN or en."),
    campaign_id: str | None = Form(default=None),
    c: Container = Depends(get_container),
) -> dict:
    if c.transcriber is None:
        raise HTTPException(status_code=503, detail="Server speech-to-text needs an LLM provider key (GROQ_API_KEY). "
                                                    "Chrome, Edge and Safari have built-in speech recognition.")
    data = await file.read(c.settings.max_audio_mb * 1024 * 1024 + 1)
    if len(data) > c.settings.max_audio_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Audio larger than {c.settings.max_audio_mb} MB.")
    if not data:
        raise HTTPException(status_code=422, detail="Empty audio.")
    campaign = resolve_campaign(campaign_id, c)
    titles = [d.title for d in c.registry.all(campaign) if d.title][:20]
    try:
        return await c.transcriber.transcribe(data, file.filename or "audio.webm", file.content_type or "",
                                              language=language, vocabulary=titles)
    except LLMError as exc:
        logger.warning("Transcription failed: %s", exc)
        raise HTTPException(status_code=502, detail="Speech-to-text provider failed; please try again.") from exc
