"""Voice endpoints.

WS /ws/voice — real-time voice turns with speculative retrieval on partial transcripts.

Client → server (JSON):
    {"type": "start", "campaign_id"?, "session_id"?, "filters"?}   configure the connection
    {"type": "partial", "text"}                                     interim transcript (any rate)
    {"type": "speech_end"}                                          optional: user stopped talking
    {"type": "final", "text", "turn_id"?}                           final transcript → answer
    {"type": "cancel"}                                              barge-in: stop the current answer
    {"type": "ping"}

Server → client (JSON):
    ready · started · speculative (stage S1|S2, sources) · retrieval · token · done · cancelled ·
    pong · error. `retrieval`/`token`/`done` carry the client's `turn_id`; `done.voice` reports the
    cache outcome, the retrieval time saved and final-transcript → first-token latency.

POST /transcribe — audio clip → text (Whisper), for browsers without the Web Speech API.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.api.deps import get_container, resolve_campaign
from app.container import Container
from app.generation.llm.base import LLMError
from app.ingestion.service import normalize_campaign
from app.schemas import FilterIn
from app.voice.controller import PartialTranscriptController

router = APIRouter(tags=["voice"])
logger = logging.getLogger(__name__)

MAX_TEXT = 1000


def _origin_allowed(ws: WebSocket, c: Container) -> bool:
    """Browsers don't apply CORS to WebSockets, so check Origin here (same policy as HTTP)."""
    origins = c.settings.cors_allow_origins()
    origin = ws.headers.get("origin")
    return not origin or "*" in origins or origin in origins


class VoiceSession:
    def __init__(self, ws: WebSocket, c: Container):
        self.ws = ws
        self.c = c
        self._send_lock = asyncio.Lock()
        self.ctrl: PartialTranscriptController | None = None
        self.answer: asyncio.Task | None = None

    async def send(self, event: dict) -> None:
        async with self._send_lock:
            await self.ws.send_json(event)

    def configure(self, msg: dict) -> PartialTranscriptController:
        s = self.c.settings
        campaign = normalize_campaign(msg.get("campaign_id") or s.default_campaign_id)
        filters = FilterIn(**msg["filters"]).to_domain() if msg.get("filters") else None
        session_id = str(msg.get("session_id") or uuid.uuid4().hex)[:100]
        self.ctrl = PartialTranscriptController(
            self.c.rag, campaign, session_id, filters, revision=lambda: self.c.ingestion.revision, emit=self.send,
            debounce_s=s.voice_debounce_ms / 1000, stable_s=s.voice_stable_ms / 1000,
            min_words=s.voice_min_words, word_step=s.voice_word_step,
        )
        return self.ctrl

    async def cancel_answer(self) -> bool:
        task, self.answer = self.answer, None
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            return True
        return False

    async def run_answer(self, text: str, turn_id: str) -> None:
        ctrl = self.ctrl
        assert ctrl is not None
        t_final = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        first_token_ms: float | None = None
        rewritten, filters = None, None
        try:
            stream = self.c.rag.stream(text, ctrl.explicit, request_id=request_id, session_id=ctrl.session_id,
                                       campaign_id=ctrl.campaign_id, reuse=ctrl.reuse)
            async for event in stream:
                event["turn_id"] = turn_id
                if event["type"] == "retrieval":
                    rewritten = event["retrieval"]["rewritten_query"]
                    filters = event["retrieval"]["filters"]
                elif event["type"] == "token" and first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - t_final) * 1000, 2)
                elif event["type"] == "done":
                    from app.domain import MetadataFilter

                    saved = ctrl.saved_ms(rewritten, MetadataFilter(**filters), event.get("cache")) \
                        if rewritten is not None and filters is not None else 0.0
                    event["voice"] = {
                        "cache": event.get("cache"), "retrieval_saved_ms": saved,
                        "final_to_first_token_ms": first_token_ms,
                        "partials": ctrl.stats.partials, "ignored_partials": ctrl.stats.ignored,
                        "speculative_searches": ctrl.stats.stage1_runs, "speculative_refines": ctrl.stats.stage2_runs,
                    }
                    self.c.metrics.record("voice", {"final_to_first_token": first_token_ms or 0.0,
                                                    "retrieval_saved": saved})
                await self.send(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Voice answer failed request_id=%s", request_id)
            await self.send({"type": "error", "turn_id": turn_id, "request_id": request_id,
                             "message": f"Something went wrong while answering. Reference: {request_id}"})


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
    })
    try:
        while True:
            try:
                msg = await ws.receive_json()
            except (ValueError, KeyError):
                await vs.send({"type": "error", "message": "Messages must be JSON objects."})
                continue
            if not isinstance(msg, dict):
                continue
            kind = msg.get("type")
            text = str(msg.get("text") or "")[:MAX_TEXT]
            try:
                if kind == "start" or vs.ctrl is None and kind in ("partial", "final", "speech_end"):
                    ctrl = vs.configure(msg if kind == "start" else {})
                    if kind == "start":
                        await vs.send({"type": "started", "campaign_id": ctrl.campaign_id,
                                       "session_id": ctrl.session_id})
                        continue
                if kind == "partial":
                    await vs.ctrl.on_partial(text)
                elif kind == "speech_end":
                    asyncio.create_task(vs.ctrl.speech_end())
                elif kind == "final":
                    if not text.strip():
                        continue
                    await vs.cancel_answer()  # a new utterance supersedes an unfinished answer
                    vs.ctrl.reset_turn()       # next partials belong to the next utterance (cache is kept)
                    turn_id = str(msg.get("turn_id") or uuid.uuid4().hex[:8])[:64]
                    vs.answer = asyncio.create_task(vs.run_answer(text.strip(), turn_id))
                elif kind == "cancel":
                    if await vs.cancel_answer():
                        await vs.send({"type": "cancelled"})
                elif kind == "ping":
                    await vs.send({"type": "pong", "t": msg.get("t")})
                else:
                    await vs.send({"type": "error", "message": f"Unknown message type: {kind!r}"})
            except (ValueError, ValidationError) as exc:  # bad campaign id / filters
                await vs.send({"type": "error", "message": str(exc)[:300]})
    except WebSocketDisconnect:
        pass
    finally:
        await vs.cancel_answer()
        if vs.ctrl:
            vs.ctrl.cancel_timers()


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
