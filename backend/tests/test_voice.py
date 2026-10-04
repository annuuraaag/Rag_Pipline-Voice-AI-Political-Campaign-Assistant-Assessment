"""Voice path: partial-transcript controller (S0–S3), WebSocket protocol, barge-in, /transcribe."""
import asyncio
import json

import httpx
import pytest

from app.voice.controller import PartialTranscriptController, normalise
from app.voice.transcriber import WhisperTranscriber, vocabulary_prompt
from tests.conftest import sample

QUESTION = "what healthcare initiatives are proposed for Vijayawada"


def seed(container) -> None:
    container.ingestion.ingest(sample("districts/vijayawada_district_plan.pdf"), "vijayawada_district_plan.pdf")
    container.ingestion.ingest(sample("faq/campaign_faq.md"), "campaign_faq.md")


def controller(container, events=None, **kw) -> PartialTranscriptController:
    async def emit(e):
        if events is not None:
            events.append(e)
    defaults = dict(debounce_s=0.01, stable_s=0.05)
    return PartialTranscriptController(container.rag, "default", "s1", revision=lambda: container.ingestion.revision,
                                       emit=emit, **{**defaults, **kw})


async def speak(ctrl, sentence: str, gap_s: float = 0.0) -> None:
    words = sentence.split()
    for i in range(1, len(words) + 1):
        await ctrl.on_partial(" ".join(words[:i]))
        if gap_s:
            await asyncio.sleep(gap_s)


def test_key_normalises_case_and_punctuation():
    assert normalise("What about Hospitals?") == normalise("what about hospitals")


def test_short_partials_and_small_talk_are_ignored(container):
    seed(container)

    async def run():
        events = []
        ctrl = controller(container, events)
        await ctrl.on_partial("what")
        await ctrl.on_partial("what health")
        await ctrl.on_partial("thank you")
        await asyncio.sleep(0.1)
        return ctrl, events

    ctrl, events = asyncio.run(run())
    assert ctrl.stats.stage1_runs == 0 and ctrl.stats.ignored >= 2
    assert not [e for e in events if e["stage"] == "S1"]


def test_partials_trigger_debounced_stage1_then_refine_and_final_hits_cache(container):
    seed(container)

    async def run():
        events = []
        ctrl = controller(container, events)
        await speak(ctrl, QUESTION)          # partials arrive faster than the debounce: few searches
        await asyncio.sleep(0.2)             # transcript stable → S2 rerank/gate ahead of time
        spec = await ctrl.reuse(*_final_plan(ctrl, QUESTION))
        return ctrl, events, spec

    ctrl, events, spec = asyncio.run(run())
    stages = [e["stage"] for e in events]
    assert "S1" in stages and "S2" in stages
    assert ctrl.stats.stage1_runs <= 3                       # debounced, not once per word
    assert spec is not None and spec.answerable              # a full RetrievalResult is ready before "final"


def _final_plan(ctrl, text):
    rw, filters, _ = ctrl.plan(text)
    return rw.query, filters


def test_different_final_wording_is_a_miss_and_uploads_invalidate(container):
    seed(container)

    async def run():
        ctrl = controller(container)
        await speak(ctrl, QUESTION)
        await ctrl.speech_end()
        assert await ctrl.reuse(*_final_plan(ctrl, QUESTION)) is not None
        assert await ctrl.reuse(*_final_plan(ctrl, "who is eligible for the Pratibha Scholarship")) is None
        container.ingestion.ingest(sample("schemes/education_schemes.txt"), "education_schemes.txt")
        return await ctrl.reuse(*_final_plan(ctrl, QUESTION))   # corpus revision changed

    assert asyncio.run(run()) is None


# ── WebSocket protocol ───────────────────────────────────────────────────
def _collect(ws, until="done", limit=500):
    out = []
    for _ in range(limit):
        e = ws.receive_json()
        out.append(e)
        if e["type"] in (until, "error"):
            break
    return out


def test_ws_full_turn_reuses_speculative_retrieval(client, container):
    seed(container)
    with client.websocket_connect("/ws/voice") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "start", "session_id": "voice-1"})
        assert ws.receive_json()["type"] == "started"
        words = QUESTION.split()
        for i in range(1, len(words) + 1):
            ws.send_json({"type": "partial", "text": " ".join(words[:i])})
        ws.send_json({"type": "speech_end"})
        spec = []
        while len([e for e in spec if e.get("stage") == "S2"]) == 0:   # wait for the refine to land
            spec.append(ws.receive_json())
        ws.send_json({"type": "final", "text": QUESTION.capitalize() + "?", "turn_id": "t1"})
        events = _collect(ws)
    retrieval = next(e for e in events if e["type"] == "retrieval")
    done = events[-1]
    assert retrieval["cache"] == "hit" and retrieval["turn_id"] == "t1"
    assert done["type"] == "done" and done["answerable"] and done["citations"]
    assert done["voice"]["cache"] == "hit" and done["voice"]["retrieval_saved_ms"] > 0
    assert done["voice"]["final_to_first_token_ms"] is not None
    assert done["conversation"]["district"] == "vijayawada"       # memory committed once, on the final


def test_ws_without_partials_still_answers(client, container):
    seed(container)
    with client.websocket_connect("/ws/voice") as ws:
        ws.receive_json()
        ws.send_json({"type": "final", "text": "What is the Sunrise Coast Alliance?"})
        events = _collect(ws)
    assert next(e for e in events if e["type"] == "retrieval")["cache"] == "miss"
    assert events[-1]["type"] == "done"


def test_ws_barge_in_cancels_the_answer(client, container, fake_llm):
    seed(container)

    async def slow_stream(messages, max_tokens=None, grounding=None):
        for i in range(200):
            await asyncio.sleep(0.01)
            yield f"word{i} "
    fake_llm.stream = slow_stream
    with client.websocket_connect("/ws/voice") as ws:
        ws.receive_json()
        ws.send_json({"type": "final", "text": QUESTION})
        while ws.receive_json()["type"] != "token":
            pass
        ws.send_json({"type": "cancel"})
        rest = _collect(ws, until="cancelled")
    assert rest[-1]["type"] == "cancelled" and not any(e["type"] == "done" for e in rest)


def test_ws_rejects_bad_input_without_dropping_the_connection(client, container):
    with client.websocket_connect("/ws/voice") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "campaign_id": "Bad Name!"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "nonsense"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "ping", "t": 1})
        assert ws.receive_json() == {"type": "pong", "t": 1}


def test_ws_checks_origin_when_cors_is_restricted(settings, container):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from app.main import create_app
    container.settings.cors_origins = "http://localhost:5173"
    with TestClient(create_app(settings, container)) as tc:
        with pytest.raises(WebSocketDisconnect):
            with tc.websocket_connect("/ws/voice", headers={"origin": "https://evil.example"}) as ws:
                ws.receive_json()
        with tc.websocket_connect("/ws/voice", headers={"origin": "http://localhost:5173"}) as ws:
            assert ws.receive_json()["type"] == "ready"


# ── /transcribe ──────────────────────────────────────────────────────────
def test_transcribe_without_provider_is_503(client):
    r = client.post("/transcribe", files={"file": ("a.webm", b"\x1a\x45\xdf\xa3", "audio/webm")})
    assert r.status_code == 503 and "Chrome" in r.json()["detail"]


def test_transcribe_sends_vocabulary_and_language(client, container):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        body = req.content.decode(errors="ignore")
        seen["prompt_has_place"] = "Vijayawada" in body
        seen["language_en"] = 'name="language"\r\n\r\nen\r\n' in body
        return httpx.Response(200, json={"text": " I live in Vijayawada, any new hospitals? "})

    container.transcriber = WhisperTranscriber("groq", "https://x/openai/v1", "k", transport=httpx.MockTransport(handler))
    r = client.post("/transcribe", data={"language": "en-IN"},
                    files={"file": ("clip.webm", b"\x1a\x45\xdf\xa3fake", "audio/webm")})
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "I live in Vijayawada, any new hospitals?"
    assert seen == {"prompt_has_place": True, "language_en": True}
    assert "Pratibha" not in vocabulary_prompt() and "Guntur" in vocabulary_prompt(["Pratibha"]) \
        and "Pratibha" in vocabulary_prompt(["Pratibha"])


def test_health_reports_voice(client):
    voice = client.get("/health").json()["components"]["voice"]
    assert voice["websocket"] == "/ws/voice" and voice["speculation"]["debounce_ms"] == 250
    assert json.dumps(voice)
