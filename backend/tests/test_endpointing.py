"""End-of-turn detection: how long a silence ends the user's turn, from the transcript alone."""
import asyncio

import pytest

from app.conversation.state import ConversationState
from app.conversation.understanding import understand
from app.voice.controller import PartialTranscriptController
from app.voice.endpointing import Endpointer
from tests.conftest import sample

E = Endpointer()


def with_history(*utterances: str) -> ConversationState:
    st = ConversationState("s")
    for u in utterances:
        st.observe_user(understand(u), u)
    return st


@pytest.mark.parametrize("text", [
    "what healthcare initiatives does the candidate propose for vijayawada",
    "what is the lean season allowance for fishermen",
    "what support is offered to chilli farmers",
    "uh so what are you guys doing for um farmers near guntur",
    "tell me about the pratibha scholarship",
    "compare the new hospital plans for guntur and visakhapatnam",
    "I'm from Vijayawada",
    "thank you",
])
def test_finished_questions_end_fast(text):
    assert E.assess(text).turn == "complete"


@pytest.mark.parametrize("text", [
    "what is planned for chilli farmers in",     # dangling preposition
    "what are the",                              # article
    "how many",                                  # opener without its noun
    "what healthcare initiatives does",          # auxiliary
    "what is the plan for farmers um",           # trailing filler
    "I live in bezawada, any",                   # an introduction that turns into a question
    "",
])
def test_unfinished_utterances_wait_longer(text):
    r = E.assess(text)
    assert r.turn == "incomplete" and r.wait_ms > 900


def test_a_place_or_topic_word_must_end_the_phrase_not_modify_it():
    assert E.assess("what are the education").turn != "complete"          # "... plans" still to come
    assert E.assess("where is the vijayawada").turn != "complete"         # "... campaign office"
    assert E.assess("what healthcare initiatives does the candidate").turn == "unsure"  # owes a verb


def test_follow_ups_use_the_conversation():
    st = with_history("What is the Pratibha Scholarship?")
    assert E.assess("who is eligible for it", st).turn == "complete"
    assert E.assess("who is eligible for it", None).turn != "complete"   # "it" resolves to nothing
    assert E.assess("what about education", with_history("I'm from Guntur")).turn == "complete"


def test_waits_are_configurable_and_ordered():
    e = Endpointer(complete_ms=250, likely_ms=500, unsure_ms=800, incomplete_ms=2000)
    assert e.assess("what is the lean season allowance for fishermen").wait_ms == 250
    assert e.assess("what is the lean season allowance for").wait_ms == 2000
    assert E.waits["complete"] < E.waits["likely"] < E.waits["unsure"] < E.waits["incomplete"]


# ── controller and WebSocket ────────────────────────────────────────────
def test_controller_emits_endpoint_hints_and_refines_early_when_complete(container):
    container.ingestion.ingest(sample("districts/vijayawada_district_plan.pdf"), "vijayawada_district_plan.pdf")

    async def run():
        events = []

        async def emit(e):
            events.append(e)
        ctrl = PartialTranscriptController(container.rag, "default", "s1", revision=lambda: container.ingestion.revision,
                                           emit=emit, debounce_s=0.01, stable_s=5.0, endpointer=Endpointer())
        await ctrl.on_partial("what healthcare initiatives are proposed for")
        await ctrl.on_partial("what healthcare initiatives are proposed for vijayawada")
        await asyncio.sleep(0.4)   # far less than stable_s: only the "complete" verdict can trigger S2
        return ctrl, events

    ctrl, events = asyncio.run(run())
    hints = [e for e in events if e["type"] == "endpoint"]
    assert [h["turn"] for h in hints] == ["incomplete", "complete"]
    assert hints[1]["wait_ms"] == 300 and hints[1]["text"].endswith("vijayawada")
    assert ctrl.stats.stage2_runs == 1 and ctrl.end_of_turn.turn == "complete"


def test_ws_sends_an_endpoint_hint_per_partial(client):
    with client.websocket_connect("/ws/voice") as ws:
        ready = ws.receive_json()
        assert ready["endpointing"]["adaptive"] and ready["endpointing"]["wait_ms"]["complete"] == 300
        ws.send_json({"type": "partial", "text": "what is planned for farmers in"})
        hint = ws.receive_json()
        assert hint["type"] == "endpoint" and hint["turn"] == "incomplete" and hint["wait_ms"] == 1600


def test_adaptive_endpointing_can_be_switched_off(settings, container):
    from fastapi.testclient import TestClient

    from app.main import create_app
    container.settings.voice_adaptive_endpointing = False
    with TestClient(create_app(settings, container)) as tc, tc.websocket_connect("/ws/voice") as ws:
        assert ws.receive_json()["endpointing"]["adaptive"] is False
        ws.send_json({"type": "partial", "text": "what is planned for farmers in"})
        ws.send_json({"type": "ping", "t": 7})
        assert ws.receive_json() == {"type": "pong", "t": 7}   # no hint was sent first
