"""Server-side speech: microphone audio over /ws/voice, end of turn, barge-in, spoken answers.

Fake recogniser and voice: the fake "hears" one scripted word per 0.25 s of loud audio (and, like
real streaming models, its partial transcripts trail by two words). The audio pipeline, VAD,
end-of-turn rules, WebSocket protocol and RAG answer are the real code.
"""
import asyncio
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from app.voice.audio import STT_RATE, float_to_pcm16, split_sentences
from app.voice.speech import AnswerSpeaker, SpeechHooks, SpeechInput, is_echo
from app.voice.vad import EnergyVAD
from tests.conftest import sample

QUESTION = "what healthcare initiatives are proposed for vijayawada"
WORD_S = 0.25


def tone(seconds: float, amp: float = 0.3) -> bytes:
    t = np.arange(int(seconds * STT_RATE)) / STT_RATE
    return float_to_pcm16(amp * np.sin(2 * np.pi * 220 * t))


def silence(seconds: float) -> bytes:
    return float_to_pcm16(np.zeros(int(seconds * STT_RATE), dtype=np.float32))


def chunks(pcm: bytes, ms: int = 40) -> list[bytes]:
    step = STT_RATE * ms // 1000 * 2
    return [pcm[i:i + step] for i in range(0, len(pcm), step)]


class FakeSTT:
    name, model, streaming = "fake", "scripted", True

    def __init__(self, text: str = QUESTION):
        self.text = text
        self.opened = 0

    async def open(self, on_partial, language="en", vocabulary=None):
        self.opened += 1
        return FakeStream(self.text.split(), on_partial)


class FakeStream:
    def __init__(self, words, on_partial):
        self.words, self.on_partial = words, on_partial
        self.loud = 0.0

    def heard(self, lag: int = 0) -> str:
        n = int(self.loud / WORD_S + 1e-9) - lag
        return " ".join(self.words[:max(0, min(n, len(self.words)))])

    async def send(self, pcm: bytes) -> None:
        x = np.frombuffer(pcm, dtype="<i2")
        if len(x) and np.abs(x).mean() > 1000:
            self.loud += len(x) / STT_RATE
            if self.heard(lag=2):
                await self.on_partial(self.heard(lag=2))

    async def flush(self) -> str:
        return self.heard()

    async def finish(self) -> str:
        return self.heard()

    async def close(self) -> None:
        pass


class FakeTTS:
    name, voice, sample_rate = "fake", "test", 16000

    def __init__(self):
        self.spoken: list[str] = []

    async def stream(self, text: str, speed: float = 1.0):
        self.spoken.append(text)
        for _ in range(2):
            await asyncio.sleep(0)
            yield silence(0.1)


def speak_question(words: int, gap_after: float) -> list[bytes]:
    return chunks(silence(0.2) + tone(words * WORD_S) + silence(gap_after))


# ── unit: helpers ──────────────────────────────────────────────────────
def test_echo_check_ignores_the_assistants_own_words():
    said = "The Health Shield covers up to 15 lakh rupees per family."
    assert is_echo("health shield covers", said) and is_echo("okay", said)
    assert not is_echo("stop what about schools", said)


def test_energy_vad_follows_the_noise_floor():
    vad = EnergyVAD()
    noise = (np.random.default_rng(1).standard_normal(320) * 0.003).astype(np.float32)
    assert not any(vad.is_speech(noise) for _ in range(50))
    assert vad.is_speech(np.frombuffer(tone(0.02), dtype="<i2").astype(np.float32) / 32768)


def test_first_piece_may_end_at_a_comma_for_earlier_audio():
    out, rest = split_sentences("According to the campaign documents, the Health Shield covers", first_clause=30)
    assert out == ["According to the campaign documents,"] and rest.strip().startswith("the Health Shield")


# ── unit: SpeechInput ──────────────────────────────────────────────────
def run_input(frames, *, assess, answering=lambda: False, spoken="", text=QUESTION, sleep=0.0):
    events, partials, finals, barged = [], [], [], []

    async def go():
        async def emit(e):
            events.append(e)

        async def on_partial(t):
            partials.append(t)

        async def on_utt(t, info):
            finals.append((t, info))

        async def barge():
            barged.append(True)
        si = SpeechInput(FakeSTT(text), EnergyVAD(), SpeechHooks(
            emit=emit, assess=assess, on_partial=on_partial, on_utterance=on_utt, answering=answering,
            spoken_text=lambda: spoken, on_barge_in=barge))
        if not answering():
            await si.start_turn()
        for f in frames:
            await si.on_audio(f)
            await asyncio.sleep(sleep)
        await asyncio.sleep(0.05)
        await si.close()
        return si
    si = asyncio.run(go())
    return si, events, partials, finals, barged


def test_a_finished_question_ends_after_the_short_wait():
    from app.voice.endpointing import Endpointer
    e = Endpointer()
    _, events, partials, finals, _ = run_input(speak_question(7, gap_after=0.5), assess=e.assess)
    assert len(finals) == 1
    text, info = finals[0]
    assert text == QUESTION and info["reason"] == "silence"
    assert 300 <= info["endpoint_ms"] < 400            # "complete" waits 300 ms, not 900
    assert partials and partials[-1] == QUESTION         # the pause probe filled in the lagging words


def test_an_unfinished_question_keeps_listening_through_the_same_pause():
    from app.voice.endpointing import Endpointer
    e = Endpointer()
    _, _, _, finals, _ = run_input(speak_question(6, gap_after=1.0), assess=e.assess)   # "... proposed for"
    assert finals == []                                   # 1 s of silence < 1.6 s for "... for"
    _, _, _, finals, _ = run_input(speak_question(6, gap_after=1.8), assess=e.assess)
    assert finals and finals[0][0].endswith("proposed for")


def test_no_speech_times_out_without_a_turn():
    si, events, _, finals, _ = run_input(chunks(silence(8.3)), assess=lambda t: None)
    assert finals == [] and events[-1] == {"type": "utterance", "text": "", "reason": "no_speech"}


def test_barge_in_needs_real_words_not_echo():
    frames = chunks(silence(0.1) + tone(1.0) + silence(0.2))
    # The user says "stop what about schools" over the answer → barge-in.
    _, _, _, _, barged = run_input(frames, assess=lambda t: None, answering=lambda: True,
                                   text="stop what about schools", spoken="The plan adds a hospital.", sleep=0.002)
    assert barged == [True]
    # The microphone hears the answer itself → ignored.
    _, _, _, _, barged = run_input(frames, assess=lambda t: None, answering=lambda: True,
                                   text="the plan adds a hospital", spoken="The plan adds a hospital.", sleep=0.002)
    assert barged == []


# ── unit: AnswerSpeaker ────────────────────────────────────────────────
def test_answer_speaker_speaks_sentences_in_order_and_can_skip_unverified_ones():
    async def go(gate=None):
        tts, frames, events = FakeTTS(), [], []

        async def send_audio(h, pcm):
            frames.append(h)

        async def emit(e):
            events.append(e)
        sp = AnswerSpeaker(tts, send_audio, emit, "t1", gate=gate)
        for tok in "The Health Shield covers 15 lakh rupees [S1]. It also pays for every visit [S2]. ".split(" "):
            sp.push(tok + " ")
        sp.end()
        await sp.wait()
        return tts, frames, events

    tts, frames, events = asyncio.run(go())
    assert tts.spoken == ["The Health Shield covers 15 lakh rupees.", "It also pays for every visit."]
    assert frames[0]["text"] == tts.spoken[0] and [f["seq"] for f in frames] == [0, 1, 2, 3]
    assert events[-1]["type"] == "audio_end" and events[-1]["sentences"] == 2

    async def gate(sentence):
        return "[S2]" not in sentence
    tts, _, events = asyncio.run(go(gate))
    assert tts.spoken == ["The Health Shield covers 15 lakh rupees."] and events[-1]["skipped"] == 1


# ── through the WebSocket ──────────────────────────────────────────────
def _frames_until(ws, kinds: set[str], limit: int = 2000):
    """Read JSON events and binary audio until an event of one of `kinds` arrives."""
    events, audio = [], []
    for _ in range(limit):
        m = ws.receive()
        if m.get("bytes") is not None:
            data = m["bytes"]
            n = struct.unpack("<I", data[:4])[0]
            audio.append((json.loads(data[4:4 + n]), data[4 + n:]))
            continue
        e = json.loads(m["text"])
        events.append(e)
        if e["type"] in kinds:
            break
    return events, audio


def test_ws_server_speech_round_trip(client, container):
    container.ingestion.ingest(sample("districts/vijayawada_district_plan.pdf"), "vijayawada_district_plan.pdf")
    container.stt, container.tts = FakeSTT(), FakeTTS()
    with client.websocket_connect("/ws/voice") as ws:
        ready = ws.receive_json()
        assert ready["speech"]["stt"]["provider"] == "fake" and ready["speech"]["tts"]["sample_rate"] == 16000
        ws.send_json({"type": "start", "session_id": "v1", "speech": {"input": "server", "output": "server"}})
        assert ws.receive_json()["speech"] == {"input": "server", "output": "server"}
        ws.send_json({"type": "audio_start"})
        for f in speak_question(7, gap_after=0.6):
            ws.send_bytes(f)
        events, audio = _frames_until(ws, {"voice_metrics", "error"})
    kinds = [e["type"] for e in events]
    assert "transcript" in kinds and "endpoint" in kinds
    utt = next(e for e in events if e["type"] == "utterance")
    assert utt["text"] == QUESTION and utt["reason"] == "silence"
    done = next(e for e in events if e["type"] == "done")
    assert done["turn_id"] == utt["turn_id"] and done["voice"]["server_voice"] is True
    assert done["voice"]["endpoint_wait_ms"] == utt["endpoint_ms"]
    assert audio and audio[0][0]["turn_id"] == utt["turn_id"] and audio[0][0]["rate"] == 16000 and "text" in audio[0][0]
    assert len(audio[0][1]) == int(0.1 * STT_RATE) * 2
    m = events[-1]
    assert m["type"] == "voice_metrics" and m["last_voice_to_first_audio_ms"] >= utt["endpoint_ms"]
    assert kinds.index("audio_end") < kinds.index("voice_metrics")


def test_ws_browser_transcript_can_ask_for_the_server_voice(client, container):
    container.ingestion.ingest(sample("faq/campaign_faq.md"), "campaign_faq.md")
    container.tts = FakeTTS()
    with client.websocket_connect("/ws/voice") as ws:
        ws.receive_json()
        ws.send_json({"type": "final", "text": "What is the Sunrise Coast Alliance?", "turn_id": "t9", "speak": True})
        events, audio = _frames_until(ws, {"voice_metrics", "error"})
    assert audio and all(h["turn_id"] == "t9" for h, _ in audio)
    assert "last_voice_to_first_audio_ms" not in events[-1]      # no server recognition, no end-to-end number


def test_ws_audio_without_server_recognition_is_refused_once(client):
    with client.websocket_connect("/ws/voice") as ws:
        ws.receive_json()
        ws.send_bytes(silence(0.04))
        ws.send_bytes(silence(0.04))
        ws.send_json({"type": "ping", "t": 1})
        first = ws.receive_json()
        assert first["type"] == "error" and "STT_PROVIDER" in first["message"]
        assert ws.receive_json() == {"type": "pong", "t": 1}


# ── real local models (optional) ───────────────────────────────────────
MODELS = Path(os.environ.get("MODEL_CACHE_DIR", Path(__file__).resolve().parents[2] / "models"))


@pytest.mark.skipif(not ((MODELS / "stt").is_dir() and (MODELS / "tts").is_dir()),
                    reason="local speech models not downloaded (scripts/download_models.py --speech)")
def test_local_models_hear_what_the_local_voice_says():
    from app.voice.audio import resample
    from app.voice.stt import SherpaSTT
    from app.voice.tts import SherpaTTS
    tts = SherpaTTS(str(MODELS / "tts"))
    stt = SherpaSTT(str(MODELS / "stt"))
    pcm = tts.synthesize("What is planned for chilli farmers?")
    x = resample(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768, tts.sample_rate, STT_RATE)
    text = stt.transcribe(np.concatenate([np.zeros(4800, np.float32), x])).lower()
    assert "farmers" in text and "planned" in text
