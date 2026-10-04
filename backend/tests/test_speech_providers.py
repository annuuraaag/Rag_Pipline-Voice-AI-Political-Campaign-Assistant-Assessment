"""Speech providers against mocked transports, and the vocabulary corrector on the sample corpus."""
import asyncio
import json
import wave
from io import BytesIO

import httpx
import pytest

from app.config import Settings
from app.voice.audio import STT_RATE, wav_bytes
from app.voice.stt import DeepgramSTT, WhisperSTT, build_stt
from app.voice.transcriber import WhisperTranscriber
from app.voice.tts import DeepgramTTS, ElevenLabsTTS, OpenAITTS, build_tts
from app.voice.vocabulary import VocabularyCorrector, build_corrector, district_terms, phonetic, similarity
from tests.conftest import sample


# ── vocabulary ─────────────────────────────────────────────────────────
def test_phonetic_spelling_absorbs_typical_mishearings():
    assert phonetic("Vizag") == phonetic("vijag") and similarity(phonetic("Vijawada"), phonetic("Vijayawada")) >= 0.8
    assert similarity(phonetic("Gunter"), phonetic("Guntur")) > 0.8
    assert similarity(phonetic("Visika Putnam"), phonetic("Visakhapatnam")) > 0.8
    assert district_terms()["Vijaywada"] == "Vijayawada"     # a misspelling alias writes the district
    assert district_terms()["Bezawada"] == "Bezawada"        # an alternative name is kept as said


@pytest.fixture
def corrector(container) -> VocabularyCorrector:
    for path, name in [("districts/vijayawada_district_plan.pdf", "vijayawada_district_plan.pdf"),
                       ("schemes/education_schemes.txt", "education_schemes.txt"),
                       ("agriculture/farmers_charter.txt", "farmers_charter.txt")]:
        container.ingestion.ingest(sample(path), name)
    return build_corrector(container.store.iter_chunks())


@pytest.mark.parametrize("heard, fixed", [
    ("What healthcare initiatives are proposed for Vid, awada", "What healthcare initiatives are proposed for Vijayawada"),
    ("Compare the hospital plans for Gunter and Visika Putnam", "Compare the hospital plans for Guntur and Visakhapatnam"),
    ("Who is eligible for the Pratiba scholarship", "Who is eligible for the Pratibha Scholarship"),
    ("I live in Bezoada", "I live in Bezawada"),
])
def test_misheard_names_are_corrected(corrector, heard, fixed):
    assert corrector.correct(heard) == fixed


@pytest.mark.parametrize("text", [
    "Where is the campaign office",            # ordinary words of the corpus are never touched
    "who won the last election",
    "What is the price of petrol today",
    "the health shield covers families",       # casing alone is not changed
])
def test_ordinary_words_are_left_alone(corrector, text):
    assert corrector.correct(text) == text


def test_the_corrector_follows_the_corpus(container):
    cache = container.vocabulary
    before = cache.get("default")
    assert "Pratibha Scholarship" not in before.names
    container.ingestion.ingest(sample("schemes/education_schemes.txt"), "education_schemes.txt")
    assert "Pratibha Scholarship" in cache.get("default").names     # rebuilt on the new corpus revision


# ── speech-to-text providers ───────────────────────────────────────────
class FakeDeepgram:
    """Speaks the Deepgram listen protocol: interim results while audio arrives, finals on Finalize."""

    def __init__(self, words):
        self.words, self.sent, self.queue = words, [], asyncio.Queue()
        self.url, self.headers = None, None

    async def __call__(self, url, additional_headers=None, open_timeout=None):
        self.url, self.headers = url, additional_headers
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        msg = await self.queue.get()
        if msg is None:
            raise StopAsyncIteration
        return msg

    async def send(self, data):
        self.sent.append(data)
        if isinstance(data, bytes):
            n = sum(isinstance(x, bytes) for x in self.sent)
            await self.queue.put(json.dumps({"type": "Results", "is_final": False,
                                             "channel": {"alternatives": [{"transcript": " ".join(self.words[:n])}]}}))
        elif json.loads(data)["type"] == "Finalize":
            await self.queue.put(json.dumps({"type": "Results", "is_final": True, "from_finalize": True,
                                             "channel": {"alternatives": [{"transcript": " ".join(self.words)}]}}))

    async def close(self):
        await self.queue.put(None)


def test_deepgram_stream_partials_and_finalize():
    fake = FakeDeepgram(["what", "is", "planned", "for", "farmers"])

    async def run():
        partials = []

        async def on_partial(t):
            partials.append(t)
        stt = DeepgramSTT("key", model="nova-3", connect=fake)
        stream = await stt.open(on_partial, language="en-IN", vocabulary=["Pratibha Scholarship"])
        for _ in range(3):
            await stream.send(b"\0\0" * 320)
        await asyncio.sleep(0.01)
        text = await stream.finish()
        return partials, text

    partials, text = asyncio.run(run())
    assert partials[:3] == ["what", "what is", "what is planned"]
    assert text == "what is planned for farmers"
    assert "keyterm=Pratibha+Scholarship" in fake.url and "language=en-IN" in fake.url and "interim_results=true" in fake.url
    assert fake.headers == {"Authorization": "Token key"}
    assert json.loads(fake.sent[-1]) == {"type": "CloseStream"}


def test_whisper_stream_partials_at_an_interval_and_a_final_transcript():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = req.content
        wav = body[body.index(b"RIFF"):]
        with wave.open(BytesIO(wav[: wav.rindex(b"\r\n--")])) as w:
            calls.append(w.getnframes() / w.getframerate())
        return httpx.Response(200, json={"text": f" HEARD {len(calls)} "})

    async def run():
        partials = []

        async def on_partial(t):
            partials.append(t)
        t = WhisperTranscriber("groq", "https://x/openai/v1", "k", transport=httpx.MockTransport(handler))
        stream = await WhisperSTT(t, partial_interval_s=0.5).open(on_partial)
        for _ in range(30):                       # 1.2 s of audio in 40 ms frames
            await stream.send(b"\1\0" * 640)
        await asyncio.sleep(0.05)
        final = await stream.finish()
        again = await stream.finish()             # nothing new: no second request
        return partials, final, again

    partials, final, again = asyncio.run(run())
    assert partials and partials[0] == "heard 1"      # upper-case output is tidied
    assert final == again and len(calls) == len(partials) + 1
    assert calls[-1] == pytest.approx(1.2, abs=0.01)


def test_build_stt_reports_why_a_provider_is_unavailable(tmp_path):
    s = Settings(data_dir=tmp_path, stt_provider="local", stt_local_model_dir=str(tmp_path / "missing"), _env_file=None)
    p, status = build_stt(s, None)
    assert p is None and status.startswith("unavailable")
    p, status = build_stt(Settings(data_dir=tmp_path, stt_provider="whisper", _env_file=None), None)
    assert p is None and "key" in status
    p, status = build_stt(Settings(data_dir=tmp_path, _env_file=None), None)
    assert p is None and status.startswith("disabled")


# ── text-to-speech providers ───────────────────────────────────────────
def collect(stream) -> list[bytes]:
    async def run():
        return [c async for c in stream]
    return asyncio.run(run())


def test_openai_tts_streams_pcm_without_splitting_samples():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, content=b"\x01" * 24001)   # odd length: the last byte is dropped

    tts = OpenAITTS("https://api.openai.com/v1", "sk", voice="alloy", transport=httpx.MockTransport(handler))
    chunks = collect(tts.stream("Hello there.", speed=1.1))
    assert all(len(c) % 2 == 0 for c in chunks) and sum(map(len, chunks)) == 24000
    assert seen["body"] == {"model": "gpt-4o-mini-tts", "input": "Hello there.", "voice": "alloy",
                            "response_format": "pcm", "speed": 1.1}
    assert seen["auth"] == "Bearer sk"


def test_openai_compatible_wav_response_sets_the_sample_rate():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=wav_bytes(b"\0\0" * 22050, 22050))

    tts = OpenAITTS("http://kokoro:8880/v1", "", response_format="wav", transport=httpx.MockTransport(handler))
    chunks = collect(tts.stream("Hi."))
    assert tts.sample_rate == 22050 and sum(map(len, chunks)) == 44100


def test_deepgram_and_elevenlabs_tts_requests():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((str(req.url), dict(req.headers), json.loads(req.content)))
        return httpx.Response(200, content=b"\0\0" * 100)

    collect(DeepgramTTS("dg", transport=httpx.MockTransport(handler)).stream("Hi."))
    collect(ElevenLabsTTS("el", voice="v1", transport=httpx.MockTransport(handler)).stream("Hi.", speed=1.5))
    (dg_url, dg_h, dg_body), (el_url, el_h, el_body) = seen
    assert "/v1/speak?" in dg_url and "encoding=linear16" in dg_url and "model=aura-2-thalia-en" in dg_url
    assert dg_h["authorization"] == "Token dg" and dg_body == {"text": "Hi."}
    assert "/text-to-speech/v1/stream?output_format=pcm_24000" in el_url and el_h["xi-api-key"] == "el"
    assert el_body["voice_settings"] == {"speed": 1.2}   # clamped to ElevenLabs' range


def test_build_tts_needs_keys(tmp_path):
    p, status = build_tts(Settings(data_dir=tmp_path, tts_provider="deepgram", _env_file=None))
    assert p is None and "DEEPGRAM_API_KEY" in status
    p, status = build_tts(Settings(data_dir=tmp_path, tts_provider="openai", tts_base_url="http://localhost:8880/v1",
                                   _env_file=None))
    assert p is not None and status == "openai/alloy"     # a self-hosted server needs no key


def test_stt_rate_constant_matches_the_browser_capture():
    assert STT_RATE == 16000   # frontend/src/lib/voice/pcm-worklet.js OUT_RATE
