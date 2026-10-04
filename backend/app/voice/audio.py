"""Audio and text helpers for the server-side speech path.

Audio travels as 16-bit little-endian mono PCM ("PCM16") in both directions: microphone audio
arrives at 16 kHz (what speech recognisers expect) and synthesized speech leaves at the voice's
native rate (the browser's Web Audio API resamples on playback).

Answers are spoken sentence by sentence while the LLM is still writing, exactly like the
browser path (frontend/src/lib/voice/tts.js): ``SentenceSplitter`` is a port of its splitter,
plus an optional early cut at the first clause so the first audio starts sooner.
"""
from __future__ import annotations

import io
import re
import wave

import numpy as np

STT_RATE = 16000


# ── PCM ──────────────────────────────────────────────────────────────────
def pcm16_to_float(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2:
        pcm = pcm[:-1]
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear-interpolation resampler: plenty for speech going into a recogniser."""
    x = np.asarray(x, dtype=np.float32)
    if src == dst or not len(x):
        return x
    n = int(round(len(x) * dst / src))
    return np.interp(np.arange(n) * (src / dst), np.arange(len(x)), x).astype(np.float32)


def rms_dbfs(x: np.ndarray) -> float:
    if not len(x):
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-9))


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def parse_wav(data: bytes) -> tuple[bytes, int]:
    """WAV file → (mono PCM16, sample rate)."""
    with wave.open(io.BytesIO(data)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"unsupported WAV sample width: {width * 8} bit")
    if channels > 1:
        x = np.frombuffer(frames, dtype="<i2").reshape(-1, channels).mean(axis=1)
        frames = x.astype("<i2").tobytes()
    return frames, rate


def frames_of(pcm: bytes, sample_rate: int, ms: int) -> list[bytes]:
    """Split PCM16 into chunks of `ms` milliseconds (the last one may be shorter)."""
    step = max(2, int(sample_rate * ms / 1000) * 2)
    return [pcm[i:i + step] for i in range(0, len(pcm), step)]


# ── text to be spoken ────────────────────────────────────────────────────
_CITATIONS = re.compile(r"\s*(?:\[S\d+\])+")
_ABBREVIATIONS = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof|Rs|St|No|Nos|vs|approx|etc|e\.g|i\.e|p|pp|Sr|Jr|Govt|Dept)\.$",
                            re.IGNORECASE)
# A sentence ends at . ! ? ; before a capital/digit, possibly after its citation markers
# ("... planned. [S1] Next"), or at the very end of the text.
_END = re.compile(r"[.!?;](?:\s*\[S\d+\])*(?=\s+[\"'(]?[A-Z0-9])|[.!?](?:\s*\[S\d+\])*(?=\s*$)")
_CLAUSE = re.compile(r"[,:;—](?=\s)")   # where the first piece may end early
MIN_SENTENCE = 24
MAX_CLAUSE = 220


def speakable(text: str) -> str:
    """Text as it should be heard: no citation markers, markdown or ellipsis artefacts."""
    t = _CITATIONS.sub("", text)
    t = re.sub(r"[*_#`>|]", "", t)
    t = re.sub(r"\s+…$", "", t)
    return re.sub(r"\s+", " ", t).strip()


def split_sentences(buffer: str, force: bool = False, first_clause: int = 0) -> tuple[list[str], str]:
    """Split complete sentences off a streaming buffer; returns (sentences, rest).

    Decimals ("6.5") and common abbreviations ("Dr.") never end a sentence. With
    ``first_clause`` > 0 the first piece may end at a comma, colon or dash once it is that long, so
    speech can start before the first full sentence is written.
    """
    out: list[str] = []
    rest = buffer
    while True:
        cut = -1
        for m in _END.finditer(rest):
            end = m.end()
            head = rest[: m.start() + 1]
            if len(head.strip()) < MIN_SENTENCE or _ABBREVIATIONS.search(head):
                continue
            if re.search(r"\d\.$", head) and re.match(r"\d", rest[m.start() + 1:]):
                continue
            if not force and not rest[end:].strip():
                break  # a trailing "." may still be "6." of "6.5"
            cut = end
            break
        if cut < 0 and first_clause and not out:
            m = _CLAUSE.search(rest, first_clause)
            if m and rest[m.end():].strip():
                cut = m.start() + 1
        if cut < 0 and len(rest) > MAX_CLAUSE:
            comma = rest.rfind(", ", 0, MAX_CLAUSE)
            if comma > MIN_SENTENCE:
                cut = comma + 1
        if cut < 0:
            break
        sentence = rest[:cut].strip()
        if sentence:
            out.append(sentence)
        rest = rest[cut:]
        first_clause = 0
    if force and rest.strip():
        out.append(rest.strip())
        rest = ""
    return out, rest


class SentenceSplitter:
    """Feed LLM tokens in, get speakable pieces out as soon as they are complete."""

    def __init__(self, first_clause: int = 0):
        self.buffer = ""
        self.first_clause = first_clause

    def push(self, delta: str) -> list[str]:
        self.buffer += delta
        out, self.buffer = split_sentences(self.buffer, first_clause=self.first_clause)
        if out:
            self.first_clause = 0
        return out

    def flush(self) -> list[str]:
        out, self.buffer = split_sentences(self.buffer, force=True)
        return out
