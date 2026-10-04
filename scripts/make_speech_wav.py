"""Speak questions with the server's local voice into a WAV file: a fake microphone for
frontend/e2e/server-speech.e2e.mjs (Chromium's --use-file-for-fake-audio-capture).

    python scripts/make_speech_wav.py --out /tmp/questions.wav \
        "What healthcare initiatives are proposed for Vijayawada?" "Who is eligible for the Pratibha Scholarship?"

Needs the local voice (python scripts/download_models.py --speech). The questions are separated by
--gap seconds of silence: long enough for the first answer to start, short enough that the second
question arrives while it is still being spoken (barge-in).
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.voice.audio import resample  # noqa: E402
from app.voice.tts import SherpaTTS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice", default=str(ROOT / "models" / "tts"), help="sherpa-onnx voice directory")
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--lead", type=float, default=1.5, help="silence before the first question (s)")
    ap.add_argument("--gap", type=float, default=5.0, help="silence between questions (s)")
    ap.add_argument("--tail", type=float, default=12.0, help="silence after the last question (s)")
    args = ap.parse_args()
    tts = SherpaTTS(args.voice)
    parts = [np.zeros(int(args.lead * args.rate), np.float32)]
    for i, q in enumerate(args.questions):
        pcm = np.frombuffer(tts.synthesize(q), dtype="<i2").astype(np.float32) / 32768
        parts.append(resample(pcm, tts.sample_rate, args.rate))
        parts.append(np.zeros(int((args.gap if i < len(args.questions) - 1 else args.tail) * args.rate), np.float32))
    audio = np.concatenate(parts)
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(args.rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    print(f"wrote {args.out}: {len(audio) / args.rate:.1f} s, {len(args.questions)} question(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
