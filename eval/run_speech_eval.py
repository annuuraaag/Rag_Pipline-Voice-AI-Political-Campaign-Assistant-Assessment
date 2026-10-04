"""End-to-end voice evaluation on real audio: recognition, end of turn, and last word → first audio.

    python eval/run_speech_eval.py [--rerank] [--barge-in] [--models models] [--test-embedder]

Every question in eval/queries.jsonl is spoken by the server's local voice and streamed in real
time (40 ms frames, as the browser streams the microphone) into /ws/voice of a server started
in-process with STT_PROVIDER=local and TTS_PROVIDER=local. History turns are sent first as typed
questions. Each question is played twice: with adaptive end-of-turn detection and with the old
fixed pause (VOICE_ADAPTIVE_ENDPOINTING=false), so the effect is measured on the same audio.

Measured per question (client timestamps unless marked server):
  recognition      word error rate of the final transcript; districts named and recognised;
                   the same audio decoded offline with and without the vocabulary correction
  end of turn      silence after the last word before the server ended the turn (server)
  latency          last word → final transcript → first token → first answer audio received
  barge-in         (--barge-in) a second question spoken over the answer: speech start → barge_in

The last word is located in the synthesised audio itself (last sample above -40 dBFS), so
"last word → first audio" is the delay a user would hear, minus network and speaker buffering.
Without GROQ_API_KEY the extractive fallback answers instantly, so "first token" then contains no
model latency (the report says which LLM ran). Needs the speech models
(python scripts/download_models.py --speech). Results: eval/results/speech_eval.{json,md}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import socket
import statistics
import struct
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from harness import CAMPAIGN, ROOT, build_eval_container, describe

from app.lexicon import detect_districts
from app.main import create_app
from app.voice.audio import STT_RATE, float_to_pcm16, resample
from app.voice.stt import SherpaSTT

FRAME_MS = 40
BARGE_IN_QUESTION = "Who is eligible for the Pratibha Scholarship?"


# ── text metrics ───────────────────────────────────────────────────────
def norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    r, h = norm_words(ref), norm_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)], len(r)


def pct(values: list[float]) -> dict:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {}

    def q(p: float) -> float:
        k = (len(v) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(v) - 1)
        return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 1)
    return {"n": len(v), "p50": q(0.5), "p90": q(0.9), "mean": round(statistics.fmean(v), 1), "max": round(v[-1], 1)}


# ── audio ──────────────────────────────────────────────────────────────
def spoken(c, text: str) -> tuple[np.ndarray, float]:
    """(16 kHz audio of `text` with 0.4 s lead and 2.5 s tail silence, time of the last word)."""
    pcm = c.tts.synthesize(text)
    x = resample(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768, c.tts.sample_rate, STT_RATE)
    loud = np.nonzero(np.abs(x) > 0.01)[0]
    end = (loud[-1] + 1) if len(loud) else len(x)
    lead = np.zeros(int(0.4 * STT_RATE), np.float32)
    return np.concatenate([lead, x[:end], np.zeros(int(2.5 * STT_RATE), np.float32)]), (len(lead) + end) / STT_RATE


def frames(x: np.ndarray) -> list[bytes]:
    pcm = float_to_pcm16(x)
    step = STT_RATE * FRAME_MS // 1000 * 2
    return [pcm[i:i + step] for i in range(0, len(pcm), step)]


# ── one spoken question through the WebSocket ──────────────────────────
async def ask(url: str, q: dict, audio: np.ndarray, last_word_s: float, barge_audio: np.ndarray | None) -> dict:
    from websockets.asyncio.client import connect

    out: dict = {"id": q["id"], "query": q["query"]}
    async with connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "session_id": uuid.uuid4().hex,
                                  "speech": {"input": "server", "output": "server"}}))
        for h in q.get("history", []):
            await ws.send(json.dumps({"type": "final", "text": h}))
            while True:
                m = await ws.recv()
                if isinstance(m, str) and json.loads(m)["type"] == "done":
                    break
        events: dict[str, float] = {}
        done = asyncio.Event()
        t0 = 0.0

        async def reader() -> None:
            async for m in ws:
                now = time.perf_counter() - t0
                if isinstance(m, bytes):
                    n = struct.unpack("<I", m[:4])[0]
                    turn = json.loads(m[4:4 + n])["turn_id"]
                    events.setdefault(f"audio:{turn}", now)
                    continue
                e = json.loads(m)
                kind = e["type"]
                if kind == "utterance" and e.get("text"):
                    out.setdefault("transcript", e["text"])
                    out.setdefault("server_endpoint_ms", e["endpoint_ms"])
                    out.setdefault("turn", e["turn_id"])
                    events.setdefault("utterance", now)
                elif kind == "token":
                    events.setdefault(f"token:{e.get('turn_id')}", now)
                elif kind == "barge_in":
                    events.setdefault("barge_in", now)
                elif kind == "voice_metrics" and e["turn_id"] == out.get("turn"):
                    out["server"] = {k: e.get(k) for k in ("final_to_first_audio_ms", "first_synth_ms",
                                                           "last_voice_to_first_audio_ms", "stt_final_ms")}
                    if barge_audio is None:
                        done.set()
                elif kind == "utterance" and not e.get("text"):
                    out["missed"] = e.get("reason")
                    done.set()
                if barge_audio is not None and "barge_in" in events and kind == "utterance" and e.get("text") \
                        and e["turn_id"] != out.get("turn"):
                    out["barge_transcript"] = e["text"]
                    done.set()

        await ws.send(json.dumps({"type": "audio_start"}))
        t0 = time.perf_counter()
        task = asyncio.create_task(reader())
        chunks = frames(audio)
        for i, f in enumerate(chunks):
            # Real time, on an absolute schedule; a microphone delivers a frame once it is recorded.
            await asyncio.sleep(max(0.0, t0 + (i + 1) * FRAME_MS / 1000 - time.perf_counter()))
            await ws.send(f)
        if barge_audio is not None:
            # Keep the microphone open (conversation mode); talk over the answer once it is heard.
            deadline = time.perf_counter() + 10
            while not any(k.startswith("audio:") for k in events) and time.perf_counter() < deadline:
                await ws.send(frames(np.zeros(STT_RATE * FRAME_MS // 1000, np.float32))[0])
                await asyncio.sleep(FRAME_MS / 1000)
            first_audio = next((v for k, v in events.items() if k.startswith("audio:")), None)
            if first_audio is not None:
                # One second into the answer, the user starts talking over it (the barge audio
                # itself opens with 0.4 s of silence).
                start = time.perf_counter()
                for i, f in enumerate(frames(np.concatenate([np.zeros(STT_RATE, np.float32), barge_audio]))):
                    await asyncio.sleep(max(0.0, start + (i + 1) * FRAME_MS / 1000 - time.perf_counter()))
                    await ws.send(f)
                out["barge_speech_start_s"] = start - t0 + 1.4
        try:
            await asyncio.wait_for(done.wait(), timeout=20)
        except TimeoutError:
            out["timeout"] = True
        task.cancel()
    turn = out.get("turn")
    lw = last_word_s
    if "utterance" in events:
        out["last_word_to_final_ms"] = round((events["utterance"] - lw) * 1000, 1)
    if turn and f"token:{turn}" in events:
        out["final_to_first_token_ms"] = round((events[f"token:{turn}"] - events["utterance"]) * 1000, 1)
    if turn and f"audio:{turn}" in events:
        out["final_to_first_audio_ms"] = round((events[f"audio:{turn}"] - events["utterance"]) * 1000, 1)
        out["last_word_to_first_audio_ms"] = round((events[f"audio:{turn}"] - lw) * 1000, 1)
    if "barge_in" in events and "barge_speech_start_s" in out:
        out["barge_in_detect_ms"] = round((events["barge_in"] - out["barge_speech_start_s"]) * 1000, 1)
    return out


# ── runner ─────────────────────────────────────────────────────────────
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(app, port: int):
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    return server, thread


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("transcript")]
    errs = sum(r["wer_errors"] for r in ok)
    ref = sum(r["wer_ref"] for r in ok)
    gold = sum(len(r["districts"]) for r in rows)
    found = sum(len(set(r["districts"]) & set(r.get("districts_heard", []))) for r in rows)
    return {
        "questions": len(rows), "recognised": len(ok), "missed": [r["id"] for r in rows if not r.get("transcript")],
        "wer": round(errs / ref, 3) if ref else None,
        "district_recall": round(found / gold, 3) if gold else None,
        **{k: pct([r.get(k) for r in ok]) for k in ("server_endpoint_ms", "last_word_to_final_ms", "final_to_first_token_ms",
                                                     "final_to_first_audio_ms", "last_word_to_first_audio_ms")},
        "first_synth_ms": pct([(r.get("server") or {}).get("first_synth_ms") for r in ok]),
        "barge_in_detect_ms": pct([r.get("barge_in_detect_ms") for r in rows]),
        "barge_in_caught": f"{sum('barge_in_detect_ms' in r for r in rows)}/{sum('barge_speech_start_s' in r for r in rows)}",
    }


def markdown(report: dict) -> str:
    cfg, off = report["config"], report["offline"]
    a, f = report["adaptive"]["summary"], report["fixed"]["summary"]

    def row(label, key):
        x, y = a.get(key) or {}, f.get(key) or {}
        return f"| {label} | {x.get('p50', '–')} | {x.get('p90', '–')} | {y.get('p50', '–')} | {y.get('p90', '–')} |"
    lines = [
        "# Voice pipeline on real audio", "",
        f"Generated by `eval/run_speech_eval.py`; do not edit by hand. {a['questions']} questions from `eval/queries.jsonl`, "
        f"spoken by the server's voice ({cfg['tts']}) and streamed in real time into `/ws/voice` "
        f"(recogniser: {cfg['stt']}; VAD: {cfg['vad']}). Machine: {report['machine']}.", "",
        f"LLM: **{cfg['llm']}** · retrieval: {cfg['retrieval']} · embedder: {cfg['embedder']}.", "",
        "## Latency (ms)", "",
        "| Measure | Adaptive p50 | p90 | Fixed 900 ms p50 | p90 |", "|---|---|---|---|---|",
        row("Silence before the turn ends (server)", "server_endpoint_ms"),
        row("Last word → final transcript", "last_word_to_final_ms"),
        row("Final transcript → first token", "final_to_first_token_ms"),
        row("Final transcript → first answer audio", "final_to_first_audio_ms"),
        row("**Last word → first answer audio**", "last_word_to_first_audio_ms"),
        row("First spoken piece synthesised in (server)", "first_synth_ms"),
        "", "## Recognition", "",
        "| Measure | Value |", "|---|---|",
        f"| Word error rate, final transcript (live, with name correction) | {a['wer']:.1%} |",
        f"| Districts named and recognised | {a['district_recall']:.0%} |",
        f"| Offline decode, word error rate without / with name correction | {off['wer_raw']:.1%} / {off['wer_corrected']:.1%} |",
        f"| Offline decode, districts recognised without / with name correction | {off['district_raw']:.0%} / {off['district_corrected']:.0%} |",
        f"| Questions not recognised at all | {', '.join(a['missed']) or 'none'} |",
    ]
    if a["barge_in_detect_ms"]:
        b = a["barge_in_detect_ms"]
        lines += ["", "## Barge-in", "",
                  f"A second question (\"{BARGE_IN_QUESTION}\") spoken 1 s after the answer starts: caught "
                  f"{a['barge_in_caught']}; from its first word to the answer being stopped p50 {b.get('p50')} ms, "
                  f"p90 {b.get('p90')} ms."]
    lines += ["", "Notes: the voice that asks is the same synthetic voice the assistant speaks with: clean, "
              "American-accented speech, so real users (Indian English, background noise) will see higher error "
              "rates. Latency excludes network and audio-device buffering. Without an LLM key the answer is "
              "extractive and instant, so model time-to-first-token must be added for an LLM.",
              "", "## Per question (adaptive)", "",
              "| id | heard | turn ends after (ms) | last word → first audio (ms) |", "|---|---|---|---|"]
    for r in report["adaptive"]["rows"]:
        lines.append(f"| {r['id']} | {r.get('transcript', '–')} | {r.get('server_endpoint_ms', '–')} | "
                     f"{r.get('last_word_to_first_audio_ms', '–')} |")
    return "\n".join(lines) + "\n"


async def run_pass(url: str, c, queries: list[dict], audios: dict, barge: np.ndarray | None, adaptive: bool) -> dict:
    c.settings.voice_adaptive_endpointing = adaptive
    rows = []
    for q in queries:
        audio, last_word = audios[q["id"]]
        r = await ask(url, q, audio, last_word, barge if q["id"] in ("q01", "q04", "q08", "q10", "q22") else None)
        e, n = word_errors(q["query"], r.get("transcript", ""))
        r.update(wer_errors=e, wer_ref=n, districts=detect_districts(q["query"]),
                 districts_heard=detect_districts(r.get("transcript", "")))
        rows.append(r)
        print(f"  {q['id']} {'adaptive' if adaptive else 'fixed'}: {r.get('transcript', '(none)')!r} "
              f"end {r.get('server_endpoint_ms')} ms, last word → audio {r.get('last_word_to_first_audio_ms')} ms")
    return {"rows": rows, "summary": summarise(rows)}


def offline(c, queries: list[dict], audios: dict) -> dict:
    """The recogniser alone (offline decode of the same audio), with and without name correction."""
    stt = c.stt
    assert isinstance(stt, SherpaSTT)
    corr = c.vocabulary.get(CAMPAIGN)
    raw_e = cor_e = ref = gold = raw_d = cor_d = 0
    for q in queries:
        text = stt.transcribe(audios[q["id"]][0])
        fixed = corr.correct(text)
        e1, n = word_errors(q["query"], text)
        e2, _ = word_errors(q["query"], fixed)
        raw_e, cor_e, ref = raw_e + e1, cor_e + e2, ref + n
        g = set(detect_districts(q["query"]))
        gold += len(g)
        raw_d += len(g & set(detect_districts(text)))
        cor_d += len(g & set(detect_districts(fixed)))
    return {"wer_raw": round(raw_e / ref, 3), "wer_corrected": round(cor_e / ref, 3),
            "district_raw": round(raw_d / gold, 3) if gold else 0.0, "district_corrected": round(cor_d / gold, 3) if gold else 0.0}


def main() -> int:
    import platform

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(ROOT / "eval" / "queries.jsonl"))
    ap.add_argument("--models", default=str(ROOT / "models"), help="MODEL_CACHE_DIR with stt/, tts/ (and vad/)")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--barge-in", action="store_true", help="also talk over five of the answers")
    ap.add_argument("--test-embedder", action="store_true", help="no bge-small available: labelled stand-in")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--machine", default=f"{platform.system()} {platform.machine()}, {__import__('os').cpu_count()} CPUs")
    args = ap.parse_args()
    queries = [json.loads(line) for line in Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        c = build_eval_container(Path(tmp), args.rerank, test_embedder=args.test_embedder, model_cache_dir=Path(args.models),
                                 stt_provider="local", tts_provider="local")
        if c.stt is None or c.tts is None:
            raise SystemExit(f"Local speech models missing in {args.models}: {c.speech_status}. "
                             "Run: python scripts/download_models.py --speech")
        audios = {q["id"]: spoken(c, q["query"]) for q in queries}
        barge = spoken(c, BARGE_IN_QUESTION)[0] if args.barge_in else None
        port = free_port()
        server, _ = serve(create_app(c.settings, c), port)
        url = f"ws://127.0.0.1:{port}/ws/voice"
        try:
            print("adaptive end of turn:")
            adaptive = asyncio.run(run_pass(url, c, queries, audios, barge, True))
            print("fixed 900 ms:")
            fixed = asyncio.run(run_pass(url, c, queries, audios, None, False))
            report = {"config": {**describe(c), "stt": f"{c.stt.name}/{c.stt.model}", "tts": f"{c.tts.name}/{c.tts.voice}",
                                 "vad": c.vad.name}, "machine": args.machine,
                      "offline": offline(c, queries, audios), "adaptive": adaptive, "fixed": fixed}
        finally:
            server.should_exit = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "speech_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = markdown(report)
    (out / "speech_eval.md").write_text(md, encoding="utf-8")
    print(md.split("## Per question")[0])
    print(f"wrote {out / 'speech_eval.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
