"""Latency benchmark: per-stage percentiles for typed questions, and the voice path with and
without speculative retrieval on partial transcripts, with adaptive or fixed end of turn, up to
the first spoken audio.

    python eval/run_latency_benchmark.py [--rerank] [--repeats 3] [--word-ms 250] [--endpoint-ms 900]
    docker compose exec backend python eval/run_latency_benchmark.py --rerank

Typed path: every query in eval/queries.jsonl through RAGService.stream() (the code behind
POST /query and the WebSocket), the query-embedding cache cleared before each run so the
embedding cost is measured, not a cache hit.

Voice path: each query is "spoken" word by word (one partial transcript every --word-ms), then
the end-of-utterance silence passes before the final transcript is sent, exactly as the web app
does: adaptive (the end-of-turn verdict on the transcript, app/voice/endpointing.py) or the fixed
--endpoint-ms. Measured from the final transcript to the first answer token, with speculation
(the controller saw the partials) and without (same final transcript, no partials). End of
speech → first token adds the end-of-turn wait.

First audio: with the local voice installed (models/tts, scripts/download_models.py --speech),
the first spoken piece of every answer (first sentence, or first clause from 24 characters, as
the server voice cuts it) is synthesised and timed: first audio = the moment that piece is
complete in the token stream + its synthesis time. With GROQ_API_KEY set, the LLM's real
time-to-first-token is inside these numbers; without it the extractive fallback answers instantly.

All numbers are wall-clock measurements on the machine running the script; the LLM part
depends on the provider (see "llm" in the report). Results: eval/results/latency_benchmark.{json,md}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import tempfile
import time
import uuid
from pathlib import Path

from harness import CAMPAIGN, ROOT, build_eval_container, describe

from app.domain import MetadataFilter
from app.voice.audio import SentenceSplitter, speakable
from app.voice.controller import PartialTranscriptController
from app.voice.endpointing import Endpointer

STAGES = ["rewrite", "embed", "dense", "bm25", "fusion", "rerank", "threshold", "context", "retrieval_total",
          "first_token", "llm", "total"]


def clear_cache(c) -> None:
    """Measure the embedding, not a cache hit (the test stand-in has no cache)."""
    getattr(c.embedder, "clear_cache", lambda: None)()


def pct(values: list[float]) -> dict:
    if not values:
        return {}
    v = sorted(values)

    def q(p: float) -> float:
        k = (len(v) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(v) - 1)
        return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 2)
    return {"n": len(v), "p50": q(0.5), "p90": q(0.9), "p95": q(0.95), "mean": round(statistics.fmean(v), 2),
            "max": round(v[-1], 2)}


async def replay_history(c, q: dict, session: str) -> None:
    for h in q.get("history", []):
        await c.rag.answer(h, session_id=session, campaign_id=CAMPAIGN)


async def stream_once(c, text: str, session: str, reuse=None, tts=None) -> tuple[dict, float | None, str | None]:
    """(stage timings, final → first token ms, cache outcome); with `tts`, timings gain
    "first_audio": final → first spoken piece synthesised."""
    t0 = time.perf_counter()
    first = None
    timings, cache = {}, None
    splitter, piece, piece_at = SentenceSplitter(first_clause=24), None, None
    async for e in c.rag.stream(text, session_id=session, campaign_id=CAMPAIGN, reuse=reuse):
        if e["type"] == "token":
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            if tts is not None and piece is None:
                done = splitter.push(e["text"])
                if done:
                    piece, piece_at = done[0], (time.perf_counter() - t0) * 1000
        elif e["type"] == "done":
            timings, cache = e["latency_ms"], e.get("cache")
    if tts is not None and timings:
        if piece is None:
            piece, piece_at = (splitter.flush() or [""])[0], (time.perf_counter() - t0) * 1000
        if speakable(piece):
            t = time.perf_counter()
            tts.synthesize(speakable(piece))
            timings = {**timings, "first_audio": round(piece_at + (time.perf_counter() - t) * 1000, 2)}
    return timings, first, cache


async def typed(c, queries: list[dict], repeats: int) -> dict:
    per_stage: dict[str, list[float]] = {s: [] for s in STAGES}
    for _ in range(repeats):
        for q in queries:
            session = uuid.uuid4().hex
            await replay_history(c, q, session)
            clear_cache(c)
            timings, _, _ = await stream_once(c, q["query"], session)
            if "retrieval_total" not in timings:   # small talk / introductions skip retrieval
                continue
            for s in STAGES:
                if s in timings:
                    per_stage[s].append(timings[s])
    return {s: pct(v) for s, v in per_stage.items() if v}


def heard(text: str) -> str:
    """A transcript as the browser's recogniser delivers it: lower case, no punctuation."""
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


async def speak(c, q: dict, session: str, word_s: float, wait_s, tts=None) -> tuple[dict, float, float, str | None]:
    """Partials while "speaking", the end-of-turn silence, then the final transcript.
    `wait_s(ctrl)` gives the silence; returns (timings, final → first token, silence, cache)."""
    words = q["query"].split()
    ctrl = PartialTranscriptController(c.rag, CAMPAIGN, session, revision=lambda: c.ingestion.revision,
                                       endpointer=Endpointer())
    for i in range(1, len(words) + 1):
        await ctrl.on_partial(heard(" ".join(words[:i])))
        await asyncio.sleep(word_s)
    wait = wait_s(ctrl)
    await asyncio.sleep(wait)
    ctrl.reset_turn()
    timings, first, cache = await stream_once(c, q["query"], session, reuse=ctrl.reuse, tts=tts)
    return timings, first, wait, cache


async def voice(c, queries: list[dict], word_s: float, endpoint_s: float, tts=None) -> dict:
    out = {k: [] for k in ("speculative", "baseline", "retrieval_spec", "retrieval_base", "cache", "saved",
                           "adaptive", "adaptive_wait", "audio_fixed", "audio_adaptive", "audio_base")}
    for q in queries:
        # speculation + the fixed end-of-utterance silence
        session = uuid.uuid4().hex
        await replay_history(c, q, session)
        clear_cache(c)
        timings, first, _, cache = await speak(c, q, session, word_s, lambda ctrl: endpoint_s, tts)
        if "retrieval_total" not in timings:
            continue
        out["speculative"].append(first)
        out["retrieval_spec"].append(timings.get("retrieval_total", 0.0))
        out["cache"].append(cache)
        out["audio_fixed"].append(timings.get("first_audio"))
        # speculation + adaptive end of turn (the verdict on the last partial sets the silence)
        session_a = uuid.uuid4().hex
        await replay_history(c, q, session_a)
        clear_cache(c)
        timings_a, first_a, wait_a, _ = await speak(
            c, q, session_a, word_s, lambda ctrl: (ctrl.end_of_turn.wait_ms / 1000) if ctrl.end_of_turn else endpoint_s, tts)
        out["adaptive"].append(first_a + wait_a * 1000)
        out["adaptive_wait"].append(wait_a * 1000)
        out["audio_adaptive"].append(timings_a.get("first_audio", 0.0) + wait_a * 1000 if tts else None)
        # baseline: identical final transcript, no partials seen
        session_b = uuid.uuid4().hex
        await replay_history(c, q, session_b)
        clear_cache(c)
        ctrl_b = PartialTranscriptController(c.rag, CAMPAIGN, session_b, revision=lambda: c.ingestion.revision)
        timings_b, first_b, _ = await stream_once(c, q["query"], session_b, reuse=ctrl_b.reuse, tts=tts)
        out["baseline"].append(first_b)
        out["retrieval_base"].append(timings_b.get("retrieval_total", 0.0))
        out["audio_base"].append(timings_b.get("first_audio"))
        out["saved"].append(timings_b.get("retrieval_total", 0.0) - timings.get("retrieval_total", 0.0))
    n = len(out["cache"])
    hits = {k: sum(1 for x in out["cache"] if x == k) for k in ("hit", "stage1", "miss")}
    report = {
        "queries": n,
        "cache_outcomes": hits,
        "reuse_rate": round((hits["hit"] + hits["stage1"]) / n, 3) if n else 0.0,
        "final_to_first_token": {"speculative": pct(out["speculative"]), "baseline": pct(out["baseline"])},
        "retrieval_on_critical_path": {"speculative": pct(out["retrieval_spec"]), "baseline": pct(out["retrieval_base"])},
        "end_of_speech_to_first_token": {
            "speculative": pct([x + endpoint_s * 1000 for x in out["speculative"]]),
            "baseline": pct([x + endpoint_s * 1000 for x in out["baseline"]]),
            "speculative_adaptive": pct(out["adaptive"]),
        },
        "adaptive_end_of_turn_wait": pct(out["adaptive_wait"]),
        "retrieval_saved_per_turn": pct(out["saved"]),
    }
    if tts is not None:
        report["end_of_speech_to_first_audio"] = {
            "speculative": pct([x + endpoint_s * 1000 for x in out["audio_fixed"] if x is not None]),
            "baseline": pct([x + endpoint_s * 1000 for x in out["audio_base"] if x is not None]),
            "speculative_adaptive": pct([x for x in out["audio_adaptive"] if x is not None]),
        }
    return report


def markdown(report: dict) -> str:
    cfg = report["config"]
    lines = [
        "# Latency benchmark", "",
        f"Generated by `eval/run_latency_benchmark.py`; do not edit by hand. Machine: {report['machine']}.",
        f"LLM: **{cfg['llm']}** · retrieval: {cfg['retrieval']} · gate: {cfg['gate']} · "
        f"embedder: {cfg.get('embedder', 'bge-small-en-v1.5')} · "
        f"{report['queries']} queries × {report['repeats']} repeats (typed), 1 pass (voice).", "",
        "## Typed questions: per stage (ms)", "",
        "| Stage | p50 | p90 | p95 | mean | max | n |", "|---|---|---|---|---|---|---|",
    ]
    for s, v in report["typed"].items():
        lines.append(f"| {s} | {v['p50']} | {v['p90']} | {v['p95']} | {v['mean']} | {v['max']} | {v['n']} |")
    v = report["voice"]
    lines += [
        "", "## Voice: speculative retrieval on partial transcripts", "",
        f"Speech simulated at {report['word_ms']} ms per word, partial transcripts as a browser recogniser delivers them; "
        f"the final transcript follows after the end-of-turn silence: fixed {report['endpoint_ms']} ms (the old UI "
        f"behaviour) or adaptive (below). Speculation reused: {v['cache_outcomes']} "
        f"→ reuse rate **{v['reuse_rate']:.0%}**.", "",
        "| Measure (ms) | Speculative p50 | p90 | p95 | Baseline p50 | p90 | p95 |", "|---|---|---|---|---|---|---|",
    ]
    rows = [("final_to_first_token", "Final transcript → first token"),
            ("retrieval_on_critical_path", "Retrieval on the critical path"),
            ("end_of_speech_to_first_token", f"End of speech → first token (fixed {report['endpoint_ms']} ms pause)")]
    if "end_of_speech_to_first_audio" in v:
        rows.append(("end_of_speech_to_first_audio", f"End of speech → first audio (fixed {report['endpoint_ms']} ms pause)"))
    for key, label in rows:
        a, b = v[key]["speculative"], v[key]["baseline"]
        lines.append(f"| {label} | {a.get('p50')} | {a.get('p90')} | {a.get('p95')} | {b.get('p50')} | {b.get('p90')} | {b.get('p95')} |")
    w = v["adaptive_end_of_turn_wait"]
    lines += ["", f"With adaptive end of turn (the verdict on each question sets the pause: p50 {w.get('p50')} ms, "
              f"mean {w.get('mean')} ms instead of {report['endpoint_ms']} ms):", "",
              "| Measure (ms), speculation + adaptive end of turn | p50 | p90 | p95 |", "|---|---|---|---|"]
    for key, label in [("end_of_speech_to_first_token", "End of speech → first token"),
                       ("end_of_speech_to_first_audio", "End of speech → first audio")]:
        if key in v:
            a = v[key]["speculative_adaptive"]
            lines.append(f"| {label} | {a.get('p50')} | {a.get('p90')} | {a.get('p95')} |")
    s = v["retrieval_saved_per_turn"]
    lines += ["", f"Retrieval time removed from the critical path per turn: p50 {s.get('p50')} ms, p90 {s.get('p90')} ms.",
              "", "Notes: 'first token' includes the LLM's time to first token, which depends on the provider; "
              "with the offline extractive fallback it is near zero, so the numbers isolate the retrieval pipeline. "
              "'First audio' adds the local voice's synthesis of the first spoken piece "
              f"({report.get('tts') or 'no local voice installed: not measured'}). "
              "Real-audio measurements (recogniser, VAD, WebSocket) are in speech_eval.md."]
    return "\n".join(lines) + "\n"


async def main_async(args) -> dict:
    queries = [json.loads(line) for line in Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        tts_dir = Path(args.models) / "tts"
        c = build_eval_container(Path(tmp), args.rerank, test_embedder=args.test_embedder, model_cache_dir=Path(args.models),
                                 tts_provider="local" if tts_dir.is_dir() else "none")
        c.retriever.retrieve("warm up", MetadataFilter(campaign_id=CAMPAIGN))  # ONNX buffers, ANN index
        if c.tts is not None:
            c.tts.synthesize("Warm up.")
        verify = getattr(c.llm, "verify_model", None)
        if verify:
            await verify()
        typed_report = await typed(c, queries, args.repeats)
        voice_report = await voice(c, queries, args.word_ms / 1000, args.endpoint_ms / 1000, tts=c.tts)
        report = {
            "config": describe(c), "machine": args.machine, "queries": len(queries), "repeats": args.repeats,
            "word_ms": args.word_ms, "endpoint_ms": args.endpoint_ms, "typed": typed_report, "voice": voice_report,
            "tts": f"{c.tts.name}/{c.tts.voice}" if c.tts else None,
        }
        aclose = getattr(c.llm, "aclose", None)
        if aclose:
            await aclose()
        c.store.close()
    return report


def main() -> int:
    import platform

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(ROOT / "eval" / "queries.jsonl"))
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--word-ms", type=int, default=250, help="time between partial transcripts (~speaking rate)")
    ap.add_argument("--endpoint-ms", type=int, default=900, help="fixed silence before the final transcript is sent")
    ap.add_argument("--models", default=str(ROOT / "models"), help="MODEL_CACHE_DIR; tts/ there enables first-audio timing")
    ap.add_argument("--test-embedder", action="store_true", help="no bge-small available: labelled stand-in")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--machine", default=f"{platform.system()} {platform.machine()}, {__import__('os').cpu_count()} CPUs")
    args = ap.parse_args()
    report = asyncio.run(main_async(args))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "latency_benchmark" + ("_rerank" if args.rerank else "")
    (out / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = markdown(report)
    (out / f"{name}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out / (name + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
