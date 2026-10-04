"""Latency benchmark: per-stage percentiles for typed questions, and the voice path with and
without speculative retrieval on partial transcripts.

    python eval/run_latency_benchmark.py [--rerank] [--repeats 3] [--word-ms 250] [--endpoint-ms 900]
    docker compose exec backend python eval/run_latency_benchmark.py --rerank

Typed path: every query in eval/queries.jsonl through RAGService.stream() (the code behind
POST /query and the WebSocket), the query-embedding cache cleared before each run so the
embedding cost is measured, not a cache hit.

Voice path: each query is "spoken" word by word (one partial transcript every --word-ms), then
the client's end-of-utterance silence (--endpoint-ms, the UI default is 900 ms) passes before
the final transcript is sent, exactly as the web app does. Measured from the final transcript
to the first answer token, with speculation (the controller saw the partials) and without
(same final transcript, no partials). End of speech → first token adds the endpoint wait.

All numbers are wall-clock measurements on the machine running the script; the LLM part
depends on the provider (see "llm" in the report). Results: eval/results/latency_benchmark.{json,md}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time
import uuid
from pathlib import Path

from harness import CAMPAIGN, ROOT, build_eval_container, describe

from app.domain import MetadataFilter
from app.voice.controller import PartialTranscriptController

STAGES = ["rewrite", "embed", "dense", "bm25", "fusion", "rerank", "threshold", "context", "retrieval_total",
          "first_token", "llm", "total"]


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


async def stream_once(c, text: str, session: str, reuse=None) -> tuple[dict, float | None, str | None]:
    t0 = time.perf_counter()
    first = None
    timings, cache = {}, None
    async for e in c.rag.stream(text, session_id=session, campaign_id=CAMPAIGN, reuse=reuse):
        if e["type"] == "token" and first is None:
            first = (time.perf_counter() - t0) * 1000
        elif e["type"] == "done":
            timings, cache = e["latency_ms"], e.get("cache")
    return timings, first, cache


async def typed(c, queries: list[dict], repeats: int) -> dict:
    per_stage: dict[str, list[float]] = {s: [] for s in STAGES}
    for _ in range(repeats):
        for q in queries:
            session = uuid.uuid4().hex
            await replay_history(c, q, session)
            c.embedder.clear_cache()
            timings, _, _ = await stream_once(c, q["query"], session)
            if "retrieval_total" not in timings:   # small talk / introductions skip retrieval
                continue
            for s in STAGES:
                if s in timings:
                    per_stage[s].append(timings[s])
    return {s: pct(v) for s, v in per_stage.items() if v}


async def voice(c, queries: list[dict], word_s: float, endpoint_s: float) -> dict:
    out = {"speculative": [], "baseline": [], "retrieval_spec": [], "retrieval_base": [], "cache": [], "saved": []}
    for q in queries:
        words = q["query"].split()
        # with speculation: partials while "speaking", then the endpoint silence, then final
        session = uuid.uuid4().hex
        await replay_history(c, q, session)
        c.embedder.clear_cache()
        ctrl = PartialTranscriptController(c.rag, CAMPAIGN, session, revision=lambda: c.ingestion.revision)
        for i in range(1, len(words) + 1):
            await ctrl.on_partial(" ".join(words[:i]))
            await asyncio.sleep(word_s)
        await asyncio.sleep(endpoint_s)
        ctrl.reset_turn()
        timings, first, cache = await stream_once(c, q["query"], session, reuse=ctrl.reuse)
        if "retrieval_total" not in timings:
            continue
        out["speculative"].append(first)
        out["retrieval_spec"].append(timings.get("retrieval_total", 0.0))
        out["cache"].append(cache)
        # baseline: identical final transcript, no partials seen
        session_b = uuid.uuid4().hex
        await replay_history(c, q, session_b)
        c.embedder.clear_cache()
        ctrl_b = PartialTranscriptController(c.rag, CAMPAIGN, session_b, revision=lambda: c.ingestion.revision)
        timings_b, first_b, _ = await stream_once(c, q["query"], session_b, reuse=ctrl_b.reuse)
        out["baseline"].append(first_b)
        out["retrieval_base"].append(timings_b.get("retrieval_total", 0.0))
        out["saved"].append(timings_b.get("retrieval_total", 0.0) - timings.get("retrieval_total", 0.0))
    n = len(out["cache"])
    hits = {k: sum(1 for x in out["cache"] if x == k) for k in ("hit", "stage1", "miss")}
    return {
        "queries": n,
        "cache_outcomes": hits,
        "reuse_rate": round((hits["hit"] + hits["stage1"]) / n, 3) if n else 0.0,
        "final_to_first_token": {"speculative": pct(out["speculative"]), "baseline": pct(out["baseline"])},
        "retrieval_on_critical_path": {"speculative": pct(out["retrieval_spec"]), "baseline": pct(out["retrieval_base"])},
        "end_of_speech_to_first_token": {
            "speculative": pct([x + endpoint_s * 1000 for x in out["speculative"]]),
            "baseline": pct([x + endpoint_s * 1000 for x in out["baseline"]]),
        },
        "retrieval_saved_per_turn": pct(out["saved"]),
    }


def markdown(report: dict) -> str:
    cfg = report["config"]
    lines = [
        "# Latency benchmark", "",
        f"Generated by `eval/run_latency_benchmark.py`; do not edit by hand. Machine: {report['machine']}.",
        f"LLM: **{cfg['llm']}** · retrieval: {cfg['retrieval']} · gate: {cfg['gate']} · "
        f"{report['queries']} queries × {report['repeats']} repeats (typed), 1 pass (voice).", "",
        "## Typed questions: per stage (ms)", "",
        "| Stage | p50 | p90 | p95 | mean | max | n |", "|---|---|---|---|---|---|---|",
    ]
    for s, v in report["typed"].items():
        lines.append(f"| {s} | {v['p50']} | {v['p90']} | {v['p95']} | {v['mean']} | {v['max']} | {v['n']} |")
    v = report["voice"]
    lines += [
        "", "## Voice: speculative retrieval on partial transcripts", "",
        f"Speech simulated at {report['word_ms']} ms per word; the final transcript is sent {report['endpoint_ms']} ms "
        f"after the last word (the UI's end-of-utterance silence). Speculation reused: {v['cache_outcomes']} "
        f"→ reuse rate **{v['reuse_rate']:.0%}**.", "",
        "| Measure (ms) | Speculative p50 | p90 | p95 | Baseline p50 | p90 | p95 |", "|---|---|---|---|---|---|---|",
    ]
    for key, label in [("final_to_first_token", "Final transcript → first token"),
                       ("retrieval_on_critical_path", "Retrieval on the critical path"),
                       ("end_of_speech_to_first_token", "End of speech → first token")]:
        a, b = v[key]["speculative"], v[key]["baseline"]
        lines.append(f"| {label} | {a.get('p50')} | {a.get('p90')} | {a.get('p95')} | {b.get('p50')} | {b.get('p90')} | {b.get('p95')} |")
    s = v["retrieval_saved_per_turn"]
    lines += ["", f"Retrieval time removed from the critical path per turn: p50 {s.get('p50')} ms, p90 {s.get('p90')} ms.",
              "", "Notes: 'first token' includes the LLM's time to first token, which depends on the provider; "
              "with the offline extractive fallback it is near zero, so the numbers isolate the retrieval pipeline."]
    return "\n".join(lines) + "\n"


async def main_async(args) -> dict:
    queries = [json.loads(line) for line in Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        c = build_eval_container(Path(tmp), args.rerank)
        c.retriever.retrieve("warm up", MetadataFilter(campaign_id=CAMPAIGN))  # ONNX buffers, ANN index
        verify = getattr(c.llm, "verify_model", None)
        if verify:
            await verify()
        typed_report = await typed(c, queries, args.repeats)
        voice_report = await voice(c, queries, args.word_ms / 1000, args.endpoint_ms / 1000)
        report = {
            "config": describe(c), "machine": args.machine, "queries": len(queries), "repeats": args.repeats,
            "word_ms": args.word_ms, "endpoint_ms": args.endpoint_ms, "typed": typed_report, "voice": voice_report,
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
    ap.add_argument("--endpoint-ms", type=int, default=900, help="silence before the final transcript is sent")
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
