"""Retrieval evaluation, separate from answer generation.

    python eval/run_retrieval_eval.py              # default experiment matrix
    python eval/run_retrieval_eval.py --rerank     # also the cross-encoder variant (needs the model)

Every configuration gets a fresh in-memory index of sample_data/ and runs the 30
labelled queries in eval/queries.jsonl. Gold labels are *evidence phrases*, not
chunk ids, so different chunkers are scored on the same footing: a retrieved
chunk is relevant if it contains a gold phrase (after normalising case/punctuation).

Metrics (ranking computed on the ranked candidate list, before the gate):
  Recall@5      share of a query's gold evidence found in the top 5
  Precision@5   share of the top 5 that contain any gold evidence (strict)
  MRR@10        1 / rank of the first relevant chunk
  nDCG@10       rank-discounted gain, binary relevance
  CtxRecall     share of gold evidence in the final context sent to the LLM (after gate + top-k)
  DocCov        multi-doc queries: share of gold documents represented in that context
  Gate          answerability decision on positives vs negatives (+ the best threshold on this set)

Results: eval/results/retrieval_eval.json and eval/results/retrieval_eval.md.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Repo layout (backend/app) or Docker image layout (/app/app)
sys.path.insert(0, str(ROOT / "backend") if (ROOT / "backend").exists() else str(ROOT))

from app.config import Settings  # noqa: E402
from app.container import Container, build_container, seed_sample_data  # noqa: E402
from app.conversation.rewriter import rewrite_with_rules  # noqa: E402
from app.conversation.state import ConversationState  # noqa: E402
from app.domain import MetadataFilter, ScoredChunk  # noqa: E402
from app.generation.llm.extractive import ExtractiveLLM  # noqa: E402
from app.ingestion.registry import DocumentRegistry  # noqa: E402
from app.retrieval.embedder import FastEmbedder  # noqa: E402
from app.retrieval.vector_store import QdrantStore  # noqa: E402

EMBEDDERS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", ROOT / "models" / "bge-small-en-v1.5"),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ROOT / "models" / "all-MiniLM-L6-v2"),
}


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def load_queries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class Variant:
    name: str
    chunker: str = "section"
    embedder: str = "bge-small"
    mode: str = "hybrid"
    rewrite: bool = True
    rerank: bool = False
    chunk_words: int = 180          # fixed chunker: max words per chunk
    query_prefix: str = "auto"      # "auto" = model's recommended query instruction, "" = none


def default_matrix(with_rerank: bool) -> list[Variant]:
    v = [
        # Chunking: same dense retriever, different chunkers (fixed at the default size and at a
        # size matched to the average section chunk, so the comparison is not about chunk size)
        Variant("fixed 180w · dense", chunker="fixed", mode="dense", rewrite=False),
        Variant("fixed 60w · dense", chunker="fixed", chunk_words=60, mode="dense", rewrite=False),
        Variant("section · dense", mode="dense", rewrite=False),
        # Embedding model and query instruction
        Variant("section · dense · bge no instruction", mode="dense", rewrite=False, query_prefix=""),
        Variant("section · dense · MiniLM", embedder="minilm", mode="dense", rewrite=False),
        # Retrieval method
        Variant("section · BM25 only", mode="bm25", rewrite=False),
        Variant("section · hybrid (RRF)", mode="hybrid", rewrite=False),
        # Conversation-aware rewriting + auto district filter
        Variant("section · hybrid + rewrite", mode="hybrid", rewrite=True),
        Variant("section · hybrid + rewrite · MiniLM", embedder="minilm", mode="hybrid", rewrite=True),
    ]
    if with_rerank:
        v.append(Variant("section · hybrid + rewrite + rerank", mode="hybrid", rewrite=True, rerank=True))
    return v


def build(v: Variant, tmp: Path) -> Container:
    model_id, path = EMBEDDERS[v.embedder]
    if not (path / "tokenizer.json").exists():
        raise SystemExit(f"Embedding model missing: {path}. Run: python scripts/download_models.py --with-baseline")
    s = Settings(data_dir=tmp, chunk_strategy=v.chunker, chunk_max_words=v.chunk_words,
                 chunk_overlap_words=max(5, v.chunk_words // 6), retrieval_mode=v.mode, llm_provider="extractive",
                 embedding_model=model_id, seed_sample_data=False, _env_file=None)
    prefix = None if v.query_prefix == "auto" else v.query_prefix
    c = build_container(
        s, embedder=FastEmbedder(model_id, model_path=str(path), query_prefix=prefix), store=QdrantStore("eval", in_memory=True),
        llm=ExtractiveLLM(), registry=DocumentRegistry(None), reranker="auto" if v.rerank else None,
    )
    if v.rerank and c.retriever.reranker is None:
        raise SystemExit(f"--rerank requested but reranker unavailable: {c.reranker_status}")
    seed_sample_data(c, ROOT / "sample_data")
    return c


def relevant_mask(chunks: list[ScoredChunk], gold: list[dict]) -> list[set[int]]:
    """For each chunk, which gold items (by index) it contains."""
    golds = [(g["doc"], norm(g["evidence"])) for g in gold]
    out = []
    for c in chunks:
        text = norm(c.chunk.text)
        out.append({i for i, (doc, ev) in enumerate(golds) if c.chunk.metadata.document_name == doc and ev in text})
    return out


def run_query(c: Container, q: dict, rewrite: bool):
    query = q["query"]
    filters = MetadataFilter(campaign_id="default")
    if rewrite:
        state = ConversationState("eval")
        for h in q.get("history", []):
            hr = rewrite_with_rules(h, state)
            state.observe_user(hr.understanding, hr.query)
        rw = rewrite_with_rules(query, state)
        query, filters = rw.query, rw.filters.model_copy(update={"campaign_id": "default"})
    elif q.get("history"):
        query = query  # literal follow-up without memory: the baseline the rewriter must beat
    t = time.perf_counter()
    r = c.retriever.retrieve(query, filters)
    return r, query, (time.perf_counter() - t) * 1000


def score_variant(c: Container, v: Variant, queries: list[dict]) -> dict:
    per_query = []
    for q in queries:
        r, used_query, ms = run_query(c, q, v.rewrite)
        ranked = r.candidates[:10]
        gold = q["gold"]
        row = {"id": q["id"], "type": q["type"], "query": q["query"], "retrieval_query": used_query,
               "answerable": q["answerable"], "gate_score": r.top_score, "gate_passed": r.answerable,
               "latency_ms": round(ms, 2),
               "top3": [f"{x.chunk.metadata.document_name} › {x.chunk.metadata.section}" for x in ranked[:3]]}
        if gold:
            mask = relevant_mask(ranked, gold)
            found5 = set().union(*mask[:5]) if mask[:5] else set()
            row["recall@5"] = len(found5) / len(gold)
            # Fair across chunk sizes: gold found within the first 300 words of the ranking.
            budget, words, found_b = 300, 0, set()
            for ch, m in zip(ranked, mask, strict=True):
                if words >= budget:
                    break
                found_b |= m
                words += ch.chunk.metadata.word_count
            row["recall@300w"] = len(found_b) / len(gold)
            row["precision@5"] = sum(1 for m in mask[:5] if m) / 5
            first = next((i for i, m in enumerate(mask) if m), None)
            row["mrr@10"] = 1 / (first + 1) if first is not None else 0.0
            dcg = sum(1 / math.log2(i + 2) for i, m in enumerate(mask) if m)
            n_rel_possible = min(10, len(gold))
            idcg = sum(1 / math.log2(i + 2) for i in range(n_rel_possible))
            row["ndcg@10"] = min(1.0, dcg / idcg) if idcg else 0.0
            ctx = r.results if r.answerable else []
            cmask = relevant_mask(ctx, gold)
            row["ctx_recall"] = len(set().union(*cmask)) / len(gold) if cmask else 0.0
            gold_docs = {g["doc"] for g in gold}
            row["doc_coverage"] = len(gold_docs & {x.chunk.metadata.document_name for x in ctx}) / len(gold_docs)
        per_query.append(row)

    pos = [r for r in per_query if r["answerable"]]
    neg = [r for r in per_query if not r["answerable"]]

    def mean(key, rows=pos):
        vals = [r[key] for r in rows if key in r]
        return round(statistics.mean(vals), 3) if vals else None

    gate = gate_report(c, pos, neg)
    multi = [r for r in pos if r["type"] == "multi_doc"]
    follow = [r for r in pos if r["type"] == "follow_up"]
    lat = sorted(r["latency_ms"] for r in per_query)
    return {
        "variant": v.__dict__, "name": v.name, "strategy": c.retriever.strategy, "chunks": c.retriever.chunk_count,
        "metrics": {
            "recall@5": mean("recall@5"), "recall@300w": mean("recall@300w"), "precision@5": mean("precision@5"), "mrr@10": mean("mrr@10"),
            "ndcg@10": mean("ndcg@10"), "ctx_recall": mean("ctx_recall"),
            "doc_coverage_multi": mean("doc_coverage", multi), "recall@5_follow_up": mean("recall@5", follow),
            "mrr@10_follow_up": mean("mrr@10", follow),
            "latency_ms_p50": round(lat[len(lat) // 2], 2), "latency_ms_max": round(lat[-1], 2),
        },
        "gate": gate,
        "per_query": per_query,
    }


def gate_report(c: Container, pos: list[dict], neg: list[dict]) -> dict:
    """Answerability gate at the configured threshold, plus the threshold that would be best on this set."""
    if c.retriever.mode == "bm25" and c.retriever.reranker is None:
        return {"applicable": False, "note": "BM25 scores are unbounded; no gate in bm25-only mode"}
    tau = c.retriever.rerank_threshold if c.retriever.reranker else c.retriever.threshold

    def acc(t):
        tp = sum(1 for r in pos if r["gate_score"] >= t)
        tn = sum(1 for r in neg if r["gate_score"] < t)
        return tp, tn, 0.5 * (tp / len(pos) + tn / len(neg))

    tp, tn, bal = acc(tau)
    scores = sorted({round(r["gate_score"], 4) for r in pos + neg})
    candidates = [(a + b) / 2 for a, b in zip(scores, scores[1:], strict=False)] or [tau]
    best_t = max(candidates, key=lambda t: (acc(t)[2], -abs(t - tau)))
    btp, btn, bbal = acc(best_t)
    return {
        "applicable": True, "score": "rerank" if c.retriever.reranker else "best dense cosine",
        "threshold": tau, "answered_positives": f"{tp}/{len(pos)}", "refused_negatives": f"{tn}/{len(neg)}",
        "balanced_accuracy": round(bal, 3),
        "best_threshold_on_this_set": round(best_t, 3), "best_balanced_accuracy": round(bbal, 3),
        "best_answered_positives": f"{btp}/{len(pos)}", "best_refused_negatives": f"{btn}/{len(neg)}",
        "max_negative_score": round(max(r["gate_score"] for r in neg), 4),
        "min_positive_score": round(min(r["gate_score"] for r in pos), 4),
        "errors": [f"{r['id']} ({'refused' if r['answerable'] else 'answered'}, {r['gate_score']:.4f}): {r['query']}"
                   for r in pos + neg if (r["gate_score"] >= tau) != r["answerable"]],
    }


def markdown(results: list[dict], n_queries: int) -> str:
    lines = [
        "# Retrieval evaluation", "",
        f"{n_queries} labelled queries (`eval/queries.jsonl`), fictional sample corpus. Generated by "
        "`eval/run_retrieval_eval.py`; do not edit by hand.", "",
        "| Variant | Chunks | Recall@5 | Recall@300w | Precision@5 | MRR@10 | nDCG@10 | Ctx recall | Multi-doc coverage | Follow-up Recall@5 | p50 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r["metrics"]
        lines.append(f"| {r['name']} | {r['chunks']} | {m['recall@5']} | {m['recall@300w']} | {m['precision@5']} | {m['mrr@10']} | "
                     f"{m['ndcg@10']} | {m['ctx_recall']} | {m['doc_coverage_multi']} | {m['recall@5_follow_up']} | "
                     f"{m['latency_ms_p50']} |")
    lines += ["", "## Answerability gate", "",
              "| Variant | Score | τ | Answered positives | Refused negatives | Balanced acc. | Max neg. score | Min pos. score | Best τ on this set |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        g = r["gate"]
        if not g.get("applicable"):
            lines.append(f"| {r['name']} | – | – | – | – | – | – | – | {g['note']} |")
            continue
        lines.append(f"| {r['name']} | {g['score']} | {g['threshold']} | {g['answered_positives']} | "
                     f"{g['refused_negatives']} | {g['balanced_accuracy']} | {g['max_negative_score']} | "
                     f"{g['min_positive_score']} | {g['best_threshold_on_this_set']} ({g['best_balanced_accuracy']}) |")
    lines += ["", "Ctx recall and gate use the configured thresholds. 'Best τ' is fitted on these same 30 queries, "
              "so it is optimistic; treat it as a direction, not a guarantee.", "",
              "Recall@300w counts gold evidence within the first 300 words of the ranking, which compares "
              "chunkers fairly (Recall@5 favours bigger chunks: five 180-word chunks cover 3x more text).", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(ROOT / "eval" / "queries.jsonl"))
    ap.add_argument("--rerank", action="store_true", help="include the cross-encoder variant")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()
    queries = load_queries(Path(args.queries))
    results = []
    cache: dict[tuple, Container] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for v in default_matrix(args.rerank):
            if not (EMBEDDERS[v.embedder][1] / "tokenizer.json").exists():
                print(f"skipping '{v.name}': model not downloaded (scripts/download_models.py --with-baseline)")
                continue
            key = (v.chunker, v.chunk_words, v.embedder, v.query_prefix, v.rerank)
            if key not in cache:
                cache[key] = build(v, Path(tmp) / f"{len(cache)}")
            c = cache[key]
            c.retriever.mode = v.mode
            res = score_variant(c, v, queries)
            results.append(res)
            m = res["metrics"]
            print(f"{v.name:42} R@5={m['recall@5']} R@300w={m['recall@300w']} P@5={m['precision@5']} MRR={m['mrr@10']} "
                  f"nDCG={m['ndcg@10']} ctx={m['ctx_recall']} gate={res['gate'].get('balanced_accuracy', '-')}")
            for err in res["gate"].get("errors", []):
                print(f"    gate error: {err}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "retrieval_eval.md").write_text(markdown(results, len(queries)), encoding="utf-8")
    report = markdown(results, len(queries))
    print("\n" + report)
    print(f"wrote {out / 'retrieval_eval.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
