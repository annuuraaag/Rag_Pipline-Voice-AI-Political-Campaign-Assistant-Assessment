"""Reciprocal Rank Fusion (Cormack et al., 2009).

    rrf(d) = Σ_lists 1 / (k + rank_list(d))

BM25 scores (unbounded, corpus-dependent) and cosine similarities (0–1) live on
different scales, so adding them with weights needs per-corpus tuning. RRF uses
only ranks: one constant, no normalisation, and a chunk found by both retrievers
beats one found by only one. Independent implementation; the idea is standard.
"""
from __future__ import annotations

from app.domain import ScoredChunk


def reciprocal_rank_fusion(ranked_lists: dict[str, list[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    """Fuse named ranked lists. Keeps every retriever's rank/score in `scores` for the trace."""
    fused: dict[str, ScoredChunk] = {}
    totals: dict[str, float] = {}
    for name, ranked in ranked_lists.items():
        seen: set[str] = set()
        for rank, sc in enumerate(ranked, start=1):
            cid = sc.chunk.metadata.chunk_id
            if cid in seen:  # a duplicate inside one list must not count twice
                continue
            seen.add(cid)
            totals[cid] = totals.get(cid, 0.0) + 1.0 / (k + rank)
            entry = fused.setdefault(cid, ScoredChunk(chunk=sc.chunk, score=0.0, retriever="hybrid", scores={}))
            entry.scores[name] = sc.scores.get(name, round(sc.score, 4))
            entry.scores[f"{name}_rank"] = rank
    # Ties (same fused score) break by the better best-rank, then by chunk id for determinism.
    order = sorted(
        fused,
        key=lambda c: (-totals[c], min(v for key, v in fused[c].scores.items() if key.endswith("_rank")), c),
    )
    out = []
    for rank, cid in enumerate(order, start=1):
        sc = fused[cid]
        sc.score = round(totals[cid], 6)
        sc.rank = rank
        sc.scores["rrf"] = sc.score
        out.append(sc)
    return out
