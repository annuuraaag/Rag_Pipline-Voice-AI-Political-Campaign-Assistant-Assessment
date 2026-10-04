"""POST /retrieve — retrieval only (no LLM), for inspecting retrieval quality independently."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from app.api.deps import get_container, resolve_campaign
from app.container import Container
from app.observability.logging import log_event
from app.schemas import RetrievedChunk, RetrieveRequest, RetrieveResponse

router = APIRouter(tags=["retrieval"])


@router.post("/retrieve", response_model=RetrieveResponse, summary="Run retrieval and return scored chunks")
async def retrieve(req: RetrieveRequest, c: Container = Depends(get_container)) -> RetrieveResponse:
    request_id = uuid.uuid4().hex[:12]
    filters = req.filters.to_domain() if req.filters else None
    campaign = resolve_campaign(req.campaign_id, c)
    p, timings = await c.rag.retrieve_only(req.query, req.session_id, filters, req.top_k, req.threshold,
                                           campaign_id=campaign)
    r = p.retrieval
    log_event("retrieve", request_id=request_id, campaign_id=campaign, strategy=r.strategy, gate=r.gate, filters=r.filters,
              counts=r.counts, top_score=r.top_score, timings_ms=timings)
    return RetrieveResponse(
        request_id=request_id,
        campaign_id=campaign,
        query=req.query,
        rewritten_query=r.query,
        rewrite_reasons=p.rewrite.reasons,
        filters=r.filters,
        filter_relaxed=r.filter_relaxed,
        strategy=r.strategy,
        gate=r.gate,
        threshold=r.threshold,
        top_score=r.top_score,
        answerable=r.answerable,
        counts=r.counts,
        results=[RetrievedChunk.from_scored(x, i) for i, x in enumerate(r.results, start=1)],
        candidates=[RetrievedChunk.from_scored(x, i) for i, x in enumerate(r.candidates, start=1)]
        if req.include_candidates else [],
        latency_ms=timings,
    )
