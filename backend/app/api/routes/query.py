"""POST /query — full RAG: retrieval → gate → grounded LLM → citations.

`stream: false` → JSON (QueryResponse). `stream: true` → Server-Sent Events:
    event: retrieval  (sources + trace, sent before generation starts)
    event: token      (answer text deltas)
    event: done       (final answer, citations, latency breakdown)
"""
from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_container, resolve_campaign
from app.container import Container
from app.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["rag"])
logger = logging.getLogger(__name__)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question from the campaign documents",
    responses={200: {"content": {"text/event-stream": {}}, "description": "JSON, or SSE when stream=true"}},
)
async def query(req: QueryRequest, c: Container = Depends(get_container)):
    filters = req.filters.to_domain() if req.filters else None
    campaign = resolve_campaign(req.campaign_id, c)
    if not req.stream:
        return await c.rag.answer(req.query, filters, req.top_k, req.threshold, session_id=req.session_id,
                                  campaign_id=campaign)

    request_id = uuid.uuid4().hex[:12]

    async def sse():
        try:
            async for event in c.rag.stream(req.query, filters, req.top_k, req.threshold, request_id=request_id,
                                            session_id=req.session_id, campaign_id=campaign):
                yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
        except Exception:
            # Full details go to the server log; the client gets a generic message plus an id
            # to quote, so internal errors (hosts, keys, stack details) never reach the browser.
            logger.exception("Streaming query failed request_id=%s", request_id)
            payload = {"type": "error", "request_id": request_id,
                       "message": f"Something went wrong while answering. Reference: {request_id}"}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
