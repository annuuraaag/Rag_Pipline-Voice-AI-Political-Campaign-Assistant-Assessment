"""POST /upload — ingest a PDF / DOCX / MD / TXT document or an image (OCR) into a campaign."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_container, require_api_key, resolve_campaign
from app.container import Container
from app.ingestion.service import UploadRejected
from app.schemas import UploadResponse

router = APIRouter(tags=["ingestion"])


@router.post("/upload", response_model=UploadResponse, summary="Upload and index a campaign document",
             dependencies=[Depends(require_api_key)])
async def upload(
    file: UploadFile = File(..., description="PDF, DOCX, MD, TXT, or a PNG/JPG image (OCR). Scanned PDF pages are OCR'd."),
    campaign_id: str | None = Form(None, description="Campaign (tenant) the document belongs to"),
    district: str | None = Form(None, description="e.g. vijayawada, guntur, statewide (auto-detected if omitted)"),
    category: str | None = Form(None, description="manifesto | district_profile | candidate_profile | scheme | policy | faq"),
    topic: str | None = Form(None, description="e.g. healthcare, education (auto-detected if omitted)"),
    source: str | None = Form(None, description="Human-readable source label"),
    replace_document_id: str | None = Form(
        None, description="Replace this document explicitly. Without it, an upload replaces only a document "
                          "with the same filename AND the same district."),
    c: Container = Depends(get_container),
) -> UploadResponse:
    max_bytes = c.ingestion.max_upload_bytes
    data = await file.read(max_bytes + 1)
    overrides = {"district": district, "category": category, "topic": topic, "source": source}
    try:
        # Parsing + embedding are CPU-bound; keep the event loop free for queries.
        result = await run_in_threadpool(c.ingestion.ingest, data, file.filename or "", overrides, None,
                                         replace_document_id or None, resolve_campaign(campaign_id, c))
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return UploadResponse(**result.model_dump())
