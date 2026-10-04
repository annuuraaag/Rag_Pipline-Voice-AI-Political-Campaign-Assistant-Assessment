"""Document browser: list documents, inspect chunks, delete."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_container, require_api_key, resolve_campaign
from app.container import Container
from app.domain import MetadataFilter
from app.ingestion.registry import DocumentRecord
from app.schemas import DocumentChunk

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=list[DocumentRecord],
            summary="List a campaign's documents with their processing status")
def list_documents(campaign_id: str | None = Query(None, description="Defaults to the server's default campaign"),
                   c: Container = Depends(get_container)) -> list[DocumentRecord]:
    return c.registry.all(resolve_campaign(campaign_id, c))


@router.get("/documents/{document_id}/chunks", response_model=list[DocumentChunk], summary="Inspect a document's chunks")
def document_chunks(document_id: str, c: Container = Depends(get_container)) -> list[DocumentChunk]:
    if c.registry.get(document_id) is None:
        raise HTTPException(404, f"Unknown document '{document_id}'")
    chunks = c.store.iter_chunks(MetadataFilter(document_ids=[document_id]))
    return [
        DocumentChunk(
            chunk_id=ch.metadata.chunk_id, chunk_index=ch.metadata.chunk_index, page=ch.metadata.page,
            page_end=ch.metadata.page_end, section=ch.metadata.section, district=ch.metadata.district,
            districts_mentioned=ch.metadata.districts_mentioned, category=ch.metadata.category,
            topic=ch.metadata.topic, word_count=ch.metadata.word_count, token_count=ch.metadata.token_count,
            content_type=ch.metadata.content_type, embedding_model=ch.metadata.embedding_model,
            campaign_id=ch.metadata.campaign_id, text=ch.text, embed_text=ch.embed_text,
        )
        for ch in chunks
    ]


@router.delete("/documents/{document_id}", response_model=DocumentRecord, summary="Delete a document and its chunks",
               dependencies=[Depends(require_api_key)])
def delete_document(document_id: str, c: Container = Depends(get_container)) -> DocumentRecord:
    rec = c.ingestion.delete(document_id)
    if rec is None:
        raise HTTPException(404, f"Unknown document '{document_id}'")
    return rec


@router.delete("/campaigns/{campaign_id}", summary="Delete every document and chunk of a campaign",
               dependencies=[Depends(require_api_key)])
def delete_campaign(campaign_id: str, c: Container = Depends(get_container)) -> dict:
    cid = resolve_campaign(campaign_id, c)
    return {"campaign_id": cid, "deleted_documents": c.ingestion.delete_campaign(cid)}
