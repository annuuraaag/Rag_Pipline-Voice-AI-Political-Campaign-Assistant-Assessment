"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, health, query, retrieve, upload, voice
from app.config import Settings, get_settings
from app.container import Container, build_container, seed_sample_data
from app.domain import MetadataFilter
from app.observability.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    origins = settings.cors_allow_origins()
    if "*" in [o.strip() for o in settings.cors_origins.split(",")]:
        if settings.api_key:
            logger.warning("CORS_ORIGINS=* is ignored because API_KEY is set; allowed origins: %s. "
                           "Set CORS_ORIGINS to your UI's origin if it is served from another host.", origins or "none")
        else:
            logger.warning("CORS_ORIGINS=* lets any website call this API from a browser. "
                           "Set CORS_ORIGINS and API_KEY before exposing it beyond localhost.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = container or build_container(settings)
        app.state.container = c
        if settings.seed_sample_data and c.registry.campaigns().get(settings.default_campaign_id, 0) == 0:
            logger.info("Seeding sample corpus: %s", seed_sample_data(c, settings.sample_data_dir))
        # Warm-up: first ONNX inference / ANN search allocate buffers; pay that at startup, not on the first query.
        c.embedder.embed_query("warm up")
        c.retriever.retrieve("warm up", MetadataFilter(campaign_id=settings.default_campaign_id))
        if c.ocr:
            await asyncio.to_thread(c.ocr.self_test)
        # Providers retire models; check now so the first user question doesn't pay for (or fail on) it.
        verify = getattr(c.llm, "verify_model", None)
        if verify:
            try:
                await asyncio.wait_for(verify(), timeout=8)
            except Exception as exc:  # never block startup on the LLM provider
                logger.warning("LLM model check skipped: %s", exc)
        logger.info("Ready: %s, %d chunks, OCR=%s, LLM=%s/%s", c.registry.campaigns(), c.store.count(),
                    "on" if c.ocr and c.ocr.available else "off", c.llm.name, c.llm.model)
        yield
        for client in (c.llm, c.transcriber):
            aclose = getattr(client, "aclose", None)
            if aclose:
                await aclose()
        if container is None:
            c.store.close()

    app = FastAPI(
        title="Campaign Voice RAG API",
        version="0.1.0",
        description=(
            "Real-time Retrieval-Augmented Generation for a voice campaign assistant. "
            "Answers are grounded in uploaded campaign documents and cite their sources. "
            "All sample data is fictional."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for module in (health, upload, documents, retrieve, query, voice):
        app.include_router(module.router)
    return app


# `uvicorn app.main:app`. Cheap to construct: heavy services are built in the lifespan.
app = create_app()
