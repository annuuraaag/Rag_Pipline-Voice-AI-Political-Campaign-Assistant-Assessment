"""Test fixtures: real pipeline, fake heavy dependencies.

Only the expensive/external pieces are faked (embedding model → feature hashing,
Qdrant server → in-memory Qdrant, LLM API → scripted responses). Parsing,
chunking, metadata, retrieval logic, citations and the API are the real code.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import build_container
from app.generation.llm.base import LLMError
from app.ingestion.registry import DocumentRegistry
from app.main import create_app
from app.retrieval.embedder import HashingEmbedder
from app.retrieval.vector_store import QdrantStore

REPO = Path(__file__).resolve().parents[2]
SAMPLES = REPO / "sample_data"


class FakeLLM:
    """Scripted LLM. `reply` may reference sources as [S1] etc. `fail=True` simulates an outage."""

    name = "fake"
    model = "fake-model"
    is_fallback = False

    def __init__(self, reply: str = "The plan adds a 200-bed hospital block [S1].", fail: bool = False):
        self.reply = reply
        self.fail = fail
        self.calls: list[list[dict[str, str]]] = []

    async def stream(self, messages, max_tokens=None, grounding=None) -> AsyncIterator[str]:
        self.calls.append(messages)
        if self.fail:
            raise LLMError("simulated outage")
        for tok in self.reply.split(" "):
            yield tok + " "

    async def complete(self, messages, max_tokens=None, grounding=None) -> str:
        self.calls.append(messages)
        if self.fail:
            raise LLMError("simulated outage")
        return self.reply


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        similarity_threshold=0.15,   # hashing-embedder cosine scale, not bge's
        llm_provider="extractive",
        seed_sample_data=False,
        log_level="WARNING",
        _env_file=None,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def container(settings, fake_llm):
    c = build_container(
        settings,
        embedder=HashingEmbedder(),
        store=QdrantStore("test", in_memory=True),
        llm=fake_llm,
        registry=DocumentRegistry(None),
        reranker=None,   # no model download in unit tests; reranking is tested with a fake below
    )
    yield c
    c.store.close()


@pytest.fixture
def client(settings, container):
    with TestClient(create_app(settings, container)) as tc:
        yield tc


def sample(path: str) -> bytes:
    return (SAMPLES / path).read_bytes()
