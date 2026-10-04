"""Queries must stay responsive while an upload is embedding/indexing."""
import threading
import time

from fastapi.testclient import TestClient

from app.container import build_container
from app.ingestion.registry import DocumentRegistry
from app.main import create_app
from app.retrieval.embedder import HashingEmbedder, PriorityLock
from app.retrieval.vector_store import QdrantStore
from tests.conftest import FakeLLM, sample


class SlowEmbedder(HashingEmbedder):
    """Uses FastEmbedder's real locking: one PriorityLock, held per passage batch, shared with queries."""

    def __init__(self, batch_delay_s: float = 0.25, batches: int = 10):
        super().__init__()
        self._lock = PriorityLock()
        self.batch_delay_s = batch_delay_s
        self.batches = batches
        self.embedding_started = threading.Event()

    def embed_query(self, text):
        with self._lock.high():
            return super().embed_query(text)

    def embed_passages(self, texts):
        self.embedding_started.set()
        for _ in range(self.batches):
            with self._lock.low():
                time.sleep(self.batch_delay_s)
        return super().embed_passages(texts)


def test_query_stays_fast_during_slow_upload(settings):
    embedder = SlowEmbedder()
    c = build_container(settings, embedder=embedder, store=QdrantStore("t", in_memory=True),
                        llm=FakeLLM(), registry=DocumentRegistry(None))
    with TestClient(create_app(settings, c)) as client:
        client.post("/upload", files={"file": ("faq.md", sample("faq/campaign_faq.md"))})

        upload_done = threading.Event()

        def slow_upload():
            client.post("/upload", files={"file": ("guntur.md", sample("districts/guntur_district_plan.md"))})
            upload_done.set()

        t = threading.Thread(target=slow_upload)
        t0 = time.perf_counter()
        t.start()
        assert embedder.embedding_started.wait(5)
        q0 = time.perf_counter()
        r = client.post("/query", json={"query": "How can I volunteer?"})
        query_s = time.perf_counter() - q0
        assert r.status_code == 200 and r.json()["answerable"]
        assert not upload_done.is_set(), "upload finished first; test did not exercise concurrency"
        t.join(10)
        upload_s = time.perf_counter() - t0
    assert upload_s >= 2.0
    assert query_s < 1.0, f"query took {query_s:.2f}s while upload took {upload_s:.2f}s"
    c.store.close()


def test_query_path_never_counts_the_store(client, container, monkeypatch):
    client.post("/upload", files={"file": ("faq.md", sample("faq/campaign_faq.md"))})

    def fail(*_a, **_k):
        raise AssertionError("store.count() called on the query path")

    monkeypatch.setattr(container.store, "count", fail)
    assert client.post("/query", json={"query": "How can I volunteer?"}).json()["answerable"]
    assert client.post("/query", json={"query": "volunteer", "stream": True}).status_code == 200
    assert client.post("/retrieve", json={"query": "volunteer"}).json()["results"]
    assert client.get("/health").json()["components"]["vector_store"]["chunks"] > 0


def test_cached_count_tracks_ingest_and_delete(client, container):
    assert container.retriever.chunk_count == 0
    doc = client.post("/upload", files={"file": ("faq.md", sample("faq/campaign_faq.md"))}).json()["document"]
    assert container.retriever.chunk_count == doc["chunk_count"]
    client.delete(f"/documents/{doc['document_id']}")
    assert container.retriever.chunk_count == 0
    assert client.post("/query", json={"query": "volunteer"}).json()["refusal_reason"] == "no_documents"


def test_priority_lock_lets_a_query_jump_ahead_of_bulk_batches():
    lock = PriorityLock()
    order: list[str] = []
    started = threading.Event()

    def bulk():
        for i in range(5):
            with lock.low():
                started.set()
                order.append(f"batch{i}")
                time.sleep(0.05)

    t = threading.Thread(target=bulk)
    t.start()
    started.wait(1)
    with lock.high():
        order.append("query")
    t.join()
    # the query waited for at most the batch in flight, not the whole bulk job
    assert order.index("query") <= 2
