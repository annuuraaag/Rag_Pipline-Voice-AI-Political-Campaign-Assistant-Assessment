import json

from tests.conftest import sample


def upload(client, path: str, **form):
    name = path.rsplit("/", 1)[-1]
    return client.post("/upload", files={"file": (name, sample(path))}, data=form)


def test_health_reports_components(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["components"]["vector_store"]["ok"] is True
    assert body["status"] == "degraded"  # no documents yet


def test_upload_dedup_and_reindex(client):
    r = upload(client, "districts/guntur_district_plan.md")
    assert r.status_code == 200, r.text
    doc = r.json()["document"]
    assert r.json()["status"] == "indexed"
    assert doc["metadata"]["district"] == "guntur" and doc["chunk_count"] > 3
    assert set(r.json()["timings_ms"]) >= {"parse", "chunk", "embed", "index", "total"}

    assert upload(client, "districts/guntur_district_plan.md").json()["status"] == "duplicate"

    changed = sample("districts/guntur_district_plan.md") + b"\n## 7. Roads\n\nNew ring road for Guntur city.\n"
    r = client.post("/upload", files={"file": ("guntur_district_plan.md", changed)})
    assert r.json()["status"] == "reindexed"
    assert r.json()["replaced_document_id"] == doc["document_id"]
    assert len(client.get("/documents").json()) == 1


def test_upload_rejections(client):
    assert client.post("/upload", files={"file": ("x.pptx", b"abc")}).status_code == 415
    assert client.post("/upload", files={"file": ("x.pdf", b"%PDF-broken")}).status_code == 422
    assert client.post("/upload", files={"file": ("x.txt", b"")}).status_code == 422


def test_form_metadata_overrides(client):
    r = upload(client, "schemes/education_schemes.txt", district="Vizag", category="scheme", topic="education")
    meta = r.json()["document"]["metadata"]
    assert meta == {**meta, "district": "visakhapatnam", "category": "scheme", "topic": "education"}


def test_retrieve_returns_scored_chunks_with_metadata_and_filters(client):
    upload(client, "districts/guntur_district_plan.md")
    upload(client, "districts/vijayawada_district_plan.pdf")
    r = client.post("/retrieve", json={"query": "hospital beds in Guntur", "filters": {"district": "guntur"}})
    assert r.status_code == 200
    body = r.json()
    assert body["filters"] == {"campaign_id": "default", "district": "guntur"}
    assert body["results"], body
    assert all(x["district"] in ("guntur", "statewide") for x in body["candidates"])
    first = body["results"][0]
    assert {"score", "document_name", "section", "page", "district", "category", "topic", "text"} <= set(first)
    assert "dense" in body["latency_ms"] and "total" in body["latency_ms"]


def test_vijayawada_pdf_citation_has_page(client, fake_llm):
    upload(client, "districts/vijayawada_district_plan.pdf")
    fake_llm.reply = "A 200-bed mother and child block is planned [S1]."
    r = client.post("/query", json={"query": "mother and child block Government General Hospital Vijayawada"})
    body = r.json()
    assert body["answerable"] is True
    cited = [c for c in body["citations"] if c["cited"]]
    assert cited and cited[0]["document_name"] == "vijayawada_district_plan.pdf" and cited[0]["page"] == 1
    # retrieved context was injected into the prompt
    assert "200-bed mother and child" in fake_llm.calls[-1][1]["content"]


def test_query_with_no_documents_refuses_without_llm(client, fake_llm):
    r = client.post("/query", json={"query": "anything"})
    assert r.json()["refusal_reason"] == "no_documents"
    assert fake_llm.calls == []


def test_below_threshold_refuses_without_calling_llm(client, fake_llm):
    upload(client, "districts/guntur_district_plan.md")
    r = client.post("/query", json={"query": "zzqx", "threshold": 0.99})
    body = r.json()
    assert body["answerable"] is False and body["refusal_reason"] == "below_threshold"
    assert body["citations"] == [] and fake_llm.calls == []


def test_llm_outage_falls_back_to_extractive(client, fake_llm):
    upload(client, "schemes/healthcare_schemes.md")
    fake_llm.fail = True
    body = client.post("/query", json={"query": "How much does the SCA Health Shield cover per family?"}).json()
    assert body["llm"]["fallback"] is True and "simulated outage" in body["llm"]["error"]
    assert "15 lakh" in body["answer"]


def test_streaming_query_emits_retrieval_tokens_done(client):
    upload(client, "districts/guntur_district_plan.md")
    with client.stream("POST", "/query", json={"query": "chilli cold storage in Guntur", "stream": True}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    types = [e["type"] for e in events]
    assert types[0] == "retrieval" and types[-1] == "done" and "token" in types
    done = events[-1]
    assert "first_token" in done["latency_ms"] and done["citations"]


def test_document_chunks_and_delete(client):
    doc = upload(client, "faq/campaign_faq.md").json()["document"]
    chunks = client.get(f"/documents/{doc['document_id']}/chunks").json()
    assert len(chunks) == doc["chunk_count"]
    assert [c["chunk_index"] for c in chunks] == sorted(c["chunk_index"] for c in chunks)
    assert client.delete(f"/documents/{doc['document_id']}").status_code == 200
    assert client.get("/documents").json() == []
    assert client.post("/retrieve", json={"query": "volunteer"}).json()["results"] == []
    assert client.delete("/documents/nope").status_code == 404


def test_stream_error_is_logged_with_request_id_and_hidden_from_client(client, container, monkeypatch, caplog):
    upload(client, "faq/campaign_faq.md")

    async def broken(*_a, **_k):
        raise RuntimeError("qdrant at 10.0.0.5 refused api-key sk-secret")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(container.rag, "stream", broken)
    with caplog.at_level("ERROR"):
        with client.stream("POST", "/query", json={"query": "volunteer", "stream": True}) as r:
            events = [json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    err = events[-1]
    assert err["type"] == "error" and err["request_id"]
    assert "sk-secret" not in err["message"] and err["request_id"] in err["message"]
    assert err["request_id"] in caplog.text and "sk-secret" in caplog.text  # full detail stays server-side


def test_health_reports_retriever_strategy(client, container):
    assert client.get("/health").json()["config"]["retrieval_strategy"] == "hybrid(dense+bm25, rrf)"
    container.retriever.mode = "dense"
    assert client.get("/health").json()["config"]["retrieval_strategy"] == "dense"
    container.retriever.mode = "hybrid"


def test_retired_groq_model_end_to_end(settings, container, monkeypatch):
    """Simulated Groq rejects the configured model: startup switches models and answers come from the LLM."""
    import httpx
    from fastapi.testclient import TestClient

    from app.generation.llm.openai_compat import OpenAICompatibleLLM
    from app.main import create_app

    def groq(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/openai/v1/models":
            return httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-20b", "active": True}]})
        if json.loads(req.content)["model"] == "llama-3.1-8b-instant":
            return httpx.Response(404, json={"error": {"code": "model_not_found"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "A 200-bed block is planned [S1]."}}]})

    llm = OpenAICompatibleLLM("groq", "https://api.groq.com/openai/v1", "llama-3.1-8b-instant",
                              transport=httpx.MockTransport(groq))
    container.llm = llm
    container.rag.llm = llm
    with TestClient(create_app(settings, container)) as client:
        health = client.get("/health").json()["components"]["llm"]
        assert health["model"] == "openai/gpt-oss-20b" and "switched" in health["model_check"]
        upload(client, "districts/vijayawada_district_plan.pdf")
        body = client.post("/query", json={"query": "mother and child block Vijayawada"}).json()
    assert body["llm"] == {"provider": "groq", "model": "openai/gpt-oss-20b", "fallback": False, "error": None}


def test_follow_up_questions_use_conversation_memory(client, fake_llm):
    upload(client, "districts/vijayawada_district_plan.pdf")
    upload(client, "districts/guntur_district_plan.md")
    sid = "demo-session"
    client.post("/query", json={"query": "I'm from Vijayawada.", "session_id": sid})
    body = client.post("/query", json={"query": "What about education?", "session_id": sid}).json()
    tr = body["retrieval"]
    assert tr["original_query"] == "What about education?"
    assert "Vijayawada" in tr["rewritten_query"] and tr["filters"] == {"campaign_id": "default", "district": "vijayawada"}
    assert body["conversation"]["district"] == "vijayawada"
    assert all(c["district"] in ("vijayawada", "statewide") for c in tr["results"])
    # the LLM answers the original question, with the interpretation and recent turns as context
    prompt = fake_llm.calls[-1][1]["content"]
    assert "QUESTION: What about education?" in prompt and "(Interpreted as:" in prompt
    assert "CONVERSATION SO FAR" in prompt

    # /retrieve can inspect the same session without changing it
    r = client.post("/retrieve", json={"query": "What about jobs?", "session_id": sid}).json()
    assert "Vijayawada" in r["rewritten_query"] and r["rewrite_reasons"]
    # without a session there is no memory
    r = client.post("/retrieve", json={"query": "What about education?"}).json()
    assert r["filters"] == {"campaign_id": "default"}


def test_explicit_filters_override_conversation_filters(client):
    upload(client, "districts/guntur_district_plan.md")
    sid = "s2"
    client.post("/query", json={"query": "I'm from Vijayawada.", "session_id": sid})
    r = client.post("/retrieve", json={"query": "hospitals", "session_id": sid,
                                       "filters": {"district": "guntur"}}).json()
    assert r["filters"]["district"] == "guntur"


def test_self_introduction_is_acknowledged_without_retrieval_or_llm(client, fake_llm):
    upload(client, "districts/vijayawada_district_plan.pdf")
    body = client.post("/query", json={"query": "I'm from Vijayawada.", "session_id": "intro"}).json()
    assert body["answer"].startswith("Got it, you're from Vijayawada")
    assert body["retrieval"]["strategy"].startswith("skipped") and body["citations"] == []
    assert fake_llm.calls == [] and body["conversation"]["district"] == "vijayawada"


def test_remembered_district_does_not_make_off_topic_questions_relevant(client):
    upload(client, "districts/vijayawada_district_plan.pdf")
    client.post("/query", json={"query": "I'm from Vijayawada.", "session_id": "neg"})
    r = client.post("/retrieve", json={"query": "What is the policy on lunar mining?", "session_id": "neg"}).json()
    assert "Vijayawada" not in r["rewritten_query"]      # no context words added to off-topic text
    assert r["filters"] == {"campaign_id": "default", "district": "vijayawada"}  # but retrieval stays scoped


def test_small_talk_gets_a_conversational_reply_not_a_refusal(client, fake_llm):
    upload(client, "districts/vijayawada_district_plan.pdf")
    sid = "smalltalk"
    client.post("/query", json={"query": "I'm from Vijayawada.", "session_id": sid})
    body = client.post("/query", json={"query": "yes", "session_id": sid}).json()
    assert body["answerable"] and body["refusal_reason"] is None
    assert "Vijayawada" in body["answer"] and "healthcare" in body["answer"]
    assert body["retrieval"]["strategy"] == "skipped (small talk: affirm)" and fake_llm.calls == []
    assert client.post("/query", json={"query": "Thank you!"}).json()["answer"].startswith("You're welcome")
    # a real question that merely starts with "yes" still goes to retrieval
    r = client.post("/query", json={"query": "yes, what about hospitals?", "session_id": sid}).json()
    assert not r["retrieval"]["strategy"].startswith("skipped")
