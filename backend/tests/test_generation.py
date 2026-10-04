import asyncio
import json

import httpx
import pytest

from app.domain import Chunk, ChunkMetadata, ScoredChunk
from app.generation.citations import finalize_answer
from app.generation.llm.base import Grounding, LLMError
from app.generation.llm.extractive import ExtractiveLLM
from app.generation.llm.openai_compat import OpenAICompatibleLLM
from app.generation.prompt import REFUSAL_TEXT, build_messages


def _sc(text: str, name: str = "doc.pdf", page: int | None = 3) -> ScoredChunk:
    meta = ChunkMetadata(document_id="d", document_name=name, document_title="D", file_type="pdf",
                         chunk_id=f"d:{abs(hash(text)) % 1000:04d}", chunk_index=0, page=page, section="Healthcare")
    return ScoredChunk(chunk=Chunk(text=text, embed_text=text, metadata=meta), score=0.8)


SOURCES = [
    _sc("Government General Hospital, Vijayawada will receive a 200-bed mother and child block."),
    _sc("The SCA Health Shield covers up to 15 lakh rupees per family.", name="schemes.md", page=None),
]


def test_prompt_numbers_sources_with_metadata_labels():
    user = build_messages("What is planned?", SOURCES)[1]["content"]
    assert "[S1] doc.pdf · p.3 · Healthcare" in user and "[S2] schemes.md · Healthcare" in user
    assert user.endswith("QUESTION: What is planned?")


def test_citations_come_from_metadata_and_invalid_markers_are_stripped():
    answer, cites, invalid = finalize_answer("A 200-bed block is planned [S1]. Cover is 15 lakh [S2, S7].", SOURCES)
    assert "[S7]" not in answer and invalid == [7]
    assert [c.cited for c in cites] == [True, True]
    assert cites[0].document_name == "doc.pdf" and cites[0].page == 3


def test_refusal_answers_cite_nothing():
    _, cites, _ = finalize_answer(REFUSAL_TEXT + " [S1]", SOURCES)
    assert not any(c.cited for c in cites)


def test_extractive_fallback_is_grounded_and_refuses_without_overlap():
    llm = ExtractiveLLM()
    q = "How much does the Health Shield cover?"
    out = asyncio.run(llm.complete(build_messages(q, SOURCES), grounding=Grounding(q, SOURCES)))
    assert "15 lakh" in out and "[S2]" in out
    out = asyncio.run(llm.complete([], grounding=Grounding("lunar mining policy", SOURCES)))
    assert out == REFUSAL_TEXT


def test_extractive_fallback_reads_structured_sources_not_prompt_text():
    llm = ExtractiveLLM()
    out = asyncio.run(llm.complete([{"role": "user", "content": "unrelated text"}],
                                   grounding=Grounding("Health Shield cover", SOURCES)))
    assert "[S2]" in out
    with pytest.raises(LLMError):
        asyncio.run(llm.complete(build_messages("x", SOURCES)))


def _sse(*deltas: str) -> bytes:
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': d}}]})}\n\n" for d in deltas]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def test_openai_compatible_streaming_parses_sse():
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["stream"] is True and body["model"] == "llama-3.1-8b-instant"
        assert req.headers["authorization"] == "Bearer k"
        return httpx.Response(200, content=_sse("Hello", " world"), headers={"content-type": "text/event-stream"})

    llm = OpenAICompatibleLLM("groq", "https://api.groq.com/openai/v1", "llama-3.1-8b-instant", api_key="k",
                              transport=httpx.MockTransport(handler))

    async def run():
        return [t async for t in llm.stream([{"role": "user", "content": "hi"}])]

    assert asyncio.run(run()) == ["Hello", " world"]


def test_openai_compatible_errors_become_llm_error():
    llm = OpenAICompatibleLLM("groq", "https://x", "m",
                              transport=httpx.MockTransport(lambda r: httpx.Response(429, text="rate limited")))
    with pytest.raises(LLMError, match="429"):
        asyncio.run(llm.complete([{"role": "user", "content": "hi"}]))


def test_retired_model_is_replaced_automatically():
    """Groq retires models; a model_not_found error triggers /models discovery and one retry."""
    seen_models = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [
                {"id": "whisper-large-v3", "active": True},
                {"id": "llama-3.3-70b-versatile", "active": True},
                {"id": "openai/gpt-oss-20b", "active": True},
            ]})
        model = json.loads(req.content)["model"]
        seen_models.append(model)
        if model == "llama-3.1-8b-instant":
            return httpx.Response(404, json={"error": {"message": "The model `llama-3.1-8b-instant` does not exist",
                                                       "code": "model_not_found"}})
        return httpx.Response(200, content=_sse("ok"), headers={"content-type": "text/event-stream"})

    llm = OpenAICompatibleLLM("groq", "https://x/openai/v1", "llama-3.1-8b-instant",
                              transport=httpx.MockTransport(handler))

    async def run():
        return [t async for t in llm.stream([{"role": "user", "content": "hi"}])]

    assert asyncio.run(run()) == ["ok"]
    # speech models skipped; a non-reasoning model is preferred over gpt-oss
    assert seen_models == ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    assert llm.model == "llama-3.3-70b-versatile"


def test_model_error_without_alternatives_still_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "whisper-large-v3"}]})
        return httpx.Response(404, json={"error": {"code": "model_not_found"}})

    llm = OpenAICompatibleLLM("groq", "https://x", "gone-model", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="404"):
        asyncio.run(llm.complete([{"role": "user", "content": "hi"}]))


def test_reasoning_models_get_low_effort_and_token_headroom():
    """gpt-oss counts hidden reasoning against max_tokens; 300 tokens cut answers mid-sentence."""
    bodies = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    llm = OpenAICompatibleLLM("groq", "https://x", "openai/gpt-oss-20b", max_tokens=300,
                              transport=httpx.MockTransport(handler))
    asyncio.run(llm.complete([{"role": "user", "content": "hi"}]))
    assert bodies[0]["reasoning_effort"] == "low" and bodies[0]["include_reasoning"] is False
    assert bodies[0]["max_tokens"] > 300

    plain = OpenAICompatibleLLM("groq", "https://x", "llama-3.3-70b-versatile", max_tokens=300,
                                transport=httpx.MockTransport(handler))
    asyncio.run(plain.complete([{"role": "user", "content": "hi"}]))
    assert bodies[1]["max_tokens"] == 300 and "reasoning_effort" not in bodies[1]


def test_rejected_reasoning_parameters_are_dropped_and_retried():
    def handler(req: httpx.Request) -> httpx.Response:
        if "reasoning_effort" in json.loads(req.content):
            return httpx.Response(400, json={"error": {"message": "unsupported parameter: reasoning_effort"}})
        return httpx.Response(200, content=_sse("fine"), headers={"content-type": "text/event-stream"})

    llm = OpenAICompatibleLLM("groq", "https://x", "openai/gpt-oss-20b", transport=httpx.MockTransport(handler))

    async def run():
        return [t async for t in llm.stream([{"role": "user", "content": "hi"}])]

    assert asyncio.run(run()) == ["fine"] and llm.send_reasoning_params is False


def test_truncated_answer_is_marked():
    def handler(req: httpx.Request) -> httpx.Response:
        chunk = {"choices": [{"delta": {"content": "These include completing a"}, "finish_reason": "length"}]}
        return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
                              headers={"content-type": "text/event-stream"})

    llm = OpenAICompatibleLLM("groq", "https://x", "llama-3.3-70b-versatile", transport=httpx.MockTransport(handler))

    async def run():
        return "".join([t async for t in llm.stream([{"role": "user", "content": "hi"}])])

    assert asyncio.run(run()).endswith(" …")
