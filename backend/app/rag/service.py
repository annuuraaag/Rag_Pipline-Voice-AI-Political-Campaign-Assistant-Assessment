"""RAG orchestration.

    utterance → conversation state → query rewrite (+ filters) → hybrid retrieval → rerank
              → answerability gate → grounded generation → citations → update memory

Three entry points share one preparation path:
* ``retrieve_only()`` – POST /retrieve (no LLM, read-only on conversation state)
* ``answer()``        – POST /query JSON
* ``stream()``        – SSE / WebSocket: ``retrieval`` event as soon as sources are known,
  then ``token`` events, then ``done`` with citations + latency.

The LLM is never called when the gate refuses, so refusals cost nothing and cannot
be hallucinated. The *original* question is what the LLM answers; the rewritten
query is used for retrieval and shown to the LLM only as an interpretation hint.
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.conversation.rewriter import LLMRewriter, Rewrite, rewrite_with_rules
from app.conversation.state import ConversationState, SessionStore
from app.domain import DEFAULT_CAMPAIGN, MetadataFilter
from app.generation.citations import finalize_answer, is_refusal
from app.generation.llm.base import Grounding, LLMError, LLMProvider
from app.generation.llm.extractive import ExtractiveLLM
from app.generation.prompt import REFUSAL_TEXT, build_messages
from app.observability.logging import log_event
from app.observability.metrics import LatencyMetrics
from app.observability.timing import StageTimer
from app.retrieval.pipeline import RetrievalResult, Retriever, Stage1
from app.schemas import LLMInfo, QueryResponse, RetrievalTrace, RetrievedChunk

NO_DOCUMENTS_TEXT = "No campaign documents are indexed yet. Please upload documents first."


def acknowledgement(district: str) -> str:
    from app.lexicon import DISTRICT_DISPLAY

    name = DISTRICT_DISPLAY.get(district, district.title())
    return f"Got it, you're from {name}. What would you like to know about the campaign's plans for {name}?"


TOPIC_HINT = "healthcare, education, jobs and skills, farmers, or flood protection and roads"


def smalltalk_reply(kind: str, district: str | None) -> str:
    """Short, speakable replies that keep a voice conversation moving instead of refusing."""
    from app.lexicon import DISTRICT_DISPLAY

    place = f" for {DISTRICT_DISPLAY.get(district, district.title())}" if district else ""
    if kind == "greeting":
        return f"Hello! I can answer questions from the campaign documents{place}, for example about {TOPIC_HINT}."
    if kind == "thanks":
        return "You're welcome. Is there anything else you'd like to know?"
    if kind == "bye":
        return "Thank you for your time. Goodbye!"
    return f"Sure. You can ask about the campaign's plans{place} on {TOPIC_HINT}. What would you like to know?"


def merge_filters(auto: MetadataFilter, explicit: MetadataFilter | None) -> MetadataFilter:
    """Explicit (UI/API) filters win field by field over filters inferred from the conversation."""
    if explicit is None:
        return auto
    return MetadataFilter(**{**auto.as_dict(), **explicit.as_dict()})


@dataclass
class Prepared:
    rewrite: Rewrite
    filters: MetadataFilter
    retrieval: RetrievalResult
    state: ConversationState | None
    ack: str | None = None  # set for a pure self-introduction: answered without retrieval or LLM
    cache: str | None = None  # voice path: "hit" (full result reused) | "stage1" (candidates reused) | "miss"


# Voice path: given (rewritten query, filters), return a speculative result computed while the
# user was still speaking — a full RetrievalResult, stage-1 candidates, or None.
Reuse = Callable[[str, MetadataFilter], Awaitable["RetrievalResult | Stage1 | None"]]


class RAGService:
    def __init__(self, retriever: Retriever, llm: LLMProvider, metrics: LatencyMetrics,
                 sessions: SessionStore | None = None, llm_rewriter: LLMRewriter | None = None,
                 log_query_text: bool = True):
        self.retriever = retriever
        self.llm = llm
        self.fallback = ExtractiveLLM()
        self.metrics = metrics
        self.sessions = sessions or SessionStore()
        self.llm_rewriter = llm_rewriter
        self.log_query_text = log_query_text
        self.default_campaign = DEFAULT_CAMPAIGN

    # ── shared steps ────────────────────────────────────────────────────
    async def prepare(self, query: str, session_id: str | None, explicit: MetadataFilter | None,
                      top_k: int | None, threshold: float | None, timer: StageTimer, *, commit: bool,
                      campaign_id: str | None = None, reuse: Reuse | None = None) -> Prepared:
        campaign = campaign_id or self.default_campaign
        if session_id:
            key = f"{campaign}:{session_id}"  # conversation memory is per campaign too
            state = self.sessions.get(key) if commit else self.sessions.peek(key)
        else:
            state = None
        with timer.stage("rewrite"):
            rw = await self.llm_rewriter.rewrite(query, state) if self.llm_rewriter else rewrite_with_rules(query, state)
        filters = merge_filters(rw.filters, explicit).model_copy(update={"campaign_id": campaign})
        u = rw.understanding
        if u is not None and u.intro_only:
            # "I'm from Vijayawada." carries context, not a question: remember it, don't search.
            r = RetrievalResult(query=rw.query, filters=filters.as_dict(), threshold=self.retriever.threshold,
                                strategy="skipped (self-introduction)", gate="none", has_documents=True)
            return Prepared(rw, filters, r, state, ack=acknowledgement(u.districts[0]))
        if u is not None and u.smalltalk:
            # "yes", "thanks", "hi": no information need; respond conversationally, skip search and LLM.
            r = RetrievalResult(query=rw.query, filters=filters.as_dict(), threshold=self.retriever.threshold,
                                strategy=f"skipped (small talk: {u.smalltalk})", gate="none", has_documents=True)
            return Prepared(rw, filters, r, state, ack=smalltalk_reply(u.smalltalk, state.district if state else None))
        cached = await reuse(rw.query, filters) if reuse and top_k is None and threshold is None else None
        # Retrieval is CPU-bound (ONNX + ANN); run it off the event loop.
        with timer.stage("retrieval_total"):
            if isinstance(cached, RetrievalResult):
                r, cache = cached, "hit"
            elif isinstance(cached, Stage1):
                r, cache = await run_in_threadpool(self.retriever.finish, cached, None, None, timer), "stage1"
            else:
                r = await run_in_threadpool(self.retriever.retrieve, rw.query, filters, top_k, threshold, timer)
                cache = "miss" if reuse else None
        return Prepared(rw, filters, r, state, cache=cache)

    @staticmethod
    def trace(p: Prepared) -> RetrievalTrace:
        r = p.retrieval
        return RetrievalTrace(
            original_query=p.rewrite.original, rewritten_query=r.query, rewrite_reasons=p.rewrite.reasons,
            rewrite_method=p.rewrite.method, filters=r.filters, filter_relaxed=r.filter_relaxed,
            strategy=r.strategy, gate=r.gate, counts=r.counts, threshold=r.threshold, top_score=r.top_score,
            candidate_count=len(r.candidates), final_count=len(r.results),
            results=[RetrievedChunk.from_scored(c, i) for i, c in enumerate(r.results, start=1)],
            candidates=[RetrievedChunk.from_scored(c, i) for i, c in enumerate(r.candidates, start=1)],
        )

    @staticmethod
    def _refusal(r: RetrievalResult) -> tuple[str, str] | None:
        if not r.has_documents:
            return NO_DOCUMENTS_TEXT, "no_documents"
        if not r.answerable:
            return REFUSAL_TEXT, "below_threshold"
        return None

    @staticmethod
    def _messages(question: str, p: Prepared) -> tuple[list[dict[str, str]], Grounding]:
        history = [(t.role, t.text) for t in list(p.state.turns)[-2:]] if p.state else None
        msgs = build_messages(question, p.retrieval.results, interpreted=p.rewrite.query, history=history)
        return msgs, Grounding(p.rewrite.query, p.retrieval.results)

    @staticmethod
    def _remember(p: Prepared, answer: str) -> None:
        if p.state is not None and p.rewrite.understanding is not None:
            p.state.observe_user(p.rewrite.understanding, p.rewrite.query)
            p.state.observe_assistant(answer)

    def _log(self, request_id: str, p: Prepared, refusal: str | None, timings: dict, llm: LLMInfo,
             citations: list) -> None:
        r = p.retrieval
        log_event(
            "query", request_id=request_id, campaign_id=p.filters.campaign_id,
            query=p.rewrite.original[:200] if self.log_query_text else None,
            rewritten_query=r.query if self.log_query_text else None,
            strategy=r.strategy, gate=r.gate, filters=r.filters, filter_relaxed=r.filter_relaxed, counts=r.counts,
            top_score=r.top_score, threshold=r.threshold, decision=refusal or "answered",
            llm=llm.model_dump(), timings_ms=timings,
            source_ids=[c.chunk_id for c in citations if getattr(c, "cited", False)],
        )

    # ── retrieval only ──────────────────────────────────────────────────
    async def retrieve_only(self, query: str, session_id: str | None = None, filters: MetadataFilter | None = None,
                            top_k: int | None = None, threshold: float | None = None,
                            campaign_id: str | None = None) -> tuple[Prepared, dict]:
        timer = StageTimer()
        p = await self.prepare(query, session_id, filters, top_k, threshold, timer, commit=False,
                               campaign_id=campaign_id)
        timings = timer.as_dict()
        self.metrics.record("retrieve", timings)
        return p, timings

    # ── JSON path ───────────────────────────────────────────────────────
    async def answer(self, query: str, filters: MetadataFilter | None = None, top_k: int | None = None,
                     threshold: float | None = None, session_id: str | None = None,
                     campaign_id: str | None = None) -> QueryResponse:
        request_id = uuid.uuid4().hex[:12]
        timer = StageTimer()
        p = await self.prepare(query, session_id, filters, top_k, threshold, timer, commit=True,
                               campaign_id=campaign_id)
        llm_info = LLMInfo(provider=self.llm.name, model=self.llm.model, fallback=self.llm.is_fallback)

        refusal = None if p.ack else self._refusal(p.retrieval)
        if p.ack:
            answer, reason, citations = p.ack, None, []
            llm_info = LLMInfo(provider="none", model="none")
        elif refusal:
            answer, reason = refusal
            citations: list = []
            llm_info = LLMInfo(provider="none", model="none")
        else:
            messages, grounding = self._messages(query, p)
            with timer.stage("llm"):
                try:
                    raw = await self.llm.complete(messages, grounding=grounding)
                except LLMError as exc:
                    raw = await self.fallback.complete(messages, grounding=grounding)
                    llm_info = LLMInfo(provider=self.fallback.name, model=self.fallback.model, fallback=True,
                                       error=str(exc)[:200])
            if not llm_info.fallback:
                llm_info.model = self.llm.model  # may differ from config if a retired model was replaced
            answer, citations, _ = finalize_answer(raw, p.retrieval.results)
            reason = "model_refused" if is_refusal(answer) else None

        self._remember(p, answer)
        timings = timer.as_dict()
        self.metrics.record("query", timings)
        self._log(request_id, p, reason, timings, llm_info, citations)
        return QueryResponse(
            request_id=request_id, query=query, answer=answer, answerable=reason is None, refusal_reason=reason,
            citations=citations, retrieval=self.trace(p), llm=llm_info, latency_ms=timings,
            conversation=p.state.summary() if p.state else None,
        )

    # ── streaming path ──────────────────────────────────────────────────
    async def stream(self, query: str, filters: MetadataFilter | None = None, top_k: int | None = None,
                     threshold: float | None = None, request_id: str | None = None,
                     session_id: str | None = None, campaign_id: str | None = None,
                     reuse: Reuse | None = None) -> AsyncIterator[dict[str, Any]]:
        # The caller may supply the id so its error handler can log and report the same one.
        request_id = request_id or uuid.uuid4().hex[:12]
        timer = StageTimer()
        p = await self.prepare(query, session_id, filters, top_k, threshold, timer, commit=True,
                               campaign_id=campaign_id, reuse=reuse)
        yield {"type": "retrieval", "request_id": request_id, "retrieval": self.trace(p).model_dump(),
               "latency_ms": timer.as_dict(), "cache": p.cache}

        refusal = None if p.ack else self._refusal(p.retrieval)
        llm_info = LLMInfo(provider=self.llm.name, model=self.llm.model, fallback=self.llm.is_fallback)
        parts: list[str] = []
        if p.ack or refusal:
            text, reason = (p.ack, None) if p.ack else refusal
            llm_info = LLMInfo(provider="none", model="none")
            timer.mark("first_token")
            yield {"type": "token", "text": text}
            citations: list = []
            answer = text
        else:
            messages, grounding = self._messages(query, p)
            with timer.stage("llm"):
                try:
                    async for tok in self.llm.stream(messages, grounding=grounding):
                        timer.mark("first_token")
                        parts.append(tok)
                        yield {"type": "token", "text": tok}
                except (asyncio.CancelledError, GeneratorExit):
                    # Barge-in: the user spoke over the answer. Keep what they heard in memory.
                    if parts:
                        self._remember(p, finalize_answer("".join(parts), p.retrieval.results)[0] + " …")
                    raise
                except LLMError as exc:
                    # Before any token: switch to the grounded fallback transparently.
                    # After tokens: keep what was streamed and report the error.
                    llm_info.error = str(exc)[:200]
                    if not parts:
                        llm_info = LLMInfo(provider=self.fallback.name, model=self.fallback.model, fallback=True,
                                           error=llm_info.error)
                        async for tok in self.fallback.stream(messages, grounding=grounding):
                            timer.mark("first_token")
                            parts.append(tok)
                            yield {"type": "token", "text": tok}
            if not llm_info.fallback:
                llm_info.model = self.llm.model
            answer, citations, _ = finalize_answer("".join(parts), p.retrieval.results)
            reason = "model_refused" if is_refusal(answer) else None

        self._remember(p, answer)
        timings = timer.as_dict()
        self.metrics.record("query_stream", timings)
        self._log(request_id, p, reason, timings, llm_info, citations)
        yield {
            "type": "done", "request_id": request_id, "answer": answer, "answerable": reason is None,
            "refusal_reason": reason, "citations": [c.model_dump() for c in citations],
            "llm": llm_info.model_dump(), "latency_ms": timings, "cache": p.cache,
            "conversation": p.state.summary() if p.state else None,
        }
