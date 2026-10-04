"""Partial-transcript controller: search while the user is still speaking.

A voice turn produces a stream of partial transcripts ("what healthcare", "what healthcare
schemes are", ...) and then one final transcript. Waiting for the final one and only then
running retrieval puts the whole retrieval pipeline (embedding, ANN, BM25, cross-encoder)
on the critical path between "user stopped talking" and "assistant starts answering".

This controller moves that work into the time the user is still talking:

    S0  ignore   fewer than ``min_words`` words, small talk / self-introduction, or nothing
                 new since the last speculative search.
    S1  search   stage-1 retrieval (dense ‖ BM25 → RRF, ~10–30 ms), debounced ``debounce_s``,
                 when the partial gains a district / topic / named entity or ``word_step``
                 more words. Results are shown in the UI as "searching ahead".
    S2  refine   when the transcript has been stable for ``stable_s`` (or the client reports
                 end of speech): cross-encoder rerank + relevance gate on the cached stage-1
                 candidates (the expensive part, ~200–400 ms on CPU).
    S3  final    the final transcript is rewritten with the committed conversation state.
                 Same normalised query + filters + corpus revision as a speculation →
                 reuse it: "hit" skips retrieval entirely, "stage1" runs only stage 2,
                 otherwise a normal retrieval ("miss"). Then the LLM streams.

End of turn: every partial is also scored by the endpointer (app/voice/endpointing.py). The
score goes to the client as an ``endpoint`` event (how long a silence should end the turn), and
when the question already sounds complete, S2 starts at once instead of after ``stable_s``.

Correctness: a speculative result is reused only for an identical cache key, and the key
includes the corpus revision, so an upload in between invalidates it. Speculation only reads
conversation state (``SessionStore.peek``); memory is committed once, on the final transcript.
The LLM is never called speculatively (it costs money and its answer depends on the final
wording).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.conversation.rewriter import Rewrite, rewrite_with_rules
from app.domain import MetadataFilter
from app.observability.timing import StageTimer
from app.rag.service import RAGService, merge_filters
from app.retrieval.pipeline import RetrievalResult, Stage1
from app.voice.endpointing import EndOfTurn, Endpointer

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")
Emit = Callable[[dict[str, Any]], Awaitable[None]]


def normalise(text: str) -> str:
    """Key form of a query: lowercase words only. Interim and final transcripts often differ
    only in casing and punctuation ("hospitals" vs "Hospitals?"); retrieval treats those alike."""
    return " ".join(_WORD.findall(text.lower()))


def word_count(text: str) -> int:
    return len(_WORD.findall(text.lower()))


@dataclass
class Speculation:
    query: str
    filters: MetadataFilter
    stage1: Stage1 | None = None
    result: RetrievalResult | None = None
    stage1_ms: float = 0.0
    stage2_ms: float = 0.0


@dataclass
class VoiceStats:
    partials: int = 0
    ignored: int = 0
    stage1_runs: int = 0
    stage2_runs: int = 0
    events: list[str] = field(default_factory=list)  # compact log of decisions, for the UI/debugging

    def note(self, what: str) -> None:
        self.events.append(what)
        del self.events[:-40]


class PartialTranscriptController:
    """One per voice connection (it holds that connection's speculative cache)."""

    def __init__(self, rag: RAGService, campaign_id: str, session_id: str | None,
                 filters: MetadataFilter | None = None, revision: Callable[[], int] = lambda: 0,
                 emit: Emit | None = None, *, debounce_s: float = 0.25, stable_s: float = 0.5,
                 min_words: int = 3, word_step: int = 3, cache_size: int = 16,
                 endpointer: Endpointer | None = None):
        self.rag = rag
        self.campaign_id = campaign_id
        self.session_id = session_id
        self.explicit = filters
        self.revision = revision
        self.emit = emit
        self.debounce_s = debounce_s
        self.stable_s = stable_s
        self.min_words = min_words
        self.word_step = word_step
        self.cache_size = cache_size
        self.endpointer = endpointer
        self.end_of_turn: EndOfTurn | None = None  # assessment of the latest partial
        self.stats = VoiceStats()
        self._cache: OrderedDict[tuple, Speculation] = OrderedDict()
        self._inflight: dict[tuple, asyncio.Task] = {}   # stage-1 jobs
        self._inflight2: dict[tuple, asyncio.Task] = {}  # stage-2 jobs
        self._latest = ""
        self._last_words = 0
        self._last_signature: tuple | None = None
        self._last_key: tuple | None = None
        self._debounce: asyncio.Task | None = None
        self._stable: asyncio.Task | None = None
        self._slot = asyncio.Semaphore(1)  # one speculative job at a time; final retrieval never waits on it

    # ── planning (pure, microseconds) ───────────────────────────────────
    def _state(self):
        if not self.session_id:
            return None
        return self.rag.sessions.peek(f"{self.campaign_id}:{self.session_id}")

    def key(self, query: str, filters: MetadataFilter) -> tuple:
        flt = json.dumps(filters.as_dict(), sort_keys=True, default=str)
        return normalise(query), flt, self.revision()

    def plan(self, text: str) -> tuple[Rewrite, MetadataFilter, tuple] | None:
        """Rewrite a partial exactly as the final transcript would be rewritten (rules only)."""
        rw = rewrite_with_rules(text, self._state())
        u = rw.understanding
        if u is not None and (u.intro_only or u.smalltalk):
            return None
        filters = merge_filters(rw.filters, self.explicit).model_copy(update={"campaign_id": self.campaign_id})
        return rw, filters, self.key(rw.query, filters)

    # ── end of turn ─────────────────────────────────────────────────────
    def assess_turn(self, text: str) -> EndOfTurn | None:
        """How finished the utterance sounds, and how long a silence should end it."""
        if self.endpointer is None:
            return None
        return self.endpointer.assess(text, self._state())

    # ── S0 / S1: partial transcripts ────────────────────────────────────
    async def on_partial(self, text: str) -> None:
        text = text.strip()[:1000]
        if not text or text == self._latest:
            return
        self._latest = text
        self.stats.partials += 1
        self.end_of_turn = self.assess_turn(text)
        if self.end_of_turn is not None:
            await self._send(self.end_of_turn.as_event(text))
        # A question that sounds finished will be final within a few hundred ms: refine (rerank + gate)
        # now so it is ready by then, instead of after the usual stability wait.
        complete = self.end_of_turn is not None and self.end_of_turn.turn == "complete"
        self._restart_stability_timer(min(self.stable_s, 0.1) if complete else self.stable_s)
        words = word_count(text)
        if words < self.min_words:
            self._ignore(f"S0 short ({words} words)")
            return
        planned = self.plan(text)
        if planned is None:
            self._ignore("S0 small talk / introduction")
            return
        rw, filters, key = planned
        if key == self._last_key or key in self._cache:
            return
        signature = self._signature(rw, filters)
        new_signal = signature != self._last_signature and any(signature)
        if not (new_signal or self._last_key is None or words - self._last_words >= self.word_step):
            self._ignore("S0 nothing new")
            return
        if self._debounce and not self._debounce.done():
            self._debounce.cancel()  # a newer partial supersedes the pending one
        self._debounce = asyncio.create_task(self._debounced_search())

    @staticmethod
    def _signature(rw: Rewrite, filters: MetadataFilter) -> tuple:
        u = rw.understanding
        return filters.district, u.topic if u else None, u.entity if u else None

    def _ignore(self, why: str) -> None:
        self.stats.ignored += 1
        self.stats.note(why)

    async def _debounced_search(self) -> None:
        await asyncio.sleep(self.debounce_s)
        planned = self.plan(self._latest)  # the latest words at fire time, not when scheduled
        if planned is None:
            return
        rw, filters, key = planned
        self._last_key, self._last_words = key, word_count(self._latest)
        self._last_signature = self._signature(rw, filters)
        await self._guard(self._ensure_stage1(rw.query, filters, key))

    async def _ensure_stage1(self, query: str, filters: MetadataFilter, key: tuple) -> Speculation:
        spec = self._cache.get(key)
        if spec and spec.stage1:
            return spec
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(self._run_stage1(query, filters, key))
            self._inflight[key] = task
            task.add_done_callback(lambda _t, k=key: self._inflight.pop(k, None))
        return await asyncio.shield(task)

    async def _run_stage1(self, query: str, filters: MetadataFilter, key: tuple) -> Speculation:
        async with self._slot:
            timer = StageTimer()
            t = time.perf_counter()
            s1 = await run_in_threadpool(self.rag.retriever.candidates, query, filters, timer)
            spec = Speculation(query=query, filters=filters, stage1=s1,
                               stage1_ms=round((time.perf_counter() - t) * 1000, 2))
        self._remember(key, spec)
        self.stats.stage1_runs += 1
        self.stats.note(f"S1 search '{query}' {spec.stage1_ms} ms")
        await self._send({
            "type": "speculative", "stage": "S1", "query": query, "latency_ms": spec.stage1_ms,
            "sources": [_brief(c) for c in s1.fused[:3]],
        })
        return spec

    # ── S2: stable transcript → rerank + gate ahead of time ─────────────
    def _restart_stability_timer(self, delay_s: float) -> None:
        if self._stable and not self._stable.done():
            self._stable.cancel()
        self._stable = asyncio.create_task(self._refine_when_stable(delay_s))

    async def _refine_when_stable(self, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        await self._guard(self.refine())

    async def speech_end(self) -> None:
        """The client detected end of speech: refine now instead of waiting for stability."""
        if self._stable and not self._stable.done():
            self._stable.cancel()
        await self._guard(self.refine())

    async def refine(self) -> Speculation | None:
        if word_count(self._latest) < self.min_words:
            return None
        planned = self.plan(self._latest)
        if planned is None:
            return None
        rw, filters, key = planned
        spec = await self._ensure_stage1(rw.query, filters, key)
        if spec.result is not None:
            return spec
        task = self._inflight2.get(key)
        if task is None:
            task = asyncio.create_task(self._run_stage2(rw.query, spec))
            self._inflight2[key] = task
            task.add_done_callback(lambda _t, k=key: self._inflight2.pop(k, None))
        # Shielded: if the final transcript arrives mid-rerank, the job keeps going and the
        # final path awaits it instead of starting the same rerank again.
        await asyncio.shield(task)
        return spec

    async def _run_stage2(self, query: str, spec: Speculation) -> None:
        async with self._slot:
            timer = StageTimer()
            t = time.perf_counter()
            result = await run_in_threadpool(self.rag.retriever.finish, spec.stage1, None, None, timer)
            spec.stage2_ms = round((time.perf_counter() - t) * 1000, 2)
            spec.result = result
        self.stats.stage2_runs += 1
        self.stats.note(f"S2 refine '{query}' {spec.stage2_ms} ms")
        await self._send({"type": "speculative", "stage": "S2", "query": query,
                          "latency_ms": spec.stage2_ms, "answerable": result.answerable,
                          "top_score": result.top_score, "sources": [_brief(c) for c in result.results[:3]]})

    # ── S3: final transcript ────────────────────────────────────────────
    def cancel_timers(self) -> None:
        for task in (self._debounce, self._stable):
            if task and not task.done():
                task.cancel()

    async def reuse(self, query: str, filters: MetadataFilter) -> RetrievalResult | Stage1 | None:
        """Called by RAGService.prepare with the *final* rewritten query and filters."""
        key = self.key(query, filters)
        for jobs in (self._inflight, self._inflight2):
            task = jobs.get(key)
            if task is not None:  # a speculation for exactly this query is running: finishing it is cheaper
                try:
                    await asyncio.shield(task)
                except Exception:  # noqa: BLE001 - a failed speculation just means a normal retrieval
                    pass
        spec = self._cache.get(key)
        if spec is None:
            return None
        return spec.result or spec.stage1

    def saved_ms(self, query: str, filters: MetadataFilter, cache: str | None) -> float:
        spec = self._cache.get(self.key(query, filters))
        if not spec or cache not in ("hit", "stage1"):
            return 0.0
        return round(spec.stage1_ms + (spec.stage2_ms if cache == "hit" else 0.0), 2)

    def reset_turn(self) -> None:
        """After a final transcript: the next utterance starts from scratch (cache is kept)."""
        self.cancel_timers()
        self._latest = ""
        self.end_of_turn = None
        self._last_words, self._last_signature, self._last_key = 0, None, None

    # ── helpers ─────────────────────────────────────────────────────────
    def _remember(self, key: tuple, spec: Speculation) -> None:
        self._cache[key] = spec
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    async def _send(self, event: dict[str, Any]) -> None:
        if self.emit is not None:
            try:
                await self.emit(event)
            except Exception:  # noqa: BLE001 - the socket may be closing; speculation is best effort
                logger.debug("voice: could not send %s", event.get("type"))

    async def _guard(self, coro) -> Any:
        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - never let speculation break the conversation
            logger.exception("voice: speculative retrieval failed")
            return None


def _brief(c) -> dict[str, Any]:
    m = c.chunk.metadata
    return {"document_name": m.document_name, "section": m.section, "page": m.page}
