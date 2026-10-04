# Design note: real-time RAG for a voice campaign assistant

**Goal.** Citizens ask a campaign assistant questions by voice ("I'm from Vijayawada, any new
hospitals?") and get a short spoken answer that is grounded in the campaign's own documents,
cites them, and says "not in the documents" instead of guessing. It has to feel like a
conversation: answers must start quickly, follow-ups must work, and the user must be able to
interrupt. All data here is fictional; the assistant is informational, never persuasive.

![Architecture](architecture.png)

## 1. Retrieval: decisions and evidence

Every choice below was measured on 32 labelled queries (direct, district, multi-document,
follow-up, disfluent voice-style, exact names, and 7 that must be refused). Labels are evidence
phrases, not chunk ids, so different chunkers are scored fairly
([retrieval_eval.md](../eval/results/retrieval_eval.md)).

| Decision | Why | Measured |
|---|---|---|
| Section-aware chunks, ≤256 tokens (model tokenizer), one-sentence overlap, contextual header for embedding only | Campaign documents are organised by district and topic; a chunk should not straddle "Healthcare" and "Education" | Recall within the first 300 retrieved words 0.85 vs 0.71 (fixed 180-word chunks) |
| bge-small-en-v1.5 on ONNX (no PyTorch), with its query instruction | 512-token window fits header + chunk; CPU inference in single-digit ms | instruction on vs off: Recall@5 0.78 vs 0.75 |
| Hybrid dense + BM25, fused with RRF (k = 60) | Scheme names, numbers and place names are exact-match problems; RRF needs no score calibration | Recall@5 0.81 vs 0.78 dense, 0.74 BM25 |
| Conversation memory + rule-based query rewriting (no LLM on the hot path) | "What about education?" means nothing on its own; a rule rewrite costs ~0.2 ms and runs on every partial transcript | Recall@5 0.81 → 0.95; follow-ups 0.33 → 0.92 |
| District filter = district ∪ statewide, relaxed if it leaves < 3 candidates; campaign filter always | A Vijayawada citizen also needs statewide schemes; tenant isolation must never be relaxed | — |
| Cross-encoder rerank of the top 12, used as the relevance gate (τ = 0.15) | Dense cosine separates on-topic from off-topic poorly (margin ~0 on this set); the cross-encoder's probability is calibrated | 7/7 off-topic questions declined vs 6/7; Recall@5 0.94, MRR 0.95 (Docker run) |
| Top-4 context with at most 2 passages per document | Questions often need the manifesto + a district plan + a scheme guide | multi-document coverage reported per query |
| Citations built from metadata; the model only writes [S#] markers | The model cannot invent a file, page or section; invalid markers are stripped | 100% of answers cite (answer eval) |
| The gate refuses before the LLM is called | A refusal costs nothing and cannot be hallucinated | — |

## 2. The real-time voice path

Latency a user perceives is **end of speech → first spoken word**. It is spent in four places:
detecting that the user finished (endpointing), retrieval, the LLM's time to first token, and
the speech engine starting. The design attacks each one.

1. **Streaming recognition, own endpointing.** The browser's Web Speech API streams partial
   transcripts. Chrome's single-shot mode cuts questions at the first pause, so recognition runs
   continuously and the client decides the end: 900 ms without new words (adjustable in
   Settings), or immediately when the user releases Space (push-to-talk).
2. **Speculative retrieval on partial transcripts** (the controller in `app/voice/controller.py`):
   - **S0:** partials under 3 words, small talk and self-introductions are ignored.
   - **S1:** when a partial gains a district, topic or named entity, or three more words, the
     cheap stage-1 search (dense ‖ BM25 → RRF) runs, debounced by 250 ms.
   - **S2:** once the transcript has been stable for 500 ms, or the recognizer reports the end of
     speech, the expensive stage (rerank + gate) runs on the cached candidates. This happens
     *during the endpointing silence*, which the user is waiting through anyway.
   - **S3:** the final transcript is rewritten with the committed memory. If the normalised query,
     filters and corpus revision match a speculation, it is reused and the LLM starts at once.
   - **Correctness rules.** Speculation only reads memory, the key includes the corpus revision,
     and the LLM is never called speculatively, because it costs money and its answer depends on
     the exact final wording.
3. **Speak while generating.** Tokens are cut into sentences as they stream; each sentence is
   spoken as soon as it is complete (decimals and "Dr." don't split). Citations are stripped from
   speech but shown on screen.
4. **Barge-in.** Tapping the mic, pressing Space, or (in conversation mode) simply speaking stops
   speech output immediately, cancels the server-side generation, and starts listening. While the
   assistant talks, a transcript that mostly repeats its own words is treated as echo, and echo
   picked up before the user cut in is trimmed from the new question.
5. **Fallbacks.** No Web Speech API (Firefox) → record with voice-activity detection and
   `POST /transcribe` (Whisper via Groq, with a vocabulary prompt of place and document names).
   WebSocket down → the same answer over SSE. No LLM key → a labelled extractive answer that quotes
   the passages.

**Measured** ([latency_benchmark.md](../eval/results/latency_benchmark.md); speech simulated at
250 ms per word, final transcript 900 ms after the last word):

- **Reuse:** the speculation was reused in full for 31 of 32 questions.
- **Effect:** retrieval left the critical path entirely (p50 0 ms vs 13.8 ms without speculation in
  that run, which had no reranker).
- **With the reranker:** stage 2 adds a measured ~230 ms p50 on CPU (Docker run). S2 hides exactly
  this cost inside the endpointing silence; run `run_latency_benchmark.py --rerank` to measure the saving.
- **Biggest cost:** endpointing (900 ms) dominates end of speech → first word. That is why it is
  configurable, and why push-to-talk skips it.

## 3. Reliability and safety

- **Tenant isolation.** `campaign_id` is enforced in the retriever: no code path can search without it.
- **Grounding.** The prompt allows only the numbered sources, treats source text as data (prompt
  injection), keeps answers neutral, and makes them 2–4 speakable sentences.
- **Hardening.**
  - Write endpoints are protected by `API_KEY`, and the WebSocket checks `Origin`.
  - Errors reach clients only as a reference id.
  - Retired Groq models are replaced automatically; reasoning models get token headroom so answers aren't cut off.
- **Ingestion is resilient.**
  - Documents move through `uploaded → … → indexed`; on failure, partial chunks are rolled back.
  - OCR runs on scanned pages, and broken emoji are cleaned.
  - Duplicates are detected by content hash, and queries take priority over uploads on the shared ONNX session.

## 4. Evaluation summary

| Area | Result | Source |
|---|---|---|
| Retrieval | Recall@5 0.95, MRR 0.91, context recall 0.89 (default pipeline, no reranker) | `retrieval_eval.md` |
| Reranker | 7/7 off-topic declined; every off-topic score ≤ 0.0008, answered questions ≈ 0.9 and above | `rerank_docker_run.md` |
| Answers | decision accuracy 97%, citation accuracy 92%, number faithfulness 100% (extractive fallback, so faithful by construction; run with Groq for the real LLM) | `answer_eval.md` |
| Voice | 31/32 speculative reuse | `latency_benchmark.md` |
| Tests | 120 backend tests; 15-check Playwright test of the full voice flow | `backend/tests`, `frontend/e2e` |

## 5. Limitations and next steps

- **Browser speech recognition.** Chrome sends audio to Google's service. A self-hosted streaming
  STT (e.g. Whisper streaming or Deepgram) would remove that dependency and add server-side
  endpointing and word timings.
- **Endpointing.** A semantic end-of-turn model, which knows "…for chilli farmers in" is
  unfinished, could cut the fixed 900 ms wait roughly in half.
- **Speculative generation.** Starting the LLM at S2 and discarding it if the final transcript
  differs would hide time to first token too, at the cost of wasted tokens.
- **Telugu and code-mixed speech.** This needs multilingual embeddings (e.g. bge-m3) and
  transliteration-aware place names.
- **Scale.**
  - BM25 is in memory per process; Qdrant sparse vectors would make it shared.
  - The evaluation set (32 queries) is small. It should grow from real logs, with an LLM judge
    calibrated against human labels.
