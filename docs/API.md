# API reference

Base URL: `http://localhost:8000` (in the Docker web app every path is also served under `/api`,
e.g. `http://localhost:5173/api/health`). Interactive docs: `/docs` (Swagger) and `/openapi.json`.

- **Auth.** None by default. When the server sets `API_KEY`, write endpoints (`POST /upload`,
  `DELETE …`) need the header `X-API-Key: <key>`. Read endpoints, `/query` and the voice socket stay open.
- **Campaigns.** Every request is scoped to one campaign (tenant). Pass `campaign_id`
  (lowercase letters, digits, `-`, `_`; up to 64 characters). Omitted → `DEFAULT_CAMPAIGN_ID` (`default`).
- **Errors.** JSON `{"detail": "..."}` with 401 (API key), 404, 413 (file too large), 422 (invalid
  input, unreadable document), 502 (upstream speech-to-text failure), 503 (feature not configured).
  Internal errors never echo stack traces; streamed errors carry a `request_id` to quote.
- **Examples.** Values in the example payloads below are illustrative; measured numbers live in `eval/results/`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | [`/upload`](#post-upload) | Ingest a document |
| `POST` | [`/retrieve`](#post-retrieve) | Retrieval only (no LLM) |
| `POST` | [`/query`](#post-query) | Grounded answer, JSON or Server-Sent Events |
| `WS` | [`/ws/voice`](#ws-wsvoice) | Real-time voice turns with speculative retrieval |
| `POST` | [`/transcribe`](#post-transcribe) | Audio → text (Whisper) for browsers without speech recognition |
| `GET` | [`/health`](#get-health) | Component diagnostics and active configuration |
| `GET` | [`/metrics`](#get-metrics) | Rolling latency percentiles |
| `GET` | `/documents?campaign_id=` | A campaign's documents with status |
| `GET` | `/documents/{id}/chunks` | Passages of a document, with metadata |
| `DELETE` | `/documents/{id}` | Remove a document and its vectors |
| `DELETE` | `/campaigns/{id}` | Remove all documents of a campaign |

---

## POST /upload

`multipart/form-data`

| Field | Required | Notes |
|---|---|---|
| `file` | yes | `.pdf .docx .md .markdown .txt .png .jpg .jpeg`, up to `MAX_UPLOAD_MB` (20) |
| `campaign_id` | no | tenant |
| `district` | no | e.g. `vijayawada`, `statewide`; aliases accepted (`vizag`, `bezawada`) |
| `category` | no | `manifesto`, `district_profile`, `candidate_profile`, `scheme`, `policy`, `faq`, `other` |
| `topic` | no | `healthcare`, `education`, `employment`, `agriculture`, `infrastructure`, `welfare`, `governance` |
| `source` | no | free-text label shown in citations |

Fields left empty are detected (front matter → `manifest.json` → gazetteer/topic lexicon).
Scanned PDF pages and images are read with OCR.

```bash
curl -s -F file=@sample_data/faq/campaign_faq.md -F district=statewide localhost:8000/upload
```

```json
{
  "status": "indexed",
  "document": {
    "campaign_id": "default", "document_id": "campaign_faq_8c1d2e3f", "filename": "campaign_faq.md",
    "title": "Campaign FAQ", "file_type": "md", "chunk_count": 8, "status": "indexed",
    "metadata": {"district": "statewide", "category": "faq", "topic": "campaign_info", "source": "..."},
    "ocr_pages": [], "content_types": {"text": 8}, "token_count": 1210, "embedding_model": "BAAI/bge-small-en-v1.5"
  },
  "replaced_document_id": null,
  "warnings": [],
  "timings_ms": {"parse": 1.1, "chunk": 6.9, "embed": 203.7, "index": 11.6, "total": 229.7}
}
```

`status` is `indexed`, `duplicate` (same content already in this campaign), or `reindexed`
(same filename and district with new content: the old version is re-indexed). A failed upload
returns 422 with the reason and stays listed under `GET /documents` with `status: "failed"`.

## POST /retrieve

Runs understanding, rewriting, filters, hybrid search, reranking and the relevance gate.
No LLM call. With `session_id` it reads (but does not change) that conversation's memory.

```json
{
  "query": "What healthcare schemes are available?",
  "campaign_id": "default",
  "session_id": "optional",
  "filters": {"district": "vijayawada", "topic": null, "category": null, "content_type": null, "document_ids": null},
  "top_k": 4,
  "threshold": null,
  "include_candidates": true
}
```

Response (abridged):

```json
{
  "request_id": "a1b2c3d4e5f6", "campaign_id": "default",
  "query": "What healthcare schemes are available?",
  "rewritten_query": "What healthcare schemes are available in Vijayawada?",
  "rewrite_reasons": ["district filter: vijayawada (+ statewide)"],
  "filters": {"campaign_id": "default", "district": "vijayawada"},
  "strategy": "hybrid(dense+bm25, rrf) + rerank", "gate": "rerank", "threshold": 0.15,
  "top_score": 0.97, "answerable": true,
  "counts": {"dense": 20, "bm25": 20, "fused": 31, "reranked": 12, "passed": 6, "final": 4},
  "results": [{"rank": 1, "score": 0.97, "scores": {"dense": 0.74, "bm25_rank": 2, "rrf": 0.032, "rerank": 0.97},
               "chunk_id": "...", "document_name": "vijayawada_district_plan.pdf", "page": 1,
               "section": "Healthcare in Vijayawada", "district": "vijayawada", "category": "district_profile",
               "topic": "healthcare", "content_type": "text", "text": "..."}],
  "candidates": ["… the fused pool, for the trace …"],
  "latency_ms": {"rewrite": 0.2, "embed": 7.1, "dense": 3.7, "bm25": 0.3, "fusion": 0.2, "rerank": 231.0,
                 "threshold": 0.0, "context": 0.0, "retrieval_total": 243.0, "total": 244.1}
}
```

## POST /query

Same request fields as `/retrieve` (without `include_candidates`) plus `stream`.
With `session_id`, the turn is added to conversation memory (district, topic, entity, last turns).

**`stream: false`** returns a `QueryResponse`:

```json
{
  "request_id": "...", "query": "...", "answer": "The plan adds a 200-bed mother and child block at Government General Hospital [S1].",
  "answerable": true, "refusal_reason": null,
  "citations": [{"source_id": "S1", "chunk_id": "...", "document_name": "vijayawada_district_plan.pdf", "page": 1,
                 "page_end": null, "section": "Healthcare in Vijayawada", "district": "vijayawada",
                 "category": "district_profile", "topic": "healthcare", "score": 0.97, "cited": true, "snippet": "..."}],
  "retrieval": {"… same trace as /retrieve …": "..."},
  "llm": {"provider": "groq", "model": "openai/gpt-oss-20b", "fallback": false, "error": null},
  "latency_ms": {"retrieval_total": 243.0, "llm": 610.2, "total": 855.3},
  "conversation": {"district": "vijayawada", "topic": "healthcare", "entity": null, "turns": 2}
}
```

`refusal_reason`: `below_threshold` (nothing relevant enough; the LLM is not called),
`no_documents`, or `model_refused` (the model said the sources don't cover it).

**`stream: true`** returns `text/event-stream`:

```
event: retrieval   data: {"type":"retrieval","request_id":"...","retrieval":{…trace…},"latency_ms":{…}}
event: token       data: {"type":"token","text":"The plan adds"}
event: token       data: {"type":"token","text":" a 200-bed …"}
event: done        data: {"type":"done","answer":"…","answerable":true,"citations":[…],"llm":{…},"latency_ms":{…,"first_token":…},"conversation":{…}}
event: error       data: {"type":"error","request_id":"...","message":"Something went wrong… Reference: …"}
```

```bash
curl -N localhost:8000/query -H 'content-type: application/json' \
  -d '{"query":"What is planned for chilli farmers in Guntur?","stream":true}'
```

## WS /ws/voice

One connection per voice client. JSON text frames both ways. The server checks the `Origin`
header against `CORS_ORIGINS`.

### Client → server

| Message | When | Effect |
|---|---|---|
| `{"type":"start","campaign_id":"default","session_id":"s-1","filters":{…}}` | after connecting, and whenever campaign / session / filters change | configures the connection; replies `started` |
| `{"type":"partial","text":"what healthcare schemes"}` | on every interim transcript (any rate) | feeds the partial-transcript controller |
| `{"type":"speech_end"}` | optional, when the recognizer reports end of speech | rerank the latest partial now (S2) |
| `{"type":"final","text":"What healthcare schemes are there?","turn_id":"t1"}` | end of utterance | answers; supersedes an unfinished answer |
| `{"type":"cancel"}` | barge-in | stops the current answer; replies `cancelled` |
| `{"type":"ping","t":123}` | keep-alive | replies `{"type":"pong","t":123}` |

`partial`, `speech_end` and `final` work without `start` (server defaults are used).

### Server → client

| Message | Meaning |
|---|---|
| `ready` | on connect: `llm`, `reranker` (bool), `transcribe` (server STT available), `speculation` settings |
| `speculative` | a speculative search finished: `stage` `S1` (stage-1 candidates) or `S2` (reranked + gated), `query`, `latency_ms`, top `sources`, and for S2 `answerable`, `top_score` |
| `retrieval` | as in SSE, plus `turn_id` and `cache`: `hit` (speculation reused in full), `stage1` (candidates reused, rerank run now), `miss` |
| `token` | answer text delta, with `turn_id` |
| `done` | as in SSE, plus `voice`: `{cache, retrieval_saved_ms, final_to_first_token_ms, partials, ignored_partials, speculative_searches, speculative_refines}` |
| `cancelled` | the answer was stopped |
| `error` | `message` (and `turn_id`, `request_id` for answer failures); the connection stays open |

### Partial-transcript strategy

| Stage | Trigger | Work |
|---|---|---|
| S0 ignore | fewer than `VOICE_MIN_WORDS` (3) words, small talk, a self-introduction, or nothing new | none |
| S1 search | a new district / topic / named entity, or `VOICE_WORD_STEP` (3) more words; debounced `VOICE_DEBOUNCE_MS` (250) | rules-only rewrite (same as the final will get) + stage-1 retrieval (dense ‖ BM25 → RRF) |
| S2 refine | transcript unchanged for `VOICE_STABLE_MS` (500), or `speech_end` | cross-encoder rerank + relevance gate on the cached candidates |
| S3 final | `final` | rewrite with the committed memory; identical normalised query + filters + corpus revision → reuse; otherwise retrieve; then stream the LLM |

The LLM is never called speculatively, and conversation memory changes only on `final`.
A speculation is reused only for an identical key, so a different final wording or an upload in
between simply falls back to a normal retrieval.

```bash
# with websocat (https://github.com/vi/websocat)
websocat ws://localhost:8000/ws/voice
{"type":"partial","text":"what healthcare initiatives"}
{"type":"partial","text":"what healthcare initiatives are proposed for vijayawada"}
{"type":"speech_end"}
{"type":"final","text":"What healthcare initiatives are proposed for Vijayawada?","turn_id":"t1"}
```

## POST /transcribe

`multipart/form-data`: `file` (webm / ogg / wav / mp3 / m4a, up to `MAX_AUDIO_MB` = 10),
`language` (`en-IN` or `en`), `campaign_id` (its document titles are added to Whisper's vocabulary
prompt, with district names, so proper nouns are spelled right).

```json
{"text": "I live in Vijayawada, any new hospitals?", "provider": "groq", "model": "whisper-large-v3-turbo", "latency_ms": 412.5}
```

503 when no provider key is configured (the web app then relies on the browser's speech recognition).

## GET /health

`status`: `ok`, `degraded` (no LLM key, or no documents) or `down` (vector store unreachable).
`components`: `vector_store`, `embedder` (model, dimension, query-cache hits), `llm` (provider, model,
model check, fallback), `reranker`, `bm25`, `documents` (per campaign, failed count), `ocr`, `voice`
(socket path, server transcription, speculation settings), `index_consistency` (chunks from another
embedding model, startup reconciliation). `config` echoes the active retrieval settings.

## GET /metrics

Rolling window (`METRICS_WINDOW`, 500) per endpoint: `query`, `query_stream`, `retrieve`, `voice`.
Each stage has `n`, `p50`, `p90`, `p95`, `max` in milliseconds. The `voice` entry tracks
`final_to_first_token` and `retrieval_saved`.

## Configuration

All settings are environment variables (see `.env.example`). The voice-specific ones:

| Variable | Default | Meaning |
|---|---|---|
| `VOICE_DEBOUNCE_MS` | 250 | S1 debounce |
| `VOICE_STABLE_MS` | 500 | S2: transcript stable this long |
| `VOICE_MIN_WORDS` | 3 | S0 threshold |
| `VOICE_WORD_STEP` | 3 | S1: re-search after this many new words |
| `STT_MODEL` | `whisper-large-v3-turbo` (Groq) / `whisper-1` (OpenAI) | `/transcribe` model |
| `MAX_AUDIO_MB` | 10 | `/transcribe` upload limit |
