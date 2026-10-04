# Rag_Pipline-Voice-AI-Political-Campaign-Assistant-Assessment
# Real-Time RAG Voice Assistant for a Political Campaign

A voice assistant that answers citizens' questions **only from a campaign's own documents**:
manifestos, district plans, scheme guides, FAQs. Every answer cites the file, page and section it
came from. When the documents don't cover a question, the assistant says so instead of guessing.

It is built for real-time voice. Retrieval starts **while the user is still speaking**, the
assistant **hears when a question is finished** instead of waiting out a fixed pause, the answer is
**spoken sentence by sentence as it is generated**, and the user can **interrupt** at any moment.
Speech can be recognised and spoken in the browser or **on the server** (private, on CPU, or with a
cloud provider), and every sentence of an answer is **checked against the passage it cites**.

> The sample corpus is **fictional** (the "Sunrise Coast Alliance" and its candidate are invented).
> The assistant is informational, never persuasive.

---

## Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [How a question is answered](#how-a-question-is-answered)
- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [API](#api)
- [Evaluation and results](#evaluation-and-results)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Security](#security)
- [Limitations and next steps](#limitations-and-next-steps)

---

## Highlights

| Area | What it does |
|---|---|
| **Ingestion** | PDF, DOCX, Markdown, TXT, plus scanned PDFs and PNG/JPG images via OCR. Structure-aware chunking with metadata: district, category, topic, source, page, section. |
| **Hybrid retrieval** | Dense vectors (bge-small, ONNX) and BM25 keyword search in parallel, merged with Reciprocal Rank Fusion. |
| **Reranking + relevance gate** | A cross-encoder rescores the top candidates. Its score decides whether to answer at all, so off-topic questions are refused before the LLM is ever called. |
| **Conversation memory** | Remembers district, topic and the last named scheme. "What about education?" becomes "What are the education plans and schemes in Vijayawada?" |
| **Multi-document answers** | Context mixes up to 4 passages (at most 2 per document), so the manifesto, a district plan and a scheme guide can all be cited together. |
| **Citations** | The model writes `[S1]`…`[S4]`; file, page and section are attached from metadata, so citations cannot be invented. |
| **Verified claims** | Each sentence is checked against the passage it cites: figures, districts and names must be there, and most of its words. Wrong citations are moved to the passage that supports the sentence; unsupported sentences are underlined (or removed and not spoken, in strict mode). |
| **Real-time voice** | Streaming speech recognition, speculative retrieval on partial transcripts, sentence-level speech output, barge-in. |
| **End of turn** | The transcript decides how long a silence ends the question: 300 ms after "…for farmers in Guntur", 1.6 s after "…for farmers in". |
| **Server-side speech** | Optional: the microphone streams to the server, which recognises (local sherpa-onnx, Deepgram or Whisper), corrects misheard place and scheme names, detects barge-in, and speaks answers with its own voice (local Piper/Kokoro, Deepgram, OpenAI-compatible, ElevenLabs). |
| **Multi-campaign** | Every document, search and conversation is isolated per campaign (tenant). |
| **Measured** | Retrieval, answers, claim checks, end of turn, latency and the voice pipeline on real audio each have an evaluation script and stored results. |

---

## Architecture

![Architecture](docs/architecture.png)

The system is one FastAPI service plus Qdrant, with a React web app in front. All models run on CPU
through ONNX Runtime; there is no GPU or PyTorch dependency.

```
┌──────────────── Browser (React) ────────────────┐
│ Mic → Web Speech API (partials) or 16 kHz PCM   │
│ Speech output ← sentences ← tokens, or PCM audio│
└───────────────┬─────────────────▲───────────────┘
                │ WebSocket /ws/voice: JSON + audio (also REST + SSE)
┌───────────────▼─────────────────┴───────────────┐
│ FastAPI                                         │
│  Server speech (optional): VAD → streaming STT  │
│   → name correction; TTS per sentence; barge-in │
│  End-of-turn verdicts · speculative retrieval   │
│  Memory + query rewrite → filters               │
│  Dense ‖ BM25 → RRF → cross-encoder → gate      │
│  Multi-document context → LLM (Groq) → citations│
│   → claim check (each sentence vs its source)   │
└───────┬───────────────────────────────┬─────────┘
        │                               │
   Qdrant (vectors)            BM25 index + document registry
        ▲
        └── Ingestion: parse (+OCR) → clean → chunk → metadata → embed
```

| Layer | Technology |
|---|---|
| API | FastAPI, Pydantic, WebSocket, Server-Sent Events |
| Vector store | Qdrant (server in Docker, embedded locally) |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (ONNX), 384 dimensions |
| Keyword search | BM25 (rank-bm25) |
| Reranker | `ms-marco-MiniLM-L-6-v2` cross-encoder (ONNX) |
| LLM | Groq (OpenAI-compatible API, streamed); offline extractive fallback |
| Speech | Browser Web Speech API, or on the server: sherpa-onnx on CPU (Kroko streaming STT, Piper/Kokoro voices, Silero VAD), Deepgram, Whisper (OpenAI-compatible), ElevenLabs |
| OCR | RapidOCR (ONNX) + pypdfium2 |
| Frontend | React + Vite, served by nginx |

More detail:
- [docs/DESIGN_NOTE.md](docs/DESIGN_NOTE.md): the reasoning behind each design decision.
- [docs/API.md](docs/API.md): full API reference.

---
## Quick start

Choose one path: **Option A (Docker)** is recommended and needs the fewest steps. **Option B**
runs everything directly with Python and Node.js.

### Option A: Run with Docker (recommended)

**Step 1: Install the tools** (one time)
- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/). After installing, open it and
  wait until it says *Engine running*. On Windows, accept the WSL 2 setup if prompted.
- Chrome, Edge or Safari, for live voice input.

**Step 2: Get a Groq API key** (free, optional)
1. Sign in at https://console.groq.com/keys.
2. Click **Create API Key** and copy it.

Without a key the app still works; answers are short quotes from the documents instead of
LLM-written answers.

**Step 3: Download the code**
```bash
git clone https://github.com/annuuraaag/Rag_Pipline-Voice-AI-Political-Campaign-Assistant-Assessment.git
cd Rag_Pipline-Voice-AI-Political-Campaign-Assistant-Assessment
```

**Step 4: Create your settings file**
```bash
cp .env.example .env            # macOS / Linux / Git Bash
copy .env.example .env          # Windows Command Prompt / PowerShell
```
Open `.env` in any text editor and set your key:
```
GROQ_API_KEY=gsk_your_key_here
```
`.env` is git-ignored, so your key is never committed.

**Step 5: Build and start**
```bash
docker compose up --build
```
The first run takes a few minutes: it downloads Python and Node images and the AI models,
including the local speech models (~140 MB; build with `SPEECH_MODELS=false docker compose build`
to skip them and leave speech to the browser). It is ready when the log shows
`Application startup complete`. Later starts take seconds.

**Step 6: Open the app**
- **Web app:** http://localhost:5173
  1. Allow microphone access when asked.
  2. Tap the orb and ask: *"What healthcare initiatives does the candidate propose for Vijayawada?"*
- **API docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health. `"status": "ok"` means everything is connected.

The 10 sample documents are indexed automatically on first start.

**Step 7: Stop or reset**
```bash
docker compose down             # stop (keeps your uploaded documents)
docker compose down -v          # stop and delete all indexed data (fresh start)
docker compose up -d --build    # start again in the background after pulling new code
```

### Option B: Run without Docker

**Requirements:** Python 3.11+, Node.js 20+, Git.

**Step 1: Clone the repository** and enter the folder (as in Option A, Step 3).

**Step 2: Set up the backend**
```bash
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
pip install -r backend/requirements-dev.txt
pip install --no-deps -r backend/requirements-ocr.txt
python scripts/download_models.py            # downloads embedding + reranker models into models/
cp .env.example .env                         # then add GROQ_API_KEY (Windows: copy)
```

Optional, server-side speech on CPU (otherwise the browser recognises and speaks):
```bash
pip install -r backend/requirements-speech.txt
python scripts/download_models.py --only-speech   # Kroko STT, Piper voice, Silero VAD → models/
# in .env: STT_PROVIDER=local and TTS_PROVIDER=local
```

**Step 3: Start the API** (embedded vector database, no separate server needed)
```bash
cd backend
SEED_SAMPLE_DATA=true uvicorn app.main:app --port 8000
```
On Windows PowerShell, set the variable first: `$env:SEED_SAMPLE_DATA="true"; uvicorn app.main:app --port 8000`

**Step 4: Start the web app** in a second terminal
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173. The dev server forwards `/api` calls to the backend on port 8000.

### Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` or `Cannot connect to the Docker daemon` | Start Docker Desktop and wait for *Engine running* |
| Port 5173, 8000 or 6333 already in use | Stop the other program, or change the left-hand port in `docker-compose.yml` |
| Microphone button does nothing | Allow the microphone in the browser (lock icon in the address bar), and use Chrome or Edge |
| Sidebar says *Offline answers* | `GROQ_API_KEY` is missing in `.env`. Add it and run `docker compose up -d` |
| Old version still showing after pulling new code | `docker compose up -d --build`, then hard-refresh the browser (Ctrl+Shift+R) |
| Answers ignore newly uploaded documents | Make sure the same campaign is selected in the sidebar for upload and questions |

## How a question is answered

Example: the user says *"I'm from Vijayawada"*, then *"what about new hospitals?"*

1. **Listen.** The browser streams partial transcripts ("what about", "what about new hospitals")
   over a WebSocket, or, with server recognition, the microphone audio itself.
2. **Understand.** Conversation memory already holds *district = Vijayawada*, so the question
   becomes *"…new hospitals in Vijayawada"*. Search is filtered to Vijayawada plus statewide documents.
3. **Search ahead.** The voice controller works through four stages:

   | Stage | When | What it does |
   |---|---|---|
   | S0 | always | Ignores fragments and small talk |
   | S1 | the transcript gains a place, topic or name | Runs the fast search (~10–30 ms) |
   | S2 | the user pauses | Runs the precise rerank (~230 ms on CPU), so it is already done when they finish |
   | S3 | the final transcript | Reuses the speculative result if the question is identical |

4. **Hear the end.** Each partial gets an end-of-turn verdict: "…new hospitals" sounds finished
   (answer after 300 ms of silence), "…new hospitals in" does not (wait 1.6 s).
5. **Decide.** If no passage passes the relevance gate, the assistant replies "not in the
   documents" and the LLM is not called.
6. **Answer.** The LLM receives 4 numbered passages and must cite them. Tokens stream back.
7. **Speak.** Each completed sentence (the first one may stop at a comma) is spoken immediately,
   by the browser or the server voice; citations show on screen.
8. **Check.** Every sentence is checked against the passage it cites; a misattributed citation is
   corrected, an unsupported sentence is underlined.
9. **Interrupt.** Tapping the mic, pressing Space, or simply talking (in conversation mode) stops
   the speech and the generation, and starts listening again.

---

## Using the app

**Assistant.** Ask a question by voice or text:
- **Talk:** tap the orb or the mic button, or hold <kbd>Space</kbd> and release it to send at once.
  <kbd>Esc</kbd> stops.
- **Citations:** click a citation number to open its source passage.
- **Details panel:** shows the sources, the retrieval trace (rewritten query, filters, candidate
  funnel, relevance gate) and the latency of each stage.

**Knowledge base.** Drag and drop documents to index them:
- Metadata is detected automatically and can be overridden.
- Click a document to see its passages.
- Scanned PDFs and photos are read with OCR.

**Insights.** Component health and live P50/P90/P95 latency.

**Settings:**
- Recognition language (English India / US / UK), voice, speaking rate.
- Speech recognition and answer voice: in the browser or on the server (when the server offers them).
- Smart end-of-question detection, and the pause used when it can't tell.
- Conversation mode: hands-free; interrupt by speaking.
- Theme, and an API key if the server requires one.

**Try these questions:**
- *"I'm from Vijayawada."*, then *"What healthcare schemes are available?"*, then *"What about education?"*
- *"Who is eligible for the Pratibha Scholarship?"*
- *"What is planned for chilli farmers in Guntur?"*
- *"What is the campaign's policy on lunar mining?"* (correctly refused)

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/upload` | Upload and index a document (multipart; optional `campaign_id`, `district`, `category`, `topic`, `source`) |
| `POST` | `/retrieve` | Retrieval only: scored passages, metadata, filters, relevance decision, per-stage latency |
| `POST` | `/query` | Grounded answer with citations; JSON, or streamed with `"stream": true` (SSE) |
| `WS` | `/ws/voice` | Real-time voice: transcripts or microphone audio in; end-of-turn hints, answers and spoken audio out |
| `POST` | `/transcribe` | Audio → text (Whisper) for browsers without speech recognition |
| `GET` | `/health` | Component diagnostics and active configuration |
| `GET` | `/metrics` | Rolling P50/P90/P95 latency per endpoint and stage |
| `GET` | `/documents` | Documents of a campaign, with indexing status |
| `GET` | `/documents/{id}/chunks` | Indexed passages of a document |
| `DELETE` | `/documents/{id}` · `/campaigns/{id}` | Remove a document or a whole campaign |

```bash
# Ask a question
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"query": "Who is eligible for the Pratibha Scholarship?"}'

# Stream the answer token by token
curl -N localhost:8000/query -H 'content-type: application/json' \
  -d '{"query": "What is planned for chilli farmers in Guntur?", "stream": true}'

# Retrieval only, filtered to a district
curl -s localhost:8000/retrieve -H 'content-type: application/json' \
  -d '{"query": "What healthcare schemes are available?", "filters": {"district": "vijayawada"}}'

# Upload a document
curl -s -F file=@sample_data/faq/campaign_faq.md -F district=statewide localhost:8000/upload
```

See [docs/API.md](docs/API.md) for request/response schemas, SSE events and the WebSocket protocol.

---

## Evaluation and results

Retrieval, answers, claim checks, end of turn and latency are measured separately, so a wrong or
slow answer can be traced to the stage that caused it.

- **Test set.** 32 labelled questions in `eval/queries.jsonl`. They cover direct facts, district
  questions, multi-document questions, follow-ups, disfluent voice-style phrasing and exact scheme
  names, plus 7 questions that must be refused.
- **Labels** are evidence phrases rather than chunk ids, so different chunking strategies are
  scored fairly.

```bash
python eval/run_retrieval_eval.py [--rerank]         # → eval/results/retrieval_eval.md
python eval/run_answer_eval.py [--rerank] [--judge]   # → answer_eval.md (claim check + LLM judge)
python eval/run_latency_benchmark.py [--rerank]      # → latency_benchmark.md (to first token and first audio)
python eval/run_endpointing_eval.py                  # → endpointing_eval.md (no models needed)
python eval/run_verifier_eval.py                     # → verifier_eval.md (no models needed)
python eval/run_speech_eval.py [--barge-in]          # → speech_eval.md (real audio; needs --speech models)
# inside Docker:  docker compose exec backend python eval/run_answer_eval.py --rerank --judge
```

The answer evaluation with a real LLM needs `GROQ_API_KEY`: with it, `--judge` has the LLM count
supported and unsupported claims, and the report shows how often the claim check agrees.

| Design choice | Measured effect |
|---|---|
| Structure-aware chunks vs fixed 180-word chunks | recall within the first 300 retrieved words **0.85 vs 0.71** |
| Hybrid (dense + BM25) vs dense only | Recall@5 **0.81 vs 0.78** |
| Conversation-aware query rewriting | Recall@5 0.81 → **0.95**; follow-up questions **0.33 → 0.92** |
| Cross-encoder rerank as the relevance gate | off-topic questions refused **7/7** (vs 6/7); MRR 0.91 → **0.95** |
| Speculative retrieval on partial transcripts | reused for **31 of 32** simulated spoken questions, taking retrieval off the critical path |
| Adaptive end of turn vs a fixed 900 ms pause | finished questions wait **484 ms** on average (transcripts); on real audio, last word → first answer audio p50 **749 ms vs 1,127 ms** |
| Claim check (each sentence vs its cited passage) | **99.6%** of 253 corrupted claims caught (changed figure, swapped district, absent from context); **0%** of 443 correct claims flagged (synthetic, from the corpus) |
| Server speech on CPU (Kroko STT + Piper voice) | word error rate **18.6%** on synthetic speech; barge-in caught **5/5**, ~0.93 s after the first word |
| Name correction for misheard places and schemes | districts recognised **0% → 31%** (offline decode of the same audio) |

The corpus is small (10 documents, 32 questions), so differences of a few points are indicative
rather than statistically significant. Full tables are in [`eval/results/`](eval/results/).

`speech_eval.md` was produced in a sandbox where neither bge-small nor an LLM key was available:
retrieval used a labelled test embedder (retrieval runs off the critical path thanks to speculation)
and answers came from the instant extractive fallback, so add your LLM's time-to-first-token to its
latencies. The voice that asks is the same synthetic voice the assistant uses, so real speakers
will see higher error rates. Re-run it, `run_latency_benchmark.py` and `run_answer_eval.py --judge`
with the real models and a key for complete numbers.

---

## Configuration

All settings are environment variables; see [`.env.example`](.env.example). The most important:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | – | LLM, and Whisper speech-to-text (`/transcribe`, `STT_PROVIDER=whisper`) |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Replaced automatically if Groq retires it |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid`, `dense` or `bm25` |
| `TOP_K` / `CANDIDATE_K` | 4 / 20 | Passages sent to the LLM / candidates per retriever |
| `RERANK_THRESHOLD` | 0.15 | Relevance gate when the reranker is loaded |
| `SIMILARITY_THRESHOLD` | 0.62 | Relevance gate without the reranker |
| `CHUNK_MAX_TOKENS` | 256 | Chunk size, counted with the embedding model's tokenizer |
| `VOICE_DEBOUNCE_MS` / `VOICE_STABLE_MS` | 250 / 500 | Speculative search timing |
| `VOICE_ADAPTIVE_ENDPOINTING` | `true` | End-of-turn verdicts; waits per verdict: `VOICE_ENDPOINT_{COMPLETE,LIKELY,UNSURE,INCOMPLETE}_MS` = 300 / 550 / 900 / 1600 |
| `STT_PROVIDER` / `TTS_PROVIDER` | `none` (`local` in Docker) | Server-side speech: `local`, `deepgram`, `whisper` / `local`, `deepgram`, `openai`, `elevenlabs` |
| `DEEPGRAM_API_KEY` · `ELEVENLABS_API_KEY` · `STT_BASE_URL` · `TTS_BASE_URL` | – | Cloud speech keys, or self-hosted OpenAI-compatible speech servers |
| `CITATION_VERIFICATION` | `flag` | `off`, `flag` (mark unsupported sentences), `strict` (also remove them, unspoken) |
| `OCR_ENABLED` | `true` | OCR for scanned pages and images |
| `API_KEY` | – | When set, uploads and deletions require `X-API-Key` |
| `CORS_ORIGINS` | `*` | Allowed browser origins (also checked for the WebSocket) |

---

## Project structure

```
backend/app/
  api/routes/      HTTP + WebSocket endpoints
  ingestion/       parsers, OCR, cleaning, chunker, metadata, document registry
  retrieval/       embedder, Qdrant store, BM25, fusion, reranker, context builder, pipeline
  conversation/    query understanding, memory, rewriter
  generation/      prompt, citations, claim check (verify.py), LLM adapters (Groq / extractive fallback)
  voice/           partial-transcript controller, end-of-turn rules, server speech: VAD, STT and
                   TTS providers, name correction, speech session (audio in, spoken answers out)
  rag/             orchestration of the full request
  observability/   stage timing, latency percentiles, structured logs
backend/tests/     unit and API tests
frontend/src/      React app (lib/voice: recognition, endpointing, speech output, PCM capture and
                   playback, socket, barge-in)
frontend/e2e/      browser tests of the voice flow (browser speech; server speech with real audio)
eval/              labelled questions, evaluation scripts, results
sample_data/       fictional campaign documents (PDF, DOCX, MD, TXT, scanned PDF, image)
scripts/           model download (--speech for the local speech models), sample document builder,
                   spoken-question WAV builder for the server-speech browser test
docs/              design note, API reference, architecture diagram
```

---

## Testing

```bash
cd backend && pytest                  # 185 tests; no network or model download needed
cd frontend && npm run test:e2e       # full voice flow in a real browser (needs the stack running)
ruff check backend eval scripts       # lint
```

The browser test replaces only the speech engine (with a scripted recognizer and synthesizer).
The app, WebSocket, speculative retrieval, end-of-turn hints, streaming, barge-in and source panel
all run for real. `frontend/e2e/server-speech.e2e.mjs` tests server speech with real audio: a
spoken question (built with `scripts/make_speech_wav.py`) is Chromium's microphone; the server
recognises it, answers in its own voice, and a second question spoken over the answer must
interrupt it. Speech providers are tested against fakes (`tests/test_speech*.py`); a test with the
real local models runs when they are downloaded.

---

## Security

- **Secrets:** no secrets in the repository. Keys are read from `.env`, which is git-ignored.
- **Network exposure:** Docker publishes ports on `127.0.0.1` only, so the vector database is not
  exposed to the network.
- **Write protection:** set `API_KEY` to protect uploads and deletions. Set `CORS_ORIGINS` before
  exposing the app beyond localhost.
- **Tenant isolation:** campaign isolation is enforced inside the retriever; a search without a
  campaign is rejected.
- **Prompt injection:** document text is treated as data in the prompt.
- **Errors:** internal errors return only a reference id.
- **Speech audio:** processed in memory and never stored; with the local providers it never leaves
  the server. Audio frames over 64 KB are dropped.
- **Web server:** nginx sends a Content-Security-Policy and other security headers. The backend
  container runs as a non-root user.

---

## Limitations and next steps

- **Local recognition accuracy:** the on-CPU recogniser is private but small; on clean synthetic
  speech it gets about one word in five wrong and still misses many place names after correction.
  Deepgram (with key terms) or Whisper are the accurate options. In-browser recognition remains
  the default where available.
- **End-of-turn rules:** rules on the transcript, tuned on 32 questions. An audio-aware turn
  detector (prosody) and real usage logs would refine them.
- **Real-LLM numbers:** answer faithfulness, the claim check's agreement with an LLM judge, and
  time-to-first-token were not measured here (no API key in the build environment); the scripts
  report them as soon as `GROQ_API_KEY` is set.
- **Speculative generation:** generating the answer speculatively would also hide the LLM's
  time-to-first-token, at extra token cost.
- **Languages:** Telugu and code-mixed speech would need multilingual embeddings.
- **Scale:** Qdrant sparse vectors instead of the in-memory BM25 index, and a larger evaluation set
  built from real usage.
