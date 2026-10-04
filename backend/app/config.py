"""Application settings, loaded from environment variables / `.env`.

Every tunable that affects retrieval quality or latency lives here so that
experiments (eval scripts) and deployments change behaviour without code edits.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.ingestion.cleaning import DEFAULT_BOILERPLATE

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Storage ──────────────────────────────────────────────────────────
    data_dir: Path = Field(default=REPO_ROOT / "data")
    # Empty → embedded (on-disk) Qdrant under data_dir; set to http://qdrant:6333 in Docker.
    qdrant_url: str = ""
    qdrant_collection: str = "campaign_chunks"

    # ── Embeddings ───────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Optional local model directory (see scripts/download_models.py). Empty → fastembed downloads.
    embedding_model_path: str = ""
    model_cache_dir: Path = Field(default=REPO_ROOT / "models")
    query_cache_size: int = 2048
    # "auto" → model's recommended query instruction (bge: "Represent this sentence…"); "" → none.
    embedding_query_prefix: str = "auto"

    # ── Ingestion / chunking ─────────────────────────────────────────────
    chunk_strategy: Literal["section", "fixed"] = "section"
    # Section chunker budget in embedding-model tokens (counted with the model's tokenizer).
    # 256 + a ~40-token contextual header stays well under bge's 512 limit, and query + chunk
    # still fits the cross-encoder's 512-token window, so neither model truncates a chunk.
    chunk_max_tokens: int = 256
    chunk_min_tokens: int = 32     # smaller sections are merged with their neighbour
    chunk_max_words: int = 180     # fixed-window baseline only
    chunk_overlap_words: int = 30  # fixed strategy only; section strategy overlaps by one sentence
    max_upload_mb: int = 20
    # OCR fallback for scanned PDF pages and image uploads (RapidOCR, runs at upload time only).
    ocr_enabled: bool = True
    ocr_max_pages: int = 50
    # Tenant used when a request does not name a campaign (single-campaign deployments, the demo).
    default_campaign_id: str = "default"
    # Regexes for corpus boilerplate (disclaimers, legal notices) removed before chunking. JSON list in env.
    boilerplate_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_BOILERPLATE))
    seed_sample_data: bool = False
    sample_data_dir: Path = Field(default=REPO_ROOT / "sample_data")

    # ── Retrieval ────────────────────────────────────────────────────────
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    top_k: int = Field(default=4, ge=1, le=20)             # chunks handed to the LLM
    candidate_k: int = Field(default=20, ge=1, le=100)      # stage-1 recall pool per retriever
    rrf_k: int = 60                                         # RRF constant (standard value)
    max_chunks_per_doc: int = 2                             # multi-document diversity in the context
    # Gate without reranker: best dense cosine. Tuned on eval/queries.jsonl (see eval/results).
    similarity_threshold: float = Field(default=0.62, ge=0.0, le=1.0)

    # ── Reranking ────────────────────────────────────────────────────────
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"  # empty → disabled
    rerank_k: int = Field(default=12, ge=1, le=50)          # candidates sent to the cross-encoder
    # Gate with reranker: per-chunk cross-encoder probability. Provisional until measured.
    rerank_threshold: float = Field(default=0.15, ge=0.0, le=1.0)

    # ── Conversation ─────────────────────────────────────────────────────
    query_rewriter: Literal["rules", "llm"] = "rules"       # llm = rules + LLM refinement with timeout
    rewrite_timeout_s: float = 1.2
    session_ttl_s: int = 1800

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_provider: Literal["groq", "openai", "ollama", "extractive"] = "groq"
    llm_model: str = "llama-3.1-8b-instant"
    llm_base_url: str = ""   # derived from provider when empty
    groq_api_key: str = ""
    openai_api_key: str = ""
    llm_timeout_s: float = 20.0
    llm_max_tokens: int = 300
    llm_temperature: float = 0.1

    # ── Voice ────────────────────────────────────────────────────────────
    # Partial-transcript controller (WS /ws/voice): see app/voice/controller.py.
    voice_debounce_ms: int = 250   # S1: wait this long after a triggering partial before searching
    voice_stable_ms: int = 500     # S2: transcript unchanged this long → rerank ahead of the final
    voice_min_words: int = 3       # S0: shorter partials are ignored
    voice_word_step: int = 3       # S1: re-search after this many new words (or a new district/topic/entity)
    # End-of-turn detection (app/voice/endpointing.py): how long a silence ends the turn, by how
    # finished the transcript sounds. "unsure" is the old fixed timeout.
    voice_adaptive_endpointing: bool = True
    voice_endpoint_complete_ms: int = Field(default=300, ge=100, le=3000)
    voice_endpoint_likely_ms: int = Field(default=550, ge=100, le=3000)
    voice_endpoint_unsure_ms: int = Field(default=900, ge=100, le=5000)
    voice_endpoint_incomplete_ms: int = Field(default=1600, ge=100, le=8000)
    # Server-side speech-to-text (/transcribe) for browsers without the Web Speech API.
    stt_model: str = ""            # default: whisper-large-v3-turbo (Groq) / whisper-1 (OpenAI)
    max_audio_mb: int = 10

    # ── API / observability ──────────────────────────────────────────────
    # Optional shared secret. When set, write endpoints (/upload, DELETE /documents/*)
    # require an X-API-Key header; read-only endpoints stay open.
    api_key: str = ""
    cors_origins: str = "*"
    log_level: str = "INFO"
    log_query_text: bool = True   # set false to keep user utterances out of logs
    metrics_window: int = 500

    def cors_allow_origins(self) -> list[str]:
        """CORS origins to apply. A wildcard is honoured only when no API key is set:
        with a key, any website could otherwise drive a logged-in operator's browser."""
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        if "*" in origins and self.api_key:
            return [o for o in origins if o != "*"]
        return origins

    @property
    def resolved_embedding_model_path(self) -> str:
        """Explicit path, else the directory scripts/download_models.py creates, else '' (fastembed downloads)."""
        if self.embedding_model_path:
            return self.embedding_model_path
        local = self.model_cache_dir / "bge-small-en-v1.5"
        if self.embedding_model == "BAAI/bge-small-en-v1.5" and (local / "tokenizer.json").exists():
            return str(local)
        return ""

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.json"

    @property
    def resolved_llm_base_url(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url.rstrip("/")
        return {
            "groq": "https://api.groq.com/openai/v1",
            "openai": "https://api.openai.com/v1",
            "ollama": "http://localhost:11434/v1",
        }.get(self.llm_provider, "")

    @property
    def llm_api_key(self) -> str:
        return {"groq": self.groq_api_key, "openai": self.openai_api_key}.get(self.llm_provider, "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
