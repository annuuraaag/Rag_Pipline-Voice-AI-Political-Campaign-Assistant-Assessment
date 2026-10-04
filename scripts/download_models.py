"""Materialise the embedding model into a fixed local directory.

    python scripts/download_models.py [--dest models]

Tries fastembed's normal download (Hugging Face) first; if that host is
unreachable, falls back to Qdrant's public GCS mirror of the same ONNX export.
Either way the model ends up in <dest>/bge-small-en-v1.5, which is what
EMBEDDING_MODEL_PATH points to (Docker sets this automatically), so runtime
never depends on network access.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

# name → (fastembed model id, local dir, GCS mirror of the same ONNX export)
MODELS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", "bge-small-en-v1.5",
                  "https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en-v1.5.tar.gz"),
    # Baseline embedder, only needed for the embedding comparison in eval/.
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2",
               "https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz"),
}
RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"


def _valid(d: Path) -> bool:
    return d.is_dir() and any(d.glob("*.onnx")) and (d / "tokenizer.json").exists()


def _fix_tokenizer_config(d: Path) -> None:
    # The GCS export stores model_max_length as a huge sentinel; fastembed needs the real limit (512).
    p = d / "tokenizer_config.json"
    if p.exists():
        cfg = json.loads(p.read_text())
        if not isinstance(cfg.get("model_max_length"), int) or cfg["model_max_length"] > 8192:
            cfg["model_max_length"] = 512
            p.write_text(json.dumps(cfg, indent=2))


def from_huggingface(model_id: str, target: Path, cache: Path) -> bool:
    try:
        from fastembed import TextEmbedding

        model = TextEmbedding(model_id, cache_dir=str(cache))
        src = Path(model.model._model_dir)  # resolved snapshot directory
        shutil.copytree(src, target, dirs_exist_ok=True)
        return _valid(target)
    except Exception as exc:  # network policy, HF outage, API change
        print(f"[download_models] Hugging Face download failed ({exc.__class__.__name__}); trying GCS mirror.")
        return False


def from_gcs(url: str, target: Path) -> bool:
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        target.mkdir(parents=True, exist_ok=True)
        for m in members:
            f = tar.extractfile(m)
            if f:
                (target / Path(m.name).name).write_bytes(f.read())
    _fix_tokenizer_config(target)
    return _valid(target)


def fetch_embedder(name: str, dest: Path) -> bool:
    model_id, local, url = MODELS[name]
    target = dest / local
    if _valid(target):
        print(f"[download_models] already present: {target}")
        return True
    ok = from_huggingface(model_id, target, dest / ".cache") or from_gcs(url, target)
    shutil.rmtree(dest / ".cache", ignore_errors=True)
    print(f"[download_models] {'ready' if ok else 'FAILED'}: {target}")
    return ok


def fetch_reranker(dest: Path) -> bool:
    """Cross-encoder into the fastembed cache the app reads (<dest>/hf). Hugging Face only."""
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        TextCrossEncoder(RERANKER, cache_dir=str(dest / "hf"))
        print(f"[download_models] reranker ready: {RERANKER}")
        return True
    except Exception as exc:
        print(f"[download_models] reranker unavailable ({exc.__class__.__name__}); "
              "the app will run without reranking.")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "models"))
    ap.add_argument("--with-baseline", action="store_true", help="also fetch all-MiniLM-L6-v2 (eval only)")
    ap.add_argument("--no-reranker", action="store_true")
    args = ap.parse_args()
    dest = Path(args.dest)
    if not fetch_embedder("bge-small", dest):
        print("[download_models] FAILED to obtain the embedding model", file=sys.stderr)
        return 1
    if args.with_baseline:
        fetch_embedder("minilm", dest)
    if not args.no_reranker:
        fetch_reranker(dest)  # optional: never fails the build
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
