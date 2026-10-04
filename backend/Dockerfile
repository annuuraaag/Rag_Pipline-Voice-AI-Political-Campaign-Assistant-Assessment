# Backend image. Build context = repository root (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY backend/requirements.txt backend/requirements-ocr.txt ./
RUN pip install -r requirements.txt && pip install --no-deps -r requirements-ocr.txt

# Bake the models into the image so containers start offline and fast: the embedder
# (required) and the cross-encoder reranker (optional: if it can't be downloaded the
# build continues and the app runs without reranking, reported in /health).
COPY scripts/download_models.py ./scripts/download_models.py
RUN python scripts/download_models.py --dest /app/models

COPY backend/app ./app
COPY sample_data ./sample_data
# Retrieval evaluation, runnable in the container: docker compose exec backend python eval/run_retrieval_eval.py --rerank
COPY eval ./eval

ENV DATA_DIR=/data \
    MODEL_CACHE_DIR=/app/models \
    EMBEDDING_MODEL_PATH=/app/models/bge-small-en-v1.5 \
    SAMPLE_DATA_DIR=/app/sample_data \
    HF_HUB_OFFLINE=1

RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app /data && chown -R app /app/models /app/eval
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
