# syntax=docker/dockerfile:1
#
# Corporate Turnaround chatbot — Railway-ready image.
#
# Three things happen at BUILD time that would otherwise cost a real user
# minutes of waiting at RUN time:
#   1. torch is installed from the CPU-only index (the default PyPI wheel pulls
#      ~2.5GB of CUDA libraries this app never uses -- there is no GPU on
#      Railway).
#   2. The embedding + reranker models are downloaded into the image, so the
#      container does not hit the HuggingFace Hub on boot.
#   3. The Chroma index is built. chroma_db/ is gitignored, so without this the
#      container starts fine and then answers every question with "not in
#      knowledge base".

FROM python:3.11-slim

# Model cache lives inside the image (see step 2 below). Setting this before
# any download makes sure build-time and run-time look in the same place.
ENV HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# build-essential is needed by some wheels; removed in the same layer to keep
# the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# ── 1. CPU-only torch, before requirements.txt ────────────────────────────────
# Installing this first means the requirements.txt install finds torch already
# satisfied and does not pull the CUDA build.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 2. Bake the models into the image ─────────────────────────────────────────
# Keep these two model ids in sync with config.py (embedding_model and
# rerank_model). If they drift, the app silently downloads at boot instead --
# slow, and it breaks entirely on a network-restricted host.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# spaCy model for Presidio PII detection. Required whenever PII_ENABLED=true --
# without it AnalyzerEngine() raises on construction and every turn 500s. Baked
# in (not a runtime download) so the container works on a network-restricted
# host and pays the cost once at build. Keep in sync with pii_detector.py.
RUN python -m spacy download en_core_web_lg

COPY . .

# ── 3. Build the Chroma index ─────────────────────────────────────────────────
# Uses local HuggingFace embeddings, so no API key is required at build time.
RUN python scripts/ingest.py --loader enriched --force

# Railway injects $PORT and expects the app to bind it on 0.0.0.0. Shell form
# (not exec form) so the variable actually expands; the fallback keeps
# `docker run -p 8000:8000` working locally.
#
# The API is the production entrypoint (the website widget talks to it).
# One worker on purpose: the embedding model, BM25 index, cross-encoder and
# conversation memory are all in-process, so a second worker doubles ~1GB of
# RAM and splits session state across processes. Scale with more containers
# plus sticky sessions, or externalize memory first.
#
# For the internal Streamlit UI instead:
#   docker run -e APP_MODE=streamlit ...
CMD if [ "$APP_MODE" = "streamlit" ]; then \
      streamlit run app.py \
        --server.port=${PORT:-8501} \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false; \
    else \
      uvicorn api.main:app \
        --host 0.0.0.0 \
        --port ${PORT:-8000} \
        --workers 1 \
        --timeout-keep-alive 65; \
    fi
