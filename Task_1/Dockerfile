FROM python:3.11-slim

WORKDIR /app

# System deps needed by some wheels (torch/onnx via sentence-transformers, pdf parsing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the embedding model at build time so the first run doesn't
# stall on a cold download inside the container. Non-fatal: if the build
# environment's network can't reach the model host, this is skipped and the
# model just downloads lazily on first real run instead.
RUN python -c "\
from chromadb.utils import embedding_functions; \
fn = embedding_functions.DefaultEmbeddingFunction(); \
fn(['warmup'])" || echo "WARN: embedding model pre-warm skipped (will download on first run)"

RUN mkdir -p /app/data /app/logs

ENV PYTHONUNBUFFERED=1
ENV ANONYMIZED_TELEMETRY=False

# Default: run the always-on scheduler service.
CMD ["python", "main.py", "serve"]