FROM python:3.11-slim

LABEL maintainer="SloughGPT"
LABEL description="SloughGPT Model Server - Self-hosted LLM infrastructure"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SLOUGHGPT_ENV="production" \
    SLOUGHGPT_LOG_LEVEL="INFO"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install additional ML dependencies
RUN pip install --no-cache-dir \
    sentence-transformers \
    chromadb \
    torch \
    torchvision

COPY server/ ./server/
COPY sloughgpt_sdk/ ./sloughgpt_sdk/
COPY training/ ./training/
COPY inference/ ./inference/
COPY monitoring/ ./monitoring/
COPY evaluation/ ./evaluation/
COPY experiments/ ./experiments/
COPY domains/ ./domains/
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

RUN mkdir -p /app/data /app/logs /app/cache /app/models /app/datasets /app/vector_store

ENV CUDA_VISIBLE_DEVICES=""

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "server/main.py"]
