# SloughGPT Production Dockerfile
# Multi-stage build for optimized production deployment

# Build Stage 1: Python Dependencies
FROM python:3.10-slim as dependencies

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Build Stage 2: Application Build
FROM python:3.10-slim as builder

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Copy virtual environment from dependencies stage
COPY --from=dependencies /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY . .

# Install application
RUN pip install -e .

# Download models (if needed)
RUN python -c "
import os
from sloughgpt.core.model_manager import ModelManager

# Pre-download essential models
manager = ModelManager()
try:
    manager.download_model('default')
    print('Default model downloaded successfully')
except Exception as e:
    print(f'Model download failed: {e}')
"

# Build Stage 3: Production Runtime
FROM python:3.10-slim as production

# Set labels
LABEL maintainer="SloughGPT Team" \
      version="1.0.0" \
      description="SloughGPT AI System" \
      org.opencontainers.image.title="SloughGPT" \
      org.opencontainers.image.description="Advanced AI conversation system" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="SloughGPT"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    SLAUGHGPT_ENV="production" \
    SLAUGHGPT_LOG_LEVEL="INFO" \
    SLAUGHGPT_HOST="0.0.0.0" \
    SLAUGHGPT_PORT="8000" \
    SLUGHGPT_MODEL_PATH="/app/models" \
    SLAUGHGPT_DATA_PATH="/app/data" \
    SLAUGHGPT_CACHE_PATH="/app/cache"

# Create non-root user for security
RUN groupadd -r sloughgpt && \
    useradd -r -g sloughgpt -d /app -s /bin/bash sloughgpt

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment and application
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/cache /app/models && \
    chown -R sloughgpt:sloughgpt /app

# Set working directory
WORKDIR /app

# Switch to non-root user
USER sloughgpt

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/opt/venv/bin/python", "-m", "sloughgpt.api_server"]

# Default command
CMD ["--host", "0.0.0.0", "--port", "8000"]