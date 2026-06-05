# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Production Image ────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Sentiment Analysis API"
LABEL version="1.0.0"
LABEL description="Production NLP Sentiment Analysis Service"

WORKDIR /app

# Copy installed packages from builder stage (multi-stage = smaller image)
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Create required directories
RUN mkdir -p src/models/saved monitoring/logs logs

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_DIR=src/models/saved
ENV LOG_LEVEL=INFO
ENV PORT=8000

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Start the API server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
