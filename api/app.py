"""
FastAPI REST API for Sentiment Analysis Model Serving
Endpoints: /predict, /batch_predict, /health, /metrics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      example="Great product! Highly recommend!")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100,
                              example=["Great product!", "Terrible experience."])

class SentimentResult(BaseModel):
    text: str
    sentiment: str
    sentiment_id: int
    emoji: str
    confidence: float
    probabilities: dict
    latency_ms: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    version: str

class MetricsResponse(BaseModel):
    total_predictions: int
    avg_latency_ms: float
    total_latency_ms: float
    requests_per_second: float
    error_rate: float
    uptime_seconds: float

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-ready NLP Sentiment Analysis — Negative / Neutral / Positive",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# App state
_start_time = time.time()
_predictor = None
_error_count = 0
_request_count = 0


@app.on_event("startup")
async def startup():
    global _predictor
    try:
        from inference.predictor import SentimentPredictor
        _predictor = SentimentPredictor()
        logger.info("Model loaded successfully at startup.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Sentiment Analysis API",
        "endpoints": ["/predict", "/batch_predict", "/health", "/metrics", "/docs"],
    }


@app.post("/predict", response_model=SentimentResult, tags=["Prediction"])
async def predict(request: PredictRequest):
    """Predict sentiment for a single text input."""
    global _error_count, _request_count
    _request_count += 1

    if _predictor is None:
        _error_count += 1
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    try:
        result = _predictor.predict_one(request.text)
        return SentimentResult(**result)
    except Exception as e:
        _error_count += 1
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=List[SentimentResult], tags=["Prediction"])
async def batch_predict(request: BatchPredictRequest):
    """Predict sentiment for up to 100 texts in a single call."""
    global _error_count, _request_count
    _request_count += 1

    if _predictor is None:
        _error_count += 1
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        results = _predictor.predict_batch(request.texts)
        return [SentimentResult(**r) for r in results]
    except Exception as e:
        _error_count += 1
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """System health check."""
    return HealthResponse(
        status="healthy" if _predictor is not None else "degraded",
        model_loaded=_predictor is not None,
        uptime_seconds=round(time.time() - _start_time, 2),
        version="1.0.0",
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def metrics():
    """Runtime performance metrics."""
    uptime = time.time() - _start_time
    stats = _predictor.stats() if _predictor else {'total_predictions': 0, 'avg_latency_ms': 0, 'total_latency_ms': 0}
    rps = _request_count / uptime if uptime > 0 else 0
    error_rate = _error_count / _request_count if _request_count > 0 else 0

    return MetricsResponse(
        total_predictions=stats['total_predictions'],
        avg_latency_ms=stats['avg_latency_ms'],
        total_latency_ms=stats['total_latency_ms'],
        requests_per_second=round(rps, 4),
        error_rate=round(error_rate, 4),
        uptime_seconds=round(uptime, 2),
    )
