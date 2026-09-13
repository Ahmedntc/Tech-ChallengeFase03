"""API de triagem automática de laudos médicos.

Etapa 3: instrumentada com prometheus_client (app/metrics.py) via
middleware HTTP, expondo contagem de requisições e latência em /metrics.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.metrics import REQUEST_COUNT, REQUEST_LATENCY
from app.model_loader import ModelNotLoadedError, is_model_ready, predict_one
from app.schemas import HealthResponse, LaudoRequest, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("triagem-api")

app = FastAPI(
    title="API de Triagem de Laudos Médicos",
    description=(
        "Classifica o texto de um laudo/exame em uma categoria clínica, "
        "servindo como base para triagem automática de urgência."
    ),
    version="0.1.0",
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Registra contagem e latência de toda requisição HTTP, exceto /metrics
    (evita ruído de uma métrica medindo a própria raspagem do Prometheus)."""
    if request.url.path == "/metrics":
        return await call_next(request)

    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    REQUEST_LATENCY.labels(method=request.method, path=request.url.path).observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
    ).inc()
    return response


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "triagem-laudos-api",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(status="ok", modelo_carregado=is_model_ready())


@app.get("/metrics", tags=["meta"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["inferencia"])
def predict(payload: LaudoRequest):
    start = time.perf_counter()
    try:
        result = predict_one(payload.texto)
    except ModelNotLoadedError as exc:
        logger.error("Modelo indisponível: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    latencia_ms = (time.perf_counter() - start) * 1000
    return PredictionResponse(
        classificacao=result["classificacao"],
        label_id=result["label_id"],
        confianca=result["confianca"],
        probabilidades=result["probabilidades"],
        latencia_ms=round(latencia_ms, 2),
    )
