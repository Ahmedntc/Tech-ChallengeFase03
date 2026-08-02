"""API de triagem automática de laudos médicos.

Etapa 1: endpoints de inferência simples, sem instrumentação de métricas
(Prometheus entra na Etapa 3, em app/metrics.py).
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, status

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
