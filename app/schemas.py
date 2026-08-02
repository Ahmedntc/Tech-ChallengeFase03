"""Schemas Pydantic da API de triagem."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LaudoRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=10,
        description="Texto do laudo/exame a ser classificado.",
        examples=[
            "Patient presents with severe chest pain radiating to the left arm, "
            "shortness of breath and diaphoresis."
        ],
    )


class ClassProbability(BaseModel):
    classe: str
    probabilidade: float


class PredictionResponse(BaseModel):
    classificacao: str
    label_id: int
    confianca: float
    probabilidades: list[ClassProbability]
    latencia_ms: float


class HealthResponse(BaseModel):
    status: str
    modelo_carregado: bool