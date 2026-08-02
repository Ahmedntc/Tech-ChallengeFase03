"""Carregamento do modelo treinado e função de inferência.

Isolado do main.py para poder ser mockado/testado sem subir a API inteira,
e para ser facilmente trocado por um runtime ONNX na Etapa 4 (mesma
interface: predict_one(texto) -> dict).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "model.pkl")))
LABELS_PATH = Path(os.getenv("LABELS_PATH", str(MODELS_DIR / "labels.json")))


class ModelNotLoadedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_pipeline():
    if not MODEL_PATH.exists():
        raise ModelNotLoadedError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Rode 'poetry run python -m training.train' antes de subir a API."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_labels_map() -> dict[int, str]:
    if not LABELS_PATH.exists():
        raise ModelNotLoadedError(f"Arquivo de labels não encontrado em {LABELS_PATH}.")
    with open(LABELS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def is_model_ready() -> bool:
    try:
        get_pipeline()
        get_labels_map()
        return True
    except ModelNotLoadedError:
        return False


def predict_one(texto: str) -> dict:
    """Roda a inferência para um único texto de laudo.

    Retorna a classe prevista, o id da classe, a confiança (probabilidade da
    classe vencedora) e a distribuição de probabilidade sobre todas as classes.
    """
    pipeline = get_pipeline()
    labels_map = get_labels_map()

    proba = pipeline.predict_proba([texto])[0]
    classes = pipeline.classes_

    best_idx = proba.argmax()
    best_label_id = int(classes[best_idx])

    probabilidades = [
        {"classe": labels_map.get(int(c), str(c)), "probabilidade": float(p)}
        for c, p in sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    ]

    return {
        "classificacao": labels_map.get(best_label_id, str(best_label_id)),
        "label_id": best_label_id,
        "confianca": float(proba[best_idx]),
        "probabilidades": probabilidades,
    }