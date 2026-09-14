"""Carregamento do modelo treinado e função de inferência.

Isolado do main.py para poder ser mockado/testado sem subir a API inteira.
Etapa 4: suporta dois backends de inferência — sklearn (.pkl, padrão) e ONNX
Runtime (.onnx, mais rápido) — selecionados pela env var MODEL_FORMAT, com a
mesma interface pública (predict_one, is_model_ready) para ambos.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_FORMAT = os.getenv("MODEL_FORMAT", "sklearn").lower()
_DEFAULT_MODEL_NAME = "model.onnx" if MODEL_FORMAT == "onnx" else "model.pkl"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / _DEFAULT_MODEL_NAME)))
LABELS_PATH = Path(os.getenv("LABELS_PATH", str(MODELS_DIR / "labels.json")))


class ModelNotLoadedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_pipeline():
    """Carrega o pipeline sklearn (.pkl) — usado quando MODEL_FORMAT=sklearn."""
    if not MODEL_PATH.exists():
        raise ModelNotLoadedError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Rode 'poetry run python -m training.train' antes de subir a API."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_onnx_session():
    """Carrega a sessão ONNX Runtime — usado quando MODEL_FORMAT=onnx."""
    import onnxruntime as ort

    if not MODEL_PATH.exists():
        raise ModelNotLoadedError(
            f"Modelo ONNX não encontrado em {MODEL_PATH}. "
            "Rode 'poetry run python -m training.optimize_onnx' antes de subir a API."
        )
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def get_labels_map() -> dict[int, str]:
    if not LABELS_PATH.exists():
        raise ModelNotLoadedError(f"Arquivo de labels não encontrado em {LABELS_PATH}.")
    with open(LABELS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def is_model_ready() -> bool:
    try:
        get_onnx_session() if MODEL_FORMAT == "onnx" else get_pipeline()
        get_labels_map()
        return True
    except ModelNotLoadedError:
        return False


def _build_result(proba, classes, labels_map: dict[int, str]) -> dict:
    best_idx = int(np.argmax(proba))
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


def _predict_sklearn(texto: str) -> dict:
    pipeline = get_pipeline()
    labels_map = get_labels_map()

    proba = pipeline.predict_proba([texto])[0]
    return _build_result(proba, pipeline.classes_, labels_map)


def _predict_onnx(texto: str) -> dict:
    session = get_onnx_session()
    labels_map = get_labels_map()

    input_name = session.get_inputs()[0].name
    entrada = np.array([[texto]], dtype=object)
    _, proba_matrix = session.run(None, {input_name: entrada})

    # RandomForestClassifier.classes_ é sempre np.unique(y) em ordem
    # crescente — mesma ordem preservada pelo skl2onnx na matriz de saída.
    classes = np.array(sorted(labels_map.keys()))
    return _build_result(proba_matrix[0], classes, labels_map)


def predict_one(texto: str) -> dict:
    """Roda a inferência para um único texto de laudo.

    Retorna a classe prevista, o id da classe, a confiança (probabilidade da
    classe vencedora) e a distribuição de probabilidade sobre todas as classes.
    Delega para o backend sklearn ou ONNX conforme MODEL_FORMAT.
    """
    if MODEL_FORMAT == "onnx":
        return _predict_onnx(texto)
    return _predict_sklearn(texto)
