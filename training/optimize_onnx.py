"""Converte o modelo sklearn treinado (TF-IDF + Random Forest) para ONNX.

Etapa 4: otimização de latência via ONNX Runtime. Reaproveita o mesmo
model.pkl gerado por training/train.py — não retreina nada, só converte o
pipeline já treinado para o formato ONNX (grafo estático, sem overhead do
runtime Python do scikit-learn na hora de servir).

Uso:
    poetry run python -m training.optimize_onnx
"""

from __future__ import annotations

from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
ONNX_PATH = MODELS_DIR / "model.onnx"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} não encontrado. Rode 'poetry run python -m training.train' antes."
        )

    print(f"Carregando pipeline sklearn de {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)

    # TfidfVectorizer espera uma coluna de texto: shape (n_amostras, 1).
    initial_type = [("texto", StringTensorType([None, 1]))]

    print("Convertendo para ONNX (pode levar alguns segundos)...")
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        options={id(pipeline): {"zipmap": False}},
        target_opset=17,
    )

    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    tamanho_pkl_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    tamanho_onnx_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"Modelo ONNX salvo em: {ONNX_PATH}")
    print(f"Tamanho: {tamanho_pkl_mb:.1f} MB (.pkl) -> {tamanho_onnx_mb:.1f} MB (.onnx)")


if __name__ == "__main__":
    main()
