"""Treino do classificador de texto (baseline).

Pipeline: TF-IDF + Logistic Regression. O enunciado do desafio cita
"TF-IDF + Random Forest ou modelo leve similar" apenas como exemplo, não
como obrigação — a comparação em training/compare_models.py mostrou que
Logistic Regression supera a Random Forest em F1-macro, confiança média e
latência de inferência sobre este dataset (ver README, seção "Escolha do
algoritmo"). Este mesmo script é reaproveitado pela DAG do Airflow
(Etapa 2) como a task de treino/salvamento do modelo.

Uso:
    poetry run python -m training.train
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from training.preprocess import LABEL_COLUMN, TEXT_COLUMN, load_dataset, load_labels_map

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
LABELS_PATH = MODELS_DIR / "labels.json"
METRICS_PATH = MODELS_DIR / "metrics_baseline.json"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20_000,
                    # Unigramas apenas: bigramas + stop_words="english" juntos
                    # quebram a conversão para ONNX. O sklearn forma bigramas
                    # a partir do fluxo de tokens JÁ SEM stopwords (ex.: em
                    # "prognosis of patients", remove "of" e cola o bigrama
                    # "prognosis patients"), mas o operador de n-gramas do
                    # ONNX (via skl2onnx) não reproduz esse "pular e colar" —
                    # é uma limitação estrutural do skl2onnx, não um bug de
                    # configuração. Testamos empiricamente (ver
                    # training/compare_models.py e o histórico do PR): sem
                    # bigramas, sklearn e ONNX batem quase exatamente, e essa
                    # configuração ainda teve a MELHOR acurácia entre as
                    # testadas (0.5076 vs 0.5028 com bigramas).
                    ngram_range=(1, 1),
                    stop_words="english",
                    sublinear_tf=True,
                    # token_pattern default do sklearn é \b\w\w+\b (2+
                    # caracteres, com fronteira de palavra). O operador
                    # Tokenizer do ONNX não suporta \b, então o skl2onnx
                    # "traduz" esse padrão para [a-zA-Z0-9_]+ (1+ caractere,
                    # sem fronteira) na conversão. Fixamos aqui o MESMO padrão
                    # simplificado que o ONNX usa, para tokenizar de forma
                    # idêntica nos dois lados por construção (ver
                    # training/optimize_onnx.py).
                    token_pattern=r"[a-zA-Z0-9_]+",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=10.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando dados...")
    train_df, test_df = load_dataset()
    labels_map = load_labels_map()

    X_train, y_train = train_df[TEXT_COLUMN], train_df[LABEL_COLUMN]
    X_test, y_test = test_df[TEXT_COLUMN], test_df[LABEL_COLUMN]

    print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

    pipeline = build_pipeline()

    print("Treinando pipeline (TF-IDF + LogisticRegression)...")
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0
    print(f"Treino concluído em {train_seconds:.1f}s")

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    report = classification_report(
        y_test,
        y_pred,
        target_names=[labels_map[k] for k in sorted(labels_map)],
        zero_division=0,
    )
    print(report)
    print(f"Acurácia: {accuracy:.4f} | F1-macro: {f1_macro:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in labels_map.items()}, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": "tfidf+logistic_regression",
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "train_seconds": round(train_seconds, 2),
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_macro), 4),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Labels salvos em: {LABELS_PATH}")
    print(f"Métricas salvas em: {METRICS_PATH}")


if __name__ == "__main__":
    main()
