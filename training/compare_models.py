"""Compara algoritmos de classificação candidatos sobre o mesmo TF-IDF.

O enunciado do desafio pede "Scikit-Learn ou framework de preferência" para
o modelo base — Random Forest aparece só como exemplo ("ex.: TF-IDF +
Random Forest ou modelo leve similar"), não como obrigação. Este script
treina três variantes candidatas sobre o dataset real, todas com o mesmo
vetorizador TF-IDF, para comparar acurácia, F1-macro e confiança média
antes de decidir qual algoritmo usar como modelo de produção.

Não faz parte do pipeline oficial de treino (training/train.py, usado pela
API e pela DAG do Airflow) — é uma etapa exploratória.

Uso:
    poetry run python -m training.compare_models
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from training.preprocess import LABEL_COLUMN, TEXT_COLUMN, load_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
COMPARISON_PATH = MODELS_DIR / "model_comparison.json"


def _tfidf() -> TfidfVectorizer:
    # Mesmos parâmetros do training/train.py, para isolar o efeito do
    # classificador — só ele muda entre os três candidatos. Config de TF-IDF
    # "forte" (bigramas + stopwords), só pra comparar algoritmos; o TF-IDF
    # de produção em training/train.py usa unigramas por motivo de
    # compatibilidade com ONNX (ver comentário lá).
    return TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )


def build_candidates() -> dict[str, Pipeline]:
    return {
        "tfidf+random_forest": Pipeline(
            steps=[
                ("tfidf", _tfidf()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=40,
                        n_jobs=-1,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "tfidf+logistic_regression": Pipeline(
            steps=[
                ("tfidf", _tfidf()),
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
        ),
        "tfidf+linear_svm": Pipeline(
            steps=[
                ("tfidf", _tfidf()),
                (
                    "clf",
                    # LinearSVC não tem predict_proba nativo (é um classificador
                    # de margem, não probabilístico) — CalibratedClassifierCV
                    # ajusta uma sigmoide/isotonic sobre a saída pra estimar
                    # probabilidades comparáveis aos outros dois candidatos.
                    CalibratedClassifierCV(
                        LinearSVC(
                            class_weight="balanced",
                            random_state=42,
                            max_iter=5000,
                        ),
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def evaluate(name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    print(f"Treinando {name}...")
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    proba = pipeline.predict_proba(X_test)
    y_pred = pipeline.classes_[np.argmax(proba, axis=1)]

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    mean_confidence = float(np.mean(np.max(proba, axis=1)))

    result = {
        "model": name,
        "train_seconds": round(train_seconds, 2),
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_macro), 4),
        "mean_confidence": round(mean_confidence, 4),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    print("Carregando dados...")
    train_df, test_df = load_dataset()

    X_train, y_train = train_df[TEXT_COLUMN], train_df[LABEL_COLUMN]
    X_test, y_test = test_df[TEXT_COLUMN], test_df[LABEL_COLUMN]
    print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras\n")

    results = [
        evaluate(name, pipeline, X_train, y_train, X_test, y_test)
        for name, pipeline in build_candidates().items()
    ]
    results.sort(key=lambda r: r["f1_macro"], reverse=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== Ranking (por F1-macro) ===")
    for r in results:
        print(
            f"{r['model']:28s} acc={r['accuracy']:.4f}  f1_macro={r['f1_macro']:.4f}  "
            f"confianca_media={r['mean_confidence']:.4f}  treino={r['train_seconds']}s"
        )
    print(f"\nResultado salvo em {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
