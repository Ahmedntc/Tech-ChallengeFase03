"""Funções de carregamento e limpeza de dados usadas pelo treino do modelo.

Mantidas separadas de train.py para que a mesma lógica possa ser reutilizada
pela DAG do Airflow (Etapa 2) sem duplicar código.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

TRAIN_FILE = DATA_DIR / "medical_tc_train.csv"
TEST_FILE = DATA_DIR / "medical_tc_test.csv"
LABELS_FILE = DATA_DIR / "medical_tc_labels.csv"

TEXT_COLUMN = "medical_abstract"
LABEL_COLUMN = "condition_label"


def clean_text(text: str) -> str:
    """Limpeza leve do texto do laudo/abstract antes da vetorização TF-IDF.

    Não removemos stopwords aqui (o TfidfVectorizer já cuida disso) — só
    normalizamos espaços e caracteres de controle que vêm do CSV original.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_labels_map(labels_path: Path = LABELS_FILE) -> dict[int, str]:
    """Carrega o mapeamento condition_label -> condition_name."""
    df = pd.read_csv(labels_path)
    return dict(zip(df["condition_label"], df["condition_name"]))


def load_dataset(
    train_path: Path = TRAIN_FILE,
    test_path: Path = TEST_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega e limpa os conjuntos de treino e teste.

    Retorna dois DataFrames com colunas [TEXT_COLUMN, LABEL_COLUMN].
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for df in (train_df, test_df):
        df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)
        df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)
        df.drop(df[df[TEXT_COLUMN] == ""].index, inplace=True)

    return train_df, test_df
