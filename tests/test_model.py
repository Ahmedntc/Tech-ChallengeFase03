"""Testes do pipeline de treino (training/).

Usam poucos dados sintéticos para rodar rápido no CI — o objetivo aqui é
validar que o código do pipeline (vetorização + classificador) funciona de
ponta a ponta, não medir a qualidade do modelo (isso é feito manualmente ao
rodar `training/train.py` sobre o dataset completo).
"""

from __future__ import annotations

from training.preprocess import clean_text, load_labels_map
from training.train import build_pipeline


def test_clean_text_normaliza_espacos_e_quebras_de_linha():
    sujo = "Texto  com\nquebras\r\nde   linha   e   espaços  extras "
    limpo = clean_text(sujo)
    assert "\n" not in limpo
    assert "\r" not in limpo
    assert "  " not in limpo
    assert limpo == limpo.strip()


def test_clean_text_com_entrada_invalida_retorna_string_vazia():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_load_labels_map_retorna_dicionario_com_5_classes():
    labels = load_labels_map()
    assert isinstance(labels, dict)
    assert len(labels) == 5
    assert all(isinstance(k, int) for k in labels)
    assert all(isinstance(v, str) for v in labels.values())


def test_build_pipeline_fit_predict_smoke():
    """Treina o pipeline real (TF-IDF + RandomForest) em dados sintéticos
    minúsculos, só para garantir que fit/predict/predict_proba funcionam.
    """
    pipeline = build_pipeline()
    # reduz o custo computacional só para o teste
    pipeline.set_params(clf__n_estimators=5, clf__max_depth=3)

    textos = [
        "chest pain and shortness of breath",
        "abdominal pain and bloody diarrhea",
        "headache and numbness in the legs",
        "persistent cough and weight loss",
    ] * 5
    labels = [4, 2, 3, 1] * 5

    pipeline.fit(textos, labels)
    preds = pipeline.predict(textos[:2])
    probas = pipeline.predict_proba(textos[:2])

    assert len(preds) == 2
    assert probas.shape[0] == 2
    assert all(abs(row.sum() - 1.0) < 1e-6 for row in probas)
