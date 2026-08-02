"""Testes da API FastAPI.

Não dependem de um modelo treinado de verdade: `app.main.predict_one` e
`app.main.is_model_ready` são mockados via monkeypatch, para que o CI rode
rápido e não precise treinar o modelo (~1-2 min) só para validar a camada
HTTP/contrato da API.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.model_loader import ModelNotLoadedError

client = TestClient(app)

FAKE_PREDICTION = {
    "classificacao": "cardiovascular diseases",
    "label_id": 4,
    "confianca": 0.87,
    "probabilidades": [
        {"classe": "cardiovascular diseases", "probabilidade": 0.87},
        {"classe": "neoplasms", "probabilidade": 0.13},
    ],
}

TEXTO_VALIDO = (
    "Patient presents with severe chest pain radiating to the left arm, "
    "shortness of breath and diaphoresis."
)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "triagem-laudos-api"


def test_health_com_modelo_carregado(monkeypatch):
    monkeypatch.setattr("app.main.is_model_ready", lambda: True)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "modelo_carregado": True}


def test_health_sem_modelo_carregado(monkeypatch):
    monkeypatch.setattr("app.main.is_model_ready", lambda: False)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "modelo_carregado": False}


def test_predict_sucesso(monkeypatch):
    monkeypatch.setattr("app.main.predict_one", lambda texto: FAKE_PREDICTION)

    resp = client.post("/predict", json={"texto": TEXTO_VALIDO})

    assert resp.status_code == 200
    body = resp.json()
    assert body["classificacao"] == "cardiovascular diseases"
    assert body["label_id"] == 4
    assert 0.0 <= body["confianca"] <= 1.0
    assert isinstance(body["probabilidades"], list)
    assert "latencia_ms" in body
    assert body["latencia_ms"] >= 0


def test_predict_texto_muito_curto():
    resp = client.post("/predict", json={"texto": "curto"})
    assert resp.status_code == 422


def test_predict_sem_texto():
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_predict_modelo_indisponivel(monkeypatch):
    def _raise(_texto):
        raise ModelNotLoadedError("modelo não encontrado")

    monkeypatch.setattr("app.main.predict_one", _raise)

    resp = client.post("/predict", json={"texto": TEXTO_VALIDO})

    assert resp.status_code == 503
