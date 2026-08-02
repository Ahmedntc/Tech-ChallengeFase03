"""DAG de treino/retreino do classificador de triagem de laudos.

Simula o pipeline pedido na Etapa 2: uma task lê o CSV de dados e outra task
treina e salva o modelo. A lógica de verdade mora em `training/` (mesmo
código usado manualmente e no CI) — a DAG só orquestra as chamadas, para não
duplicar a implementação do treino em dois lugares.

Como rodar localmente (fora do Docker Compose, que é só API+Prometheus+Grafana):

    pip install -r airflow/requirements.txt
    export AIRFLOW_HOME=$(pwd)/airflow
    airflow standalone
    # a UI sobe em http://localhost:8080 com a DAG "triagem_laudos_retrain"

Ou, para testar sem subir o webserver/scheduler:

    airflow dags test triagem_laudos_retrain 2024-01-01
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task

# Garante que os pacotes `training/` e `app/` da raiz do repositório sejam
# importáveis a partir do container/processo do Airflow, que roda a partir de
# AIRFLOW_HOME (airflow/) e não da raiz do projeto.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


default_args = {
    "owner": "triagem-laudos",
    "retries": 1,
}


@dag(
    dag_id="triagem_laudos_retrain",
    description="Pipeline de retreino do classificador de triagem de laudos médicos",
    default_args=default_args,
    schedule=None,  # disparo manual; trocar por "@weekly" para retreino periódico
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlet", "tech-challenge", "triagem"],
)
def triagem_laudos_retrain():
    @task
    def carregar_dados() -> dict:
        """Task 1: lê os CSVs de treino/teste e reporta o volume de dados."""
        from training.preprocess import load_dataset

        train_df, test_df = load_dataset()
        info = {"n_train": len(train_df), "n_test": len(test_df)}
        print(f"Dados carregados: {info}")
        return info

    @task
    def treinar_modelo(dados_info: dict) -> dict:
        """Task 2: treina o pipeline (TF-IDF + Random Forest) e salva o modelo."""
        from training.train import main as treinar_e_salvar

        print(f"Iniciando treino com {dados_info['n_train']} amostras de treino...")
        treinar_e_salvar()
        return {"status": "modelo treinado e salvo em models/model.pkl"}

    dados_info = carregar_dados()
    treinar_modelo(dados_info)


triagem_laudos_retrain()
