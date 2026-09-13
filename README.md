# Triagem Automática de Laudos Médicos — Tech Challenge Fase 3 (MLET)

Sistema de triagem automática de exames de texto (laudos médicos), classificando
urgência a partir do texto. Modelo de NLP leve servido via API REST em container
Docker, com pipeline de CI/CD, orquestração de retreino via Airflow, monitoramento
com Prometheus + Grafana e otimização de latência (ONNX).

> Status: projeto em desenvolvimento incremental (ver seção "Etapas" abaixo).

## Dataset

[Medical Abstracts TC Corpus](https://www.kaggle.com/datasets/chaitanyakck/medical-text)
— abstracts médicos rotulados por condição/especialidade (5 classes: neoplasms,
digestive system diseases, nervous system diseases, cardiovascular diseases,
general pathological conditions). Usado aqui como proxy do cenário de triagem por
texto: mesma estrutura (texto → classe), suficiente para demonstrar o pipeline de
ponta a ponta pedido no desafio (≥ 2.000 amostras, coluna de texto + coluna de
target).

Arquivos em `data/raw/`:
- `medical_tc_train.csv` (~11.5k linhas)
- `medical_tc_test.csv` (~2.9k linhas)
- `medical_tc_labels.csv` (mapeamento label → nome da condição)

## Decisão Arquitetural de Deploy em Nuvem

**Estratégia escolhida: real-time (API síncrona), não batch.**

O cenário é triagem clínica: um laudo chega e precisa de uma classificação de
urgência para apoiar priorização de atendimento. Isso é, por definição, uma
necessidade de resposta imediata (segundos), não um job que roda em lote a cada
X horas — um laudo "urgente" processado em batch noturno perde o propósito do
sistema. Por isso a arquitetura é: API REST stateless, containerizada, atrás de
um load balancer, escalando horizontalmente por número de réplicas.

**Nuvem escolhida: AWS.**

Justificativa comparativa (AWS vs Azure vs GCP): as três nuvens resolvem o
problema de forma equivalente; a escolha por AWS aqui é por ser a mais comum em
ambientes de ensino/portfólio e ter o caminho mais direto entre "container
Docker local" e "produção" sem reescrever nada:

| Necessidade | Serviço AWS | Por quê |
|---|---|---|
| Rodar o container da API 24/7, escalável | **ECS Fargate** | Serverless (sem gerenciar EC2/nós), mesma imagem Docker usada localmente, autoscaling por CPU/latência |
| Registro da imagem | **ECR** | Integração nativa com ECS e com o workflow do GitHub Actions (build → push → deploy) |
| Entrada de tráfego / TLS / health checks | **Application Load Balancer** | Distribui requisições entre réplicas, dá o endpoint público estável |
| Orquestração do retreino (DAG Airflow) | **MWAA** (Managed Workflows for Apache Airflow) *ou* Airflow em container no ECS (mais barato para o escopo do desafio) | Mesma DAG usada localmente, sem reescrever a lógica de treino |
| Armazenamento de dados/modelos | **S3** | Fonte dos CSVs de treino e destino dos artefatos de modelo (`.pkl` / `.onnx`) versionados por execução da DAG |
| Monitoramento | **Prometheus + Grafana** auto-hospedados (Fargate/EC2) ou **Amazon Managed Service for Prometheus + Amazon Managed Grafana** | Localmente usamos Prometheus/Grafana via Docker Compose; em produção os serviços gerenciados equivalentes evitam operar a stack de observabilidade manualmente |
| CI/CD | **GitHub Actions** (fora da AWS) → deploy via `aws ecs update-service` | Já é requisito do desafio; a Action publica a imagem no ECR e atualiza o serviço ECS |

Trade-off consciente: Fargate tem cold-start e custo por hora maior que um único
EC2 fixo, mas evita gestão de infraestrutura e escala melhor com picos de
requisições — aceitável para um serviço clínico onde disponibilidade importa mais
que custo mínimo. Para o ambiente **local/desenvolvimento** (o que este
repositório efetivamente roda), tudo é reproduzido com Docker Compose, sem
depender de conta AWS.

## Arquitetura local (o que este repo roda)

```
Cliente ──HTTP──▶ FastAPI (Uvicorn) ──▶ modelo (sklearn/ONNX)
                        │
                        ├──/metrics──▶ Prometheus ──▶ Grafana (dashboards)
                        │
Airflow (DAG) ──▶ lê CSV ──▶ treina ──▶ salva modelo em models/
```

## Estrutura do repositório

```
.
├── app/                 # API FastAPI (inferência + métricas)
├── training/            # Pré-processamento, treino e otimização ONNX
├── airflow/dags/        # DAG de treino/retreino
├── monitoring/          # Configuração Prometheus + Grafana
├── data/raw/            # Dataset (Medical Abstracts TC Corpus)
├── models/              # Modelos treinados (não versionados no git)
├── tests/               # Testes automatizados (pytest)
├── scripts/             # Scripts utilitários (ex.: benchmark de latência)
├── docs/                # Decisões e roteiro do vídeo STAR
├── .github/workflows/   # CI/CD (lint + testes)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml / poetry.lock
```

## Requisitos

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation) `2.x`
- Docker + Docker Compose

## Setup local

```bash
# instalar dependências
poetry install

# ativar o ambiente virtual do Poetry
poetry shell

# treinar o modelo baseline (gera models/model.pkl e models/labels.json)
poetry run python -m training.train

# subir a API localmente
poetry run uvicorn app.main:app --reload

# rodar os testes
poetry run pytest

# lint
poetry run ruff check .
```

Depois de subir a API, teste em outro terminal:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "Patient presents with severe chest pain radiating to the left arm, shortness of breath and diaphoresis."}'
```

## Rodando com Docker

```bash
# build da imagem da API (requer models/model.pkl já treinado — passo acima)
docker build -t triagem-laudos-api .

# subir a API isolada
docker run -p 8000:8000 triagem-laudos-api

# subir a stack completa (API + Prometheus + Grafana) — feito na Etapa 3
docker compose up --build
```

## Modelo baseline e latência (Etapa 1)

Métricas do modelo (TF-IDF + Random Forest, 200 árvores, treinado sobre as
11.550 amostras de treino, avaliado nas 2.888 de teste):

| Métrica | Valor |
|---|---|
| Acurácia | 0.4771 |
| F1-macro | 0.4795 |
| Tempo de treino | 20.0 s |

Acurácia moderada é esperada aqui: são 5 classes de condição clínica com
sobreposição de vocabulário (ex.: "general pathological conditions" é uma
classe "genérica" que puxa recall de outras). Serve como baseline honesto
para comparar contra a versão otimizada em ONNX na Etapa 4 — o foco do
desafio é o pipeline (deploy, CI/CD, monitoramento, latência), não a métrica
de classificação em si.

Latência do `/predict`, medida com `scripts/benchmark_latency.py` **contra o
container Docker** (100 requisições, após warmup):

| Métrica | Valor |
|---|---|
| Média | 46.52 ms |
| Mediana | 48.99 ms |
| p95 | 52.64 ms |
| p99 | 62.87 ms |
| Min | 38.61 ms |
| Max | 99.21 ms |

```bash
docker build -t triagem-laudos-api .
docker run -p 8000:8000 triagem-laudos-api
poetry run python scripts/benchmark_latency.py --url http://localhost:8000 --n 100 --tag baseline
```

Essa é a baseline oficial (rodando no container, como pedido no enunciado) a
ser comparada com o modelo otimizado em ONNX na Etapa 4.

## CI/CD (Etapa 2)

Workflow em `.github/workflows/ci.yml`, disparado em todo push e pull request
para `main`, com 3 jobs em sequência:

1. **lint** — `ruff check .`
2. **test** — `pytest -v` (11 testes: contrato da API mockando o modelo + smoke
   test do pipeline de treino real em dados sintéticos)
3. **build** — treina o modelo baseline e builda a imagem Docker, validando
   que o Dockerfile funciona de ponta a ponta

Rodar localmente antes de dar push:

```bash
poetry run ruff check .
poetry run pytest -v
```

## Orquestração de retreino com Airflow (Etapa 2)

DAG em `airflow/dags/retrain_pipeline_dag.py`, com duas tasks encadeadas
(TaskFlow API): `carregar_dados` → `treinar_modelo`. A lógica de verdade fica
em `training/` — a DAG só chama essas funções, para não duplicar código entre
o treino manual, os testes e o retreino orquestrado.

```bash
# instalar o Airflow num ambiente isolado (não usa o Poetry do projeto —
# ver o comentário no topo de airflow/requirements.txt)
AIRFLOW_VERSION=2.10.3
PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
pip install "apache-airflow==${AIRFLOW_VERSION}" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install -r airflow/requirements.txt

export AIRFLOW_HOME=$(pwd)/airflow
airflow db migrate

# validar que a DAG carrega sem erros
airflow dags list-import-errors

# rodar a DAG completa uma vez (carrega dados + treina + salva o modelo)
airflow dags test triagem_laudos_retrain 2024-01-01

# ou subir a UI (http://localhost:8080) e disparar manualmente
airflow standalone
```

Validado neste ambiente: a DAG carrega sem erros de import, o grafo de tasks
fica correto (`carregar_dados >> treinar_modelo`) e a task `carregar_dados`
roda de ponta a ponta via `airflow tasks test`, retornando
`{'n_train': 11550, 'n_test': 2888}`. A task `treinar_modelo` reaproveita o
`training/train.py` já validado manualmente (Etapa 1) — não foi reexecutada
aqui via Airflow por já levar ~1-2 min com os hiperparâmetros finais.

## Monitoramento (Etapa 3)

A API é instrumentada com `prometheus_client` via middleware HTTP
(`app/metrics.py` + `app/main.py`), expondo em `/metrics`:

- `http_requests_total{method, path, status_code}` — contador de requisições
- `http_request_duration_seconds{method, path}` — histograma de latência

A rota `/metrics` fica de fora da própria instrumentação (senão a raspagem
do Prometheus inflaria as próprias métricas).

```bash
# sobe API + Prometheus + Grafana juntos
docker compose up --build

# API:        http://localhost:8000  (docs em /docs, métricas em /metrics)
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (login anônimo habilitado como Viewer;
#                                      admin/admin por padrão, ver .env)

# gerar tráfego para popular o dashboard
poetry run python scripts/benchmark_latency.py --url http://localhost:8000 --n 100
```

O Prometheus (`monitoring/prometheus/prometheus.yml`) raspa `api:8000/metrics`
a cada 15s. O Grafana já sobe com o datasource do Prometheus e o dashboard
`monitoring/grafana/dashboards/api_dashboard.json` provisionados
automaticamente (`monitoring/grafana/provisioning/`), com 4 painéis:

1. **Total de Requisições** — `sum(http_requests_total)`
2. **Latência (p50/p95/p99)** — `histogram_quantile(...,
   http_request_duration_seconds_bucket)`
3. **Taxa de Erro (5xx)** — proporção de respostas 5xx sobre o total
4. **Requisições por segundo, por endpoint** — `rate(http_requests_total[5m])
   by (path)`

## Etapas do desafio

- [x] **Estrutura do projeto** — pastas, `.gitignore`, Docker, Poetry, README
- [x] **Etapa 1** — API FastAPI (`/health`, `/predict`) + modelo baseline (TF-IDF + Random Forest) + Dockerfile + latência baseline medida
- [x] **Etapa 2** — GitHub Actions (lint + test + build) + DAG Airflow de treino
- [x] **Etapa 3** — Docker Compose (API + Prometheus + Grafana) + dashboard
- [ ] **Etapa 4** — Otimização ONNX + comparação de latência + vídeo STAR

## Vídeo STAR

Link: _a adicionar na Etapa 4_.