
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

| Necessidade                               | Serviço AWS                                                                                                                             | Por quê                                                                                                                                                          |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rodar o container da API 24/7, escalável | **ECS Fargate**                                                                                                                    | Serverless (sem gerenciar EC2/nós), mesma imagem Docker usada localmente, autoscaling por CPU/latência                                                          |
| Registro da imagem                        | **ECR**                                                                                                                            | Integração nativa com ECS e com o workflow do GitHub Actions (build → push → deploy)                                                                          |
| Entrada de tráfego / TLS / health checks | **Application Load Balancer**                                                                                                      | Distribui requisições entre réplicas, dá o endpoint público estável                                                                                         |
| Orquestração do retreino (DAG Airflow)  | **MWAA** (Managed Workflows for Apache Airflow) *ou* Airflow em container no ECS (mais barato para o escopo do desafio)          | Mesma DAG usada localmente, sem reescrever a lógica de treino                                                                                                    |
| Armazenamento de dados/modelos            | **S3**                                                                                                                             | Fonte dos CSVs de treino e destino dos artefatos de modelo (`.pkl` / `.onnx`) versionados por execução da DAG                                               |
| Monitoramento                             | **Prometheus + Grafana** auto-hospedados (Fargate/EC2) ou **Amazon Managed Service for Prometheus + Amazon Managed Grafana** | Localmente usamos Prometheus/Grafana via Docker Compose; em produção os serviços gerenciados equivalentes evitam operar a stack de observabilidade manualmente |
| CI/CD                                     | **GitHub Actions** (fora da AWS) → deploy via `aws ecs update-service`                                                          | Já é requisito do desafio; a Action publica a imagem no ECR e atualiza o serviço ECS                                                                           |

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

# rodar os testes
poetry run pytest

# lint
poetry run ruff check .
```

## Rodando com Docker

```bash
# build da imagem da API
docker build -t triagem-laudos-api .

# subir a API isolada
docker run -p 8000:8000 triagem-laudos-api

# subir a stack completa (API + Prometheus + Grafana) — feito na Etapa 3
docker compose up --build
```

## Etapas do desafio

- [X] **Estrutura do projeto** — pastas, `.gitignore`, Docker, Poetry, README
- [ ] **Etapa 1** — API FastAPI + Dockerfile + medição de latência baseline
- [ ] **Etapa 2** — GitHub Actions (lint + test) + DAG Airflow de treino
- [ ] **Etapa 3** — Docker Compose (API + Prometheus + Grafana) + dashboard
- [ ] **Etapa 4** — Treino do modelo + otimização ONNX + comparação de latência + vídeo STAR

## Vídeo STAR

Link: _a adicionar na Etapa 4_.
