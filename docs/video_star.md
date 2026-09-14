# Roteiro do Vídeo STAR (≤ 5 minutos)

Roteiro de apoio para a gravação exigida pelo Tech Challenge Fase 3. Cada
bloco indica o tempo sugerido e o que mostrar na tela.

## Situation (~45s)

- Um hospital de referência precisa triar laudos médicos por urgência
  (normal / atenção / urgente) mais rápido do que um processo manual permite.
- Cada minuto de atraso na triagem de um caso urgente tem custo clínico real
  — o problema não é só "classificar texto", é fazer isso em tempo de
  resposta compatível com atendimento em produção.
- Mostrar na tela: o dataset (`data/raw/medical_tc_*.csv`, 5 classes de
  condição clínica) como proxy desse cenário de triagem.

## Task (~45s)

- Requisitos técnicos da fase: API real-time em container Docker, pipeline
  de CI/CD, orquestração de retreino, monitoramento com Prometheus/Grafana e
  otimização de latência.
- Mostrar na tela: seção "Etapas do desafio" do `README.md` (checklist das
  4 etapas).

## Action (~2min30s)

1. **Arquitetura e API (Etapa 1)** — Mostrar `app/main.py` (`/health`,
   `/predict`) e a decisão de deploy real-time em AWS (ECS Fargate) no
   README, justificando por que batch não serve para triagem de urgência.
2. **CI/CD e Airflow (Etapa 2)** — Mostrar o workflow
   `.github/workflows/main.yml` rodando (lint → test → build) e a DAG
   `airflow/dags/retrain_pipeline_dag.py` (`carregar_dados >>
   treinar_modelo`).
3. **Monitoramento (Etapa 3)** — Rodar `docker compose up` e mostrar o
   dashboard do Grafana (`monitoring/grafana/dashboards/api_dashboard.json`)
   com os 4 painéis enquanto dispara requisições com
   `scripts/benchmark_latency.py`.
4. **Otimização de latência (Etapa 4)** — Mostrar a conversão
   (`poetry run python -m training.optimize_onnx`) e o `app/model_loader.py`
   com os dois backends (sklearn vs ONNX Runtime) por trás da mesma
   interface `predict_one`.

## Result (~1min)

- Mostrar a tabela comparativa de latência (README, seção "Otimização de
  latência com ONNX"): sklearn ~43.8 ms médios vs ONNX Runtime ~1.8 ms
  médios — ganho de ~24x, sem mudar a classificação retornada.
- Lições aprendidas a mencionar:
  - Ganho de latência de otimizar o *runtime* de inferência foi muito maior
    que qualquer ajuste de hiperparâmetro do modelo.
  - Compatibilidade de versões (onnx/protobuf/skl2onnx) foi o maior atrito
    técnico da conversão — documentado em `pyproject.toml`.
  - Observabilidade (Prometheus/Grafana) desde a Etapa 3 permitiu enxergar
    o efeito da otimização em métricas reais, não só em benchmark isolado.
- Fechar com o checklist final do README todo marcado.
