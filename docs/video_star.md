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
4. **Escolha do algoritmo + otimização de latência (Etapa 4)** — Mostrar
   `training/compare_models.py` comparando Random Forest, Logistic
   Regression e Linear SVM sobre o mesmo TF-IDF (README, seção "Escolha do
   algoritmo"), a troca do modelo de produção pra Logistic Regression, a
   conversão (`poetry run python -m training.optimize_onnx`) e o
   `app/model_loader.py` com os dois backends (sklearn vs ONNX Runtime) por
   trás da mesma interface `predict_one`.

## Result (~1min)

- Mostrar a tabela comparativa de algoritmos (README, "Escolha do
  algoritmo"): Logistic Regression venceu Random Forest em acurácia,
  F1-macro, confiança média **e** treina ~12x mais rápido.
- Mostrar a tabela de latência ONNX (README, "Otimização de latência com
  ONNX"): Logistic Regression sklearn ~2.9ms vs ONNX ~2.2ms.
- Lições aprendidas a mencionar:
  - **A escolha do algoritmo importou mais que a otimização de runtime**:
    trocar Random Forest por Logistic Regression, sozinho, já derrubou a
    latência de inferência pura de ~54ms pra ~1ms — maior que o ganho que a
    conversão ONNX trouxe para a Random Forest original.
  - **ONNX rende menos quando o modelo já é rápido**: para Logistic
    Regression, boa parte do tempo de resposta é overhead de HTTP/FastAPI,
    não inferência — não dá pra otimizar isso convertendo o modelo.
  - **Validar em lote, não só com 2-3 exemplos**: rodando a comparação
    sklearn-vs-ONNX nas 2.888 amostras de teste (não só em exemplos
    manuais), descobrimos que ~3% das classificações mudavam de classe —
    um bug real na conversão (tokenização + formação de bigramas com
    stopwords removidas), não arredondamento. Corrigido ajustando o TF-IDF
    de treino pra tokenizar de forma idêntica ao ONNX; depois da correção,
    a divergência caiu pra 0.69% (resíduo esperado de float32 vs float64).
  - Observabilidade (Prometheus/Grafana) desde a Etapa 3 permitiu enxergar
    o efeito de cada mudança em métricas reais, não só em benchmark isolado.
- Fechar com o checklist final do README todo marcado.
