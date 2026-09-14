"""Métricas Prometheus da API de triagem.

Definidas num módulo à parte (em vez de direto em app/main.py) para que o
middleware e os testes importem sempre o mesmo objeto Counter/Histogram —
registrar a mesma métrica duas vezes no registry default do
prometheus_client levanta ValueError ("Duplicated timeseries").
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total de requisições HTTP recebidas pela API",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Latência das requisições HTTP em segundos",
    ["method", "path"],
)
