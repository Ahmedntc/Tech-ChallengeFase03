"""Mede a latência do endpoint /predict da API já em execução (local ou Docker).

Uso:
    poetry run python scripts/benchmark_latency.py --url http://localhost:8000 --n 100

Salva o resultado em models/latency_baseline.json (Etapa 1) ou, se --tag onnx
for passado, em models/latency_onnx.json (usado na comparação da Etapa 4).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SAMPLE_TEXTS = [
    "Patient presents with severe chest pain radiating to the left arm, "
    "shortness of breath and diaphoresis, suggestive of acute myocardial infarction.",
    "Chronic abdominal pain with recurrent episodes of bloody diarrhea and weight "
    "loss, consistent with inflammatory bowel disease.",
    "Progressive weakness and numbness in the lower extremities, with imaging "
    "showing demyelinating lesions in the spinal cord.",
    "Persistent cough, unintentional weight loss and a mass identified on chest "
    "X-ray, with biopsy pending to rule out malignancy.",
    "Patient reports fatigue, joint pain and low-grade fever for the past three "
    "weeks, laboratory workup in progress.",
]


def run_benchmark(base_url: str, n_requests: int, warmup: int = 5) -> dict:
    # trust_env=False: ignora proxies do sistema (ex.: HTTP_PROXY/ALL_PROXY),
    # já que estamos sempre falando com a API local/container, nunca com a internet.
    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        # warmup: primeira(s) chamada(s) costuma(m) ser mais lenta (import lazy, cache)
        for i in range(warmup):
            client.post("/predict", json={"texto": SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]})

        latencies_ms = []
        for i in range(n_requests):
            texto = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
            t0 = time.perf_counter()
            resp = client.post("/predict", json={"texto": texto})
            elapsed_ms = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            latencies_ms.append(elapsed_ms)

    latencies_ms.sort()

    def pct(p: float) -> float:
        idx = int(len(latencies_ms) * p) - 1
        return latencies_ms[max(idx, 0)]

    return {
        "n_requests": n_requests,
        "mean_ms": round(statistics.mean(latencies_ms), 2),
        "median_ms": round(statistics.median(latencies_ms), 2),
        "p95_ms": round(pct(0.95), 2),
        "p99_ms": round(pct(0.99), 2),
        "min_ms": round(min(latencies_ms), 2),
        "max_ms": round(max(latencies_ms), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark de latência da API de triagem")
    parser.add_argument("--url", default="http://localhost:8000", help="URL base da API")
    parser.add_argument("--n", type=int, default=100, help="Número de requisições")
    parser.add_argument(
        "--tag",
        default="baseline",
        help="Nome do arquivo de saída: models/latency_<tag>.json",
    )
    args = parser.parse_args()

    print(f"Rodando benchmark contra {args.url} ({args.n} requisições)...")
    result = run_benchmark(args.url, args.n)
    print(json.dumps(result, indent=2))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / f"latency_{args.tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Resultado salvo em {out_path}")


if __name__ == "__main__":
    main()