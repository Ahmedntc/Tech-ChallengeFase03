# syntax=docker/dockerfile:1

# ---------- Stage 1: build das dependências com Poetry ----------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.4.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

# Copia apenas os arquivos de dependências primeiro (cache de layer)
COPY pyproject.toml poetry.lock ./

# Instala só as dependências de produção (sem grupo dev), sem instalar o projeto em si
RUN poetry install --only main --no-root --no-ansi

# ---------- Stage 2: imagem final, enxuta ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Usuário não-root
RUN groupadd -r app && useradd -r -g app app

# Copia o virtualenv já resolvido do estágio de build
COPY --from=builder /app/.venv /app/.venv

# Copia o código da aplicação e o modelo treinado
COPY app ./app
COPY models ./models

RUN chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health').status == 200 else sys.exit(1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]