FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /svc

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM base AS api

ENV PYTHONPATH=/svc/app
COPY app/ /svc/app/
WORKDIR /svc/app
EXPOSE 8080
CMD ["uv", "run", "--project", "/svc", "python", "api.py"]


FROM base AS ui

ENV API_BASE_URL=http://app:8080
COPY webview/ /svc/webview/
WORKDIR /svc/webview
EXPOSE 8501
CMD ["uv", "run", "--project", "/svc", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


FROM base AS worker

ENV PYTHONPATH=/svc/ml_worker:/svc/app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group ml

COPY app/ /svc/app/
COPY ml_worker/ /svc/ml_worker/
WORKDIR /svc/ml_worker
CMD ["uv", "run", "--project", "/svc", "python", "main.py"]
