FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY .python-version ./
RUN uv sync --frozen --extra oslo --no-install-project

COPY src ./src
COPY codecs/oslo-360-degree-image-compression ./codecs/oslo-360-degree-image-compression
RUN uv sync --frozen --extra oslo

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["python", "-m", "n360ic_benchmark.adapters.oslo"]
CMD ["--help"]
