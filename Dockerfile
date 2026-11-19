# CPU-only image that installs the locked environment (uv.lock, cpu extra) and runs the
# test suite. It does not reproduce the GPU results; see results/PROVENANCE.md for those.
# Base image pinned by digest (python:3.13-slim, pulled 2026-11-19).
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PATH=/opt/venv/bin:$PATH

RUN useradd --create-home --uid 10001 app \
    && python -m pip install "uv==0.11.32"
WORKDIR /app

# Dependencies from the lock in their own layer, so source edits do not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --extra cpu --no-install-project

COPY src ./src
RUN uv sync --locked --extra cpu

# Test inputs: the suite reads configs, committed results and the scripts.
COPY tests ./tests
COPY configs ./configs
COPY results ./results
COPY scripts ./scripts

USER app
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
