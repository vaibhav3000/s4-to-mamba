# CPU-only image that installs the package and runs the test suite. It does not
# reproduce the GPU results; see results/PROVENANCE.md for those commands.
# Base image pinned by digest (python:3.13-slim, pulled 2026-11-19).
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 10001 app
WORKDIR /app

# CPU torch in its own layer, so source edits do not re-download it.
RUN python -m pip install "torch==2.14.*" --index-url https://download.pytorch.org/whl/cpu

# Non-editable install: the tests import the installed package.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install ".[dev]"

# Test inputs: the suite reads configs, committed results and the figure script.
COPY tests ./tests
COPY configs ./configs
COPY results ./results
COPY scripts ./scripts

USER app
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
