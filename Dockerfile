# The deployment: the approval service and the dashboard, not an evaluation rig.
#
# The harness itself imports no agent framework, but the service does: it runs
# real retail tools through the tau2 environment and speaks tau2's message
# models, so the pinned checkout (third_party/PIN.md) is installed here. Only
# the retail domain's data comes with it -- see .dockerignore, which keeps the
# other domains and the knowledge corpus out of a 1.6 GB context. Evaluation
# stays on the host, where the API keys and the full checkout live.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.9.10 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first: the layer survives every source edit. The project itself
# is installed from source afterwards, so pyproject alone is what invalidates it.
# tau2 first and on its own layer: it is the heaviest install and the one that
# changes least -- its commit is pinned. It brings litellm, fastapi and pandas
# with it, which is why those are not repeated here. plotly is only in tau2's
# `experiments` extra (with scikit-learn and seaborn), so the dashboard's charts
# install it themselves below.
COPY third_party/tau2-bench/ ./third_party/tau2-bench/
RUN uv pip install --system --no-cache ./third_party/tau2-bench

COPY pyproject.toml ./
RUN uv pip install --system --no-cache \
      "pydantic>=2.9" "deepdiff>=8.4.2" "pyyaml>=6.0" \
      "sqlalchemy>=2.0" "alembic>=1.13" "psycopg[binary]>=3.2" \
      "langgraph>=0.2" "langgraph-checkpoint-postgres>=2.0" \
      "uvicorn>=0.30" "streamlit>=1.38" "plotly>=6.0"

COPY src/ ./src/
ENV PYTHONPATH=/app/src
# tau2 resolves its data relative to its own source tree unless told otherwise,
# and this install is not editable, so point it at the checkout that came in.
ENV TAU2_DATA_DIR=/app/third_party/tau2-bench/data

# A non-root user, and a home for the audit database when it is SQLite.
RUN useradd --create-home --uid 10001 policygate \
    && mkdir -p /data && chown policygate /data
USER policygate
ENV POLICYGATE_AUDIT_URL=sqlite:////data/audit.db

FROM base AS service
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/approvals')"
CMD ["uvicorn", "policygate.service.api:build", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]

FROM base AS dashboard
EXPOSE 8501
# Read-only: the dashboard reads run directories mounted from the host.
ENV POLICYGATE_RUN_ROOT=/runs POLICYGATE_RUN_DIR=/runs/w4_v4 POLICYGATE_RAG_RESULTS=/rag
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "src/policygate/viz/app.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0", \
     "--server.headless", "true", "--browser.gatherUsageStats", "false"]
