UV ?= uv
PY := .venv/bin/python

.PHONY: sync test lint baseline analyze viz ablate ablate-watched purity service

sync:
	$(UV) sync --extra bench --extra dev

test:
	$(PY) -m pytest

lint:
	$(UV) run ruff check src tests scripts

# W1 feasibility baseline.
MODEL ?= openai/gpt-4.1-mini
baseline:
	$(PY) scripts/w1_baseline.py --agent-llm $(MODEL) --user-llm $(MODEL)

analyze:
	$(PY) scripts/w1_analyze.py $(RESULTS)

# --- W4 and after -----------------------------------------------------------

# .env is the project's provider configuration, and a shell that exports
# OPENAI_API_KEY would otherwise shadow it: load_dotenv never overrides what is
# already set, so the process would quietly use another account's key against
# this project's base URL. Sourcing it here makes the environment explicit.
ENV := set -a; . ./.env; set +a;

# The dashboard over experiments/. POLICYGATE_RUN_DIR picks the run it opens.
viz:
	$(ENV) $(UV) run streamlit run src/policygate/viz/app.py --server.port 8501

# One ablation run. Costs money: --dry-run first, and mind --budget.
OUT ?= experiments/run
BUDGET ?= 30
ablate:
	$(UV) run python scripts/w4_run.py --budget $(BUDGET) --out $(OUT)

# The same, restarted whenever it stops making progress (see the script).
ablate-watched:
	bash scripts/w4_watchdog.sh $(OUT) $(BUDGET)

# harness/ imports no agent framework. CI runs this too.
purity:
	$(UV) run python scripts/check_no_framework_deps.py

# The deployment path: the approval service the Live session page talks to.
service:
	$(ENV) $(UV) run uvicorn policygate.service.api:build --factory --port 8000
