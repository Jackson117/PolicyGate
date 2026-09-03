UV ?= uv
REPO := $(shell pwd)
# Import paths are passed explicitly rather than relying on the .pth files that
# this machine keeps re-hiding. See scripts/fix_editable_installs.sh.
export PYTHONPATH := $(REPO)/src:$(REPO)/third_party/tau2-bench/src
PY := .venv/bin/python

.PHONY: fix sync test lint baseline analyze

fix:
	@./scripts/fix_editable_installs.sh

sync:
	$(UV) sync --extra bench --extra dev
	@./scripts/fix_editable_installs.sh

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
