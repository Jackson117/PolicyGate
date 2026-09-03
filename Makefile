UV ?= uv
PY := .venv/bin/python

.PHONY: sync test lint baseline analyze

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
