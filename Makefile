# Variables
PY=python
POETRY=poetry

.PHONY: dev eval install

install:
	$(POETRY) install

# Start dev server and open browser automatically
# Uses scripts/dev.py to handle cross-platform browser open and server spawn
dev: install
	$(POETRY) run $(PY) scripts/dev.py

# Run evaluator and print concise metrics
# Requires the dev server running (localhost:8000 by default)
eval: install
	$(POETRY) run $(PY) scripts/eval.py 