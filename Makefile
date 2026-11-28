# Lint só nos .py alterados desde BASE: achados em arquivos que a tarefa não tocou
# não bloqueiam (make lint-all mostra a árvore toda).
BASE ?= HEAD
PY_ALTERADOS := $(shell { git diff --name-only --diff-filter=d $(BASE) -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | sort -u)

.PHONY: lint lint-all test smoke

lint:
ifeq ($(strip $(PY_ALTERADOS)),)
	@echo "Nenhum .py alterado desde $(BASE)."
else
	uvx ruff check --output-format=concise $(PY_ALTERADOS)
	uvx ruff format --check $(PY_ALTERADOS)
endif

lint-all:
	uvx ruff check --output-format=concise .
	uvx ruff format --check .

test:
	uv run --no-sync pytest -q --tb=short

smoke: lint test
