.PHONY: setup test lint evaluate benchmark verify
setup:
	uv sync --frozen --extra dev
test:
	uv run pytest --cov=rag_lab --cov-report=term-missing --cov-fail-under=85
lint:
	uv run ruff check .
evaluate:
	uv run rag-lab evaluate --output reports/evaluation.json
benchmark:
	uv run rag-lab benchmark --requests 200 --workers 8 --output reports/benchmark.json
verify: lint test evaluate benchmark

.PHONY: neural-setup neural-evaluate neural-test
neural-setup:
	uv sync --frozen --extra dev --extra neural
neural-evaluate:
	uv run --no-sync rag-lab neural-evaluate
neural-test:
	RAG_LAB_REAL_MODELS=1 uv run --no-sync pytest -m real_models -v
