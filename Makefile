.PHONY: install test lint typecheck docs repro clean

install:
	pip install -e ".[dev,diff]"

test:
	pytest -v

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy src/mcl_bio

docs:
	mkdocs build

repro:
	python -m mcl_bio.cli quickstart
	python -m mcl_bio.cli vex

clean:
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache site htmlcov
