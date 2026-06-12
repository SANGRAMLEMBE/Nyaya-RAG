.PHONY: setup lint test download-dry download probe smoke

setup:
	uv pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	mypy nyaya

test:
	pytest

download-dry:
	python -m nyaya.pipelines.download --dry-run

download:
	python -m nyaya.pipelines.download --priority 1

probe:
	python -m nyaya.pipelines.probe

smoke: lint test download-dry
