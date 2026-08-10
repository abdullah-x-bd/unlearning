.PHONY: install test smoke lint-env

install:
	python -m pip install -e '.[llm,dev]'

test:
	pytest

smoke:
	python -m unlearning_at_scale.cli core-smoke --output runs/core-smoke

lint-env:
	python scripts/capture_env.py --output runs/environment.json
