.PHONY: install train card run-api test lint

install:
	python -m venv .venv && . .venv/bin/activate && pip install -e .[dev]

train:
	python scripts/train_model.py

card:
	python scripts/make_model_card.py

run-api:
	uvicorn loan_approval.api.main:app --reload --port 8300

test:
	pytest

lint:
	ruff check src tests scripts
