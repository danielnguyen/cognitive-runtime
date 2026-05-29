SHELL := /usr/bin/env bash

.PHONY: dev-test dev-install dev-lint dev-check dev-start dev-start-reload

dev-test:
	@cd api && ./.venv/bin/python -m pytest -q

dev-install:
	@cd api && ./.venv/bin/python -m pip install -r requirements.txt

dev-lint:
	@cd api && ./.venv/bin/python -m ruff check .

dev-check: dev-lint dev-test

dev-start:
	@cd api && ./.venv/bin/uvicorn main:app --host 0.0.0.0 --port "$${APP_PORT:-4371}"

dev-start-reload:
	@cd api && ./.venv/bin/uvicorn main:app --host 0.0.0.0 --port "$${APP_PORT:-4371}" --reload
