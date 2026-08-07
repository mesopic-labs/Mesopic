# The same gate as CI, one command. If `make check` is green, CI will be too.
.DEFAULT_GOAL := help
.PHONY: help setup fmt lint types arch test check cov image run demo clean

UV ?= uv
ALL := muster-engine/src muster-engine/tests

help:  ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the workspace and the git hooks
	$(UV) sync
	$(UV) run pre-commit install

fmt:  ## Format
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

lint:  ## Lint and check formatting
	$(UV) run ruff check .
	$(UV) run ruff format --check .

types:  ## mypy --strict
	$(UV) run mypy $(ALL)

arch:  ## Enforce the module boundaries
	$(UV) run lint-imports --verbose

test:  ## Fast tests (excludes golden clips and the perf harness)
	$(UV) run pytest -m "not slow"

cov:  ## Fast tests with coverage
	$(UV) run pytest -m "not slow" --cov --cov-report=term-missing

check: lint types arch test  ## The merge gate

image:  ## Build the engine container and assert its invariants
	docker build -f docker/engine.Dockerfile -t muster-engine:dev .
	scripts/assert-no-model-weights.sh muster-engine:dev
	docker run --rm muster-engine:dev version

run:  ## Run the engine against ./muster.yaml
	$(UV) run muster run --config ./muster.yaml

demo:  ## Bring up the engine plus a synthetic camera and an MQTT broker
	docker compose --profile demo up --build

clean:  ## Remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov dist build
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
