# The same gate as CI, one command. If `make check` is green, CI will be too.
.DEFAULT_GOAL := help
.PHONY: help setup fmt lint types arch test check cov image run demo gate gif compose-check test-stream clean

UV ?= uv
ALL := mesopic-engine/src mesopic-engine/tests

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
	docker build -f docker/engine.Dockerfile -t mesopic-engine:dev .
	scripts/assert-no-model-weights.sh mesopic-engine:dev
	docker run --rm mesopic-engine:dev version

run:  ## Run the engine against ./mesopic.yaml
	$(UV) run mesopic run --config ./mesopic.yaml

demo:  ## THE one command: engine + dashboard + sample stream + broker
	@command -v openssl >/dev/null || { echo "error: openssl not found; export MESOPIC_ADMIN_PASSWORD yourself" >&2; exit 1; }
	@set -eu; \
	: "$${MESOPIC_ADMIN_PASSWORD:=$$(openssl rand -hex 16)}"; \
	: "$${MESOPIC_RTSP_URL:=rtsp://mediamtx:8554/synthetic}"; \
	export MESOPIC_ADMIN_PASSWORD MESOPIC_RTSP_URL; \
	docker compose --profile demo build; \
	docker compose --profile demo run --rm seed; \
	printf '\n  dashboard   http://localhost:8080\n'; \
	printf '  password    %s\n' "$$MESOPIC_ADMIN_PASSWORD"; \
	printf '  stream      %s\n\n' "$$MESOPIC_RTSP_URL"; \
	printf '  Reading the dashboard needs no password. The password guards changes —\n'; \
	printf '  drawing zones and lines. It is generated per run unless you export one.\n\n'; \
	printf '  THE COUNTS WILL READ ZERO. The bundled stream is a test pattern with no\n'; \
	printf '  people in it, so there is nothing to count; what this shows is the pipeline\n'; \
	printf '  running end to end. Export MESOPIC_RTSP_URL to point it at a real camera.\n\n'; \
	docker compose --profile demo up

gate:  ## The M2 gate (P4.5): both ingest paths on one clip, both adjacencies, every exporter
	@command -v openssl >/dev/null || { echo "error: openssl not found; export MESOPIC_ADMIN_PASSWORD yourself" >&2; exit 1; }
	@test -f examples/clips/sample.mp4 || { echo "error: examples/clips/sample.mp4 is missing — the gate needs footage with people in it" >&2; exit 1; }
	@set -eu; \
	: "$${MESOPIC_ADMIN_PASSWORD:=$$(openssl rand -hex 16)}"; \
	: "$${MESOPIC_RTSP_URL:=rtsp://mediamtx:8554/sample}"; \
	export MESOPIC_ADMIN_PASSWORD MESOPIC_RTSP_URL; \
	MESOPIC_SEED=gate.yaml docker compose --profile gate build; \
	MESOPIC_SEED=gate.yaml docker compose --profile gate run --rm seed; \
	printf '\n  dashboard   http://localhost:8080\n'; \
	printf '  frigate     http://localhost:5000\n'; \
	printf '  password    %s\n\n' "$$MESOPIC_ADMIN_PASSWORD"; \
	printf '  Two cameras watch the SAME clip: `rtsp-door` decodes it here, `frigate-door`\n'; \
	printf '  consumes what Frigate publishes about it. Their geometry is identical, so\n'; \
	printf '  their counts should be too — a disagreement is the coordinate-space\n'; \
	printf '  assumption ADR-0022 recorded as unverified.\n\n'; \
	printf '  broker:  docker compose --profile gate exec mosquitto mosquitto_sub -v -t "mesopic/#" -t "homeassistant/#"\n'; \
	printf '  stop:    docker compose --profile gate down -v\n\n'; \
	MESOPIC_SEED=gate.yaml docker compose --profile gate up

gif:  ## Bring up exactly what the launch demo GIF is recorded against (P5.5)
	@command -v openssl >/dev/null || { echo "error: openssl not found; export MESOPIC_ADMIN_PASSWORD yourself" >&2; exit 1; }
	@test -f examples/clips/sample.mp4 || { echo "error: examples/clips/sample.mp4 is missing — the GIF needs footage with people in it" >&2; exit 1; }
	@set -eu; \
	: "$${MESOPIC_ADMIN_PASSWORD:=$$(openssl rand -hex 16)}"; \
	: "$${MESOPIC_RTSP_URL:=rtsp://mediamtx:8554/sample}"; \
	export MESOPIC_ADMIN_PASSWORD MESOPIC_RTSP_URL; \
	MESOPIC_SEED=gif.yaml docker compose --profile gif build; \
	MESOPIC_SEED=gif.yaml docker compose --profile gif run --rm seed; \
	printf '\n  dashboard   http://localhost:8080\n'; \
	printf '  password    %s\n\n' "$$MESOPIC_ADMIN_PASSWORD"; \
	printf '  Footfall, line-crossings, live occupancy, dwell and a zone heatmap, on a\n'; \
	printf '  real storefront feed. NO QUEUE TILE: nobody waits in this clip, so the\n'; \
	printf '  config does not claim one — see the header of docker/gif.yaml.\n\n'; \
	printf '  record:  scripts/record-demo-gif.sh\n'; \
	printf '  stop:    docker compose --profile gif down -v\n\n'; \
	MESOPIC_SEED=gif.yaml docker compose --profile gif up

compose-check:  ## Validate the compose file and its demo profile
	docker compose --profile demo config -q
	docker compose --profile gate config -q
	docker compose --profile gif config -q
	docker compose config -q

test-stream:  ## Serve a synthetic RTSP camera on :8554 (no engine, no clip needed)
	@docker compose --profile camera up -d mediamtx
	@printf '\n  rtsp://127.0.0.1:8554/synthetic   1080p25, generated live\n'
	@printf '  rtsp://127.0.0.1:8554/sample      your own clip, if examples/clips/sample.mp4 exists\n'
	@printf '\n  probe:  ffprobe -rtsp_transport tcp rtsp://127.0.0.1:8554/synthetic\n'
	@printf '  stop:   docker compose --profile camera down\n\n'

clean:  ## Remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov dist build
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
