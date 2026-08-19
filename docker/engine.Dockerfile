# syntax=docker/dockerfile:1
#
# The Muster engine image (implements P0.6, published by CI at P5.1).
#
# Two properties this file exists to guarantee:
#   1. NO MODEL WEIGHTS ARE BAKED IN. They are fetched and exported at runtime into
#      `/data/models` on the volume — which is what keeps the model's licence a separate
#      artefact from this MIT image (ADR-0008, ADR-0013). A test asserts both halves:
#      that nothing is baked in, and that what is fetched lands somewhere it survives.
#   2. The dependency graph is exactly the one CI resolved. `uv sync --frozen` fails
#      rather than re-resolving, because a build we cannot reproduce is a build we
#      cannot debug on a box we cannot SSH into (tech-stack.md §7).

ARG PYTHON_VERSION=3.12

# --- builder ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Build the venv at the SAME path it will live at in the runtime stage. uv writes
# absolute shebangs into the console scripts, so a venv built at /build and copied to
# /app produces `exec muster: No such file or directory` — the script is there, but its
# interpreter path is not.
WORKDIR /app

# Dependencies first, in their own layer: they change far less often than our source.
COPY pyproject.toml uv.lock ./
COPY muster-engine/pyproject.toml muster-engine/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --package muster

COPY muster-engine/ muster-engine/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package muster

# --- runtime ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# ffmpeg's shared libraries only — PyAV binds them; we never shell out to the binary.
# tini reaps the camera worker processes so a restart never leaves zombies behind.
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
      libglib2.0-0 tini \
 && rm -rf /var/lib/apt/lists/*

# Unprivileged: the engine reads a stream and writes one SQLite file. It needs nothing else.
RUN useradd --create-home --uid 10001 muster \
 && mkdir -p /data \
 && chown muster:muster /data

# The model cache belongs on the volume, not in the container's writable layer. Its
# default is `~/.cache/muster/models`, which is right for a developer and wrong here: the
# home directory dies with the container, so every recreate re-downloads and re-exports
# the graph. Measured before this was set — `docker compose down && up` against the same
# volume kept the store and fetched the model again. On an airgapped box that is not a
# cost but a failure, because the second start has nowhere to fetch from.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUSTER_DATA_DIR=/data \
    MUSTER_MODEL_CACHE=/data/models

COPY --from=builder --chown=muster:muster /app/.venv /app/.venv
COPY --from=builder --chown=muster:muster /app/muster-engine /app/muster-engine

WORKDIR /app
USER muster

# The metric store, muster.yaml, and the model cache. The only writable path.
VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=4)"

ENTRYPOINT ["/usr/bin/tini", "--", "muster"]
CMD ["run", "--config", "/data/muster.yaml"]
