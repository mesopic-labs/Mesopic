"""What the published image must promise, asserted against the Dockerfile itself.

The image is the artefact a stranger runs (P5.1), and the properties below are ones no
Python test can reach from inside the process: they are decided by the build, hold for the
lifetime of a container, and fail silently — the engine runs perfectly either way, and the
cost only shows up on the second `docker run`.

Read rather than built, deliberately. `make check` runs everywhere, including where no
daemon is available; the CI image job already does the things that need one (no weights
baked in, the entrypoint resolves).

Implements P5.1.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "engine.Dockerfile"

DATA_DIR = "/data"


def _runtime_env(dockerfile: str) -> dict[str, str]:
    """Every `ENV k=v` pair in the runtime stage, which is the one that ships."""
    runtime = dockerfile.split("AS runtime", 1)[-1]
    return dict(re.findall(r"^\s*(?:ENV\s+)?([A-Z_][A-Z0-9_]*)=(\S+)", runtime, re.M))


def test_the_model_cache_lives_on_the_data_volume() -> None:
    """Otherwise the model is re-fetched on every container recreate — forever.

    `DEFAULT_MODEL_CACHE` is `~/.cache/muster/models`, which is the right default for a
    developer and the wrong one inside a container: the home directory is in the writable
    layer, so it dies with the container while `/data` survives. Measured on a real image
    before this was set — a `docker compose down && up` against the same volume kept the
    store and downloaded the model again.

    The engine works either way, which is what makes it worth a test. What it costs is a
    download on every restart, and on the airgapped installs this product claims to
    support, that is not a cost but a failure: the box runs once and never again.
    """
    env = _runtime_env(DOCKERFILE.read_text(encoding="utf-8"))

    cache = env.get("MUSTER_MODEL_CACHE")
    assert cache is not None, "the image must not leave the model cache in the container layer"
    assert cache.startswith(DATA_DIR), f"{cache} is not on the volume the image declares"


def test_the_data_directory_the_image_declares_is_the_one_it_mounts() -> None:
    """Two names for one path is how the cache above ends up outside the volume."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    env = _runtime_env(dockerfile)

    assert env.get("MUSTER_DATA_DIR") == DATA_DIR
    assert f'VOLUME ["{DATA_DIR}"]' in dockerfile
