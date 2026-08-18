"""The configuration we document must be configuration that actually works.

The README's example is the first thing a new user copies, and `examples/muster.yaml` is
the second. Both are prose from the code's point of view, so they drift silently: a key
gets renamed in the schema, and the documented example keeps saying the old name until
someone files an issue saying "your quickstart doesn't work".

These tests make that drift a build failure instead. Until P2.1 there was no loader to
check them against, so they compared *names* — sections, identity keys, the metric
vocabulary — and said in this docstring that the right move, once `load_config` existed,
was to run both documents through it for real. That is what they now do: every document
we publish is validated by the same code path a user's file takes, so a renamed key, an
out-of-range coordinate or an invented metric name fails here first.

What is left alongside that is only what validation cannot see: that the README and the
worked example describe the same schema as each other, and that none of these documents
inlines a secret.

`fixtures/muster.yaml` is checked here too, though nobody reads it as documentation. It
is the geometry the ground-truth clips are labelled against, so drift there does not
mislead a reader — it silently invalidates every label made against it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from muster.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"
DEMO_CONFIG = REPO_ROOT / "docker" / "demo.yaml"
"""The config `make demo` seeds into the container's volume (P3.6).

Held to the same bar as the documents a reader copies, for a sharper reason: nobody reads
this one, so drift in it is invisible until the one-command bring-up dies on a fresh
machine — which is exactly the machine nobody is watching.
"""

FIXTURE_CONFIG = REPO_ROOT / "fixtures" / "muster.yaml"
"""The geometry the ground-truth clips are labelled against.

Not documentation, but it drifts the same way and the cost of drift is worse: a renamed
key here does not confuse a reader, it silently invalidates every label made against the
line this file defines.
"""


def _readme_config_block() -> dict[str, Any]:
    """The `muster.yaml` example embedded in the README."""
    match = re.search(r"```yaml\n(# muster\.yaml.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert match, "the README no longer contains a muster.yaml example block"
    parsed = yaml.safe_load(match.group(1))
    assert isinstance(parsed, dict)
    return parsed


@pytest.fixture(scope="module")
def documented() -> dict[str, Any]:
    return _readme_config_block()


def _yaml_document(path: Path) -> dict[str, Any]:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


@pytest.fixture(scope="module")
def example() -> dict[str, Any]:
    return _yaml_document(EXAMPLE_CONFIG)


@pytest.fixture(scope="module")
def documents(documented: dict[str, Any], example: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every config this repository commits, by the name the tests parametrize over."""
    return {
        "readme": documented,
        "example": example,
        "fixtures": _yaml_document(FIXTURE_CONFIG),
        "demo": _yaml_document(DEMO_CONFIG),
    }


@pytest.mark.parametrize("document", ["readme", "example", "fixtures", "demo"])
def test_every_documented_config_validates(
    document: str, documents: dict[str, dict[str, Any]], tmp_path: Path
) -> None:
    """The documents we publish go through the loader a user's file goes through.

    This is the check the earlier name comparisons were standing in for: `exports` for
    `exporters`, a coordinate outside `[0, 1]`, `dwell_time` for `dwell_seconds` — each
    is now rejected by the schema itself rather than by a test re-implementing it.
    """
    config = documents[document]
    path = tmp_path / "muster.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.cameras, f"{document} describes a site with no cameras"


def test_readme_and_example_agree_on_structure(
    documented: dict[str, Any], example: dict[str, Any]
) -> None:
    """Two documents describing one schema should not describe two schemas."""
    assert set(documented) <= set(example), (
        f"README uses sections the worked example does not: {set(documented) - set(example)}"
    )
    assert set(documented["cameras"][0]) <= set(example["cameras"][0])
    assert set(documented["lines"][0]) <= set(example["lines"][0])
    assert set(documented["zones"][0]) <= set(example["zones"][0])


@pytest.mark.parametrize("document", ["readme", "example", "fixtures", "demo"])
def test_every_documented_config_shows_some_geometry(
    document: str, documents: dict[str, dict[str, Any]]
) -> None:
    """Validation accepts a site with no lines or zones; a *worked example* must not be one.

    The engine counts nothing without geometry, so a document that shows none is not
    showing the reader how to use it — which is a documentation bug the schema cannot
    have an opinion about. The fixture config is held to the same bar for a different
    reason: it exists only to give the labels a geometry to mean something against.
    """
    config = documents[document]

    assert config.get("lines"), f"{document} defines no lines"
    assert config.get("zones"), f"{document} defines no zones"


@pytest.mark.privacy
@pytest.mark.parametrize("document", ["readme", "example", "fixtures", "demo"])
def test_no_document_inlines_a_secret(document: str, documents: dict[str, dict[str, Any]]) -> None:
    """Secrets are referenced by env-var name, never written into the config file.

    The RTSP URL is the deliberate exception: it is the one credential that has to be in
    the file, and the documented form uses the `user:pass` placeholder.
    """
    config = documents[document]
    sync = config.get("cloud_sync", {})

    assert "site_token" not in sync, "the site token must be a `site_token_env` reference"
    for exporter in config.get("exporters", {}).values():
        if isinstance(exporter, dict):
            assert "secret" not in exporter, "webhook secrets are `secret_env` references"

    assert "password" not in config.get("api", {}), (
        "the dashboard password is a `password_env` reference"
    )

    for camera in config["cameras"]:
        url = camera.get("source", {}).get("url", "")
        if url.startswith("rtsp://") and "@" in url:
            assert "user:pass@" in url, f"{document} embeds real-looking camera credentials"


# --- The demo config the container actually boots (P3.6) ----------------------


def test_the_demo_config_binds_every_interface(documents: dict[str, dict[str, Any]]) -> None:
    """A published port cannot reach a loopback bind inside a container.

    The default is loopback for a good reason (§13.1), so the demo overriding it is a
    deliberate act that belongs in exactly one file and nowhere else. Pinned because the
    failure it prevents looks like "the dashboard is down" rather than like a bind address.
    """
    assert documents["demo"]["api"]["host"] == "0.0.0.0"  # noqa: S104 - the opt-in this key exists for


def test_the_demo_config_requires_a_password(documents: dict[str, dict[str, Any]]) -> None:
    """Binding every interface is exactly when the write surface needs its credential.

    ADR-0019 made this card responsible for the pair: the bring-up that removes the
    loopback mitigation is the bring-up that must supply the thing replacing it.
    """
    assert documents["demo"]["api"]["password_env"] == "MUSTER_ADMIN_PASSWORD"  # noqa: S105 - an env-var name


def test_the_demo_camera_takes_its_url_from_the_environment(
    documents: dict[str, dict[str, Any]],
) -> None:
    """One config serves both the sidecar and a real camera, and holds no URL either way.

    `make demo` defaults the variable to the mediamtx sidecar; anyone who exports their own
    stream gets it through the identical file. An inline URL would carry the camera's
    credentials into a committed document (hard invariant 6).
    """
    source = documents["demo"]["cameras"][0]["source"]

    assert source["url_env"] == "MUSTER_RTSP_URL"
    assert "url" not in source
