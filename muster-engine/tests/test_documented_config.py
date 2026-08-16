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

What is left alongside that is only what validation cannot see: that the two documents
describe the same schema as each other, and that neither of them inlines a secret.
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


@pytest.fixture(scope="module")
def example() -> dict[str, Any]:
    parsed = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


@pytest.mark.parametrize("document", ["readme", "example"])
def test_every_documented_config_validates(
    document: str, documented: dict[str, Any], example: dict[str, Any], tmp_path: Path
) -> None:
    """The documents we publish go through the loader a user's file goes through.

    This is the check the earlier name comparisons were standing in for: `exports` for
    `exporters`, a coordinate outside `[0, 1]`, `dwell_time` for `dwell_seconds` — each
    is now rejected by the schema itself rather than by a test re-implementing it.
    """
    config = documented if document == "readme" else example
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


@pytest.mark.parametrize("document", ["readme", "example"])
def test_every_documented_config_shows_some_geometry(
    document: str, documented: dict[str, Any], example: dict[str, Any]
) -> None:
    """Validation accepts a site with no lines or zones; a *worked example* must not be one.

    The engine counts nothing without geometry, so a document that shows none is not
    showing the reader how to use it — which is a documentation bug the schema cannot
    have an opinion about.
    """
    config = documented if document == "readme" else example

    assert config.get("lines"), f"{document} defines no lines"
    assert config.get("zones"), f"{document} defines no zones"


@pytest.mark.privacy
@pytest.mark.parametrize("document", ["readme", "example"])
def test_no_document_inlines_a_secret(
    document: str, documented: dict[str, Any], example: dict[str, Any]
) -> None:
    """Secrets are referenced by env-var name, never written into the config file.

    The RTSP URL is the deliberate exception: it is the one credential that has to be in
    the file, and the documented form uses the `user:pass` placeholder.
    """
    config = documented if document == "readme" else example
    sync = config.get("cloud_sync", {})

    assert "site_token" not in sync, "the site token must be a `site_token_env` reference"
    for exporter in config.get("exporters", {}).values():
        if isinstance(exporter, dict):
            assert "secret" not in exporter, "webhook secrets are `secret_env` references"

    for camera in config["cameras"]:
        url = camera.get("source", {}).get("url", "")
        if url.startswith("rtsp://") and "@" in url:
            assert "user:pass@" in url, f"{document} embeds real-looking camera credentials"
