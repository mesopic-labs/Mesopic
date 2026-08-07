"""The configuration we document must be configuration that actually works.

The README's example is the first thing a new user copies, and `examples/muster.yaml` is
the second. Both are prose from the code's point of view, so they drift silently: a key
gets renamed in the schema, and the documented example keeps saying the old name until
someone files an issue saying "your quickstart doesn't work".

These tests make that drift a build failure instead. They deliberately check *names* —
top-level sections, identity keys, and the metric vocabulary — rather than trying to
re-implement validation, which is `muster.config`'s job (P2.1). When the loader exists,
the right move is to replace the key comparisons here with a real `load_config()` call on
both documents.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from muster.types import MetricName

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


def test_readme_example_is_valid_yaml(documented: dict[str, Any]) -> None:
    """It parses at all — and parses with `safe_load`, which is what the loader uses."""
    assert documented


@pytest.mark.parametrize("section", ["site", "cameras", "lines", "zones", "exporters"])
def test_readme_uses_the_real_top_level_sections(documented: dict[str, Any], section: str) -> None:
    """`exports` vs `exporters` and `cloud` vs `cloud_sync` are exactly the silent drift."""
    assert section in documented, f"README example is missing the `{section}` section"


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
def test_documented_metric_names_exist(
    document: str, documented: dict[str, Any], example: dict[str, Any]
) -> None:
    """Every metric named in a document is one the engine actually produces.

    `dwell_time` and `queue_length` read naturally and are not real; the vocabulary is
    `dwell_seconds` and `queue_len`, and it is a shared contract with the cloud.
    """
    config = documented if document == "readme" else example
    valid = {metric.value for metric in MetricName}

    named: set[str] = set()
    for section in ("lines", "zones"):
        for entry in config.get(section, []):
            named.update(entry.get("metrics", []))

    assert named, f"{document} names no metrics at all"
    assert named <= valid, f"{document} names metrics that do not exist: {sorted(named - valid)}"


@pytest.mark.parametrize("document", ["readme", "example"])
def test_documented_geometry_is_normalized(
    document: str, documented: dict[str, Any], example: dict[str, Any]
) -> None:
    """Coordinates are `[0, 1]` so geometry survives a resolution change (engine arch §3)."""
    config = documented if document == "readme" else example

    points: list[list[float]] = []
    for line in config.get("lines", []):
        points += [line["a"], line["b"]]
    for zone in config.get("zones", []):
        points += zone["polygon"]

    assert points, f"{document} defines no geometry"
    for x, y in points:
        assert 0.0 <= x <= 1.0, f"{document} has an out-of-range x: {x}"
        assert 0.0 <= y <= 1.0, f"{document} has an out-of-range y: {y}"


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
