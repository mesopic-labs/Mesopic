"""Failing tests for `muster.yaml` validation — red-first for P2.1.

`MusterConfig` has no fields yet and `load_config` raises `NotImplementedError`
unconditionally (`muster.config.schema`, `muster.config.loader`). These tests describe
the validated shape P2.1 must produce: they are red now and must go green when P2.1
lands, not before.

Fixtures build on the worked example (`examples/muster.yaml`) rather than inventing a
second schema by hand, so this file cannot drift from `test_documented_config.py`'s
notion of "the real config" — the same reason that file checks names instead of
re-implementing validation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from muster.config.loader import load_config
from muster.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"


@pytest.fixture
def valid_config() -> dict[str, Any]:
    """A deep copy of the worked example, safe for each test to mutate."""
    parsed = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return copy.deepcopy(parsed)


def _write(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "muster.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_the_worked_example_loads_cleanly(valid_config: dict[str, Any], tmp_path: Path) -> None:
    """The example every user copies must actually validate."""
    path = _write(tmp_path, valid_config)

    config = load_config(path)

    # P2.1 hasn't added MusterConfig's `site` sub-model yet — this is the red-first
    # assertion for the shape it must have (engine-architecture.md §13.1).
    assert config.site.site_id == "acme-camden"  # type: ignore[attr-defined]


def test_dangling_camera_id_on_a_line_is_rejected(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    """A line pointing at a camera nobody defined must fail loud, not silently mis-count."""
    valid_config["lines"][0]["camera_id"] = "no-such-camera"
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError, match="no-such-camera"):
        load_config(path)


def test_dangling_camera_id_on_a_zone_is_rejected(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    valid_config["zones"][0]["camera_id"] = "no-such-camera"
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError, match="no-such-camera"):
        load_config(path)


def test_inline_webhook_secret_is_rejected(valid_config: dict[str, Any], tmp_path: Path) -> None:
    """`secret_env` is the only accepted form; a literal `secret` key is a startup error.

    The camera above still needs a source, so this fixture keeps the documented
    `rtsp://user:pass@` placeholder — the one spelling `.gitleaks.toml` allowlists.
    """
    valid_config["exporters"]["webhook"]["secret"] = "not-an-env-reference"  # noqa: S105 - fixture proving this exact shape is rejected, not a real credential
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError):
        load_config(path)


def test_inline_site_token_is_rejected(valid_config: dict[str, Any], tmp_path: Path) -> None:
    """Same rule for the cloud-sync token: `site_token_env`, never `site_token`."""
    valid_config["cloud_sync"]["site_token"] = "not-an-env-reference"  # noqa: S105 - fixture proving this exact shape is rejected, not a real credential
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize("bad_point", [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)])
def test_out_of_range_line_geometry_is_rejected(
    valid_config: dict[str, Any], tmp_path: Path, bad_point: tuple[float, float]
) -> None:
    valid_config["lines"][0]["a"] = list(bad_point)
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize("bad_point", [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)])
def test_out_of_range_zone_geometry_is_rejected(
    valid_config: dict[str, Any], tmp_path: Path, bad_point: tuple[float, float]
) -> None:
    valid_config["zones"][0]["polygon"][0] = list(bad_point)
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_required_key_names_it(valid_config: dict[str, Any], tmp_path: Path) -> None:
    """The error must name the key, not just say "invalid config"."""
    del valid_config["site"]["site_id"]
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError, match="site_id"):
        load_config(path)


@pytest.mark.privacy
def test_error_never_echoes_the_rtsp_url(valid_config: dict[str, Any], tmp_path: Path) -> None:
    """A validation failure elsewhere must not leak the camera's credentials into the error.

    The RTSP URL always carries a credential; an error that echoes the failing document
    verbatim, rather than naming just the field that's wrong, would leak it into logs.
    """
    valid_config["lines"][0]["camera_id"] = "no-such-camera"
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError) as exc_info:
        load_config(path)

    assert "user:pass@" not in str(exc_info.value)
