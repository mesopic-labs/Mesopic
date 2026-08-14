"""Failing tests for `muster.yaml` validation — red-first for P2.1.

`MusterConfig` has no fields yet and `load_config` raises `NotImplementedError`
unconditionally (`muster.config.schema`, `muster.config.loader`). These tests describe
the validated shape P2.1 must produce: they are red now and must go green when P2.1
lands, not before.

Every test in this module is `xfail(strict=True)`: `make test` / CI run the whole fast
suite unfiltered, so an un-marked red file would leave `main` failing from the moment
this merges until P2.1 lands. `strict=True` keeps the red-first intent honest the other
way too — the first test P2.1 makes pass turns into a loud XPASS failure, which is
exactly the signal to delete that test's marker rather than leaving it stale.

Fixtures build on the worked example (`examples/muster.yaml`) rather than inventing a
second schema by hand, so this file cannot drift from `test_documented_config.py`'s
notion of "the real config" — the same reason that file checks names instead of
re-implementing validation. Entries are selected by id, not list position, so
reordering the worked example can't silently point a test at the wrong camera/line/zone.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import pytest
import yaml

from muster.config.loader import load_config
from muster.errors import ConfigError

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="P2.1 not implemented yet (MK.3 red-first tests) — remove this marker, "
    "test by test, as P2.1 lands.",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"


@pytest.fixture
def valid_config() -> dict[str, Any]:
    """The worked example, parsed fresh for each test — no shared mutable state."""
    parsed = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def _by_id(items: list[dict[str, Any]], id_key: str, value: str) -> dict[str, Any]:
    """Select a camera/line/zone by its id rather than its position in the list."""
    for item in items:
        if item[id_key] == value:
            return item
    msg = f"no entry with {id_key}={value!r} in the worked example"
    raise AssertionError(msg)


def _write(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "muster.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_the_worked_example_loads_cleanly(valid_config: dict[str, Any], tmp_path: Path) -> None:
    """The example every user copies must actually validate."""
    path = _write(tmp_path, valid_config)

    config = load_config(path)

    # P2.1 hasn't added MusterConfig's `site` sub-model yet, so this line is also
    # a mypy failure today. Once P2.1 adds `site`, `warn_unused_ignores = true`
    # turns *that* into a different failure ("unused ignore") — expected; delete
    # the ignore comment then, not this assertion.
    assert config.site.site_id == "acme-camden"  # type: ignore[attr-defined]


def test_dangling_camera_id_on_a_line_is_rejected(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    """A line pointing at a camera nobody defined must fail loud, not silently mis-count."""
    _by_id(valid_config["lines"], "line_id", "door-count")["camera_id"] = "no-such-camera"
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError, match="no-such-camera"):
        load_config(path)


def test_dangling_camera_id_on_a_zone_is_rejected(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    _by_id(valid_config["zones"], "zone_id", "shop-floor")["camera_id"] = "no-such-camera"
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError, match="no-such-camera"):
        load_config(path)


def test_inline_webhook_secret_is_rejected(valid_config: dict[str, Any], tmp_path: Path) -> None:
    """A bare `secret` with no `secret_env` at all is the actual violation of the rule.

    Leaving `secret_env` in place too would only prove the *pair* is rejected — a
    loader that treats them as mutually exclusive would pass while still accepting a
    config with `secret` and no `secret_env`, the real hard-invariant #6 violation.
    """
    webhook = valid_config["exporters"]["webhook"]
    del webhook["secret_env"]
    webhook["secret"] = "not-an-env-reference"  # noqa: S105 - fixture proving this exact shape is rejected, not a real credential
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError):
        load_config(path)


def test_inline_site_token_is_rejected(valid_config: dict[str, Any], tmp_path: Path) -> None:
    """Same rule for the cloud-sync token: `site_token_env`, never `site_token`.

    `enabled: true` so this exercises real validation rather than whatever a disabled
    `cloud_sync` block happens to skip.
    """
    sync = valid_config["cloud_sync"]
    sync["enabled"] = True
    del sync["site_token_env"]
    sync["site_token"] = "not-an-env-reference"  # noqa: S105 - fixture proving this exact shape is rejected, not a real credential
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize("bad_point", [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)])
@pytest.mark.parametrize("endpoint", ["a", "b"])
def test_out_of_range_line_geometry_is_rejected(
    valid_config: dict[str, Any],
    tmp_path: Path,
    endpoint: str,
    bad_point: tuple[float, float],
) -> None:
    """Both endpoints are checked — an implementation that only range-checks `a`
    and forgets `b` must not pass."""
    _by_id(valid_config["lines"], "line_id", "door-count")[endpoint] = list(bad_point)
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize("bad_point", [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)])
@pytest.mark.parametrize("vertex_index", [0, 2])
def test_out_of_range_zone_geometry_is_rejected(
    valid_config: dict[str, Any],
    tmp_path: Path,
    vertex_index: int,
    bad_point: tuple[float, float],
) -> None:
    """Covers a non-zero vertex too — an implementation that only checks `polygon[0]`
    must not pass."""
    _by_id(valid_config["zones"], "zone_id", "shop-floor")["polygon"][vertex_index] = list(
        bad_point
    )
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_required_key_names_it(valid_config: dict[str, Any], tmp_path: Path) -> None:
    """The error must name the key, not just say "invalid config"."""
    del valid_config["site"]["site_id"]
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError, match="site_id"):
        load_config(path)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    """A missing path is a `ConfigError`, not a bare `FileNotFoundError` at the CLI."""
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does-not-exist.yaml")


def test_unparseable_yaml_raises_config_error(tmp_path: Path) -> None:
    """Malformed YAML is a `ConfigError`, not a bare `yaml.YAMLError` at the CLI."""
    path = tmp_path / "muster.yaml"
    path.write_text("site: [this is not: valid: yaml", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.privacy
def test_error_never_echoes_the_rtsp_url(valid_config: dict[str, Any], tmp_path: Path) -> None:
    """A validation failure inside the camera block must not leak its credentials.

    The failure is induced in the camera's own `source` block, not an unrelated field,
    so the error's context has every opportunity to include the URL if the
    implementation is careless. Asserted over the full formatted exception chain, not
    just `str(exc)`: `raise ConfigError(msg) from validation_error` is the natural
    implementation, and pydantic's `ValidationError` renders `input_value=` — which any
    traceback or `logger.exception(...)` call prints even when the top-level message is
    clean.
    """
    camera = _by_id(valid_config["cameras"], "camera_id", "front-door")
    camera["source"]["kind"] = "bogus"
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError) as exc_info:
        load_config(path)

    rendered = "".join(traceback.format_exception(exc_info.value))
    assert "user:pass@" not in rendered
