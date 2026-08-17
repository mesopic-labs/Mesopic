"""What `muster.yaml` validation must reject, and what it must accept.

Written red-first as MK.3, against a `MusterConfig` that had no fields and a
`load_config` that raised `NotImplementedError`; P2.1 turned them green and removed the
`xfail(strict=True)` markers that kept `main` honest in between.

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
from muster.config.schema import RtspSource
from muster.errors import ConfigError

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

    assert config.site.site_id == "acme-camden"


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


def test_camera_url_may_be_an_env_reference(valid_config: dict[str, Any], tmp_path: Path) -> None:
    """The RTSP URL carries the camera's credentials, so it gets the `*_env` form too.

    The worked example inlines it — with a `user:pass` placeholder — because a config
    has to be readable to be a worked example. A real deployment should not have to
    write a live credential into a file it will `git add`.
    """
    source = _by_id(valid_config["cameras"], "camera_id", "front-door")["source"]
    del source["url"]
    source["url_env"] = "MUSTER_FRONT_DOOR_RTSP"
    path = _write(tmp_path, valid_config)

    config = load_config(path)

    source = next(c for c in config.cameras if c.camera_id == "front-door").source
    assert isinstance(source, RtspSource)
    assert source.url_env == "MUSTER_FRONT_DOOR_RTSP"
    assert source.url is None


def test_camera_with_neither_url_nor_url_env_is_rejected(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    source = _by_id(valid_config["cameras"], "camera_id", "front-door")["source"]
    del source["url"]
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError, match="url"):
        load_config(path)


def test_camera_with_both_url_and_url_env_is_rejected(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    """Two sources of truth for one credential is a silent "which one won?" bug."""
    source = _by_id(valid_config["cameras"], "camera_id", "front-door")["source"]
    source["url_env"] = "MUSTER_FRONT_DOOR_RTSP"
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError, match="url"):
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


@pytest.mark.privacy
def test_no_validation_error_is_chained_into_the_traceback(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    """No pydantic rendering of the input may reach the traceback at all.

    The test above searches for the credential itself, and that is not sufficient here
    for the same reason a taint pattern does not survive JPEG encoding in
    `test_frame_lifetime.py`: **pydantic elides the middle of a long `input_value=`
    repr**, which is exactly where `user:pass@` sits in the worked example's URL. So a
    loader that chains `from error` passes the taint check by luck, and stops passing
    the day someone shortens a URL.

    A failure on a key *next to* `source` is the shape that carries the whole camera
    block — URL included — into `input_value=`. Asserting that no such rendering exists
    is the structural version of the rule, and it does not depend on where a repr
    happens to be truncated.
    """
    camera = _by_id(valid_config["cameras"], "camera_id", "front-door")
    del camera["reference_resolution"]
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError) as exc_info:
        load_config(path)

    rendered = "".join(traceback.format_exception(exc_info.value))
    assert "input_value=" not in rendered, "the loader chained pydantic's input rendering"
    assert "user:pass@" not in rendered


# --- Storage retention (P2.8) -----------------------------------------------


def test_event_retention_defaults_when_the_storage_section_is_absent(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    """Every config written before this key existed must keep loading, unchanged."""
    valid_config.pop("storage", None)
    path = _write(tmp_path, valid_config)

    config = load_config(path)

    assert config.storage.event_retention_hours == 72


def test_event_retention_hours_is_read_from_the_storage_section(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    valid_config["storage"] = {"event_retention_hours": 6}
    path = _write(tmp_path, valid_config)

    config = load_config(path)

    assert config.storage.event_retention_hours == 6


@pytest.mark.parametrize("hours", [0, -1])
def test_a_non_positive_retention_window_is_rejected(
    valid_config: dict[str, Any], tmp_path: Path, hours: int
) -> None:
    """Zero would trim every event the moment it is written, which reads as "off"."""
    valid_config["storage"] = {"event_retention_hours": hours}
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError, match="event_retention_hours"):
        load_config(path)


def test_a_retention_window_beyond_a_year_is_rejected(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    """`events` is the short-retention log; an unbounded window is a full disk."""
    valid_config["storage"] = {"event_retention_hours": 8761}
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError, match="event_retention_hours"):
        load_config(path)


# --- The local API (P3.1) ---------------------------------------------------


def test_the_api_binds_to_loopback_when_the_section_is_absent(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    """The dashboard is unauthenticated, so off-box reachability is an opt-in act.

    A default of `0.0.0.0` would put camera topology and occupancy history in front of
    the whole LAN on first run. The container case is handled in the compose file, not
    by weakening this (P3.6).
    """
    valid_config.pop("api", None)
    path = _write(tmp_path, valid_config)

    config = load_config(path)

    assert config.api.host == "127.0.0.1"
    assert config.api.port == 8080
    assert config.api.enabled is True


def test_the_api_host_and_port_are_read_from_the_api_section(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    valid_config["api"] = {"host": "0.0.0.0", "port": 9000}  # noqa: S104 - the opt-in this key exists for
    path = _write(tmp_path, valid_config)

    config = load_config(path)

    assert config.api.host == "0.0.0.0"  # noqa: S104 - see above
    assert config.api.port == 9000


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_a_port_outside_the_tcp_range_is_rejected(
    valid_config: dict[str, Any], tmp_path: Path, port: int
) -> None:
    valid_config["api"] = {"port": port}
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError, match=r"api\.port"):
        load_config(path)


def test_an_empty_api_host_is_rejected(valid_config: dict[str, Any], tmp_path: Path) -> None:
    """An empty string binds every interface in some stacks — the opposite of the default."""
    valid_config["api"] = {"host": ""}
    path = _write(tmp_path, valid_config)

    with pytest.raises(ConfigError, match=r"api\.host"):
        load_config(path)
