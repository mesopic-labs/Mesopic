"""The `/config` view replaces the whole document, and validates before it lands.

P3.3's `write_geometry` edits two keys and preserves everything around them, because the
browser was drawing polygons rather than editing text. Here the operator *is* editing the
text, so there is nothing to preserve around: what they typed is the document. That makes
the round trip unnecessary and the validation more important — the editor can break any
key in the file, not just the two the canvas owns.

The guarantee is the same one P3.3 established and is the reason both routes share
`_reject_an_invalid_result` and `_replace_atomically` rather than repeating them: a config
the engine wrote never fails at the next start, and a refusal never quotes what it refused.

Implements P3.4.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from muster.config import load_config
from muster.config.changes import LIVE_SECTIONS, sections_needing_restart
from muster.config.schema import MusterConfig, ZoneConfig
from muster.config.writer import read_document, write_document
from muster.errors import ConfigError
from muster.types import CameraId, MetricName, ZoneId, ZoneRole

# `rtsp://user:pass@` is the one credential form allowed in this repository — the
# documented placeholder, exempted by name in `.gitleaks.toml`. Any other spelling is a
# leak as far as the scanner is concerned, and it is right to insist. The host is in RFC
# 5737's documentation range, so this URL cannot address a real camera either.
CREDENTIAL = "user:pass"
CAMERA_HOST = "192.0.2.10"

DOCUMENT = """\
# The site's config. Hand-annotated, and it stays that way.
site:
  site_id: "acme-camden"
  timezone: "Europe/London"

budget:
  cpu_budget: 0.75                  # the share of the box we may take
  fps_min: 1.0
  fps_max: 5.0

cameras:
  - camera_id: "front-door"
    name: "Front door"
    source:
      kind: rtsp
      url_env: MUSTER_FRONT_DOOR_RTSP   # never inline
      transport: tcp
    reference_resolution: [1920, 1080]
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "muster.yaml"
    path.write_text(DOCUMENT, encoding="utf-8")
    return path


def test_the_document_lands_byte_for_byte(config_path: Path) -> None:
    """No re-render, so no diff nobody asked for.

    `write_geometry` round-trips through ruamel because it edits a document it did not
    write. This route writes the text the operator submitted, which means their layout,
    their comments and their key order survive by construction rather than by effort.
    """
    edited = DOCUMENT.replace("fps_max: 5.0", "fps_max: 4.0")

    write_document(config_path, edited)

    assert config_path.read_text(encoding="utf-8") == edited


def test_a_document_that_would_not_validate_never_lands(config_path: Path) -> None:
    before = config_path.read_text(encoding="utf-8")

    with pytest.raises(ConfigError):
        write_document(config_path, DOCUMENT.replace("fps_max: 5.0", "fps_max: 0.5"))

    assert config_path.read_text(encoding="utf-8") == before


def test_text_that_is_not_yaml_never_lands(config_path: Path) -> None:
    before = config_path.read_text(encoding="utf-8")

    with pytest.raises(ConfigError):
        write_document(config_path, "site: [unclosed\n")

    assert config_path.read_text(encoding="utf-8") == before


def test_a_refused_document_leaves_no_partial_file_behind(config_path: Path) -> None:
    """A half-written config is a box that will not start, found at the next restart."""
    with pytest.raises(ConfigError):
        write_document(config_path, DOCUMENT.replace("fps_max: 5.0", "fps_max: 0.5"))

    assert [path.name for path in config_path.parent.iterdir()] == ["muster.yaml"]


def test_the_refusal_names_the_key_path(config_path: Path) -> None:
    """The operator has to be able to fix it, which means being told where it is wrong."""
    with pytest.raises(ConfigError) as caught:
        write_document(config_path, DOCUMENT.replace("fps_max: 5.0", "fps_max: 0.5"))

    assert "budget" in str(caught.value)


def test_the_refusal_never_quotes_the_submitted_document(config_path: Path) -> None:
    """This message is rendered to a browser, which `describe_validation_error` allows.

    It allows it because `include_input=False` is what P2.1 built it for. The document an
    operator submits may hold an inline RTSP URL — legitimate, since it is also the
    address — so a message that quoted the offending input would put the camera's password
    on the page and into whatever logs the exception. The failure below is in `budget`,
    two keys away from the credential, which is exactly the case a naive formatter leaks:
    pydantic reports the *model's* input, not the field's.

    The assertions are on the host and the path rather than on the password, because the
    password here is the literal string the repository allows and would be a weak needle.
    """
    inlined = DOCUMENT.replace(
        "      url_env: MUSTER_FRONT_DOOR_RTSP   # never inline",
        f"      url: rtsp://{CREDENTIAL}@{CAMERA_HOST}:554/Streaming/Channels/102",
    ).replace("fps_max: 5.0", "fps_max: 0.5")

    with pytest.raises(ConfigError) as caught:
        write_document(config_path, inlined)

    message = str(caught.value)
    assert CAMERA_HOST not in message
    assert "Streaming/Channels" not in message


def test_what_landed_is_what_a_restart_would_read(config_path: Path) -> None:
    write_document(config_path, DOCUMENT.replace("fps_max: 5.0", "fps_max: 4.0"))

    assert load_config(config_path).budget.fps_max == 4.0


def test_read_document_returns_the_text_on_disk(config_path: Path) -> None:
    assert read_document(config_path) == DOCUMENT


def test_reading_a_config_that_is_not_there_is_a_config_error(tmp_path: Path) -> None:
    """The one place an OSError becomes a ConfigError, so no route has to know about both."""
    with pytest.raises(ConfigError):
        read_document(tmp_path / "absent.yaml")


# --- What a save cannot swap (P3.4) -----------------------------------------

CONFIG = MusterConfig.model_validate(yaml.safe_load(DOCUMENT))


def test_a_geometry_change_needs_no_restart() -> None:
    """Zones and lines are what `Supervisor.reload` recompiles and pushes."""
    moved = CONFIG.model_copy(
        update={
            "zones": [
                ZoneConfig(
                    zone_id=ZoneId("shop-floor"),
                    camera_id=CameraId("front-door"),
                    role=ZoneRole.AREA,
                    polygon=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)],
                    metrics=[MetricName("occupancy")],
                )
            ]
        }
    )

    assert sections_needing_restart(CONFIG, moved) == ()


def test_a_budget_change_needs_no_restart() -> None:
    """The half `Retarget` carries down the control channel."""
    slower = CONFIG.model_copy(update={"budget": CONFIG.budget.model_copy(update={"fps_max": 2.0})})

    assert sections_needing_restart(CONFIG, slower) == ()


def test_a_camera_change_needs_a_restart() -> None:
    """A source is opened once, when the worker spawns. Nothing can swap it under a
    running RTSP session, and saying the save took effect would be a lie the operator
    only discovers when the numbers do not move."""
    renamed = CONFIG.model_copy(
        update={"cameras": [CONFIG.cameras[0].model_copy(update={"name": "Side door"})]}
    )

    assert sections_needing_restart(CONFIG, renamed) == ("cameras",)


def test_every_changed_section_is_named_not_just_the_first() -> None:
    """An operator who has to restart deserves the whole reason, once."""
    changed = CONFIG.model_copy(
        update={
            "site": CONFIG.site.model_copy(update={"timezone": "UTC"}),
            "thresholds": CONFIG.thresholds.model_copy(update={"dwell_min_s": 30.0}),
        }
    )

    assert sections_needing_restart(CONFIG, changed) == ("site", "thresholds")


def test_an_unchanged_config_needs_nothing() -> None:
    assert sections_needing_restart(CONFIG, CONFIG) == ()


def test_only_the_three_sections_the_reload_carries_are_live() -> None:
    """The safe default, and the reason this is a deny-list of three rather than an
    allow-list of nine: `sections_needing_restart` walks `MusterConfig`'s fields and
    excludes these, so a section the schema grows later is reported until someone teaches
    the reload to carry it, rather than silently claiming to be live.
    """
    assert set(LIVE_SECTIONS) == {"budget", "lines", "zones"}
    assert set(LIVE_SECTIONS) <= set(MusterConfig.model_fields), "a typo here is a silent lie"
