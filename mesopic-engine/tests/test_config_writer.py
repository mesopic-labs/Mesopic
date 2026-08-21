"""The geometry writer edits `mesopic.yaml` in place without destroying it.

§13.1 makes the config file authoritative, so the zone editor cannot save to the store
alone — the next `apply_config` would overwrite it from the file. It has to write the
file, which means the file an operator hand-annotated is now something the engine
rewrites. That is only acceptable if the rewrite is surgical.

Three properties define "surgical" here, and each has a test below: comments and layout
outside the geometry blocks survive untouched, a polygon stays on one readable line
rather than exploding into one point per pair of lines, and a document that would not
load is never allowed to land on disk in the first place.

Implements P3.3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mesopic.config import load_config
from mesopic.config.schema import LineConfig, ZoneConfig
from mesopic.config.writer import write_geometry
from mesopic.errors import ConfigError
from mesopic.types import CameraId, Direction, LineId, MetricName, ZoneId, ZoneRole

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
      url_env: MESOPIC_FRONT_DOOR_RTSP   # never inline
      transport: tcp
    reference_resolution: [1920, 1080]

lines:
  - line_id: "door-count"
    camera_id: "front-door"
    a: [0.10, 0.80]                 # normalized endpoints
    b: [0.90, 0.80]
    positive_dir: in
    metrics: [line_cross, footfall]

zones:
  - zone_id: "shop-floor"
    camera_id: "front-door"
    role: area
    polygon: [[0.05, 0.30], [0.95, 0.30], [0.95, 0.95], [0.05, 0.95]]
    metrics: [occupancy, dwell_seconds]
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "mesopic.yaml"
    path.write_text(DOCUMENT, encoding="utf-8")
    return path


def _zone(zone_id: str = "shop-floor", camera_id: str = "front-door") -> ZoneConfig:
    return ZoneConfig(
        zone_id=ZoneId(zone_id),
        camera_id=CameraId(camera_id),
        role=ZoneRole.AREA,
        polygon=[(0.10, 0.20), (0.80, 0.20), (0.80, 0.90)],
        metrics=[MetricName("occupancy")],
    )


def _line(line_id: str = "door-count", camera_id: str = "front-door") -> LineConfig:
    return LineConfig(
        line_id=LineId(line_id),
        camera_id=CameraId(camera_id),
        a=(0.15, 0.75),
        b=(0.85, 0.75),
        positive_dir=Direction.IN,
        metrics=[MetricName("footfall")],
    )


def test_comments_outside_the_geometry_blocks_survive_a_save(config_path: Path) -> None:
    """The operator's annotations are not ours to delete.

    `safe_dump` of the whole document would pass every other test in this file and still
    be wrong: it silently discards every comment, including the one marking the RTSP URL
    as an env reference rather than an inline secret.
    """
    write_geometry(config_path, zones=[_zone()], lines=[_line()])

    text = config_path.read_text(encoding="utf-8")
    assert "# The site's config. Hand-annotated, and it stays that way." in text
    assert "# the share of the box we may take" in text
    assert "# never inline" in text


def test_a_saved_polygon_stays_on_one_line(config_path: Path) -> None:
    """Block style for coordinates turns a four-point zone into eight lines of noise.

    The file has to stay hand-editable after the editor has touched it, which is the
    whole reason the config is the authority rather than the store.
    """
    write_geometry(config_path, zones=[_zone()], lines=[_line()])

    text = config_path.read_text(encoding="utf-8")
    assert "polygon: [[0.1, 0.2], [0.8, 0.2], [0.8, 0.9]]" in text
    assert "a: [0.15, 0.75]" in text


def test_the_written_document_reloads_to_the_geometry_that_was_saved(config_path: Path) -> None:
    write_geometry(config_path, zones=[_zone()], lines=[_line()])

    config = load_config(config_path)
    assert [zone.polygon for zone in config.zones] == [[(0.10, 0.20), (0.80, 0.20), (0.80, 0.90)]]
    assert [line.a for line in config.lines] == [(0.15, 0.75)]


def test_saving_removes_the_geometry_that_is_no_longer_there(config_path: Path) -> None:
    """A save replaces the blocks, it does not merge into them.

    Deleting a zone in the editor and finding it still counting after a reload is the
    failure this pins.
    """
    write_geometry(config_path, zones=[], lines=[_line()])

    assert load_config(config_path).zones == []
    assert "shop-floor" not in config_path.read_text(encoding="utf-8")


def test_geometry_naming_an_unknown_camera_never_reaches_the_file(config_path: Path) -> None:
    """Validate the whole candidate document, not the fragment.

    Referential integrity is a property of the document, so a zone pointing at a camera
    that does not exist can only be caught by validating the result — and it has to be
    caught before the write, not after, or the engine's own config is left broken.
    """
    before = config_path.read_text(encoding="utf-8")

    with pytest.raises(ConfigError):
        write_geometry(config_path, zones=[_zone(camera_id="does-not-exist")], lines=[_line()])

    assert config_path.read_text(encoding="utf-8") == before


def test_a_failed_save_leaves_no_partial_file_behind(config_path: Path) -> None:
    """A half-written config is worse than an unchanged one: it fails at the next start."""
    with pytest.raises(ConfigError):
        write_geometry(config_path, zones=[_zone(camera_id="does-not-exist")], lines=[])

    assert [path.name for path in config_path.parent.iterdir()] == ["mesopic.yaml"]


def test_the_error_does_not_quote_the_document(config_path: Path) -> None:
    """The same leak P2.1 closed in the loader.

    The candidate document holds every camera block, and a validation error that renders
    its `input_value=` puts an RTSP URL — credentials and all — into whatever logs the
    exception. Here the URL is an env reference, so the assertion is on the *shape*: no
    part of the document body appears in the message.
    """
    with pytest.raises(ConfigError) as caught:
        write_geometry(config_path, zones=[_zone(camera_id="does-not-exist")], lines=[])

    message = str(caught.value)
    assert "MESOPIC_FRONT_DOOR_RTSP" not in message
    assert "acme-camden" not in message


# --- Emptying a block that a comment introduces (found by P3.6) ---------------

COMMENTED_DOCUMENT = DOCUMENT.replace(
    "lines:\n", "lines:\n  # Across the lower third, where feet cross in a doorway view.\n", 1
)
"""The same document, with a comment on its own line between `lines:` and its first item.

Nothing in the worked example is shaped this way — its comments sit *inside* items — which
is why every test above passes while this shape does not. `docker/demo.yaml` is written
this way, which is how it was found.
"""


@pytest.fixture
def commented_path(tmp_path: Path) -> Path:
    path = tmp_path / "mesopic.yaml"
    path.write_text(COMMENTED_DOCUMENT, encoding="utf-8")
    return path


def test_emptying_a_block_a_comment_introduces_still_loads(commented_path: Path) -> None:
    """Deleting the last line from a camera must not corrupt the file.

    An introducing comment has nothing left to introduce once its block is empty, and
    ruamel keeps emitting it — leaving the comment as the key's value line and the `[]`
    dangling at column zero, which is not YAML at all. The document the engine wrote then
    fails to parse on the next start, on a box nobody is watching.
    """
    write_geometry(commented_path, zones=[_zone()], lines=[])

    assert load_config(commented_path).lines == []


def test_a_comment_introducing_a_block_survives_a_non_empty_save(
    commented_path: Path,
) -> None:
    """The fix must not become "delete the comment on every save".

    Only an *empty* replacement drops it, because only then does it introduce nothing.
    A redraw that still has lines in it keeps the operator's annotation.
    """
    write_geometry(commented_path, zones=[_zone()], lines=[_line()])

    assert "feet cross in a doorway view" in commented_path.read_text(encoding="utf-8")


def test_a_document_the_writer_itself_mangles_never_lands(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard validates what would land, so it has to survive invalid *syntax* too.

    It parsed the candidate and caught only `ValidationError`, so a render that was not
    YAML escaped as a raw `ScannerError` — a 500 out of the save route rather than the
    refusal the route is built around, and the one case where the engine broke the file
    itself rather than being handed something bad.
    """
    monkeypatch.setattr("mesopic.config.writer._render", lambda _: "not: [valid: yaml")
    original = config_path.read_text(encoding="utf-8")

    with pytest.raises(ConfigError):
        write_geometry(config_path, zones=[_zone()], lines=[_line()])

    assert config_path.read_text(encoding="utf-8") == original
