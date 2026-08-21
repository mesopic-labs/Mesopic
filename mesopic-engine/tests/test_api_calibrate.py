"""What the calibration canvas is allowed to send, and what it may change.

This is the engine's first endpoint that changes state, so its body is the first
untrusted structure that ends up written to a file the engine itself reads back. It is
parsed into a strict typed shape at the edge and rejected on first inconsistency —
never sanitised and passed on.

Two rules here are not about types at all:

* **The camera comes from the path, never from the body.** Otherwise a request
  calibrating one camera could rewrite another camera's geometry, and the URL an
  operator is looking at would not be the thing they changed.
* **A save replaces one camera's geometry and leaves every other camera's alone.**
  `write_geometry` takes whole lists, so the merge has to happen somewhere; doing it in
  a pure function keeps it testable without a browser, a supervisor or a config file.

Implements P3.3.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mesopic.api.calibrate import (
    MAX_POLYGON_VERTICES,
    MAX_SHAPES_PER_CAMERA,
    GeometryEdit,
    merge_geometry,
)
from mesopic.config.schema import MesopicConfig
from mesopic.types import CameraId

CONFIG = MesopicConfig.model_validate(
    {
        "site": {"site_id": "acme", "timezone": "UTC"},
        "cameras": [
            {
                "camera_id": "front-door",
                "name": "Front door",
                "source": {"kind": "rtsp", "url_env": "FRONT_DOOR_RTSP"},
                "reference_resolution": [1920, 1080],
            },
            {
                "camera_id": "till",
                "name": "Till",
                "source": {"kind": "rtsp", "url_env": "TILL_RTSP"},
                "reference_resolution": [1280, 720],
            },
        ],
        "lines": [
            {
                "line_id": "door-count",
                "camera_id": "front-door",
                "a": [0.1, 0.8],
                "b": [0.9, 0.8],
                "metrics": ["footfall"],
            },
            {
                "line_id": "till-approach",
                "camera_id": "till",
                "a": [0.2, 0.5],
                "b": [0.8, 0.5],
                "metrics": ["line_cross"],
            },
        ],
        "zones": [
            {
                "zone_id": "shop-floor",
                "camera_id": "front-door",
                "role": "area",
                "polygon": [[0.05, 0.3], [0.95, 0.3], [0.95, 0.95]],
                "metrics": ["occupancy"],
            },
            {
                "zone_id": "queue-till",
                "camera_id": "till",
                "role": "queue",
                "polygon": [[0.2, 0.4], [0.8, 0.4], [0.8, 0.9]],
                "metrics": ["queue_len"],
            },
        ],
    }
)

A_SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


def _edit(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "zones": [
            {
                "zone_id": "redrawn",
                "role": "area",
                "polygon": A_SQUARE,
                "metrics": ["occupancy"],
            }
        ],
        "lines": [
            {
                "line_id": "redrawn-line",
                "a": [0.0, 0.5],
                "b": [1.0, 0.5],
                "positive_dir": "in",
                "metrics": ["footfall"],
            }
        ],
    }
    body.update(overrides)
    return body


# --- The body -----------------------------------------------------------------


def test_a_coordinate_outside_the_unit_square_is_rejected() -> None:
    """Normalized means normalized. A pixel value here would silently mis-place a zone."""
    body = _edit(
        zones=[{"zone_id": "z", "role": "area", "polygon": [[0.1, 0.1], [1.4, 0.1], [0.9, 0.9]]}]
    )

    with pytest.raises(ValidationError):
        GeometryEdit.model_validate(body)


def test_a_polygon_with_fewer_than_three_points_is_rejected() -> None:
    body = _edit(zones=[{"zone_id": "z", "role": "area", "polygon": [[0.1, 0.1], [0.9, 0.9]]}])

    with pytest.raises(ValidationError):
        GeometryEdit.model_validate(body)


def test_a_polygon_larger_than_the_vertex_limit_is_rejected() -> None:
    """Unbounded input becomes an unbounded config file and an unbounded ray cast."""
    too_many = [[0.5, 0.5]] * (MAX_POLYGON_VERTICES + 1)
    body = _edit(zones=[{"zone_id": "z", "role": "area", "polygon": too_many}])

    with pytest.raises(ValidationError):
        GeometryEdit.model_validate(body)


def test_more_shapes_than_a_camera_may_hold_are_rejected() -> None:
    zone = {"zone_id": "z", "role": "area", "polygon": A_SQUARE}
    body = _edit(zones=[zone] * (MAX_SHAPES_PER_CAMERA + 1))

    with pytest.raises(ValidationError):
        GeometryEdit.model_validate(body)


def test_an_unknown_key_is_rejected() -> None:
    """`extra="forbid"`, for the same reason the config schema has it."""
    with pytest.raises(ValidationError):
        GeometryEdit.model_validate(_edit(camera_id="till"))


def test_an_empty_edit_is_allowed() -> None:
    """Clearing every zone on a camera is a legitimate thing to save."""
    edit = GeometryEdit.model_validate({"zones": [], "lines": []})

    assert edit.zones == []
    assert edit.lines == []


# --- The merge ----------------------------------------------------------------


def test_the_edit_replaces_only_the_named_camera() -> None:
    edit = GeometryEdit.model_validate(_edit())

    zones, lines = merge_geometry(CONFIG, CameraId("front-door"), edit)

    assert [zone.zone_id for zone in zones] == ["queue-till", "redrawn"]
    assert [line.line_id for line in lines] == ["till-approach", "redrawn-line"]


def test_the_camera_comes_from_the_path_not_the_body() -> None:
    """The body has no camera field at all, so the merged shapes carry the path's."""
    edit = GeometryEdit.model_validate(_edit())

    zones, lines = merge_geometry(CONFIG, CameraId("front-door"), edit)

    assert {zone.camera_id for zone in zones if zone.zone_id == "redrawn"} == {"front-door"}
    assert {line.camera_id for line in lines if line.line_id == "redrawn-line"} == {"front-door"}


def test_clearing_a_camera_leaves_the_other_cameras_intact() -> None:
    edit = GeometryEdit.model_validate({"zones": [], "lines": []})

    zones, lines = merge_geometry(CONFIG, CameraId("front-door"), edit)

    assert [zone.zone_id for zone in zones] == ["queue-till"]
    assert [line.line_id for line in lines] == ["till-approach"]
