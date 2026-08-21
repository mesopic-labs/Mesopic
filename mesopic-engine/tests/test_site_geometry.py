"""What the compiled form of a site's geometry must answer, and how.

`SiteGeometry` is the one place config becomes queryable geometry, so these tests are
about two separable things: that compiling *partitions* config correctly per camera, and
that the two predicates built on the prepared form — point-in-zone and line-side — give
the answers algorithms.md §5 and §6 specify, on fixtures worked out by hand rather than
by running the implementation and blessing its output.

Fixtures build on `examples/mesopic.yaml` for the same reason `test_config_validation.py`
does: a second hand-written schema in the test suite is a second schema to drift.
Geometry-specific shapes (a triangle, a concave zone, two zones sharing an edge) are
substituted into that example, so a config change breaks these tests loudly.

Red-first for P2.2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from mesopic.analytics import site_geometry
from mesopic.analytics.site_geometry import SiteGeometry
from mesopic.config.schema import MesopicConfig
from mesopic.types import CameraId, LineId, MetricName, NormPoint, ZoneId, ZoneRole

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "mesopic.yaml"

FRONT_DOOR = CameraId("front-door")
TILL = CameraId("till")

SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
"""A plain axis-aligned box, so "inside" and "outside" need no working out."""

TRIANGLE = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
"""Half of the unit square: its bounding box is the whole square, so the corner at
`(0.9, 0.9)` is inside the box and outside the polygon — which is exactly the case a
bounding-box prefilter gets wrong if it is treated as an answer instead of a filter."""

L_SHAPE = [[0.0, 0.0], [0.6, 0.0], [0.6, 0.6], [0.4, 0.6], [0.4, 0.2], [0.0, 0.2]]
"""Concave: a bar across the top (`y <= 0.2`) and a leg down its right (`x >= 0.4`). The
notch at `(0.2, 0.4)` is inside the bounding box, inside the convex hull, and outside the
polygon — the case ray casting handles and a hull test does not."""


@pytest.fixture
def example() -> dict[str, Any]:
    """The worked example, parsed fresh per test — no shared mutable state."""
    parsed = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def _compile(example: dict[str, Any], **sections: list[dict[str, Any]]) -> SiteGeometry:
    """Compile the worked example with some of its geometry sections replaced."""
    return SiteGeometry.compile(MesopicConfig.model_validate({**example, **sections}))


def _zone(zone_id: str, camera_id: str, polygon: list[list[float]], **extra: Any) -> dict[str, Any]:
    return {"zone_id": zone_id, "camera_id": camera_id, "polygon": polygon, **extra}


def _line(line_id: str, camera_id: str, a: list[float], b: list[float]) -> dict[str, Any]:
    return {"line_id": line_id, "camera_id": camera_id, "a": a, "b": b}


# --- Compiling: config in, per-camera prepared geometry out ------------------


def test_compile_groups_geometry_by_the_camera_it_belongs_to(example: dict[str, Any]) -> None:
    """The worked example puts one zone and one line on `front-door` and two zones on `till`."""
    geometry = SiteGeometry.compile(MesopicConfig.model_validate(example))

    assert [zone.zone_id for zone in geometry.zones_for(FRONT_DOOR)] == [ZoneId("shop-floor")]
    assert [line.line_id for line in geometry.lines_for(FRONT_DOOR)] == [LineId("door-count")]
    assert [zone.zone_id for zone in geometry.zones_for(TILL)] == [
        ZoneId("queue-till"),
        ZoneId("behind-counter"),
    ]


def test_a_camera_with_no_lines_compiles_to_an_empty_tuple(example: dict[str, Any]) -> None:
    """Absence of geometry is legitimate and must not be confused with an unknown camera."""
    geometry = SiteGeometry.compile(MesopicConfig.model_validate(example))

    assert geometry.lines_for(TILL) == ()


def test_an_unknown_camera_is_an_error_not_an_empty_result(example: dict[str, Any]) -> None:
    """Silently answering "no zones" for a typo is how a camera stops counting unnoticed."""
    geometry = SiteGeometry.compile(MesopicConfig.model_validate(example))

    with pytest.raises(KeyError, match="no-such-camera"):
        geometry.zones_for(CameraId("no-such-camera"))


def test_a_disabled_camera_keeps_its_compiled_geometry(example: dict[str, Any]) -> None:
    """`enabled` is a scheduling decision for the supervisor, not a geometry one.

    Dropping the geometry here would mean re-compiling the site to re-enable a camera.
    """
    example["cameras"][0]["enabled"] = False

    geometry = SiteGeometry.compile(MesopicConfig.model_validate(example))

    assert [zone.zone_id for zone in geometry.zones_for(FRONT_DOOR)] == [ZoneId("shop-floor")]


def test_compile_carries_the_role_and_metrics_a_zone_was_configured_with(
    example: dict[str, Any],
) -> None:
    """The metric layer reads these off the prepared zone; losing them loses the metric."""
    geometry = SiteGeometry.compile(MesopicConfig.model_validate(example))

    (queue,) = [zone for zone in geometry.zones_for(TILL) if zone.zone_id == ZoneId("queue-till")]

    assert queue.role is ZoneRole.QUEUE
    assert queue.metrics == (MetricName.QUEUE_LEN, MetricName.DWELL_SECONDS)


def test_a_zones_bounds_are_the_axis_aligned_box_of_its_polygon(example: dict[str, Any]) -> None:
    """Hand-checked: the L-shape spans x in [0, 0.6] and y in [0, 0.6]."""
    geometry = _compile(example, zones=[_zone("l", "front-door", L_SHAPE)], lines=[])

    (zone,) = geometry.zones_for(FRONT_DOOR)

    assert zone.bounds == (0.0, 0.0, 0.6, 0.6)


# --- Point-in-zone (algorithms.md §6) ---------------------------------------


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0.5, 0.5), True),
        ((0.25, 0.75), True),
        ((0.1, 0.5), False),
        ((0.9, 0.5), False),
        ((0.5, 0.1), False),
        ((0.5, 0.9), False),
    ],
)
def test_a_square_zone_contains_exactly_the_points_inside_it(
    example: dict[str, Any], point: NormPoint, expected: bool
) -> None:
    geometry = _compile(example, zones=[_zone("square", "front-door", SQUARE)], lines=[])

    (zone,) = geometry.zones_for(FRONT_DOOR)

    assert zone.contains(point) is expected


def test_a_point_inside_the_bounding_box_but_outside_the_polygon_is_not_contained(
    example: dict[str, Any],
) -> None:
    """The AABB is a prefilter, never the answer — the triangle's far corner proves it."""
    geometry = _compile(example, zones=[_zone("triangle", "front-door", TRIANGLE)], lines=[])

    (zone,) = geometry.zones_for(FRONT_DOOR)

    assert zone.bounds == (0.0, 0.0, 1.0, 1.0)
    assert zone.contains((0.9, 0.9)) is False
    assert zone.contains((0.1, 0.1)) is True


def test_a_concave_zone_excludes_the_notch_between_its_arms(example: dict[str, Any]) -> None:
    """Ray casting handles arbitrary simple polygons; `(0.2, 0.4)` sits in the notch."""
    geometry = _compile(example, zones=[_zone("l", "front-door", L_SHAPE)], lines=[])

    (zone,) = geometry.zones_for(FRONT_DOOR)

    assert zone.contains((0.2, 0.4)) is False
    assert zone.contains((0.2, 0.1)) is True
    assert zone.contains((0.5, 0.4)) is True


def test_a_point_on_a_shared_edge_belongs_to_exactly_one_of_two_touching_zones(
    example: dict[str, Any],
) -> None:
    """Two zones drawn edge-to-edge must not both count the person standing on the seam.

    The even-odd rule with a half-open crossing test gives this for free, and it is the
    property that matters — not which of the two wins, which is arbitrary.
    """
    geometry = _compile(
        example,
        zones=[
            _zone("left", "front-door", [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]),
            _zone("right", "front-door", [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]),
        ],
        lines=[],
    )

    on_the_seam = geometry.zones_containing(FRONT_DOOR, (0.5, 0.5))

    assert len(on_the_seam) == 1


def test_zones_containing_returns_every_zone_covering_the_point(example: dict[str, Any]) -> None:
    """Zones may overlap — a queue inside a shop floor is the ordinary case, not an error."""
    geometry = _compile(
        example,
        zones=[
            _zone("floor", "front-door", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            _zone("queue", "front-door", SQUARE, role="queue"),
            _zone("corner", "front-door", [[0.9, 0.9], [1.0, 0.9], [1.0, 1.0], [0.9, 1.0]]),
        ],
        lines=[],
    )

    assert geometry.zones_containing(FRONT_DOOR, (0.5, 0.5)) == [ZoneId("floor"), ZoneId("queue")]


def test_zones_containing_ignores_zones_on_another_camera(example: dict[str, Any]) -> None:
    """Geometry is per-camera: the same normalized point means different things per view."""
    geometry = _compile(
        example,
        zones=[
            _zone("front", "front-door", SQUARE),
            _zone("till-side", "till", SQUARE),
        ],
        lines=[],
    )

    assert geometry.zones_containing(FRONT_DOOR, (0.5, 0.5)) == [ZoneId("front")]
    assert geometry.zones_containing(TILL, (0.5, 0.5)) == [ZoneId("till-side")]


# --- Line side (algorithms.md §5) -------------------------------------------


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0.5, 0.9), 1),
        ((0.5, 0.1), -1),
        ((0.05, 0.9), 1),
    ],
)
def test_side_of_is_the_sign_of_the_cross_product_of_ab_and_ap(
    example: dict[str, Any], point: NormPoint, expected: int
) -> None:
    """`a=(0.1,0.5) b=(0.9,0.5)` runs left to right, so with y down, below it is `+1`.

    Hand-checked: `(Bx-Ax)(Py-Ay) - (By-Ay)(Px-Ax)` = `0.8 * (Py - 0.5) - 0`, whose sign
    is the sign of `Py - 0.5` — and does not depend on `Px` at all, which is the last
    case: a point off the end of the segment still has a side of the infinite line.
    """
    geometry = _compile(example, lines=[_line("door", "front-door", [0.1, 0.5], [0.9, 0.5])])

    (line,) = geometry.lines_for(FRONT_DOOR)

    assert line.side_of(point) == expected


def test_a_point_exactly_on_the_line_has_side_zero(example: dict[str, Any]) -> None:
    """Zero is a real answer, not a rounding artefact: §5c holds the sticky side on it."""
    geometry = _compile(example, lines=[_line("door", "front-door", [0.1, 0.5], [0.9, 0.5])])

    (line,) = geometry.lines_for(FRONT_DOOR)

    assert line.side_of((0.5, 0.5)) == 0


def test_a_point_collinear_beyond_the_endpoints_still_has_side_zero(
    example: dict[str, Any],
) -> None:
    """`side_of` is the infinite line's half-plane test; the segment test is P2.3's job."""
    geometry = _compile(example, lines=[_line("door", "front-door", [0.1, 0.5], [0.9, 0.5])])

    (line,) = geometry.lines_for(FRONT_DOOR)

    assert line.side_of((0.99, 0.5)) == 0


def test_reversing_a_lines_endpoints_flips_which_side_a_point_is_on(
    example: dict[str, Any],
) -> None:
    """`positive_dir` labels the `+1` sense, so the sign has to follow `a -> b`."""
    point = (0.5, 0.9)
    forward = _compile(example, lines=[_line("door", "front-door", [0.1, 0.5], [0.9, 0.5])])
    reversed_ = _compile(example, lines=[_line("door", "front-door", [0.9, 0.5], [0.1, 0.5])])

    (forward_line,) = forward.lines_for(FRONT_DOOR)
    (reversed_line,) = reversed_.lines_for(FRONT_DOOR)

    assert forward_line.side_of(point) == -reversed_line.side_of(point)


def test_a_line_carries_the_positive_direction_it_was_configured_with(
    example: dict[str, Any],
) -> None:
    """Which geometric side means "in" is user semantics; the engine must not guess it."""
    geometry = SiteGeometry.compile(MesopicConfig.model_validate(example))

    (line,) = geometry.lines_for(FRONT_DOOR)

    assert line.positive_dir.value == "in"
    assert line.metrics == (MetricName.LINE_CROSS, MetricName.FOOTFALL)


# --- Prepared once, queried many times --------------------------------------


def test_polygons_are_prepared_at_compile_time_and_never_per_query(
    example: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of a compile step: querying must not re-derive the prepared form.

    Counted rather than asserted structurally, because "it happens to be fast" and "it
    provably happens once" are different claims and only the second one survives someone
    moving the bounding-box computation into `contains`.
    """
    calls = 0
    real = site_geometry._bounding_box

    def counting(polygon: tuple[NormPoint, ...]) -> tuple[float, float, float, float]:
        nonlocal calls
        calls += 1
        return real(polygon)

    monkeypatch.setattr(site_geometry, "_bounding_box", counting)

    geometry = _compile(
        example,
        zones=[_zone("square", "front-door", SQUARE), _zone("l", "front-door", L_SHAPE)],
        lines=[],
    )
    at_compile = calls
    for _ in range(200):
        geometry.zones_containing(FRONT_DOOR, (0.5, 0.5))

    assert at_compile == 2
    assert calls == at_compile
