"""The compiled, per-camera form of the config's lines and zones.

Built once from config and rebuilt only when config changes — never per frame. Polygons
are bounding-boxed so the common "not even close" case is a cheap rejection before any
point-in-polygon work.

The two predicates here are the ones every metric is eventually built from, and both are
algorithms.md's, imported rather than re-derived: containment is ray casting under the
even-odd rule (§6), and a line's side is the sign of the 2-D cross product of `AB` and
`AP` (§5). Both are deliberately *stateless* — the sticky-side bookkeeping, the crossing
debounce and the coasted-track policy are per-track state and belong to P2.3, which also
owns the segment-intersection test so that its tie-break for a zero side can never
disagree with the direction test sitting next to it.

Implements P2.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mesopic.config.schema import LineConfig, MesopicConfig, ZoneConfig
from mesopic.types import (
    CameraId,
    Direction,
    GridCell,
    LineId,
    MetricName,
    NormPoint,
    ZoneId,
    ZoneRole,
)

Bounds = tuple[float, float, float, float]
"""``(min_x, min_y, max_x, max_y)`` — a polygon's axis-aligned bounding box."""

GRID_W = 32
GRID_H = 32
"""The heatmap grid, fixed rather than configurable (algorithms.md §10's parameter table
calls it expert-level, and the plan requires a bounded blob). 32x32 `uint16` is 2 KB per
zone per minute, which is what the coarse sync cadence is sized against.

The grid spans the **whole frame**, not the zone's bounding box. That is forced by the
schema rather than chosen: `heatmap_minute` stores `grid_w`/`grid_h` and no origin, so a
zone-relative grid would silently misrender against every stored minute the moment the
calibration editor moved the zone. Cells outside the polygon simply stay zero.
"""


@dataclass(frozen=True, slots=True)
class PreparedZone:
    """One zone, with its bounding box computed once so `contains` can reject cheaply."""

    zone_id: ZoneId
    camera_id: CameraId
    role: ZoneRole
    polygon: tuple[NormPoint, ...]
    bounds: Bounds
    metrics: tuple[MetricName, ...]

    def contains(self, point: NormPoint) -> bool:
        """Is `point` inside this zone? Bounding box first, then ray casting."""
        x, y = point
        min_x, min_y, max_x, max_y = self.bounds
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
        return _point_in_polygon(point, self.polygon)


@dataclass(frozen=True, slots=True)
class PreparedLine:
    """One directed counting line. `positive_dir` labels the `+1` sense, it does not set it."""

    line_id: LineId
    camera_id: CameraId
    a: NormPoint
    b: NormPoint
    positive_dir: Direction
    metrics: tuple[MetricName, ...]

    def side_of(self, point: NormPoint) -> int:
        """Which half-plane of the infinite line `a -> b` holds `point`.

        `+1` is left of `a -> b`, `-1` is right, `0` is exactly on it. Zero is a real
        answer and callers must treat it as one: algorithms.md §5c holds the sticky side
        on a zero rather than folding it into a direction.
        """
        return side_of(self.a, self.b, point)

    def distance_to(self, point: NormPoint) -> float:
        """Unsigned normalized distance from `point` to the infinite line `a -> b`.

        The hysteresis band of algorithms.md §5(d) is expressed in these units, so the
        cross product has to be divided by `|AB|` rather than used raw — otherwise the
        band would silently scale with the length of the line the user happened to draw.
        """
        (ax, ay), (bx, by), (px, py) = self.a, self.b, point
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        return abs(cross) / math.hypot(bx - ax, by - ay)


@dataclass(frozen=True, slots=True, eq=False)
class SiteGeometry:
    """Prepared geometry for one site, queryable per camera.

    Construct with `compile`; the mappings are keyed by every camera in the config, so a
    camera with no geometry answers with an empty tuple while an unknown one raises.
    """

    _zones: dict[CameraId, tuple[PreparedZone, ...]]
    _lines: dict[CameraId, tuple[PreparedLine, ...]]

    @classmethod
    def compile(cls, config: MesopicConfig) -> SiteGeometry:
        """Build prepared geometry from validated config.

        Every camera gets an entry, including disabled ones: `enabled` is the
        supervisor's scheduling decision, and re-enabling a camera should not require
        recompiling the site.
        """
        zones: dict[CameraId, list[PreparedZone]] = {c.camera_id: [] for c in config.cameras}
        lines: dict[CameraId, list[PreparedLine]] = {c.camera_id: [] for c in config.cameras}
        for zone in config.zones:
            zones[zone.camera_id].append(_prepare_zone(zone))
        for line in config.lines:
            lines[line.camera_id].append(_prepare_line(line))
        return cls(
            _zones={camera: tuple(prepared) for camera, prepared in zones.items()},
            _lines={camera: tuple(prepared) for camera, prepared in lines.items()},
        )

    def zones_for(self, camera_id: CameraId) -> tuple[PreparedZone, ...]:
        """This camera's zones, in the order the config declared them."""
        self._require_known(camera_id)
        return self._zones[camera_id]

    def lines_for(self, camera_id: CameraId) -> tuple[PreparedLine, ...]:
        """This camera's lines, in the order the config declared them."""
        self._require_known(camera_id)
        return self._lines[camera_id]

    def zones_containing(self, camera_id: CameraId, point: NormPoint) -> list[ZoneId]:
        """Zone ids whose polygon contains `point`. Zones may overlap, so this is a list."""
        return [zone.zone_id for zone in self.zones_for(camera_id) if zone.contains(point)]

    def has_staff_zone(self, camera_id: CameraId) -> bool:
        """Whether this camera could tag a staff member at all (ADR-0021, P4.7).

        Asked per camera and not per site: a track is tagged by the zone it *originated*
        in, so on a camera with no `role: staff` zone nobody can ever be tagged, whoever
        they are. A staff sub-count there is not a count of no staff — it is a
        measurement that was not made, and the two must not render alike.
        """
        return any(zone.role is ZoneRole.STAFF for zone in self.zones_for(camera_id))

    def _require_known(self, camera_id: CameraId) -> None:
        """Guard a lookup, because "no geometry" and "no such camera" must not look alike.

        A camera that silently has no geometry is a camera that silently stops counting.
        Both maps are keyed by every configured camera, so either one answers this.
        """
        if camera_id not in self._zones:
            msg = f"unknown camera {camera_id!r}"
            raise KeyError(msg)


def _prepare_zone(zone: ZoneConfig) -> PreparedZone:
    polygon = tuple(zone.polygon)
    return PreparedZone(
        zone_id=zone.zone_id,
        camera_id=zone.camera_id,
        role=zone.role,
        polygon=polygon,
        bounds=_bounding_box(polygon),
        metrics=tuple(zone.metrics),
    )


def _prepare_line(line: LineConfig) -> PreparedLine:
    return PreparedLine(
        line_id=line.line_id,
        camera_id=line.camera_id,
        a=line.a,
        b=line.b,
        positive_dir=line.positive_dir,
        metrics=tuple(line.metrics),
    )


def side_of(a: NormPoint, b: NormPoint, point: NormPoint) -> int:
    """The sign of the 2-D cross product of `AB` and `AP` (algorithms.md §5).

    Module-level and shared on purpose. The segment-intersection test in `geometry` needs
    the orientation of a line's endpoints about a *track's* segment as well as the other
    way round, and §5(b) warns that the two must never disagree about which side a zero
    belongs to. One implementation is the only way to keep that true.
    """
    (ax, ay), (bx, by), (px, py) = a, b, point
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if cross > 0.0:
        return 1
    if cross < 0.0:
        return -1
    return 0


def _bounding_box(polygon: tuple[NormPoint, ...]) -> Bounds:
    """The prefilter, computed once per zone at compile time and never per query."""
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def _point_in_polygon(point: NormPoint, polygon: tuple[NormPoint, ...]) -> bool:
    """Ray casting under the even-odd rule (algorithms.md §6).

    The crossing test is half-open — an edge counts only where it straddles the ray's
    height — so two zones sharing an edge cannot both claim a point standing on the seam.
    """
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        (xi, yi), (xj, yj) = current, previous
        if (yi > y) != (yj > y):
            x_crossing = xj + (y - yj) * (xi - xj) / (yi - yj)
            if x < x_crossing:
                inside = not inside
        previous = current
    return inside


def cell_of(point: NormPoint, *, grid_w: int = GRID_W, grid_h: int = GRID_H) -> GridCell:
    """Which grid cell a normalized foot-point falls in (algorithms.md §10).

    Lives here, with the other normalized-space math, so the emitter (analytics) and the
    reducer (the heatmap accumulator) discretize identically — a grid built with one
    convention and rendered with another is off by a cell everywhere and looks plausible.

    The far edge is clamped rather than allowed to overflow: `floor(1.0 * 32)` is `32`,
    one past the last cell, and a foot-point exactly on the frame's right or bottom edge
    is a real observation rather than an error.
    """
    x, y = point
    column = min(math.floor(x * grid_w), grid_w - 1)
    line = min(math.floor(y * grid_h), grid_h - 1)
    return (max(column, 0), max(line, 0))
