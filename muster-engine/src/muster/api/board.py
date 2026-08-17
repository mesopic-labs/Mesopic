"""The dashboard board: what the polled fragment shows and how its numbers are reached.

Presentation only. Nothing here computes a metric — §13 makes the API a reader, and every
value below is either copied from a stored bucket or summed across the buckets already
returned for the charts. A number the board cannot get this way is a number that belongs
in a plugin.

Two decisions worth stating, because both are easy to get subtly wrong and neither is
visible in the rendered page:

* **Tiles come from the config, not from the rows.** `camera_health` follows the same
  rule for the same reason: a scope missing from the board reads as a scope that does not
  exist, so the operator's config decides who appears and the store only fills values in.
  The consequence is the next point.
* **A scope with no rows has value `None`, and renders as `—`.** Zero is a measurement —
  it says nobody walked past. `None` says nothing was reported. Collapsing the two tells
  an operator with a dead camera that their shop is empty.

Implements P3.2.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from muster.config.schema import MusterConfig
from muster.types import CameraId, MetricName, MetricRow, MinuteBucket, ScopeId


class BoardWindow(StrEnum):
    """How much history the board shows.

    A closed set rather than a duration the caller names: the window bounds the query, so
    letting the page ask for an arbitrary span is how a dashboard poll reads a year of
    minute buckets. `/api/metrics` takes the general case and bounds it explicitly;
    this is the page's own control, and three fixed choices are all it offers.
    """

    HOUR = "1h"
    SIX_HOURS = "6h"
    DAY = "24h"

    @property
    def span(self) -> timedelta:
        return _SPANS[self]


_SPANS = {
    BoardWindow.HOUR: timedelta(hours=1),
    BoardWindow.SIX_HOURS: timedelta(hours=6),
    BoardWindow.DAY: timedelta(hours=24),
}

DEFAULT_WINDOW = BoardWindow.SIX_HOURS

COUNTING_METRICS = frozenset({MetricName.FOOTFALL, MetricName.LINE_CROSS})
"""Metrics whose tile is the window's total. Everything else is a level, and a level is
shown as its newest bucket — summing occupancy would report a quiet afternoon as a
crowd."""

UNTILED_METRICS = frozenset({MetricName.HEATMAP})
"""`heatmap` is a packed grid blob, not a number, and has no scalar to put on a tile.
Rendering it is P4.1's overlay."""


@dataclass(frozen=True, slots=True)
class Tile:
    """One configured `(camera, metric, scope)` and its current reading."""

    camera_id: CameraId
    metric: MetricName
    scope_id: ScopeId | None
    label: str
    value: float | None
    """`None` means nothing was reported for this scope in the window — not zero."""

    @property
    def total(self) -> bool:
        return self.metric in COUNTING_METRICS


def tiles_for(config: MusterConfig, *, rows: Sequence[MetricRow]) -> tuple[Tile, ...]:
    """Every configured scope, in config order, filled from the window's rows."""
    readings = _fold(rows)
    return tuple(
        Tile(
            camera_id=camera_id,
            metric=metric,
            scope_id=scope_id,
            label=scope_id or camera_id,
            value=readings.get((camera_id, metric, scope_id)),
        )
        for camera_id, metric, scope_id in _configured_scopes(config)
    )


EXPOSURE_CELLS = 60
"""How many cells the exposure strip is divided into — a minute each at the shortest
window, twenty-four at the longest."""


@dataclass(frozen=True, slots=True)
class Exposure:
    """Which parts of the window the engine actually has rows for.

    Presence, not magnitude: a cell is lit if *any* scope reported inside it. It answers
    the question that comes before every number on the page — has this box been seeing
    anything, and for how long — which no individual metric can answer, because a quiet
    zone and a dead camera produce the same empty chart.
    """

    cells: tuple[bool, ...]
    """Oldest first."""

    @property
    def covered(self) -> int:
        return sum(self.cells)


def exposure_of(
    rows: Sequence[MetricRow],
    *,
    end: datetime,
    window: BoardWindow,
    cells: int = EXPOSURE_CELLS,
) -> Exposure:
    """Fold every row into the cell of the strip its bucket falls in.

    Rows outside the window are dropped rather than clamped: a row from before the window
    is not evidence that the window has data, and clamping it into the first cell would
    draw exactly that claim.
    """
    span = window.span / cells
    start = MinuteBucket(end - window.span)
    lit = [False] * cells
    for row in rows:
        if not start <= row.bucket <= end:
            continue
        # A bucket at exactly `end` indexes one past the last cell; it belongs to the
        # strip's final cell, not off the end of it.
        lit[min(int((row.bucket - start) / span), cells - 1)] = True
    return Exposure(cells=tuple(lit))


def human_duration(seconds: float) -> str:
    """Seconds as the coarsest two units that still say something.

    An uptime is read at a glance and never acted on to the second, so `14h 22m` is the
    whole of what the operator wants from `51720.4`.
    """
    whole = max(int(seconds), 0)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def scope_slots(config: MusterConfig) -> dict[str, int]:
    """A stable colour slot per scope, in config order.

    Colour follows the scope, never its rank inside one chart: assigning by position
    within a chart makes `shop-floor` the first series on the occupancy plate and the
    second on the dwell plate, so it changes colour between two panels the operator reads
    side by side. From the config for the same reason the tiles are — a scope that is
    silent today must not be handed a different colour tomorrow when it starts reporting.
    """
    slots: dict[str, int] = {}
    for camera_id, _, scope_id in _configured_scopes(config):
        slots.setdefault(scope_id or camera_id, len(slots))
    return slots


def as_series(rows: Sequence[MetricRow]) -> list[dict[str, Any]]:
    """Group rows into one columnar series per `(camera, metric, scope)`.

    Columnar because uPlot consumes parallel arrays; one flat list would plot two cameras'
    footfall as a single sawtooth. Grouping is a reshape of what the store returned in
    order, not a computation — no value here is derived from another.
    """
    series: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for row in rows:
        key = (row.camera_id, row.metric.value, row.scope_id)
        if key not in series:
            series[key] = {
                "camera_id": row.camera_id,
                "metric": row.metric.value,
                "scope_id": row.scope_id,
                "t": [],
                "v": [],
            }
        series[key]["t"].append(int(row.bucket.timestamp()))
        series[key]["v"].append(row.value)
    return list(series.values())


def charts_of(config: MusterConfig, *, rows: Sequence[MetricRow]) -> list[dict[str, Any]]:
    """One chart per configured metric, in uPlot's `[xs, ys…]` shape.

    Built from the config for the same reason the tiles are: a metric the operator asked
    for and is not getting should show an empty chart, which is a visible absence, rather
    than no chart at all, which is indistinguishable from never having asked.

    Every scope in a chart shares one x axis, and a scope with no row at some bucket gets
    `None` there rather than a shorter array. A shorter array is the bug worth naming: it
    slides every later point one bucket earlier, so a camera that dropped a minute draws
    a plausible chart of events that happened at the wrong time.
    """
    by_metric: dict[MetricName, list[MetricRow]] = {}
    for row in rows:
        by_metric.setdefault(row.metric, []).append(row)

    slots = scope_slots(config)
    return [
        _chart(metric, by_metric.get(metric, []), slots) for metric in _configured_metrics(config)
    ]


def _chart(metric: MetricName, rows: Sequence[MetricRow], slots: dict[str, int]) -> dict[str, Any]:
    points: dict[str, dict[int, float]] = {}
    for row in rows:
        label = row.scope_id or row.camera_id
        points.setdefault(label, {})[int(row.bucket.timestamp())] = row.value

    axis = sorted({at for scope in points.values() for at in scope})
    labels = sorted(points)
    return {
        "metric": metric.value,
        # A count per bucket is a bar and a level is a line — the same split the readings
        # columns make, drawn rather than captioned. A count joined bucket to bucket is a
        # sawtooth that implies the floor emptied and refilled every single minute.
        "total": metric in COUNTING_METRICS,
        "labels": labels,
        # Parallel to `labels`, so the plate's swatches and the plot's strokes cannot
        # drift apart. A scope the config does not know about folds into the last slot.
        "slots": [slots.get(label, len(slots)) for label in labels],
        "t": axis,
        "v": [[points[label].get(at) for at in axis] for label in labels],
    }


def _configured_scopes(
    config: MusterConfig,
) -> Iterator[tuple[CameraId, MetricName, ScopeId | None]]:
    for line in config.lines:
        for metric in line.metrics:
            if metric not in UNTILED_METRICS:
                yield line.camera_id, metric, ScopeId(line.line_id)
    for zone in config.zones:
        for metric in zone.metrics:
            if metric not in UNTILED_METRICS:
                yield zone.camera_id, metric, ScopeId(zone.zone_id)


def _configured_metrics(config: MusterConfig) -> list[MetricName]:
    """The metrics this site actually collects, deduplicated but in config order."""
    seen: dict[MetricName, None] = {}
    for _, metric, _ in _configured_scopes(config):
        seen.setdefault(metric, None)
    return list(seen)


_Key = tuple[CameraId, MetricName, ScopeId | None]


def _fold(rows: Sequence[MetricRow]) -> dict[_Key, float]:
    """Reduce each scope's buckets to the one number its tile shows.

    Bucket order is compared rather than assumed: the store returns rows oldest-first
    today, and a tile that silently depends on that would be wrong the first time
    somebody adds an `ORDER BY`.
    """
    totals: dict[_Key, float] = {}
    newest: dict[_Key, tuple[MinuteBucket, float]] = {}
    for row in rows:
        key = (row.camera_id, row.metric, row.scope_id)
        if row.metric in COUNTING_METRICS:
            totals[key] = totals.get(key, 0.0) + row.value
        else:
            seen = newest.get(key)
            if seen is None or row.bucket >= seen[0]:
                newest[key] = (row.bucket, row.value)
    return totals | {key: value for key, (_, value) in newest.items()}


__all__ = [
    "COUNTING_METRICS",
    "DEFAULT_WINDOW",
    "EXPOSURE_CELLS",
    "UNTILED_METRICS",
    "BoardWindow",
    "Exposure",
    "Tile",
    "as_series",
    "charts_of",
    "exposure_of",
    "human_duration",
    "scope_slots",
    "tiles_for",
]
