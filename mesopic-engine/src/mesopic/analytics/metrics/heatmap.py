"""Zone heatmaps — the density grid, accumulated per minute (adjacency A1).

Deliberately **not** a `MetricPlugin`. Every other metric reduces to a scalar `MetricRow`
upserted into `metrics_minute`; a heatmap is a blob on a different natural key in a
different table (engine-architecture.md §11). Widening the plugin protocol to return two
row shapes would put a case into every existing plugin's signature that none of them ever
uses, and would make `metrics_minute` stop being the scalar, queryable thing §11 split it
out to be. So this is a second fold over the same bucket, driven by the aggregator
alongside the registry, and it inherits the scalar path's replay-safety by folding the
same retained events rather than by keeping a running total of its own.

The unit is **deciseconds of foot-point presence**, `Δt`-weighted, and the count-weighted
version is not a worse approximation — it is a different picture. The adaptive sampler
lowers fps when the box is under load, and the box is under load when the scene is busy,
so a `+= 1` grid is brightest where the sampler was *fastest*: a rendering of the load
controller rather than of the floor (algorithms.md §0.6, §10).

Storage is raw and undecayed. Decay and normalization are render-time choices applied to
a rolled-up window (§10), so any window can be reconstructed from stored minutes — the
same store-raw/derive-on-read rule the rest of the engine follows.

Implements P4.1.
"""

from __future__ import annotations

import struct
from collections import defaultdict
from collections.abc import Sequence

from mesopic.analytics.site_geometry import GRID_H, GRID_W, PreparedZone, SiteGeometry
from mesopic.types import (
    CameraId,
    EventKind,
    GridCell,
    HeatmapRow,
    MetricName,
    MinuteBucket,
    RawEvent,
    ZoneId,
)

MAX_CELL = 65535
"""What a `uint16` cell saturates at: 65535 ds is ~1.8 h of presence in one cell.

Unreachable inside a one-minute bucket — you would need ~109 people standing in the same
cell for the whole minute — so the clamp is a guarantee about the type rather than a
behaviour anyone will observe. It exists because saturating reads as "very hot" while
wrapping reads as "cold", and the wrap would be invisible.
"""

DECISECONDS_PER_SECOND = 10.0

_CellCounts = dict[GridCell, float]


def pack_counts(counts: Sequence[float], *, width: int, height: int) -> bytes:
    """Pack a row-major grid of deciseconds into little-endian `uint16`.

    **Explicitly little-endian, not native.** These bytes sync to the cloud (ADR-0010),
    so native order would make the wire format depend on which box happened to write it.

    Rounding happens here, once, and never per hit: at 20 fps a single hit is 0.5 ds and
    `round(0.5)` is `0` under banker's rounding, so a per-hit round would quietly render
    an empty grid at exactly the frame rates the engine is fastest at.
    """
    if len(counts) != width * height:
        msg = f"grid is {width}x{height} but {len(counts)} cells were given"
        raise ValueError(msg)
    return struct.pack(f"<{len(counts)}H", *(min(round(count), MAX_CELL) for count in counts))


def unpack_counts(blob: bytes) -> tuple[int, ...]:
    """Read a packed grid back. The inverse of `pack_counts`, for readers and tests."""
    return struct.unpack(f"<{len(blob) // 2}H", blob)


class HeatmapAccumulator:
    """Folds one bucket's `heatmap_hit` events into one grid blob per zone."""

    def __init__(
        self, geometry: SiteGeometry, *, grid_w: int = GRID_W, grid_h: int = GRID_H
    ) -> None:
        self._geometry = geometry
        self._grid_w = grid_w
        self._grid_h = grid_h

    def fold(self, events: Sequence[RawEvent], bucket: MinuteBucket) -> list[HeatmapRow]:
        """Sum the presence time each cell accumulated in this minute.

        Deterministic over the events it is given, so re-folding a bucket after a late
        arrival produces the same grid rather than a doubled one — the property that
        makes the store's upsert safe to replay (P2.6).

        A zone nobody was seen in produces **no row**, not a grid of zeros: an empty zone
        renders cold from a missing row, and writing 2 KB of zeros per zone per minute to
        say "nothing happened" is a cost with no reader.
        """
        totals: dict[tuple[CameraId, ZoneId], _CellCounts] = defaultdict(lambda: defaultdict(float))
        for event in events:
            key = self._key_of(event)
            if key is None or event.cell is None:
                continue
            totals[key][event.cell] += (event.dt_s or 0.0) * DECISECONDS_PER_SECOND

        return [
            self._row(camera_id, zone_id, cells, bucket)
            for (camera_id, zone_id), cells in sorted(totals.items())
            if cells
        ]

    def _key_of(self, event: RawEvent) -> tuple[CameraId, ZoneId] | None:
        """The zone a hit belongs to, or `None` if it is not one we accumulate."""
        if event.kind is not EventKind.HEATMAP_HIT or event.zone_id is None:
            return None
        if not self._accumulates(event.camera_id, event.zone_id):
            return None
        return (event.camera_id, event.zone_id)

    def _accumulates(self, camera_id: CameraId, zone_id: ZoneId) -> bool:
        """Only zones whose config asked for `heatmap`. Drawing the zone is what enables it."""
        return any(
            zone.zone_id == zone_id and MetricName.HEATMAP in zone.metrics
            for zone in self._zones_for(camera_id)
        )

    def _zones_for(self, camera_id: CameraId) -> tuple[PreparedZone, ...]:
        try:
            return self._geometry.zones_for(camera_id)
        except KeyError:
            # A camera whose geometry was removed mid-bucket. Its retained events are
            # still foldable for the scalar metrics; there is simply no grid to build.
            return ()

    def _row(
        self,
        camera_id: CameraId,
        zone_id: ZoneId,
        cells: _CellCounts,
        bucket: MinuteBucket,
    ) -> HeatmapRow:
        grid = [0.0] * (self._grid_w * self._grid_h)
        for (column, line), deciseconds in cells.items():
            grid[line * self._grid_w + column] += deciseconds
        return HeatmapRow(
            camera_id=camera_id,
            bucket=bucket,
            zone_id=zone_id,
            grid_w=self._grid_w,
            grid_h=self._grid_h,
            counts=pack_counts(grid, width=self._grid_w, height=self._grid_h),
        )
