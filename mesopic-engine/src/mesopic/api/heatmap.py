"""Turning stored minute grids into something a browser can paint.

Everything here is a **render choice**, which is why it lives on the read side and not in
the accumulator. `heatmap_minute` holds raw, undecayed deciseconds; decay, normalization
and smoothing are applied to a rolled-up window on the way out, so any window can be
rebuilt from the same stored rows and a display tweak never means re-accumulating
(algorithms.md §10 — store raw, derive on read).

Two of those choices are load-bearing rather than cosmetic:

* **Decay is exponential in the age of the minute**, so a window reads as a moving
  picture rather than an all-time integral in which yesterday's rush drowns this
  morning's.
* **Normalization is logarithmic against a p99 clip.** Foot-point density is heavy-tailed
  — a till cell is routinely 100x any other — so linear normalization against the maximum
  renders everything except the single hottest cell as cold, which is a picture of one
  cell rather than of the floor.

Implements P4.1.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from mesopic.analytics.metrics.heatmap import unpack_counts
from mesopic.types import CameraId, HeatmapRow, ZoneId

DEFAULT_DECAY = 0.95
"""Per-minute decay (algorithms.md §10's alpha). A minute `t` ago contributes
`decay ** t`, so the half-life is about 13.5 minutes. Expert-level in §10's parameter
table, so it is an argument here rather than a `mesopic.yaml` key."""

CLIP_QUANTILE = 0.99
"""Where the colour scale tops out, over the *occupied* cells only.

Taken over non-zero cells rather than over the whole grid, because most of a frame is
floor nobody walks on: with 95% of cells at zero, a p99 over everything lands in the
noise and the scale saturates on contact.
"""


@dataclass(frozen=True, slots=True)
class HeatmapView:
    """One zone's rolled-up grid, ready to paint.

    `cells` is row-major and normalized to `[0, 1]`. `peak_ds` is the hottest *decayed*
    cell in deciseconds, carried so the legend can state what full intensity means — a
    heatmap with no units is a picture that cannot be argued with.
    """

    camera_id: CameraId
    zone_id: ZoneId
    grid_w: int
    grid_h: int
    cells: tuple[float, ...]
    peak_ds: float
    minutes: int
    """How many stored minutes went into this view. A thin window should read as thin."""


def roll_up(
    rows: Sequence[HeatmapRow],
    *,
    end: datetime,
    decay: float = DEFAULT_DECAY,
) -> list[HeatmapView]:
    """Sum stored minutes into one decayed, normalized grid per zone.

    `end` is what ages are measured from — the window's end, not wall-now, so a view of
    a past window is not silently decayed into nothing by how long ago it happened.

    A zone whose rows disagree about grid dimensions is not merged: the newest dimensions
    win and older minutes are dropped, because summing a 32x32 into a 64x64 would place
    every historical cell somewhere it does not belong. That only happens across a change
    to the grid constants, and a visibly shorter history is the honest outcome.
    """
    grouped: dict[tuple[CameraId, ZoneId], list[HeatmapRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.camera_id, row.zone_id)].append(row)
    return [_view(zone_rows, end=end, decay=decay) for zone_rows in grouped.values()]


def _view(rows: list[HeatmapRow], *, end: datetime, decay: float) -> HeatmapView:
    newest = max(rows, key=lambda row: row.bucket)
    usable = [row for row in rows if (row.grid_w, row.grid_h) == (newest.grid_w, newest.grid_h)]
    totals = [0.0] * (newest.grid_w * newest.grid_h)
    for row in usable:
        weight = decay ** _age_minutes(row.bucket, end)
        for index, count in enumerate(unpack_counts(row.counts)):
            totals[index] += count * weight
    return HeatmapView(
        camera_id=newest.camera_id,
        zone_id=newest.zone_id,
        grid_w=newest.grid_w,
        grid_h=newest.grid_h,
        cells=normalize(totals),
        peak_ds=max(totals, default=0.0),
        minutes=len(usable),
    )


def _age_minutes(bucket: datetime, end: datetime) -> float:
    """How many minutes before the window's end this bucket sits. Never negative."""
    return max((end - bucket).total_seconds() / 60.0, 0.0)


def normalize(totals: Sequence[float]) -> tuple[float, ...]:
    """Map deciseconds to `[0, 1]`, logarithmically, clipped at the p99 occupied cell.

    An all-zero grid normalizes to all zeros rather than dividing by one: an empty zone
    renders uniformly cold, which is a real answer and not an error (§10).
    """
    clip = _clip_at(totals)
    if clip <= 0.0:
        return tuple(0.0 for _ in totals)
    scale = math.log1p(clip)
    return tuple(min(math.log1p(max(total, 0.0)) / scale, 1.0) for total in totals)


def _clip_at(totals: Sequence[float]) -> float:
    occupied = sorted(total for total in totals if total > 0.0)
    if not occupied:
        return 0.0
    index = min(int(CLIP_QUANTILE * len(occupied)), len(occupied) - 1)
    return occupied[index]
