"""The engine's shared vocabulary.

Domain-meaningful values are ``NewType`` aliases so a ``CameraId`` can never be passed
where a ``ZoneId`` is expected and ``mypy --strict`` catches the mix-up. Everything the
pipeline hands between modules is defined here, once, and imported at the top of every
file that needs it (engine-architecture.md §3).

Two conventions this module exists to make unbreakable:

* **Time is UTC.** ``FrameTs`` and ``MinuteBucket`` are timezone-aware UTC datetimes.
  The site's ``timezone`` config is display-only.
* **Geometry is normalized.** The detector and tracker work in ``PixelPoint`` space of
  the (possibly downscaled) inference frame; everything downstream works in
  ``NormPoint`` space, ``[0.0, 1.0]``, origin top-left, so zones and lines authored once
  survive a resolution change. The conversion happens at exactly one boundary — the
  tracker (engine-architecture.md §7) — and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import NewType

import numpy as np
from numpy.typing import NDArray

# --- Identities -------------------------------------------------------------

CameraId = NewType("CameraId", str)
"""Stable per-camera identity, from config."""

ZoneId = NewType("ZoneId", str)
"""A polygon region on a camera's frame."""

LineId = NewType("LineId", str)
"""A directional counting line."""

TrackId = NewType("TrackId", int)
"""Tracker-assigned. Unique per camera *per run* only — never an identity."""

SiteId = NewType("SiteId", str)
"""One engine == one site. Also the cloud-sync identity."""

ScopeId = NewType("ScopeId", str)
"""The zone or line a metric is scoped to. ``None`` for camera-wide metrics."""

ClipId = NewType("ClipId", str)
"""A ground-truth clip. Names a manifest in the repository, never the footage itself —
the bytes live outside it and are resolved at run time (MK.2)."""

# --- Time -------------------------------------------------------------------

FrameTs = NewType("FrameTs", datetime)
"""Capture time of a sampled frame. Timezone-aware, UTC."""

MinuteBucket = NewType("MinuteBucket", datetime)
"""``FrameTs`` floored to the minute. Timezone-aware, UTC."""

# --- Space ------------------------------------------------------------------

PixelPoint = tuple[int, int]
"""``(x, y)`` in inference-resolution pixels. Detector/tracker only."""

NormPoint = tuple[float, float]
"""``(x, y)`` in ``[0.0, 1.0]``, origin top-left. Everything downstream."""

PixelBox = tuple[int, int, int, int]
"""``(x1, y1, x2, y2)`` in inference-resolution pixels."""

GridCell = tuple[int, int]
"""``(column, line)`` in a heatmap grid, discretized from a ``NormPoint``.

Frame-normalized, never zone-relative: ``heatmap_minute`` stores ``grid_w``/``grid_h``
and no origin, so a zone-relative grid could not be rendered back after the calibration
editor moved the zone (algorithms.md §10, engine-architecture.md §11).
"""

BgrImage = NDArray[np.uint8]
"""HxWx3 BGR. Its lifetime is one pipeline tick — see ``DecodedFrame``."""


# --- Enumerations -----------------------------------------------------------


class SourceKind(StrEnum):
    """Where a camera's tracks come from (engine-architecture.md §4)."""

    RTSP = "rtsp"
    ONVIF = "onvif"
    FRIGATE = "frigate"


class ZoneRole(StrEnum):
    """What a zone means. ``STAFF`` drives the staff-vs-customer adjacency."""

    AREA = "area"
    QUEUE = "queue"
    STAFF = "staff"


class Direction(StrEnum):
    """The semantic label for a line's ``+1`` crossing sense."""

    IN = "in"
    OUT = "out"


class EventKind(StrEnum):
    """The raw events analytics emits (engine-architecture.md §8).

    Most are *transitions* — something changed, here is what. Two are not, and the
    difference is worth knowing before adding a third (ADR-0016):

    * ``OCCUPANCY_SAMPLE`` is a *state*, emitted per zone per tick whether or not
      anything changed. It is the only dense kind, and it exists because a reducer that
      sees transitions alone cannot report a minute in which nobody moved.
    * ``ZONE_CONFIRMED`` is sparse but derived: the moment a residency passes
      ``dwell_min_s``, which the sample's bare count cannot attribute to a track.
    """

    LINE_CROSS = "line_cross"
    ZONE_ENTER = "zone_enter"
    ZONE_EXIT = "zone_exit"
    ZONE_CONFIRMED = "zone_confirmed"
    OCCUPANCY_SAMPLE = "occupancy_sample"
    DWELL_SAMPLE = "dwell_sample"
    HEATMAP_HIT = "heatmap_hit"


class MetricName(StrEnum):
    """The metric vocabulary. Shared contract with the cloud (cloud-architecture.md §4).

    The core six plus the two cheap adjacencies. Anything not on this list does not
    exist in v1 — scope is locked (engine-architecture.md §1).

    Occupancy is three series rather than one (algorithms.md §6.1) and queue length
    follows the same split: the confirmed series lags by ``dwell_min_s``, which is
    harmless for a mean and wrong for a peak, because peaks form in exactly the
    fast-turnover moments the confirmation rule suppresses. So ``OCCUPANCY`` carries the
    mean and ``OCCUPANCY_RAW`` the peak. The third, ``net_occupancy``, is deliberately
    absent — it is opt-in, it is the only metric in the set that accumulates drift, and
    whether it should exist in v1 at all is still open (ADR-0016).

    ``TRANSACTIONS`` is conversion's numerator, carried as its own series because a ratio
    cannot be rolled up from ratios: an hour's conversion is ``Σ txns / Σ footfall``, and
    nothing downstream can recover the numerator from a number that has already been
    divided (cloud-architecture.md §3.4). The cloud learns this name *before* an engine
    emits it — the reverse refuses the whole batch with a 422.
    """

    FOOTFALL = "footfall"
    OCCUPANCY = "occupancy"
    OCCUPANCY_RAW = "occupancy_raw"
    QUEUE_LEN = "queue_len"
    QUEUE_LEN_RAW = "queue_len_raw"
    DWELL_SECONDS = "dwell_seconds"
    LINE_CROSS = "line_cross"
    CONVERSION = "conversion"
    TRANSACTIONS = "transactions"
    HEATMAP = "heatmap"


class CameraState(StrEnum):
    """Ingest state machine (engine-architecture.md §4). Reported on ``/healthz``."""

    CONNECT = "connect"
    STREAMING = "streaming"
    STALLED = "stalled"
    BACKOFF = "backoff"
    DISABLED = "disabled"


class Runtime(StrEnum):
    """Which native runtime executes the detector's graph (ADR-0012).

    ``ORT_CPU`` is the guaranteed path: it is a hard dependency, it runs on Intel, AMD and
    ARM, and it is what the N100 perf gate is measured against. Everything else is an
    accelerator — selected when explicitly asked for, never required.
    """

    ORT_CPU = "ort-cpu"
    OPENVINO = "openvino"


# --- Pipeline payloads ------------------------------------------------------
#
# Every payload is frozen: nothing downstream may mutate what it was handed. Slots
# because these are allocated per frame on a CPU budget.


@dataclass(frozen=True, slots=True, eq=False)
class DecodedFrame:
    """One decoded frame, on its way to the detector.

    **Its lifetime is one pipeline tick.** It is a local variable inside the camera
    worker loop: never stored, never buffered across ticks, never sent anywhere. This is
    the "frames never hit disk" invariant in its most concrete form (ADR-0005).

    ``eq=False`` because ``image`` is an ndarray and structural equality on frames is
    meaningless here — identity comparison is what callers actually want.
    """

    camera_id: CameraId
    ts: FrameTs
    image: BgrImage
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Detection:
    """One person box from the detector, in inference-frame pixel space."""

    box: PixelBox
    score: float


@dataclass(frozen=True, slots=True)
class Track:
    """A tracked person at one instant, in normalized space.

    ``foot_point`` is the bottom-centre of the bounding box — approximately where the
    person meets the floor. It is *the* geometric convention: every metric is computed
    from it, never from the box centroid. Defined once in algorithms.md; the engine
    imports that definition and never re-derives it.
    """

    camera_id: CameraId
    track_id: TrackId
    ts: FrameTs
    foot_point: NormPoint
    score: float
    time_since_update: int = 0
    """Ticks since this track last matched a real detection.

    ``0`` is an observed position. ``> 0`` is dead reckoning: the track is being
    predicted with no observation behind it. Geometry may use coasted positions for
    path continuity but must refuse to emit an event from a segment whose endpoints
    are *both* unobserved (algorithms.md §3.4, engine-architecture.md §8).
    """

    # There is deliberately no `is_staff` here. The staff tag is a polygon test, and the
    # import-linter contract keeps the tracker away from geometry — so a field on the
    # tracker's own output could never be anything but `False`, which is worse than
    # absent for whoever reads it next. `GeometryAnalytics` owns the tag (ADR-0021).


@dataclass(frozen=True, slots=True)
class RawEvent:
    """What analytics emits and the aggregator reduces (engine-architecture.md §8, §10).

    Small by construction: this is what crosses the process boundary from a camera
    worker to the supervisor. Never a frame, never a crop, never a pixel.
    """

    camera_id: CameraId
    ts: FrameTs
    kind: EventKind
    track_id: TrackId | None
    """The track the event is about. ``None`` for ``OCCUPANCY_SAMPLE``, which is a count
    of a zone rather than a fact about anyone in it."""
    zone_id: ZoneId | None = None
    line_id: LineId | None = None
    direction: int | None = None
    """``+1`` / ``-1`` for line crossings, matching the line's ``positive_dir``."""
    value: float | None = None
    """The magnitude carried by the kinds that have one: a completed dwell in seconds for
    ``DWELL_SAMPLE`` (algorithms.md §7), and the number of tracks inside the zone for
    ``OCCUPANCY_SAMPLE``.

    ``None`` everywhere else, because most events *are* the fact: a crossing has a
    direction and a zone entry has neither size nor duration. Note the ``events`` table
    has no matching column — dwell samples are derived inside the aggregator and reduced
    there, never appended to the raw log — so persisting one would need a migration.
    """
    confirmed_value: float | None = None
    """The confirmed half of a count that has two halves.

    Set on ``OCCUPANCY_SAMPLE`` alongside ``value``: how many of the tracks inside the
    zone have been there for at least ``dwell_min_s``. Both travel on one event because
    algorithms.md §6.1 needs both — the peak reads the raw count, the mean reads this one
    — and two events could disagree about the same instant.
    """
    staff_value: float | None = None
    """How many of ``value`` were staff. Set on ``OCCUPANCY_SAMPLE`` only.

    A sampled state is the one shape the ``is_staff`` flag cannot express: the sample
    counts a *zone* and names no track, so there is nothing to filter on and the split
    has to ride the sample itself. Every other kind is a fact about one track and carries
    ``is_staff`` instead (ADR-0016, ADR-0021).
    """
    staff_confirmed_value: float | None = None
    """The staff half of ``confirmed_value``.

    A second field rather than a ratio applied to one, because occupancy is two series on
    two different bases — the peak reads the raw count and the mean reads the confirmed
    one — and a staff sub-count has to sit on the same basis as the series it belongs to
    or it is a number derived from a different population.
    """
    dt_s: float | None = None
    """The wall-clock interval this sample represents, in seconds — the gap since the
    previous sampled frame on the same camera.

    Set on ``OCCUPANCY_SAMPLE`` and on ``HEATMAP_HIT``, and nowhere else. It is what
    makes a state metric honest: the adaptive sampler slows down when the scene is busy,
    so an estimator that averages over *samples* systematically under-weights the rush
    (algorithms.md §0.6).
    """
    cell: GridCell | None = None
    """Which heatmap cell the foot-point fell in. Set on ``HEATMAP_HIT`` and nowhere else.

    Analytics discretizes rather than shipping the foot-point, because the grid is a
    property of the site's geometry and the aggregator has none. The ``events`` table has
    no column for it — like a dwell's duration, a hit is folded in memory and never
    appended to the raw log, which is also what keeps the densest event kind off disk.
    """
    is_staff: bool = False


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One aggregated bucket — the durable unit and the only thing that syncs.

    Natural key is ``(camera_id, metric, scope_id, bucket)``; writes are idempotent
    upserts on it, so replaying a bucket after a crash cannot double-count
    (engine-architecture.md §10, §11).
    """

    camera_id: CameraId
    bucket: MinuteBucket
    metric: MetricName
    scope_id: ScopeId | None
    value: float
    staff_value: float | None = None
    sample_count: int = 0


@dataclass(frozen=True, slots=True)
class HeatmapRow:
    """One zone's density grid for one minute — the other durable unit.

    Deliberately not a ``MetricRow``. A grid is a blob on a different natural key
    (``camera_id, zone_id, bucket``) in a different table, and widening ``MetricRow`` to
    carry one would put a case every scalar metric never uses into every plugin's
    signature. Keeping ``metrics_minute`` scalar is why engine-architecture.md §11 split
    the tables in the first place.

    ``counts`` is packed little-endian ``uint16``, ``grid_w * grid_h`` cells in row-major
    order, in **deciseconds of foot-point presence** — never a frame count (§10).
    """

    camera_id: CameraId
    bucket: MinuteBucket
    zone_id: ZoneId
    grid_w: int
    grid_h: int
    counts: bytes
