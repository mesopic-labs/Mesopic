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
    """The raw events analytics emits (engine-architecture.md §8)."""

    LINE_CROSS = "line_cross"
    ZONE_ENTER = "zone_enter"
    ZONE_EXIT = "zone_exit"
    DWELL_SAMPLE = "dwell_sample"
    HEATMAP_HIT = "heatmap_hit"


class MetricName(StrEnum):
    """The metric vocabulary. Shared contract with the cloud (cloud-architecture.md §4).

    The core six plus the two cheap adjacencies. Anything not on this list does not
    exist in v1 — scope is locked (engine-architecture.md §1).
    """

    FOOTFALL = "footfall"
    OCCUPANCY = "occupancy"
    QUEUE_LEN = "queue_len"
    DWELL_SECONDS = "dwell_seconds"
    LINE_CROSS = "line_cross"
    CONVERSION = "conversion"
    HEATMAP = "heatmap"


class CameraState(StrEnum):
    """Ingest state machine (engine-architecture.md §4). Reported on ``/healthz``."""

    CONNECT = "connect"
    STREAMING = "streaming"
    STALLED = "stalled"
    BACKOFF = "backoff"
    DISABLED = "disabled"


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
    is_staff: bool = False


@dataclass(frozen=True, slots=True)
class RawEvent:
    """What analytics emits and the aggregator reduces (engine-architecture.md §8, §10).

    Small by construction: this is what crosses the process boundary from a camera
    worker to the supervisor. Never a frame, never a crop, never a pixel.
    """

    camera_id: CameraId
    ts: FrameTs
    kind: EventKind
    track_id: TrackId
    zone_id: ZoneId | None = None
    line_id: LineId | None = None
    direction: int | None = None
    """``+1`` / ``-1`` for line crossings, matching the line's ``positive_dir``."""
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
