"""ByteTrack wrapper — the default tracker, benchmark-gated (ADR-0014).

This module owns two things nothing else may duplicate:

* **Foot-point derivation** — bottom-centre of the bounding box, never the centroid.
* **Pixel -> normalized conversion** — the *only* place in the engine it happens.

Implements P1.5.
"""

from __future__ import annotations

from itertools import count

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from muster.tracker.cost import CostFunction, giou_cost
from muster.tracker.track import TrackRecord, TrackState
from muster.types import (
    DecodedFrame,
    Detection,
    FrameTs,
    NormPoint,
    PixelBox,
    Track,
    TrackId,
)

_BIRTH_MARGIN = 0.1
"""How much more confident a detection must be to *create* an identity than to sustain
one (algorithms.md §3.2). A false birth is a permanent overcount; a false sustain
self-corrects."""


def foot_point(box: PixelBox, frame_width: int, frame_height: int) -> NormPoint:
    """Bottom-centre of `box`, in normalized `[0, 1]` coordinates.

    The single most important geometric convention in the engine: approximately where
    the person meets the floor, which is what every metric is actually about. Defined in
    algorithms.md; this is its one implementation.

    Args:
        box: `(x1, y1, x2, y2)` in inference-frame pixels.
        frame_width: Inference-frame width in pixels.
        frame_height: Inference-frame height in pixels.

    Returns:
        `(x, y)` in `[0.0, 1.0]`, origin top-left.

    Raises:
        ValueError: If either frame dimension is not positive.
    """
    if frame_width <= 0 or frame_height <= 0:
        msg = f"frame dimensions must be positive, got {frame_width}x{frame_height}"
        raise ValueError(msg)
    x1, _, x2, y2 = box
    x = (x1 + x2) / 2.0 / frame_width
    y = y2 / frame_height
    return (min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))


class ByteTrackTracker:
    """A `Tracker` implementing BYTE association with a pluggable cost (ADR-0014)."""

    def __init__(
        self,
        *,
        track_thresh: float = 0.5,
        max_cost: float = 0.8,
        max_cost_low: float = 0.5,
        n_init: int = 2,
        track_memory_s: float = 2.0,
        cost: CostFunction = giou_cost,
        gate_widening_per_second: float = 2.0,
    ) -> None:
        # `max_cost` gates the association COST (1 - similarity), not the similarity.
        # The upstream name (`match_thresh`) invites exactly the wrong reading, and that
        # misreading was a real bug in the design drafts (algorithms.md §3.3). Stage 1
        # is the LOOSE gate and stage 2 the tight one -- also the opposite of what the
        # names suggest.
        self.cost = cost
        self._track_thresh = track_thresh
        self._det_thresh = track_thresh + _BIRTH_MARGIN
        self._max_cost = max_cost
        self._max_cost_low = max_cost_low
        self._n_init = n_init
        self._track_memory_s = track_memory_s
        # kappa (ADR-0014 Decision #1): "accept iff cost < max_cost + kappa * dt_s". The
        # gate must widen with the sampling gap because displacement grows with it -- a
        # newly-born track has no velocity estimate yet and predicts "it didn't move",
        # so a flat gate refuses exactly the step that matters most. 1.0 measured to
        # REFUSE a brisk 2.0 m/s walker (kalman.py's _MAX_WALK_SPEED_MS -- two modules
        # must not silently disagree about how fast a person may walk) at 2 and 3 fps.
        # Fix round 1 raised this to 1.5 against a pixel scale that was NOT derived from
        # kalman.py's own metres-to-pixels formula (scale = height_px / _PERSON_HEIGHT_M)
        # and turned out to understate the true displacement -- 1.5 still REFUSES that
        # same walker at 3 fps once computed honestly (cost 1.3245 vs gate 1.3000,
        # margin -0.0245; see test_a_brisk_walker_is_admitted_at_the_worst_case_fps and
        # the kappa table in task-7-report.md's fix round 2). 2.0 clears 1/2/3/5 fps
        # with the corrected scale (worst-case margin +0.1189 at 5 fps). kappa is a
        # SWEPT axis, not a tuned constant: raising it trades birth-survival against
        # ID-swap risk -- the tradeoff ADR-0014 names -- and only the Task 9 sweep can
        # settle where it should sit.
        self._gate_widening_per_second = gate_widening_per_second
        self._tracks: list[TrackRecord] = []
        self._ids = count(1)
        self._last_ts: FrameTs | None = None

    def update(self, frame: DecodedFrame, detections: list[Detection]) -> list[Track]:
        """Advance the tracker one tick and return the currently live tracks."""
        dt_s = self._elapsed(frame.ts)
        for track in self._tracks:
            track.predict(dt_s)

        high = [d for d in detections if d.score >= self._track_thresh]
        low = [d for d in detections if d.score < self._track_thresh]

        unmatched, claimed = self._associate(self._tracks, high, dt_s, self._max_cost, frame.ts)
        # Stage 2 may only SUSTAIN an identity that already exists, never confirm one.
        # A false birth is a permanent overcount, so creating an identity stays on the
        # strict path (a high-confidence match, or det_thresh at birth) even during
        # recovery -- an unproven track offered only a low-confidence detection falls
        # straight through to a miss and dies, exactly as if nothing had matched at all.
        recoverable = [t for t in unmatched if t.state is not TrackState.TENTATIVE]
        unrecoverable = [t for t in unmatched if t.state is TrackState.TENTATIVE]
        still_unmatched, _ = self._associate(recoverable, low, dt_s, self._max_cost_low, frame.ts)

        for track in [*still_unmatched, *unrecoverable]:
            track.mark_missed()
        # Only detections no track claimed may start one. Anything else double-counts
        # the same person as both a continuation and a birth.
        self._birth([d for i, d in enumerate(high) if i not in claimed], frame.ts)
        self._tracks = [t for t in self._tracks if not t.is_expired(frame.ts, self._track_memory_s)]
        return [
            self._publish(t, frame) for t in self._tracks if t.state is not TrackState.TENTATIVE
        ]

    def _elapsed(self, ts: FrameTs) -> float:
        """Seconds since the previous tick. The first tick has no gap to advance over.

        A frame arriving out of order (an RTSP reconnect glitch, not a rare corner
        case in practice) must degrade the tracker, not crash the camera worker: a
        dropped or reordered frame should cost a missed tick, never a process. Clamp
        the gap to zero -- "no time passed" -- rather than raising, and never let
        `_last_ts` move backward: rewinding it would silently inflate the *next*
        well-ordered frame's gap by however early this one arrived.
        """
        previous = self._last_ts
        if previous is None or ts > previous:
            self._last_ts = ts
        return 0.0 if previous is None else max((ts - previous).total_seconds(), 0.0)

    def _associate(
        self,
        tracks: list[TrackRecord],
        detections: list[Detection],
        dt_s: float,
        max_cost: float,
        ts: FrameTs,
    ) -> tuple[list[TrackRecord], set[int]]:
        """Match `tracks` to `detections`.

        Returns:
            The tracks left unmatched, and the indices of the detections that were
            claimed. The caller needs the second half to keep a detection from both
            continuing a track and starting a new one.
        """
        if not tracks or not detections:
            return tracks, set()
        det_boxes = np.array([d.box for d in detections], dtype=np.float64)
        costs = self._cost_matrix(tracks, det_boxes, dt_s)
        gate = self._gate(tracks, max_cost, ts)

        rows, cols = linear_sum_assignment(costs)
        matched_tracks: set[int] = set()
        claimed_dets: set[int] = set()
        for row, col in zip(rows, cols, strict=True):
            if costs[row, col] >= gate[row]:
                continue
            tracks[row].mark_matched(det_boxes[col], detections[col].score, ts)
            matched_tracks.add(row)
            claimed_dets.add(int(col))
        return [t for i, t in enumerate(tracks) if i not in matched_tracks], claimed_dets

    def _cost_matrix(
        self, tracks: list[TrackRecord], det_boxes: NDArray[np.float64], dt_s: float
    ) -> NDArray[np.float64]:
        """Pairwise cost, `(len(tracks), len(det_boxes))`.

        KNOWN LIMITATION (fix round 2, item 6): `dt_s` here is the TICK's interval,
        while `_gate` widens by each track's own OBSERVED gap -- the two diverge only
        for a coasted track (a fresh match has gap == dt_s by definition). Harmless for
        `giou_cost`, which ignores `dt_s` entirely, but `expansion_iou_cost` (a Task 9
        sweep candidate) inflates boxes by a dt_s-scaled margin, so for a reacquired
        track it will under-inflate relative to the gap the gate actually bridges.
        Bounded: within one sweep cell fps is fixed, so only coasted tracks (not the
        common case) diverge. Not fixed here on purpose -- doing it properly makes the
        expansion per-pair rather than per-box, a real `cost.py` redesign that is not
        landing unreviewed against the M0 date. Task 9 must carry this caveat into the
        ADR if `expansion_iou_cost` is the sweep's chosen candidate.
        """
        return self.cost(np.array([t.box for t in tracks]), det_boxes, dt_s)

    def _gate(self, tracks: list[TrackRecord], max_cost: float, ts: FrameTs) -> NDArray[np.float64]:
        """Per-track accept threshold: `max_cost + kappa * (gap since last observed)`.

        Widens by the gap since each track was last OBSERVED, not since the last tick:
        a track missed for three ticks must bridge three ticks' worth of displacement,
        so a reacquisition after a multi-tick occlusion is gated proportionally to what
        it needs to bridge, not to a single missed frame (ADR-0014 Decision #1: dt is
        the gap being bridged, not the tick interval).
        """
        gaps = np.array(
            [max((ts - t.last_observed_ts).total_seconds(), 0.0) for t in tracks],
            dtype=np.float64,
        )
        gate: NDArray[np.float64] = max_cost + self._gate_widening_per_second * gaps
        return gate

    def _birth(self, unclaimed: list[Detection], ts: FrameTs) -> None:
        """Start tracks from confident detections no existing track claimed."""
        for detection in unclaimed:
            if detection.score < self._det_thresh:
                continue
            self._tracks.append(
                TrackRecord(
                    track_id=TrackId(next(self._ids)),
                    box=np.array(detection.box, dtype=np.float64),
                    score=detection.score,
                    ts=ts,
                    n_init=self._n_init,
                )
            )

    def _publish(self, track: TrackRecord, frame: DecodedFrame) -> Track:
        """Convert to the frozen, normalized payload geometry consumes."""
        return Track(
            camera_id=frame.camera_id,
            track_id=track.track_id,
            ts=frame.ts,
            foot_point=self._foot_point(track.box, frame),
            score=track.score,
            time_since_update=track.time_since_update,
        )

    @staticmethod
    def _foot_point(box: NDArray[np.float64], frame: DecodedFrame) -> NormPoint:
        x1, y1, x2, y2 = (round(v) for v in box)
        return foot_point((x1, y1, x2, y2), frame.width, frame.height)
