"""ByteTrack wrapper — the default tracker, benchmark-gated (ADR-0014).

This module owns two things nothing else may duplicate:

* **Foot-point derivation** — bottom-centre of the bounding box, never the centroid.
* **Pixel -> normalized conversion** — the *only* place in the engine it happens.

Implements P1.5.
"""

from __future__ import annotations

from itertools import count
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from muster.tracker.cost import CostFunction, ceiling_for, centre_distance_cost
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
        max_cost: float = 0.4,
        max_cost_low: float = 0.25,
        n_init: int = 2,
        track_memory_s: float = 2.0,
        cost: CostFunction = centre_distance_cost,
        gate_widening_per_second: float = 1.5,
        gate_form: Literal["additive", "saturating"] = "additive",
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
        # so a flat gate refuses exactly the step that matters most.
        #
        # `cost=centre_distance_cost, max_cost=0.4, kappa=1.5, gate_form="additive"` is
        # this cost's own measured operating point, not a scaled-down guess from GIoU's
        # numbers -- its cost is on a different scale entirely (normalized centre
        # distance plus a scale-agreement penalty, not `1 - similarity`). ADR-0014
        # records the full benchmark this came from: every candidate cost tuned at its
        # own operating point, both gate forms, swept across the product's fps/n_init/
        # scenario grid under detection corruption. `centre_distance_cost` won on
        # identity stability, by a margin the ADR confirms is statistically significant,
        # and its additive gate form beat the saturating alternative once both were
        # correctly evaluated. `n_init=2` is unchanged from the pre-benchmark default --
        # `n_init=1` was measured and was not a clear enough win to justify moving off it
        # (ADR-0014 Decision #4's own standard).
        #
        # HONEST CAVEAT, recorded in ADR-0014 and not smoothed over here:
        # `centre_distance_cost`'s advantage is not uniform across fps. At the product's
        # 1 fps floor specifically, plain `iou_cost` scores better on the primary
        # identity-stability metric; `centre_distance_cost` wins decisively at 2/3/5 fps
        # and in aggregate over the full grid, but 1 fps is a real exception, not a
        # rounding error, and the ADR says so rather than claiming a universal win.
        self._gate_widening_per_second = gate_widening_per_second
        # Gate FORM (ADR-0014): see `_gate`'s docstring for what "additive" vs
        # "saturating" mean and why the latter needs a cost ceiling.
        self._gate_form = gate_form
        self._ceiling = self._resolve_ceiling(gate_form, cost)
        self._tracks: list[TrackRecord] = []
        self._ids = count(1)
        self._last_ts: FrameTs | None = None

    @staticmethod
    def _resolve_ceiling(gate_form: str, cost: CostFunction) -> float | None:
        """The cost's ceiling, required only by the saturating gate form.

        Resolved eagerly so a misconfiguration (a saturating gate paired with an
        unbounded cost) fails at construction, not on the first call to `update`. Every
        cost `cost.py`'s `COSTS` currently registers has a finite ceiling, so this guard
        is not reachable today, but stays here for whatever candidate joins `COSTS` next.
        """
        if gate_form != "saturating":
            return None
        ceiling = ceiling_for(cost)
        if ceiling is None:
            msg = (
                "gate_form='saturating' requires a cost function with a finite ceiling; "
                "this cost has none -- use 'additive' instead"
            )
            raise ValueError(msg)
        return ceiling

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

        KNOWN LIMITATION: `dt_s` here is the TICK's interval, while `_gate` widens by
        each track's own OBSERVED gap -- the two diverge only for a coasted track (a
        fresh match has gap == dt_s by definition). Harmless for `giou_cost`, which
        ignores `dt_s` entirely, but `expansion_iou_cost` inflates boxes by a
        dt_s-scaled margin, so for a reacquired track it will under-inflate relative to
        the gap the gate actually bridges. Bounded: within one fps regime fps is fixed,
        so only coasted tracks (not the common case) diverge. Not fixed here on purpose
        -- doing it properly makes the expansion per-pair rather than per-box, a real
        `cost.py` redesign that is not landing unreviewed against the M0 date. ADR-0014
        records this as an open caveat on `expansion_iou_cost`'s benchmark standing.
        """
        return self.cost(np.array([t.box for t in tracks]), det_boxes, dt_s)

    def _gate(self, tracks: list[TrackRecord], max_cost: float, ts: FrameTs) -> NDArray[np.float64]:
        """Per-track accept threshold, widened by the gap since each track was last
        OBSERVED (not since the last tick): a track missed for three ticks must bridge
        three ticks' worth of displacement, so a reacquisition after a multi-tick
        occlusion is gated proportionally to what it needs to bridge, not to a single
        missed frame (ADR-0014 Decision #1: dt is the gap being bridged, not the tick
        interval).

        Two forms (`gate_form`, ADR-0014's gate-form decision):

        * `additive`: `max_cost + kappa * gap`. ADR-0014 Decision #1's own formula.
          Unbounded, so past some gap it exceeds a bounded cost's ceiling and refuses
          nothing at all -- a dead zone, not a typo (see `__init__`).
        * `saturating`: `ceiling - (ceiling - max_cost) * exp(-kappa * gap)`. Starts at
          `max_cost` when `gap == 0`, same as the additive form, then approaches (never
          reaches) the cost's ceiling as the gap grows -- so there is always some margin
          left to refuse an implausible match, however large the gap.
        """
        gaps = np.array(
            [max((ts - t.last_observed_ts).total_seconds(), 0.0) for t in tracks],
            dtype=np.float64,
        )
        if self._gate_form == "additive":
            gate: NDArray[np.float64] = max_cost + self._gate_widening_per_second * gaps
            return gate
        if self._ceiling is None:
            # Unreachable: __init__ raises before a saturating tracker with no ceiling
            # can exist. Guards mypy's narrowing rather than expressing a real runtime
            # possibility.
            msg = "saturating gate requires a ceiling; __init__ should have refused this"
            raise RuntimeError(msg)
        saturating: NDArray[np.float64] = self._ceiling - (self._ceiling - max_cost) * np.exp(
            -self._gate_widening_per_second * gaps
        )
        return saturating

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
