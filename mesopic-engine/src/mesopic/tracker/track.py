"""One track's state machine (algorithms.md §3.4).

The only mutable state in the tracker package. Everything that crosses a module
boundary is a frozen `mesopic.types.Track`; this record stays inside.

**Death is wall-clock.** §3.4 expresses track memory as `track_memory_s * target_fps`
frames, but its stated intent is that memory stay stable in wall-clock terms as fps
flexes. Every tick already carries a UTC timestamp, so comparing timestamps directly is
that intent with the conversion -- and its whole class of off-by-fps bugs -- removed.

The lifecycle below (tentative -> confirmed -> lost -> expired, `hits`, `n_init`,
`time_since_update`) follows the state machine common to the motion-only MOT
literature (SORT/DeepSORT/ByteTrack all share it); the names are that field's
conventional vocabulary, not lineage from any one implementation.
"""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from mesopic.tracker.kalman import BoxKalmanFilter
from mesopic.types import FrameTs, TrackId


class TrackState(StrEnum):
    """Where a track sits in its lifecycle."""

    TENTATIVE = "tentative"
    """Born but unproven. Never reaches geometry; dies on its first miss."""

    CONFIRMED = "confirmed"
    """Matched `n_init` times. Reaches geometry."""

    LOST = "lost"
    """Confirmed, then missed. Dead-reckoned until `track_memory_s` runs out."""


class TrackRecord:
    """A single track: its filter, its identity, and its lifecycle bookkeeping."""

    def __init__(
        self,
        *,
        track_id: TrackId,
        box: NDArray[np.float64],
        score: float,
        ts: FrameTs,
        n_init: int,
    ) -> None:
        self.track_id = track_id
        self.score = score
        self.time_since_update = 0
        self.last_observed_ts = ts
        self.hits = 1
        self._n_init = n_init
        self._filter = BoxKalmanFilter(box)
        self.state = TrackState.CONFIRMED if self.hits >= n_init else TrackState.TENTATIVE

    @property
    def box(self) -> NDArray[np.float64]:
        """Current estimate as `(x1, y1, x2, y2)`."""
        return self._filter.box

    def predict(self, dt_s: float) -> None:
        """Advance the motion estimate to this tick."""
        self._filter.predict(dt_s)

    def mark_matched(self, box: NDArray[np.float64], score: float, ts: FrameTs) -> None:
        """Fold in an observation, promoting or reviving the track as earned."""
        self._filter.update(box)
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        # Never let an out-of-order frame (an RTSP reconnect glitch) rewind this: `_gate`
        # and `is_expired` both read `last_observed_ts` to size a gap, and a rewind would
        # shrink that gap -- tightening the gate exactly when a stray late frame should
        # have no effect at all -- and could expire the track early. `bytetrack.py`'s
        # `_elapsed` guards the tracker-wide clock the same way, for the same reason.
        self.last_observed_ts = max(ts, self.last_observed_ts)
        # `CONFIRMED` only ever transitions to `LOST` after `hits >= self._n_init` already
        # held (see `mark_missed`), and `hits` only grows -- so `hits >= self._n_init` is
        # already true on every revival from `LOST`, making a separate `state is LOST`
        # check redundant with the counter check on the very next line.
        if self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        """Record that no detection matched this tick."""
        self.time_since_update += 1
        if self.state is TrackState.CONFIRMED:
            self.state = TrackState.LOST

    def is_expired(self, now: FrameTs, track_memory_s: float) -> bool:
        """Whether this track should be deleted.

        An unproven track dies on its first miss: a false birth is a permanent
        overcount, so identity creation is held to a stricter standard than identity
        continuation.
        """
        if self.state is TrackState.TENTATIVE and self.time_since_update > 0:
            return True
        return now - self.last_observed_ts > timedelta(seconds=track_memory_s)
