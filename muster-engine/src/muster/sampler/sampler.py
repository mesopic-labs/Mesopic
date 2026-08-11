"""Adaptive frame sampling — the single biggest performance lever in the engine.

Target is ~2-5 fps *effective* (frames actually detected), not frames off the wire. The
core six tolerate that because counting, dwell, and queue are integrals over time, not
per-frame events (ADR-0003).

Two mechanisms, both of which matter:

1. **Timestamp-gated admission.** A per-camera `next_due` advances by `1 / target_fps`
   and is driven off the frame's *capture* timestamp, so a burst of buffered frames after
   a reconnect cannot cause a detection storm.
2. **Backpressure.** When the supervisor's event queue fills, the *sampler* sheds load —
   never the decoder — so the pipeline degrades in fps rather than in correctness.

Kept as its own module precisely so the policy is testable with a fake clock and no
camera. P1.3 ships the fixed-rate gate; the adaptive controller lands with P2.7.
"""

from __future__ import annotations

import math
from datetime import timedelta

from muster.types import FrameTs


def _period_for(target_fps: float) -> timedelta:
    """The minimum spacing between two admitted frames.

    Raises:
        ValueError: If `target_fps` is not a positive, finite rate.
    """
    if not math.isfinite(target_fps) or target_fps <= 0.0:
        msg = f"target_fps must be a positive, finite rate, got {target_fps!r}"
        raise ValueError(msg)
    return timedelta(seconds=1.0 / target_fps)


class FrameSampler:
    """Timestamp-gated admission control for one camera.

    Stateful, and deliberately not thread-safe: one sampler belongs to one camera worker
    and is driven from that worker's loop alone.
    """

    def __init__(self, target_fps: float) -> None:
        self._period = _period_for(target_fps)
        self._last_admitted: FrameTs | None = None

    def is_due(self, ts: FrameTs) -> bool:
        """Whether a frame captured at `ts` should be detected.

        Consumes as well as answers: a `True` re-bases the gate on `ts`, so the caller
        must ask exactly once per frame. Re-basing on the admitted frame rather than
        advancing by whole periods is what keeps a reconnect cheap — an outage banks no
        credit to spend on the buffered frames that follow it.

        Args:
            ts: The frame's capture timestamp. Never its arrival time: the two diverge
                by exactly the backlog that would cause a storm.

        Returns:
            `True` if the frame should be detected.

        Raises:
            ValueError: If `ts` is not timezone-aware.
        """
        if ts.tzinfo is None:
            msg = f"frame timestamps must be timezone-aware UTC, got {ts!r}"
            raise ValueError(msg)
        if self._last_admitted is not None:
            elapsed = ts - self._last_admitted
            # The lower bound re-arms the gate on a backwards step larger than one
            # period: a camera clock that resets would otherwise silence this sampler
            # until real time caught up. Smaller backwards steps are ordinary arrival
            # jitter, and fall inside the rate limit anyway.
            if -self._period <= elapsed < self._period:
                return False
        self._last_admitted = ts
        return True

    def set_target_fps(self, target_fps: float) -> None:
        """Adjust the cadence — how the supervisor applies backpressure.

        Takes effect on the next frame, measured from the last admitted one, so shedding
        bites immediately rather than a period later.

        Raises:
            ValueError: If `target_fps` is not a positive, finite rate.
        """
        self._period = _period_for(target_fps)
