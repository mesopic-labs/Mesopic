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

from muster.types import FrameTs


class FrameSampler:
    """Timestamp-gated admission control for one camera."""

    def __init__(self, target_fps: float) -> None:
        raise NotImplementedError

    def is_due(self, ts: FrameTs) -> bool:
        """Whether a frame captured at `ts` should be detected."""
        raise NotImplementedError

    def set_target_fps(self, target_fps: float) -> None:
        """Adjust the cadence — how the supervisor applies backpressure."""
        raise NotImplementedError
