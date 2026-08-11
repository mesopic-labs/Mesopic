"""The P1 perf spike: one camera, hard-coded, straight through the pipeline.

This is scaffolding with a job and an expiry date. Its job is to make the M0 gate
(P1.7) measurable — one 1080p stream, decode -> sample -> detect -> track, with the
numbers that gate needs printed where a harness can read them. Its expiry date is
P2.7, when the supervisor and camera workers replace it with the real thing.

Because it exists to be *measured*, two things are deliberate:

* **Every clock and counter is injectable.** Stats are testable without sleeping, and
  a fake clock cannot drift into the numbers the gate is read from.
* **Output is JSON lines on stdout, one object per line.** `muster spike | jq` and
  `muster spike > run.jsonl` both work, and P1.7's harness parses rather than scrapes.

What it prints is ts, fps, track ids and foot-points — never pixels, and never the
RTSP URL, which carries the camera's credentials.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass

from muster.detector.detector import Detector
from muster.errors import StreamDropped
from muster.ingest.source import FrameSource
from muster.sampler.sampler import FrameSampler
from muster.tracker.tracker import Tracker
from muster.types import FrameTs, Track

STATS_PERIOD_S = 1.0
"""How often `--stats` emits, per the P1.6 acceptance criteria."""


def _rss_bytes() -> int:
    """Peak resident set size, in bytes on every platform.

    `ru_maxrss` is kilobytes on Linux and bytes on macOS/BSD — a difference of 1024x in
    the number P1.7 asserts "no OOM" against, so it is normalised here rather than at
    the point someone reads a graph.
    """
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss if sys.platform == "darwin" else max_rss * 1024


@dataclass(frozen=True, slots=True)
class StatsSnapshot:
    """One second's worth of the numbers the M0 gate is read from."""

    effective_fps: float
    mean_latency_ms: float
    max_latency_ms: float
    cpu_percent: float
    rss_bytes: int

    def to_line(self) -> str:
        """One JSON line, tagged so a consumer can tell it from a frame line."""
        return json.dumps(
            {
                "stats": True,
                "fps": round(self.effective_fps, 3),
                "mean_latency_ms": round(self.mean_latency_ms, 3),
                "max_latency_ms": round(self.max_latency_ms, 3),
                "cpu_percent": round(self.cpu_percent, 2),
                "rss_bytes": self.rss_bytes,
            },
            separators=(",", ":"),
        )


class SpikeStats:
    """Rolling per-second pipeline statistics.

    The window is *rolling*, not cumulative: a cumulative mean converges and stops
    moving, which would hide exactly the slow fps decay P1.7's 30-minute soak exists
    to catch.
    """

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        cpu_seconds: Callable[[], float] = time.process_time,
        rss_bytes: Callable[[], int] = _rss_bytes,
    ) -> None:
        self._monotonic = monotonic
        self._cpu_seconds = cpu_seconds
        self._rss_bytes = rss_bytes
        self._latencies: list[float] = []
        self._window_start = monotonic()
        self._window_cpu_start = cpu_seconds()

    def record(self, latency_s: float) -> None:
        """Record one admitted frame's detect+track latency."""
        self._latencies.append(latency_s)

    @property
    def effective_fps(self) -> float:
        """Admitted frames per second so far this window — read without resetting it."""
        elapsed = self._monotonic() - self._window_start
        if elapsed <= 0.0:
            return 0.0
        return len(self._latencies) / elapsed

    def is_due(self) -> bool:
        """Whether a `--stats` line is owed."""
        return (self._monotonic() - self._window_start) >= STATS_PERIOD_S

    def snapshot(self) -> StatsSnapshot:
        """Read the window and start a new one."""
        now = self._monotonic()
        cpu_now = self._cpu_seconds()
        elapsed = now - self._window_start
        cpu_elapsed = cpu_now - self._window_cpu_start

        count = len(self._latencies)
        snapshot = StatsSnapshot(
            effective_fps=count / elapsed if elapsed > 0.0 else 0.0,
            mean_latency_ms=(sum(self._latencies) / count * 1000.0) if count else 0.0,
            max_latency_ms=(max(self._latencies) * 1000.0) if count else 0.0,
            # May exceed 100%: the detector is multi-threaded, and on the N100 that is
            # the number that matters — four cores means a 400% ceiling.
            cpu_percent=(cpu_elapsed / elapsed * 100.0) if elapsed > 0.0 else 0.0,
            rss_bytes=self._rss_bytes(),
        )

        self._latencies = []
        self._window_start = now
        self._window_cpu_start = cpu_now
        return snapshot


def frame_line(*, ts: FrameTs, tracks: list[Track], effective_fps: float) -> str:
    """One admitted frame as a JSON line.

    The payload is the whole privacy story in miniature: a timestamp, a rate, and a
    foot-point per track. No box, no crop, no image — a foot-point is a coordinate,
    and a coordinate is not a person (ADR-0005).
    """
    return json.dumps(
        {
            "ts": ts.isoformat(),
            "fps": round(effective_fps, 3),
            "tracks": [
                {"id": int(track.track_id), "foot": [track.foot_point[0], track.foot_point[1]]}
                for track in tracks
            ],
        },
        separators=(",", ":"),
    )


def run_spike(
    *,
    source: FrameSource,
    sampler: FrameSampler,
    detector: Detector,
    tracker: Tracker,
    stats: SpikeStats | None = None,
    emit_stats: bool = False,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> Generator[str, None, None]:
    """Run one camera through the pipeline, yielding a JSON line per admitted frame.

    A generator rather than a print loop so the caller owns the output stream — which
    is what lets the tests read the lines and P1.7 redirect them to a file. `Generator`
    rather than `Iterator` in the signature is deliberate: closing it early is part of
    the contract, because that is what releases the camera socket.
    """
    stats = stats if stats is not None else SpikeStats()
    try:
        try:
            for frame in source.frames():
                # The sampler consumes as well as answers, so this is asked exactly once
                # per decoded frame (P1.3).
                if not sampler.is_due(frame.ts):
                    continue

                started = perf_counter()
                detections = detector.detect(frame)
                tracks = tracker.update(frame, detections)
                stats.record(latency_s=perf_counter() - started)

                yield frame_line(ts=frame.ts, tracks=tracks, effective_fps=stats.effective_fps)

                if emit_stats and stats.is_due():
                    yield stats.snapshot().to_line()
        except StreamDropped:
            # The normal end of a real run: a camera stream does not finish, it drops.
            # The partial run's numbers are still the measurement P1.7 came for, so
            # report before propagating — a soak that dies at minute 20 must not also
            # lose the twenty minutes it did manage.
            if emit_stats:
                yield stats.snapshot().to_line()
            raise
        else:
            # A run shorter than one stats period would otherwise print frames and no
            # numbers — useless for a quick check against a short clip.
            if emit_stats:
                yield stats.snapshot().to_line()
    finally:
        # Also runs on GeneratorExit — `muster spike | head -1` must still release the
        # socket, or the next run finds the camera's session limit reached.
        source.close()
