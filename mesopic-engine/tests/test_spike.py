"""The P1.6 spike runner: does it wire the pipeline together honestly?

Two things are worth testing here and one is not. Worth testing: that the runner
respects the sampler (the whole point of P1.3 is that the detector does *not* see
every frame) and that what it prints carries no pixels. Not worth testing: the
detector and tracker themselves, which have their own suites — here they are fakes,
so a failure in this file means the *wiring* is wrong.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from mesopic.errors import StreamDropped
from mesopic.sampler.sampler import FrameSampler
from mesopic.spike import SpikeStats, frame_line, run_spike
from mesopic.types import (
    CameraId,
    DecodedFrame,
    Detection,
    FrameTs,
    NormPoint,
    Track,
    TrackId,
)

CAMERA = CameraId("spike-cam")
T0 = datetime(2026, 8, 11, 12, 0, 0, tzinfo=UTC)


def _frame(offset_s: float) -> DecodedFrame:
    """A frame whose pixels are recognisable if they ever leak into output."""
    return DecodedFrame(
        camera_id=CAMERA,
        ts=FrameTs(T0 + timedelta(seconds=offset_s)),
        # 7 is arbitrary but non-zero: a leaked buffer would show up as "7"s.
        image=np.full((4, 4, 3), 7, dtype=np.uint8),
        width=4,
        height=4,
    )


def _track(track_id: int, foot_point: NormPoint) -> Track:
    return Track(
        camera_id=CAMERA,
        track_id=TrackId(track_id),
        ts=FrameTs(T0),
        foot_point=foot_point,
        score=0.9,
    )


class _FakeSource:
    """A finite frame source that ends the way a real one does.

    `FrameSource.frames()` raises `StreamDropped` at EOF rather than returning — for a
    camera, "the stream ended" *is* a drop (ingest/rtsp.py). A fake that returns
    normally instead would hide every bug on the only exit path a real run takes.
    """

    def __init__(self, frames: list[DecodedFrame], *, drop_at_end: bool = True) -> None:
        self._frames = frames
        self._drop_at_end = drop_at_end
        self.closed = False

    def frames(self) -> Iterator[DecodedFrame]:
        yield from self._frames
        if self._drop_at_end:
            message = "camera 'spike-cam': stream ended"
            raise StreamDropped(message)

    def close(self) -> None:
        self.closed = True


def _collect(lines: Iterator[str]) -> list[str]:
    """Drain a spike run that ends in the drop every real stream ends in."""
    collected: list[str] = []
    with pytest.raises(StreamDropped):
        collected.extend(lines)
    return collected


class _CountingDetector:
    """Records how often it ran, which is what proves the sampler is honoured."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, _frame: DecodedFrame) -> list[Detection]:
        self.calls += 1
        return [Detection(box=(0, 0, 2, 4), score=0.9)]

    def close(self) -> None:
        return None


class _FakeTracker:
    def __init__(self) -> None:
        self.calls = 0

    def update(self, _frame: DecodedFrame, _detections: list[Detection]) -> list[Track]:
        self.calls += 1
        return [_track(1, (0.25, 0.5))]


# --- The per-frame JSON line ------------------------------------------------


def test_frame_line_carries_ts_fps_and_track_foot_points() -> None:
    """The acceptance criterion: ts, fps, track ids + foot-points, one JSON line."""
    line = frame_line(
        ts=FrameTs(T0),
        tracks=[_track(1, (0.25, 0.5)), _track(2, (0.75, 0.125))],
        effective_fps=2.5,
    )

    assert "\n" not in line, "one frame is one line, or `mesopic spike | jq` breaks"
    payload = json.loads(line)

    assert payload["ts"] == "2026-08-11T12:00:00+00:00"
    assert payload["fps"] == pytest.approx(2.5)
    assert payload["tracks"] == [
        {"id": 1, "foot": [0.25, 0.5]},
        {"id": 2, "foot": [0.75, 0.125]},
    ]


def test_frame_line_emits_utc_offset_explicitly() -> None:
    """A bare naive timestamp downstream is how UTC drift enters a metrics pipeline."""
    payload = json.loads(frame_line(ts=FrameTs(T0), tracks=[], effective_fps=0.0))

    assert payload["ts"].endswith("+00:00")


def test_frame_line_carries_no_pixels() -> None:
    """The categorical claim, asserted at the one place the spike prints anything."""
    frame = _frame(0.0)
    line = frame_line(ts=frame.ts, tracks=[_track(1, (0.5, 0.5))], effective_fps=1.0)
    payload = json.loads(line)

    assert set(payload) == {"ts", "fps", "tracks"}
    for forbidden in ("image", "frame", "crop", "box", "bbox", "pixels"):
        assert forbidden not in payload


# --- The runner -------------------------------------------------------------


def test_run_spike_detects_only_the_frames_the_sampler_admits() -> None:
    """P1.3 exists to keep the detector off most frames. Prove the runner honours it."""
    # 10 frames at 10 fps == 1.0s of wall time; at target 2 fps that is ~2 admissions.
    frames = [_frame(i / 10.0) for i in range(10)]
    source, detector, tracker = _FakeSource(frames), _CountingDetector(), _FakeTracker()

    lines = _collect(
        run_spike(
            source=source,
            sampler=FrameSampler(target_fps=2.0),
            detector=detector,
            tracker=tracker,
        )
    )

    assert detector.calls == 2
    assert tracker.calls == 2
    assert len(lines) == 2, "a line is printed per *admitted* frame, not per decoded one"


def test_run_spike_emits_a_final_stats_line_when_the_stream_ends() -> None:
    """A run shorter than the stats period must still report. Otherwise a quick
    `mesopic spike --stats` against a short clip prints frames and no numbers at all."""
    frames = [_frame(i / 10.0) for i in range(4)]

    lines = _collect(
        run_spike(
            source=_FakeSource(frames),
            sampler=FrameSampler(target_fps=100.0),
            detector=_CountingDetector(),
            tracker=_FakeTracker(),
            emit_stats=True,
        )
    )

    assert json.loads(lines[-1])["stats"] is True
    assert sum(1 for line in lines if json.loads(line).get("stats")) == 1


def test_run_spike_emits_no_stats_line_when_stats_are_off() -> None:
    lines = _collect(
        run_spike(
            source=_FakeSource([_frame(i / 10.0) for i in range(4)]),
            sampler=FrameSampler(target_fps=100.0),
            detector=_CountingDetector(),
            tracker=_FakeTracker(),
        )
    )

    assert all("stats" not in json.loads(line) for line in lines)


def test_run_spike_closes_the_source_when_the_stream_ends() -> None:
    """A spike that leaks the socket cannot be run twice in a row on one box."""
    source = _FakeSource([_frame(0.0)])

    _collect(
        run_spike(
            source=source,
            sampler=FrameSampler(2.0),
            detector=_CountingDetector(),
            tracker=_FakeTracker(),
        )
    )

    assert source.closed


def test_run_spike_closes_the_source_even_when_the_consumer_stops_early() -> None:
    """`mesopic spike | head -1` closes the generator; the socket must still be released."""
    source = _FakeSource([_frame(i / 10.0) for i in range(10)])

    runner = run_spike(
        source=source,
        sampler=FrameSampler(target_fps=100.0),
        detector=_CountingDetector(),
        tracker=_FakeTracker(),
    )
    next(runner)
    runner.close()

    assert source.closed


# --- Stats ------------------------------------------------------------------


class _FakeClock:
    """Wall time and process CPU time, both advanced by hand."""

    def __init__(self) -> None:
        self.wall = 0.0
        self.cpu = 0.0

    def advance(self, wall: float, cpu: float) -> None:
        self.wall += wall
        self.cpu += cpu


def test_stats_reports_effective_fps_over_the_elapsed_window() -> None:
    clock = _FakeClock()
    stats = SpikeStats(monotonic=lambda: clock.wall, cpu_seconds=lambda: clock.cpu)

    for _ in range(6):
        clock.advance(wall=0.5, cpu=0.25)
        stats.record(latency_s=0.1)

    snapshot = stats.snapshot()

    assert snapshot.effective_fps == pytest.approx(2.0)


def test_stats_reports_cpu_percent_from_process_cpu_time() -> None:
    """Half a CPU-second per wall second is 50%, and may exceed 100% on many cores."""
    clock = _FakeClock()
    stats = SpikeStats(monotonic=lambda: clock.wall, cpu_seconds=lambda: clock.cpu)

    clock.advance(wall=2.0, cpu=1.0)
    stats.record(latency_s=0.1)

    assert stats.snapshot().cpu_percent == pytest.approx(50.0)


def test_stats_reports_mean_and_max_detect_track_latency() -> None:
    clock = _FakeClock()
    stats = SpikeStats(monotonic=lambda: clock.wall, cpu_seconds=lambda: clock.cpu)

    for latency in (0.1, 0.2, 0.3):
        clock.advance(wall=1.0, cpu=0.5)
        stats.record(latency_s=latency)

    snapshot = stats.snapshot()

    assert snapshot.mean_latency_ms == pytest.approx(200.0)
    assert snapshot.max_latency_ms == pytest.approx(300.0)


def test_stats_is_due_once_per_second() -> None:
    """`--stats` logs *each second*, not each frame."""
    clock = _FakeClock()
    stats = SpikeStats(monotonic=lambda: clock.wall, cpu_seconds=lambda: clock.cpu)

    clock.advance(wall=0.4, cpu=0.2)
    assert not stats.is_due()

    clock.advance(wall=0.7, cpu=0.2)
    assert stats.is_due()


def test_snapshot_resets_the_window_so_fps_is_rolling_not_cumulative() -> None:
    """A cumulative average hides the fps decay P1.7's 30-minute soak exists to catch."""
    clock = _FakeClock()
    stats = SpikeStats(monotonic=lambda: clock.wall, cpu_seconds=lambda: clock.cpu)

    for _ in range(10):
        clock.advance(wall=0.1, cpu=0.05)
        stats.record(latency_s=0.01)
    assert stats.snapshot().effective_fps == pytest.approx(10.0)

    for _ in range(2):
        clock.advance(wall=0.5, cpu=0.05)
        stats.record(latency_s=0.01)

    assert stats.snapshot().effective_fps == pytest.approx(2.0)


def test_snapshot_reports_rss_in_bytes() -> None:
    """P1.7 asserts no OOM over 30 minutes, so RSS has to be a real, comparable number."""
    clock = _FakeClock()
    stats = SpikeStats(monotonic=lambda: clock.wall, cpu_seconds=lambda: clock.cpu)

    clock.advance(wall=1.0, cpu=0.5)
    stats.record(latency_s=0.1)

    # A live process is larger than a megabyte and smaller than a terabyte.
    assert 1_000_000 < stats.snapshot().rss_bytes < 1_000_000_000_000


def test_stats_line_is_one_json_line_with_the_criterion_fields() -> None:
    clock = _FakeClock()
    stats = SpikeStats(monotonic=lambda: clock.wall, cpu_seconds=lambda: clock.cpu)
    clock.advance(wall=1.0, cpu=0.5)
    stats.record(latency_s=0.1)

    line = stats.snapshot().to_line()

    assert "\n" not in line
    payload = json.loads(line)
    assert payload["stats"] is True
    assert set(payload) >= {
        "stats",
        "fps",
        "mean_latency_ms",
        "max_latency_ms",
        "cpu_percent",
        "rss_bytes",
    }


def test_snapshot_of_an_empty_window_does_not_divide_by_zero() -> None:
    """The first `--stats` tick can land before any frame was admitted."""
    clock = _FakeClock()
    stats = SpikeStats(monotonic=lambda: clock.wall, cpu_seconds=lambda: clock.cpu)

    clock.advance(wall=1.0, cpu=0.1)
    snapshot = stats.snapshot()

    assert snapshot.effective_fps == 0.0
    assert snapshot.mean_latency_ms == 0.0
    assert snapshot.max_latency_ms == 0.0
