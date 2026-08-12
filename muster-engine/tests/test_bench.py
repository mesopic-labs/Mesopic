"""The P1.7 bench harness: does it measure the M0 gate honestly?

The gate this harness reports on can only be *run* on an N100, but it has to be
*trusted* from a laptop, so everything it reads — the clock, resident memory, the
throttle counter, the frame source — is injected. A thirty-minute soak, a mid-run
stream drop, a memory leak and a thermal throttle all happen here in milliseconds
and without hardware.

The distinction this file exists to defend: a threshold that could not be measured
must never read as a pass. "Unknown" is its own verdict.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from muster.bench import (
    BenchResult,
    BenchRun,
    BenchThresholds,
    Verdict,
    drive_bench,
    evaluate,
    percentile,
    read_current_rss_bytes,
    read_throttle_count,
)
from muster.errors import StreamDropped
from muster.spike import StatsSnapshot

MIB = 1024 * 1024

_DROP_MESSAGE = "camera 'x': dropped"
"""What `RtspFrameSource` raises: the camera id, never the URL."""


def _snapshot(
    *,
    fps: float = 2.5,
    mean_latency_ms: float = 20.0,
    max_latency_ms: float = 25.0,
    cpu_percent: float = 100.0,
    rss_bytes: int = 200 * MIB,
) -> StatsSnapshot:
    return StatsSnapshot(
        effective_fps=fps,
        mean_latency_ms=mean_latency_ms,
        max_latency_ms=max_latency_ms,
        cpu_percent=cpu_percent,
        rss_bytes=rss_bytes,
    )


def _soak(
    *,
    seconds: int = 1800,
    fps: float = 2.5,
    rss: Iterator[int] | None = None,
    cpu_percent: float = 100.0,
) -> BenchRun:
    """A completed run of `seconds` seconds at a steady rate, one snapshot per second.

    Frames are emitted at `fps` for real, accumulated across seconds so a fractional
    rate lands the right total — the sustained figure is frames over wall clock, so a
    helper that emitted one frame a second would quietly measure 1.0 fps whatever it
    was asked for.
    """
    run = BenchRun(started_monotonic=0.0)
    owed = 0.0
    for second in range(seconds):
        owed += fps
        while owed >= 1.0:
            run.consume_line(
                json.dumps({"ts": "2026-08-12T00:00:00+00:00", "fps": fps, "tracks": []})
            )
            owed -= 1.0
        run.consume_line(
            _snapshot(
                fps=fps,
                cpu_percent=cpu_percent,
                rss_bytes=next(rss) if rss is not None else 200 * MIB,
            ).to_line()
        )
        run.mark_elapsed(float(second + 1))
    return run


def _leg_of(result: BenchResult, name: str) -> Verdict:
    """The single named leg of a verdict, so a test asserts on one threshold at a time."""
    return next(v for v in result.verdicts if v.name == name)


def _leg(run: BenchRun, thresholds: BenchThresholds, name: str) -> Verdict:
    return _leg_of(evaluate(run, thresholds), name)


class TestPercentile:
    """p95 is the number the latency budget is read from, so its edges matter."""

    def test_interpolates_between_samples(self) -> None:
        assert percentile([10.0, 20.0], 0.5) == pytest.approx(15.0)

    def test_p95_of_a_flat_series_is_the_value(self) -> None:
        assert percentile([7.0] * 100, 0.95) == pytest.approx(7.0)

    def test_single_sample(self) -> None:
        assert percentile([42.0], 0.95) == pytest.approx(42.0)

    def test_empty_series_has_no_percentile(self) -> None:
        # Not 0.0: a run that produced no samples has an *absent* latency, and a
        # zero would read as a spectacularly good one.
        assert percentile([], 0.95) is None


class TestFrameAccounting:
    """Effective fps is frames-actually-detected over wall clock (test-strategy §9)."""

    def test_counts_frame_lines_and_ignores_stats_lines(self) -> None:
        run = _soak(seconds=10, fps=1.0)
        assert run.frames == 10

    def test_sustained_fps_is_frames_over_wall_clock(self) -> None:
        run = _soak(seconds=100, fps=2.5)
        # 250 frame lines over 100s of wall clock.
        assert run.frames == 250
        assert run.sustained_fps == pytest.approx(2.5)

    def test_downtime_counts_against_sustained_fps(self) -> None:
        """A reconnect gap is dead air the camera did not cover, not a pause."""
        run = BenchRun(started_monotonic=0.0)
        for _ in range(10):
            run.consume_line(json.dumps({"ts": "t", "fps": 2.5, "tracks": []}))
        run.record_drop(downtime_s=5.0)
        run.mark_elapsed(10.0)

        assert run.drops == 1
        assert run.downtime_s == pytest.approx(5.0)
        # 10 frames / 10s wall clock — the 5s hole is inside the wall clock, not excluded.
        assert run.sustained_fps == pytest.approx(1.0)


class TestMemoryGrowth:
    """The leak check, which is why the gate says 'no OOM' and not 'peak RSS was fine'."""

    def test_warmup_growth_does_not_count_as_a_leak(self) -> None:
        """RSS climbing while the model loads, then flat, is a healthy run."""
        warmup = iter([150 * MIB + i * MIB for i in range(50)] + [200 * MIB] * 50)
        run = _soak(seconds=100, rss=warmup)

        # Measured on the second half only, which is flat.
        assert run.mem_growth_bytes() == pytest.approx(0.0, abs=MIB)

    def test_steady_growth_across_the_second_half_is_a_leak(self) -> None:
        leak = iter([150 * MIB + i * MIB for i in range(100)])
        run = _soak(seconds=100, rss=leak)

        growth = run.mem_growth_bytes()
        assert growth is not None
        assert growth > 40 * MIB

    def test_growth_needs_a_second_half_to_measure(self) -> None:
        run = _soak(seconds=1)
        assert run.mem_growth_bytes() is None


class TestThrottleVerdict:
    """A throttle counter that could not be read is unknown, never a pass."""

    def test_unchanged_counter_passes(self) -> None:
        run = _soak(seconds=10)
        run.record_throttle(start=3, end=3)

        verdict = _leg(run, BenchThresholds(no_throttle=True), "no_throttle")
        assert verdict.passed is True

    def test_increased_counter_fails(self) -> None:
        run = _soak(seconds=10)
        run.record_throttle(start=3, end=9)

        verdict = _leg(run, BenchThresholds(no_throttle=True), "no_throttle")
        assert verdict.passed is False
        assert "6" in verdict.detail

    def test_unreadable_counter_is_unknown_not_passed(self) -> None:
        run = _soak(seconds=10)
        run.record_throttle(start=None, end=None)

        verdict = _leg(run, BenchThresholds(no_throttle=True), "no_throttle")
        assert verdict.passed is None
        assert not evaluate(run, BenchThresholds(no_throttle=True)).passed


class TestVerdict:
    """The gate: every asserted leg must pass, and a failure must name itself."""

    def test_a_clean_soak_passes_every_leg(self) -> None:
        run = _soak(seconds=1800, fps=2.5)
        run.record_throttle(start=0, end=0)
        result = evaluate(
            run,
            BenchThresholds(
                min_fps=2.0, max_cpu_fraction=0.75, no_mem_growth=True, no_throttle=True
            ),
        )

        assert result.passed
        assert all(v.passed for v in result.verdicts)

    def test_fps_below_the_floor_fails_and_names_the_leg(self) -> None:
        # 1800 frames over 1800s = 1.0 fps sustained, under a 2.0 floor.
        run = _soak(seconds=1800, fps=1.0)
        result = evaluate(run, BenchThresholds(min_fps=2.0))

        assert not result.passed
        assert _leg(run, BenchThresholds(min_fps=2.0), "min_fps").passed is False

    def test_cpu_fraction_is_normalised_by_core_count(self) -> None:
        """400% on four cores is the ceiling, not four times over budget."""
        run = _soak(seconds=100, cpu_percent=200.0)
        result = evaluate(run, BenchThresholds(max_cpu_fraction=0.75), cpu_count=4)

        # 200% of a 400% ceiling = 0.5, inside a 0.75 budget.
        assert _leg_of(result, "max_cpu").passed is True

    def test_cpu_over_budget_fails(self) -> None:
        run = _soak(seconds=100, cpu_percent=350.0)
        result = evaluate(run, BenchThresholds(max_cpu_fraction=0.75), cpu_count=4)

        assert _leg_of(result, "max_cpu").passed is False

    def test_unasserted_thresholds_produce_no_leg(self) -> None:
        run = _soak(seconds=10)
        result = evaluate(run, BenchThresholds())

        assert result.verdicts == []
        # Nothing asserted is not a gate pass.
        assert not result.passed


class _FakeClock:
    """A monotonic clock that only moves when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def tick(self, seconds: float) -> None:
        self.now += seconds


class TestDriver:
    """Duration control and the reconnect policy the frame source deliberately lacks."""

    def test_stops_at_the_deadline(self) -> None:
        clock = _FakeClock()

        def lines() -> Iterator[str]:
            while True:
                clock.tick(1.0)
                yield json.dumps({"ts": "t", "fps": 1.0, "tracks": []})

        run = drive_bench(
            open_lines=lines,
            duration_s=10.0,
            run=BenchRun(started_monotonic=0.0),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            throttle_reader=lambda: 0,
        )

        assert run.frames == 10
        assert run.elapsed_s == pytest.approx(10.0)

    def test_reconnects_across_a_drop_and_records_downtime(self) -> None:
        clock = _FakeClock()
        attempts = 0

        def lines() -> Iterator[str]:
            nonlocal attempts
            attempts += 1
            clock.tick(1.0)
            yield json.dumps({"ts": "t", "fps": 1.0, "tracks": []})
            if attempts == 1:
                raise StreamDropped(_DROP_MESSAGE)

        run = drive_bench(
            open_lines=lines,
            duration_s=5.0,
            run=BenchRun(started_monotonic=0.0),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            throttle_reader=lambda: 0,
        )

        assert run.drops == 1
        assert run.downtime_s == pytest.approx(1.0)
        assert attempts > 1

    def test_backoff_grows_while_the_camera_stays_down(self) -> None:
        clock = _FakeClock()

        def lines() -> Iterator[str]:
            # `yield from ()` makes this a generator without an unreachable statement:
            # the camera is down, so it drops on the first frame every time.
            yield from ()
            raise StreamDropped(_DROP_MESSAGE)

        drive_bench(
            open_lines=lines,
            duration_s=100.0,
            run=BenchRun(started_monotonic=0.0),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            throttle_reader=lambda: 0,
            max_backoff_s=8.0,
        )

        assert clock.slept[:4] == [1.0, 2.0, 4.0, 8.0]
        assert max(clock.slept) == 8.0

    def test_closes_the_generator_so_the_camera_socket_is_released(self) -> None:
        clock = _FakeClock()
        closed = False

        def lines() -> Iterator[str]:
            nonlocal closed
            try:
                while True:
                    clock.tick(1.0)
                    yield json.dumps({"ts": "t", "fps": 1.0, "tracks": []})
            finally:
                closed = True

        drive_bench(
            open_lines=lines,
            duration_s=3.0,
            run=BenchRun(started_monotonic=0.0),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            throttle_reader=lambda: 0,
        )

        assert closed


class TestProbes:
    """The two platform readings, and their refusal to guess."""

    def test_throttle_counter_sums_cores(self, tmp_path: Path) -> None:
        for core, count in enumerate([2, 5]):
            path = tmp_path / f"cpu{core}" / "thermal_throttle"
            path.mkdir(parents=True)
            (path / "core_throttle_count").write_text(f"{count}\n")

        assert read_throttle_count(cpu_sysfs=tmp_path) == 7

    def test_throttle_counter_absent_is_none_not_zero(self, tmp_path: Path) -> None:
        assert read_throttle_count(cpu_sysfs=tmp_path) is None

    def test_current_rss_reads_resident_pages(self, tmp_path: Path) -> None:
        statm = tmp_path / "statm"
        statm.write_text("1000 512 100 10 0 200 0\n")

        assert read_current_rss_bytes(statm=statm, page_size=4096) == 512 * 4096

    def test_current_rss_absent_is_none(self, tmp_path: Path) -> None:
        assert read_current_rss_bytes(statm=tmp_path / "nope", page_size=4096) is None

    def test_current_rss_overrides_the_spikes_peak_figure(self) -> None:
        """Peak RSS cannot fall, so the leak check must read the current figure."""
        climbing = iter([100 * MIB, 101 * MIB, 102 * MIB, 103 * MIB])
        run = BenchRun(started_monotonic=0.0, rss_reader=lambda: next(climbing))
        for _ in range(4):
            # A flat peak figure, which on its own would report no growth at all.
            run.consume_line(_snapshot(rss_bytes=500 * MIB).to_line())
        run.mark_elapsed(4.0)

        assert run.mem_growth_bytes() == pytest.approx(float(MIB))


class TestArtifact:
    """The artifact is quotable evidence, so it must disclose what it is not."""

    def test_records_that_a_non_baseline_run_is_not_the_gate(self) -> None:
        run = _soak(seconds=10)
        result = evaluate(run, BenchThresholds(min_fps=2.0), baseline_hardware=False)
        payload = json.loads(result.to_json())

        assert payload["baseline_hardware"] is False

    def test_records_the_rss_source_so_peak_is_never_read_as_current(self) -> None:
        run = _soak(seconds=10)
        run.record_rss_source("peak-ru_maxrss")
        payload = json.loads(evaluate(run, BenchThresholds()).to_json())

        assert payload["rss_source"] == "peak-ru_maxrss"

    def test_carries_the_measured_numbers(self) -> None:
        run = _soak(seconds=100, fps=2.5)
        payload = json.loads(evaluate(run, BenchThresholds(min_fps=2.0)).to_json())

        assert payload["sustained_fps"] == pytest.approx(2.5)
        assert payload["p95_latency_ms"] == pytest.approx(20.0)
        assert payload["frames"] == 250
        assert payload["duration_s"] == pytest.approx(100.0)

    def test_carries_no_rtsp_url(self) -> None:
        """The URL is a credential (CLAUDE.md); an artifact is a file people paste."""
        run = _soak(seconds=10)
        payload = evaluate(run, BenchThresholds()).to_json()

        assert "rtsp://" not in payload
