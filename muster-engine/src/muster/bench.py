"""The P1.7 bench harness: turning a spike run into the M0 gate's verdict.

The M0 gate (implementation-plan P1.7) asks one question with a yes/no answer: does
one 1080p camera sustain >= 2 fps effective, decode->detect->track, for >= 30 minutes
on the N100, without running out of memory or thermally throttling? This module is
the half of the answer that does not need hardware — it consumes the JSON lines
`muster spike` already emits (spike.py's stated contract: "P1.7's harness parses
rather than scrapes") and reduces them to measured numbers and per-threshold verdicts.

Three deliberate positions, each of which exists to stop a number being read as more
than it is:

* **A threshold that could not be measured is `None`, not `False` and never `True`.**
  A throttle counter absent from a non-Linux `/sys` means "unknown"; a run that cannot
  see whether the CPU throttled has not proved that it did not.
* **Nothing asserted is not a pass.** `evaluate` with no thresholds returns a result
  whose `passed` is false, because a bench that checked nothing gates nothing.
* **The artifact discloses its own provenance** — which machine, whether that machine
  is the baseline, and where the RSS figure came from. A run on a laptop is quotable
  as a laptop number and never as the M0 gate.

The measurement logic here is pure and injectable end to end, so a thirty-minute soak,
a stream drop, a leak and a throttle event are all unit-testable in milliseconds.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from muster.errors import StreamDropped

_MIN_SAMPLES_FOR_GROWTH = 2
"""Two points make a trend line; one makes none."""

MEM_GROWTH_TOLERANCE_BYTES = 8 * 1024 * 1024
"""Slack on the leak check.

Allocator high-water marks and arena churn move RSS by a few MiB across a long run
without anything leaking. Set high enough not to cry wolf, far below what thirty
minutes of a real leak would reach.
"""


THROTTLE_GLOB = "cpu*/thermal_throttle/core_throttle_count"
"""Intel's per-core throttle event counter, which is what the N100 exposes."""

_CPU_SYSFS = Path("/sys/devices/system/cpu")
_STATM = Path("/proc/self/statm")


def read_throttle_count(*, cpu_sysfs: Path = _CPU_SYSFS) -> int | None:
    """Total thermal-throttle events across cores, or `None` where unreadable.

    `None` is load-bearing: on macOS, in a container without `/sys`, or on a non-Intel
    part, there is no counter, and the honest answer to "did it throttle" is that we
    do not know. Returning 0 there would silently convert ignorance into a passing
    gate leg on the one threshold nobody can eyeball afterwards.
    """
    try:
        counters = sorted(cpu_sysfs.glob(THROTTLE_GLOB))
        if not counters:
            return None
        return sum(int(path.read_text().strip()) for path in counters)
    except (OSError, ValueError):
        return None


def read_current_rss_bytes(*, statm: Path = _STATM, page_size: int | None = None) -> int | None:
    """Current (not peak) resident set size in bytes, or `None` off Linux.

    The leak check needs a figure that can go *down*; `resource.getrusage` reports a
    high-water mark that cannot. `/proc/self/statm` field 2 is resident pages, and
    costs one small read per second — cheap enough for a soak, and no new dependency
    (CLAUDE.md prefers stdlib over `psutil`, which §9 otherwise suggests).
    """
    try:
        resident_pages = int(statm.read_text().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return resident_pages * (page_size if page_size is not None else os.sysconf("SC_PAGE_SIZE"))


def describe_environment() -> dict[str, Any]:
    """What the numbers were measured on, so a result cannot be quoted out of context."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }


def percentile(values: list[float], quantile: float) -> float | None:
    """Linear-interpolated percentile of `values`, or `None` if there are none.

    `None` rather than `0.0` on an empty series: a run that produced no samples has an
    *absent* latency, and a zero would read as a spectacularly good one.
    """
    if not values:
        return None

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


@dataclass(frozen=True, slots=True)
class BenchThresholds:
    """The gate's legs. Every one left `None`/`False` is simply not asserted."""

    min_fps: float | None = None
    max_cpu_fraction: float | None = None
    no_mem_growth: bool = False
    no_throttle: bool = False


@dataclass(frozen=True, slots=True)
class Verdict:
    """One threshold's outcome. `passed is None` means it could not be measured."""

    name: str
    passed: bool | None
    detail: str


@dataclass
class BenchRun:
    """Accumulated state of one bench run.

    Fed the spike's JSON lines as they arrive, so a soak that dies at minute 20 still
    has twenty minutes of measurement rather than nothing.
    """

    started_monotonic: float
    frames: int = 0
    drops: int = 0
    downtime_s: float = 0.0
    elapsed_s: float = 0.0
    rss_source: str = "unknown"
    # When set, overrides the spike's `ru_maxrss` figure with a *current* RSS reading.
    # The spike reports peak, which never falls and so can only ever show that growth
    # stopped — not enough for the leak check the M0 gate's "no OOM" leg rests on.
    rss_reader: Callable[[], int | None] | None = None
    _mean_latencies: list[float] = field(default_factory=list)
    _max_latencies: list[float] = field(default_factory=list)
    _cpu_percents: list[float] = field(default_factory=list)
    _rss_series: list[int] = field(default_factory=list)
    _throttle_start: int | None = None
    _throttle_end: int | None = None

    def consume_line(self, line: str) -> None:
        """Route one spike JSON line: a stats snapshot, or an admitted frame."""
        payload = json.loads(line)
        if payload.get("stats"):
            self._mean_latencies.append(float(payload["mean_latency_ms"]))
            self._max_latencies.append(float(payload["max_latency_ms"]))
            self._cpu_percents.append(float(payload["cpu_percent"]))
            self._rss_series.append(self._read_rss(int(payload["rss_bytes"])))
        else:
            self.frames += 1

    def _read_rss(self, reported: int) -> int:
        """Prefer a current-RSS reading; fall back to what the spike reported."""
        if self.rss_reader is None:
            return reported
        current = self.rss_reader()
        return current if current is not None else reported

    def mark_elapsed(self, elapsed_s: float) -> None:
        """Set wall-clock elapsed. The caller owns the clock so tests need no sleeping."""
        self.elapsed_s = elapsed_s

    def record_drop(self, *, downtime_s: float) -> None:
        """Record a stream drop and the dead air before the reconnect succeeded."""
        self.drops += 1
        self.downtime_s += downtime_s

    def record_throttle(self, *, start: int | None, end: int | None) -> None:
        """Record the thermal-throttle counter either side of the run."""
        self._throttle_start = start
        self._throttle_end = end

    def record_rss_source(self, source: str) -> None:
        """Record where the RSS figure came from, so peak is never read as current."""
        self.rss_source = source

    @property
    def sustained_fps(self) -> float:
        """Frames actually detected over wall clock (test-strategy §9).

        Reconnect downtime sits *inside* the wall clock rather than being excluded: a
        camera that was disconnected was not covering the doorway, and the gate is
        about coverage, not about the pipeline's form when it happened to be running.
        """
        if self.elapsed_s <= 0.0:
            return 0.0
        return self.frames / self.elapsed_s

    @property
    def cpu_percent_mean(self) -> float | None:
        """Mean CPU across the run. May exceed 100%: four cores means a 400% ceiling."""
        if not self._cpu_percents:
            return None
        return sum(self._cpu_percents) / len(self._cpu_percents)

    def latency_p50_ms(self) -> float | None:
        """Median of the per-second *mean* latencies — see `latency_p95_ms`."""
        return percentile(self._mean_latencies, 0.5)

    def latency_p95_ms(self) -> float | None:
        """95th percentile of the per-second mean latencies.

        Not a true per-frame p95: the spike emits per-second aggregates, so this is
        "the 95th-percentile second", which is a coarser statistic than the one
        test-strategy §9 ultimately wants. `worst_latency_ms` carries the tail.
        """
        return percentile(self._mean_latencies, 0.95)

    def worst_latency_ms(self) -> float | None:
        """The worst single frame the run saw, across every window."""
        return max(self._max_latencies) if self._max_latencies else None

    def throttle_delta(self) -> int | None:
        """Throttle events during the run, or `None` if the counter was unreadable."""
        if self._throttle_start is None or self._throttle_end is None:
            return None
        return self._throttle_end - self._throttle_start

    def mem_growth_bytes(self) -> float | None:
        """RSS growth across the run's second half, or `None` if too short to tell.

        The second half only, because the first is warmup: model sessions load, arenas
        size themselves, and RSS climbs steeply and legitimately. Measuring end-to-end
        would fail a healthy run; measuring after the pipeline has settled is what
        actually answers "is this leaking".
        """
        half = self._rss_series[len(self._rss_series) // 2 :]
        if len(half) < _MIN_SAMPLES_FOR_GROWTH:
            return None
        return float(half[-1] - half[0])


@dataclass(frozen=True, slots=True)
class BenchResult:
    """The measured run plus its verdicts — the thing written to disk and quoted."""

    verdicts: list[Verdict]
    payload: dict[str, Any]

    @property
    def passed(self) -> bool:
        """True only if something was asserted and every asserted leg passed."""
        return bool(self.verdicts) and all(v.passed is True for v in self.verdicts)

    def to_json(self) -> str:
        """The artifact, as indented JSON — a file a human reads and pastes."""
        return json.dumps({**self.payload, "passed": self.passed}, indent=2, sort_keys=True)


def _fps_verdict(run: BenchRun, floor: float) -> Verdict:
    measured = run.sustained_fps
    return Verdict(
        name="min_fps",
        passed=measured >= floor,
        detail=f"sustained {measured:.3f} fps against a {floor:.3f} fps floor",
    )


def _cpu_verdict(run: BenchRun, budget: float, cpu_count: int) -> Verdict:
    mean = run.cpu_percent_mean
    if mean is None:
        return Verdict(name="max_cpu", passed=None, detail="no CPU samples recorded")

    # A 400% reading on four cores is the ceiling, not four times over budget.
    fraction = mean / (100.0 * cpu_count)
    return Verdict(
        name="max_cpu",
        passed=fraction <= budget,
        detail=f"mean {fraction:.3f} of {cpu_count} cores against a {budget:.3f} budget",
    )


def _mem_verdict(run: BenchRun) -> Verdict:
    growth = run.mem_growth_bytes()
    if growth is None:
        return Verdict(
            name="no_mem_growth",
            passed=None,
            detail="run too short to measure growth over its second half",
        )

    mib = growth / (1024 * 1024)
    return Verdict(
        name="no_mem_growth",
        passed=growth <= MEM_GROWTH_TOLERANCE_BYTES,
        detail=f"second-half RSS moved {mib:+.1f} MiB",
    )


def _throttle_verdict(run: BenchRun) -> Verdict:
    delta = run.throttle_delta()
    if delta is None:
        return Verdict(
            name="no_throttle",
            passed=None,
            detail="throttle counter unreadable on this platform",
        )
    return Verdict(
        name="no_throttle",
        passed=delta == 0,
        detail=f"{delta} throttle events during the run",
    )


INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0


def _consume_until(
    run: BenchRun,
    lines: Iterator[str],
    deadline_reached: Callable[[], bool],
) -> None:
    """Feed spike lines into `run` until the deadline, then release the camera socket.

    Closing the generator is what runs `run_spike`'s `finally` and frees the RTSP
    session; a soak that walked away from it would find the camera's session limit
    reached on the next attempt.
    """
    try:
        for line in lines:
            run.consume_line(line)
            if deadline_reached():
                return
    finally:
        close = getattr(lines, "close", None)
        if close is not None:
            close()


def drive_bench(
    *,
    open_lines: Callable[[], Iterator[str]],
    duration_s: float,
    run: BenchRun,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    throttle_reader: Callable[[], int | None] = read_throttle_count,
    max_backoff_s: float = MAX_BACKOFF_S,
) -> BenchRun:
    """Run the pipeline for `duration_s`, reconnecting across drops.

    A thirty-minute soak outlives the average RTSP session, and `RtspFrameSource`
    deliberately holds no reconnect policy of its own (ingest/rtsp.py: backoff and
    `CameraState` belong to the supervisor, which is P2.7). Until that exists, the
    bench owns it — otherwise one blip at minute three ends the M0 measurement and
    reports it as a pipeline that could not keep up.
    """
    started = monotonic()
    throttle_start = throttle_reader()
    backoff = INITIAL_BACKOFF_S

    def past_deadline() -> bool:
        return (monotonic() - started) >= duration_s

    while not past_deadline():
        try:
            before = run.frames
            _consume_until(run, open_lines(), past_deadline)
            if run.frames > before:
                backoff = INITIAL_BACKOFF_S
        except StreamDropped:
            if past_deadline():
                break
            dropped_at = monotonic()
            sleep(backoff)
            run.record_drop(downtime_s=monotonic() - dropped_at)
            backoff = min(backoff * 2.0, max_backoff_s)

    run.record_throttle(start=throttle_start, end=throttle_reader())
    run.mark_elapsed(monotonic() - started)
    return run


def evaluate(
    run: BenchRun,
    thresholds: BenchThresholds,
    *,
    cpu_count: int | None = None,
    baseline_hardware: bool = False,
    environment: dict[str, Any] | None = None,
) -> BenchResult:
    """Reduce a run to its measured numbers and the verdict on each asserted leg."""
    cores = cpu_count if cpu_count is not None else (os.cpu_count() or 1)

    verdicts: list[Verdict] = []
    if thresholds.min_fps is not None:
        verdicts.append(_fps_verdict(run, thresholds.min_fps))
    if thresholds.max_cpu_fraction is not None:
        verdicts.append(_cpu_verdict(run, thresholds.max_cpu_fraction, cores))
    if thresholds.no_mem_growth:
        verdicts.append(_mem_verdict(run))
    if thresholds.no_throttle:
        verdicts.append(_throttle_verdict(run))

    payload: dict[str, Any] = {
        "baseline_hardware": baseline_hardware,
        "cpu_count": cores,
        "duration_s": run.elapsed_s,
        "frames": run.frames,
        "sustained_fps": run.sustained_fps,
        "cpu_percent_mean": run.cpu_percent_mean,
        "p50_latency_ms": run.latency_p50_ms(),
        "p95_latency_ms": run.latency_p95_ms(),
        "worst_latency_ms": run.worst_latency_ms(),
        "mem_growth_bytes": run.mem_growth_bytes(),
        "rss_source": run.rss_source,
        "throttle_events": run.throttle_delta(),
        "drops": run.drops,
        "downtime_s": run.downtime_s,
        "environment": environment or {},
        "verdicts": [{"name": v.name, "passed": v.passed, "detail": v.detail} for v in verdicts],
    }
    return BenchResult(verdicts=verdicts, payload=payload)
