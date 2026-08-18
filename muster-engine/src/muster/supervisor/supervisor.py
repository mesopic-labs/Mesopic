"""Spawn, watch, and restart camera workers; own the shared sinks.

The bounded event queue is the backpressure valve. When the supervisor falls behind, the
queue fills and the *sampler* sheds fps — never the decoder, and never the store. Only
small `RawEvent`s cross the process boundary; a frame never does.

Three things here are load-bearing and easy to get subtly wrong:

* **Buckets close on a lag, not on the minute.** A dwell ending at 09:30:59 is only known
  once its exit grace lapses at 09:31:01, so closing 09:30 at 09:31:00 would miss it. The
  lag is derived from the aggregator's own grace rather than written down twice.
* **The aggregator has no clock**, on purpose: it advances on the timestamps it is fed. A
  zone the last person left is therefore waiting on an event that will never arrive, and
  the tick's `flush(now)` is what resolves it.
* **A bucket that has been closed and forgotten cannot be re-opened.** Memory is bounded
  by forgetting, and a late event folded alone into a forgotten bucket would upsert a
  partial value *over* the correct one. Late events are dropped and counted instead.

Implements P2.7 (engine-architecture.md §9).
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from muster.aggregator.aggregator import EXIT_GRACE_S, Aggregator, bucket_of
from muster.analytics.metrics import build_registry
from muster.analytics.metrics.heatmap import HeatmapAccumulator
from muster.analytics.site_geometry import SiteGeometry
from muster.config.schema import MusterConfig
from muster.errors import SnapshotUnavailableError
from muster.exporters.fanout import ExporterFanout
from muster.store.store import Store
from muster.supervisor.handle import WorkerEntry, WorkerHandle, WorkerReport
from muster.supervisor.worker import run_camera_worker
from muster.types import CameraId, EventKind, FrameTs, MinuteBucket, RawEvent

_DENSE_KINDS = frozenset({EventKind.OCCUPANCY_SAMPLE, EventKind.HEATMAP_HIT})
"""Sampled state, emitted every tick whether or not anything changed — never logged.

`events` is a log of per-track *transitions*, and these two are neither. An occupancy
sample names no track at all, and a heatmap hit names one but carries a grid cell the
table has no column for, at one row per resident per zone per tick — the densest kind the
engine emits. Both are folded in memory and reduced; the raw log would gain nothing from
them but size (ADR-0016, and P4.1 for the hit).
"""

BUCKET_S = 60.0

CLOSE_LAG_S = EXIT_GRACE_S + 3.0
"""How long after a minute ends before its bucket is closed.

Derived from the aggregator's exit grace rather than chosen independently: a lag shorter
than the grace closes buckets before the dwells inside them are known, and two numbers
that must agree are two numbers that eventually will not.
"""

TICK_S = 1.0
"""How often `run` wakes. Cheap: a tick with nothing due does almost nothing."""

DRAIN_LIMIT = 512
"""Events taken from one worker per drain, so a busy camera cannot starve the others."""

SNAPSHOT_TIMEOUT_S = 5.0
"""How long a calibration request waits for a frame.

Generous next to a 2 fps sampling grid and short next to a human's patience. A camera
that has not answered in five seconds is not slow, it is stuck, and the operator is
better served by being told that than by a spinner."""

TRIM_INTERVAL_S = 3600.0
"""How often the raw event log is trimmed. A constant rather than a config key: the
retention *window* is what a self-hoster tunes (`storage.event_retention_hours`), and
this only decides how promptly expiry is enforced. A DELETE on every one-second tick
would be pure waste against a window measured in hours."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Supervisor:
    """Runs one site: N camera workers plus the shared aggregator, store, and sinks."""

    def __init__(
        self,
        config: MusterConfig,
        *,
        store: Store,
        entry: WorkerEntry = run_camera_worker,
        now: Callable[[], datetime] = _utcnow,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._store = store
        self._entry = entry
        self._now = now
        self._monotonic = monotonic
        geometry = SiteGeometry.compile(config)
        self._aggregator = Aggregator(
            build_registry(geometry),
            dwell_min_s=config.thresholds.dwell_min_s,
            heatmaps=HeatmapAccumulator(geometry),
        )
        self._handles = [
            WorkerHandle(camera.camera_id, config=config, entry=entry, monotonic=monotonic)
            for camera in config.cameras
            if camera.enabled
        ]
        self.exporters = ExporterFanout({})
        """Where committed rows go next. Replaced by `build_exporters(config)` at the
        composition root; empty here so a supervisor is usable without any peer."""
        self._closed_through: MinuteBucket | None = None
        self._stopping = False
        self.skipped_untracked_events = 0
        """Events the raw log deliberately does not hold — the dense sampled-state kinds,
        plus any event without a track. See `_DENSE_KINDS`."""
        self.late_events = 0
        """Events for a bucket already closed and forgotten. Dropped, never re-folded."""
        self.restarts = 0
        self.trimmed_events = 0
        """Raw events deleted by the retention job. Visible, like every other loss here."""
        self.trim_runs = 0
        self._last_trim_at: float | None = None

    # --- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        self.exporters.start()
        for handle in self._handles:
            handle.start()
            handle.note_started(self._monotonic())

    async def run(self) -> None:
        """Run until shutdown, restarting failed workers with backoff."""
        await self.start()
        try:
            while not self._stopping:
                await self.tick()
                await self.supervise(monotonic=self._monotonic())
                await asyncio.sleep(TICK_S)
        finally:
            await self.stop()

    def request_shutdown(self) -> None:
        self._stopping = True

    async def stop(self, *, flush: bool = True) -> None:
        """Stop the workers, take what they already produced, and close what is open.

        `flush=False` is the crash-stop: leave the open minute open rather than write a
        bucket whose remaining events were never going to arrive.
        """
        self._stopping = True
        for handle in self._handles:
            handle.stop(timeout=2.0)
        await self.drain_once(timeout=0.0)
        if flush:
            self._aggregator.flush(FrameTs(self._now()))
            await self._close(self._aggregator.pending_buckets())
        self.exporters.shutdown()

    # --- Calibration and reconfiguration ------------------------------------

    async def snapshot(
        self,
        camera_id: CameraId,
        *,
        timeout: float = SNAPSHOT_TIMEOUT_S,  # noqa: ASYNC109 - bounds a thread, not a task
    ) -> bytes:
        """One encoded frame from a running camera, for the calibration view (§13).

        The frame is grabbed by the worker that already owns the stream, so no second
        RTSP session is opened and `muster.api` never imports a codec. What comes back is
        held in memory, streamed once, and never persisted.

        A camera with no live worker is refused rather than served from anywhere else. An
        operator calibrating a camera they have not got streaming yet is the wrong order,
        and the alternative — a second out-of-band capture path — is a second thing that
        can quietly write a frame somewhere.
        """
        handle = self._handle_for(camera_id)
        if handle is None:
            msg = f"camera {camera_id!r} is not configured"
            raise SnapshotUnavailableError(msg)
        if not handle.is_alive():
            msg = f"camera {camera_id!r} is {handle.state.value} — no snapshot available"
            raise SnapshotUnavailableError(msg)

        request_id = secrets.token_hex(8)
        if not handle.request_snapshot(request_id):
            msg = f"camera {camera_id!r} is not accepting requests"
            raise SnapshotUnavailableError(msg)

        # The blocking wait runs off the loop thread: the API serves this route and the
        # supervisor's tick shares that loop, so blocking here stalls every camera's
        # bucket close for as long as a camera takes to produce a frame.
        #
        # `asyncio.timeout` cannot replace the parameter above (ASYNC109): it would
        # abandon the `to_thread` call rather than end it — a thread blocked on a
        # `Queue.get` is not cancellable — leaving a thread alive holding the reply. The
        # deadline has to be inside the blocking call, which is where it is.
        reply = await asyncio.to_thread(handle.take_snapshot, request_id, timeout)
        if reply is None:
            msg = f"camera {camera_id!r} did not answer in {timeout:g}s"
            raise SnapshotUnavailableError(msg)
        if reply.jpeg is None:
            msg = reply.error or f"camera {camera_id!r} could not produce a snapshot"
            raise SnapshotUnavailableError(msg)
        return reply.jpeg

    async def reload(self, config: MusterConfig) -> None:
        """Validate-then-swap: rebuild geometry and push it to every live worker.

        The config is compiled here, before anything is sent, so a config that cannot
        compile fails where an operator can be told rather than inside a worker that can
        only die. Each worker then closes what its old geometry had open and adopts the
        new — see `GeometryAnalytics.reconfigure` for why closing is not optional.

        **This is the geometry half only.** Pushing new budgets to workers is P3.4's, and
        it rides on the same channel.
        """
        geometry = SiteGeometry.compile(config)
        self._aggregator.retarget(build_registry(geometry), heatmaps=HeatmapAccumulator(geometry))
        for handle in self._handles:
            if handle.is_alive():
                handle.request_reconfigure(geometry)

    def _handle_for(self, camera_id: CameraId) -> WorkerHandle | None:
        return next((h for h in self._handles if h.camera_id == camera_id), None)

    # --- The tick -----------------------------------------------------------

    async def tick(self) -> None:
        """One pass: take what the workers said, advance the clock, close what is due."""
        for handle in self._handles:
            # Heartbeats are drained on the tick because the tick is the only thing that
            # runs unattended. Draining them where `/healthz` is served instead would
            # make a camera's freshness depend on someone loading the dashboard.
            handle.drain_heartbeats()
        await self.drain_once(timeout=0.0)
        now = self._now()
        # The aggregator advances on the timestamps it is fed, so a zone the last person
        # left resolves here or not at all.
        self._aggregator.flush(FrameTs(now))
        await self._close(self._due_buckets(now))
        self._trim(now, self._monotonic())

    async def drain_once(
        self,
        *,
        timeout: float,  # noqa: ASYNC109 - see below; `asyncio.timeout` cannot bound this
        expected: int | None = None,
    ) -> list[RawEvent]:
        """Take what is waiting on every worker's queue and fold it in.

        Per-worker queues mean one noisy camera cannot starve another's events, and the
        round-robin here keeps that true when several are busy at once.

        `timeout` bounds a poll over `multiprocessing.Queue`s, which are not awaitables:
        there is nothing for `asyncio.timeout` to cancel, and wrapping the poll in one
        would leave the underlying `get_nowait` loop running regardless.
        """
        received: list[RawEvent] = []
        # A REAL clock, not the injected one. This deadline is a safety bound on polling
        # real queues for events a real process may never send; the injected clock is for
        # policy (how often work repeats) and a test is entitled to freeze it. Reading it
        # here made the bound unreachable under a frozen clock, so a worker that failed to
        # deliver spun this loop forever instead of timing out -- which is exactly how it
        # hung CI rather than failing it.
        deadline = time.monotonic() + timeout
        while True:
            batch = [event for handle in self._handles for event in handle.drain(DRAIN_LIMIT)]
            received.extend(batch)
            if expected is not None and len(received) >= expected:
                break
            if time.monotonic() >= deadline:
                break
            if not batch and timeout > 0.0:
                await asyncio.sleep(0.01)
        self._absorb(received)
        return received

    def _absorb(self, events: list[RawEvent]) -> None:
        loggable: list[RawEvent] = []
        for event in events:
            if self._is_late(event):
                self.late_events += 1
                continue
            self._aggregator.ingest(event)
            if event.kind in _DENSE_KINDS or event.track_id is None:
                # The raw log is a log of per-track transitions, and `events.track_id` is
                # NOT NULL. Counting the skip is what keeps the omission visible.
                self.skipped_untracked_events += 1
            else:
                loggable.append(event)
        if loggable:
            self._store.append_events(loggable)

    def _is_late(self, event: RawEvent) -> bool:
        return self._closed_through is not None and bucket_of(event.ts) <= self._closed_through

    # --- Closing ------------------------------------------------------------

    def _due_buckets(self, now: datetime) -> list[MinuteBucket]:
        cutoff = now - timedelta(seconds=BUCKET_S + CLOSE_LAG_S)
        return [bucket for bucket in self._aggregator.pending_buckets() if bucket <= cutoff]

    async def _close(self, buckets: list[MinuteBucket]) -> None:
        for bucket in buckets:
            rows = self._aggregator.close_bucket(bucket)
            grids = self._aggregator.close_grids(bucket)
            if rows:
                # Written on the loop thread, not in an executor. §9 says the blocking
                # SQLite calls belong in one, but `sqlite3` connections are thread-affine
                # (`check_same_thread`), so `to_thread` here raises ProgrammingError
                # against the store P2.5 built. A minute's rows are a handful of upserts
                # into a WAL database; moving them off-thread would need the store to own
                # a dedicated writer thread, which is a change to the store's contract
                # rather than to this file.
                self._store.upsert_metrics(rows)
                # AFTER the write, never before: an exporter that announces a number the
                # store rejected has told the outside world something the engine does not
                # believe. The fan-out contains its own failures, so a dead broker cannot
                # turn into a missed bucket here (§12).
                self.exporters.on_metrics(rows)
            if grids:
                # Not fanned out: an exporter takes scalar rows, and a 2 KB blob per zone
                # per minute is not something an MQTT topic or a Prometheus gauge has any
                # use for. The dashboard and the sync client read it from the store.
                self._store.upsert_heatmaps(grids)
            self._closed_through = bucket
            self._aggregator.forget_before(MinuteBucket(bucket + timedelta(seconds=BUCKET_S)))

    # --- Retention ----------------------------------------------------------

    def _trim(self, now: datetime, monotonic: float) -> None:
        """Enforce `storage.event_retention_hours`, at most once an interval.

        Two clocks, deliberately: the *cutoff* is wall-clock, because that is what an
        event's timestamp is measured against, while *how often* is monotonic, because a
        wall clock that steps backwards over an NTP correction would stop trimming.

        The first tick always trims, so an engine that was off for a week expires its
        backlog on startup rather than an hour into the run.
        """
        if self._last_trim_at is not None and monotonic - self._last_trim_at < TRIM_INTERVAL_S:
            return
        self._last_trim_at = monotonic
        self.trim_runs += 1
        window = timedelta(hours=self._config.storage.event_retention_hours)
        self.trimmed_events += self._store.trim(before=now - window)

    # --- Keeping workers alive ----------------------------------------------

    @property
    def live_workers(self) -> int:
        """How many workers are running. A health signal, and what lets a caller wait
        for a crash to have actually happened rather than assume it has."""
        return sum(1 for handle in self._handles if handle.is_alive())

    def camera_reports(self) -> dict[CameraId, WorkerReport]:
        """Per-camera state for `/healthz`, for the workers this supervisor runs.

        A camera disabled in config has no worker and is deliberately absent rather than
        reported as broken — describing a camera the operator switched off as `BACKOFF`
        is how a health endpoint pages someone at 3am about a decision they made. Naming
        it `DISABLED` needs config, which the health builder has and this does not.
        """
        return {handle.camera_id: handle.report() for handle in self._handles}

    async def supervise(self, *, monotonic: float) -> None:
        """Restart what died, once its backoff has elapsed.

        A camera whose URL is wrong fails immediately and forever, so respawning it in a
        tight loop would spend the whole box on one broken stream.
        """
        for handle in self._handles:
            if handle.is_alive():
                continue
            if not handle.death_noted:
                handle.note_failure(monotonic)
                handle.death_noted = True
                continue
            if handle.restart_due(monotonic):
                handle.start()
                handle.death_noted = False
                self.restarts += 1
