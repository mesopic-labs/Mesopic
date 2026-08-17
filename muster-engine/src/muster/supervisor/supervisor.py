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
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from muster.aggregator.aggregator import EXIT_GRACE_S, Aggregator, bucket_of
from muster.analytics.metrics import build_registry
from muster.analytics.site_geometry import SiteGeometry
from muster.config.schema import MusterConfig
from muster.store.store import Store
from muster.supervisor.handle import WorkerEntry, WorkerHandle
from muster.supervisor.worker import run_camera_worker
from muster.types import FrameTs, MinuteBucket, RawEvent

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
        self._aggregator = Aggregator(
            build_registry(SiteGeometry.compile(config)),
            dwell_min_s=config.thresholds.dwell_min_s,
        )
        self._handles = [
            WorkerHandle(camera.camera_id, config=config, entry=entry)
            for camera in config.cameras
            if camera.enabled
        ]
        self._closed_through: MinuteBucket | None = None
        self._stopping = False
        self.skipped_untracked_events = 0
        """Events the raw log cannot hold — trackless occupancy samples (ADR-0016)."""
        self.late_events = 0
        """Events for a bucket already closed and forgotten. Dropped, never re-folded."""
        self.restarts = 0

    # --- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
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

    async def reload(self, config: MusterConfig) -> None:
        """Validate-then-swap: rebuild geometry and push new budgets without dropping streams."""
        raise NotImplementedError

    # --- The tick -----------------------------------------------------------

    async def tick(self) -> None:
        """One pass: take what the workers produced, advance the clock, close what is due."""
        await self.drain_once(timeout=0.0)
        now = self._now()
        # The aggregator advances on the timestamps it is fed, so a zone the last person
        # left resolves here or not at all.
        self._aggregator.flush(FrameTs(now))
        await self._close(self._due_buckets(now))

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
        deadline = self._monotonic() + timeout
        while True:
            batch = [event for handle in self._handles for event in handle.drain(DRAIN_LIMIT)]
            received.extend(batch)
            if expected is not None and len(received) >= expected:
                break
            if not batch and self._monotonic() >= deadline:
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
            if event.track_id is None:
                # The raw log is a log of per-track facts and `events.track_id` is NOT
                # NULL. Counting the skip is what keeps the omission visible (ADR-0016).
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
            if rows:
                # Written on the loop thread, not in an executor. §9 says the blocking
                # SQLite calls belong in one, but `sqlite3` connections are thread-affine
                # (`check_same_thread`), so `to_thread` here raises ProgrammingError
                # against the store P2.5 built. A minute's rows are a handful of upserts
                # into a WAL database; moving them off-thread would need the store to own
                # a dedicated writer thread, which is a change to the store's contract
                # rather than to this file.
                self._store.upsert_metrics(rows)
            self._closed_through = bucket
            self._aggregator.forget_before(MinuteBucket(bucket + timedelta(seconds=BUCKET_S)))

    # --- Keeping workers alive ----------------------------------------------

    @property
    def live_workers(self) -> int:
        """How many workers are running. A health signal, and what lets a caller wait
        for a crash to have actually happened rather than assume it has."""
        return sum(1 for handle in self._handles if handle.is_alive())

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
