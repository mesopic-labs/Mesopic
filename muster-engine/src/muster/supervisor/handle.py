"""One camera's process, its bounded queue, and its restart policy.

Split out of `Supervisor` because these are three separable concerns with one owner: the
process lifecycle, the events crossing the boundary, and how long to wait before trying a
camera again. The supervisor composes handles; it does not reach inside one.

The start method is pinned to `spawn` rather than inherited from the platform default.
Two reasons, both of which bite silently otherwise: `fork` copies the parent's threads
and locks into a child that never ran their owners, which deadlocks under an asyncio loop
plus a SQLite connection; and `fork` would let a worker inherit parent state by accident,
so an argument that is not picklable — a frame, say — would keep working on Linux and
fail only on someone else's laptop. Under `spawn`, everything crossing the boundary must
be picklable by construction, and a worker entry point must be importable by name.

Implements P2.7 (engine-architecture.md §9).
"""

from __future__ import annotations

import multiprocessing
import queue
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import Queue

from muster.config.schema import MusterConfig
from muster.types import CameraId, CameraState, RawEvent

WorkerEntry = Callable[[CameraId, MusterConfig, "Queue[RawEvent]", "Queue[str]"], None]
"""What a camera worker process runs. Must be importable by name — see the module note."""


@dataclass(frozen=True, slots=True)
class WorkerReport:
    """Everything the supervisor knows about one camera, for `/healthz` (§15).

    Two fields rather than a bare state because the count is what explains the state: a
    camera in `BACKOFF` with one failure is a stream that hiccuped, and the same camera
    with forty is a URL that has been wrong since install.
    """

    state: CameraState
    consecutive_failures: int


_SPAWN = multiprocessing.get_context("spawn")


class WorkerHandle:
    """The supervisor's grip on one camera worker."""

    STOP = "stop"
    """The control message a worker loops on. Its value is part of the worker contract."""

    QUEUE_SIZE = 1000
    """Bounded on purpose: this is the backpressure valve, not a buffer (§9)."""

    BASE_BACKOFF_S = 1.0
    MAX_BACKOFF_S = 30.0
    """Capped so a camera that recovers after a long outage is still retried."""

    def __init__(
        self,
        camera_id: CameraId,
        *,
        config: MusterConfig,
        entry: WorkerEntry,
        queue_size: int = QUEUE_SIZE,
    ) -> None:
        self.camera_id = camera_id
        self.stopped_cleanly = False
        self.death_noted = False
        """Set when the supervisor has already counted this death, so one crash is one
        failure rather than one per tick until the backoff lapses."""
        self._config = config
        self._entry = entry
        self._events: Queue[RawEvent] = _SPAWN.Queue(maxsize=queue_size)
        self._control: Queue[str] = _SPAWN.Queue(maxsize=16)
        self._process: SpawnProcess | None = None
        self._failures = 0
        self._failed_at: float | None = None

    # --- Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self._process = _SPAWN.Process(
            target=self._entry,
            args=(self.camera_id, self._config, self._events, self._control),
            name=f"muster-worker-{self.camera_id}",
            daemon=True,
        )
        self._process.start()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def state(self) -> CameraState:
        """What `/healthz` reports for this camera (engine-architecture.md §15).

        **This is process liveness standing in for stream state, and it over-claims.** A
        worker whose RTSP connection is reconnecting inside its own loop is still a live
        process, and is reported here as `STREAMING`. The honest facts — last frame time
        and effective fps — live inside the worker and are never sent back to the
        supervisor, so `STALLED` is unreachable from here and `/healthz` serialises those
        two fields as `null`. A worker→supervisor status heartbeat is what replaces this
        with the real thing; until then this distinguishes "running" from "not running",
        which is what the restart policy already knows.
        """
        if self.is_alive():
            return CameraState.STREAMING
        return CameraState.BACKOFF

    @property
    def consecutive_failures(self) -> int:
        """Deaths since this worker last ran successfully. Reset by `note_started`."""
        return self._failures

    def report(self) -> WorkerReport:
        return WorkerReport(state=self.state, consecutive_failures=self._failures)

    def wait_exit(self, timeout: float) -> None:
        if self._process is not None:
            self._process.join(timeout)

    def request_stop(self) -> None:
        """Ask the worker to finish its current frame and return. Never blocks."""
        with suppress(queue.Full):
            self._control.put_nowait(self.STOP)

    def stop(self, timeout: float) -> None:
        """Ask first, kill second. A worker holding a camera socket deserves the ask."""
        if self._process is None:
            return
        self.request_stop()
        self.wait_exit(timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout)
        else:
            self.stopped_cleanly = True

    # --- Events -------------------------------------------------------------

    def drain(self, limit: int) -> list[RawEvent]:
        """Take what is waiting, up to `limit`. Never blocks — the loop has a tick."""
        events: list[RawEvent] = []
        while len(events) < limit:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return events

    def drain_blocking(self, expected: int, timeout: float) -> list[RawEvent]:
        """Wait for `expected` events. For tests and for a drain at shutdown."""
        events: list[RawEvent] = []
        while len(events) < expected:
            try:
                events.append(self._events.get(timeout=timeout))
            except queue.Empty:
                break
        return events

    # --- Restart policy -----------------------------------------------------

    def note_failure(self, now: float) -> None:
        self._failures += 1
        self._failed_at = now

    def note_started(self, now: float) -> None:
        """A worker that ran is not a worker that keeps failing."""
        del now  # only the reset matters; the timestamp is the caller's log line
        self._failures = 0
        self._failed_at = None

    def restart_due(self, now: float) -> bool:
        if self._failed_at is None:
            return True
        return now - self._failed_at >= self.backoff_s

    @property
    def backoff_s(self) -> float:
        if self._failures == 0:
            return 0.0
        grown = self.BASE_BACKOFF_S * float(2 ** (self._failures - 1))
        return min(grown, self.MAX_BACKOFF_S)
