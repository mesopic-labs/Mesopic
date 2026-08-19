"""One camera's process, its bounded queues, and its restart policy.

Split out of `Supervisor` because these are three separable concerns with one owner: the
process lifecycle, the messages crossing the boundary, and how long to wait before trying
a camera again. The supervisor composes handles; it does not reach inside one.

Since P3.7 a handle also holds the newest thing its worker said about itself, which is
what makes `STALLED` reachable: process liveness alone cannot tell a camera that is
counting from one whose stream has wedged.

The start method is pinned to `spawn` rather than inherited from the platform default.
Two reasons, both of which bite silently otherwise: `fork` copies the parent's threads
and locks into a child that never ran their owners, which deadlocks under an asyncio loop
plus a SQLite connection; and `fork` would let a worker inherit parent state by accident,
so an argument that is not picklable — a frame, say — would keep working on Linux and
fail only on someone else's laptop. Under `spawn`, everything crossing the boundary must
be picklable by construction, and a worker entry point must be importable by name.

Implements P2.7, extended by P3.7 (engine-architecture.md §9, §15).
"""

from __future__ import annotations

import multiprocessing
import queue
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import Queue

from muster.analytics.site_geometry import SiteGeometry
from muster.config.schema import MusterConfig
from muster.supervisor.control import (
    ControlMessage,
    Heartbeat,
    Reconfigure,
    Retarget,
    Snapshot,
    SnapshotReply,
    Stop,
    WorkerChannels,
)
from muster.types import CameraId, CameraState, FrameTs, RawEvent

WorkerEntry = Callable[[CameraId, MusterConfig, WorkerChannels], None]
"""What a camera worker process runs. Must be importable by name — see the module note."""


@dataclass(frozen=True, slots=True)
class WorkerReport:
    """Everything the supervisor knows about one camera, for `/healthz` (§15).

    The count is what explains the state: a camera in `BACKOFF` with one failure is a
    stream that hiccuped, and the same camera with forty is a URL that has been wrong
    since install.

    `last_frame_ts` and `effective_fps` come from the worker's own heartbeat (P3.7) and
    are `None` until it has sent one. They are deliberately **not** cleared when a worker
    goes `STALLED`: the time of the last frame is what says when it stopped, and the
    `state` beside it is what stops the number being read as current.
    """

    state: CameraState
    consecutive_failures: int
    last_frame_ts: FrameTs | None = None
    effective_fps: float | None = None


HEARTBEAT_QUEUE_SIZE = 4
"""Heartbeats buffered before the worker drops one. A heartbeat is a latest-value
signal, not a stream: the supervisor drains the whole queue every tick and keeps the
newest, so a dropped one costs an intermediate reading nobody reads and never a pulse."""

STALL_AFTER_S = 10.0
"""How long a live worker may say nothing before it is called stalled.

Ten missed heartbeats at `worker.HEARTBEAT_INTERVAL_S`. Generous on purpose: this decides
whether an operator is told a camera is broken, and a threshold tight enough to trip on a
loaded box would teach them to ignore it. It is also the grace a freshly spawned worker
gets to open its RTSP session before silence stops reading as `CONNECT`.
"""


def camera_state(
    *, alive: bool, since_last_beat_s: float | None, since_start_s: float
) -> CameraState:
    """What `/healthz` should say about one worker (§15).

    A free function because it is the whole rule and nothing else: given liveness and two
    durations it is total, and every branch is reachable without a process in a
    particular condition.

    `since_last_beat_s` is `None` when this worker has never reported. That is the case
    that must not collapse into the stale one — a worker still dialling and a worker that
    wedged after ten minutes are both silent, and only the second is a fault.
    """
    if not alive:
        return CameraState.BACKOFF
    if since_last_beat_s is None:
        return CameraState.CONNECT if since_start_s <= STALL_AFTER_S else CameraState.STALLED
    return CameraState.STREAMING if since_last_beat_s <= STALL_AFTER_S else CameraState.STALLED


SNAPSHOT_QUEUE_SIZE = 4
"""Replies buffered before the worker drops one. Tiny on purpose: a snapshot is answered
to a browser that is waiting right now, so a backlog of them is stale by definition and
holding several encoded frames is exactly the memory nobody budgeted for."""

_SPAWN = multiprocessing.get_context("spawn")


class WorkerHandle:
    """The supervisor's grip on one camera worker."""

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
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera_id = camera_id
        self.stopped_cleanly = False
        self.death_noted = False
        """Set when the supervisor has already counted this death, so one crash is one
        failure rather than one per tick until the backoff lapses."""
        self._config = config
        self._entry = entry
        self._monotonic = monotonic
        self._events: Queue[RawEvent] = _SPAWN.Queue(maxsize=queue_size)
        self._control: Queue[ControlMessage] = _SPAWN.Queue(maxsize=16)
        self._snapshots: Queue[SnapshotReply] = _SPAWN.Queue(maxsize=SNAPSHOT_QUEUE_SIZE)
        self._beats: Queue[Heartbeat] = _SPAWN.Queue(maxsize=HEARTBEAT_QUEUE_SIZE)
        self._channels = WorkerChannels(
            events=self._events,
            control=self._control,
            snapshots=self._snapshots,
            heartbeats=self._beats,
        )
        self._process: SpawnProcess | None = None
        self._failures = 0
        self._failed_at: float | None = None
        self._beat: Heartbeat | None = None
        self._beat_at: float | None = None
        """When the newest heartbeat was *drained*, on the monotonic clock. Receipt
        rather than the frame's own timestamp: a camera whose clock is skewed is not a
        camera that has stopped, and comparing a `FrameTs` against wall-now conflates the
        two."""
        self._started_at = monotonic()

    # --- Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        # A heartbeat belongs to a process, not to a camera. Left in place, a respawned
        # worker would inherit its predecessor's last reading and report a pulse it never
        # produced — fastest exactly when the camera is flapping.
        self._forget_heartbeat()
        self._started_at = self._monotonic()
        self._process = _SPAWN.Process(
            target=self._entry,
            args=(self.camera_id, self._config, self._channels),
            name=f"muster-worker-{self.camera_id}",
            daemon=True,
        )
        self._process.start()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def state(self) -> CameraState:
        """What `/healthz` reports for this camera (engine-architecture.md §15).

        Process liveness alone over-claimed: a worker whose RTSP connection had wedged
        was still a live process and read as `STREAMING`. Since P3.7 the worker reports
        itself, so silence is visible and `STALLED` is reachable — see `camera_state` for
        the rule and `drain_heartbeats` for what feeds it.
        """
        return camera_state(
            alive=self.is_alive(),
            since_last_beat_s=None if self._beat_at is None else self._monotonic() - self._beat_at,
            since_start_s=self._monotonic() - self._started_at,
        )

    @property
    def consecutive_failures(self) -> int:
        """Deaths since this worker last ran successfully. Reset by `note_started`."""
        return self._failures

    def report(self) -> WorkerReport:
        return WorkerReport(
            state=self.state,
            consecutive_failures=self._failures,
            last_frame_ts=None if self._beat is None else self._beat.last_frame_ts,
            effective_fps=None if self._beat is None else self._beat.effective_fps,
        )

    def drain_heartbeats(self) -> None:
        """Take every heartbeat waiting and keep the newest. Never blocks.

        The whole queue is drained rather than one item taken: a worker beats faster than
        the supervisor ticks, so taking the first would report a reading that is already
        superseded — and would do it under exactly the load that makes several pile up.
        """
        latest: Heartbeat | None = None
        while True:
            try:
                latest = self._beats.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._beat = latest
            self._beat_at = self._monotonic()

    def _forget_heartbeat(self) -> None:
        self._beat = None
        self._beat_at = None

    def wait_exit(self, timeout: float) -> None:
        if self._process is not None:
            self._process.join(timeout)

    def request_stop(self) -> None:
        """Ask the worker to finish its current frame and return. Never blocks."""
        with suppress(queue.Full):
            self._control.put_nowait(Stop())

    # --- Calibration and reconfiguration ------------------------------------

    def request_snapshot(self, request_id: str) -> bool:
        """Ask for one encoded frame. Returns whether the request was accepted.

        A full control queue means a worker that is not draining it, and saying so is
        better than queueing a request nobody will answer.
        """
        try:
            self._control.put_nowait(Snapshot(request_id=request_id))
        except queue.Full:
            return False
        return True

    def take_snapshot(self, request_id: str, timeout: float) -> SnapshotReply | None:
        """Wait for the reply to `request_id`, discarding any that are stale.

        **Correlation is the point.** A caller that timed out and retried would otherwise
        read the previous request's answer — a stale frame presented as the live one,
        which is precisely the wrong thing to draw a counting line on. Replies for other
        requests are dropped rather than requeued: nobody is waiting for them.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                reply = self._snapshots.get(timeout=remaining)
            except queue.Empty:
                return None
            if reply.request_id == request_id:
                return reply

    def request_reconfigure(self, geometry: SiteGeometry) -> bool:
        """Ask the worker to close what is open and adopt `geometry`. Never blocks."""
        try:
            self._control.put_nowait(Reconfigure(geometry=geometry))
        except queue.Full:
            return False
        return True

    def request_retarget(self, fps_min: float, fps_max: float) -> bool:
        """Ask the worker to adopt a new fps envelope. Never blocks."""
        try:
            self._control.put_nowait(Retarget(fps_min=fps_min, fps_max=fps_max))
        except queue.Full:
            return False
        return True

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
