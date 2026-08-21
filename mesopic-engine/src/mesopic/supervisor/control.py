"""What the supervisor can say to a worker, and what comes back.

Until P3.8 this channel carried one string. Two features needed more: the calibration
view needs a frame from the camera that is already streaming, and the zone editor needs
geometry to change without dropping the stream. P3.7 added the return direction proper —
a `Heartbeat` on its own queue, so a camera that wedges is visible rather than merely
still running.

Every type here is a frozen dataclass of picklable parts, which is not decoration. The
`spawn` start method is pinned (see `handle`), so the child re-imports rather than
inheriting, and anything crossing this boundary must be picklable by construction — the
rule that also keeps a frame from ever being an argument.

`SiteGeometry` rather than `MesopicConfig` rides on `Reconfigure`: the worker would only
compile the config into geometry anyway, and sending the compiled form means a config
that fails to compile fails in the supervisor, where an operator can be told, rather than
inside a worker that can only die.

Implements P3.8 and P3.7 (engine-architecture.md §9, §13, §15).
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.queues import Queue

from mesopic.analytics.site_geometry import SiteGeometry
from mesopic.types import FrameTs, RawEvent

STOP = "stop"
"""The stop message's wire identity, unchanged from P2.7. Kept as a name because it is
part of the worker contract and a rename would be a silent protocol change."""


@dataclass(frozen=True, slots=True)
class Stop:
    """Finish the current frame and return."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Encode the frame in hand and put it on the reply queue.

    `request_id` correlates the reply. Without it a caller that timed out and retried
    would read the previous request's answer — a stale frame presented as the live one,
    which is exactly the wrong thing to draw a counting line on.
    """

    request_id: str


@dataclass(frozen=True, slots=True)
class Reconfigure:
    """Close what is open, then adopt this geometry (see `GeometryAnalytics.reconfigure`)."""

    geometry: SiteGeometry


@dataclass(frozen=True, slots=True)
class Retarget:
    """Adopt a new fps envelope without restarting (P3.4).

    Two floats rather than a `BudgetConfig`, for the same reason `Reconfigure` carries
    compiled geometry: the worker uses the envelope and nothing else in that section, and
    a message carrying only what its receiver reads cannot grow a second meaning later.

    `cpu_budget` is deliberately absent. It is the supervisor's scheduling input, not a
    worker's — a worker sheds on its own outbox pressure (`Backpressure`), which is the
    local signal, and handing it a site-wide fraction would invite a second opinion about
    the same decision.
    """

    fps_min: float
    fps_max: float


ControlMessage = Stop | Snapshot | Reconfigure | Retarget


@dataclass(frozen=True, slots=True)
class Heartbeat:
    """One worker saying what it has actually managed to do lately (§15).

    Sent worker→supervisor on its own queue, never on the events queue: that queue is the
    backpressure valve (`WorkerHandle.QUEUE_SIZE`), and a chatty camera putting status on
    it would make a camera evict its own metrics to report its health.

    `effective_fps` is the rate **achieved** — admitted frames over the window just
    elapsed — not `Backpressure.target_fps`, which is the rate asked for. The two diverge
    exactly when something is wrong, which is the only time anybody reads this.
    """

    last_frame_ts: FrameTs
    effective_fps: float


@dataclass(frozen=True, slots=True)
class SnapshotReply:
    """One answer to one `Snapshot`.

    `jpeg` is `None` when the worker could not produce one, and `error` says why in terms
    an operator can act on. A reply always comes back: a caller waiting on a queue that
    silently never answers cannot tell a slow camera from a dead one.
    """

    request_id: str
    jpeg: bytes | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerChannels:
    """Every queue a worker is given, as one argument.

    Bundled because the entry-point contract had grown twice — snapshots in P3.8,
    heartbeats in P3.7 — and each time every scripted worker in the tests had to grow a
    parameter it did not use. A channel added from here on changes this class and the
    workers that read the new channel, and nothing else.

    Passed whole through `Process(args=...)`, so it is pickled by the spawn pickler with
    the queues inside it. That is the one context in which a `multiprocessing.Queue` may
    be pickled at all, which is also why this class must never be sent anywhere else.
    """

    events: Queue[RawEvent]
    control: Queue[ControlMessage]
    snapshots: Queue[SnapshotReply]
    heartbeats: Queue[Heartbeat]


__all__ = [
    "STOP",
    "ControlMessage",
    "Heartbeat",
    "Reconfigure",
    "Retarget",
    "Snapshot",
    "SnapshotReply",
    "Stop",
    "WorkerChannels",
]
