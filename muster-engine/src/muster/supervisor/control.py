"""What the supervisor can say to a worker, and what comes back.

Until P3.8 this channel carried one string. Two features need more: the calibration view
needs a frame from the camera that is already streaming, and the zone editor needs
geometry to change without dropping the stream.

Every type here is a frozen dataclass of picklable parts, which is not decoration. The
`spawn` start method is pinned (see `handle`), so the child re-imports rather than
inheriting, and anything crossing this boundary must be picklable by construction — the
rule that also keeps a frame from ever being an argument.

`SiteGeometry` rather than `MusterConfig` rides on `Reconfigure`: the worker would only
compile the config into geometry anyway, and sending the compiled form means a config
that fails to compile fails in the supervisor, where an operator can be told, rather than
inside a worker that can only die.

Implements P3.8 (engine-architecture.md §9, §13).
"""

from __future__ import annotations

from dataclasses import dataclass

from muster.analytics.site_geometry import SiteGeometry

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


ControlMessage = Stop | Snapshot | Reconfigure


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


__all__ = [
    "STOP",
    "ControlMessage",
    "Reconfigure",
    "Snapshot",
    "SnapshotReply",
    "Stop",
]
