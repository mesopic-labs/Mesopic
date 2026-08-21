"""The interfaces a camera's input implements.

There are two, not one, because there are two kinds of upstream. Most sources hand over
frames and the engine detects and tracks them. Frigate has already done both, so it hands
over tracks and the detector and tracker are skipped entirely (engine-architecture.md §4,
§7) — a source that yielded frames it does not have could only fake them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from mesopic.types import DecodedFrame, FrameTs, Track


class FrameSource(Protocol):
    """One camera's frames.

    Blocking iterator. Raises `StreamDropped` on loss — a routine event the caller
    responds to with the reconnect/backoff state machine, not a crash.
    """

    def frames(self) -> Iterator[DecodedFrame]:
        """Yield decoded frames with capture timestamps until the stream drops."""
        ...

    def close(self) -> None:
        """Tear down the container and release the socket."""
        ...


class TrackSource(Protocol):
    """One camera's tracks, from an upstream that already detected them.

    Yields a **tick at a time — every track the camera can currently see** — not one
    track per message. `GeometryAnalytics.on_tracks` diffs zone membership against the
    previous call, so a partial tick reads as everybody else having left.

    The timestamp rides alongside because a tick with no tracks in it still has a time,
    and a quiet camera still has to report that it is quiet.
    """

    def ticks(self) -> Iterator[tuple[FrameTs, list[Track]]]:
        """Yield `(capture time, every live track)` until the source is closed."""
        ...

    def close(self) -> None:
        """Release the connection."""
        ...
