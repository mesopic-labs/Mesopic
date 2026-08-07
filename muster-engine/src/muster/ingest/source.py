"""The one interface every frame source implements."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from muster.types import DecodedFrame


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
