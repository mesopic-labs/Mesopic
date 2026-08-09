"""RTSP ingest over PyAV (FFmpeg).

Opens with TCP transport — UDP packet loss on a cheap LAN produces artefacts a detector
reads as noise — and low-latency flags, so no buffer accumulates that the sampler would
only discard.

Yields decoded frames until the stream ends, stalls, or errors, then raises
`StreamDropped`. A drop is a routine event with a defined response rather than a crash:
the caller reconnects by calling `frames()` again. The socket timeout doubles as the
stall watchdog, so "socket open but frames stopped" — wedged camera firmware — surfaces
as a drop instead of blocking forever.

The reconnect *policy* is deliberately not here. Backoff, jitter, and the `CameraState`
transitions belong to the camera worker that owns that state and reports it on
`/healthz`; this module's whole contract is to raise once the stream is gone.

Implements P1.2.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any, Protocol, cast

from muster.errors import StreamDropped
from muster.types import BgrImage, CameraId, DecodedFrame, FrameTs

PIXEL_FORMAT = "bgr24"
"""What the detector expects. Converting anywhere else would be a second convention."""

RTSP_OPTIONS = {
    # TCP: UDP loss on a cheap LAN reads as detector noise (engine-architecture §4).
    "rtsp_transport": "tcp",
    # Do not accumulate a buffer we would only throw away — the sampler decides
    # which frames survive, and a deep buffer just makes every frame staler.
    "fflags": "nobuffer",
    "flags": "low_delay",
}


class VideoFrame(Protocol):
    """The slice of a PyAV video frame this module actually touches.

    Narrow on purpose: it is what lets a test drive the decode loop with hand-built
    frames and no FFmpeg anywhere in the loop.
    """

    time: float | None
    """Presentation time in seconds from the start of the stream. `None` if unset."""

    def to_ndarray(self, *, format: str) -> Any: ...  # noqa: A002 - PyAV's parameter name


class VideoContainer(Protocol):
    """An open PyAV container."""

    def decode(self, *, video: int) -> Iterator[VideoFrame]: ...

    def close(self) -> None: ...


Opener = Callable[[str], VideoContainer]
"""Open a stream URL. Injected so the decode loop is testable without a camera."""

Clock = Callable[[], datetime]
"""Reads wall time once per connection, to anchor the stream's own clock to UTC."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _open_rtsp(url: str, *, open_timeout_s: float) -> VideoContainer:
    """Open an RTSP URL with the transport settings a cheap camera needs.

    The timeout is required rather than defaulted: FFmpeg exposes a single socket-timeout
    knob, the caller is the only one that knows what it should be, and a default here
    would quietly disagree with the stall timeout the source was configured with.

    Imported lazily so that constructing a source with an injected container — which is
    what the fast tests do — does not pay to import the FFmpeg bindings.
    """
    import av  # noqa: PLC0415

    container = av.open(url, options=RTSP_OPTIONS, timeout=open_timeout_s)
    # PyAV types `decode` as a positional overload set returning a union of video,
    # audio and subtitle frames. This module only ever calls it as `decode(video=0)`
    # and only ever sees video frames, and `VideoContainer` says exactly that much.
    # Narrowing once here keeps the mismatch at the boundary instead of forcing every
    # test fake to mirror PyAV's overloads.
    return cast("VideoContainer", container)


class RtspFrameSource:
    """A `FrameSource` backed by an RTSP URL."""

    def __init__(
        self,
        camera_id: CameraId,
        url: str,
        *,
        stall_timeout_s: float = 5.0,
        opener: Opener | None = None,
        now: Clock | None = None,
    ) -> None:
        self._camera_id = camera_id
        self._url = url
        # FFmpeg has one socket-timeout knob, and the stall timeout is what it means:
        # a read that outlives it is the wedged-firmware case, which counts as a drop.
        # Bound here rather than at call time so an injected opener stays a
        # one-argument callable.
        self._opener: Opener = (
            opener if opener is not None else partial(_open_rtsp, open_timeout_s=stall_timeout_s)
        )
        self._now: Clock = now if now is not None else _utc_now
        self._container: VideoContainer | None = None

    def frames(self) -> Iterator[DecodedFrame]:
        """Yield decoded frames with capture timestamps until the stream drops.

        Capture time is the stream's own presentation clock anchored to UTC once, at
        connect — not the wall clock read per frame. A camera that buffers and then
        bursts after a hiccup must hand downstream the spacing the frames were
        *captured* at, or the sampler's timestamp gate admits the whole burst at once.
        """
        # Every reconnect re-enters this method, so release any container left over from
        # the connection that just dropped. Without it a flaky camera strands one open
        # socket per drop for the life of the process.
        self.close()

        try:
            container = self._opener(self._url)
        except Exception as exc:  # noqa: BLE001 - PyAV raises a wide, unstable set
            reason = f"{type(exc).__name__} while connecting"
            raise self._dropped(reason) from None

        self._container = container
        anchor = self._now()
        offset = 0.0
        decoded = container.decode(video=0)

        while True:
            try:
                frame = next(decoded)
            except StopIteration:
                break
            except Exception as exc:  # noqa: BLE001 - PyAV raises a wide, unstable set
                reason = f"{type(exc).__name__} while decoding"
                raise self._dropped(reason) from None

            # A frame with no presentation time carries the previous one's offset
            # rather than collapsing to the anchor, so time never runs backwards.
            if frame.time is not None:
                offset = frame.time
            image: BgrImage = frame.to_ndarray(format=PIXEL_FORMAT)
            height, width = int(image.shape[0]), int(image.shape[1])
            yield DecodedFrame(
                camera_id=self._camera_id,
                ts=FrameTs(anchor + timedelta(seconds=offset)),
                image=image,
                width=width,
                height=height,
            )

        # Falling off the end of the decode loop is EOF: the camera closed the stream.
        reason = "stream ended"
        raise self._dropped(reason)

    def _dropped(self, reason: str) -> StreamDropped:
        """Build a `StreamDropped` that identifies the camera and never the URL.

        An RTSP URL carries the camera's credentials, and FFmpeg embeds the URL it was
        given in its own error strings — so the cause is deliberately *not* chained at
        the raise site. Anything logging this with `exc_info` would otherwise print
        those credentials. The exception type is carried in `reason` instead, which is
        what actually distinguishes a reset socket from a corrupt stream.
        """
        return StreamDropped(f"camera {self._camera_id!r}: {reason}")

    def close(self) -> None:
        """Tear down the container and release the socket.

        Idempotent, and safe on a source that never connected: a failed open and a
        caller reconnecting after a drop both unwind through here.
        """
        container, self._container = self._container, None
        if container is not None:
            container.close()
