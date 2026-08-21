"""`RtspFrameSource` — decode, timestamps, and the drop that is a routine event.

The real thing talks to a camera, so the container is injected: these tests drive the
decode loop with hand-built frames and no FFmpeg in the loop at all. What cannot be
faked away — that the PyAV options really do say TCP — is asserted separately against
the default opener.

Implements the test half of P1.2.
"""

from __future__ import annotations

import socket
import traceback
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any

import av
import numpy as np
import pytest

from mesopic.errors import StreamDropped
from mesopic.ingest import FrameSource
from mesopic.ingest.rtsp import RtspFrameSource, _open_rtsp
from mesopic.types import CameraId, DecodedFrame

CAMERA = CameraId("front-door")

# The `user:pass@` placeholder documented in .gitleaks.toml, on the RFC 2606 reserved
# TLD so it can never resolve. Credential-shaped on purpose — it is the fixture that
# proves credentials do not leak — but it is the one spelling the secret scanner
# allowlists, so the rule forbidding real RTSP URLs stays sharp.
URL = "rtsp://user:pass@camera.invalid:554/stream"

# The dev harness `make test-stream` serves this; loopback, no credentials.
SYNTHETIC_URL = "rtsp://127.0.0.1:8554/synthetic"


def _synthetic_stream_is_up() -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", 8554)) == 0


class FakeVideoFrame:
    """The slice of a PyAV `VideoFrame` the source actually touches."""

    def __init__(self, time: float | None, *, width: int = 1920, height: int = 1080) -> None:
        self.time = time
        self.width = width
        self.height = height
        self.requested_format: str | None = None

    def to_ndarray(self, *, format: str) -> Any:  # noqa: A002 - PyAV's own parameter name
        self.requested_format = format
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


class FakeContainer:
    """A stand-in for an open PyAV container over an RTSP stream."""

    def __init__(self, frames: list[FakeVideoFrame]) -> None:
        self._frames = frames
        self.closed = False

    def decode(self, *, video: int) -> Iterator[FakeVideoFrame]:
        assert video == 0
        yield from self._frames

    def close(self) -> None:
        self.closed = True


class ExplodingContainer(FakeContainer):
    """Yields what it has, then fails the way a dropped network stream does."""

    def __init__(self, frames: list[FakeVideoFrame], error: Exception) -> None:
        super().__init__(frames)
        self._error = error

    def decode(self, *, video: int) -> Iterator[FakeVideoFrame]:
        assert video == 0
        yield from self._frames
        raise self._error


def take(frames: Iterator[DecodedFrame], count: int) -> list[DecodedFrame]:
    """Pull `count` frames and stop, leaving the stream mid-flight.

    Draining a source to exhaustion raises `StreamDropped` by design, so a test that
    only cares about the frames must stop short of the end.
    """
    taken: list[DecodedFrame] = []
    for frame in frames:
        taken.append(frame)
        if len(taken) == count:
            break
    return taken


def test_frames_yields_one_decoded_frame_per_video_frame() -> None:
    container = FakeContainer([FakeVideoFrame(0.0), FakeVideoFrame(0.04)])
    source = RtspFrameSource(CAMERA, URL, opener=lambda _url: container)

    frames = take(source.frames(), 2)

    assert [f.camera_id for f in frames] == [CAMERA, CAMERA]
    assert frames[0].width == 1920
    assert frames[0].height == 1080
    assert frames[0].image.shape == (1080, 1920, 3)


def test_frames_are_decoded_as_bgr_for_the_detector() -> None:
    """The detector's graph takes raw BGR; converting anywhere else is a second convention."""
    frame = FakeVideoFrame(0.0)
    source = RtspFrameSource(CAMERA, URL, opener=lambda _url: FakeContainer([frame]))

    take(source.frames(), 1)

    assert frame.requested_format == "bgr24"


def test_reconnecting_releases_the_previous_container() -> None:
    """`frames()` is re-entered on every reconnect, and must not strand the old socket.

    The drop-and-retry loop calls `frames()` again after `StreamDropped`. Without this,
    each reconnect abandons an open container and the process leaks a socket per drop —
    on a flaky camera that is a slow death over days, which is exactly how long it would
    take to notice.
    """
    first = FakeContainer([FakeVideoFrame(0.0)])
    second = FakeContainer([FakeVideoFrame(0.0)])
    containers = iter([first, second])
    source = RtspFrameSource(CAMERA, URL, opener=lambda _url: next(containers))

    take(source.frames(), 1)
    take(source.frames(), 1)

    assert first.closed, "the previous connection's container was abandoned"


def test_default_opener_demands_tcp_transport_and_low_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UDP loss on a cheap LAN produces artefacts a detector reads as noise (§4)."""
    captured: dict[str, Any] = {}

    def fake_open(url: str, **kwargs: Any) -> FakeContainer:
        captured["url"] = url
        captured.update(kwargs)
        return FakeContainer([])

    monkeypatch.setattr(av, "open", fake_open)

    _open_rtsp(URL, open_timeout_s=4.0)

    assert captured["url"] == URL
    assert captured["options"]["rtsp_transport"] == "tcp"
    assert captured["options"]["flags"] == "low_delay"
    assert captured["options"]["fflags"] == "nobuffer"
    assert captured["timeout"] == 4.0


def test_capture_timestamps_track_stream_time_not_wall_clock() -> None:
    """The property the sampler depends on (sampler.py, ADR-0003).

    The fake clock never advances, so if these timestamps were read off the wall clock
    all three frames would collide on one instant and a post-reconnect burst of buffered
    frames would look simultaneous — which is exactly the detection storm the sampler's
    timestamp gate exists to prevent.
    """
    anchor = datetime(2026, 8, 9, 12, 0, 0, tzinfo=UTC)
    container = FakeContainer([FakeVideoFrame(0.0), FakeVideoFrame(0.5), FakeVideoFrame(2.0)])
    source = RtspFrameSource(CAMERA, URL, opener=lambda _url: container, now=lambda: anchor)

    stamps = [frame.ts for frame in take(source.frames(), 3)]

    assert stamps == [
        anchor,
        anchor + timedelta(seconds=0.5),
        anchor + timedelta(seconds=2.0),
    ]
    assert all(ts.tzinfo == UTC for ts in stamps)


def test_stream_ending_raises_stream_dropped() -> None:
    """A drop is a routine event with a defined response, not a crash (§4)."""
    container = FakeContainer([FakeVideoFrame(0.0)])
    source = RtspFrameSource(CAMERA, URL, opener=lambda _url: container)

    with pytest.raises(StreamDropped):
        list(source.frames())


def test_decode_error_mid_stream_raises_stream_dropped() -> None:
    container = ExplodingContainer([FakeVideoFrame(0.0)], OSError("connection reset"))
    source = RtspFrameSource(CAMERA, URL, opener=lambda _url: container)

    with pytest.raises(StreamDropped):
        list(source.frames())


def test_stream_dropped_never_leaks_the_rtsp_url() -> None:
    """An RTSP URL carries the camera's credentials, so it may not reach a log line.

    FFmpeg puts the URL it was given into its own error strings, so the leak vector is
    not our message but the *chained* cause: anything logging this with `exc_info` would
    print the credentials. The whole rendered chain has to be clean, not just our half.
    """
    container = ExplodingContainer([], OSError(f"Connection refused: {URL}"))
    source = RtspFrameSource(CAMERA, URL, opener=lambda _url: container)

    with pytest.raises(StreamDropped) as excinfo:
        list(source.frames())

    rendered = "".join(traceback.format_exception(excinfo.value))
    assert URL not in rendered
    assert "camera.invalid" not in rendered
    assert CAMERA in rendered, "the camera must still be identifiable"
    assert "OSError" in rendered, "the failure type is what makes a drop debuggable"


def test_rtsp_source_satisfies_the_frame_source_protocol() -> None:
    """Three ingest paths, one interface, so nothing downstream knows the difference.

    A structural guard rather than a behaviour test: the assignment is what `mypy
    --strict` checks, and it fails the build if `frames()` or `close()` ever drift.
    """
    source: FrameSource = RtspFrameSource(CAMERA, URL, opener=lambda _url: FakeContainer([]))

    assert isinstance(source, RtspFrameSource)


def test_open_failure_raises_stream_dropped_without_leaking_credentials() -> None:
    """The commonest failure of all — wrong password, camera offline, bad path.

    Verified against a real stream: PyAV raises `av.error.HTTPBadRequestError` whose
    message is `Server returned 400 Bad Request: '<the full URL>'`, password included.
    Left unwrapped that propagates straight out of `frames()` and into whatever logs it.
    """

    def refuse(_url: str) -> FakeContainer:
        message = f"Server returned 400 Bad Request: '{URL}'"
        raise OSError(message)

    source = RtspFrameSource(CAMERA, URL, opener=refuse)

    with pytest.raises(StreamDropped) as excinfo:
        list(source.frames())

    rendered = "".join(traceback.format_exception(excinfo.value))
    assert URL not in rendered
    assert "camera.invalid" not in rendered
    assert CAMERA in rendered


def test_close_releases_the_container() -> None:
    container = FakeContainer([FakeVideoFrame(0.0), FakeVideoFrame(0.04)])
    source = RtspFrameSource(CAMERA, URL, opener=lambda _url: container)
    take(source.frames(), 1)

    source.close()

    assert container.closed


def test_close_is_idempotent_and_safe_before_connecting() -> None:
    """Callers close on paths that never opened — a failed connect unwinds here too."""
    container = FakeContainer([FakeVideoFrame(0.0)])
    source = RtspFrameSource(CAMERA, URL, opener=lambda _url: container)

    source.close()
    take(source.frames(), 1)
    source.close()
    source.close()

    assert container.closed


def test_stall_timeout_becomes_the_socket_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A socket that stays open while frames stop is a wedged camera, not a healthy one.

    FFmpeg enforces this one: without a read timeout the decode call blocks forever and
    the camera never transitions out of STREAMING, so `/healthz` reports a dead camera
    as live.
    """
    captured: dict[str, Any] = {}

    def fake_open(url: str, **kwargs: Any) -> FakeContainer:
        captured.update(kwargs)
        return FakeContainer([])

    monkeypatch.setattr(av, "open", fake_open)
    source = RtspFrameSource(CAMERA, URL, stall_timeout_s=3.0)

    with pytest.raises(StreamDropped):
        list(source.frames())

    assert captured["timeout"] == 3.0


@pytest.mark.integration
@pytest.mark.slow
def test_decodes_a_real_rtsp_stream_with_monotonic_capture_time() -> None:
    """P1.2's acceptance criterion, against mediamtx instead of a fake.

    Everything above this line proves the decode loop's logic; only this proves the
    PyAV options, the TCP transport and the presentation clock survive contact with a
    real server. Bring the stream up with `make test-stream`.
    """
    if not _synthetic_stream_is_up():
        pytest.skip("no RTSP server on :8554 — run `make test-stream`")

    source = RtspFrameSource(CameraId("synthetic"), SYNTHETIC_URL, stall_timeout_s=10.0)
    try:
        frames = take(source.frames(), 40)
    finally:
        source.close()

    stamps = [frame.ts for frame in frames]
    assert len(frames) == 40
    assert frames[0].image.shape == (1080, 1920, 3)
    assert frames[0].image.dtype == np.uint8
    assert all(a <= b for a, b in pairwise(stamps))
    assert all(ts.tzinfo is not None for ts in stamps)

    # The source is 1080p25, so capture time spread over ~40 frames has to come out
    # near 25 fps. Reading the wall clock per frame would instead report however fast
    # this machine happened to decode, which on a warm buffer is far higher.
    span_s = (stamps[-1] - stamps[0]).total_seconds()
    assert 20.0 <= (len(frames) - 1) / span_s <= 30.0
