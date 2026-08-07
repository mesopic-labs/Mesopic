"""RTSP/ONVIF ingest over PyAV (FFmpeg).

Opens with TCP transport (UDP packet loss on a cheap LAN produces artefacts a detector
reads as noise), a low-latency flag set, and a hardware-decode probe that falls back to
software silently.

Owns the reconnect state machine — CONNECT -> STREAMING -> {STALLED, BACKOFF} -> CONNECT
— with exponential backoff plus jitter, capped, so a site-wide outage does not make every
camera reconnect in lockstep. A watchdog treats "socket open but no frames" (wedged
camera firmware) exactly like a drop.

Implements P1.2.
"""

from __future__ import annotations

from collections.abc import Iterator

from muster.types import CameraId, DecodedFrame


class RtspFrameSource:
    """A `FrameSource` backed by an RTSP URL."""

    def __init__(self, camera_id: CameraId, url: str, *, stall_timeout_s: float = 5.0) -> None:
        raise NotImplementedError

    def frames(self) -> Iterator[DecodedFrame]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
