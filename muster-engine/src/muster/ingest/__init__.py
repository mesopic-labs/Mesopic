"""Get decoded frames off a stream, reliably (engine-architecture.md §4).

Three code paths — direct RTSP, ONVIF-discovered RTSP, and Frigate-over-MQTT — behind
**one** `FrameSource` interface, so nothing downstream knows where a frame came from.

This package must not sample, detect, or interpret pixels.
"""

from __future__ import annotations

from muster.ingest.source import FrameSource

__all__ = ["FrameSource"]
