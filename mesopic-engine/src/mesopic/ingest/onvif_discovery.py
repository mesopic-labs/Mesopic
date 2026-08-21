"""WS-Discovery probe to enumerate cameras and their RTSP profile URIs on the LAN.

A **setup-time** concern only, so the config can be pre-populated (`mesopic discover`).
It never runs in the frame loop.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiscoveredCamera:
    """One camera found on the LAN. Credentials are never stored here."""

    address: str
    name: str
    rtsp_uri: str


def discover(timeout_s: float = 5.0) -> list[DiscoveredCamera]:
    """Probe the local network for ONVIF devices."""
    raise NotImplementedError
