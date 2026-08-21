"""Mesopic — video-intelligence for the cameras you already own.

The engine runs on the customer's box. Footage never leaves the building: frames are
discarded the instant they are processed, and only anonymous foot-points and the metrics
derived from them are ever stored (ADR-0005).

Pipeline, per camera:  ingest -> sampler -> detector -> tracker -> analytics -> events
Supervisor, per site:  events -> aggregator -> store -> {exporters, api, sync}

See ../Mesopic-docs/docs/01-architecture/engine-architecture.md.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]
