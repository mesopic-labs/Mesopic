"""Mesopic — video-intelligence for the cameras you already own.

The engine runs on the customer's box. Footage never leaves the building: frames are
discarded the instant they are processed, and only anonymous foot-points and the metrics
derived from them are ever stored (ADR-0005).

Pipeline, per camera:  ingest -> sampler -> detector -> tracker -> analytics -> events
Supervisor, per site:  events -> aggregator -> store -> {exporters, api, sync}

See ../Mesopic-docs/docs/01-architecture/engine-architecture.md.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# Read from the installed distribution rather than restated here, so a release bump in
# pyproject.toml cannot leave the CLI reporting a stale number. It already had: this
# module said 0.0.1 while the project shipped 0.1.0, so `mesopic version` was wrong for
# the whole of the first release.
try:
    __version__ = version("mesopic")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
