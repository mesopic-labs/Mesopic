"""Process supervision and backpressure (engine-architecture.md §9, §15).

One OS process per camera, one supervisor. Per-process isolation means one camera's
decode load or one camera's crash cannot stall or take down another.

The supervisor does no CV work itself: it spawns and restarts workers, owns the single
SQLite writer, and runs the aggregator, exporters, API, and sync client.
"""

from __future__ import annotations

from muster.supervisor.supervisor import Supervisor

__all__ = ["Supervisor"]
