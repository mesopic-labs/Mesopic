"""Store-and-forward of metrics JSON to the paid cloud (engine-architecture.md §14).

Optional and **off by default** — the free, local experience never touches it (ADR-0001).

This package reads metrics tables and nothing else. It has no access to frames, and an
import-linter contract forbids it from importing anything that does. "Video never leaves
the building" is a structural property here, not a promise (ADR-0005).

The guarantee is made twice over, at different layers, because one of them is easy to
weaken without noticing: the contract stops this package importing a decoder, and
`MetricsSource` — the protocol the loop is typed against — stops it *asking the store* for
anything but metrics.
"""

from __future__ import annotations

from mesopic.sync.build import build_sync_loop
from mesopic.sync.client import MetricsSyncClient, ThrottledError
from mesopic.sync.loop import MetricsSource, SyncLoop, drain_once
from mesopic.sync.payload import SYNC_FORMAT, batch_for, idempotency_key

__all__ = [
    "SYNC_FORMAT",
    "MetricsSource",
    "MetricsSyncClient",
    "SyncLoop",
    "ThrottledError",
    "batch_for",
    "build_sync_loop",
    "drain_once",
    "idempotency_key",
]
