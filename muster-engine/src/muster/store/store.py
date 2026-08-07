"""The narrow store API over one SQLite file.

WAL mode: many readers (dashboard, exporters, sync client) concurrent with the one
writer (aggregator, in the supervisor process). `STRICT` tables catch type bugs at the
boundary; `WITHOUT ROWID` on the composite-key time series clusters storage by the key we
always range-scan.

There is no method here that reads or writes an image, a crop, or a bbox pixel — the
privacy guarantee is made structural rather than promised (ADR-0005).

Implements P2.5.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from muster.types import MetricRow, MinuteBucket, RawEvent


class Store:
    """Owns the SQLite file. One instance, in the supervisor process."""

    def __init__(self, path: Path) -> None:
        raise NotImplementedError

    def migrate(self) -> None:
        """Apply forward-only migrations and stamp `schema_meta.schema_version`."""
        raise NotImplementedError

    def append_events(self, events: Sequence[RawEvent]) -> None:
        """Write to the short-retention raw event log. Never synced."""
        raise NotImplementedError

    def upsert_metrics(self, rows: Sequence[MetricRow]) -> None:
        """Idempotent upsert on `(camera_id, metric, scope_id, bucket)`."""
        raise NotImplementedError

    def unsynced_metrics(self, limit: int) -> list[MetricRow]:
        """Rows awaiting cloud sync, in bucket order. Drives the sync cursor."""
        raise NotImplementedError

    def mark_synced(self, rows: Sequence[MetricRow], synced_at: MinuteBucket) -> None:
        """Stamp `synced_at` after a confirmed 2xx from the cloud."""
        raise NotImplementedError

    def trim(self) -> None:
        """Enforce event retention and the disk-bounded metric cap."""
        raise NotImplementedError
