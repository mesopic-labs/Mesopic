"""Fold raw events into minute buckets.

Two properties this module exists to guarantee:

* **Bucketing is by capture time, UTC** — never wall time. A frame processed late still
  lands in the minute it was captured in, which is what makes the metric honest and what
  makes replay deterministic.
* **Writes are idempotent** on the natural key `(camera_id, metric, scope_id, bucket)`.
  A mid-minute crash and restart must not double-count.

Open dwells and live occupancy are carried in memory and resolve into the bucket they
end in.

Implements P2.6.
"""

from __future__ import annotations

from muster.types import MetricRow, MinuteBucket, RawEvent


class Aggregator:
    """Reduces the event stream into durable metric rows."""

    def ingest(self, event: RawEvent) -> None:
        """Accept one event. Cheap: the fold happens on bucket close."""
        raise NotImplementedError

    def close_bucket(self, bucket: MinuteBucket) -> list[MetricRow]:
        """Finalise a minute and return its rows for the store to upsert."""
        raise NotImplementedError
