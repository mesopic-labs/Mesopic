"""Conversion — the one core-six metric with no camera input.

It is *derived, not sensed* (engine-architecture.md §8, §10): a join of footfall, which
this engine measures, with a transaction count that arrives from a till. P2.4 takes the
count from an injected mapping; P3.5 replaces that with the signed webhook ingress.

**The unifying rule is that absence of a signal is absence of a row, never a zero**
(algorithms.md §9). A missing till reading rendered as `0.0` reads as "nobody bought
anything", which is a damaging lie in a way that a gap in the chart is not. Likewise
`0` footfall gives an undefined ratio rather than an infinite one.

A ratio above 1.0 is passed through rather than clamped: group baskets and staff
purchases make it genuinely possible, and where it is *not*, it is revealing a footfall
undercount that clamping would hide.

**A minute-grain ratio does not roll up by averaging.** Hourly conversion is
`Σ txns / Σ footfall` over the hour, not the mean of sixty ratios, and the transaction
counts are not stored as a series today — so a correct hourly figure needs the numerator
kept. That is P3.5's to fix when the ingress it belongs to exists.

Implements P2.4.
"""

from __future__ import annotations

from collections.abc import Mapping

from mesopic.analytics.metrics.footfall import FootfallPlugin
from mesopic.types import (
    CameraId,
    MetricName,
    MetricRow,
    MinuteBucket,
    RawEvent,
)

TransactionCounts = Mapping[tuple[CameraId, MinuteBucket], int]
"""Transactions per camera per minute, from the till. P3.5 makes it a live feed."""


class ConversionPlugin:
    """Joins one bucket's footfall with the transactions recorded against it."""

    def __init__(self, footfall: FootfallPlugin, transactions: TransactionCounts) -> None:
        self._footfall = footfall
        self._transactions = transactions

    @property
    def names(self) -> frozenset[MetricName]:
        return frozenset({MetricName.CONVERSION})

    def reduce(self, events: list[RawEvent], bucket: MinuteBucket) -> list[MetricRow]:
        """Camera-wide: a till belongs to no single door, so neither does the ratio.

        Footfall is recomputed here rather than read back from the store because a
        reducer folds one bucket's events and nothing else — a store read would make the
        result depend on what had already been written, and re-folding after a crash
        would stop being deterministic.
        """
        rows = []
        for camera_id in sorted(self._footfall.cameras_in(events)):
            transactions = self._transactions.get((camera_id, bucket))
            visitors = sum(
                row.value
                for row in self._footfall.reduce(events, bucket)
                if row.camera_id == camera_id
            )
            if transactions is None or visitors == 0:
                continue
            rows.append(
                MetricRow(
                    camera_id=camera_id,
                    bucket=bucket,
                    metric=MetricName.CONVERSION,
                    scope_id=None,
                    value=transactions / visitors,
                    sample_count=int(visitors),
                )
            )
        return rows
