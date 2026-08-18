"""Dwell time — how long people stay, per zone.

A dwell is a duration bracketed by two timestamps, so it is already in wall-clock terms
and `Δt`-weighting does not apply to it (algorithms.md §0.6). The aggregator's state
machine has already closed each one and attributed it to the minute it *ended* in; this
plugin only reduces the durations it is handed.

**It reports the mean, and algorithms.md §7 says the mean is the misleading summary.**
Dwell durations are heavy-tailed — most people are quick, a few linger a long time — so
the median and p90 are the honest answer, and p90 is the number a manager actually wants.
Both need metric names the vocabulary does not have, so they are deliberately not
invented here. `sample_count` carries how many stays produced the mean, which at least
makes a mean-of-one visible as one.

Implements P2.4.
"""

from __future__ import annotations

from collections.abc import Sequence

from muster.analytics.metrics.staff import staff_only
from muster.analytics.site_geometry import SiteGeometry
from muster.types import (
    EventKind,
    MetricName,
    MetricRow,
    MinuteBucket,
    RawEvent,
    ScopeId,
    ZoneId,
)


class DwellPlugin:
    """Reduces completed dwells into one row per zone."""

    def __init__(self, geometry: SiteGeometry) -> None:
        self._geometry = geometry

    @property
    def names(self) -> frozenset[MetricName]:
        return frozenset({MetricName.DWELL_SECONDS})

    def reduce(self, events: list[RawEvent], bucket: MinuteBucket) -> list[MetricRow]:
        rows = []
        for camera_id in sorted({event.camera_id for event in events}):
            for zone in self._geometry.zones_for(camera_id):
                if MetricName.DWELL_SECONDS not in zone.metrics:
                    continue
                durations = _durations(events, zone.zone_id)
                staff = _durations(staff_only(events), zone.zone_id)
                if not durations:
                    # Most minutes close no dwell at all — a stay is attributed to the
                    # minute it ended in, not to every minute it spanned.
                    continue
                rows.append(
                    MetricRow(
                        camera_id=camera_id,
                        bucket=bucket,
                        metric=MetricName.DWELL_SECONDS,
                        scope_id=ScopeId(zone.zone_id),
                        value=sum(durations) / len(durations),
                        # A MEAN, not a count: with no staff dwell there is nothing to
                        # average, and `0.0` would claim staff stayed for no time at all
                        # rather than that none of them stayed (staff.py).
                        staff_value=sum(staff) / len(staff) if staff else None,
                        sample_count=len(durations),
                    )
                )
        return rows


def _durations(events: Sequence[RawEvent], zone_id: ZoneId) -> list[float]:
    return [
        event.value
        for event in events
        if event.kind is EventKind.DWELL_SAMPLE
        and event.zone_id == zone_id
        and event.value is not None
    ]
