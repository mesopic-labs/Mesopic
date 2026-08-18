"""Occupancy and queue length — the two sampled states.

Both fold the same input (`occupancy_sample`, one per zone per tick) into the same two
series, and differ only in which zones they read and what they call the result. Queue
length *is* occupancy of a `role: queue` zone (algorithms.md §8); giving it its own
plugin keeps the two from claiming the same metric name and stops a till queue being
added to the shop floor it sits inside.

Two rules carry this module, both from the plugin review checklist
(engine-architecture.md §17), and both fail quietly rather than loudly:

* **The mean is `Δt`-weighted, never sample-averaged** (algorithms.md §0.6). The adaptive
  sampler lowers fps under load and load correlates with a busy scene, so a sample-mean
  under-weights precisely the rush the customer is paying to see.
* **Peak reads the raw count, mean reads the confirmed one** (§6.1). Confirmation lags by
  `dwell_min_s`, which is harmless in an average and wrong in a peak — peaks form in the
  fast-turnover moments the confirmation rule suppresses.

`max` is unaffected by uneven spacing but *is* affected by missing samples, so
`sample_count` rides along on both rows: a thin bucket should read as thin rather than as
confident (§6.2).

Implements P2.4.
"""

from __future__ import annotations

from muster.analytics.site_geometry import PreparedZone, SiteGeometry
from muster.types import (
    CameraId,
    EventKind,
    MetricName,
    MetricRow,
    MinuteBucket,
    RawEvent,
    ScopeId,
    ZoneRole,
)


class _SampledZoneMetric:
    """The shared fold. Subclasses choose the zones and name the two series."""

    role: ZoneRole
    mean_metric: MetricName
    peak_metric: MetricName

    def __init__(self, geometry: SiteGeometry) -> None:
        self._geometry = geometry

    @property
    def names(self) -> frozenset[MetricName]:
        """Both series, from one plugin.

        Splitting them would mean two folds over one sample stream, and nothing would
        keep the pair consistent once one of them changed.
        """
        return frozenset({self.mean_metric, self.peak_metric})

    def reduce(self, events: list[RawEvent], bucket: MinuteBucket) -> list[MetricRow]:
        rows: list[MetricRow] = []
        for camera_id in sorted({event.camera_id for event in events}):
            for zone in self._zones_for(camera_id):
                rows += self._rows_for(camera_id, zone, events, bucket)
        return rows

    def _zones_for(self, camera_id: CameraId) -> list[PreparedZone]:
        return [
            zone
            for zone in self._geometry.zones_for(camera_id)
            if zone.role is self.role and self.mean_metric in zone.metrics
        ]

    def _rows_for(
        self,
        camera_id: CameraId,
        zone: PreparedZone,
        events: list[RawEvent],
        bucket: MinuteBucket,
    ) -> list[MetricRow]:
        samples = _samples_of(zone, events)
        elapsed = sum(sample.dt_s or 0.0 for sample in samples)
        if not samples or elapsed <= 0.0:
            # No observation is not an observation of zero, and a bucket of zero-width
            # samples has nothing to average over.
            return []
        weighted = sum((sample.confirmed_value or 0.0) * (sample.dt_s or 0.0) for sample in samples)
        peak = max(sample.value or 0.0 for sample in samples)
        # Each staff sub-count on the same basis as the series it belongs to: the mean is
        # `Δt`-weighted over the confirmed count, so its staff half must be too, and the
        # peak reads the raw count. Crossing them would report a staff number derived from
        # a different population than the total beside it (staff.py, ADR-0021).
        staff_weighted = sum(
            (sample.staff_confirmed_value or 0.0) * (sample.dt_s or 0.0) for sample in samples
        )
        staff_peak = max(sample.staff_value or 0.0 for sample in samples)
        return [
            MetricRow(
                camera_id=camera_id,
                bucket=bucket,
                metric=metric,
                scope_id=ScopeId(zone.zone_id),
                value=value,
                staff_value=staff,
                sample_count=len(samples),
            )
            for metric, value, staff in (
                (self.mean_metric, weighted / elapsed, staff_weighted / elapsed),
                (self.peak_metric, peak, staff_peak),
            )
        ]


class OccupancyPlugin(_SampledZoneMetric):
    """How many people are in an area: the `Δt`-weighted mean, and the peak."""

    role = ZoneRole.AREA
    mean_metric = MetricName.OCCUPANCY
    peak_metric = MetricName.OCCUPANCY_RAW


class QueueLengthPlugin(_SampledZoneMetric):
    """The same fold on a `role: queue` zone (algorithms.md §8).

    The `min_dwell_to_count` filter the confirmed series already applies is what keeps
    someone browsing past the till out of the queue — §8's default "plain 0/1 headcount
    past `dwell_min_s`". The soft dwell-weighted ramp is opt-in and not built.
    """

    role = ZoneRole.QUEUE
    mean_metric = MetricName.QUEUE_LEN
    peak_metric = MetricName.QUEUE_LEN_RAW


def _samples_of(zone: PreparedZone, events: list[RawEvent]) -> list[RawEvent]:
    return [
        event
        for event in events
        if event.kind is EventKind.OCCUPANCY_SAMPLE and event.zone_id == zone.zone_id
    ]
