"""Line crossings — the signed net across each counting line.

Net rather than a count, because the direction is the information: three in and one out
is a net of two, which is what makes a line usable as an occupancy integrator
(algorithms.md §5, engine-architecture.md §10).

`sample_count` carries how many crossings produced that net, so a net of zero from forty
crossings reads as a busy doorway rather than a deserted one.

Implements P2.4.
"""

from __future__ import annotations

from collections.abc import Sequence

from mesopic.analytics.metrics.staff import staff_only
from mesopic.analytics.site_geometry import SiteGeometry
from mesopic.types import (
    EventKind,
    LineId,
    MetricName,
    MetricRow,
    MinuteBucket,
    RawEvent,
    ScopeId,
)


class LineCrossPlugin:
    """Reduces directional crossings into one row per line."""

    def __init__(self, geometry: SiteGeometry) -> None:
        self._geometry = geometry

    @property
    def names(self) -> frozenset[MetricName]:
        return frozenset({MetricName.LINE_CROSS})

    def reduce(self, events: list[RawEvent], bucket: MinuteBucket) -> list[MetricRow]:
        rows = []
        for camera_id in sorted({event.camera_id for event in events}):
            measurable = self._geometry.has_staff_zone(camera_id)
            for line in self._geometry.lines_for(camera_id):
                if MetricName.LINE_CROSS not in line.metrics:
                    continue
                crossings = _directions(events, line.line_id)
                rows.append(
                    MetricRow(
                        camera_id=camera_id,
                        bucket=bucket,
                        metric=MetricName.LINE_CROSS,
                        scope_id=ScopeId(line.line_id),
                        value=float(sum(crossings)),
                        # A count, so no staff is zero — but only where staff could have
                        # been seen at all. On a camera with no staff zone the split is
                        # absent rather than zero (staff.py, ADR-0021).
                        staff_value=(
                            float(sum(_directions(staff_only(events), line.line_id)))
                            if measurable
                            else None
                        ),
                        sample_count=len(crossings),
                    )
                )
        return rows


def _directions(events: Sequence[RawEvent], line_id: LineId) -> list[int]:
    return [
        event.direction
        for event in events
        if event.kind is EventKind.LINE_CROSS
        and event.line_id == line_id
        and event.direction is not None
    ]
