"""Footfall — distinct people arriving, per doorway or per area.

Two sources, unified (algorithms.md §7). A camera with a counting line uses it: each
inbound crossing is one arrival. A camera without one falls back to its zones, where an
arrival is a *confirmed* entry — someone clipping a corner has not visited.

**A camera with a line ignores its zones entirely.** Both would otherwise count the same
person walking through the same doorway, and the sum would read as two visitors.

Rows are scoped to the geometry that produced them and never to the camera as a whole.
Two doors are two rows that sum to the visit count; a camera-wide row would be that same
quantity measured a second way, so a consumer adding both would double it.

Implements P2.4.
"""

from __future__ import annotations

from muster.analytics.site_geometry import PreparedLine, PreparedZone, SiteGeometry
from muster.types import (
    CameraId,
    Direction,
    EventKind,
    MetricName,
    MetricRow,
    MinuteBucket,
    RawEvent,
    ScopeId,
    TrackId,
)


class FootfallPlugin:
    """Reduces arrivals into one row per counting line, or per zone where none exists."""

    def __init__(self, geometry: SiteGeometry) -> None:
        self._geometry = geometry

    @property
    def names(self) -> frozenset[MetricName]:
        return frozenset({MetricName.FOOTFALL})

    def reduce(self, events: list[RawEvent], bucket: MinuteBucket) -> list[MetricRow]:
        return [
            MetricRow(
                camera_id=camera_id,
                bucket=bucket,
                metric=MetricName.FOOTFALL,
                scope_id=scope_id,
                value=float(len(entrants)),
                sample_count=len(entrants),
            )
            for camera_id in sorted(self.cameras_in(events))
            for scope_id, entrants in self._arrivals(camera_id, events)
        ]

    @staticmethod
    def cameras_in(events: list[RawEvent]) -> set[CameraId]:
        """Which cameras this bucket has any evidence of.

        A camera that sent nothing gets no row at all, not a zero: an outage reported as
        a confident zero every minute is how it becomes a quiet trading day in a chart.
        """
        return {event.camera_id for event in events}

    def _arrivals(
        self, camera_id: CameraId, events: list[RawEvent]
    ) -> list[tuple[ScopeId, set[TrackId]]]:
        lines = self._lines_for(camera_id)
        if lines:
            return [(ScopeId(line.line_id), _entrants_through(line, events)) for line in lines]
        return [
            (ScopeId(zone.zone_id), _entrants_into(zone, events))
            for zone in self._zones_for(camera_id)
        ]

    def _lines_for(self, camera_id: CameraId) -> list[PreparedLine]:
        return [
            line
            for line in self._geometry.lines_for(camera_id)
            if MetricName.FOOTFALL in line.metrics
        ]

    def _zones_for(self, camera_id: CameraId) -> list[PreparedZone]:
        return [
            zone
            for zone in self._geometry.zones_for(camera_id)
            if MetricName.FOOTFALL in zone.metrics
        ]


def _entrants_through(line: PreparedLine, events: list[RawEvent]) -> set[TrackId]:
    """Distinct tracks crossing inwards. Distinct, because one person pacing a doorway
    would otherwise be a busy morning."""
    return {
        event.track_id
        for event in events
        if event.kind is EventKind.LINE_CROSS
        and event.line_id == line.line_id
        and event.track_id is not None
        and _is_arrival(event.direction, line.positive_dir)
    }


def _entrants_into(zone: PreparedZone, events: list[RawEvent]) -> set[TrackId]:
    return {
        event.track_id
        for event in events
        if event.kind is EventKind.ZONE_CONFIRMED
        and event.zone_id == zone.zone_id
        and event.track_id is not None
    }


def _is_arrival(direction: int | None, positive_dir: Direction) -> bool:
    """`positive_dir` labels which sense of the crossing is "in"; it does not set it."""
    if direction is None:
        return False
    inbound = 1 if positive_dir is Direction.IN else -1
    return direction == inbound
