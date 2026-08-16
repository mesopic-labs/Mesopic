"""The shared vocabulary's guarantees.

`muster.types` is imported by every module in the engine, so the properties it promises
— immutable payloads, a locked metric vocabulary — are worth asserting once, here.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import Any

import pytest

from muster.types import (
    CameraId,
    Detection,
    FrameTs,
    MetricName,
    MetricRow,
    MinuteBucket,
    RawEvent,
    ScopeId,
    Track,
    TrackId,
)

PIPELINE_PAYLOADS: tuple[Any, ...] = (Detection, Track, RawEvent, MetricRow)


@pytest.mark.parametrize("payload", PIPELINE_PAYLOADS)
def test_pipeline_payloads_are_frozen_and_slotted(payload: Any) -> None:
    """Nothing downstream may mutate what it was handed, and these are allocated per frame."""
    assert dataclasses.fields(payload), f"{payload.__name__} declares no fields"
    assert payload.__dataclass_params__.frozen, f"{payload.__name__} must be frozen"
    assert hasattr(payload, "__slots__"), f"{payload.__name__} must use slots"


def test_metric_vocabulary_is_locked() -> None:
    """The core six plus the two adjacencies. Scope is locked for v1.

    A tripwire, not a formality: adding a metric changes the edge/cloud sync contract
    (cloud-architecture.md §4), so it should require deliberately editing this list.
    """
    assert {metric.value for metric in MetricName} == {
        "footfall",
        "occupancy",
        "occupancy_raw",
        "queue_len",
        "queue_len_raw",
        "dwell_seconds",
        "line_cross",
        "conversion",
        "heatmap",
    }


def test_a_metric_row_carries_its_full_natural_key() -> None:
    """Idempotent upsert depends on `(camera_id, metric, scope_id, bucket)` (§10, §11)."""
    row = MetricRow(
        camera_id=CameraId("front-door"),
        bucket=MinuteBucket(datetime(2026, 8, 7, 9, 0, tzinfo=UTC)),
        metric=MetricName.FOOTFALL,
        scope_id=ScopeId("door-count"),
        value=12.0,
    )

    assert row.camera_id
    assert row.metric
    assert row.scope_id
    assert row.bucket.tzinfo is UTC, "storage is UTC everywhere; timezone is display-only"
    assert row.bucket.second == 0, "a MinuteBucket is floored to the minute"


def test_a_raw_event_carries_no_pixels() -> None:
    """What crosses the worker/supervisor boundary is small and pixel-free (ADR-0005).

    An exact set, not a subset: growing `RawEvent` should cost a line in this test and a
    moment's thought about whether the new field could carry image data. `value` was
    added for P2.6's completed dwell durations and earned that moment; `dt_s` was added
    for P2.4's occupancy samples and is a duration in seconds, which cannot encode a
    frame however it is abused.
    """
    assert {field.name for field in dataclasses.fields(RawEvent)} == {
        "camera_id",
        "ts",
        "kind",
        "track_id",
        "zone_id",
        "line_id",
        "direction",
        "value",
        "confirmed_value",
        "dt_s",
        "is_staff",
    }


def test_a_track_reports_a_normalized_foot_point() -> None:
    """Geometry is normalized so zones authored once survive a resolution change (§3)."""
    track = Track(
        camera_id=CameraId("front-door"),
        track_id=TrackId(1),
        ts=FrameTs(datetime(2026, 8, 7, 9, 0, tzinfo=UTC)),
        foot_point=(0.5, 0.9),
        score=0.8,
    )

    x, y = track.foot_point
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0
    assert not track.is_staff, "staff is a per-track boolean, defaulted off, never an identity"


def test_track_defaults_to_observed() -> None:
    """A Track with no explicit status is an observed one, not a coasted one."""
    track = Track(
        camera_id=CameraId("cam-1"),
        track_id=TrackId(1),
        ts=FrameTs(datetime(2026, 8, 10, 12, 0, tzinfo=UTC)),
        foot_point=(0.5, 0.9),
        score=0.9,
    )
    assert track.time_since_update == 0


def test_track_records_dead_reckoning_depth() -> None:
    """Geometry needs the count, not just the fact, to refuse fabricated events."""
    track = Track(
        camera_id=CameraId("cam-1"),
        track_id=TrackId(1),
        ts=FrameTs(datetime(2026, 8, 10, 12, 0, tzinfo=UTC)),
        foot_point=(0.5, 0.9),
        score=0.9,
        time_since_update=3,
    )
    assert track.time_since_update == 3
