"""Wiring a Frigate camera, and the loop that runs it.

The adapter's own logic is `test_frigate_objects.py`. This is the seams around it: that
config refuses a Frigate camera with nowhere to connect, that the worker picks the loop
matching what the source can hand over, and that the calibration editor says why it
cannot help rather than showing an empty canvas.

Red-first for P4.3.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from muster.analytics.geometry import GeometryAnalytics
from muster.analytics.site_geometry import SiteGeometry
from muster.config.schema import MusterConfig
from muster.errors import ConfigError
from muster.ingest.frigate import FrigateObjects
from muster.sampler.sampler import FrameSampler
from muster.supervisor.pipeline import CameraPipeline, TrackPipeline, build_pipeline
from muster.supervisor.worker import track_loop
from muster.types import CameraId, EventKind, FrameTs, RawEvent, Track, TrackId

TILL = CameraId("till")
DOOR = CameraId("front-door")
FLOOR = "shop-floor"

T0 = datetime(2026, 8, 18, 9, 30, tzinfo=UTC)


def _config(**frigate: Any) -> MusterConfig:
    document: dict[str, Any] = {
        "site": {"site_id": "test-site"},
        "cameras": [
            {
                "camera_id": TILL,
                "name": "Till",
                "source": {"kind": "frigate", "mqtt_topic": "frigate/events"},
                "reference_resolution": [1280, 720],
            },
            {
                "camera_id": DOOR,
                "name": "Front door",
                "source": {"kind": "rtsp", "url_env": "MUSTER_TEST_RTSP"},
                "reference_resolution": [1920, 1080],
            },
        ],
        "zones": [
            {
                "zone_id": FLOOR,
                "camera_id": TILL,
                "role": "area",
                "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                "metrics": ["occupancy"],
            }
        ],
    }
    if frigate:
        document["frigate"] = frigate
    return MusterConfig.model_validate(document)


# --- Config -----------------------------------------------------------------


def test_a_frigate_camera_with_no_broker_is_refused_at_load() -> None:
    """It could never produce a single event, and a worker restart loop is a worse place
    to learn that than the error the operator is already looking at."""
    with pytest.raises(ValueError, match=r"frigate\.broker"):
        _config()


def test_a_frigate_camera_builds_a_track_pipeline() -> None:
    pipeline = build_pipeline(_config(broker="mosquitto"), TILL)
    assert isinstance(pipeline, TrackPipeline)


def test_an_rtsp_camera_still_builds_a_frame_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dispatch must not have quietly captured every camera."""
    monkeypatch.setenv("MUSTER_TEST_RTSP", "rtsp://127.0.0.1:8554/x")
    assert isinstance(build_pipeline(_config(broker="mosquitto"), DOOR), CameraPipeline)


def test_broker_credentials_name_the_variable_never_the_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MUSTER_FRIGATE_PASSWORD", raising=False)
    # The value here is an environment variable's NAME, not a password — which is the
    # whole point of the `*_env` convention, and why the linter's guess is wrong.
    config = _config(
        broker="mosquitto",
        username_env="MUSTER_FRIGATE_USER",
        password_env="MUSTER_FRIGATE_PASSWORD",  # noqa: S106 - an env var name, not a secret
    )
    monkeypatch.setenv("MUSTER_FRIGATE_USER", "muster")

    with pytest.raises(ConfigError) as raised:
        build_pipeline(config, TILL)

    assert "MUSTER_FRIGATE_PASSWORD" in str(raised.value)


# --- The loop ---------------------------------------------------------------


class _ScriptedSource:
    """A `TrackSource` that yields a fixed script, then stops. No broker, no thread."""

    def __init__(self, script: list[tuple[FrameTs, list[Track]]]) -> None:
        self._script = script
        self.closed = False

    def ticks(self) -> Iterator[tuple[FrameTs, list[Track]]]:
        yield from self._script

    def close(self) -> None:
        self.closed = True


class _Sink:
    def __init__(self) -> None:
        self.events: list[RawEvent] = []

    def put_nowait(self, item: RawEvent) -> None:
        self.events.append(item)


def _track(foot: tuple[float, float], *, ts: datetime, track_id: int = 1) -> Track:
    return Track(
        camera_id=TILL,
        track_id=TrackId(track_id),
        ts=FrameTs(ts),
        foot_point=foot,
        score=0.9,
    )


def _loop(script: list[tuple[FrameTs, list[Track]]]) -> tuple[_Sink, _ScriptedSource]:
    source = _ScriptedSource(script)
    sink = _Sink()
    track_loop(
        TILL,
        source=source,
        sampler=FrameSampler(target_fps=1000.0),
        analytics=GeometryAnalytics(SiteGeometry.compile(_config(broker="m"))),
        sink=sink,
        fps_min=1.0,
        fps_max=1000.0,
    )
    return sink, source


def test_the_loop_turns_ticks_into_events() -> None:
    inside = (0.5, 0.5)
    sink, source = _loop(
        [
            (FrameTs(T0), [_track(inside, ts=T0)]),
            (FrameTs(T0 + timedelta(seconds=1)), [_track(inside, ts=T0 + timedelta(seconds=1))]),
        ]
    )

    kinds = {event.kind for event in sink.events}
    assert EventKind.ZONE_ENTER in kinds
    assert source.closed, "the source must be released however the loop ends"


def test_an_empty_tick_still_reports_the_zone() -> None:
    """The reason the idle tick exists: a quiet camera has to be measured as quiet, or a
    shut shop and a dead broker render identically."""
    sink, _ = _loop([(FrameTs(T0), []), (FrameTs(T0 + timedelta(seconds=1)), [])])

    samples = [e for e in sink.events if e.kind is EventKind.OCCUPANCY_SAMPLE]
    assert samples
    assert samples[0].value == 0.0


def test_the_loop_runs_no_detector_at_all() -> None:
    """There is nothing to detect. If this ever needs one, the source is lying about
    what it hands over."""
    assert "detector" not in inspect.signature(track_loop).parameters
    assert "tracker" not in inspect.signature(track_loop).parameters


# --- Calibration ------------------------------------------------------------


def test_the_adapter_and_the_loop_agree_on_the_tick_shape() -> None:
    """`FrigateObjects` produces what `track_loop` consumes — the two are wired by the
    pipeline and never otherwise exercised together."""
    objects = FrigateObjects(TILL, frigate_camera="till", width=1280, height=720)
    ts = objects.tick_ts(now=0.0)
    tracks = objects.tick(now=0.0)

    sink, _ = _loop([(ts, tracks), (FrameTs(ts + timedelta(seconds=1)), tracks)])

    assert [e for e in sink.events if e.kind is EventKind.OCCUPANCY_SAMPLE]
