"""The per-camera loop and the config→pipeline wiring it runs.

Two things are asserted here that nothing else can assert:

* the loop emits events and sheds fps under pressure, rather than blocking on a full
  queue or dropping frames the detector already paid for;
* `build_pipeline` fails on a source it cannot serve **without ever putting the RTSP URL
  in the error**, because an RTSP URL carries the camera's credentials.

Red-first for P2.7.
"""

from __future__ import annotations

import queue
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from muster.config.schema import MusterConfig
from muster.errors import ConfigError, StreamDropped
from muster.supervisor.pipeline import build_pipeline
from muster.supervisor.worker import camera_loop
from muster.types import (
    CameraId,
    DecodedFrame,
    Detection,
    EventKind,
    FrameTs,
    NormPoint,
    RawEvent,
    Track,
    TrackId,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"

FRONT_DOOR = CameraId("front-door")
EXAMPLE_RTSP_URL = "rtsp://user:pass@192.168.1.40:554/Streaming/Channels/102"
T0 = datetime(2026, 8, 16, 9, 30, tzinfo=UTC)


@pytest.fixture
def raw_config() -> dict[str, Any]:
    parsed = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def _frame(offset_s: float) -> DecodedFrame:
    return DecodedFrame(
        camera_id=FRONT_DOOR,
        ts=FrameTs(T0 + timedelta(seconds=offset_s)),
        image=np.full((4, 4, 3), 7, dtype=np.uint8),
        width=4,
        height=4,
    )


class _Source:
    def __init__(self, count: int) -> None:
        self._count = count
        self.closed = False

    def frames(self) -> Iterator[DecodedFrame]:
        for index in range(self._count):
            yield _frame(index)
        message = "camera 'front-door': stream ended"
        raise StreamDropped(message)

    def close(self) -> None:
        self.closed = True


class _Detector:
    def detect(self, _frame: DecodedFrame) -> list[Detection]:
        return [Detection(box=(0, 0, 2, 4), score=0.9)]

    def close(self) -> None:
        return None


class _Tracker:
    def update(self, frame: DecodedFrame, _detections: list[Detection]) -> list[Track]:
        point: NormPoint = (0.25, 0.5)
        return [
            Track(
                camera_id=FRONT_DOOR,
                track_id=TrackId(1),
                ts=frame.ts,
                foot_point=point,
                score=0.9,
            )
        ]


class _Analytics:
    """Emits one event per tick, so the loop's plumbing is what is under test."""

    def __init__(self) -> None:
        self.calls = 0

    def on_tracks(self, camera_id: CameraId, tracks: list[Track]) -> list[RawEvent]:
        self.calls += 1
        return [
            RawEvent(
                camera_id=camera_id,
                ts=tracks[0].ts,
                kind=EventKind.LINE_CROSS,
                track_id=TrackId(self.calls),
                direction=1,
            )
        ]


class _Sampler:
    """Admits everything, and records the rates backpressure asks for."""

    def __init__(self) -> None:
        self.rates: list[float] = []

    def is_due(self, _ts: FrameTs) -> bool:
        return True

    def set_target_fps(self, target_fps: float) -> None:
        self.rates.append(target_fps)


class _Sink:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: list[object] = []

    def put_nowait(self, item: object) -> None:
        if len(self.items) >= self.capacity:
            raise queue.Full
        self.items.append(item)


def _parts(sampler: _Sampler, source: _Source, analytics: _Analytics) -> dict[str, Any]:
    return {
        "source": source,
        "sampler": sampler,
        "detector": _Detector(),
        "tracker": _Tracker(),
        "analytics": analytics,
    }


# --- The loop ---------------------------------------------------------------


def test_the_loop_emits_one_event_per_admitted_frame() -> None:
    sink = _Sink(capacity=100)
    analytics = _Analytics()

    stats = camera_loop(
        FRONT_DOOR, sink=sink, fps_min=1.0, fps_max=5.0, **_parts(_Sampler(), _Source(3), analytics)
    )

    assert len(sink.items) == 3
    assert stats.frames_admitted == 3
    assert stats.dropped_events == 0


def test_a_stream_drop_ends_the_loop_rather_than_escaping() -> None:
    """A dropped stream is how every real run ends; the process must exit, not crash."""
    sink = _Sink(capacity=100)
    source = _Source(2)

    stats = camera_loop(
        FRONT_DOOR, sink=sink, fps_min=1.0, fps_max=5.0, **_parts(_Sampler(), source, _Analytics())
    )

    assert stats.stream_dropped
    assert source.closed, "the camera socket must be released on the way out"


def test_a_full_sink_sheds_fps_instead_of_blocking() -> None:
    """The response to a slow supervisor is to sample less, not to stall the decoder."""
    sampler = _Sampler()
    sink = _Sink(capacity=1)

    camera_loop(
        FRONT_DOOR, sink=sink, fps_min=1.0, fps_max=5.0, **_parts(sampler, _Source(8), _Analytics())
    )

    assert sampler.rates, "a full sink must reach the sampler"
    assert sampler.rates == sorted(sampler.rates, reverse=True)
    assert min(sampler.rates) >= 1.0


def test_events_beyond_the_outbox_are_counted_as_dropped() -> None:
    """Dropping counts is a degraded metric — visible, never silent (§9)."""
    sink = _Sink(capacity=0)

    stats = camera_loop(
        FRONT_DOOR,
        sink=sink,
        fps_min=1.0,
        fps_max=5.0,
        outbox_size=2,
        **_parts(_Sampler(), _Source(6), _Analytics()),
    )

    assert stats.dropped_events == 4


# --- Building the pipeline from config --------------------------------------


def test_an_rtsp_camera_gets_a_source_built_from_its_url(raw_config: dict[str, Any]) -> None:
    config = MusterConfig.model_validate(raw_config)

    pipeline = build_pipeline(config, FRONT_DOOR)

    assert pipeline.source is not None
    assert pipeline.camera_id == FRONT_DOOR


def test_a_url_env_camera_reads_its_url_from_the_environment(
    raw_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The form a real deployment uses: the URL never appears in the config file."""
    monkeypatch.setenv("MUSTER_FRONT_DOOR_RTSP", EXAMPLE_RTSP_URL)
    camera = next(c for c in raw_config["cameras"] if c["camera_id"] == "front-door")
    del camera["source"]["url"]
    camera["source"]["url_env"] = "MUSTER_FRONT_DOOR_RTSP"
    config = MusterConfig.model_validate(raw_config)

    pipeline = build_pipeline(config, FRONT_DOOR)

    assert pipeline.source is not None


def test_a_missing_url_env_var_is_refused_without_naming_its_value(
    raw_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MUSTER_FRONT_DOOR_RTSP", raising=False)
    camera = next(c for c in raw_config["cameras"] if c["camera_id"] == "front-door")
    del camera["source"]["url"]
    camera["source"]["url_env"] = "MUSTER_FRONT_DOOR_RTSP"
    config = MusterConfig.model_validate(raw_config)

    with pytest.raises(ConfigError, match="MUSTER_FRONT_DOOR_RTSP"):
        build_pipeline(config, FRONT_DOOR)


@pytest.mark.privacy
def test_an_unsupported_source_is_refused_without_leaking_the_url(
    raw_config: dict[str, Any],
) -> None:
    """A Frigate source is P4.3's, and an RTSP URL carries the camera's credentials."""
    config = MusterConfig.model_validate(raw_config)

    with pytest.raises(ConfigError) as exc_info:
        build_pipeline(config, CameraId("till"))

    assert "user:pass@" not in str(exc_info.value)
    assert "frigate" in str(exc_info.value).lower()


def test_an_unknown_camera_is_refused(raw_config: dict[str, Any]) -> None:
    """A silently-absent camera is a camera that silently stops counting."""
    config = MusterConfig.model_validate(raw_config)

    with pytest.raises(ConfigError, match="no-such-camera"):
        build_pipeline(config, CameraId("no-such-camera"))
