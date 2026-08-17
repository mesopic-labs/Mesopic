"""The worker control channel: typed messages in, a snapshot back.

Until P3.8 the boundary was one-way in every useful sense — `Queue[str]` carrying a stop,
`Queue[RawEvent]` carrying events out. The calibration view needs a frame, and the zone
editor needs geometry to change under a running camera, and neither fits through that.

The constraint that shapes all of it is the import contract "The dashboard serves
numbers, never frames": `muster.api` may not import a codec, so the worker that already
owns the stream encodes, and the API only ever handles bytes.

Two things these tests exist to hold:

* **A snapshot is answered from the frame the loop already has.** No second capture, no
  second RTSP session, and nothing retained after the reply is put.
* **Reconfigure closes before it swaps.** `test_reconfigure.py` owns that rule at the
  analytics level; here it is checked where it actually has to survive — inside the loop.

Red-first for P3.8.
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
from scripted_worker import run_until_stopped

from muster.analytics.site_geometry import SiteGeometry
from muster.config.schema import MusterConfig
from muster.errors import SnapshotUnavailableError, StreamDropped
from muster.store.store import Store
from muster.supervisor.control import Reconfigure, Snapshot, SnapshotReply, Stop
from muster.supervisor.snapshot import MAX_EDGE_PX, encode_snapshot
from muster.supervisor.supervisor import Supervisor
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

FRONT_DOOR = CameraId("front-door")
T0 = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"


@pytest.fixture
def config() -> MusterConfig:
    parsed = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    return MusterConfig.model_validate(parsed)


def _frame(offset_s: float, *, width: int = 64, height: int = 48) -> DecodedFrame:
    """A gradient rather than a flat fill: a flat image encodes to almost nothing and
    would make a size assertion pass for the wrong reason."""
    image = np.tile(np.linspace(0, 255, width, dtype=np.uint8).reshape(1, width, 1), (height, 1, 3))
    return DecodedFrame(
        camera_id=FRONT_DOOR,
        ts=FrameTs(T0 + timedelta(seconds=offset_s)),
        image=image,
        width=width,
        height=height,
    )


class _Source:
    def __init__(self, count: int) -> None:
        self._count = count

    def frames(self) -> Iterator[DecodedFrame]:
        for index in range(self._count):
            yield _frame(index)
        message = "camera 'front-door': stream ended"
        raise StreamDropped(message)

    def close(self) -> None:
        return None


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
    def __init__(self) -> None:
        self.reconfigured: list[Any] = []

    def on_tracks(self, camera_id: CameraId, tracks: list[Track]) -> list[RawEvent]:
        del tracks
        del camera_id
        return []

    def reconfigure(self, geometry: Any, *, camera_id: CameraId, ts: FrameTs) -> list[RawEvent]:
        self.reconfigured.append(geometry)
        return [
            RawEvent(
                camera_id=camera_id,
                ts=ts,
                kind=EventKind.ZONE_EXIT,
                track_id=TrackId(1),
            )
        ]


class _Sampler:
    def is_due(self, _ts: FrameTs) -> bool:
        return True

    def set_target_fps(self, target_fps: float) -> None:
        del target_fps


class _Sink:
    def __init__(self) -> None:
        self.items: list[object] = []

    def put_nowait(self, item: object) -> None:
        self.items.append(item)


class _Q:
    """A stand-in for a `multiprocessing.Queue` with the two methods the loop uses.

    The real queue is exercised across a real process boundary in `test_supervisor.py`;
    here the loop's own behaviour is what is under test, and a real queue would only add
    a scheduler to it.
    """

    def __init__(self, *items: object) -> None:
        self.items = list(items)
        self.put_items: list[Any] = []

    def get_nowait(self) -> object:
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)

    def put_nowait(self, item: object) -> None:
        self.put_items.append(item)


def _geometry() -> SiteGeometry:
    """A real compiled geometry, so `Reconfigure` carries what it will carry in life."""
    return SiteGeometry.compile(
        MusterConfig.model_validate(
            {
                "site": {"site_id": "test-site"},
                "cameras": [
                    {
                        "camera_id": FRONT_DOOR,
                        "name": "Front door",
                        "source": {"kind": "rtsp", "url_env": "MUSTER_TEST_RTSP"},
                        "reference_resolution": [1920, 1080],
                    }
                ],
            }
        )
    )


def _run(control: _Q | None = None, snapshots: _Q | None = None, **kw: Any) -> Any:
    """The fakes are deliberately untyped here: `camera_loop` takes protocols, and the
    point of these tests is the loop's own plumbing, not the parts' conformance."""
    analytics = kw.pop("analytics", _Analytics())
    sink = kw.pop("sink", _Sink())
    parts: dict[str, Any] = {
        "source": _Source(kw.pop("frames", 3)),
        "sampler": _Sampler(),
        "detector": _Detector(),
        "tracker": _Tracker(),
        "analytics": analytics,
        "sink": sink,
        "control": control,
        "snapshots": snapshots,
    }
    return camera_loop(FRONT_DOOR, fps_min=1.0, fps_max=5.0, **parts, **kw), analytics, sink


# --- Encoding ---------------------------------------------------------------


def test_a_snapshot_encodes_to_jpeg_bytes() -> None:
    encoded = encode_snapshot(_frame(0))

    assert encoded[:2] == b"\xff\xd8"  # SOI: this is a JPEG, not a raw buffer
    assert encoded[-2:] == b"\xff\xd9"


def test_a_large_frame_is_scaled_down_before_encoding() -> None:
    """The browser needs a backdrop to draw on, not the sensor's full resolution. An
    uncapped 4K snapshot is megabytes across a queue for a canvas a fraction the size."""
    encoded = encode_snapshot(_frame(0, width=4096, height=2160))
    small = encode_snapshot(_frame(0, width=MAX_EDGE_PX, height=MAX_EDGE_PX // 2))

    assert len(encoded) <= len(small) * 2


# --- The loop answers ------------------------------------------------------


def test_a_snapshot_request_is_answered_with_the_frame_in_hand() -> None:
    snapshots = _Q()
    _run(control=_Q(Snapshot(request_id="abc")), snapshots=snapshots)

    assert len(snapshots.put_items) == 1
    reply = snapshots.put_items[0]
    assert isinstance(reply, SnapshotReply)
    assert reply.request_id == "abc"
    assert reply.jpeg is not None


def test_a_snapshot_does_not_stop_the_loop() -> None:
    """A calibration grab must not cost the operator their stream."""
    stats, _, _ = _run(control=_Q(Snapshot(request_id="abc")), snapshots=_Q(), frames=3)

    assert stats.frames_admitted == 3


def test_a_stop_still_stops_the_loop() -> None:
    stats, _, _ = _run(control=_Q(Stop()), frames=5)

    assert stats.frames_admitted == 0


def test_a_snapshot_with_nowhere_to_reply_is_dropped_not_raised() -> None:
    """A reply queue is optional in the same way `control` is, and a worker must not die
    because nobody was listening."""
    stats, _, _ = _run(control=_Q(Snapshot(request_id="abc")), snapshots=None, frames=2)

    assert stats.frames_admitted == 2


# --- Reconfigure ------------------------------------------------------------


def test_a_reconfigure_reaches_the_analytics() -> None:
    geometry = _geometry()
    _, analytics, _ = _run(control=_Q(Reconfigure(geometry=geometry)), frames=2)

    assert analytics.reconfigured == [geometry]


def test_the_events_a_reconfigure_closes_are_emitted() -> None:
    """The whole point of closing before swapping is that the exits reach the store. A
    reconfigure that dropped them on the floor would look identical from inside the loop.
    """
    _, _, sink = _run(control=_Q(Reconfigure(geometry=_geometry())), frames=2)

    assert [event.kind for event in sink.items] == [EventKind.ZONE_EXIT]


def test_one_message_is_handled_per_frame() -> None:
    """`control` is drained one message per iteration, so a burst cannot starve the
    stream. Two requests, three frames: both are answered."""
    snapshots = _Q()
    _run(
        control=_Q(Snapshot(request_id="a"), Snapshot(request_id="b")),
        snapshots=snapshots,
        frames=3,
    )

    assert [reply.request_id for reply in snapshots.put_items] == ["a", "b"]


# --- The supervisor's side --------------------------------------------------


async def test_a_snapshot_of_an_unconfigured_camera_is_refused(
    tmp_path: Path, config: MusterConfig
) -> None:
    with Store(tmp_path / "muster.db") as store:
        store.migrate()
        supervisor = Supervisor(config, store=store, entry=run_until_stopped)
        with pytest.raises(SnapshotUnavailableError, match="not configured"):
            await supervisor.snapshot(CameraId("no-such-camera"))


async def test_a_snapshot_of_a_dead_camera_names_its_state(
    tmp_path: Path, config: MusterConfig
) -> None:
    """The operator's next action is to fix the stream, so the refusal has to say that
    rather than time out behind a spinner."""
    with Store(tmp_path / "muster.db") as store:
        store.migrate()
        supervisor = Supervisor(config, store=store, entry=run_until_stopped)
        with pytest.raises(SnapshotUnavailableError, match="backoff"):
            await supervisor.snapshot(CameraId("front-door"))


async def test_a_refusal_never_names_the_source_url(tmp_path: Path, config: MusterConfig) -> None:
    """An RTSP URL carries the camera's credentials, and this message reaches a browser."""
    with Store(tmp_path / "muster.db") as store:
        store.migrate()
        supervisor = Supervisor(config, store=store, entry=run_until_stopped)
        with pytest.raises(SnapshotUnavailableError) as raised:
            await supervisor.snapshot(CameraId("front-door"))

    assert "rtsp://" not in str(raised.value)
    assert "user:pass" not in str(raised.value)


async def test_reload_compiles_before_it_sends(tmp_path: Path, config: MusterConfig) -> None:
    """A config that cannot compile must fail where an operator can be told, not inside a
    worker whose only way of complaining is to die."""
    with Store(tmp_path / "muster.db") as store:
        store.migrate()
        supervisor = Supervisor(config, store=store, entry=run_until_stopped)

        await supervisor.reload(config)

    assert supervisor.live_workers == 0
