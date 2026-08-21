"""The calibration surface over HTTP: one snapshot out, one geometry edit in.

The snapshot route is the only place in the engine where frame bytes reach a socket, and
§13 permits it exactly once: grabbed on demand, streamed once, never persisted. The
tests here cover the HTTP half of that promise — the filesystem half is asserted in
`test_frame_lifetime.py`, which drives the same path under a recorder.

The save route is the engine's first state-changing endpoint, so it carries the checks
that go with that: a body parsed strictly, a camera named by the path, a session, and a
same-origin guard. The session arrived with P3.10 and the origin check now sits behind it
as belt-and-braces (ADR-0018 item 8, discharged by ADR-0019) rather than as the only
thing standing between another LAN host's page and this site's geometry.

Red-first for P3.3.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from mesopic.api.app import create_app
from mesopic.api.auth import Credential
from mesopic.config.schema import LineConfig, MesopicConfig, ZoneConfig
from mesopic.errors import SnapshotUnavailableError
from mesopic.store.store import Store
from mesopic.supervisor.handle import WorkerReport
from mesopic.types import CameraId, CameraState

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "mesopic.yaml"

FRONT_DOOR = CameraId("front-door")
TILL = CameraId("till")

A_JPEG = b"\xff\xd8\xff\xe0 not really a jpeg, but bytes that came from a worker \xff\xd9"

PASSWORD = "correct-horse-battery-staple"  # noqa: S105 - a fixture, not a credential
"""P3.10 put the save route behind a session, so the client here logs in once.

What that guard does — and what happens without it — is `test_api_auth_routes.py`. These
tests are about the save itself, so they start from an operator who is already signed in.
"""

AN_EDIT: dict[str, Any] = {
    "zones": [
        {
            "zone_id": "redrawn",
            "role": "area",
            "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            "metrics": ["occupancy"],
        }
    ],
    "lines": [
        {
            "line_id": "redrawn-line",
            "a": [0.0, 0.5],
            "b": [1.0, 0.5],
            "positive_dir": "in",
            "metrics": ["footfall"],
        }
    ],
}


class FakeEngine:
    """Stands in for the supervisor and the writer, recording what the routes asked for."""

    def __init__(self, config: MesopicConfig) -> None:
        self.config = config
        self.snapshots: list[CameraId] = []
        self.saves: list[tuple[list[ZoneConfig], list[LineConfig]]] = []
        self.unavailable: str | None = None

    async def snapshot(self, camera_id: CameraId) -> bytes:
        self.snapshots.append(camera_id)
        if self.unavailable is not None:
            raise SnapshotUnavailableError(self.unavailable)
        return A_JPEG

    async def save_geometry(
        self, zones: Sequence[ZoneConfig], lines: Sequence[LineConfig]
    ) -> MesopicConfig:
        self.saves.append((list(zones), list(lines)))
        self.config = self.config.model_copy(update={"zones": list(zones), "lines": list(lines)})
        return self.config


@pytest.fixture
def config() -> MesopicConfig:
    parsed: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    return MesopicConfig.model_validate(parsed)


@pytest.fixture
def store(tmp_path: Path, config: MesopicConfig) -> Iterator[Store]:
    with Store(tmp_path / "mesopic.db") as store:
        store.migrate()
        store.apply_config(config)
        yield store


@pytest.fixture
def engine(config: MesopicConfig) -> FakeEngine:
    return FakeEngine(config)


def _reports() -> dict[CameraId, WorkerReport]:
    return {
        FRONT_DOOR: WorkerReport(state=CameraState.STREAMING, consecutive_failures=0),
        TILL: WorkerReport(state=CameraState.STREAMING, consecutive_failures=0),
    }


@pytest.fixture
async def client(
    config: MesopicConfig, store: Store, engine: FakeEngine
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(
        config=config,
        store=store,
        camera_reports=_reports,
        snapshot=engine.snapshot,
        save_geometry=engine.save_geometry,
        credential=Credential(PASSWORD),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        await client.post("/login", data={"password": PASSWORD})
        yield client


# --- The snapshot -------------------------------------------------------------


async def test_the_snapshot_route_serves_the_bytes_the_worker_encoded(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    response = await client.get(f"/api/cameras/{FRONT_DOOR}/snapshot")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == A_JPEG
    assert engine.snapshots == [FRONT_DOOR]


async def test_the_snapshot_is_never_cached(client: httpx.AsyncClient) -> None:
    """ "Never persisted" has to hold at the browser's end of the wire too.

    Without `no-store` the one frame §13 allows out of the process lands in a disk cache,
    which is a frame written to disk by any reading of the invariant that matters.
    """
    response = await client.get(f"/api/cameras/{FRONT_DOOR}/snapshot")

    assert response.headers["cache-control"] == "no-store"


async def test_a_camera_that_cannot_produce_a_frame_is_refused(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    engine.unavailable = "camera 'front-door' is stopped — no snapshot available"

    response = await client.get(f"/api/cameras/{FRONT_DOOR}/snapshot")

    assert response.status_code == 503


async def test_a_refusal_never_quotes_a_stream_url(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    """An RTSP URL carries the camera's credentials, and this is a browser-facing body."""
    engine.unavailable = "rtsp://user:pass@192.168.1.40:554/Streaming/Channels/102 failed"

    response = await client.get(f"/api/cameras/{FRONT_DOOR}/snapshot")

    assert "rtsp://" not in response.text
    assert "pass" not in response.text


async def test_an_unknown_camera_has_no_snapshot(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/cameras/not-a-camera/snapshot")

    assert response.status_code == 404


# --- The page -----------------------------------------------------------------


async def test_the_calibration_page_renders_for_a_configured_camera(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get(f"/calibrate/{FRONT_DOOR}")

    assert response.status_code == 200
    assert "Front door" in response.text


async def test_the_calibration_page_warns_that_a_save_empties_every_zone(
    client: httpx.AsyncClient,
) -> None:
    """P3.8's `Reconfigure` closes what is open before it swaps.

    A save therefore emits an exit for everyone currently resident: dwell and occupancy
    read as everybody leaving and coming back. That is correct and load-bearing, and an
    operator calibrating during trading hours has to be told before they click.
    """
    response = await client.get(f"/calibrate/{FRONT_DOOR}")

    assert "re-enter" in response.text.lower()


async def test_an_unknown_camera_has_no_calibration_page(client: httpx.AsyncClient) -> None:
    response = await client.get("/calibrate/not-a-camera")

    assert response.status_code == 404


# --- The save -----------------------------------------------------------------


async def test_saving_geometry_hands_the_whole_site_to_the_writer(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    """The writer replaces whole blocks, so a save must carry every camera's geometry."""
    response = await client.post(f"/calibrate/{FRONT_DOOR}", json=AN_EDIT)

    assert response.status_code == 204
    ((zones, lines),) = engine.saves
    assert [zone.zone_id for zone in zones] == ["queue-till", "behind-counter", "redrawn"]
    assert [line.line_id for line in lines] == ["redrawn-line"]


async def test_a_saved_shape_belongs_to_the_camera_in_the_path(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    await client.post(f"/calibrate/{FRONT_DOOR}", json=AN_EDIT)

    ((zones, _),) = engine.saves
    assert [zone.camera_id for zone in zones if zone.zone_id == "redrawn"] == [FRONT_DOOR]


async def test_an_invalid_edit_changes_nothing(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    outside_the_unit_square = {
        "zones": [{"zone_id": "z", "polygon": [[0.1, 0.1], [9.9, 0.1], [0.5, 0.5]]}]
    }

    response = await client.post(f"/calibrate/{FRONT_DOOR}", json=outside_the_unit_square)

    assert response.status_code == 400
    assert engine.saves == []


async def test_saving_to_an_unknown_camera_is_refused(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    response = await client.post("/calibrate/not-a-camera", json=AN_EDIT)

    assert response.status_code == 404
    assert engine.saves == []


async def test_a_cross_origin_save_is_refused(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    """The engine has no auth yet and P3.6 binds it to every interface in a container.

    Without this, any page on the LAN could rewrite the site's geometry with a fetch the
    operator never sees.
    """
    response = await client.post(
        f"/calibrate/{FRONT_DOOR}", json=AN_EDIT, headers={"Origin": "http://evil.example"}
    )

    assert response.status_code == 403
    assert engine.saves == []


async def test_a_same_origin_save_is_allowed(client: httpx.AsyncClient, engine: FakeEngine) -> None:
    response = await client.post(
        f"/calibrate/{FRONT_DOOR}", json=AN_EDIT, headers={"Origin": "http://engine"}
    )

    assert response.status_code == 204
    assert len(engine.saves) == 1


async def test_the_dashboard_shows_saved_geometry_without_a_restart(
    client: httpx.AsyncClient,
) -> None:
    """The app is handed a config once, and a save replaces it.

    Without rebinding, every surface built from config — the tiles, the scope list, this
    page's own shape list — keeps rendering the geometry the process started with, and
    the operator's first impression of the editor is that it did nothing.
    """
    await client.post(f"/calibrate/{FRONT_DOOR}", json=AN_EDIT)

    page = await client.get(f"/calibrate/{FRONT_DOOR}")

    assert "redrawn" in page.text


async def test_the_save_route_is_unavailable_without_an_engine(
    config: MesopicConfig, store: Store
) -> None:
    """A dashboard opened against a store with no supervisor says so, rather than 500s."""
    app = create_app(config=config, store=store, camera_reports=_reports)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        assert (await client.post(f"/calibrate/{FRONT_DOOR}", json=AN_EDIT)).status_code == 503
        assert (await client.get(f"/api/cameras/{FRONT_DOOR}/snapshot")).status_code == 503


# --- The geometry list views --------------------------------------------------


async def test_the_camera_list_links_each_camera_to_its_calibration_page(
    client: httpx.AsyncClient,
) -> None:
    """`/cameras` is how an operator reaches the editor without typing a URL."""
    response = await client.get("/cameras")

    assert response.status_code == 200
    assert "Front door" in response.text
    assert f'href="/calibrate/{FRONT_DOOR}"' in response.text


async def test_the_zone_list_shows_every_zone_on_the_site(client: httpx.AsyncClient) -> None:
    response = await client.get("/zones")

    assert response.status_code == 200
    assert "shop-floor" in response.text
    assert "queue-till" in response.text


async def test_the_line_list_shows_every_line_on_the_site(client: httpx.AsyncClient) -> None:
    response = await client.get("/lines")

    assert response.status_code == 200
    assert "door-count" in response.text


async def test_the_lists_follow_a_save(client: httpx.AsyncClient) -> None:
    """Same live-config rule as the dashboard: a list rendered from a stale config lies."""
    await client.post(f"/calibrate/{FRONT_DOOR}", json=AN_EDIT)

    assert "redrawn" in (await client.get("/zones")).text


async def test_a_frigate_camera_is_refused_a_snapshot_with_a_reason(
    client: httpx.AsyncClient,
) -> None:
    """A Frigate camera has tracks and no frames. Sending the request anyway would hang
    until it timed out and leave the editor showing a spinner where a reason belongs."""
    response = await client.get("/api/cameras/till/snapshot")

    assert response.status_code == 422
    assert "Frigate" in response.json()["detail"]
