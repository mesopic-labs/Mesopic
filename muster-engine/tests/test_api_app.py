"""The app itself: what it serves, what it refuses to serve, and on which thread.

`test_every_route_is_a_coroutine` is the load-bearing one. Starlette runs a `def` handler
in a threadpool, and P2.7 established that `sqlite3` connections are thread-affine — so a
handler written without `async` reaches the store from the wrong thread and raises
`ProgrammingError`, but only under a real server. Called directly, as a unit test calls
it, it works perfectly. That combination is how the bug ships.

Red-first for P3.1.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from starlette.routing import Route

from muster.api.app import create_app
from muster.config.schema import MusterConfig
from muster.store.store import Store
from muster.supervisor.handle import WorkerReport
from muster.types import CameraId, CameraState

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"
STATIC_DIR = REPO_ROOT / "muster-engine" / "src" / "muster" / "api" / "static"

FRONT_DOOR = CameraId("front-door")
TILL = CameraId("till")


@pytest.fixture
def config() -> MusterConfig:
    parsed: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    return MusterConfig.model_validate(parsed)


@pytest.fixture
def store(tmp_path: Path, config: MusterConfig) -> Iterator[Store]:
    with Store(tmp_path / "muster.db") as store:
        store.migrate()
        store.apply_config(config)
        yield store


def _streaming() -> WorkerReport:
    return WorkerReport(state=CameraState.STREAMING, consecutive_failures=0)


def _all_streaming() -> dict[CameraId, WorkerReport]:
    return {FRONT_DOOR: _streaming(), TILL: _streaming()}


@pytest.fixture
async def client(config: MusterConfig, store: Store) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(config=config, store=store, camera_reports=_all_streaming)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        yield client


# --- The threading trap -----------------------------------------------------


def test_every_route_is_a_coroutine(config: MusterConfig, store: Store) -> None:
    """A sync handler runs in a threadpool, and the store's connection is thread-affine.

    The failure this prevents is invisible in a unit test and certain in production: the
    handler works when called directly and raises `sqlite3.ProgrammingError` the moment a
    real server runs it off the loop thread.
    """
    app = create_app(config=config, store=store, camera_reports=_all_streaming)

    sync_routes = [
        route.path
        for route in app.routes
        if isinstance(route, Route) and not inspect.iscoroutinefunction(route.endpoint)
    ]

    assert sync_routes == []


# --- /healthz ---------------------------------------------------------------


async def test_healthz_reports_every_configured_camera(client: httpx.AsyncClient) -> None:
    body = (await client.get("/healthz")).json()

    assert {camera["camera_id"] for camera in body["cameras"]} == {FRONT_DOOR, TILL}


async def test_healthz_is_ok_when_every_camera_streams(client: httpx.AsyncClient) -> None:
    body = (await client.get("/healthz")).json()

    assert body["status"] == "ok"


async def test_healthz_degrades_when_a_camera_is_in_backoff(
    config: MusterConfig, store: Store
) -> None:
    app = create_app(
        config=config,
        store=store,
        camera_reports=lambda: {
            FRONT_DOOR: _streaming(),
            TILL: WorkerReport(state=CameraState.BACKOFF, consecutive_failures=7),
        },
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        body = (await client.get("/healthz")).json()

    assert body["status"] == "degraded"


async def test_healthz_answers_200_even_when_degraded(config: MusterConfig, store: Store) -> None:
    """The payload is the signal. A non-200 would make a monitor drop the detail that says why.

    Splitting liveness from readiness is P3.6's, where compose declares the probes.
    """
    app = create_app(config=config, store=store, camera_reports=dict)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        response = await client.get("/healthz")

    assert response.status_code == httpx.codes.OK
    assert response.json()["status"] == "down"


async def test_healthz_reports_the_unsynced_backlog(
    client: httpx.AsyncClient, store: Store
) -> None:
    del store  # the empty store is the point: a fresh box owes the cloud nothing
    body = (await client.get("/healthz")).json()

    assert body["sync"] == {"enabled": False, "unsynced_rows": 0}


async def test_healthz_nulls_what_the_supervisor_cannot_know(
    client: httpx.AsyncClient,
) -> None:
    body = (await client.get("/healthz")).json()

    assert all(camera["last_frame_ts"] is None for camera in body["cameras"])
    assert all(camera["effective_fps"] is None for camera in body["cameras"])


async def test_healthz_never_names_a_source_url(
    client: httpx.AsyncClient, config: MusterConfig
) -> None:
    """An RTSP URL carries the camera's credentials and must not ride out on a health poll."""
    del config
    text = (await client.get("/healthz")).text

    assert "rtsp://" not in text
    assert "_env" not in text


# --- The page ---------------------------------------------------------------


async def test_the_dashboard_renders_with_the_hud_stylesheet(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/")

    assert response.status_code == httpx.codes.OK
    assert "hud.css" in response.text


async def test_the_dashboard_names_every_camera(client: httpx.AsyncClient) -> None:
    response = await client.get("/")

    assert "front-door" in response.text
    assert "till" in response.text


async def test_the_stylesheet_is_served(client: httpx.AsyncClient) -> None:
    response = await client.get("/static/hud.css")

    assert response.status_code == httpx.codes.OK
    assert "--safelight" in response.text


async def test_the_page_loads_nothing_from_the_internet(client: httpx.AsyncClient) -> None:
    """The engine runs on a box with no outbound path by design (ADR-0005).

    A CDN font or a jsdelivr script tag would turn a dashboard load into an off-box
    request, and would break entirely on the airgapped installs this product is for. The
    same assertion guards the labelling clicker (`test_clicker_is_offline.py`).
    """
    text = (await client.get("/")).text

    assert "http://" not in text
    assert "https://" not in text
    assert "//cdn" not in text


# --- /metrics ---------------------------------------------------------------


async def test_prometheus_is_served_when_an_exporter_is_wired(
    config: MusterConfig, store: Store
) -> None:
    """P3.5 renders the exposition and deliberately does not serve it; this is the route."""
    app = create_app(
        config=config,
        store=store,
        camera_reports=_all_streaming,
        render_prometheus=lambda: "muster_up 1.0\n",
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        response = await client.get("/metrics")

    assert response.status_code == httpx.codes.OK
    assert response.text == "muster_up 1.0\n"
    assert response.headers["content-type"].startswith("text/plain")


async def test_prometheus_is_absent_when_the_exporter_is_disabled(
    client: httpx.AsyncClient,
) -> None:
    """404 is the honest answer for a surface the operator turned off in config."""
    response = await client.get("/metrics")

    assert response.status_code == httpx.codes.NOT_FOUND


# --- The stylesheet itself --------------------------------------------------


def test_the_hud_stylesheet_defines_both_themes() -> None:
    """Light, `prefers-color-scheme: dark`, and an explicit `data-theme` override each way.

    A token defined only inside a media query takes its value from the host's theme when
    the query does not match, which is how a dashboard ends up with black text on a black
    panel for exactly the users who set a preference.
    """
    css = (STATIC_DIR / "hud.css").read_text(encoding="utf-8")

    assert "prefers-color-scheme: dark" in css
    assert '[data-theme="dark"]' in css
    assert '[data-theme="light"]' in css


def test_the_hud_stylesheet_honours_reduced_motion() -> None:
    css = (STATIC_DIR / "hud.css").read_text(encoding="utf-8")

    assert "prefers-reduced-motion" in css
