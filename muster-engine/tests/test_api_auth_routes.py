"""Login, logout, and what the write guard does to the engine's one write route.

ADR-0019's decisions, as tests: writes need a session and reads do not, an unconfigured
credential refuses writes rather than allowing them, the refusal says which door is shut
and never why, and the password reaches no log line.

The structural test at the bottom is the one that matters longest. P3.4's `/config`
editor is a strictly more powerful write than geometry, and it will be written by
somebody who has forgotten this file exists — so a new state-changing route that carries
no session dependency fails here rather than shipping.

Red-first for P3.10.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Iterator, Sequence
from http import HTTPStatus
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.routing import APIRoute

from muster.api.app import create_app
from muster.api.auth import SESSION_COOKIE, Credential, WriteGuard
from muster.config.schema import LineConfig, MusterConfig, ZoneConfig
from muster.store.store import Store
from muster.supervisor.handle import WorkerReport
from muster.types import CameraId, CameraState

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"

FRONT_DOOR = CameraId("front-door")
PASSWORD = "correct-horse-battery-staple"  # noqa: S105 - a fixture, not a credential

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

EXEMPT_WRITES = frozenset({("POST", "/login"), ("POST", "/logout")})
"""The only state-changing routes that may skip the session guard.

Both are how a session is obtained and disposed of, so requiring one would be circular.
Anything else added here is a decision to ship an unauthenticated write."""

AN_EDIT: dict[str, Any] = {
    "zones": [
        {
            "zone_id": "redrawn",
            "role": "area",
            "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            "metrics": ["occupancy"],
        }
    ],
    "lines": [],
}


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


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


async def _save_geometry(zones: Sequence[ZoneConfig], lines: Sequence[LineConfig]) -> MusterConfig:
    del zones, lines
    msg = "the guard should have refused before the writer was reached"
    raise AssertionError(msg)


def _reports() -> dict[CameraId, WorkerReport]:
    return {FRONT_DOOR: WorkerReport(state=CameraState.STREAMING, consecutive_failures=0)}


def _app(
    config: MusterConfig,
    store: Store,
    *,
    credential: Credential | None,
    clock: FakeClock,
    save_geometry: Any = _save_geometry,
) -> FastAPI:
    return create_app(
        config=config,
        store=store,
        camera_reports=_reports,
        credential=credential,
        save_geometry=save_geometry,
        monotonic=clock,
    )


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://engine")


@pytest.fixture
async def client(
    config: MusterConfig, store: Store, clock: FakeClock
) -> AsyncIterator[httpx.AsyncClient]:
    """An engine with a credential configured and nobody logged in."""
    async with _client(_app(config, store, credential=Credential(PASSWORD), clock=clock)) as c:
        yield c


@pytest.fixture
async def open_client(
    config: MusterConfig, store: Store, clock: FakeClock
) -> AsyncIterator[httpx.AsyncClient]:
    """An engine with no credential configured at all."""
    async with _client(_app(config, store, credential=None, clock=clock)) as c:
        yield c


async def _log_in(client: httpx.AsyncClient) -> httpx.Response:
    return await client.post("/login", data={"password": PASSWORD})


# --- The write guard ----------------------------------------------------------


async def test_a_write_without_a_session_is_refused(client: httpx.AsyncClient) -> None:
    response = await client.post(f"/calibrate/{FRONT_DOOR}", json=AN_EDIT)

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_a_session_admits_a_write(
    config: MusterConfig, store: Store, clock: FakeClock
) -> None:
    saved: list[str] = []

    async def save(zones: Sequence[ZoneConfig], lines: Sequence[LineConfig]) -> MusterConfig:
        # The writer is handed the whole site's geometry, not one camera's — the other
        # cameras' zones ride along, so the edit is identified rather than counted.
        saved.extend(str(zone.zone_id) for zone in zones)
        return config.model_copy(update={"zones": list(zones), "lines": list(lines)})

    app = _app(config, store, credential=Credential(PASSWORD), clock=clock, save_geometry=save)
    async with _client(app) as client:
        await _log_in(client)

        response = await client.post(f"/calibrate/{FRONT_DOOR}", json=AN_EDIT)

    assert response.status_code == HTTPStatus.NO_CONTENT
    assert "redrawn" in saved


async def test_the_guard_runs_before_the_body_is_trusted(client: httpx.AsyncClient) -> None:
    """An unauthenticated caller must not learn whether their payload would have parsed.

    A 400 here rather than a 401 would turn the write route into a free validator for the
    engine's geometry schema, and would mean a body reached the parser before anyone
    established the caller was allowed to send one.
    """
    response = await client.post(f"/calibrate/{FRONT_DOOR}", json={"zones": "not a list"})

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_an_expired_session_no_longer_admits_a_write(
    client: httpx.AsyncClient, clock: FakeClock
) -> None:
    await _log_in(client)

    clock.advance(13 * 60 * 60)

    response = await client.post(f"/calibrate/{FRONT_DOOR}", json=AN_EDIT)
    assert response.status_code == HTTPStatus.UNAUTHORIZED


# --- Logging in ---------------------------------------------------------------


async def test_the_login_page_renders_without_a_session(client: httpx.AsyncClient) -> None:
    response = await client.get("/login")

    assert response.status_code == HTTPStatus.OK
    assert "password" in response.text


async def test_the_correct_password_opens_a_session(client: httpx.AsyncClient) -> None:
    response = await _log_in(client)

    assert response.status_code == HTTPStatus.SEE_OTHER
    assert response.headers["location"] == "/"
    assert client.cookies.get(SESSION_COOKIE)


async def test_the_session_cookie_is_not_reachable_from_script(
    client: httpx.AsyncClient,
) -> None:
    """HttpOnly and SameSite are the whole CSRF/XSS story for an opaque cookie.

    `Secure` is deliberately absent: the LAN default is plain http, and a cookie the
    browser refuses to send back is a login loop with no error message.
    """
    response = await _log_in(client)

    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert "secure" not in cookie


async def test_the_wrong_password_opens_nothing(client: httpx.AsyncClient) -> None:
    response = await client.post("/login", data={"password": "wrong"})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert not client.cookies.get(SESSION_COOKIE)


async def test_a_login_with_no_password_at_all_is_refused(client: httpx.AsyncClient) -> None:
    response = await client.post("/login", data={})

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_an_oversized_login_body_is_refused_before_it_is_parsed(
    client: httpx.AsyncClient,
) -> None:
    """Unbounded input at a trust boundary gets an explicit limit, like every other one."""
    response = await client.post("/login", data={"password": "x" * 10_000})

    assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE


async def test_an_oversized_body_with_no_declared_length_is_still_refused(
    client: httpx.AsyncClient,
) -> None:
    """A chunked request carries no `content-length`, so the pre-check cannot see it.

    Reading the body first and measuring it after would mean the ceiling is enforced only
    once the bytes are already in memory — which is the thing a ceiling exists to prevent.
    """

    sent = 0

    async def chunks() -> AsyncIterator[bytes]:
        nonlocal sent
        for _ in range(20):
            sent += 1
            yield b"password=" + b"x" * 1_000

    response = await client.post(
        "/login",
        content=chunks(),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    # The status code alone would pass with the whole 20 KB already in memory. What is
    # actually being asserted is that the engine stopped reading once the ceiling was hit.
    assert sent < 20


async def test_the_password_never_reaches_a_log_record(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The one value the operator typed, held to the rule that keeps RTSP URLs out of logs."""
    with caplog.at_level(logging.DEBUG):
        await _log_in(client)
        await client.post("/login", data={"password": "a-wrong-one"})

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert PASSWORD not in logged
    assert "a-wrong-one" not in logged


async def test_the_password_never_reaches_a_response_body(client: httpx.AsyncClient) -> None:
    response = await client.post("/login", data={"password": "a-wrong-one"})

    assert "a-wrong-one" not in response.text


# --- The throttle over HTTP ---------------------------------------------------


async def test_enough_failures_refuse_even_the_correct_password(
    client: httpx.AsyncClient,
) -> None:
    for _ in range(5):
        await client.post("/login", data={"password": "wrong"})

    response = await _log_in(client)

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert not client.cookies.get(SESSION_COOKIE)


async def test_the_cooldown_lets_the_operator_back_in(
    client: httpx.AsyncClient, clock: FakeClock
) -> None:
    for _ in range(5):
        await client.post("/login", data={"password": "wrong"})

    clock.advance(61.0)
    response = await _log_in(client)

    assert response.status_code == HTTPStatus.SEE_OTHER


async def test_a_throttled_refusal_is_indistinguishable_from_a_wrong_password(
    client: httpx.AsyncClient,
) -> None:
    """Telling the caller they are throttled tells them the password was right."""
    wrong = await client.post("/login", data={"password": "wrong"})
    for _ in range(5):
        await client.post("/login", data={"password": "wrong"})
    throttled = await _log_in(client)

    assert throttled.status_code == wrong.status_code
    assert throttled.text == wrong.text


# --- Logging out --------------------------------------------------------------


async def test_logout_revokes_the_session(client: httpx.AsyncClient) -> None:
    await _log_in(client)

    await client.post("/logout")

    response = await client.post(f"/calibrate/{FRONT_DOOR}", json=AN_EDIT)
    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_logout_without_a_session_is_not_an_error(client: httpx.AsyncClient) -> None:
    response = await client.post("/logout")

    assert response.status_code == HTTPStatus.SEE_OTHER


# --- No credential configured -------------------------------------------------


async def test_without_a_credential_a_write_is_refused_as_unavailable(
    open_client: httpx.AsyncClient,
) -> None:
    """503, not 401: there is no session that could ever satisfy this, so a login prompt lies."""
    response = await open_client.post(f"/calibrate/{FRONT_DOOR}", json=AN_EDIT)

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE


def _save_button(page: str) -> str:
    found = re.search(r"<button[^>]*data-act=\"save\"[^>]*>", page)
    assert found is not None, "the calibration page has no save control at all"
    return found.group(0)


async def test_without_a_credential_the_calibration_page_offers_no_save(
    open_client: httpx.AsyncClient,
) -> None:
    """The page already renders read-only for an engine with no writer; no credential joins it.

    Asserted on the control rather than on the word "save", which the page's own prose
    uses while explaining what a save does.
    """
    response = await open_client.get(f"/calibrate/{FRONT_DOOR}")

    assert response.status_code == HTTPStatus.OK
    assert "disabled" in _save_button(response.text)


async def test_with_a_credential_the_calibration_page_offers_a_save(
    client: httpx.AsyncClient,
) -> None:
    """Enabled even before login: the 401 is what sends the operator to the login page.

    A control disabled until you are logged in would need the page to know about the
    session, and the redirect already covers it with one code path instead of two.
    """
    response = await client.get(f"/calibrate/{FRONT_DOOR}")

    assert "disabled" not in _save_button(response.text)


async def test_without_a_credential_there_is_no_login_page(
    open_client: httpx.AsyncClient,
) -> None:
    """A form that cannot succeed is worse than an honest 404."""
    response = await open_client.get("/login")

    assert response.status_code == HTTPStatus.NOT_FOUND


# --- What stays open ----------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/fragments/board",
        "/healthz",
        "/cameras",
        "/zones",
        "/lines",
        f"/calibrate/{FRONT_DOOR}",
    ],
)
async def test_the_read_surface_needs_no_session(client: httpx.AsyncClient, path: str) -> None:
    """ADR-0019 item 1: a LAN dashboard that needs a password to look at is a worse product."""
    response = await client.get(path)

    assert response.status_code == HTTPStatus.OK


async def test_the_metrics_api_needs_no_session(client: httpx.AsyncClient) -> None:
    response = await client.get(
        "/api/metrics", params={"from": "2026-08-18T00:00:00Z", "to": "2026-08-18T01:00:00Z"}
    )

    assert response.status_code == HTTPStatus.OK


# --- The structural guarantee -------------------------------------------------


def test_every_write_route_requires_a_session(config: MusterConfig, store: Store) -> None:
    """No state-changing route ships without the guard, including ones not written yet.

    This is the same shape as `test_every_route_is_a_coroutine`: the property is true of
    the app rather than of a route somebody remembered, so it is asserted over the route
    table instead of once per endpoint.
    """
    app = _app(config, store, credential=Credential(PASSWORD), clock=FakeClock())

    unguarded = [
        (method, route.path)
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods or set()
        if method not in SAFE_METHODS
        and (method, route.path) not in EXEMPT_WRITES
        and not any(
            isinstance(dependency.call, WriteGuard) for dependency in route.dependant.dependencies
        )
    ]

    assert unguarded == []
