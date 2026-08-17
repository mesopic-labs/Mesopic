"""`muster run` — the composition root, and how it comes apart.

`make run` has shelled to `muster run --config ./muster.yaml` since P0; until P3.1 the
command did not exist. Everything the engine needs is assembled exactly once, here, so
these tests are mostly about the joins:

* the store is migrated and the config applied *before* anything reads either;
* `/healthz` reports the running supervisor's cameras, which is only true if the app was
  handed `camera_states` rather than a snapshot taken at startup;
* whichever half exits first takes the other down. A dead HTTP server with live camera
  workers behind it is a box that looks off and is still recording;
* SIGTERM reaches the supervisor. uvicorn installs its own signal handlers by default
  and would otherwise swallow it, leaving workers orphaned on every `docker stop`.

Red-first for P3.1.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from scripted_worker import run_until_stopped
from typer.testing import CliRunner

from muster import cli
from muster.config.schema import ApiConfig, MusterConfig
from muster.runner import DATA_DIR_ENV_VAR, Engine, build_server, store_path
from muster.store.store import Store
from muster.types import MetricName, MetricRow, MinuteBucket

RUNNER = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"


@pytest.fixture
def config() -> MusterConfig:
    """The worked example with its exporters off — they need secrets and peers.

    `build_exporters` refuses rather than degrades when an enabled exporter has no
    credential, which is right, and which would otherwise make every test here about the
    webhook's signing key instead of about the wiring.
    """
    parsed: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    parsed["exporters"] = {}
    return MusterConfig.model_validate(parsed)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    """Deliberately NOT migrated: whether the engine migrates it is what is under test."""
    with Store(tmp_path / "muster.db") as store:
        yield store


class SpyServer:
    """Stands in for uvicorn. Records what it was told to serve, and how it ended."""

    def __init__(self, *, runs_for_s: float = 0.05) -> None:
        self.app: Any = None
        self.api: ApiConfig | None = None
        self.cancelled = False
        self._runs_for_s = runs_for_s

    async def __call__(self, app: Any, api: ApiConfig) -> None:
        self.app = app
        self.api = api
        try:
            await asyncio.sleep(self._runs_for_s)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


async def _stop_shortly(engine: Engine, after_s: float = 0.05) -> None:
    await asyncio.sleep(after_s)
    engine.supervisor.request_shutdown()


# --- Wiring -----------------------------------------------------------------


async def test_the_store_is_migrated_and_the_config_applied(
    config: MusterConfig, store: Store
) -> None:
    """A fresh box has no database. Reading one before it exists is the first thing tried."""
    engine = Engine(config, store, entry=run_until_stopped)

    await engine.run(serve=SpyServer())

    cameras = {
        row[0] for row in store._connection.execute("SELECT camera_id FROM cameras").fetchall()
    }
    assert cameras == {camera.camera_id for camera in config.cameras}


async def test_the_server_is_given_the_configured_bind(config: MusterConfig, store: Store) -> None:
    engine = Engine(config, store, entry=run_until_stopped)
    server = SpyServer()

    await engine.run(serve=server)

    assert server.api is not None
    assert (server.api.host, server.api.port) == (config.api.host, config.api.port)


async def test_healthz_sees_the_running_supervisors_cameras(
    config: MusterConfig, store: Store
) -> None:
    """Proves the app holds `camera_states` itself, not a copy read once at startup."""
    engine = Engine(config, store, entry=run_until_stopped)
    seen: dict[str, Any] = {}

    async def capture(app: Any, api: ApiConfig) -> None:
        del api
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
            seen.update((await client.get("/healthz")).json())

    await engine.run(serve=capture)

    assert {camera["camera_id"] for camera in seen["cameras"]} == {
        camera.camera_id for camera in config.cameras
    }
    assert seen["status"] == "ok"


async def test_the_api_can_be_switched_off(config: MusterConfig, store: Store) -> None:
    """An engine with no dashboard is a supported deployment, not a broken one."""
    headless = MusterConfig.model_validate({**config.model_dump(), "api": {"enabled": False}})
    engine = Engine(headless, store, entry=run_until_stopped)
    server = SpyServer()

    await asyncio.gather(engine.run(serve=server), _stop_shortly(engine))

    assert server.app is None


async def test_metrics_is_served_from_the_fanouts_own_prometheus_exporter(
    config: MusterConfig, store: Store
) -> None:
    """A second `PrometheusExporter` would render an empty page forever.

    Each instance owns a private `CollectorRegistry`, so wiring the route to a fresh one
    scrapes clean and reports nothing while the real counters climb somewhere else. The
    test crosses back into the shipped path deliberately: it pushes a row through the
    supervisor's own fan-out and then reads the HTTP route (the P3.5 lesson).
    """
    scraped = MusterConfig.model_validate(
        {**config.model_dump(), "exporters": {"prometheus": {"enabled": True}}}
    )
    engine = Engine(scraped, store, entry=run_until_stopped)
    row = MetricRow(
        camera_id=config.cameras[0].camera_id,
        bucket=MinuteBucket(datetime(2026, 8, 17, 9, 30, tzinfo=UTC)),
        metric=MetricName.FOOTFALL,
        scope_id=None,
        value=7.0,
        sample_count=1,
    )

    engine.supervisor.exporters.on_metrics([row])
    transport = httpx.ASGITransport(app=engine.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        exposition = (await client.get("/metrics")).text

    assert "muster_metric" in exposition
    assert "7.0" in exposition


# --- Coming apart -----------------------------------------------------------


async def test_a_server_that_exits_stops_the_workers(config: MusterConfig, store: Store) -> None:
    """A dead dashboard with live workers behind it is a box that looks off and is not."""
    engine = Engine(config, store, entry=run_until_stopped)

    await engine.run(serve=SpyServer(runs_for_s=0.05))

    assert engine.supervisor.live_workers == 0


async def test_a_supervisor_that_stops_cancels_the_server(
    config: MusterConfig, store: Store
) -> None:
    engine = Engine(config, store, entry=run_until_stopped)
    server = SpyServer(runs_for_s=30.0)

    await asyncio.gather(engine.run(serve=server), _stop_shortly(engine))

    assert server.cancelled is True


async def test_sigterm_shuts_the_engine_down(config: MusterConfig, store: Store) -> None:
    """`docker stop` sends SIGTERM. uvicorn would eat it and leave the workers running."""
    engine = Engine(config, store, entry=run_until_stopped)
    server = SpyServer(runs_for_s=30.0)

    async def sigterm_shortly() -> None:
        await asyncio.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)

    await asyncio.wait_for(
        asyncio.gather(engine.run(serve=server), sigterm_shortly()), timeout=10.0
    )

    assert server.cancelled is True
    assert engine.supervisor.live_workers == 0


def test_the_real_server_leaves_signal_handling_to_the_engine(
    config: MusterConfig, store: Store
) -> None:
    """The one test that crosses back into the shipped server instead of the spy.

    Every other test here injects `SpyServer`, which cannot notice what uvicorn does to
    the process's signal handlers — and uvicorn's `capture_signals` installs its own with
    `signal.signal`, *inside* `serve()`, so it lands after the engine's and wins. SIGTERM
    would reach uvicorn and never the supervisor: the workers would keep decoding until
    `docker stop` ran out of patience and killed the container.

    (The API moved: uvicorn ≤ 0.51 called this `install_signal_handlers`. Overriding the
    old name on 0.52 silently does nothing at all, which is how this was found.)
    """
    engine = Engine(config, store, entry=run_until_stopped)
    server = build_server(engine.app, config.api)
    before = signal.getsignal(signal.SIGTERM)

    with server.capture_signals():
        during = signal.getsignal(signal.SIGTERM)

    assert during is before


# --- Where the database goes ------------------------------------------------


def test_the_store_lives_beside_the_config_by_default(tmp_path: Path) -> None:
    """A self-hoster who ran `muster run --config ./muster.yaml` gets `./muster.db`."""
    assert store_path(config_path=tmp_path / "muster.yaml", data_dir=None, env={}) == (
        tmp_path / "muster.db"
    )


def test_the_data_dir_env_var_wins_over_the_config_location(tmp_path: Path) -> None:
    """The image sets `MUSTER_DATA_DIR=/data` and mounts one volume there."""
    volume = tmp_path / "data"

    resolved = store_path(
        config_path=tmp_path / "etc" / "muster.yaml",
        data_dir=None,
        env={DATA_DIR_ENV_VAR: str(volume)},
    )

    assert resolved == volume / "muster.db"


def test_an_explicit_data_dir_wins_over_the_env_var(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"

    resolved = store_path(
        config_path=tmp_path / "muster.yaml",
        data_dir=explicit,
        env={DATA_DIR_ENV_VAR: str(tmp_path / "ignored")},
    )

    assert resolved == explicit / "muster.db"


# --- The command itself -----------------------------------------------------


def test_run_reports_a_missing_config_without_a_traceback(tmp_path: Path) -> None:
    """The first thing a new user gets wrong is the path. It deserves a sentence."""
    missing = tmp_path / "nope.yaml"

    result = RUNNER.invoke(cli.app, ["run", "--config", str(missing)])

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert str(missing) in result.output


def test_run_never_prints_a_source_url(tmp_path: Path, config: MusterConfig) -> None:
    """An RTSP URL is a credential, and a rejected config is where one gets echoed.

    The leaking shape P2.1 found: the failure is on a key *next to* `source`, so the
    whole camera block — credentials included — becomes pydantic's `input_value`.
    """
    document = config.model_dump(mode="json")
    document["cameras"][0]["source"] = {
        "kind": "rtsp",
        "url": "rtsp://user:pass@192.0.2.10:554/Streaming/Channels/101",
    }
    document["cameras"][0]["unknown_key"] = "boom"
    path = tmp_path / "muster.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    result = RUNNER.invoke(cli.app, ["run", "--config", str(path)])

    assert result.exit_code != 0
    assert "unknown_key" in result.output, "the rejection must still say what was wrong"
    assert "rtsp://" not in result.output
    assert "user:pass" not in result.output
