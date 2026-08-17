"""The composition root: one config, one store, one supervisor, one HTTP server.

Everything the engine needs is assembled here and nowhere else, so there is exactly one
place to read to learn what a running box consists of. `muster run` is a thin caller.

**One process, one event loop** (engine-architecture.md §9). The supervisor's work is
I/O-bound — draining queues, writing SQLite, pushing to exporters — and the HTTP server
is the same shape, so they share a loop rather than a thread or a second process. That
also keeps the single-writer rule intact: the API reads the store on the loop thread the
supervisor writes it on, and `sqlite3`'s thread affinity is never tested.

**Whichever half exits first takes the other down.** A dead HTTP server in front of live
camera workers is a box that looks off and is still recording; live workers with no
server is a box nobody can see into. Neither is a state to keep running in.

Implements P3.1.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Callable, Coroutine, Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn

from muster.api.app import create_app
from muster.config.schema import ApiConfig, MusterConfig
from muster.exporters.fanout import build_exporters
from muster.exporters.prometheus import PrometheusExporter
from muster.store.store import Store
from muster.supervisor.handle import WorkerEntry
from muster.supervisor.supervisor import Supervisor
from muster.supervisor.worker import run_camera_worker

if TYPE_CHECKING:
    from fastapi import FastAPI

Serve = Callable[["FastAPI", ApiConfig], Coroutine[object, object, None]]
"""How the app gets served. `serve_uvicorn` in production, a spy in tests."""

SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM)

DATA_DIR_ENV_VAR = "MUSTER_DATA_DIR"
"""Where the writable state goes. The image sets it to `/data` and mounts one volume
there; outside a container the config file's own directory is the obvious answer."""

STORE_FILENAME = "muster.db"


def store_path(
    *,
    config_path: Path,
    data_dir: Path | None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Decide where the SQLite file lives: explicit flag, then env var, then next to config.

    Deliberately not a config key. `muster.yaml` describes the *site* — cameras, geometry,
    thresholds — and where its own compiled form is written is a property of the
    deployment, which is exactly what the image already expresses with `MUSTER_DATA_DIR`.
    Putting it in the file would also mean the path to the database could only be learnt
    by first finding and parsing the database's own config.
    """
    environment = os.environ if env is None else env
    if data_dir is not None:
        return data_dir.expanduser().resolve() / STORE_FILENAME
    from_env = environment.get(DATA_DIR_ENV_VAR)
    if from_env:
        return Path(from_env).expanduser().resolve() / STORE_FILENAME
    return config_path.expanduser().resolve().parent / STORE_FILENAME


class EngineManagedServer(uvicorn.Server):
    """uvicorn, with its signal handling removed so the engine keeps exactly one.

    uvicorn installs handlers with `signal.signal` from inside `serve()` — *after* the
    engine has installed its own — so by default it wins, and the SIGTERM `docker stop`
    sends stops the HTTP server while the camera workers decode on until the container is
    killed. Overriding the hook is what leaves shutdown to `Engine.run`.

    The method is `capture_signals` as of uvicorn 0.52; it was `install_signal_handlers`
    before. Overriding the old name on a new uvicorn is a silent no-op, which is why
    `test_the_real_server_leaves_signal_handling_to_the_engine` asserts the process's
    handler is untouched rather than that some method was overridden.
    """

    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def build_server(app: FastAPI, api: ApiConfig) -> EngineManagedServer:
    """The configured server, unstarted — so a test can inspect it without a port."""
    return EngineManagedServer(
        uvicorn.Config(
            app,
            host=api.host,
            port=api.port,
            log_level="info",
            access_log=False,
        )
    )


async def serve_uvicorn(app: FastAPI, api: ApiConfig) -> None:
    """Run uvicorn on the loop that is already running."""
    await build_server(app, api).serve()


class Engine:
    """One running site: the workers, the store behind them, and the API in front."""

    def __init__(
        self,
        config: MusterConfig,
        store: Store,
        *,
        entry: WorkerEntry = run_camera_worker,
    ) -> None:
        store.migrate()
        store.apply_config(config)
        self.config = config
        self.store = store
        self.supervisor = Supervisor(config, store=store, entry=entry)
        self.supervisor.exporters = build_exporters(config)
        self.app = create_app(
            config=config,
            store=store,
            camera_reports=self.supervisor.camera_reports,
            render_prometheus=_prometheus_renderer(self.supervisor),
        )

    async def run(self, *, serve: Serve = serve_uvicorn) -> None:
        """Run until a signal, a stopped supervisor, or a server that gave up."""
        with _shutdown_on_signal(self.supervisor.request_shutdown):
            engine = asyncio.create_task(self.supervisor.run(), name="supervisor")
            if not self.config.api.enabled:
                await engine
                return
            server = asyncio.create_task(serve(self.app, self.config.api), name="api")
            await _first_to_finish(engine, server, on_stop=self.supervisor.request_shutdown)


async def _first_to_finish(
    engine: asyncio.Task[None], server: asyncio.Task[None], *, on_stop: Callable[[], None]
) -> None:
    """Wait for either half, wind the other down, then re-raise whatever ended it.

    The supervisor is asked to stop rather than cancelled: it has workers to signal, a
    final drain to do and an open minute to close, and a cancellation mid-`stop()` would
    abandon all three.
    """
    done, _ = await asyncio.wait({engine, server}, return_when=asyncio.FIRST_COMPLETED)
    on_stop()
    if not engine.done():
        await engine
    if not server.done():
        server.cancel()
        with suppress(asyncio.CancelledError):
            await server
    for task in done:
        task.result()


def _prometheus_renderer(supervisor: Supervisor) -> Callable[[], str] | None:
    """The `/metrics` route, but only when config enabled that exporter.

    P3.5 built the exporter to render and deliberately not to serve; this is the other
    half. The instance has to be the fan-out's own — each `PrometheusExporter` owns a
    private `CollectorRegistry`, so a second one here would render an empty page while
    the real counters went up somewhere else.
    """
    exporter = supervisor.exporters.get("prometheus")
    if isinstance(exporter, PrometheusExporter):
        return exporter.render
    return None


class _shutdown_on_signal:  # noqa: N801 - a context manager used as a statement, not a type
    """Route SIGINT/SIGTERM to one callback for as long as the block runs.

    Restores whatever was installed before, so a caller that runs an engine twice in one
    process — a test, mostly — does not leave a handler pointing at a dead supervisor.
    """

    def __init__(self, on_signal: Callable[[], None]) -> None:
        self._on_signal = on_signal

    def __enter__(self) -> None:
        loop = asyncio.get_running_loop()
        for received in SHUTDOWN_SIGNALS:
            with suppress(NotImplementedError):  # not every platform has add_signal_handler
                loop.add_signal_handler(received, self._on_signal)

    def __exit__(self, *_: object) -> None:
        loop = asyncio.get_running_loop()
        for received in SHUTDOWN_SIGNALS:
            with suppress(NotImplementedError, RuntimeError):
                loop.remove_signal_handler(received)


__all__ = [
    "DATA_DIR_ENV_VAR",
    "Engine",
    "EngineManagedServer",
    "Serve",
    "build_server",
    "serve_uvicorn",
    "store_path",
]
