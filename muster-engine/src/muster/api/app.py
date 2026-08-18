"""FastAPI app factory for the local dashboard.

Routes (engine-architecture.md §13):

| Route            | Purpose                                                          |
|------------------|------------------------------------------------------------------|
| `/`              | Live occupancy tiles, core-six charts, heatmap, staff/customer    |
| `/cameras` `/zones` `/lines` | The site's geometry, each row linking to its canvas   |
| `/calibrate/{camera_id}` | Draw zones and lines over one ephemeral snapshot          |
| `/config`        | Render, validate, and hot-reload `muster.yaml`                    |
| `/api/metrics`   | Read-only time-series JSON                                        |
| `/healthz`       | Machine-readable engine and per-camera health                     |
| `/metrics`       | Prometheus exposition                                             |

**The calibration view is the one place a frame is briefly shown.** A single snapshot is
grabbed on demand, streamed to the browser to draw zones and lines on, and never written
to disk or uploaded. Geometry saves as normalized coordinates, so the snapshot is
disposable — which is what keeps the "frames discarded immediately" guarantee intact even
during setup.

Two rules this module is built around:

* **Every handler is `async def`.** Starlette runs a sync handler in a threadpool, and
  `sqlite3` connections are thread-affine (the same fact that keeps the supervisor's
  writes on its loop thread), so a sync handler touching the store raises
  `ProgrammingError` — but only under a real server, never when a test calls it directly.
  `test_every_route_is_a_coroutine` is what keeps this true.
* **The engine is injected as callables, not as objects.** The app never imports the
  supervisor or an exporter; it is handed `camera_states` and `render_prometheus` by the
  composition root. Keeps the dependency arrow pointing one way, and makes every route
  testable without spawning a worker.

Implements P3.1 (`/`, `/healthz`, `/api/metrics`, `/metrics`). P3.2 added the board;
P3.3's calibration routes live in `calibration.py` and are included here; P3.4 owns
`/config`.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from muster.api.board import (
    DEFAULT_WINDOW,
    BoardWindow,
    as_series,
    charts_of,
    exposure_of,
    freshness_for,
    human_duration,
    scope_slots,
    tiles_for,
)
from muster.api.calibration import calibration_router
from muster.api.health import (
    DiskHealth,
    EngineHealth,
    SyncHealth,
    camera_health,
    engine_health,
)
from muster.config.schema import LineConfig, MusterConfig, ZoneConfig
from muster.supervisor.handle import WorkerReport
from muster.types import CameraId, MetricName

Snapshotter = Callable[[CameraId], Awaitable[bytes]]
"""`Supervisor.snapshot` — one frame from the worker that already owns the stream."""

GeometrySaver = Callable[[Sequence[ZoneConfig], Sequence[LineConfig]], Awaitable[MusterConfig]]
"""Write the site's geometry and hot-reload it, returning the config that landed.

Returning the *reloaded* config rather than `None` is what keeps the dashboard honest
after a save: the file on disk is the authority (§13.1), so what the app should render
afterwards is what came back off it, not what the browser asked for."""

if TYPE_CHECKING:
    from typing import Self

    from muster.store import Store

STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

MAX_LIMIT = 20_000
"""Rows one request may return. A minute-bucketed month of the core six is under this;
an unbounded read of a long history is how a dashboard poll OOMs the box."""

DEFAULT_LIMIT = 5_000

MAX_WINDOW = timedelta(days=31)
"""Widest range a single request may span, for the reason `MAX_LIMIT` exists. A longer
view is a pre-aggregated one, which is the cloud's job (ADR-0005)."""

MAX_ID_LENGTH = 128
"""Long enough for any id a config can name, short enough that nothing large is echoed
into a log line or a query."""

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _utc_now() -> datetime:
    """The board's window is wall-clock, and everything stored is UTC.

    Injectable for the same reason the supervisor's clocks are: a test that has to wait
    for real minutes to pass is a test nobody runs.
    """
    return datetime.now(UTC)


class MetricsQuery(BaseModel):
    """The `/api/metrics` query string, parsed into a typed window at the edge.

    Untrusted input becomes a strict structure here and is rejected on first
    inconsistency — it is never sanitised and passed on.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    start: AwareDatetime = Field(alias="from")
    """Timezone-aware on purpose: everything stored is UTC, and a bare local timestamp
    would silently shift the window by the operator's offset."""
    end: AwareDatetime = Field(alias="to")
    camera_id: Annotated[str, Field(max_length=MAX_ID_LENGTH)] | None = None
    metric: list[MetricName] | None = None
    limit: Annotated[int, Field(gt=0, le=MAX_LIMIT)] = DEFAULT_LIMIT

    @model_validator(mode="after")
    def _window_must_be_forward_and_bounded(self) -> Self:
        if self.end <= self.start:
            msg = "the window must end after it starts"
            raise ValueError(msg)
        if self.end - self.start > MAX_WINDOW:
            msg = "the window is too wide"
            raise ValueError(msg)
        return self


def create_app(
    *,
    config: MusterConfig,
    store: Store,
    camera_reports: Callable[[], Mapping[CameraId, WorkerReport]] = dict,
    render_prometheus: Callable[[], str] | None = None,
    snapshot: Snapshotter | None = None,
    save_geometry: GeometrySaver | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    clock: Callable[[], datetime] = _utc_now,
) -> FastAPI:
    """Build the local dashboard app around an already-open store.

    `camera_reports` is `Supervisor.camera_reports` in production and a literal in tests.
    `render_prometheus` is `PrometheusExporter.render` when that exporter is enabled and
    `None` when it is not — in which case `/metrics` is not registered at all, because a
    404 is the honest answer for a surface the operator turned off.

    `snapshot` and `save_geometry` are the calibration view's two halves, and both are
    `None` for an app built without a supervisor behind it. Their routes still exist in
    that case and answer 503: the surface is real, the engine behind it is not running.
    """
    started_at = monotonic()
    live_config = config
    """The config the app renders from.

    Rebound by a successful save, because after one the file on disk no longer matches
    the value this app was constructed with — and every surface built from config (the
    tiles, the scope colours, the shape list) would otherwise keep rendering the geometry
    the process started with until someone restarted it."""

    def current() -> MusterConfig:
        return live_config

    def _adopt(saved: MusterConfig) -> None:
        nonlocal live_config
        live_config = saved

    app = FastAPI(title="Muster", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=TEMPLATES_DIR)

    @app.exception_handler(RequestValidationError)
    async def _generic_rejection(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Say no without saying what was sent.

        FastAPI's default body quotes the offending input straight back, and an RTSP URL
        carries the camera's credentials — the same leak P2.1 closed in the config
        loader's tracebacks. Detail belongs in the log, not in the response.
        """
        del request, exc
        return JSONResponse(status_code=400, content={"detail": "invalid request"})

    @app.get("/healthz")
    async def healthz() -> EngineHealth:
        return _health(
            current(),
            store,
            reports=camera_reports(),
            uptime_s=monotonic() - started_at,
        )

    @app.get("/api/metrics")
    async def api_metrics(query: Annotated[MetricsQuery, Query()]) -> dict[str, Any]:
        rows = store.metrics_between(
            start=query.start,
            end=query.end,
            limit=query.limit,
            camera_id=CameraId(query.camera_id) if query.camera_id else None,
            metrics=query.metric,
        )
        return {"series": as_series(rows), "truncated": len(rows) >= query.limit}

    def _board_context(window: BoardWindow) -> dict[str, Any]:
        end = clock()
        rows = store.metrics_between(start=end - window.span, end=end, limit=MAX_LIMIT)
        health = _health(
            current(), store, reports=camera_reports(), uptime_s=monotonic() - started_at
        )
        return {
            "site_id": current().site.site_id,
            "window": window,
            "windows": list(BoardWindow),
            # The exposure strip's axis. UTC on the page because UTC is what is stored —
            # a dashboard that silently localises one clock and not the other is worse
            # than one that is consistently in a timezone you have to know.
            "since": end - window.span,
            "now": end,
            "tiles": tiles_for(current(), rows=rows),
            "charts": charts_of(current(), rows=rows),
            "slots": scope_slots(current()),
            "exposure": exposure_of(rows, end=end, window=window),
            "health": health,
            "freshness": freshness_for(health.cameras, now=end),
            "uptime": human_duration(monotonic() - started_at),
        }

    @app.get("/")
    async def dashboard(request: Request, window: BoardWindow = DEFAULT_WINDOW) -> Any:
        """The board is rendered inline, not fetched.

        A page that is blank until the first poll lands looks broken for exactly as long
        as the poll interval, which is the first thing a new self-hoster would see.

        `window` is accepted here as well as on the fragment so the range controls are
        real links: with scripting off they reload the page at the chosen range instead
        of doing nothing.
        """
        return templates.TemplateResponse(
            request=request, name="dashboard.html", context=_board_context(window)
        )

    @app.get("/fragments/board")
    async def board(request: Request, window: BoardWindow = DEFAULT_WINDOW) -> Any:
        return templates.TemplateResponse(
            request=request, name="_board.html", context=_board_context(window)
        )

    app.include_router(
        calibration_router(
            current=current,
            adopt=_adopt,
            templates=templates,
            snapshot=snapshot,
            save_geometry=save_geometry,
        )
    )

    if render_prometheus is not None:

        @app.get("/metrics", response_class=PlainTextResponse)
        async def prometheus() -> PlainTextResponse:
            return PlainTextResponse(
                content=render_prometheus(), media_type=PROMETHEUS_CONTENT_TYPE
            )

    return app


def _health(
    config: MusterConfig,
    store: Store,
    *,
    reports: Mapping[CameraId, WorkerReport],
    uptime_s: float,
) -> EngineHealth:
    return engine_health(
        camera_health(config, reports),
        uptime_s=uptime_s,
        schema_version=store.schema_version(),
        sync=SyncHealth(enabled=config.cloud_sync.enabled, unsynced_rows=store.unsynced_count()),
        disk=DiskHealth.of(store.path),
    )


__all__ = ["DEFAULT_LIMIT", "MAX_LIMIT", "MAX_WINDOW", "create_app"]
