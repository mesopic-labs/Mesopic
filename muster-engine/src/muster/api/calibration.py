"""The calibration surface: the geometry list views and the drawing canvas.

Split out of `app.py` rather than living beside the dashboard routes, because this is
where the engine stops being read-only. Everything here either shows a frame or changes
the site's configuration, and both deserve to be read in one place.

Three rules hold across every route below:

* **The camera comes from the path.** `_camera` resolves it first, so an id naming
  nothing is refused before it can reach the supervisor or the writer.
* **The snapshot is `no-store`.** §13 allows one frame out of the process; a cached
  response would put it on a disk, which is the thing the invariant forbids.
* **A refusal says nothing about the camera.** An RTSP URL carries credentials, so the
  detail goes to the log and the browser gets a generic body (the P2.1 rule).

Implements P3.3. P3.10 put the save behind a session (ADR-0019).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi import Path as PathParam

from muster.api.auth import WriteGuard
from muster.api.calibrate import GeometryEdit, merge_geometry
from muster.api.rows import GeometryRow, count_for
from muster.config.schema import CameraConfig, FrigateSource, MusterConfig
from muster.errors import ConfigError, SnapshotUnavailableError
from muster.types import CameraId

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

    from muster.api.app import GeometrySaver, Snapshotter

logger = logging.getLogger(__name__)

MAX_ID_LENGTH = 128
"""Long enough for any id a config can name, short enough that nothing large reaches a
log line or a template."""


def _same_origin(request: Request) -> bool:
    """A cheap CSRF floor for the engine's only state-changing endpoint.

    A browser attaches `Origin` to every cross-origin write, so a mismatch is a request
    the operator did not make from the page they are looking at. A missing header is not
    a browser and therefore not a CSRF vector — `curl` and the tests land there.

    This is a floor, not authentication, and since P3.10 it no longer has to be: the
    session guard is what establishes the caller is the operator. ADR-0018 item 8 said
    this check should be re-described rather than deleted once that landed, because the
    two fail independently — a stolen session still cannot be spent from another host's
    page, and a same-origin request still cannot be made without one.
    """
    origin = request.headers.get("origin")
    if origin is None:
        return True
    return urlsplit(origin).netloc == request.headers.get("host")


def calibration_router(
    *,
    current: Callable[[], MusterConfig],
    adopt: Callable[[MusterConfig], None],
    templates: Jinja2Templates,
    snapshot: Snapshotter | None,
    save_geometry: GeometrySaver | None,
    guard: WriteGuard,
) -> APIRouter:
    """The calibration routes, bound to one app's config cell.

    `current` and `adopt` are the two halves of that cell: the app owns the live config
    and this router reads it and replaces it, rather than holding a copy that would go
    stale the moment a save landed.
    """
    router = APIRouter()

    def _camera(camera_id: str) -> CameraConfig:
        """Resolve the path's camera, or 404.

        Every calibration route starts here, so an id that names nothing is refused
        before it can reach the supervisor or the writer.
        """
        found = next(
            (camera for camera in current().cameras if camera.camera_id == camera_id), None
        )
        if found is None:
            raise HTTPException(status_code=404, detail="no such camera")
        return found

    def _geometry_page(request: Request, heading: str, rows: list[GeometryRow]) -> Any:
        return templates.TemplateResponse(
            request=request,
            name="geometry.html",
            context={"site_id": current().site.site_id, "heading": heading, "rows": rows},
        )

    @router.get("/cameras")
    async def cameras(request: Request) -> Any:
        site = current()
        return _geometry_page(
            request,
            "Cameras",
            [
                GeometryRow(
                    camera_id=camera.camera_id,
                    label=camera.name,
                    detail=camera.source.kind.value,
                    extent=f"{count_for(site, camera.camera_id)} shapes",
                )
                for camera in site.cameras
            ],
        )

    @router.get("/zones")
    async def zones(request: Request) -> Any:
        return _geometry_page(
            request,
            "Zones",
            [
                GeometryRow(
                    camera_id=zone.camera_id,
                    label=zone.zone_id,
                    detail=f"{zone.camera_id} · {zone.role.value}",
                    extent=f"{len(zone.polygon)} pts",
                )
                for zone in current().zones
            ],
        )

    @router.get("/lines")
    async def lines(request: Request) -> Any:
        return _geometry_page(
            request,
            "Lines",
            [
                GeometryRow(
                    camera_id=line.camera_id,
                    label=line.line_id,
                    detail=f"{line.camera_id} · {line.positive_dir.value}",
                    extent="2 pts",
                )
                for line in current().lines
            ],
        )

    @router.get("/api/cameras/{camera_id}/snapshot")
    async def camera_snapshot(
        camera_id: Annotated[str, PathParam(max_length=MAX_ID_LENGTH)],
    ) -> Response:
        """The one frame §13 lets out of the process.

        `no-store` is part of the promise, not a nicety: a cached response is a frame on
        the viewer's disk, which is the thing the invariant forbids however it got there.
        """
        camera = _camera(camera_id)
        if isinstance(camera.source, FrigateSource):
            # Refused here rather than sent to a worker that could never answer it. A
            # Frigate camera has tracks and no frames, so the request would hang until it
            # timed out and the editor would show a spinner where a reason belongs.
            raise HTTPException(
                status_code=422,
                detail="this camera's video is handled by Frigate, so the engine has no "
                "frame to draw on — set its zones and lines in muster.yaml",
            )
        if snapshot is None:
            raise HTTPException(status_code=503, detail="the engine is not running")
        try:
            jpeg = await snapshot(CameraId(camera_id))
        except SnapshotUnavailableError:
            # Deliberately without the exception: its message is engine-authored today,
            # but this is the one log line in reach of a camera's own error text, and an
            # RTSP URL carries the camera's credentials.
            logger.info("no snapshot available for camera %s", camera_id)
            raise HTTPException(status_code=503, detail="no snapshot available") from None
        return Response(
            content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
        )

    @router.get("/calibrate/{camera_id}")
    async def calibrate(
        request: Request, camera_id: Annotated[str, PathParam(max_length=MAX_ID_LENGTH)]
    ) -> Any:
        camera = _camera(camera_id)
        site = current()
        return templates.TemplateResponse(
            request=request,
            name="calibrate.html",
            context={
                "site_id": site.site.site_id,
                "camera": camera,
                "zones": [zone for zone in site.zones if zone.camera_id == camera.camera_id],
                "lines": [line for line in site.lines if line.camera_id == camera.camera_id],
                "can_save": save_geometry is not None and guard.configured,
            },
        )

    @router.post("/calibrate/{camera_id}", status_code=204, dependencies=[Depends(guard)])
    async def save_calibration(
        request: Request,
        camera_id: Annotated[str, PathParam(max_length=MAX_ID_LENGTH)],
        edit: GeometryEdit,
    ) -> Response:
        """Merge the edit into the site's geometry, write it, and hot-reload.

        Two independent guards stand in front of this, and neither subsumes the other.
        The session (P3.10, ADR-0019) establishes that the caller is the operator at all;
        the origin check establishes that the request came from the page they are looking
        at rather than from another host's page driving their browser. ADR-0018 item 8
        promised the second would be re-described rather than deleted once the first
        existed, which is what this is.
        """
        camera = _camera(camera_id)
        if not _same_origin(request):
            raise HTTPException(status_code=403, detail="cross-origin request refused")
        if save_geometry is None:
            raise HTTPException(status_code=503, detail="the engine is not running")

        zones, lines = merge_geometry(current(), camera.camera_id, edit)
        try:
            adopt(await save_geometry(zones, lines))
        except ConfigError:
            # The message names the file and the failing key path, both of which belong
            # in the log rather than in a browser.
            logger.warning("refused a geometry save for camera %s", camera_id)
            raise HTTPException(status_code=400, detail="invalid geometry") from None
        return Response(status_code=204)

    return router


__all__ = ["calibration_router"]
