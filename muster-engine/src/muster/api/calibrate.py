"""The calibration canvas's request body, and the rule for folding it into a config.

The browser sends geometry it drew over a snapshot; this module turns that into the same
`ZoneConfig`/`LineConfig` objects a hand-written `muster.yaml` produces, so everything
downstream — validation, compilation, the writer — sees one shape regardless of origin.

The camera is deliberately **not** a field of the body. It comes from the path, so a
request cannot address a camera other than the one whose URL the operator is looking at,
and there is no second copy of it to disagree with the first.

Implements P3.3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field

from muster.config.schema import LineConfig, NormalizedPoint, ZoneConfig
from muster.types import CameraId, Direction, LineId, MetricName, ZoneId, ZoneRole

if TYPE_CHECKING:
    from muster.config.schema import MusterConfig

MAX_POLYGON_VERTICES = 64
"""A hand-drawn zone is a handful of points. The ceiling exists because every vertex is
a segment the ray cast walks per track per frame (§6), and because an unbounded polygon
is an unbounded line in a config file."""

MAX_SHAPES_PER_CAMERA = 32
"""Zones or lines on one camera. Well past what a real site draws, and short of what
would make a config unreadable or a geometry compile slow."""

MAX_ID_LENGTH = 64
"""Ids become YAML keys, store rows and metric scopes. Long enough to be descriptive."""

IdText = Annotated[str, Field(min_length=1, max_length=MAX_ID_LENGTH, pattern=r"^[A-Za-z0-9_-]+$")]
"""Ids are drawn by a human into a text box and end up in three other systems. The
charset is the intersection of what YAML writes unquoted, what reads back as itself, and
what a URL can carry without escaping."""


class CanvasShape(BaseModel):
    """Base for what the canvas sends: closed to unknown keys, like every config section."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metrics: Annotated[list[MetricName], Field(max_length=MAX_SHAPES_PER_CAMERA)] = Field(
        default_factory=list
    )


class ZoneEdit(CanvasShape):
    zone_id: IdText
    role: ZoneRole = ZoneRole.AREA
    polygon: Annotated[list[NormalizedPoint], Field(min_length=3, max_length=MAX_POLYGON_VERTICES)]

    def to_config(self, camera_id: CameraId) -> ZoneConfig:
        return ZoneConfig(
            zone_id=ZoneId(self.zone_id),
            camera_id=camera_id,
            role=self.role,
            polygon=self.polygon,
            metrics=self.metrics,
        )


class LineEdit(CanvasShape):
    line_id: IdText
    a: NormalizedPoint
    b: NormalizedPoint
    positive_dir: Direction = Direction.IN

    def to_config(self, camera_id: CameraId) -> LineConfig:
        return LineConfig(
            line_id=LineId(self.line_id),
            camera_id=camera_id,
            a=self.a,
            b=self.b,
            positive_dir=self.positive_dir,
            metrics=self.metrics,
        )


class GeometryEdit(BaseModel):
    """One camera's geometry, as drawn. Empty lists are a legitimate save."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    zones: Annotated[list[ZoneEdit], Field(max_length=MAX_SHAPES_PER_CAMERA)] = Field(
        default_factory=list
    )
    lines: Annotated[list[LineEdit], Field(max_length=MAX_SHAPES_PER_CAMERA)] = Field(
        default_factory=list
    )


def merge_geometry(
    config: MusterConfig, camera_id: CameraId, edit: GeometryEdit
) -> tuple[list[ZoneConfig], list[LineConfig]]:
    """This camera's geometry replaced by the edit, every other camera's carried through.

    The writer replaces whole `zones:`/`lines:` blocks, so it has to be handed the site's
    complete geometry. Building that here — rather than in the route — keeps the rule
    that a two-camera site does not lose a camera when the other one is calibrated in a
    place where it can be tested without a browser.
    """
    zones = [zone for zone in config.zones if zone.camera_id != camera_id]
    lines = [line for line in config.lines if line.camera_id != camera_id]
    zones.extend(zone.to_config(camera_id) for zone in edit.zones)
    lines.extend(line.to_config(camera_id) for line in edit.lines)
    return zones, lines


__all__ = [
    "MAX_ID_LENGTH",
    "MAX_POLYGON_VERTICES",
    "MAX_SHAPES_PER_CAMERA",
    "GeometryEdit",
    "LineEdit",
    "ZoneEdit",
    "merge_geometry",
]
