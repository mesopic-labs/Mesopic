"""The row shape the geometry list views render.

`/cameras`, `/zones` and `/lines` differ only in what they put in each column, so they
share one row and one template rather than three near-identical ones that drift apart.

Implements P3.3.
"""

from __future__ import annotations

from dataclasses import dataclass

from muster.config.schema import MusterConfig
from muster.types import CameraId


@dataclass(frozen=True, slots=True)
class GeometryRow:
    """One line of a geometry table, already rendered to strings."""

    camera_id: str
    label: str
    detail: str
    extent: str


def count_for(config: MusterConfig, camera_id: CameraId) -> int:
    """How many shapes a camera carries — the only number `/cameras` needs.

    Counted per collection rather than over one chained sequence: zones and lines have no
    common base carrying `camera_id`, so chaining them widens the element type to the
    section base and the attribute stops type-checking.
    """
    zones = sum(1 for zone in config.zones if zone.camera_id == camera_id)
    lines = sum(1 for line in config.lines if line.camera_id == camera_id)
    return zones + lines


__all__ = ["GeometryRow", "count_for"]
