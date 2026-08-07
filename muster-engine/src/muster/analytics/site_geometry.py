"""The compiled, per-camera form of the config's lines and zones.

Built once from config and rebuilt only when config changes — never per frame. Polygons
are bounding-boxed so the common "not even close" case is a cheap rejection before any
point-in-polygon work.

Implements P2.2.
"""

from __future__ import annotations

from muster.config.schema import MusterConfig
from muster.types import CameraId, NormPoint


class SiteGeometry:
    """Prepared geometry for one site, queryable per camera."""

    @classmethod
    def compile(cls, config: MusterConfig) -> SiteGeometry:
        """Build prepared geometry from validated config."""
        raise NotImplementedError

    def zones_containing(self, camera_id: CameraId, point: NormPoint) -> list[str]:
        """Zone ids whose polygon contains `point`."""
        raise NotImplementedError
