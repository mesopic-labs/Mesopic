"""The pure `(tracks, geometry) -> events` function.

Pure by design: no camera, no model, no database in the loop, so it is testable against
recorded `Track` sequences alone. Crossing debounce (a foot-point jittering across a
line) is handled per algorithms.md, not invented here.

Implements P2.3.
"""

from __future__ import annotations

from muster.analytics.site_geometry import SiteGeometry
from muster.types import RawEvent, Track


def events_from_tracks(tracks: list[Track], geometry: SiteGeometry) -> list[RawEvent]:
    """Derive raw events from one tick's tracks. Pure — no I/O, no state on disk."""
    raise NotImplementedError
