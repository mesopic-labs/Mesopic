"""FastAPI app factory for the local dashboard.

Routes (engine-architecture.md §13):

| Route            | Purpose                                                          |
|------------------|------------------------------------------------------------------|
| `/`              | Live occupancy tiles, core-six charts, heatmap, staff/customer    |
| `/cameras` `/zones` `/lines` | View and draw the site geometry                      |
| `/config`        | Render, validate, and hot-reload `muster.yaml`                    |
| `/api/metrics`   | Read-only time-series JSON                                        |
| `/healthz`       | Machine-readable engine and per-camera health                     |
| `/metrics`       | Prometheus exposition                                             |

**The calibration view is the one place a frame is briefly shown.** A single snapshot is
grabbed on demand, streamed to the browser to draw zones and lines on, and never written
to disk or uploaded. Geometry saves as normalized coordinates, so the snapshot is
disposable — which is what keeps the "frames discarded immediately" guarantee intact even
during setup.

Implements P3.1 onward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

    from muster.store import Store


def create_app(store: Store) -> FastAPI:
    """Build the local dashboard app around an already-open store."""
    raise NotImplementedError
