"""The local web API and HUD dashboard (engine-architecture.md §13).

The whole free-tier UI, with **no dependency on the cloud**. Built on the same stack the
cloud dashboard uses — FastAPI + Jinja + HTMX + uPlot over the HUD design system — so a
tile built once is reused on both sides (ADR-0007).

Handlers are thin: query the store, render a fragment. They compute nothing.
"""

from __future__ import annotations

from muster.api.app import create_app

__all__ = ["create_app"]
