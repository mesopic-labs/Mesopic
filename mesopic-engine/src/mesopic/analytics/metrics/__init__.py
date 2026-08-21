"""The core six and the two adjacencies, each as a registered plugin.

Every metric is a `MetricPlugin` + reducer, not special-cased engine code — the plugin
seam (engine-architecture.md §17) is proved by building the first six against it, so the
seventh is a plugin rather than a refactor.

Scope is locked: footfall, occupancy, queue length, dwell, line crossings, conversion,
plus zone heatmaps and staff-vs-customer. Nothing else exists in v1.
"""

from __future__ import annotations

from mesopic.analytics.metrics.conversion import ConversionPlugin, TransactionCounts
from mesopic.analytics.metrics.dwell import DwellPlugin
from mesopic.analytics.metrics.footfall import FootfallPlugin
from mesopic.analytics.metrics.line_cross import LineCrossPlugin
from mesopic.analytics.metrics.occupancy import OccupancyPlugin, QueueLengthPlugin
from mesopic.analytics.metrics.registry import MetricPlugin, MetricRegistry
from mesopic.analytics.site_geometry import SiteGeometry

__all__ = [
    "ConversionPlugin",
    "DwellPlugin",
    "FootfallPlugin",
    "LineCrossPlugin",
    "MetricPlugin",
    "MetricRegistry",
    "OccupancyPlugin",
    "QueueLengthPlugin",
    "build_registry",
]


def build_registry(
    geometry: SiteGeometry, *, transactions: TransactionCounts | None = None
) -> MetricRegistry:
    """Register the core six against one site's geometry.

    Which *rows* each plugin produces is still decided per zone and per line by the
    `metrics` list in config — registering a plugin enables the metric for the site,
    drawing the geometry is what asks for it somewhere.

    `transactions` is the till feed conversion divides into. Absent, conversion is built
    but never produces a row, which is exactly algorithms.md §9's rule for a site with no
    POS wired up: not an error, and footfall is unaffected. P3.5 replaces the mapping
    with the signed webhook ingress.

    Heatmap and the staff split are **absent rather than stubbed** (P4.1, P4.2). A
    registered plugin that emits nothing would read as a working metric reporting no
    activity, which is the kind of confident silence this engine tries not to produce.
    """
    footfall = FootfallPlugin(geometry)
    registry = MetricRegistry()
    for plugin in (
        footfall,
        LineCrossPlugin(geometry),
        OccupancyPlugin(geometry),
        QueueLengthPlugin(geometry),
        DwellPlugin(geometry),
        ConversionPlugin(footfall, {} if transactions is None else transactions),
    ):
        registry.register(plugin)
    return registry
