"""The core six and the two adjacencies, each as a registered plugin.

Every metric is a `MetricPlugin` + reducer, not special-cased engine code — the plugin
seam (engine-architecture.md §17) is proved by building the first six against it, so the
seventh is a plugin rather than a refactor.

Scope is locked: footfall, occupancy, queue length, dwell, line crossings, conversion,
plus zone heatmaps and staff-vs-customer. Nothing else exists in v1.
"""

from __future__ import annotations

from muster.analytics.metrics.registry import MetricPlugin, MetricRegistry

__all__ = ["MetricPlugin", "MetricRegistry"]
