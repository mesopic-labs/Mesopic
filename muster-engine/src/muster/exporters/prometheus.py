"""`/metrics` text exposition — business metrics *and* engine health.

Exporting health is deliberate: the self-hoster's existing Grafana becomes the free
"is my box healthy" view (per-camera fps, dropped events, camera state), and it does so
without us ever seeing a frame or phoning home.

Implements part of P3.5.
"""

from __future__ import annotations

from muster.types import MetricRow


class PrometheusExporter:
    """Maintains the gauges and counters scraped from the local API's `/metrics`."""

    def start(self) -> None:
        raise NotImplementedError

    def on_metric(self, row: MetricRow) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError
