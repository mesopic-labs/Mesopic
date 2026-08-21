"""`/metrics` text exposition — business metrics *and* engine health.

Exporting health is deliberate: the self-hoster's existing Grafana becomes the free
"is my box healthy" view (per-camera fps, dropped events, camera state), and it does so
without us ever seeing a frame or phoning home.

This class **renders** the exposition and does not serve it. The local API owns the HTTP
route (P3.1); putting a second web server in the box here would give the operator two
ports and two lifecycles, and P3.1 would then have to displace one of them.

Implements part of P3.5.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Gauge, generate_latest

from mesopic.types import MetricRow

METRIC_NAME = "mesopic_metric"
HEALTH_PREFIX = "mesopic_"


class PrometheusExporter:
    """Maintains the gauges and counters scraped from the local API's `/metrics`."""

    def __init__(self) -> None:
        # Its OWN registry, never `prometheus_client`'s process-global default. The
        # default makes series leak between instances -- across a config reload, and
        # between tests, where it also makes duplicate registration an error rather than
        # a fresh start.
        self._registry = CollectorRegistry()
        self._metric = Gauge(
            METRIC_NAME,
            "Latest value per camera, metric and scope.",
            ["camera", "metric", "scope"],
            registry=self._registry,
        )
        self._health: dict[str, Gauge] = {}

    def start(self) -> None:
        """Nothing to open: a pull exporter has no peer to connect to."""

    def on_metric(self, row: MetricRow) -> None:
        """Set, never increment. A scrape reports the last known value of a gauge."""
        self._metric.labels(
            camera=row.camera_id,
            metric=row.metric.value,
            scope="" if row.scope_id is None else row.scope_id,
        ).set(row.value)

    def shutdown(self) -> None:
        """Nothing to flush."""

    def set_health(self, values: dict[str, float]) -> None:
        """Publish engine health alongside the business metrics (§12).

        Gauges are created on first sight rather than declared up front, so the
        supervisor can add a counter without this file needing to know its name.
        """
        for name, value in values.items():
            gauge = self._health.get(name)
            if gauge is None:
                gauge = Gauge(
                    f"{HEALTH_PREFIX}{name}", f"Engine health: {name}.", registry=self._registry
                )
                self._health[name] = gauge
            gauge.set(value)

    def render(self) -> str:
        """The text exposition the local API serves at `/metrics`."""
        return generate_latest(self._registry).decode()
