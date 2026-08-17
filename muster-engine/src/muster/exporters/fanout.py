"""One place that hands committed buckets to every enabled exporter.

The fan-out exists to make §12's independence rule true rather than aspirational: a dead
MQTT broker must not stall the Prometheus scrape, the CSV rotation, or the pipeline. So
every call into an exporter is contained here, and a peer's outage becomes a counter
rather than an exception travelling up into the supervisor's tick.

`Exception` is caught deliberately and broadly. An exporter talks to something outside
the box — a broker, an HTTP endpoint, a filesystem — and the set of ways that fails is
open. Narrowing the catch to the errors we thought of is how the first unanticipated one
takes the engine down with it. `BaseException` is *not* caught: a `KeyboardInterrupt` or
`SystemExit` is the operator, not a broken peer.

Implements P3.5 (engine-architecture.md §12).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

from muster.config.schema import MusterConfig
from muster.errors import ConfigError
from muster.exporters.csv_export import CsvExporter
from muster.exporters.exporter import Exporter
from muster.exporters.mqtt import MqttExporter
from muster.exporters.prometheus import PrometheusExporter
from muster.exporters.webhook import WebhookExporter
from muster.types import MetricRow

_log = logging.getLogger(__name__)


class ExporterFanout:
    """Holds the enabled exporters and isolates each one's failures."""

    def __init__(self, exporters: dict[str, Exporter]) -> None:
        self._exporters = exporters
        self.failures: dict[str, int] = {}
        """Failed calls per exporter name. A degraded exporter is a number, not silence."""

    def names(self) -> tuple[str, ...]:
        return tuple(self._exporters)

    def get(self, name: str) -> Exporter | None:
        """The exporter config enabled under `name`, if any.

        For the one exporter that is also a *surface*: Prometheus is scraped from the
        local API's `/metrics`, so the route has to render the very instance the fan-out
        feeds. A second one would own a second `CollectorRegistry` and scrape clean
        forever while the real counters climbed out of sight.
        """
        return self._exporters.get(name)

    def healthy_names(self) -> tuple[str, ...]:
        """Exporters that have not failed. What `/healthz` will report (P3.1)."""
        return tuple(name for name in self._exporters if name not in self.failures)

    def start(self) -> None:
        for name, exporter in self._exporters.items():
            self._guard(name, exporter.start)

    def on_metrics(self, rows: Sequence[MetricRow]) -> None:
        """Deliver committed rows. Called by the supervisor *after* the store accepts them."""
        for name, exporter in self._exporters.items():
            for row in rows:
                self._guard(name, exporter.on_metric, row)

    def shutdown(self) -> None:
        for name, exporter in self._exporters.items():
            self._guard(name, exporter.shutdown)

    def _guard(self, name: str, call: object, *args: object) -> None:
        assert callable(call)  # noqa: S101 - narrowing a Protocol member for mypy
        try:
            call(*args)
        except Exception:  # noqa: BLE001 - see the module docstring; the failure set is open
            self.failures[name] = self.failures.get(name, 0) + 1
            # The exporter's name and nothing else: a webhook's exception message can
            # carry its URL, and a URL is a secret here (ADR-0005).
            _log.warning("exporter %s failed; continuing", name)


def build_exporters(config: MusterConfig) -> ExporterFanout:
    """Construct the exporters config enables, and only those.

    Refuses rather than degrades when an enabled exporter cannot be configured: a webhook
    that posts unsigned because its secret was missing is worse than one that never
    starts, because nothing downstream can tell the difference until it matters.
    """
    exporters: dict[str, Exporter] = {}
    settings = config.exporters

    if settings.csv.enabled:
        if settings.csv.dir is None:  # pragma: no cover - the schema requires it
            msg = "csv exporter is enabled but has no dir"
            raise ConfigError(msg)
        exporters["csv"] = CsvExporter(Path(settings.csv.dir))

    if settings.prometheus.enabled:
        exporters["prometheus"] = PrometheusExporter()

    if settings.webhook.enabled:
        if settings.webhook.url is None or settings.webhook.secret_env is None:
            msg = "webhook exporter is enabled but has no url or secret_env"
            raise ConfigError(msg)
        exporters["webhook"] = WebhookExporter(
            settings.webhook.url, _secret(settings.webhook.secret_env)
        )

    if settings.mqtt.enabled:
        if settings.mqtt.broker is None:  # pragma: no cover - the schema requires it
            msg = "mqtt exporter is enabled but has no broker"
            raise ConfigError(msg)
        exporters["mqtt"] = MqttExporter(
            settings.mqtt.broker, base_topic=settings.mqtt.base_topic, port=settings.mqtt.port
        )

    return ExporterFanout(exporters)


def _secret(env_var: str) -> str:
    secret = os.environ.get(env_var)
    if not secret:
        # Names the variable, never its value.
        msg = f"${env_var} is unset or empty, and the webhook exporter needs it to sign"
        raise ConfigError(msg)
    return secret
