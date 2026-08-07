"""One topic per camera and metric — the day-one Frigate and Home Assistant bus.

Messages are published **retained**, so Home Assistant sees last-known values the moment
it connects and can automate on them without any bespoke integration. MQTT discovery
config is published so HA auto-creates the sensors.

Implements part of P3.5 and P4.4.
"""

from __future__ import annotations

from muster.types import MetricRow


class MqttExporter:
    """Publishes retained metric topics to a broker."""

    def __init__(self, broker: str, base_topic: str = "muster", port: int = 1883) -> None:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def on_metric(self, row: MetricRow) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError
