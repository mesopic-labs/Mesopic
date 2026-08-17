"""One topic per camera and metric — the day-one Frigate and Home Assistant bus.

Messages are published **retained**, so Home Assistant sees last-known values the moment
it connects and can automate on them without any bespoke integration. MQTT discovery
config is published so HA auto-creates the sensors.

Implements part of P3.5 and P4.4.
"""

from __future__ import annotations

from typing import Protocol

from paho.mqtt.client import Client
from paho.mqtt.enums import CallbackAPIVersion

from muster.types import MetricRow

KEEPALIVE_S = 60


class MqttClient(Protocol):
    """The part of `paho`'s client this exporter uses, so a test can stand in for it."""

    def connect(self, host: str, port: int, keepalive: int) -> object: ...
    def loop_start(self) -> object: ...
    def publish(self, topic: str, payload: str, *, retain: bool = False) -> object: ...
    def disconnect(self) -> object: ...
    def loop_stop(self) -> object: ...


class MqttExporter:
    """Publishes retained metric topics to a broker."""

    def __init__(
        self,
        broker: str,
        base_topic: str = "muster",
        port: int = 1883,
        *,
        client: MqttClient | None = None,
    ) -> None:
        self._broker = broker
        self._base_topic = base_topic.rstrip("/")
        self._port = port
        default = Client(CallbackAPIVersion.VERSION2)
        self._client: MqttClient = client if client is not None else default

    def start(self) -> None:
        """Connect and hand the socket to paho's own loop thread.

        `loop_start` is what gives this exporter its reconnect behaviour for free: paho
        retries in the background, so a broker that comes back does not need the engine
        restarted, and a broker that is down costs a failed publish rather than a stall.
        """
        self._client.connect(self._broker, self._port, KEEPALIVE_S)
        self._client.loop_start()

    def on_metric(self, row: MetricRow) -> None:
        self._client.publish(self._topic(row), str(row.value), retain=True)

    def shutdown(self) -> None:
        self._client.disconnect()
        self._client.loop_stop()

    def _topic(self, row: MetricRow) -> str:
        """`base/camera/metric[/scope]`, with no empty segment for a camera-wide metric.

        A trailing empty segment (`muster/front-door/occupancy/`) is a topic nobody can
        subscribe to sensibly and that Home Assistant renders as a blank entity name.
        """
        parts = [self._base_topic, row.camera_id, row.metric.value]
        if row.scope_id is not None:
            parts.append(row.scope_id)
        return "/".join(parts)
