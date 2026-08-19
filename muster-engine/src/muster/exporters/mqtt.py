"""One topic per camera and metric — the day-one Frigate and Home Assistant bus.

Messages are published **retained**, so Home Assistant sees last-known values the moment
it connects and can automate on them without any bespoke integration.

Since P4.4 the exporter also announces itself. Home Assistant creates an entity when it
sees a retained config on `<prefix>/sensor/<id>/config`, so a sensor is announced the
first time a metric for that scope arrives and never again — the rows are the truth about
what this site measures, and announcing from them cannot disagree with what is published.
The cost is that an entity appears after the first bucket closes rather than at start-up;
the alternative was enumerating the config here, which is a second place to be wrong about
which scopes produce which metrics.

**Availability is the other half, and the less obvious one.** Every value is retained, so
a broker holds the last reading forever — including the reading from an engine that died
an hour ago. Without an availability topic Home Assistant shows that number as current,
which is the same stale-reads-as-live failure the dashboard's tiles and heatmaps were both
built to avoid. The will is registered *before* connecting, because a will set afterwards
is one the broker never received, and the case it exists for is exactly the ungraceful
death that never gets to say goodbye.

Implements part of P3.5 and P4.4 (ADR-0006 — integrate, do not fork).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol

from paho.mqtt.client import Client
from paho.mqtt.enums import CallbackAPIVersion

from muster.types import CameraId, MetricName, MetricRow, SiteId

if TYPE_CHECKING:
    from collections.abc import Mapping

KEEPALIVE_S = 60

DISCOVERY_PREFIX = "homeassistant"
"""Where Home Assistant listens for discovery configs, and its own default."""

ONLINE = "online"
OFFLINE = "offline"

UNNAMED_SITE = SiteId("muster")
"""What an exporter built without a site announces under.

`build_exporters` always passes the real one, so this is reached only by a caller
constructing the exporter directly. It is a constant rather than a literal in the
signature so that the value two boxes would collide on has a name and a reason next to
it."""

PEOPLE = "people"
"""Not an HA unit with a device class behind it — there isn't one for a person count.

Stated anyway: an axis labelled `people` is readable, and a bare number in a history graph
beside a temperature in °C is not."""

_UNITS: dict[MetricName, tuple[str, str | None]] = {
    MetricName.FOOTFALL: (PEOPLE, None),
    MetricName.LINE_CROSS: (PEOPLE, None),
    MetricName.OCCUPANCY: (PEOPLE, None),
    MetricName.OCCUPANCY_RAW: (PEOPLE, None),
    MetricName.QUEUE_LEN: (PEOPLE, None),
    MetricName.QUEUE_LEN_RAW: (PEOPLE, None),
    MetricName.DWELL_SECONDS: ("s", "duration"),
    MetricName.CONVERSION: ("%", None),
}
"""Unit and HA device class per metric. `HEATMAP` is deliberately absent — a packed grid
is not a number, so it gets no entity rather than an entity holding nonsense."""


class MqttClient(Protocol):
    """The part of `paho`'s client this exporter uses, so a test can stand in for it."""

    def will_set(self, topic: str, payload: str, *, retain: bool = False) -> object: ...
    def connect(self, host: str, port: int, keepalive: int) -> object: ...
    def loop_start(self) -> object: ...
    def publish(self, topic: str, payload: str, *, retain: bool = False) -> object: ...
    def disconnect(self) -> object: ...
    def loop_stop(self) -> object: ...


class MqttExporter:
    """Publishes retained metric topics to a broker, and announces them to Home Assistant.

    **Known limitation:** a renamed or deleted zone leaves its retained discovery config on
    the broker, so Home Assistant keeps an entity for a scope that no longer exists.
    Clearing it means publishing an empty payload to the *old* topic, which needs a record
    of what was announced before this process started — state the engine deliberately does
    not keep. Removing the stale entity is a one-line `mosquitto_pub` an operator can run,
    and pretending otherwise would be worse than saying so.
    """

    def __init__(
        self,
        broker: str,
        base_topic: str = "muster",
        port: int = 1883,
        *,
        client: MqttClient | None = None,
        site_id: SiteId = UNNAMED_SITE,
        camera_names: Mapping[CameraId, str] | None = None,
        discovery: bool = True,
        discovery_prefix: str = DISCOVERY_PREFIX,
    ) -> None:
        self._broker = broker
        self._base_topic = base_topic.rstrip("/")
        self._port = port
        default = Client(CallbackAPIVersion.VERSION2)
        self._client: MqttClient = client if client is not None else default
        self._site_id = site_id
        self._camera_names = dict(camera_names or {})
        self._discovery = discovery
        self._discovery_prefix = discovery_prefix.rstrip("/")
        self._announced: set[str] = set()
        """Which sensors have been announced, so a bucket a minute is not a retained
        discovery config a minute that the broker then keeps."""

    @property
    def status_topic(self) -> str:
        """Where this engine says whether it is running. Also HA's availability topic."""
        return f"{self._base_topic}/status"

    def start(self) -> None:
        """Register the will, connect, and hand the socket to paho's own loop thread.

        `loop_start` is what gives this exporter its reconnect behaviour for free: paho
        retries in the background, so a broker that comes back does not need the engine
        restarted, and a broker that is down costs a failed publish rather than a stall.
        """
        self._client.will_set(self.status_topic, OFFLINE, retain=True)
        self._client.connect(self._broker, self._port, KEEPALIVE_S)
        self._client.loop_start()
        self._client.publish(self.status_topic, ONLINE, retain=True)

    def on_metric(self, row: MetricRow) -> None:
        self._announce(row)
        self._client.publish(self._topic(row), str(row.value), retain=True)

    def shutdown(self) -> None:
        """Say goodbye before closing, so a planned stop is not read as a crash."""
        self._client.publish(self.status_topic, OFFLINE, retain=True)
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

    def _announce(self, row: MetricRow) -> None:
        """Publish this scope's discovery config, once, the first time it is seen."""
        if not self._discovery or row.metric not in _UNITS:
            return
        unique_id = self._unique_id(row)
        if unique_id in self._announced:
            return
        self._announced.add(unique_id)
        self._client.publish(
            f"{self._discovery_prefix}/sensor/{unique_id}/config",
            json.dumps(self._config(row, unique_id)),
            retain=True,
        )

    def _unique_id(self, row: MetricRow) -> str:
        parts = ["muster", self._site_id, row.camera_id, row.metric.value]
        if row.scope_id is not None:
            parts.append(row.scope_id)
        return "_".join(parts)

    def _config(self, row: MetricRow, unique_id: str) -> dict[str, object]:
        """One entity, described the way Home Assistant's discovery schema expects.

        `state_topic` is built by `_topic`, the same call `on_metric` publishes with,
        rather than by formatting the same string twice. The two drifting apart is the
        failure with no symptom: HA creates the entity, subscribes to a topic nothing
        publishes to, and shows `unknown` forever without either side erroring.
        """
        unit, device_class = _UNITS[row.metric]
        config: dict[str, object] = {
            "name": self._entity_name(row),
            "unique_id": unique_id,
            "object_id": unique_id,
            "state_topic": self._topic(row),
            "availability_topic": self.status_topic,
            "payload_available": ONLINE,
            "payload_not_available": OFFLINE,
            # Every row is one minute's value, not a running total: `measurement` is what
            # makes HA's statistics treat it as a level to average rather than a counter
            # to difference.
            "state_class": "measurement",
            "unit_of_measurement": unit,
            "device": {
                "identifiers": [f"muster_{self._site_id}_{row.camera_id}"],
                "name": self._camera_names.get(row.camera_id, row.camera_id),
                "manufacturer": "Muster",
                "model": "Camera",
            },
        }
        if device_class is not None:
            config["device_class"] = device_class
        return config

    def _entity_name(self, row: MetricRow) -> str:
        """`Footfall door-count` — the metric first, because the device is the camera.

        HA prefixes the device name itself, so repeating the camera here would render as
        "Front door Front door footfall" in every entity list.
        """
        readable = row.metric.value.replace("_", " ").capitalize()
        return f"{readable} {row.scope_id}" if row.scope_id is not None else readable


__all__ = ["DISCOVERY_PREFIX", "MqttExporter"]
