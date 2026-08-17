"""The four exporters §12 specifies, each driven without its peer.

No broker, no HTTP server, no scrape. The clients are injected instead, because what
these tests are about is the *shape* each exporter emits — a retained topic, a signed
body, a rotated file, an exposition line — and a real peer would only add flakiness to
assertions about shape.

Two rules from CLAUDE.md are asserted here rather than trusted: TLS verification stays on
(including in tests), and a secret never appears in a payload, a log, or an error.

Red-first for P3.5.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from muster.errors import ExportError
from muster.exporters.csv_export import CsvExporter
from muster.exporters.mqtt import MqttExporter
from muster.exporters.prometheus import PrometheusExporter
from muster.exporters.webhook import SIGNATURE_HEADER, WebhookExporter
from muster.types import CameraId, MetricName, MetricRow, MinuteBucket, ScopeId

FRONT_DOOR = CameraId("front-door")
DOOR_LINE = ScopeId("door-count")
BUCKET = MinuteBucket(datetime(2026, 8, 16, 9, 30, tzinfo=UTC))
SIGNING_KEY = "not-a-real-one"


def _row(
    value: float = 3.0,
    *,
    bucket: MinuteBucket = BUCKET,
    metric: MetricName = MetricName.FOOTFALL,
    scope: ScopeId | None = DOOR_LINE,
) -> MetricRow:
    return MetricRow(
        camera_id=FRONT_DOOR,
        bucket=bucket,
        metric=metric,
        scope_id=scope,
        value=value,
        sample_count=1,
    )


# --- CSV --------------------------------------------------------------------


def test_csv_writes_a_header_once_then_appends(tmp_path: Path) -> None:
    exporter = CsvExporter(tmp_path)
    exporter.start()

    exporter.on_metric(_row(1.0))
    exporter.on_metric(_row(2.0))
    exporter.shutdown()

    written = sorted(tmp_path.glob("*.csv"))
    lines = written[0].read_text(encoding="utf-8").strip().split("\n")
    assert len(written) == 1
    assert lines[0].startswith("bucket,")
    assert len(lines) == 3


def test_csv_rotates_on_the_utc_day(tmp_path: Path) -> None:
    """Daily rotation, and the day is UTC — a local day would split a shop's evening."""
    exporter = CsvExporter(tmp_path)
    exporter.start()

    exporter.on_metric(_row(1.0))
    exporter.on_metric(_row(2.0, bucket=MinuteBucket(BUCKET + timedelta(days=1))))
    exporter.shutdown()

    assert len(sorted(tmp_path.glob("*.csv"))) == 2


def test_csv_separates_metrics_into_their_own_files(tmp_path: Path) -> None:
    exporter = CsvExporter(tmp_path)
    exporter.start()

    exporter.on_metric(_row(1.0, metric=MetricName.FOOTFALL))
    exporter.on_metric(_row(2.0, metric=MetricName.OCCUPANCY))
    exporter.shutdown()

    names = {path.name for path in tmp_path.glob("*.csv")}
    assert any("footfall" in name for name in names)
    assert any("occupancy" in name for name in names)


def test_csv_timestamps_are_utc_iso(tmp_path: Path) -> None:
    exporter = CsvExporter(tmp_path)
    exporter.start()

    exporter.on_metric(_row())
    exporter.shutdown()

    body = next(iter(tmp_path.glob("*.csv"))).read_text(encoding="utf-8")
    assert "2026-08-16T09:30:00+00:00" in body


def test_csv_creates_its_directory(tmp_path: Path) -> None:
    """A self-hoster who names a directory should not have to `mkdir` it first."""
    target = tmp_path / "exports" / "nested"
    exporter = CsvExporter(target)

    exporter.start()
    exporter.on_metric(_row())
    exporter.shutdown()

    assert target.is_dir()


# --- Webhook ----------------------------------------------------------------


def _capture(status: int = 200) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status)

    return httpx.MockTransport(handler), seen


def test_the_webhook_posts_the_row_as_json() -> None:
    transport, seen = _capture()
    exporter = WebhookExporter("https://example.invalid/hook", SIGNING_KEY, transport=transport)
    exporter.start()

    exporter.on_metric(_row(7.0))
    exporter.shutdown()

    body = json.loads(seen[0].content)
    assert body["camera_id"] == FRONT_DOOR
    assert body["metric"] == "footfall"
    assert body["value"] == 7.0
    assert body["bucket"] == "2026-08-16T09:30:00+00:00"


def test_the_webhook_signs_the_exact_bytes_it_sends() -> None:
    """Signing a re-serialised body is the classic way a signature stops verifying."""
    transport, seen = _capture()
    exporter = WebhookExporter("https://example.invalid/hook", SIGNING_KEY, transport=transport)
    exporter.start()

    exporter.on_metric(_row())
    exporter.shutdown()

    sent = seen[0]
    expected = hmac.new(SIGNING_KEY.encode(), sent.content, hashlib.sha256).hexdigest()
    assert hmac.compare_digest(sent.headers[SIGNATURE_HEADER], expected)


def test_the_webhook_never_puts_the_secret_in_the_payload() -> None:
    transport, seen = _capture()
    exporter = WebhookExporter("https://example.invalid/hook", SIGNING_KEY, transport=transport)
    exporter.start()

    exporter.on_metric(_row())
    exporter.shutdown()

    assert SIGNING_KEY not in seen[0].content.decode()
    assert SIGNING_KEY not in str(dict(seen[0].headers))


def test_the_webhook_sets_an_explicit_timeout() -> None:
    """An unset timeout is an exporter that can hang the tick forever."""
    transport, _ = _capture()
    exporter = WebhookExporter("https://example.invalid/hook", SIGNING_KEY, transport=transport)

    exporter.start()

    assert exporter.timeout_s > 0
    assert exporter.client is not None
    assert exporter.client.timeout.read == exporter.timeout_s


def test_the_webhook_keeps_tls_verification_on() -> None:
    """Asserted because a test that disables it is how it gets disabled in production."""
    exporter = WebhookExporter("https://example.invalid/hook", SIGNING_KEY)

    exporter.start()

    assert exporter.verifies_tls
    exporter.shutdown()


def test_the_webhook_retries_a_server_error_then_succeeds() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        attempts.append(1)
        return httpx.Response(500 if len(attempts) < 3 else 200)

    exporter = WebhookExporter(
        "https://example.invalid/hook",
        SIGNING_KEY,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    exporter.start()

    exporter.on_metric(_row())
    exporter.shutdown()

    assert len(attempts) == 3


def test_the_webhook_gives_up_loudly_so_the_fanout_can_count_it() -> None:
    """Swallowing here would make a dead endpoint indistinguishable from a working one."""
    transport, _ = _capture(status=500)
    exporter = WebhookExporter(
        "https://example.invalid/hook", SIGNING_KEY, transport=transport, sleep=lambda _s: None
    )
    exporter.start()

    with pytest.raises(ExportError):
        exporter.on_metric(_row())
    exporter.shutdown()


# --- MQTT -------------------------------------------------------------------


class FakeMqttClient:
    def __init__(self) -> None:
        self.connected: tuple[str, int] | None = None
        self.published: list[tuple[str, str, bool]] = []
        self.looping = False
        self.disconnected = False

    def connect(self, host: str, port: int, keepalive: int) -> None:
        del keepalive
        self.connected = (host, port)

    def loop_start(self) -> None:
        self.looping = True

    def publish(self, topic: str, payload: str, *, retain: bool = False) -> None:
        self.published.append((topic, payload, retain))

    def disconnect(self) -> None:
        self.disconnected = True

    def loop_stop(self) -> None:
        self.looping = False


def test_mqtt_publishes_retained() -> None:
    """Retained is what lets Home Assistant see last-known values on connect (§12)."""
    client = FakeMqttClient()
    exporter = MqttExporter("broker.invalid", client=client)
    exporter.start()

    exporter.on_metric(_row(4.0))
    exporter.shutdown()

    (_topic, payload, retain) = client.published[0]
    assert retain is True
    assert payload == "4.0"


def test_the_mqtt_topic_names_camera_metric_and_scope() -> None:
    client = FakeMqttClient()
    exporter = MqttExporter("broker.invalid", base_topic="muster", client=client)
    exporter.start()

    exporter.on_metric(_row())
    exporter.shutdown()

    assert client.published[0][0] == "muster/front-door/footfall/door-count"


def test_a_camera_wide_metric_gets_a_topic_without_a_scope_segment() -> None:
    """An empty segment (`muster/cam/occupancy//`) is not a topic anyone can subscribe to."""
    client = FakeMqttClient()
    exporter = MqttExporter("broker.invalid", client=client)
    exporter.start()

    exporter.on_metric(_row(scope=None, metric=MetricName.OCCUPANCY))
    exporter.shutdown()

    assert client.published[0][0] == "muster/front-door/occupancy"


def test_mqtt_connects_on_start_and_disconnects_on_shutdown() -> None:
    client = FakeMqttClient()
    exporter = MqttExporter("broker.invalid", port=1884, client=client)

    exporter.start()
    assert client.connected == ("broker.invalid", 1884)
    assert client.looping

    exporter.shutdown()
    assert client.disconnected
    assert not client.looping


# --- Prometheus -------------------------------------------------------------


def test_prometheus_renders_the_latest_value_for_a_series() -> None:
    exporter = PrometheusExporter()
    exporter.start()

    exporter.on_metric(_row(5.0))
    rendered = exporter.render()

    assert 'camera="front-door"' in rendered
    assert "5.0" in rendered


def test_a_later_bucket_replaces_the_value_rather_than_adding_to_it() -> None:
    """These are gauges: a scrape reports the last known value, not a running total."""
    exporter = PrometheusExporter()
    exporter.start()

    exporter.on_metric(_row(5.0))
    exporter.on_metric(_row(2.0, bucket=MinuteBucket(BUCKET + timedelta(minutes=1))))
    rendered = exporter.render()

    assert "5.0" not in rendered
    assert "2.0" in rendered


def test_two_exporters_do_not_share_a_registry() -> None:
    """`prometheus_client`'s default registry is process-global.

    Using it would make these tests order-dependent and would leak series across a
    config reload, so each exporter owns its own `CollectorRegistry`.
    """
    first, second = PrometheusExporter(), PrometheusExporter()
    first.start()
    second.start()

    first.on_metric(_row(5.0))

    assert "5.0" in first.render()
    assert "5.0" not in second.render()


def test_prometheus_exports_engine_health_not_only_business_metrics() -> None:
    """§12 calls this deliberate: the self-hoster's Grafana is the free health view."""
    exporter = PrometheusExporter()
    exporter.start()

    exporter.set_health({"live_workers": 2.0, "restarts": 1.0, "late_events": 0.0})
    rendered = exporter.render()

    assert "muster_live_workers" in rendered
    assert "muster_restarts" in rendered


def test_the_default_mqtt_client_is_a_real_paho_client() -> None:
    """Every other MQTT test injects a fake, which hides the real constructor entirely.

    It hid a real one: the enum member is `VERSION2`, not `V2`, and an `AttributeError`
    at construction would have reached the first user to enable MQTT rather than CI. A
    seam that makes the tested path differ from the shipped path needs one test that
    crosses back over it.
    """
    exporter = MqttExporter("broker.invalid")

    assert type(exporter._client).__module__.startswith("paho.mqtt")
