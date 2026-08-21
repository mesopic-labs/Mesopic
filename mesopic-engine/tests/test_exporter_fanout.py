"""Exporters fail independently, or they are not exporters.

§12's rule is the whole point of this file: a dead MQTT broker must not stall the
Prometheus scrape, the CSV rotation, or the pipeline. So the fan-out swallows and counts
per-exporter failures rather than letting one peer's outage become an engine outage.

The other load-bearing property is ordering: rows are handed to exporters only after the
store has accepted them. An exporter that announces a number the store rejected has told
the outside world something the engine does not believe.

Red-first for P3.5.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from mesopic.config.schema import MesopicConfig
from mesopic.errors import ConfigError
from mesopic.exporters.fanout import ExporterFanout, build_exporters
from mesopic.types import CameraId, MetricName, MetricRow, MinuteBucket, ScopeId

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "mesopic.yaml"

FRONT_DOOR = CameraId("front-door")
BUCKET = MinuteBucket(datetime(2026, 8, 16, 9, 30, tzinfo=UTC))


def _row(value: float = 3.0, *, metric: MetricName = MetricName.FOOTFALL) -> MetricRow:
    return MetricRow(
        camera_id=FRONT_DOOR,
        bucket=BUCKET,
        metric=metric,
        scope_id=ScopeId("door-count"),
        value=value,
        sample_count=1,
    )


class Recorder:
    """An exporter that remembers what it was given."""

    def __init__(self) -> None:
        self.rows: list[MetricRow] = []
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def on_metric(self, row: MetricRow) -> None:
        self.rows.append(row)

    def shutdown(self) -> None:
        self.stopped += 1


class Broken:
    """An exporter whose peer is down — the case §12 is written about."""

    def __init__(self) -> None:
        self.calls = 0

    def start(self) -> None:
        message = "broker unreachable"
        raise ConnectionError(message)

    def on_metric(self, row: MetricRow) -> None:
        del row
        self.calls += 1
        message = "broker unreachable"
        raise ConnectionError(message)

    def shutdown(self) -> None:
        message = "broker unreachable"
        raise ConnectionError(message)


@pytest.fixture
def raw_config() -> dict[str, Any]:
    parsed = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


# --- Fan-out ----------------------------------------------------------------


def test_every_exporter_sees_every_row() -> None:
    first, second = Recorder(), Recorder()
    fanout = ExporterFanout({"first": first, "second": second})

    fanout.on_metrics([_row(1.0), _row(2.0)])

    assert [row.value for row in first.rows] == [1.0, 2.0]
    assert [row.value for row in second.rows] == [1.0, 2.0]


def test_a_broken_exporter_does_not_stop_the_others() -> None:
    """The §12 rule, stated as a test: one peer's outage is not an engine outage."""
    healthy, broken = Recorder(), Broken()
    fanout = ExporterFanout({"healthy": healthy, "broken": broken})

    fanout.on_metrics([_row()])

    assert len(healthy.rows) == 1
    assert broken.calls == 1


def test_a_broken_exporter_is_counted_not_silent() -> None:
    """A degraded exporter must be a number someone can read off /healthz later."""
    fanout = ExporterFanout({"broken": Broken(), "healthy": Recorder()})

    fanout.on_metrics([_row(), _row()])

    assert fanout.failures == {"broken": 2}
    assert fanout.healthy_names() == ("healthy",)


def test_a_broken_exporter_does_not_break_start_or_shutdown() -> None:
    """Startup order must not depend on whether a broker happened to be up."""
    healthy = Recorder()
    fanout = ExporterFanout({"broken": Broken(), "healthy": healthy})

    fanout.start()
    fanout.shutdown()

    assert healthy.started == 1
    assert healthy.stopped == 1
    assert fanout.failures["broken"] == 2


def test_an_empty_fanout_is_not_an_error() -> None:
    """Every exporter disabled is the default config, not a misconfiguration."""
    fanout = ExporterFanout({})

    fanout.start()
    fanout.on_metrics([_row()])
    fanout.shutdown()

    assert fanout.failures == {}


# --- Building from config ---------------------------------------------------


def test_only_enabled_exporters_are_built(
    raw_config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MESOPIC_WEBHOOK_SECRET", "not-a-real-one")
    raw_config["exporters"]["csv"] = {"enabled": True, "dir": str(tmp_path)}
    raw_config["exporters"]["mqtt"] = {"enabled": False}
    config = MesopicConfig.model_validate(raw_config)

    fanout = build_exporters(config)

    assert "csv" in fanout.names()
    assert "mqtt" not in fanout.names()


def test_a_config_with_no_exporters_builds_an_empty_fanout(raw_config: dict[str, Any]) -> None:
    raw_config["exporters"] = {}
    config = MesopicConfig.model_validate(raw_config)

    fanout = build_exporters(config)

    assert fanout.names() == ()


def test_a_webhook_whose_secret_env_var_is_unset_is_refused(
    raw_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently posting unsigned is worse than refusing to start."""
    monkeypatch.delenv("MESOPIC_WEBHOOK_SECRET", raising=False)
    config = MesopicConfig.model_validate(raw_config)

    with pytest.raises(ConfigError, match="MESOPIC_WEBHOOK_SECRET"):
        build_exporters(config)


@pytest.mark.privacy
def test_the_exporter_interface_carries_no_pixels(raw_config: dict[str, Any]) -> None:
    """`on_metric` takes a `MetricRow`, whose fields are scalars and ids only.

    Structural rather than aspirational: the day someone adds an image-shaped field to
    `MetricRow` so an exporter can "just send a thumbnail", this fails.
    """
    row = _row()

    for field in fields(row):
        value = getattr(row, field.name)
        assert not hasattr(value, "shape"), f"{field.name} carries an array"
        assert not isinstance(value, bytes | bytearray | memoryview)
