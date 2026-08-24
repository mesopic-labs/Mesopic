"""Turning `cloud_sync:` in the YAML into a running loop, or into a clear refusal.

Implements part of C5.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

import pytest

from mesopic.config.schema import CloudSyncConfig
from mesopic.errors import ConfigError
from mesopic.sync.build import build_sync_loop
from mesopic.types import MetricRow

TOKEN_VAR = "MESOPIC_TEST_SITE_TOKEN"  # noqa: S105 - the name of a variable, not a token


class FakeStore:
    """The three methods `MetricsSource` allows, each actually using its argument — a
    stub that ignored them would stop satisfying the protocol the moment mypy looked."""

    def __init__(self) -> None:
        self.rows: list[MetricRow] = []
        self.stamped: list[MetricRow] = []

    def unsynced_metrics(self, limit: int) -> list[MetricRow]:
        return self.rows[:limit]

    def mark_synced(self, rows: Sequence[MetricRow], synced_at: datetime) -> None:
        self.stamped.extend(rows)
        del synced_at

    def unsynced_count(self) -> int:
        return len(self.rows)


def a_config(**overrides: object) -> CloudSyncConfig:
    fields: dict[str, object] = {
        "enabled": True,
        "endpoint": "https://cloud.example/ingest",
        "site_token_env": TOKEN_VAR,
    } | overrides
    return CloudSyncConfig(**fields)  # type: ignore[arg-type]


def test_sync_is_off_by_default() -> None:
    """The free, local engine never talks to the cloud (ADR-0001), and "off" has to mean
    no loop rather than a loop that happens not to be pointed anywhere."""
    assert build_sync_loop(CloudSyncConfig(), FakeStore()) is None


def test_an_enabled_sync_builds_a_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_VAR, "msk_deadbeef_notreal")
    assert build_sync_loop(a_config(), FakeStore()) is not None


def test_an_enabled_sync_with_no_token_in_the_environment_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loud at load rather than quiet at runtime.

    The alternative — start, fail every request, climb `unsynced_count` — is the shape of
    an outage nobody notices for a week, and it is the operator's deploy that is wrong
    rather than the network.
    """
    monkeypatch.delenv(TOKEN_VAR, raising=False)

    with pytest.raises(ConfigError):
        build_sync_loop(a_config(), FakeStore())


def test_the_refusal_names_the_variable_and_never_a_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_VAR, "")

    with pytest.raises(ConfigError) as raised:
        build_sync_loop(a_config(), FakeStore())

    assert TOKEN_VAR in str(raised.value)


def test_a_token_is_never_written_to_the_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "msk_deadbeef_averydistinctivesecret"  # noqa: S105 - a fabricated token in a test, not a credential
    monkeypatch.setenv(TOKEN_VAR, secret)

    with caplog.at_level(logging.DEBUG, logger="mesopic"):
        build_sync_loop(a_config(), FakeStore())

    assert secret not in caplog.text
