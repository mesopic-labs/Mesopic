"""The batch this engine sends, and the key that makes a resend free.

The payload is one half of a versioned contract with a service in another repository
(ADR-0010), so these tests are written as the contract rather than as the implementation:
what the fields are called, what is absent, and what a retry reproduces.

Implements part of C5.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime

import pytest

from mesopic.sync.payload import SYNC_FORMAT, batch_for, idempotency_key
from mesopic.types import CameraId, MetricName, MetricRow, MinuteBucket, ScopeId

pytestmark = pytest.mark.privacy

A_BUCKET = MinuteBucket(datetime(2026, 7, 13, 9, 15, tzinfo=UTC))
LATER = MinuteBucket(datetime(2026, 7, 13, 9, 16, tzinfo=UTC))


def a_row(**overrides: object) -> MetricRow:
    fields: dict[str, object] = {
        "camera_id": CameraId("cam-door"),
        "bucket": A_BUCKET,
        "metric": MetricName.FOOTFALL,
        "scope_id": None,
        "value": 4.0,
        "staff_value": None,
        "sample_count": 120,
    } | overrides
    return MetricRow(**fields)  # type: ignore[arg-type]


# --- What goes on the wire ---------------------------------------------------


def test_the_batch_declares_the_format_it_speaks() -> None:
    """A mismatch has to fail loud rather than mis-store (ADR-0009)."""
    assert batch_for([a_row()])["sync_format"] == SYNC_FORMAT


def test_a_camera_wide_scope_travels_as_an_empty_string_not_null() -> None:
    """The store maps `'' -> None` on read; the wire wants `''` back.

    A `None` here would serialise as JSON `null`, which the cloud refuses — and a naive
    `str(None)` would store the literal text "None" as a zone name, which it would not.
    """
    sample = batch_for([a_row(scope_id=None)])["samples"][0]
    assert sample["scope_id"] == ""


def test_a_named_scope_travels_unchanged() -> None:
    sample = batch_for([a_row(scope_id=ScopeId("till"))])["samples"][0]
    assert sample["scope_id"] == "till"


def test_a_bucket_travels_as_utc_iso8601() -> None:
    sample = batch_for([a_row()])["samples"][0]
    assert sample["bucket_ts"] == "2026-07-13T09:15:00+00:00"


def test_the_frame_count_and_staff_split_are_carried() -> None:
    """Both exist on the edge row and both have a home in the cloud schema now."""
    sample = batch_for([a_row(staff_value=1.0, sample_count=99)])["samples"][0]
    assert sample["staff_value"] == 1.0
    assert sample["sample_count"] == 99


def test_the_payload_carries_no_site_id() -> None:
    """The cloud derives the site from the credential and forbids unknown fields, so a
    `site_id` here would not be ignored — it would make every batch a 422."""
    assert "site_id" not in batch_for([a_row()])


def test_the_payload_carries_nothing_but_the_agreed_fields() -> None:
    """The absence *is* the GDPR posture, and the endpoint forbids extras.

    Written as an exact set rather than a blocklist: a field added here without the cloud
    knowing about it would fail every sync, and a field that could carry a pixel would be
    worse.
    """
    assert set(batch_for([a_row()])) == {"sync_format", "idempotency_key", "samples"}
    assert set(batch_for([a_row()])["samples"][0]) == {
        "camera_id",
        "metric",
        "scope_id",
        "bucket_ts",
        "value",
        "staff_value",
        "sample_count",
    }


def test_rows_keep_their_order() -> None:
    """The store hands them back in bucket order and the batch preserves it, so a partial
    failure leaves the oldest unsynced rather than a hole in the middle."""
    batch = batch_for([a_row(bucket=A_BUCKET), a_row(bucket=LATER)])
    assert [s["bucket_ts"] for s in batch["samples"]] == [
        "2026-07-13T09:15:00+00:00",
        "2026-07-13T09:16:00+00:00",
    ]


# --- The key that makes a resend free ----------------------------------------


def test_the_same_rows_produce_the_same_key() -> None:
    """The whole point. After a lost 2xx the client re-reads the same unsynced rows, so
    the key must be reproduced exactly or the cloud's batch log never dedupes and the
    second layer of idempotency is dead weight.
    """
    rows = [a_row(), a_row(bucket=LATER)]
    assert idempotency_key(rows) == idempotency_key(list(rows))


def test_different_rows_produce_different_keys() -> None:
    assert idempotency_key([a_row()]) != idempotency_key([a_row(bucket=LATER)])


def test_the_key_ignores_the_value_and_tracks_identity() -> None:
    """A corrected bucket keeps the batch's identity: the natural keys are what the cloud
    upserts on, and re-sending a corrected value under the same key must still overwrite.
    """
    assert idempotency_key([a_row(value=4.0)]) == idempotency_key([a_row(value=9.0)])


def test_the_key_fits_the_field_the_cloud_accepts() -> None:
    """The cloud bounds `idempotency_key` at 128 characters and refuses a longer one, so a
    thousand-row batch must not produce a key that grows with it."""
    many = [a_row(bucket=MinuteBucket(datetime(2026, 7, 13, 9, m, tzinfo=UTC))) for m in range(60)]
    assert 0 < len(idempotency_key(many)) <= 128


def test_the_key_is_stable_across_processes() -> None:
    """Derived by hashing, never by `hash()`: Python salts string hashing per process, so
    a restart mid-retry would mint a new key for the same rows and defeat the dedup at
    exactly the moment it is needed.

    Two interpreters with deliberately different hash seeds, because that is the only way
    to tell a `hashlib` digest from a `hash()` that looks stable within one process.
    """
    script = (
        "from datetime import UTC, datetime;"
        "from mesopic.sync.payload import idempotency_key;"
        "from mesopic.types import CameraId, MetricName, MetricRow, MinuteBucket;"
        "print(idempotency_key([MetricRow("
        "camera_id=CameraId('cam-door'),"
        "bucket=MinuteBucket(datetime(2026, 7, 13, 9, 15, tzinfo=UTC)),"
        "metric=MetricName.FOOTFALL, scope_id=None, value=4.0)]))"
    )
    runs = [
        subprocess.run(  # noqa: S603 - a fixed argv running this interpreter, no shell
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=os.environ | {"PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "12345")
    ]

    assert runs[0] == runs[1] != ""
