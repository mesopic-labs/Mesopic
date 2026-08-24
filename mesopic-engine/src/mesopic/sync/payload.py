"""The batch this engine sends, and the key that makes a resend free.

One half of a versioned contract with a service that lives in another repository
(ADR-0010). Nothing here imports that service — the cloud is not a dependency and never
will be — so the contract is kept by the field names below and by the tests beside them.

What is absent is as load-bearing as what is present. There is no `site_id`: the cloud
derives the site from the credential and refuses a body that names one. There are no
coordinates, no crop, no frame. The receiving endpoint forbids unknown fields outright, so
a field added here without the other side knowing about it does not degrade — it fails
every sync until someone notices, which is the failure mode we want.

Implements part of C5.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mesopic.types import MetricRow

SYNC_FORMAT: Final = 1
"""The envelope version the cloud must agree with. Named `sync_format` rather than
`schema_version` because both sides already have a schema version of their own, and one
name for two unrelated things is how a mismatch gets misread (ADR-0010)."""

CAMERA_WIDE: Final = ""
"""What a camera-wide metric's scope looks like on the wire.

The store maps `'' <-> None` at its own boundary, so rows arrive here with `None`. It has
to go back as `''`: JSON `null` is refused by the cloud, and `str(None)` would store the
literal text `"None"` as a zone name, which it would cheerfully accept.
"""

_KEY_DIGEST_CHARS: Final = 16
"""64 bits of the digest. The cloud bounds the key at 128 characters and the risk being
managed is accidental collision between two batches from one site, not an adversary — a
site would need billions of distinct batches for this to be close."""


def batch_for(rows: Sequence[MetricRow]) -> dict[str, Any]:
    """Build the request body for one batch of unsynced rows.

    Order is preserved from the store, which hands them back in bucket order, so a batch
    that fails leaves the oldest rows unsynced rather than a hole in the middle.
    """
    return {
        "sync_format": SYNC_FORMAT,
        "idempotency_key": idempotency_key(rows),
        "samples": [_sample_for(row) for row in rows],
    }


def idempotency_key(rows: Sequence[MetricRow]) -> str:
    """A name for this batch that a retry of the same rows reproduces exactly.

    Derived rather than minted, and that is the whole design. After a lost 2xx the client
    re-reads the same unsynced rows and sends them again; if the key were a fresh UUID the
    cloud's batch log could never match it, and the second layer of idempotency would be
    dead weight behind the row-level upsert.

    It covers the rows' **natural keys only**, not their values. A corrected bucket keeps
    the batch's identity, which is right: the cloud upserts on the same natural key, so a
    correction resent under the same batch name still overwrites.

    `hashlib`, never `hash()`: Python salts string hashing per process, so a restart
    mid-retry would mint a new key for identical rows — defeating the dedup at exactly the
    moment it is there for.
    """
    digest = hashlib.sha256()
    for row in rows:
        # The separator matters: without it, ("ab", "c") and ("a", "bc") hash alike, and
        # a camera named for a zone could collide with the zone.
        digest.update(
            "\x1f".join(
                (
                    row.camera_id,
                    row.metric.value,
                    CAMERA_WIDE if row.scope_id is None else row.scope_id,
                    row.bucket.isoformat(),
                )
            ).encode()
        )
        digest.update(b"\x1e")
    return f"{_stamp(rows)}_{digest.hexdigest()[:_KEY_DIGEST_CHARS]}"


def _stamp(rows: Sequence[MetricRow]) -> str:
    """The first bucket, so a key is legible in a log without being looked up.

    Fixed width whatever the batch size, which is what keeps the key inside the 128
    characters the cloud accepts even for a batch of five thousand.
    """
    if not rows:
        return "empty"
    return rows[0].bucket.strftime("%Y%m%dT%H%M")


def _sample_for(row: MetricRow) -> dict[str, Any]:
    return {
        "camera_id": row.camera_id,
        "metric": row.metric.value,
        "scope_id": CAMERA_WIDE if row.scope_id is None else row.scope_id,
        "bucket_ts": row.bucket.isoformat(),
        "value": row.value,
        "staff_value": row.staff_value,
        "sample_count": row.sample_count,
    }
