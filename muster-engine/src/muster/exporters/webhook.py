"""HMAC-signed JSON POST per bucket, retried with backoff, at-least-once.

This exporter is **bidirectional**, and that is deliberate. Conversion is the one metric
the cameras cannot sense: its *ingress* side is a small authenticated endpoint a till or
POS bridge calls with transaction counts, which the aggregator joins against footfall per
bucket. Conversion stays honest — derived from a real external signal, never invented
from video. The ingress half is the local API's (P3.1/P3.4); what lives here is egress.

Two details that are easy to get subtly wrong:

* **The signature covers the exact bytes on the wire.** The body is serialised once and
  both signed and sent; signing a re-serialised copy is the classic way a signature stops
  verifying against a receiver that hashes what it actually received.
* **Failure is raised, not swallowed.** The fan-out counts it (§12). An exporter that
  quietly gave up would be indistinguishable from one that worked.

Implements part of P3.5.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Callable

import httpx

from muster.errors import ExportError
from muster.types import MetricRow

SIGNATURE_HEADER = "X-Muster-Signature"
"""Lowercase hex HMAC-SHA256 of the request body, keyed by the shared secret."""

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_ATTEMPTS = 3
BACKOFF_BASE_S = 0.5

_RETRYABLE_FROM = 500


class WebhookExporter:
    """Signed, retrying, at-least-once delivery of committed buckets."""

    def __init__(
        self,
        url: str,
        secret: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
        attempts: int = DEFAULT_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # `secret` is resolved from the env var named in config, never read from YAML.
        self._url = url
        self._secret = secret.encode()
        self._transport = transport
        self._attempts = attempts
        self._sleep = sleep
        self.timeout_s = timeout_s
        self.verifies_tls = True
        """Never turned off, including in tests. A transport is injected instead."""
        self.client: httpx.Client | None = None

    def start(self) -> None:
        self.client = httpx.Client(
            timeout=httpx.Timeout(self.timeout_s),
            verify=self.verifies_tls,
            transport=self._transport,
        )

    def on_metric(self, row: MetricRow) -> None:
        if self.client is None:  # pragma: no cover - the fan-out always starts first
            msg = "webhook exporter used before start()"
            raise ExportError(msg)
        body = _body(row)
        headers = {
            "content-type": "application/json",
            SIGNATURE_HEADER: hmac.new(self._secret, body, hashlib.sha256).hexdigest(),
        }
        self._deliver(body, headers)

    def shutdown(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def _deliver(self, body: bytes, headers: dict[str, str]) -> None:
        assert self.client is not None  # noqa: S101 - checked by the only caller
        for attempt in range(1, self._attempts + 1):
            try:
                response = self.client.post(self._url, content=body, headers=headers)
            except httpx.HTTPError:
                # Never chained and never formatted: an httpx error message carries the
                # URL, and the URL is a secret here (ADR-0005, CLAUDE.md).
                if attempt == self._attempts:
                    msg = "webhook delivery failed after retries"
                    raise ExportError(msg) from None
            else:
                if response.status_code < _RETRYABLE_FROM:
                    return
                if attempt == self._attempts:
                    msg = f"webhook endpoint returned {response.status_code}"
                    raise ExportError(msg)
            self._sleep(BACKOFF_BASE_S * 2 ** (attempt - 1))


def _body(row: MetricRow) -> bytes:
    """One row as compact, key-sorted JSON — serialised once, signed and sent."""
    payload = {
        "bucket": row.bucket.isoformat(),
        "camera_id": row.camera_id,
        "metric": row.metric.value,
        "scope_id": row.scope_id,
        "value": row.value,
        "staff_value": row.staff_value,
        "sample_count": row.sample_count,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
