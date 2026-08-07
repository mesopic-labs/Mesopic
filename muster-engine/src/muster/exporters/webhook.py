"""HMAC-signed JSON POST per bucket, retried with backoff, at-least-once.

This exporter is **bidirectional**, and that is deliberate. Conversion is the one metric
the cameras cannot sense: its *ingress* side is a small authenticated endpoint a till or
POS bridge calls with transaction counts, which the aggregator joins against footfall per
bucket. Conversion stays honest — derived from a real external signal, never invented
from video.

Implements part of P3.5.
"""

from __future__ import annotations

from muster.types import MetricRow


class WebhookExporter:
    """Signed, retrying, at-least-once delivery of committed buckets."""

    def __init__(self, url: str, secret: str, *, timeout_s: float = 10.0) -> None:
        # `secret` is resolved from the env var named in config, never read from YAML.
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def on_metric(self, row: MetricRow) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError
