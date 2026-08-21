"""Ship unsynced metric rows to the cloud `/ingest` endpoint.

Offline is not an error state: rows simply pile up unsynced and drain in bucket order on
reconnect. Every batch carries an idempotency key so a retry after a lost 2xx cannot
double-send.

The site token comes from an environment variable named in config, never from the YAML.

Implements C5.
"""

from __future__ import annotations

from mesopic.types import MetricRow, SiteId


class MetricsSyncClient:
    """Authenticated, metrics-only, one-way client. Nothing is ever pulled down."""

    def __init__(self, endpoint: str, site_id: SiteId, token: str) -> None:
        raise NotImplementedError

    async def send(self, rows: list[MetricRow]) -> None:
        """POST one batch. Raises `SyncError` on anything but a 2xx."""
        raise NotImplementedError
