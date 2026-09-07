"""Ship unsynced metric rows to the cloud `/ingest` endpoint.

Offline is not an error state: rows simply pile up unsynced and drain in bucket order on
reconnect. Every batch carries an idempotency key so a retry after a lost 2xx cannot
double-send.

The site token comes from an environment variable named in config, never from the YAML.

This module is the HTTP half only — one request, one answer. The deciding what to send and
when lives in `loop.py`, so that a transport swap (ADR-0010 keeps MQTT as a possibility it
rejected rather than one it made impossible) does not touch the draining logic.

Implements C5.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any, Final, Self

import httpx

from mesopic.errors import SyncError
from mesopic.sync.payload import batch_for
from mesopic.types import MetricRow

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S: Final = 15.0
"""Generous for a batch of a few thousand rows on a poor uplink, and finite, which is the
part that matters: without it a stalled connection parks the sync task for the life of the
process while `unsynced_count` climbs and nothing explains why."""

_OK_FROM: Final = 200
_OK_BELOW: Final = 300
_THROTTLED: Final = 429


class ThrottledError(SyncError):
    """The cloud asked us to slow down. Not a failure — a schedule.

    Carries `Retry-After` when the response gave one, because discarding it turns a polite
    refusal into the retry loop the refusal was trying to prevent.
    """

    def __init__(self, message: str, retry_after_s: float | None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class MetricsSyncClient:
    """Authenticated, metrics-only, one-way client. Nothing is ever pulled down.

    It cannot read a frame, and that is structural rather than promised: this module
    imports the store not at all, an import-linter contract forbids `mesopic.sync` from
    reaching `mesopic.ingest`, `mesopic.detector`, `av` or `cv2`, and the loop that feeds
    it is typed against a source exposing three metrics methods and nothing else
    (ADR-0005).
    """

    def __init__(
        self,
        endpoint: str,
        token: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not endpoint.startswith("https://"):
            # Config validation says the same thing, so this is the second lock on one
            # door: it stops a caller constructing the client directly from doing what the
            # YAML is not allowed to.
            msg = "cloud sync endpoint must be https"
            raise SyncError(msg)
        self._endpoint = endpoint
        # Resolved from the env var named in config, never read from YAML, and never put
        # in a log line or an exception message — this endpoint fails routinely, so a
        # chatty failure path would write the credential out on every reconnect.
        self._token = token
        self.timeout_s = timeout_s
        self.verify = True
        """Stated rather than left to the default. A bearer token to a public endpoint
        over an unverified connection is the token handed to whoever holds the network."""
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._client = self._build_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send(self, rows: list[MetricRow]) -> None:
        """POST one batch. Raises `SyncError` on anything but a 2xx.

        Raising rather than returning a status is deliberate: the caller's next action is
        to stamp these rows as delivered, and that must be unreachable unless the cloud
        actually confirmed them.
        """
        if not rows:
            # The loop should not call this with nothing; if it does, silence is the right
            # answer rather than a request that asks the cloud to remember an empty batch.
            return
        body = batch_for(rows)
        try:
            response = await self._post(body)
        except httpx.HTTPError as error:
            # `type(error).__name__` and not the error itself: httpx puts the full URL in
            # some messages, and the endpoint is about to be joined by a credential in
            # somebody's mental model of "what is safe to log".
            msg = f"cloud sync request failed: {type(error).__name__}"
            raise SyncError(msg) from error
        _raise_for_status(response)

    async def _post(self, body: dict[str, Any]) -> httpx.Response:
        if self._client is not None:
            return await self._client.post(self._endpoint, json=body, headers=self._headers())
        async with self._build_client() as client:
            return await client.post(self._endpoint, json=body, headers=self._headers())

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self.timeout_s, verify=self.verify, transport=self._transport
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}


def _raise_for_status(response: httpx.Response) -> None:
    """Turn the answer into either a return or one of two errors.

    The status is the only detail carried out. A body from a rejected batch may quote what
    we sent, and this runs on a path that logs.
    """
    if _OK_FROM <= response.status_code < _OK_BELOW:
        return
    if response.status_code == _THROTTLED:
        msg = "cloud sync throttled"
        raise ThrottledError(msg, _retry_after(response))
    msg = f"cloud sync rejected the batch with status {response.status_code}"
    raise SyncError(msg)


def _retry_after(response: httpx.Response) -> float | None:
    """`Retry-After` in seconds, or `None` if it was absent or unparseable.

    Only the delta-seconds form is read. The HTTP-date form is legal and nothing we serve
    emits it; guessing at a date against an edge clock we already know drifts would be
    worse than falling back to our own backoff.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
