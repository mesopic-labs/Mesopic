"""The HTTP half of the sync client: what it sends, and how it fails.

Driven through `httpx.MockTransport`, so a real `httpx.AsyncClient` runs its real request
path against a fake network. Nothing here opens a socket and nothing here is a mock of our
own code — the assertions are on the request that would have gone out.

Implements part of C5.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import pytest

from mesopic.errors import SyncError
from mesopic.sync.client import MetricsSyncClient, ThrottledError
from mesopic.types import CameraId, MetricName, MetricRow, MinuteBucket

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.privacy

ENDPOINT = "https://cloud.example/ingest"
A_TOKEN = "msk_deadbeef_thisisnotarealsecret"  # noqa: S105 - a fabricated token in a test, not a credential
A_BUCKET = MinuteBucket(datetime(2026, 7, 13, 9, 15, tzinfo=UTC))


def a_row() -> MetricRow:
    return MetricRow(
        camera_id=CameraId("cam-door"),
        bucket=A_BUCKET,
        metric=MetricName.FOOTFALL,
        scope_id=None,
        value=4.0,
    )


def client_over(handler: Callable[[httpx.Request], httpx.Response]) -> MetricsSyncClient:
    return MetricsSyncClient(ENDPOINT, A_TOKEN, transport=httpx.MockTransport(handler))


def ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"accepted": 1, "duplicate_batch": False})


# --- What goes out -----------------------------------------------------------


async def test_the_batch_is_posted_to_the_configured_endpoint() -> None:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok(request)

    await client_over(record).send([a_row()])

    assert seen[0].method == "POST"
    assert str(seen[0].url) == ENDPOINT


async def test_the_site_token_travels_as_a_bearer_credential() -> None:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok(request)

    await client_over(record).send([a_row()])

    assert seen[0].headers["Authorization"] == f"Bearer {A_TOKEN}"


async def test_an_empty_batch_is_not_sent_at_all() -> None:
    """Nothing to store and a key for the cloud to remember. The loop should never call
    this with nothing, and if it does the right answer is silence, not a request."""

    def refuse(request: httpx.Request) -> httpx.Response:
        pytest.fail("an empty batch should not reach the network")

    await client_over(refuse).send([])


# --- How it fails ------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 500, 503])
async def test_any_non_2xx_is_an_error_rather_than_a_silent_success(status: int) -> None:
    """The caller must never stamp rows it did not confirm, so every one of these has to
    raise rather than return."""
    with pytest.raises(SyncError):
        await client_over(lambda _: httpx.Response(status)).send([a_row()])


async def test_a_throttle_carries_how_long_to_wait() -> None:
    """The cloud's `Retry-After` is the whole backpressure signal; discarding it would
    turn a polite refusal into a retry loop."""
    response = httpx.Response(429, headers={"Retry-After": "42"})

    with pytest.raises(ThrottledError) as raised:
        await client_over(lambda _: response).send([a_row()])

    assert raised.value.retry_after_s == 42


async def test_a_throttle_without_a_header_still_raises() -> None:
    with pytest.raises(ThrottledError) as raised:
        await client_over(lambda _: httpx.Response(429)).send([a_row()])

    assert raised.value.retry_after_s is None


async def test_a_network_failure_is_a_sync_error_not_a_crash() -> None:
    """Offline is the expected state, not an exceptional one — it must arrive as the same
    error type as a refusal so the loop has one thing to handle."""

    def unreachable(request: httpx.Request) -> httpx.Response:
        msg = "network is unreachable"
        raise httpx.ConnectError(msg, request=request)

    with pytest.raises(SyncError):
        await client_over(unreachable).send([a_row()])


# --- What must never be said out loud ----------------------------------------


async def test_the_token_never_reaches_a_log_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A site token is a credential. This endpoint fails routinely — offline is normal —
    so a failure path that logged the request would write it out on every reconnect."""
    with caplog.at_level(logging.DEBUG, logger="mesopic"), pytest.raises(SyncError):
        await client_over(lambda _: httpx.Response(401)).send([a_row()])

    assert A_TOKEN not in caplog.text


async def test_an_error_message_does_not_carry_the_token() -> None:
    """Exception text reaches logs and tracebacks by routes nobody planned."""
    with pytest.raises(SyncError) as raised:
        await client_over(lambda _: httpx.Response(500)).send([a_row()])

    assert A_TOKEN not in str(raised.value)


# --- Transport posture -------------------------------------------------------


async def test_certificate_verification_is_never_disabled() -> None:
    """The channel carries a bearer token to a public endpoint. `verify=False` here would
    hand it to whoever holds the network."""
    plain = MetricsSyncClient(ENDPOINT, A_TOKEN)
    assert plain.verify is True


async def test_a_request_cannot_hang_forever() -> None:
    """No timeout means a stalled uplink parks the sync task for the process lifetime, and
    `unsynced_count` climbs with nothing in the log to explain it."""
    plain = MetricsSyncClient(ENDPOINT, A_TOKEN)
    assert plain.timeout_s > 0


async def test_a_plaintext_endpoint_is_refused() -> None:
    """Config validation already requires https, so this is the second lock on the same
    door — it stops a caller constructing the client directly from doing what the YAML
    cannot."""
    with pytest.raises(SyncError):
        MetricsSyncClient("http://cloud.example/ingest", A_TOKEN)
