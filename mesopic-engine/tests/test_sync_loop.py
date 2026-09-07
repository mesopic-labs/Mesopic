"""The draining loop: what it stamps, what it does not, and what it does when offline.

C5's acceptance criteria live here. They are written as the situations a shop actually
produces — a dropped uplink, a router reboot mid-request, a cloud asking for quiet — rather
than as calls to the functions under test.

Implements part of C5.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, get_type_hints

import httpx
import pytest

from mesopic.errors import SyncError
from mesopic.store import Store
from mesopic.sync.client import MetricsSyncClient
from mesopic.sync.loop import MetricsSource, SyncLoop, drain_once
from mesopic.types import CameraId, MetricName, MetricRow, MinuteBucket

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

pytestmark = pytest.mark.privacy

ENDPOINT = "https://cloud.example/ingest"
A_TOKEN = "msk_deadbeef_notreal"  # noqa: S105 - a fabricated token in a test, not a credential
START = datetime(2026, 7, 13, 9, 15, tzinfo=UTC)


def rows_from(count: int) -> list[MetricRow]:
    return [
        MetricRow(
            camera_id=CameraId("cam-door"),
            bucket=MinuteBucket(START + timedelta(minutes=n)),
            metric=MetricName.FOOTFALL,
            scope_id=None,
            value=float(n),
        )
        for n in range(count)
    ]


class FakeStore:
    """A store that holds rows and remembers which were stamped.

    Deliberately implements only the three methods the loop is allowed to reach — it would
    not satisfy `MetricsSource` if the loop needed anything else, which is the point.
    """

    def __init__(self, rows: list[MetricRow]) -> None:
        self.rows = rows
        self.stamped: list[MetricRow] = []
        self.stamps: list[datetime] = []

    def unsynced_metrics(self, limit: int) -> list[MetricRow]:
        pending = [row for row in self.rows if row not in self.stamped]
        return pending[:limit]

    def mark_synced(self, rows: Sequence[MetricRow], synced_at: datetime) -> None:
        self.stamped.extend(rows)
        self.stamps.append(synced_at)

    def unsynced_count(self) -> int:
        return len([row for row in self.rows if row not in self.stamped])


def client_over(handler: Callable[[httpx.Request], httpx.Response]) -> MetricsSyncClient:
    return MetricsSyncClient(ENDPOINT, A_TOKEN, transport=httpx.MockTransport(handler))


def _ignore(_seconds: float) -> None:
    """A sleep that does not, for the idle case where the delay is not what is asserted."""


def ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"accepted": 1, "duplicate_batch": False})


def offline(request: httpx.Request) -> httpx.Response:
    msg = "network is unreachable"
    raise httpx.ConnectError(msg, request=request)


# --- Offline is a non-event --------------------------------------------------


async def test_while_offline_nothing_is_stamped() -> None:
    """The first acceptance clause. Rows must survive an outage as unsynced rather than
    being lost or marked delivered to a cloud that never saw them.

    `drain_once` raises rather than returning zero: "the cloud did not confirm" and
    "there was nothing to send" must not be one value a caller can forget to check.
    """
    store = FakeStore(rows_from(3))

    with pytest.raises(SyncError):
        await drain_once(store, client_over(offline), batch_size=100)

    assert store.stamped == []
    assert store.unsynced_count() == 3


async def test_a_rejected_batch_stamps_nothing_either() -> None:
    """A 500 is not a delivery. Stamping here would lose the rows permanently — the edge
    is the only place they exist."""
    store = FakeStore(rows_from(3))

    with pytest.raises(SyncError):
        await drain_once(store, client_over(lambda _: httpx.Response(500)), batch_size=100)

    assert store.stamped == []


# --- Reconnect drains ---------------------------------------------------------


async def test_on_reconnect_rows_drain_and_are_stamped() -> None:
    store = FakeStore(rows_from(3))

    synced = await drain_once(store, client_over(ok), batch_size=100)

    assert synced == 3
    assert store.unsynced_count() == 0


async def test_rows_are_sent_in_bucket_order() -> None:
    """Order is the store's, and the loop must not disturb it: a batch that fails leaves
    the oldest rows unsynced rather than a hole in the middle of the series."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return ok(request)

    await drain_once(FakeStore(rows_from(3)), client_over(record), batch_size=100)

    buckets = [s["bucket_ts"] for s in seen[0]["samples"]]
    assert buckets == sorted(buckets)


async def test_a_batch_is_capped_at_the_configured_size() -> None:
    """Small and frequent beats one enormous POST on a shop's uplink (engine §14)."""
    store = FakeStore(rows_from(10))

    synced = await drain_once(store, client_over(ok), batch_size=4)

    assert synced == 4
    assert store.unsynced_count() == 6


# --- The lost 2xx, which is the case the whole design exists for -------------


async def test_a_retry_after_a_lost_ack_reuses_the_batch_key() -> None:
    """The cloud committed, the response never arrived, the edge tries again.

    The rows are still unsynced, so the retry re-reads exactly the same ones — and the key
    must come out identical, or the cloud's batch log cannot recognise the resend and the
    second idempotency layer does nothing.
    """
    keys: list[str] = []

    def lose_the_ack(request: httpx.Request) -> httpx.Response:
        keys.append(json.loads(request.content)["idempotency_key"])
        msg = "connection reset while reading the response"
        raise httpx.ReadError(msg, request=request)

    store = FakeStore(rows_from(3))
    for _ in range(2):
        with pytest.raises(SyncError):
            await drain_once(store, client_over(lose_the_ack), batch_size=100)

    assert len(keys) == 2
    assert keys[0] == keys[1]
    assert store.stamped == []


async def test_new_rows_arriving_between_attempts_change_the_key() -> None:
    """Not a flaw — the batch genuinely differs. The overlap is safe because the cloud
    upserts each row on its natural key, so a re-sent row overwrites itself."""
    keys: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        keys.append(json.loads(request.content)["idempotency_key"])
        return ok(request)

    store = FakeStore(rows_from(2))
    await drain_once(store, client_over(record), batch_size=100)
    store.rows.extend(rows_from(4)[2:])
    await drain_once(store, client_over(record), batch_size=100)

    assert keys[0] != keys[1]


# --- Backpressure ------------------------------------------------------------


async def test_the_loop_backs_off_when_the_cloud_is_unreachable() -> None:
    slept: list[float] = []
    store = FakeStore(rows_from(1))

    loop = SyncLoop(
        store,
        client_over(offline),
        interval_s=30.0,
        batch_size=100,
        sleep=slept.append,
    )
    await loop.run(iterations=3)

    assert slept == sorted(slept), "a backoff that does not grow is a retry storm"
    assert slept[-1] > 30.0


async def test_the_loop_returns_to_its_normal_interval_after_a_success() -> None:
    """A single blip must not leave a healthy site syncing every twenty minutes."""
    slept: list[float] = []
    store = FakeStore(rows_from(2))
    responses = iter([httpx.Response(503), httpx.Response(200, json={"accepted": 2})])

    loop = SyncLoop(
        store,
        client_over(lambda _: next(responses)),
        interval_s=30.0,
        batch_size=100,
        sleep=slept.append,
    )
    await loop.run(iterations=2)

    assert slept[-1] == 30.0


async def test_a_throttle_is_honoured_rather_than_guessed_at() -> None:
    slept: list[float] = []
    store = FakeStore(rows_from(1))
    throttle = httpx.Response(429, headers={"Retry-After": "90"})

    loop = SyncLoop(
        store,
        client_over(lambda _: throttle),
        interval_s=30.0,
        batch_size=100,
        sleep=slept.append,
    )
    await loop.run(iterations=1)

    assert slept == [90.0]


async def test_an_idle_site_does_not_post_an_empty_batch() -> None:
    """Nothing to say is not a reason to speak: an empty POST would still cost a request
    and ask the cloud to remember a batch key for nothing."""

    def refuse(request: httpx.Request) -> httpx.Response:
        pytest.fail("an idle loop should not reach the network")

    loop = SyncLoop(
        FakeStore([]), client_over(refuse), interval_s=30.0, batch_size=100, sleep=_ignore
    )
    await loop.run(iterations=2)


# --- The structural guarantee ------------------------------------------------


def test_the_loop_can_only_ask_the_store_for_metrics() -> None:
    """ADR-0005 made structural: "video never leaves the site" is a property of what this
    component is *able* to read.

    The loop is typed against `MetricsSource`, so widening it is what a reviewer would have
    to do to give the sync path a frame — and this is the test that makes that deliberate
    rather than incidental.
    """
    assert set(get_type_hints(MetricsSource).keys()) == set()
    exposed = {name for name in vars(MetricsSource) if not name.startswith("_")}
    assert exposed == {"unsynced_metrics", "mark_synced", "unsynced_count"}


def test_the_real_store_satisfies_the_narrow_source() -> None:
    """The protocol is only worth anything if the actual store still fits through it."""
    for name in ("unsynced_metrics", "mark_synced", "unsynced_count"):
        assert callable(getattr(Store, name))
