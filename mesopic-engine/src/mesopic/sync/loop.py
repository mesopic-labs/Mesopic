"""Deciding what to send and when — the half of the sync client that is not HTTP.

Split from `client.py` so the draining logic is testable without a transport and a
transport swap does not touch it (ADR-0010 rejected MQTT for this leg rather than making
it impossible).

The loop is where "video never leaves the site" stops being a promise. It is typed against
`MetricsSource`, a protocol with three metrics methods and nothing else, so giving this
component access to a frame is not an oversight anyone can make in passing — it means
widening a protocol whose entire purpose is to be narrow (ADR-0005).

Implements part of C5.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Protocol

from mesopic.errors import SyncError
from mesopic.sync.client import ThrottledError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mesopic.sync.client import MetricsSyncClient
    from mesopic.types import MetricRow

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S: Final = 30.0
"""How often a healthy site syncs. Minute buckets mean there is nothing new more often
than once a minute, so this is already twice as eager as the data warrants."""

DEFAULT_BATCH_SIZE: Final = 500
"""Roughly four hours of a busy multi-camera site. Small enough not to saturate a shop's
uplink in one POST, large enough that a day-long outage drains in a handful of rounds."""

MAX_BACKOFF_S: Final = 900.0
"""Fifteen minutes. An outage longer than that is not going to be fixed by asking more
often, and the rows are safe in SQLite meanwhile."""

_BACKOFF_FACTOR: Final = 2.0
_JITTER = 0.25
"""A quarter, so a fleet that lost the same uplink does not return in lockstep."""


class MetricsSource(Protocol):
    """What the sync path is allowed to read: three methods, all of them metrics.

    This is deliberately not `Store`. `Store` can also read raw events and heatmap blobs,
    and a component that holds one has to be trusted not to; a component that holds this
    cannot, because there is no method to call. The narrowness is the feature — do not add
    to it without reading ADR-0005 first.
    """

    def unsynced_metrics(self, limit: int) -> list[MetricRow]: ...

    def mark_synced(self, rows: Sequence[MetricRow], synced_at: datetime) -> None: ...

    def unsynced_count(self) -> int: ...


async def drain_once(
    source: MetricsSource,
    client: MetricsSyncClient,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    now: Callable[[], datetime] | None = None,
) -> int:
    """Send at most one batch. Returns how many rows the cloud confirmed.

    Zero means either nothing to send or nothing delivered, and the caller cannot tell
    them apart — deliberately, because the response to both is the same: wait. What it
    must not do is stamp, and the only `mark_synced` in this module sits after an
    exception-free `send`.
    """
    stamp = _utcnow if now is None else now
    rows = source.unsynced_metrics(batch_size)
    if not rows:
        return 0
    await client.send(rows)
    # Only reachable when `send` did not raise, which is only when the cloud answered 2xx.
    source.mark_synced(rows, stamp())
    return len(rows)


class SyncLoop:
    """Drain, wait, drain again — widening the wait when the cloud is unhappy.

    Failure here is ordinary. A shop's uplink drops, a router reboots, the cloud asks for
    quiet; none of it is exceptional and none of it should reach the engine, which is why
    `run` swallows `SyncError` rather than propagating it.
    """

    def __init__(
        self,
        source: MetricsSource,
        client: MetricsSyncClient,
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sleep: Callable[[float], object] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._source = source
        self._client = client
        self._interval_s = interval_s
        self._batch_size = batch_size
        self._sleep = sleep
        self._now = now
        self._backoff_s = interval_s

    async def run(self, *, iterations: int | None = None) -> None:
        """Loop until cancelled, or for a fixed number of rounds in a test.

        `iterations` exists so the loop's scheduling can be asserted without a real clock;
        production passes nothing and the loop ends only by cancellation.
        """
        rounds = 0
        while iterations is None or rounds < iterations:
            await self._wait(await self._one_round())
            rounds += 1

    async def _one_round(self) -> float:
        """Do one drain and return how long to wait before the next."""
        try:
            sent = await drain_once(
                self._source, self._client, batch_size=self._batch_size, now=self._now
            )
        except ThrottledError as throttle:
            # The cloud named a number. Preferring our own backoff over it would be
            # ignoring the one piece of information the other side actually has.
            wait = throttle.retry_after_s or self._widen()
            logger.info("cloud sync throttled; next attempt in %.0fs", wait)
            return wait
        except SyncError:
            wait = self._widen()
            # `info`, not `error`: an unreachable cloud is the expected state of a shop
            # with a flaky uplink, and logging it as a fault trains people to ignore the
            # log. `unsynced_count` is the signal worth alarming on.
            logger.info("cloud sync unavailable; next attempt in %.0fs", wait)
            return wait
        if sent:
            logger.info("cloud sync delivered %d rows", sent)
        self._backoff_s = self._interval_s
        return self._interval_s

    def _widen(self) -> float:
        """Exponential with jitter, capped. Jitter because a whole fleet loses the same
        uplink at the same moment and must not come back in lockstep."""
        self._backoff_s = min(MAX_BACKOFF_S, self._backoff_s * _BACKOFF_FACTOR)
        # `random` and not `secrets`: this is scheduling, not a credential, and the
        # security ruleset's objection does not apply to spreading retries.
        return self._backoff_s * (1 + random.uniform(0, _JITTER))  # noqa: S311

    async def _wait(self, seconds: float) -> None:
        if self._sleep is not None:
            self._sleep(seconds)
            return
        await asyncio.sleep(seconds)


def _utcnow() -> datetime:
    return datetime.now(UTC)
