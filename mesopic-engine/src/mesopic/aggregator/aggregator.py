"""Fold raw events into minute buckets.

Two properties this module exists to guarantee:

* **Bucketing is by capture time, UTC** — never wall time. A frame processed late still
  lands in the minute it was captured in, which is what makes the metric honest and what
  makes replay deterministic.
* **Writes are idempotent** on the natural key `(camera_id, metric, scope_id, bucket)`.
  A mid-minute crash and restart must not double-count.

Open dwells and live occupancy are carried in memory and resolve into the bucket they
end in.

Closing a bucket does not seal it. A dwell that ends at 09:30:59 is only *known* once its
grace window lapses at 09:31:01, and a reconnect can deliver a whole minute's backlog
late — so a bucket can be re-folded at any time and the fold is deterministic over the
events it holds. The store's upsert is what makes re-folding harmless; `forget_before` is
what stops the kept events growing without bound.

Implements P2.6.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from mesopic.analytics.metrics.heatmap import HeatmapAccumulator
from mesopic.analytics.metrics.registry import MetricRegistry
from mesopic.types import (
    CameraId,
    EventKind,
    FrameTs,
    HeatmapRow,
    MetricRow,
    MinuteBucket,
    RawEvent,
    TrackId,
    ZoneId,
)

EXIT_GRACE_S = 2.0
"""How long a zone exit stays provisional before the dwell closes (algorithms.md §7).

Someone stepping behind a pillar should be one dwell, not two. Expert-level in the doc's
parameter table, so it is a constructor argument rather than a `mesopic.yaml` key.
"""

_DWELL_KINDS = frozenset({EventKind.ZONE_ENTER, EventKind.ZONE_EXIT})
"""The only kinds that move the dwell state machine.

Named rather than implied, because the zone events are no longer the only ones carrying a
`zone_id`: an occupancy sample names a zone and no track at all (ADR-0016).
"""

_DwellKey = tuple[CameraId, ZoneId, TrackId]


@dataclass(slots=True)
class _DwellState:
    """One open dwell. `pending_exit_ts` is set on exit and cleared by a re-entry."""

    entered_at: FrameTs
    pending_exit_ts: FrameTs | None = None
    is_staff: bool = False
    """Carried from the entry that opened this dwell.

    A completed dwell is synthesised here rather than by geometry, so without this the
    sample is built fresh with the default and every dwell in the building reads as a
    customer's — including the eight hours a barista spends behind the counter, which is
    the single largest thing the staff split exists to remove.
    """


def bucket_of(ts: datetime) -> MinuteBucket:
    """Floor a capture time to its minute. The one place the grain is decided."""
    return MinuteBucket(ts.replace(second=0, microsecond=0))


class Aggregator:
    """Reduces the event stream into durable metric rows."""

    def __init__(
        self,
        registry: MetricRegistry,
        *,
        dwell_min_s: float,
        exit_grace_s: float = EXIT_GRACE_S,
        heatmaps: HeatmapAccumulator | None = None,
    ) -> None:
        self._registry = registry
        self._heatmaps = heatmaps
        self._dwell_min_s = dwell_min_s
        self._exit_grace = timedelta(seconds=exit_grace_s)
        self._buckets: dict[MinuteBucket, list[RawEvent]] = defaultdict(list)
        self._open: dict[_DwellKey, _DwellState] = {}

    # --- Ingest -------------------------------------------------------------

    def ingest(self, event: RawEvent) -> None:
        """Accept one event. Cheap: the fold happens on bucket close.

        The flush comes *first*: this event's timestamp is what advances the clock, and a
        dwell whose grace lapsed before it has to close at its own exit time rather than
        be mistaken for the thing this event continues.
        """
        self._buckets[bucket_of(event.ts)].append(event)
        self.flush(event.ts)
        self._track_dwell(event)

    def flush(self, now: FrameTs) -> None:
        """Close any dwell whose grace window has lapsed as of `now`.

        Called on every ingest and, by the supervisor, on a tick — a zone the last person
        left is otherwise waiting on an event that will never come.
        """
        for key, state in list(self._open.items()):
            if state.pending_exit_ts is None or now - state.pending_exit_ts <= self._exit_grace:
                continue
            del self._open[key]
            completed = self._completed_dwell(key, state)
            if completed is not None:
                self._buckets[bucket_of(completed.ts)].append(completed)

    def _track_dwell(self, event: RawEvent) -> None:
        """algorithms.md §7's state machine, kept in memory and lost on restart (§10)."""
        if event.kind not in _DWELL_KINDS or event.zone_id is None or event.track_id is None:
            return
        key = (event.camera_id, event.zone_id, event.track_id)
        state = self._open.get(key)

        if event.kind is EventKind.ZONE_ENTER:
            if state is None:
                self._open[key] = _DwellState(entered_at=event.ts, is_staff=event.is_staff)
            elif state.pending_exit_ts is not None:
                # A re-entry against a still-open dwell bridges the gap. It can only be
                # inside the grace window: `ingest` flushed at this event's timestamp
                # first, so anything past its grace has already closed and been emitted.
                # §7's snippet re-tests the window here because it has no such ordering.
                state.pending_exit_ts = None
            # A duplicate enter for an already-open dwell does nothing. Restarting the
            # clock here would silently report a fraction of the real dwell — §7 names
            # this as a defect found in the v0 draft, because the output is merely wrong.
        elif event.kind is EventKind.ZONE_EXIT and state is not None:
            state.pending_exit_ts = event.ts

    def _completed_dwell(self, key: _DwellKey, state: _DwellState) -> RawEvent | None:
        """A dwell shorter than `dwell_min_s` produces no record at all (§7)."""
        camera_id, zone_id, track_id = key
        assert state.pending_exit_ts is not None  # noqa: S101 - flush only calls this when set
        seconds = (state.pending_exit_ts - state.entered_at).total_seconds()
        if seconds < self._dwell_min_s:
            return None
        return RawEvent(
            camera_id=camera_id,
            ts=state.pending_exit_ts,
            kind=EventKind.DWELL_SAMPLE,
            track_id=track_id,
            zone_id=zone_id,
            value=seconds,
            is_staff=state.is_staff,
        )

    # --- Fold ---------------------------------------------------------------

    def close_bucket(self, bucket: MinuteBucket) -> list[MetricRow]:
        """Finalise a minute and return its rows for the store to upsert.

        Deterministic over the events the bucket holds, so closing twice returns the same
        rows rather than doubled ones — the property that makes a crash mid-minute safe.
        """
        events = self._buckets.get(bucket)
        if not events:
            return []
        return self._registry.reduce_all(list(events), bucket)

    def close_grids(self, bucket: MinuteBucket) -> list[HeatmapRow]:
        """The same minute's heatmap grids, folded from the same retained events.

        Separate from `close_bucket` because the two produce different row types bound
        for different tables, and a `MetricRow` cannot carry a blob. Both fold the same
        retained events, so both are equally re-foldable — closing twice returns the same
        grids rather than doubled ones.

        Without an accumulator this is empty rather than an error: a site whose zones
        never asked for `heatmap` has no grids, which is not a failure to have any.
        """
        events = self._buckets.get(bucket)
        if not events or self._heatmaps is None:
            return []
        return self._heatmaps.fold(list(events), bucket)

    def retarget(self, registry: MetricRegistry, *, heatmaps: HeatmapAccumulator | None) -> None:
        """Adopt plugins built from edited geometry (P3.8).

        Open buckets keep their events and are reduced by the new registry when they
        close. That is the honest reading: a zone the operator has just drawn should
        count the people already standing in it, and one they deleted should stop
        counting mid-minute rather than emit a final partial value nobody asked for.

        `heatmaps` is not optional-by-omission: a caller that passed an accumulator once
        and forgets it here would silently stop accumulating on the first geometry edit,
        which is the kind of failure that shows up as a cold heatmap a week later.
        """
        self._registry = registry
        self._heatmaps = heatmaps

    def pending_buckets(self) -> list[MinuteBucket]:
        """Buckets holding events, oldest first. The supervisor's closing cursor."""
        return sorted(bucket for bucket, events in self._buckets.items() if events)

    def forget_before(self, bucket: MinuteBucket) -> None:
        """Drop buckets older than `bucket`, which can no longer be re-folded."""
        for stale in [held for held in self._buckets if held < bucket]:
            del self._buckets[stale]

    def open_dwells(self) -> int:
        """How many dwells are still open. Health signal, and a leak check in tests."""
        return len(self._open)
