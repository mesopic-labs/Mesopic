"""Tracks plus `SiteGeometry` become `RawEvent`s.

Pure in the sense engine-architecture.md §8 means it: no camera, no model, no database,
no clock. Everything this module decides, it decides from the tracks it has been shown
and the geometry it was built with, which is what makes it testable against a recorded
track sequence alone.

It is *not* stateless, and cannot be. A crossing is a property of two consecutive
foot-points, the hysteresis band is a property of what the track did last, and a zone
enter is the difference between two membership sets — so the per-track bookkeeping
algorithms.md §5(c), §5(d) and §6 describe lives here, keyed by `(camera, track)` because
a `TrackId` is unique per camera per run and nothing more.

The math is imported, not re-derived: containment and `side_of` come from
`site_geometry`, and the segment-intersection test below is built on that same `side_of`
so its tie-break for a zero side cannot drift from the direction test beside it — which
is the failure §5(b) explicitly warns about.

Implements P2.3.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from muster.analytics.site_geometry import PreparedLine, SiteGeometry, side_of
from muster.types import (
    CameraId,
    EventKind,
    FrameTs,
    LineId,
    NormPoint,
    RawEvent,
    Track,
    TrackId,
    ZoneId,
)

HYSTERESIS_DELTA = 0.02
"""How far past a line a foot-point must get before a re-cross counts, in normalized
units (algorithms.md §5(d), "small, e.g. 0.02"). The band is the primary defence against
a person loitering on the threshold; the timer below is the backstop."""

CROSSING_DEBOUNCE_S = 1.0
"""How long the same track is ignored on the same line after crossing it (§5(d))."""

DWELL_MIN_S = 3.0
"""How long a track must be *continuously* inside a zone before it counts (§6, §7).

The same threshold the dwell state machine uses, applied to the live count — one rule,
two consumers. Confirmation here is deliberately not the aggregator's dwell machine: it
has no re-entry grace, because §6 asks for continuous residency and a round trip out of
the zone genuinely restarts it.
"""


@dataclass(slots=True)
class _LineState:
    """Per `(track, line)`: the sticky side and the debounce window."""

    sticky_side: int | None = None
    """Last *non-zero* side. A graze must not move it, or one step onto the line and
    back becomes two half-crossings (§5c)."""

    debounced_until: datetime | None = None


@dataclass(slots=True)
class _Residency:
    """Per `(zone, track)`: when this stay started, and whether it has been confirmed."""

    entered_at: FrameTs
    confirmed: bool = False


@dataclass(slots=True)
class _TrackState:
    """Per track: where it last was, and whether that was an observation or a guess."""

    last_observed_foot: NormPoint | None = None
    was_observed: bool = False
    lines: dict[LineId, _LineState] = field(default_factory=dict)


class GeometryAnalytics:
    """Turns each tick's tracks into raw events for one site.

    Holds the per-track state the crossing and membership tests need. One instance per
    engine; `on_tracks` is called once per camera per tick.
    """

    def __init__(
        self,
        geometry: SiteGeometry,
        *,
        hysteresis_delta: float = HYSTERESIS_DELTA,
        crossing_debounce_s: float = CROSSING_DEBOUNCE_S,
        dwell_min_s: float = DWELL_MIN_S,
    ) -> None:
        self._geometry = geometry
        self._hysteresis_delta = hysteresis_delta
        self._crossing_debounce_s = crossing_debounce_s
        self._dwell_min_s = timedelta(seconds=dwell_min_s)
        self._tracks: dict[tuple[CameraId, TrackId], _TrackState] = {}
        self._inside: dict[ZoneId, dict[TrackId, _Residency]] = {}
        self._last_ts: dict[CameraId, datetime] = {}

    def on_tracks(
        self, camera_id: CameraId, tracks: Sequence[Track], *, ts: FrameTs | None = None
    ) -> list[RawEvent]:
        """Derive this tick's events for one camera. No I/O, no persistence.

        `ts` is the sampled frame's capture time. It is optional only because a tick that
        carries tracks can take its timestamp from the newest of them — but a tick with
        *no* tracks cannot, and an empty camera still has to report that its zones held
        nobody. Pass it wherever the caller knows it, which is everywhere real.
        """
        for track in tracks:
            if track.camera_id != camera_id:
                msg = (
                    f"track {track.track_id} belongs to camera "
                    f"{track.camera_id!r}, not {camera_id!r}"
                )
                raise ValueError(msg)

        previous_ts = self._last_ts.get(camera_id)
        tick_ts = self._tick_ts(camera_id, tracks, ts)
        events = self._line_events(camera_id, tracks)
        events += self._zone_events(camera_id, tracks, tick_ts)
        events += self._occupancy_samples(camera_id, tick_ts, previous_ts)
        self._forget_dead_tracks(camera_id, tracks)
        return events

    def _tick_ts(
        self, camera_id: CameraId, tracks: Sequence[Track], ts: FrameTs | None
    ) -> datetime | None:
        """This tick's capture time — the caller's, the newest track's, or the last seen.

        A tick with no tracks still has to stamp the exits it produces, and this module
        has no clock by design. The last capture time is the honest answer: it is when
        the departing track was last actually seen.
        """
        if ts is not None:
            self._last_ts[camera_id] = ts
        elif tracks:
            self._last_ts[camera_id] = max(track.ts for track in tracks)
        return self._last_ts.get(camera_id)

    # --- Lines --------------------------------------------------------------

    def _line_events(self, camera_id: CameraId, tracks: Sequence[Track]) -> list[RawEvent]:
        lines = self._geometry.lines_for(camera_id)
        events: list[RawEvent] = []
        for track in tracks:
            state = self._tracks.setdefault((camera_id, track.track_id), _TrackState())
            observed = track.time_since_update == 0
            if observed or state.was_observed:
                events += [
                    event
                    for line in lines
                    if (event := self._crossing(track, line, state)) is not None
                ]
            self._remember(state, track, observed=observed)
        return events

    def _crossing(self, track: Track, line: PreparedLine, state: _TrackState) -> RawEvent | None:
        """One track against one line, following algorithms.md §5 step for step."""
        line_state = state.lines.setdefault(line.line_id, _LineState())
        current_side = line.side_of(track.foot_point)

        previous = state.last_observed_foot
        if previous is None:
            # First sight of this track: seed the side it started on. Without this the
            # track's first real crossing has nothing to flip from and is swallowed.
            self._update_sticky(line_state, current_side)
            return None

        if not _segments_intersect(previous, track.foot_point, line.a, line.b):
            self._update_sticky(line_state, current_side)
            return None

        sticky = line_state.sticky_side
        if current_side == 0 or sticky is None or (current_side > 0) == (sticky > 0):
            self._update_sticky(line_state, current_side)
            return None

        if line.distance_to(track.foot_point) < self._hysteresis_delta:
            return None
        if line_state.debounced_until is not None and track.ts < line_state.debounced_until:
            return None

        line_state.sticky_side = current_side
        line_state.debounced_until = track.ts + timedelta(seconds=self._crossing_debounce_s)
        return RawEvent(
            camera_id=track.camera_id,
            ts=track.ts,
            kind=EventKind.LINE_CROSS,
            track_id=track.track_id,
            line_id=line.line_id,
            direction=1 if sticky < 0 else -1,
            is_staff=track.is_staff,
        )

    @staticmethod
    def _update_sticky(line_state: _LineState, side: int) -> None:
        """Zero is not a side. Holding through it is what makes a graze silent (§5c)."""
        if side != 0:
            line_state.sticky_side = side

    # --- Zones --------------------------------------------------------------

    def _zone_events(
        self, camera_id: CameraId, tracks: Sequence[Track], tick_ts: datetime | None
    ) -> list[RawEvent]:
        """Membership diffing, which is also the single owner of track-death -> exit.

        algorithms.md §7's dwell state machine depends on that: a track that dies inside a
        zone is absent from `tracks`, so it falls out of `now_inside` and closes its dwell
        normally. Without it `open_dwells` leaks forever.
        """
        events: list[RawEvent] = []
        coasted = {t.track_id for t in tracks if t.time_since_update > 0}
        by_id = {track.track_id: track for track in tracks}

        for zone in self._geometry.zones_for(camera_id):
            residents = self._inside.setdefault(zone.zone_id, {})
            was_inside = set(residents)
            now_inside = {
                track.track_id
                for track in tracks
                if track.time_since_update == 0 and zone.contains(track.foot_point)
            }
            # A dead-reckoned position may neither create nor destroy membership (§3.4).
            now_inside |= was_inside & coasted

            events += [
                _zone_event(EventKind.ZONE_ENTER, zone.zone_id, by_id[track_id])
                for track_id in sorted(now_inside - was_inside)
            ]
            events += [
                _zone_event(
                    EventKind.ZONE_EXIT,
                    zone.zone_id,
                    by_id.get(track_id),
                    departed=(camera_id, track_id, tick_ts),
                )
                for track_id in sorted(was_inside - now_inside)
            ]
            self._inside[zone.zone_id] = self._residents_after(
                residents, now_inside, by_id, tick_ts
            )
            events += self._confirmations(camera_id, zone.zone_id, tick_ts)
        return events

    @staticmethod
    def _residents_after(
        residents: dict[TrackId, _Residency],
        now_inside: set[TrackId],
        by_id: dict[TrackId, Track],
        tick_ts: datetime | None,
    ) -> dict[TrackId, _Residency]:
        """Carry the residency of everyone who stayed; start a clock for everyone new.

        A track that left is simply dropped, which is what restarts its confirmation
        clock on a re-entry — §6 asks for *continuous* residency, and a round trip out of
        the zone is not continuous however brief it was.
        """
        kept: dict[TrackId, _Residency] = {}
        for track_id in now_inside:
            existing = residents.get(track_id)
            if existing is not None:
                kept[track_id] = existing
                continue
            track = by_id.get(track_id)
            entered_at = track.ts if track is not None else tick_ts
            if entered_at is not None:
                kept[track_id] = _Residency(entered_at=FrameTs(entered_at))
        return kept

    def _confirmations(
        self, camera_id: CameraId, zone_id: ZoneId, tick_ts: datetime | None
    ) -> list[RawEvent]:
        """Announce, once, that a stay has outlasted `dwell_min_s`.

        The occupancy sample carries how many tracks are confirmed but not *which*, and
        zone-derived footfall counts distinct confirmed entries (§7) — so the identity
        has to arrive as its own event or not at all.
        """
        if tick_ts is None:
            return []
        events = []
        for track_id, residency in sorted(self._inside[zone_id].items()):
            if residency.confirmed or tick_ts - residency.entered_at < self._dwell_min_s:
                continue
            residency.confirmed = True
            events.append(
                RawEvent(
                    camera_id=camera_id,
                    ts=FrameTs(tick_ts),
                    kind=EventKind.ZONE_CONFIRMED,
                    track_id=track_id,
                    zone_id=zone_id,
                )
            )
        return events

    # --- Sampled state ------------------------------------------------------

    def _occupancy_samples(
        self, camera_id: CameraId, tick_ts: datetime | None, previous_ts: datetime | None
    ) -> list[RawEvent]:
        """One count per zone per tick, carrying the interval it stands for.

        Emitted whether or not anything changed, because that is the entire point: a
        reducer fed only transitions reports nothing for a minute in which nobody moved,
        and "nobody moved" is not the same fact as "the camera was down".

        Nothing is emitted for a tick that did not advance the clock — the first tick of
        a run, or a repeated timestamp. A zero-width interval weights nothing and would
        only inflate `sample_count` into false confidence (algorithms.md §6.2).
        """
        if tick_ts is None or previous_ts is None or tick_ts <= previous_ts:
            return []
        dt_s = (tick_ts - previous_ts).total_seconds()
        samples = []
        for zone in self._geometry.zones_for(camera_id):
            residents = self._inside.get(zone.zone_id, {})
            samples.append(
                RawEvent(
                    camera_id=camera_id,
                    ts=FrameTs(tick_ts),
                    kind=EventKind.OCCUPANCY_SAMPLE,
                    track_id=None,
                    zone_id=zone.zone_id,
                    value=float(len(residents)),
                    confirmed_value=float(
                        sum(1 for residency in residents.values() if residency.confirmed)
                    ),
                    dt_s=dt_s,
                )
            )
        return samples

    # --- Bookkeeping --------------------------------------------------------

    @staticmethod
    def _remember(state: _TrackState, track: Track, *, observed: bool) -> None:
        """Only an observation moves the anchor a crossing is measured from.

        That is what recovers a crossing hidden by an occlusion: on re-match the segment
        runs from the last *real* position to the re-found one, so the crossing is
        evaluated once against evidence rather than against the Kalman filter's guess
        (§3.4(4)).
        """
        if observed:
            state.last_observed_foot = track.foot_point
        state.was_observed = observed

    def _forget_dead_tracks(self, camera_id: CameraId, tracks: Sequence[Track]) -> None:
        """A `TrackId` is unique per run only, so state for a dead track is a leak."""
        alive = {track.track_id for track in tracks}
        for key in [k for k in self._tracks if k[0] == camera_id and k[1] not in alive]:
            del self._tracks[key]


def _zone_event(
    kind: EventKind,
    zone_id: ZoneId,
    track: Track | None,
    *,
    departed: tuple[CameraId, TrackId, datetime | None] | None = None,
) -> RawEvent:
    """Build a zone event, for a track that may already be gone.

    An exit is routinely emitted for a track that has left the tick's list entirely — a
    death inside the zone is how a dwell closes — so its identity and timestamp come from
    `departed` rather than from a `Track` that no longer exists.
    """
    if track is not None:
        return RawEvent(
            camera_id=track.camera_id,
            ts=track.ts,
            kind=kind,
            track_id=track.track_id,
            zone_id=zone_id,
            is_staff=track.is_staff,
        )
    if departed is None:  # pragma: no cover - the caller always supplies one of the two
        msg = "a zone event needs either a live track or the identity of a departed one"
        raise ValueError(msg)
    camera_id, track_id, ts = departed
    if ts is None:  # pragma: no cover - a track can only depart after it was once seen
        msg = f"no capture time to stamp the exit of track {track_id}"
        raise ValueError(msg)
    return RawEvent(
        camera_id=camera_id, ts=FrameTs(ts), kind=kind, track_id=track_id, zone_id=zone_id
    )


def _segments_intersect(p1: NormPoint, p2: NormPoint, a: NormPoint, b: NormPoint) -> bool:
    """Do segments `p1->p2` and `a->b` properly cross (algorithms.md §5(b))?

    Degenerate cases are resolved rather than left to the reader: `(d > 0)` folds a zero
    into the negative side, the same tie-break the sticky side uses, so a graze produces
    no crossing on the grazing frame and a normal one on the frame it commits.
    """
    d1 = side_of(a, b, p1)
    d2 = side_of(a, b, p2)
    d3 = side_of(p1, p2, a)
    d4 = side_of(p1, p2, b)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))
