"""Load-shedding on the worker side: a local outbox and the fps policy above it.

§9's rule is that backpressure flows upstream to the *sampler*, never to the decoder and
never to the store. When the supervisor falls behind, the response is to sample fewer
frames — which means decoding and inferring less, which is where the CPU is going. The
alternative, dropping events at the boundary, sheds the cheap tail work and keeps paying
for the expensive part.

The outbox exists because a producer cannot drop the *oldest* item from a
`multiprocessing.Queue` it is writing to — only the consumer sees that end. Buffering
locally in a bounded deque gives "drop oldest" its natural meaning and keeps a momentary
stall from costing counts at all.

Implements P2.7 (engine-architecture.md §9).
"""

from __future__ import annotations

import queue
from collections import deque
from typing import Protocol, TypeVar

T_contra = TypeVar("T_contra", contravariant=True)


class Sink(Protocol[T_contra]):
    """The write end of the worker's event queue.

    Generic and contravariant so a `multiprocessing.Queue[RawEvent]` satisfies it: a
    queue that accepts `RawEvent` is not a queue that accepts `object`, and typing it
    as the latter would let a frame past the type checker.
    """

    def put_nowait(self, item: T_contra, /) -> None: ...


class Sampler(Protocol):
    """The half of `FrameSampler` backpressure touches (P1.3 added it for this)."""

    def set_target_fps(self, target_fps: float) -> None: ...


class Outbox[T]:
    """A bounded local buffer in front of the event queue.

    Bounded rather than growing: an unbounded outbox turns "the supervisor is slow" into
    "the worker runs out of memory", which is the death spiral the queue bound exists to
    prevent in the first place.
    """

    def __init__(self, maxlen: int) -> None:
        self._pending: deque[T] = deque(maxlen=maxlen)
        self._blocked = False
        self.dropped = 0
        """Events lost to overflow. A degraded metric — logged and visible, never silent."""

    def push(self, event: T) -> None:
        if len(self._pending) == self._pending.maxlen:
            # `deque` would evict the oldest silently; counting it is the difference
            # between a degraded metric and a mystery.
            self.dropped += 1
        self._pending.append(event)

    def flush(self, sink: Sink[T]) -> int:
        """Move as much as the sink will take. Returns how many crossed."""
        sent = 0
        while self._pending:
            item = self._pending[0]
            try:
                sink.put_nowait(item)
            except queue.Full:
                self._blocked = True
                return sent
            self._pending.popleft()
            sent += 1
        self._blocked = False
        return sent

    @property
    def pending(self) -> int:
        return len(self._pending)

    @property
    def under_pressure(self) -> bool:
        """True when the sink refused, which is the signal the sampler acts on."""
        return self._blocked


class Backpressure:
    """Maps outbox pressure onto the sampler's target frame rate."""

    SHED_FACTOR = 0.8
    RECOVER_FACTOR = 1.25
    RELIEF_TICKS = 10
    """How many clear observations before climbing back. Recovering on the first one
    oscillates: shed, recover, shed, recover, at the period of the queue."""

    def __init__(
        self, sampler: Sampler, *, fps_min: float, fps_max: float, start_fps: float
    ) -> None:
        self._sampler = sampler
        self._fps_min = fps_min
        self._fps_max = fps_max
        self._clear_ticks = 0
        self.target_fps = start_fps

    def observe(self, *, under_pressure: bool) -> None:
        if under_pressure:
            self._clear_ticks = 0
            self._retarget(max(self._fps_min, self.target_fps * self.SHED_FACTOR))
            return
        self._clear_ticks += 1
        if self._clear_ticks < self.RELIEF_TICKS:
            return
        self._clear_ticks = 0
        self._retarget(min(self._fps_max, self.target_fps * self.RECOVER_FACTOR))

    def _retarget(self, target_fps: float) -> None:
        """Only tell the sampler when the number actually moved."""
        if target_fps == self.target_fps:
            return
        self.target_fps = target_fps
        self._sampler.set_target_fps(target_fps)
