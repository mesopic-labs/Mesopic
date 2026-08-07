"""Frigate as an upstream detector, over MQTT (ADR-0006).

This source decodes nothing. On a box already running Frigate we subscribe to its object
topics and adapt its detections to the same shape as every other path, skipping our own
decode+detect entirely — the largest free performance win available on the reference box.

We talk to Frigate across an API boundary and neither bundle nor derive from its code:
that boundary is what keeps Frigate's AGPL away from the MIT engine.

Implements P4.3.
"""

from __future__ import annotations

from collections.abc import Iterator

from muster.types import CameraId, Track


class FrigateTrackSource:
    """Adapts Frigate's MQTT object events to `Track`s.

    Note this yields `Track`s, not `DecodedFrame`s: Frigate has already done the detect
    (and often the track), so this source joins the pipeline downstream of the detector.
    """

    def __init__(self, camera_id: CameraId, broker: str, topic: str) -> None:
        raise NotImplementedError

    def tracks(self) -> Iterator[Track]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
