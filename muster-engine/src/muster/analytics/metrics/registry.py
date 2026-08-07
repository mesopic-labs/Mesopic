"""The metric plugin seam.

Implements P2.4.
"""

from __future__ import annotations

from typing import Protocol

from muster.types import MetricName, MetricRow, MinuteBucket, RawEvent


class MetricPlugin(Protocol):
    """Reduces a bucket's raw events into metric rows."""

    @property
    def name(self) -> MetricName:
        """The metric this plugin produces."""
        ...

    def reduce(self, events: list[RawEvent], bucket: MinuteBucket) -> list[MetricRow]:
        """Fold one minute's events into rows. Must be deterministic and replay-safe."""
        ...


class MetricRegistry:
    """Holds the enabled plugins for a site."""

    def register(self, plugin: MetricPlugin) -> None:
        """Add a plugin. Registering the same metric twice is an error."""
        raise NotImplementedError

    def reduce_all(self, events: list[RawEvent], bucket: MinuteBucket) -> list[MetricRow]:
        """Run every registered plugin over one bucket's events."""
        raise NotImplementedError
