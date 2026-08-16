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
    """Holds the enabled plugins for a site.

    The mechanism only. The core six that will be registered into it are P2.4, and their
    arithmetic is checked against labelled footage rather than against this seam.
    """

    def __init__(self) -> None:
        self._plugins: dict[MetricName, MetricPlugin] = {}

    def register(self, plugin: MetricPlugin) -> None:
        """Add a plugin. Registering the same metric twice is an error.

        Two reducers writing one metric name would collide on the natural key
        `(camera_id, metric, scope_id, bucket)` — the later upsert silently replacing the
        earlier one, per bucket, forever.
        """
        if plugin.name in self._plugins:
            msg = f"a plugin for metric {plugin.name.value!r} is already registered"
            raise ValueError(msg)
        self._plugins[plugin.name] = plugin

    def reduce_all(self, events: list[RawEvent], bucket: MinuteBucket) -> list[MetricRow]:
        """Run every registered plugin over one bucket's events.

        Every plugin sees every event: filtering by kind is the plugin's own business,
        because a metric may legitimately need more than one kind (footfall reads line
        crossings *and* confirmed zone entries, algorithms.md §7).
        """
        rows: list[MetricRow] = []
        for plugin in self._plugins.values():
            rows += plugin.reduce(events, bucket)
        return rows
