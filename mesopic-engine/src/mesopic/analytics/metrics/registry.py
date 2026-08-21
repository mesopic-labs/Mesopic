"""The metric plugin seam.

Implements P2.4.
"""

from __future__ import annotations

from typing import Protocol

from mesopic.types import MetricName, MetricRow, MinuteBucket, RawEvent


class MetricPlugin(Protocol):
    """Reduces a bucket's raw events into metric rows."""

    @property
    def names(self) -> frozenset[MetricName]:
        """Every metric this plugin produces.

        Plural because a metric concept is not always one series: occupancy is a peak
        taken from the raw count and a mean taken from the confirmed one (algorithms.md
        §6.1), and both come from one fold over one sample stream. Splitting them across
        two plugins would read the same events twice and let the pair drift apart.
        """
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
        self._plugins: list[MetricPlugin] = []
        self._claimed: dict[MetricName, MetricPlugin] = {}

    def register(self, plugin: MetricPlugin) -> None:
        """Add a plugin. Claiming a metric another plugin already claims is an error.

        Two reducers writing one metric name would collide on the natural key
        `(camera_id, metric, scope_id, bucket)` — the later upsert silently replacing the
        earlier one, per bucket, forever.
        """
        for name in sorted(plugin.names):
            if name in self._claimed:
                msg = f"a plugin for metric {name.value!r} is already registered"
                raise ValueError(msg)
        self._plugins.append(plugin)
        self._claimed.update(dict.fromkeys(plugin.names, plugin))

    def metrics(self) -> frozenset[MetricName]:
        """Every metric this registry can produce. The site's declared vocabulary."""
        return frozenset(self._claimed)

    def reduce_all(self, events: list[RawEvent], bucket: MinuteBucket) -> list[MetricRow]:
        """Run every registered plugin over one bucket's events.

        Every plugin sees every event: filtering by kind is the plugin's own business,
        because a metric may legitimately need more than one kind (footfall reads line
        crossings *and* confirmed zone entries, algorithms.md §7).
        """
        rows: list[MetricRow] = []
        for plugin in self._plugins:
            rows += plugin.reduce(events, bucket)
        return rows
