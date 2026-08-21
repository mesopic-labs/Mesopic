"""One interface, so a fifth exporter is a plugin rather than a refactor."""

from __future__ import annotations

from typing import Protocol

from mesopic.types import MetricRow


class Exporter(Protocol):
    """Delivers metric rows somewhere outside the engine."""

    def start(self) -> None:
        """Open connections. Must not raise on a dead peer — retry in the background."""
        ...

    def on_metric(self, row: MetricRow) -> None:
        """Handle one committed bucket. Push exporters deliver; pull exporters cache."""
        ...

    def shutdown(self) -> None:
        """Flush and close."""
        ...
