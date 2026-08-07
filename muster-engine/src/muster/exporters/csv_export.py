"""Flat CSV files, one per metric per day, append-only with daily rotation.

For spreadsheet users and offline analysis. UTC ISO timestamps throughout.

Implements part of P3.5.
"""

from __future__ import annotations

from pathlib import Path

from muster.types import MetricRow


class CsvExporter:
    """Writes committed buckets to daily-rotated CSV files."""

    def __init__(self, directory: Path) -> None:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def on_metric(self, row: MetricRow) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError
