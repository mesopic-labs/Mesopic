"""Flat CSV files, one per metric per day, append-only with daily rotation.

For spreadsheet users and offline analysis. UTC ISO timestamps throughout.

The day is **UTC**, matching the bucket it is derived from rather than the box's local
time. A local day would put a shop's evening on either side of a rotation depending on
where the box happens to sit, and would silently change behaviour twice a year.

Implements part of P3.5.
"""

from __future__ import annotations

import csv
from pathlib import Path

from muster.errors import ExportError
from muster.types import MetricRow

HEADER = ("bucket", "camera_id", "metric", "scope_id", "value", "staff_value", "sample_count")


class CsvExporter:
    """Writes committed buckets to daily-rotated CSV files."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def start(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)

    def on_metric(self, row: MetricRow) -> None:
        path = self._path_for(row)
        is_new = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(HEADER)
            writer.writerow(
                (
                    row.bucket.isoformat(),
                    row.camera_id,
                    row.metric.value,
                    "" if row.scope_id is None else row.scope_id,
                    row.value,
                    "" if row.staff_value is None else row.staff_value,
                    row.sample_count,
                )
            )

    def shutdown(self) -> None:
        """Nothing to flush: each row is written and closed as it arrives.

        Deliberate. Holding a handle open across a daily rotation is how an exporter ends
        up writing yesterday's file forever, and the write rate here is a handful of rows
        a minute — there is no throughput problem to solve.
        """

    def _path_for(self, row: MetricRow) -> Path:
        """Name the file from the metric and the UTC day, and verify it stays put.

        Both components are engine-controlled — a `MetricName` is an enum and the date is
        derived from a `datetime` — so this cannot traverse today. The containment check
        is here anyway because the day it *can* traverse is the day someone makes the
        filename configurable, and this is the boundary that would have to notice.
        """
        root = self._directory.resolve()
        candidate = (root / f"{row.metric.value}-{row.bucket.date().isoformat()}.csv").resolve()
        if not candidate.is_relative_to(root):
            msg = f"csv path for {row.metric.value} resolves outside the export directory"
            raise ExportError(msg)
        return candidate
