"""Store-and-forward of metrics JSON to the paid cloud (engine-architecture.md §14).

Optional and **off by default** — the free, local experience never touches it (ADR-0001).

This package reads metrics tables and nothing else. It has no access to frames, and an
import-linter contract forbids it from importing anything that does. "Video never leaves
the building" is a structural property here, not a promise (ADR-0005).
"""

from __future__ import annotations

from mesopic.sync.client import MetricsSyncClient

__all__ = ["MetricsSyncClient"]
