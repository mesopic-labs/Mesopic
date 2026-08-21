"""SQLite as the local source of truth (engine-architecture.md §11, ADR-0009).

Nothing outside this package touches the database file. The store is the **single
writer**, the source of truth, and — via the `synced_at` cursor — the offline
store-and-forward buffer for the cloud sync, for free.
"""

from __future__ import annotations

from mesopic.store.store import Store

__all__ = ["Store"]
