"""Stable `TrackId`s across frames (engine-architecture.md §7).

Must not compute metrics and must not know about geometry.
"""

from __future__ import annotations

from mesopic.tracker.tracker import Tracker

__all__ = ["Tracker"]
