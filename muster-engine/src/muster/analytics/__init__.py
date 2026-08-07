"""Tracks + geometry -> raw events (engine-architecture.md §8).

Deliberately **thin and delegating**: this package wires "for each track, for each line
and zone, call the predicate, emit a `RawEvent`". The metric *math* — crossing tests,
containment, debounce, dwell weighting — lives in algorithms.md and is imported, never
re-derived here.

This package must not persist anything and must not talk to the network. That is
enforced by an import-linter contract, not by convention.
"""

from __future__ import annotations

from muster.analytics.site_geometry import SiteGeometry

__all__ = ["SiteGeometry"]
