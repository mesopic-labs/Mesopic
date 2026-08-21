"""What a saved config changes, and what a running engine can actually adopt.

`Supervisor.reload` carries two things to a live worker: compiled geometry, and the fps
envelope. Everything else in `MesopicConfig` is read once — a source is opened when the
worker spawns, exporters are built at the composition root, `storage.path` decides which
database was opened. Saving those is legitimate and takes effect at the next start.

What is not legitimate is letting the operator believe otherwise. A save that reported
success while half of it sat inert on disk is the kind of lie that is only discovered when
the numbers do not move, so `/config` names the sections it could not swap.

Implements P3.4 (engine-architecture.md §13).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesopic.config.schema import MesopicConfig

LIVE_SECTIONS = ("budget", "lines", "zones")
"""The sections a running engine adopts without a restart.

A deny-list of three rather than an allow-list of the rest, so a section added to
`MesopicConfig` after this was written is reported as needing a restart until someone
teaches the reload to carry it. Wrong in the direction that tells the truth."""


def sections_needing_restart(before: MesopicConfig, after: MesopicConfig) -> tuple[str, ...]:
    """The top-level sections that changed and cannot take effect until the next start.

    Compared section by section rather than key by key: the operator's next action is the
    same either way, and a per-key diff would be a second, subtly different notion of
    "changed" living next to pydantic's own equality.
    """
    return tuple(
        name
        for name in type(before).model_fields
        if name not in LIVE_SECTIONS and getattr(before, name) != getattr(after, name)
    )


__all__ = ["LIVE_SECTIONS", "sections_needing_restart"]
