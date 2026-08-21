"""Load and validate the one human-editable `mesopic.yaml` (engine-architecture.md §13.1)."""

from __future__ import annotations

from mesopic.config.loader import load_config
from mesopic.config.schema import MesopicConfig

__all__ = ["MesopicConfig", "load_config"]
