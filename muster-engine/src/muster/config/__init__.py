"""Load and validate the one human-editable `muster.yaml` (engine-architecture.md §13.1)."""

from __future__ import annotations

from muster.config.loader import load_config
from muster.config.schema import MusterConfig

__all__ = ["MusterConfig", "load_config"]
