"""Read `muster.yaml` from disk into a validated `MusterConfig`.

`yaml.safe_load` only — never `yaml.load`. The path is canonicalised and checked for
containment before it is opened.

Implements P2.1.
"""

from __future__ import annotations

from pathlib import Path

from muster.config.schema import MusterConfig


def load_config(path: Path) -> MusterConfig:
    """Parse and validate a config file.

    Raises:
        ConfigError: the file is missing, unparseable, or fails validation.
    """
    raise NotImplementedError
