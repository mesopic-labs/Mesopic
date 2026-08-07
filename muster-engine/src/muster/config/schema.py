"""The `muster.yaml` schema as pydantic models.

`muster.yaml` is the authoritative site description; the store's `cameras`/`zones`/
`lines` tables are its compiled form. Validation is strict and happens once, at load: a
bad config fails loud at startup rather than silently mis-counting.

Three conventions the models must enforce, not merely document:

* **Secrets are `*_env` references only.** An inline secret is a validation error.
* **All geometry is normalized** `[0, 1]`, so it survives a resolution change.
* **Referential integrity**: every `camera_id` on a line or zone must exist.

Implements P2.1. Fields below are the shape from engine-architecture.md §13.1; the
validators that make the three rules above real are the work.
"""

from __future__ import annotations

from pydantic import BaseModel


class MusterConfig(BaseModel):
    """Root of the validated configuration tree."""

    # TODO(P2.1): site / budget / cameras / lines / zones / thresholds / exporters /
    # cloud_sync sub-models, with the validators described in the module docstring.
