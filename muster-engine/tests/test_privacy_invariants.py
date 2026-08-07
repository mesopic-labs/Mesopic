"""The privacy invariants, asserted rather than promised.

"Footage never leaves the building" is the product's categorical claim. These tests are
the mechanism that keeps it true as the code grows: they fail in CI, before review, if
someone adds a column that could hold a pixel or a call that could write one to disk.

The full runtime assertion — run the pipeline, prove nothing touched the filesystem —
lands with P1.8 once there is a pipeline to run. These are the static half, and they are
armed from the first commit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "muster"
SCHEMA = SRC / "store" / "schema.sql"

# Column names that would mean the store can hold pixels. `counts` (the heatmap grid) is
# fine: it is deciseconds of presence per cell, derived from foot-points, not imagery.
FORBIDDEN_COLUMN_TOKENS = (
    "image",
    "frame",
    "crop",
    "thumbnail",
    "thumb",
    "snapshot",
    "jpeg",
    "jpg",
    "png",
    "pixel",
    "bbox",
)

# Calls that write image data to disk. If one of these is ever needed (it should not be),
# it needs an ADR first, not a `# noqa`.
FORBIDDEN_CALLS = (
    "cv2.imwrite",
    "imageio.imwrite",
    "PIL.Image.save",
    ".tofile(",
    "np.save",
)


@pytest.mark.privacy
def test_store_schema_has_no_column_that_could_hold_a_pixel() -> None:
    """ADR-0005: the edge store is metrics and events. Never imagery."""
    schema = SCHEMA.read_text(encoding="utf-8")

    # Strip comments first: the schema explains this invariant in prose, and that prose
    # legitimately contains the very words we are banning.
    ddl = re.sub(r"--[^\n]*", "", schema).lower()

    offenders = [token for token in FORBIDDEN_COLUMN_TOKENS if token in ddl]
    assert not offenders, (
        f"schema.sql declares something pixel-shaped: {offenders}. "
        "The edge store holds metrics and events only (ADR-0005)."
    )


@pytest.mark.privacy
def test_no_module_writes_image_bytes_to_disk() -> None:
    """A frame is a local variable in the worker loop and is never persisted."""
    offenders: list[str] = []
    for module in SRC.rglob("*.py"):
        source = module.read_text(encoding="utf-8")
        # Docstrings in this repo describe the invariant; only flag real call sites.
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith(("#", "*"))
        )
        offenders += [
            f"{module.relative_to(SRC)}: {call}" for call in FORBIDDEN_CALLS if call in code
        ]

    assert not offenders, (
        f"image-writing call sites found: {offenders}. "
        "Frames are discarded the instant they are processed (ADR-0005)."
    )


@pytest.mark.privacy
def test_sync_package_cannot_reach_a_frame() -> None:
    """The metrics-sync client reads metrics tables and nothing else.

    Structurally enforced by an import-linter contract too; this is the cheap, fast
    version that runs on every commit.
    """
    banned = ("import av", "import cv2", "from muster.ingest", "from muster.detector")
    for module in (SRC / "sync").rglob("*.py"):
        source = module.read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith(("#", "*"))
        )
        for banned_import in banned:
            assert banned_import not in code, (
                f"{module.name} imports {banned_import!r}: the sync client must have no "
                "path to a frame (ADR-0005)."
            )
