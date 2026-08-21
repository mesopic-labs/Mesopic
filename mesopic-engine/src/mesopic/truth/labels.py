"""Truth files: a human's account of what walked through the door, and when.

**Media time, deliberately.** ``t_s`` is an offset from the start of the clip, which reads
against this repository's "timestamps are UTC everywhere" rule and has to. Replaying a
clip through the test rig stamps every frame with the wall clock of the replay, so a UTC
instant in a truth file would describe the labelling session rather than the footage;
offsets are the only thing that aligns truth to engine output. Conversion to UTC happens
at exactly one boundary, inside scoring, in the same spirit as the tracker's single
pixel-to-normalized conversion. Do not "fix" this.

Scope is v1's gate and nothing more: crossing timestamps and directions, because count
error is the only number that gates a release. Dwell intervals and per-minute occupancy
are not optional fields here — they arrive as ``schema_version: 2`` when something needs
them, because a dead optional field nobody fills in is worse than a version bump.
"""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise
from math import isclose
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from mesopic.errors import TruthError
from mesopic.truth._json import parse, read_json
from mesopic.truth.clips import CLIP_ID_PATTERN, ClipManifest, TruthModel
from mesopic.types import ClipId, Direction, LineId

MAX_TRUTH_BYTES = 4 * 1024 * 1024
"""Generous for JSON, and still an explicit ceiling: an hour of a busy doorway is a few
thousand crossings, which is tens of kilobytes."""

MAX_CROSSINGS = 100_000
"""No human labels more than this, and an unbounded list is an unbounded allocation."""

DRAFT_RATER = "draft-unverified"
"""The rater name a label carries when nobody has stood behind it yet.

Load-bearing rather than decorative: scoring refuses to *gate* on a truth file carrying
it. Deliberately not a boolean field — ADR-0015's argument against a stored
``gate_eligible`` applies here too. Promoting a draft means a human replacing this with
their own name, which is an act of accountability rather than a flag anybody can flip.
"""

DURATION_TOLERANCE_S = 1e-3
"""Durations are recorded from the container to a few decimal places; a millisecond of
float noise is not a disagreement, and anything larger means two different cuts."""


class Crossing(TruthModel):
    """One person crossing one line, once, in one direction."""

    t_s: float = Field(ge=0.0)
    """Media time: seconds from the start of the clip. Never a UTC instant."""

    line_id: LineId = Field(min_length=1, max_length=64)
    direction: Direction


class TruthFile(TruthModel):
    """Everything a human observed in one clip. The only thing accuracy is measured against."""

    schema_version: Literal[1]
    clip_id: Annotated[ClipId, Field(pattern=CLIP_ID_PATTERN, max_length=64)]
    labelled_by: str = Field(min_length=1, max_length=100)
    """Free text, identifying the rater so two passes over one clip can be compared."""

    labelled_at_utc: AwareDatetime
    duration_s: float = Field(gt=0.0)
    crossings: tuple[Crossing, ...] = Field(max_length=MAX_CROSSINGS)

    @field_validator("labelled_at_utc")
    @classmethod
    def _normalise_to_utc(cls, value: datetime) -> datetime:
        """Aware is enforced by the annotation; this makes the stored value actually UTC."""
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _crossings_must_lie_inside_the_clip(self) -> Self:
        """A mark past the end of the clip means the labeller was watching something else."""
        for crossing in self.crossings:
            if crossing.t_s > self.duration_s:
                message = (
                    f"crossing at {crossing.t_s}s is past the end of a {self.duration_s}s clip"
                )
                raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _crossings_must_be_non_decreasing(self) -> Self:
        """Non-decreasing, not increasing: two people abreast through a wide door share
        an instant, but time never runs backwards."""
        times = [crossing.t_s for crossing in self.crossings]
        if any(later < earlier for earlier, later in pairwise(times)):
            message = "crossings are not in time order"
            raise ValueError(message)
        return self


def load_truth(path: Path) -> TruthFile:
    """Parse a truth file, rejecting on the first inconsistency."""
    return parse(TruthFile, read_json(path, MAX_TRUTH_BYTES), path)


def check_pairing(truth: TruthFile, manifest: ClipManifest) -> None:
    """Confirm a truth file describes the clip it claims to.

    Two ways a labelling session goes wrong silently: the file is scored against a
    different clip entirely, or against a different *cut* of the same one. Both produce a
    complete, plausible accuracy number from labels that never matched the footage.
    """
    if truth.clip_id != manifest.clip_id:
        message = f"truth file describes {truth.clip_id}, not {manifest.clip_id}"
        raise TruthError(message)
    if not isclose(
        truth.duration_s, manifest.duration_s, rel_tol=0.0, abs_tol=DURATION_TOLERANCE_S
    ):
        message = (
            f"truth file for {truth.clip_id} was labelled against a {truth.duration_s}s cut, "
            f"but the manifest describes {manifest.duration_s}s"
        )
        raise TruthError(message)
