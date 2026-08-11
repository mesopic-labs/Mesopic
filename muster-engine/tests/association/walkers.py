"""Synthetic walkers, projected through a plausible retail camera.

Ground truth is exact by construction, which is the point: a hand-labelled clip has
label noise on the order of the effect ADR-0014 is trying to measure. Boxes only --
this module never allocates an image, so the privacy invariant is untouched.

Camera model: 3 m mount, 30 degrees down-tilt, 1080p, ~70 degrees horizontal FOV.
A person is a 1.7 m x 0.5 m cylinder walking at 1.5 m/s (algorithms.md §3.3.1's
reference walker). Box size therefore falls off with distance the way real footage
does, which matters because centre-distance costs are scale-sensitive. Reproducible
directly from `_project`: a 6 m walker projects to ~348 px tall, well inside a
plausible 150-350 px band for a 1.7 m adult on this rig, and height falls monotonically
with depth (568 px at 3 m, 348 px at 6 m, 251 px at 9 m, 196 px at 12 m).

`_project` is a simplified pinhole, not a rigid rotation -- `height_m` enters the
image-vertical term unrotated and is left out of `depth`, an approximation good to
within a few percent at these ranges but not exact. One symptom: box aspect ratio
(height/width) comes out ~3.4 at every depth here, where a correct rotated pinhole
gives a ratio that itself changes with depth (e.g. ~3.14 at 3 m, ~3.70 at 12 m) as
foreshortening bites the vertical and horizontal extents differently. The constant
ratio this module produces is an artifact of the simplification, not evidence the
projection is exact -- fine for a relative benchmark that compares costs against each
other under one camera model, not fine as a claim of geometric correctness.

**`crossing` on clean (uncorrupted) input ties across all four candidate costs** and
is not evidence for ranking them: the walkers move at exact constant velocity with
zero process noise, so `BoxKalmanFilter`'s constant-velocity prediction is already
correct every tick, leaving nothing for any cost function to disambiguate. The costs
only diverge once detection corruption (`recall < 1.0` and/or `box_sigma > 0.0`) makes
the prediction genuinely uncertain: plain `iou_cost` swaps and merges identities on a
corrupted `crossing` run where `giou_cost`, `centre_distance_cost` and
`expansion_iou_cost` do not. **ADR-0014's decision numbers must come from corrupted
runs.** Clean-input `crossing` stays in the sweep as a sanity check -- an all-tie
result there proves the harness is not fabricating discrimination it has no basis for
-- but a clean-input tie is not a finding about which cost to prefer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from muster.types import CameraId, DecodedFrame, Detection, FrameTs, PixelBox

WIDTH, HEIGHT = 1920, 1080
CAM = CameraId("bench")
T0 = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)

_MOUNT_HEIGHT_M = 3.0
_TILT_RAD = math.radians(30.0)
_FOCAL_PX = WIDTH / (2.0 * math.tan(math.radians(70.0) / 2.0))
_PERSON_HEIGHT_M = 1.7
_PERSON_WIDTH_M = 0.5
_SPEED_MS = 1.5

SCENARIOS = ("single", "crossing", "group", "entering")

_GROUP_DEPTH_SPACING_M = 0.7
"""`group`'s three walkers' depth spacing.

A narrower `0.5` gives a true minimum foot-point separation of **43.1 px** at closest
approach (`t≈2.67s`, between the two nearest walkers) -- half of that, 21.5 px, is the
budget `score.py`'s match radius must stay under to tell them apart. That budget could
not also clear the sweep's own detection jitter at its upper bound (`box_sigma=8`px):
the noise floor there is `3 * 8 = 24` px, which exceeds 21.5 px, so `score_run` raises
`ValueError` rather than return a `NEVER`/`MT` number that would not mean what it
claimed.

The spacing had to widen, not the jitter this scenario is swept at -- `group` is the
one scenario built to stress crowding, and capping its `box_sigma` to dodge the problem
would mean the sweep's most crowded case never actually runs at the sweep's own upper
jitter bound, hiding exactly the regime ADR-0014 cares about.

`0.7` gives a true minimum foot-point separation of **56.1 px** (half: 28.0 px) --
comfortably (not razor-thin: ~4 px / ~14% headroom) above the 24 px `sigma=8` noise
floor, while still recognisably "three people close together" rather than a spacing
chosen to make the problem disappear. Reproducible directly by scanning the scenario's
own ground truth at `dt=0.01s` over its full 6 s run (`_box_for`/`_walkers`, above).
"""

# One empty frame, reused for every tick. The tracker reads ts/width/height only.
_NO_PIXELS = np.zeros((1, 1, 3), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class _Walker:
    """A straight-line walk on the ground plane, in metres."""

    walker_id: int
    start: tuple[float, float]
    heading: tuple[float, float]
    t_enter: float


def _project(x_m: float, z_m: float, height_m: float) -> tuple[float, float]:
    """Ground-plane point -> image point, for a camera looking down the +z axis."""
    depth = max(z_m * math.cos(_TILT_RAD) + _MOUNT_HEIGHT_M * math.sin(_TILT_RAD), 0.5)
    vertical = _MOUNT_HEIGHT_M * math.cos(_TILT_RAD) - z_m * math.sin(_TILT_RAD)
    u = WIDTH / 2.0 + _FOCAL_PX * x_m / depth
    v = HEIGHT / 2.0 + _FOCAL_PX * (vertical - height_m) / depth
    return u, v


def _box_for(walker: _Walker, t_s: float) -> PixelBox | None:
    """The walker's box at time `t_s`, or None if they are outside the frame."""
    if t_s < walker.t_enter:
        return None
    travelled = _SPEED_MS * (t_s - walker.t_enter)
    x_m = walker.start[0] + walker.heading[0] * travelled
    z_m = walker.start[1] + walker.heading[1] * travelled

    _, foot_v = _project(x_m, z_m, 0.0)
    _, head_v = _project(x_m, z_m, _PERSON_HEIGHT_M)
    left_u, _ = _project(x_m - _PERSON_WIDTH_M / 2.0, z_m, 0.0)
    right_u, _ = _project(x_m + _PERSON_WIDTH_M / 2.0, z_m, 0.0)

    box = (int(left_u), int(head_v), int(right_u), int(foot_v))
    if box[2] < 0 or box[0] > WIDTH or box[3] < 0 or box[1] > HEIGHT:
        return None
    return box


def _walkers(name: str) -> tuple[list[_Walker], float]:
    """The cast for a scenario, and how many seconds it runs."""
    if name == "single":
        return [_Walker(0, (-4.0, 6.0), (1.0, 0.0), 0.0)], 6.0
    if name == "crossing":
        # Two people on opposing paths, offset slightly in depth (z=5.7 vs 6.3) so
        # they pass as a near miss rather than a byte-identical coincidence: two
        # people at exactly the same point produce an all-zeros 2x2 cost matrix under
        # every candidate, which every cost function scores identically and the
        # resulting assignment is scipy's row order, not evidence about the cost.
        # Not timed to land on any particular tick grid -- a scenario tuned to a
        # sampling rate is measuring the grid, not the tracker, and this one is
        # deliberately checked off-grid as well as on it.
        return [
            _Walker(0, (-4.0, 5.7), (1.0, 0.0), 0.0),
            _Walker(1, (4.0, 6.3), (-1.0, 0.0), 0.0),
        ], 6.0
    if name == "group":
        # Three abreast, _GROUP_DEPTH_SPACING_M apart: every box is a plausible match
        # for its neighbour, which is where centre-distance costs are weakest.
        return [
            _Walker(i, (-4.0, 5.5 + i * _GROUP_DEPTH_SPACING_M), (1.0, 0.0), 0.0) for i in range(3)
        ], 6.0
    if name == "entering":
        # Staggered arrivals at the frame edge -- the birth case ADR-0014 says the
        # damage concentrates on, and what N_init is actually trading against.
        return [_Walker(i, (-5.0, 6.0), (1.0, 0.0), t_enter=i * 1.5) for i in range(4)], 9.0
    msg = f"unknown scenario {name!r}"
    raise ValueError(msg)


def walk_scenario(
    name: str,
    *,
    dt_s: float,
    seed: int,
    recall: float = 1.0,
    box_sigma: float = 0.0,
    false_positive_rate: float = 0.0,
) -> list[tuple[DecodedFrame, list[Detection], dict[int, PixelBox]]]:
    """Generate one corrupted run: frames, detections, and exact ground truth.

    Corruption mirrors algorithms.md §13.4's `corrupt()`, minus the knobs that only
    affect geometry: `recall` drops boxes, `box_sigma` jitters them, and
    `false_positive_rate` invents them.

    Args:
        name: One of `SCENARIOS`.
        dt_s: Seconds between sampled ticks.
        seed: Seed for the corruption RNG. Ground-truth geometry is deterministic;
            only detection corruption is randomized.
        recall: Probability `[0, 1]` that a present walker is actually detected.
        box_sigma: Standard deviation, in pixels, of per-coordinate detection jitter.
        false_positive_rate: Probability `[0, 1]` of a spurious detection each tick.

    Returns:
        One `(frame, detections, truth)` tuple per tick, where `truth` maps
        `walker_id -> PixelBox` for every walker present that tick, corrupted or not.
    """
    rng = np.random.default_rng(seed)
    cast, duration_s = _walkers(name)
    run: list[tuple[DecodedFrame, list[Detection], dict[int, PixelBox]]] = []

    for tick in range(int(duration_s / dt_s)):
        t_s = tick * dt_s
        frame = DecodedFrame(
            camera_id=CAM,
            ts=FrameTs(T0 + timedelta(seconds=t_s)),
            image=_NO_PIXELS,
            width=WIDTH,
            height=HEIGHT,
        )
        truth: dict[int, PixelBox] = {}
        detections: list[Detection] = []
        for walker in cast:
            box = _box_for(walker, t_s)
            if box is None:
                continue
            truth[walker.walker_id] = box
            if rng.random() > recall:
                continue
            detections.append(
                Detection(box=_jitter(box, box_sigma, rng), score=float(rng.uniform(0.7, 0.95)))
            )
        if rng.random() < false_positive_rate:
            detections.append(Detection(box=_random_box(rng), score=float(rng.uniform(0.5, 0.65))))
        run.append((frame, detections, truth))
    return run


def _jitter(box: PixelBox, sigma: float, rng: np.random.Generator) -> PixelBox:
    """Perturb each coordinate independently, keeping the box non-degenerate."""
    if sigma <= 0.0:
        return box
    noise: NDArray[np.float64] = rng.normal(0.0, sigma, size=4)
    x1, y1, x2, y2 = (int(v + float(n)) for v, n in zip(box, noise, strict=True))
    return (min(x1, x2 - 1), min(y1, y2 - 1), max(x2, x1 + 1), max(y2, y1 + 1))


def _random_box(rng: np.random.Generator) -> PixelBox:
    """A spurious detection somewhere plausible in the frame."""
    x = int(rng.uniform(0, WIDTH - 60))
    y = int(rng.uniform(HEIGHT / 3, HEIGHT - 160))
    return (x, y, x + 60, y + 150)
