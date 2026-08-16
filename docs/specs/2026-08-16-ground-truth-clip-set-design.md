# Ground-truth clip set: design

**Written:** 2026-08-16

How Muster's accuracy is measured against reality: where the footage comes from, how it
is labelled, what may and may not back a released accuracy claim, and why the video
itself is not in this repository.

---

## 1. The problem

The accuracy gate and every metric plugin are specified to be written test-first against
a labelled ground-truth clip set. That set does not exist, and the work to produce it is
uncosted. Nothing downstream can be written test-first until it does.

The gate itself is narrow, and that narrowness is the key to making this tractable: it is
a single number — hour-grain footfall count error on a good-doorway scene, at the sampled
frame rate the engine actually runs at. Tracking quality, dwell, queue and occupancy are
recorded alongside it but do not gate. The metric definitions live in
`Muster-docs/docs/02-algorithms/algorithms.md` §13, the target values in
`Muster-docs/docs/07-operations/accuracy-targets-and-sla.md` §4 and §9, and the
measurement procedure in `Muster-docs/docs/04-testing/test-strategy.md` §3, §8 and §17.
This document does not restate any of them.

## 2. Findings that shaped the design

### 2.1 The raw-footage licence question was already settled

MOT17/MOT20 are CC BY-NC-SA — non-commercial. MOTSynth and JTA inherit GTA V's
non-commercial EULA. Under the existing licence policy these are development signal only
and may never back a gate. Pexels and Pixabay stock footage is commercially licensed and
usable as raw footage.

### 2.2 The *labelled* set has no off-the-shelf answer — verified 2026-08-16

Whether a commercially-licensed labelled set could be found was left open. It cannot.
Checked live:

| Candidate | Why it looked promising | Verdict |
|---|---|---|
| **PersonPath22** (Amazon) | 236 real-world videos, largely fixed surveillance-style cameras, full track ids — by far the closest fit to the deployment | **CC BY-NC 4.0.** Same category as MOT17/20: development signal only, never a gate. |
| **CrowdHuman** | Large, richly annotated, crowded scenes | **Detection only** — head / visible-region / full-body boxes per *image*, no track ids across frames. Cannot produce counting ground truth at any licence. Its site states no licence at all. |
| **Pexels / Pixabay** | Commercially licensed, no attribution required, modification permitted | Usable as raw footage, with the caveats in §2.3. |

**Conclusion: we label our own.**

### 2.3 Stock footage carries a consent gap, not just a copyright licence

- **Pexels does not guarantee model releases** for identifiable people. Its licence
  addresses copyright and prohibits selling unaltered copies or redistributing on other
  stock platforms; it says nothing about consent, and Pexels explicitly declines to
  warrant that releases exist. Pixabay requires releases for identifiable people.
- **The licence prohibits redistributing the unaltered file** on stock platforms. A git
  repository is arguably not one — but the question does not need answering if the bytes
  never enter the repository (§4.2).

For a product positioned on "footage never leaves your premises", gating a released
accuracy claim on footage of people who never consented would be indefensible. The Oxford
Town Centre dataset — withdrawn in 2020 on exactly those grounds — was already removed
from the test strategy for this reason. Stock footage is therefore a **development
fixture only**, never gate-eligible, and that is enforced in code rather than by
convention (§4.3).

### 2.4 Perception ground truth is not on the gate's path

Two ground truths are kept distinct:

- **Perception ground truth** — per-frame boxes and ids. Feeds tracking metrics.
- **Metric ground truth** — the business number: crossing timestamps and directions.
  Feeds count error.

The gate is a count-error number, so it needs **metric ground truth only**. That is the
difference between roughly half an hour of clicking through a 30-minute clip and roughly
twenty hours of drawing boxes. Treating perception ground truth as in-scope is what makes
this look impossible.

## 3. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Ground truth is **metric ground truth only** in v1 — crossing timestamps and directions | The gate needs nothing more (§2.4) |
| D2 | We **label our own footage**; no public dataset backs a gate | No commercially-licensed labelled set exists (§2.2) |
| D3 | **Staged sourcing**: a stock-footage development fixture first, an own-rig clip as the gate artefact | Plugin work needs something concrete immediately; only own footage can be gate-eligible |
| D4 | **Clip bytes never enter the repository**; a manifest does | This repository goes public; committing footage of real people would contradict the product's central claim |
| D5 | **Gate-eligibility is derived, never stored** | A stored boolean can be set to `true`; a derived one cannot |
| D6 | Labels are in **media time** (offset from clip start), converted to UTC at one boundary | Replay stamps frames with wall-clock *now*; only offsets align truth to engine output |
| D7 | The labelling tool is a **dependency-free local HTML page** | No new dependency; no frame ever touches the network, including in our own tooling |

## 4. The design

### 4.1 The truth file

`fixtures/truth/<clip_id>.truth.json`, committed — it is numbers, not pixels.

```json
{
  "schema_version": 1,
  "clip_id": "doorway-daylight-01",
  "labelled_by": "mark",
  "labelled_at_utc": "2026-08-17T14:02:11Z",
  "duration_s": 1800.0,
  "crossings": [
    { "t_s": 12.480, "line_id": "entrance", "direction": "in" },
    { "t_s": 31.200, "line_id": "entrance", "direction": "out" }
  ]
}
```

| Field | Type | Constraint |
|---|---|---|
| `schema_version` | `int` | Exactly `1`; any other value is rejected, not coerced |
| `clip_id` | `str` | Must resolve to a manifest in `fixtures/clips/` |
| `labelled_by` | `str` | Free text; identifies the rater for inter-rater agreement |
| `labelled_at_utc` | `datetime` | Timezone-aware UTC |
| `duration_s` | `float` | `> 0`; must equal the manifest's `duration_s` |
| `crossings[].t_s` | `float` | `0 ≤ t_s ≤ duration_s`; list must be non-decreasing |
| `crossings[].line_id` | `LineId` | — |
| `crossings[].direction` | `Direction` | Reuses the engine's existing `IN` / `OUT` `StrEnum` |

Dwell intervals and per-minute occupancy — the other two things the ground-truth format
eventually wants — are **not** optional fields here. They land as `schema_version: 2`
when something needs them. Dead optional fields that nobody fills in are worse than a
version bump.

**On D6, media time.** `t_s` is an offset from clip start, which reads against this
repository's "timestamps are UTC everywhere" rule. It has to be: replaying a clip through
the mediamtx rig stamps every frame with the current wall clock, so engine output and
truth are alignable only by offset-from-stream-start. The truth file is therefore in media
time and conversion happens at exactly one boundary inside the scoring code — the same
discipline the tracker already applies to pixel→normalized. The module docstring must say
so, or a future reader will "fix" it.

### 4.2 The clip manifest

`fixtures/clips/<clip_id>.clip.json`, committed. The video itself is resolved at run time
from `MUSTER_CLIPS_DIR` and verified by SHA-256.

```json
{
  "schema_version": 1,
  "clip_id": "doorway-daylight-01",
  "sha256": "…",
  "duration_s": 1800.0,
  "width": 1920, "height": 1080, "fps": 25.0,
  "provenance": {
    "kind": "own_rig",
    "licence": "Owned — recorded by us",
    "licence_verified_utc": "2026-08-16",
    "url": null
  },
  "consent": { "model_release": "obtained", "note": "signed, held offline" },
  "scene": {
    "reference": "good_doorway",
    "mount_height_m": 2.8,
    "mount_angle_deg": 42,
    "lighting": "even_daylight"
  }
}
```

`provenance.kind` ∈ `own_rig | pilot | stock | synthetic`.
`consent.model_release` ∈ `obtained | not_required | unknown`.
`scene.reference` ∈ `good_doorway | typical | hard`, matching the reference conditions in
`Muster-docs/docs/07-operations/accuracy-targets-and-sla.md` §1.1.

`synthetic` covers `testsrc2`-style generated clips, which contain no people and no
consent question but also no ground truth worth labelling — they exist for performance
work, not accuracy, and the manifest records that distinction rather than leaving it to a
filename.

### 4.3 Gate-eligibility

There is **no `gate_eligible` field**. It is computed:

```
gate_eligible(manifest) ⟺ provenance.kind ∈ {own_rig, pilot}
                        ∧ consent.model_release ∈ {obtained, not_required}
```

Stock footage can never be gate-eligible, however good it looks. Anything with unknown
consent can never be gate-eligible. The scoring API takes an explicit `gating: bool` and
raises when asked to gate on an ineligible clip. This makes "no release is gated on
non-consented footage" a property the type system enforces, in the same spirit as the
import-linter contract that makes "video never leaves the building" a property of the
dependency graph.

This supersedes the existing instruction to commit a golden `.mp4` into the repository,
and therefore needs an ADR.

### 4.4 Derivation and scoring

`muster.truth` exposes:

- `load_truth(path) -> TruthFile` — pydantic, strict, rejects on first inconsistency
- `load_manifest(path) -> ClipManifest` — parses the JSON only; it does not touch the
  video, so it works in CI where no clip is present
- `resolve_clip(manifest, clips_dir) -> Path` — the *only* place SHA-256 is verified,
  raising on mismatch or absence. Separating the two means the manifest tests and the
  scoring tests run without any video on disk, which is what keeps this suite in the
  per-PR gate rather than the nightly one
- `footfall_per_minute(truth) -> dict[int, int]` — minute index from clip start → count
- `mape(predicted, truth) -> float` — with the `max(true, 1)` divide-by-zero guard the
  test strategy specifies for empty windows
- `score(truth, metrics, *, gating: bool) -> Score`

Not in scope: the evaluation CLI, TrackEval, and the tracking metrics. Those are a
substantially larger piece of work.

### 4.5 The labelling tool

`tools/clicker/index.html` — one self-contained file, no server, no build step, no
dependency. Open in a browser, pick a local video, mark crossings:

| Key | Action |
|---|---|
| `space` | play / pause |
| `←` `→` | frame step |
| `f` | mark crossing `in` |
| `j` | mark crossing `out` |
| `u` | undo last mark |
| `s` | save `.truth.json` |

The video is read locally via `createObjectURL` and never touches the network. That is
not incidental — it is the product's own posture applied to its own tooling, and
`fixtures/README.md` should say so plainly.

Python side, following the existing Typer command style in `muster/cli.py`:

```
muster truth validate <file>
muster truth score --truth <file> --metrics <file> [--gating]
```

### 4.6 Module layout and the import contract

New package `muster.truth`. A new import-linter contract forbids it from importing
`muster.ingest`, `av`, or `cv2` — it handles labels, never pixels. This mirrors the
existing "sync client reads metrics, never frames" contract.

### 4.7 No new dependencies

`pydantic` and `typer` are already engine dependencies. The clicker has none.

## 5. Tests, written first

1. **Truth schema** — rejects unknown `direction`; negative `t_s`; `t_s > duration_s`;
   unsorted crossings; unknown `schema_version`; missing `clip_id`; `duration_s`
   disagreeing with the manifest.
2. **Manifest** — gate-eligibility false for `stock`; false for `synthetic`; false for
   `unknown` consent; true only for `own_rig`/`pilot` with consent obtained.
   `resolve_clip` raises on a SHA-256 mismatch and on a missing file (this one test
   writes a temporary byte file; it needs no real footage).
3. **Derivation** — crossings → per-minute footfall across minute boundaries; count error
   against a known series; the `max(true, 1)` guard on an empty window.
4. **The gate refuses ineligible input** — `score(..., gating=True)` raises on a
   non-gate-eligible manifest. This is the test that makes §4.3 real rather than
   documentary.

## 6. Documentation

- **ADR-0015** — ground-truth clip sets: manifests in the repository, bytes outside it,
  gate-eligibility derived. It supersedes the existing instruction to commit the golden
  `.mp4`, which cannot survive this repository going public.
- **`fixtures/README.md`** — how to obtain a clip, how to label one, and why the bytes are
  not in the repository.

## 7. Residual human work

This delivers everything up to and including the tool that makes labelling fast. It
cannot deliver the labels themselves. What remains:

1. **Confirm the stock candidates** are actually inside the reference envelope — fixed
   mount, 30–60° from horizontal, single-file passage, feet visible. Requires watching
   them.
2. **Label the development fixture** — roughly 30–45 minutes of clicking per 30 minutes
   of clip.
3. **Record the gate clip** on the doorway rig, with consent obtained. Only own-rig
   footage can be gate-eligible, so the accuracy gate cannot be met without this session.
4. **A second rater on one clip**, for inter-rater agreement. Low agreement means the
   *scene* is ambiguous and its target should widen — not that the engine is wrong.

## 8. Open questions

- **Who records the gate clip, and where?** The session is unscheduled and the gate
  depends on it.
- **Does the good-doorway clip need to span an IR-night transition?** The test strategy
  lists it among four goldens "easy to forget", but the gate is specified on the
  good-doorway scene, which is even daylight. Read here as a separate later golden, not a
  blocker for this one.
- **Hour-grain vs minute-grain.** The gate is stated at hour grain; the engine's durable
  unit is `metrics_minute`. Scoring aggregates minutes to hours, so a 30-minute clip
  yields a partial hour. Either the gate clip runs ≥ 1 hour, or the gate is evaluated on a
  minute-grain series and the hour-grain figure derived. Needs a call before the recording
  session.

## 9. References

Design documents live in the private `Muster-docs` corpus and are cited by path, never
restated here:

- `docs/04-testing/test-strategy.md` §3, §7.1, §8, §12, §13
- `docs/07-operations/accuracy-targets-and-sla.md` §1.1, §4, §9
- `docs/02-algorithms/algorithms.md` §13
- `docs/01-architecture/adr/0013-permissive-detector-fallback.md` — the licence standard
  of care this design applies
