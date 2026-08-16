# Ground-truth fixtures

Accuracy is a number measured against footage a human watched and labelled. This
directory holds the labels and the descriptions of that footage. It does not hold the
footage.

```
fixtures/
  clips/<clip_id>.clip.json    what a clip is, where it came from, whether it may gate
  truth/<clip_id>.truth.json   what a human saw happen in it
```

## Why the video is not here

This repository is public, and the product's central claim is that footage never leaves
the customer's premises. Committing video of real people into it would contradict that on
day one, permanently — git history is not something you take footage back out of.

So a clip is described here and stored elsewhere. Point `MUSTER_CLIPS_DIR` at a local
directory containing `<clip_id>.mp4`, and the manifest's SHA-256 confirms it is the same
file the labels were made against. `*.mp4` is in `.gitignore`; keep it that way.

## Gate-eligibility is derived, never stored

There is no `gate_eligible` field. It is computed, every time it is asked for:

```
gate_eligible ⟺ provenance.kind ∈ {own_rig, pilot}
              ∧ consent.model_release ∈ {obtained, not_required}
```

A stored boolean can be set to `true` by a hurried afternoon. A derived one cannot. The
scoring API takes an explicit `gating` flag and raises when asked to gate on a clip that
fails either half, so "no released accuracy claim rests on footage of people who did not
consent" is enforced by the code rather than remembered by a person.

**Stock footage can never be gate-eligible**, however good it looks. A stock licence
addresses copyright; it says nothing about whether the people in frame agreed to be
filmed, and the major libraries explicitly decline to warrant that releases exist. Stock
clips are a development fixture: fine to iterate metric plugins against, never a source
for a published number.

## What is here today

| Clip | Provenance | Consent | Scene | Labels | Gate-eligible |
|---|---|---|---|---|---|
| `storefront-oblique-01` | stock | unknown | hard | 3 in, 1 out — draft | no |
| `entrance-headon-01` | stock | unknown | typical | 1 in — draft | no |
| `entrance-headon-loop30-01` | stock | unknown | typical | none | no |

The first two are ~19 s of a shop entrance and exist so metric-plugin work has something
concrete to run against. Neither is inside the reference mounting envelope (both are near
eye-level rather than 30–60° from horizontal) and neither is long enough to fill an hour,
so neither can produce the gating number. **The accuracy gate still needs an own-rig
recording session with consent obtained.**

`entrance-headon-loop30-01` is `entrance-headon-01` repeated 98 times to reach 30 minutes.
It exists for soak-style runs that want a byte-reproducible file rather than the
indefinite loop `docker/mediamtx.yml` already serves. **It is not an accuracy fixture**:
98 copies of one scene is a single observation repeated, and every loop seam is a hard cut
that breaks tracks and manufactures crossings that are not in the source.

### The labels here are unverified drafts

Both truth files carry `labelled_by: draft-unverified`. They were derived by stepping
through sampled frames rather than by watching the video end to end, and nobody has
checked them since. They are good enough to iterate a metric against and are **not** good
enough to be anybody's reference for a published number — which the gate-eligibility rule
already makes structurally impossible for these two clips.

Verify them in the clicker and re-save under your own name. The judgement calls worth
re-checking are the ones a detector will also find hard:

- **`storefront-oblique-01` @ 0.1–1.9 s** — a man walks right-to-left across the front of
  the store, over the mat, without entering. Recorded as *no crossing*. Whether that is
  right depends entirely on where the `entrance` line is drawn (see below).
- **`storefront-oblique-01` @ 17.0 s** — a man walks *out*. Recorded as `out`.
- Timestamps are ±0.3 s throughout.

### A gap this surfaced: the labeller cannot see the line

A truth file records a crossing of `line_id`, but the line's geometry lives in
`muster.yaml`, and the clicker does not show it. For a head-on doorway that is harmless —
the threshold is obvious. For an oblique storefront it is not: someone walking along the
pavement passes within a metre of the door, and whether they count depends on where the
line sits. Until the clicker can overlay the configured line, an oblique clip's labels are
only meaningful alongside the config they were made against.

## Labelling a clip

`tools/clicker/index.html` — open it in a browser, no server and no build step. Pick a
local video, play it, and mark each crossing:

| Key | Action |
|---|---|
| `space` | play / pause |
| `←` `→` | step one frame |
| `f` / `j` | mark a crossing in / out |
| `u` | undo the last mark |
| `s` | save the `.truth.json` |

The video is read with `createObjectURL` and never touches the network — the page has no
code that could send it anywhere, and a test in the engine suite fails if any appears.
That is the product's own posture applied to its own tooling.

Expect roughly 30–45 minutes of clicking per 30 minutes of clip.

Then check what you produced:

```
muster truth validate fixtures/clips/*.clip.json fixtures/truth/*.truth.json
```

## Times are media time

`t_s` is seconds from the start of the clip, not a UTC instant — the one place in this
repository where "timestamps are UTC everywhere" does not apply, and it has to be.
Replaying a clip stamps every frame with the wall clock of the replay, so a UTC timestamp
in a truth file would describe the labelling session rather than the footage. Conversion
happens at exactly one boundary, inside `muster.truth.score`.

## Adding a clip

1. Put the video somewhere outside the repository, named `<clip_id>.mp4`.
2. Write `clips/<clip_id>.clip.json`. Record the licence **and the date a human read it** —
   a licence nobody checked on a stated day is a licence nobody checked.
3. Be honest about `consent.model_release`. `unknown` is the correct answer far more often
   than it is comfortable, and it costs nothing except the ability to gate on that clip.
4. Label it, and commit the truth file.
5. Run `muster truth validate` on both.
