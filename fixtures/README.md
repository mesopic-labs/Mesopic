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

So a clip is described here and stored elsewhere. Point `MESOPIC_CLIPS_DIR` at a local
directory containing `<clip_id>.mp4`, and the manifest's SHA-256 confirms it is the same
file the labels were made against. `*.mp4` is in `.gitignore`; keep it that way.

That directory is also what the RTSP sidecar serves, each clip under its own id, so
replaying one is two commands:

```
export MESOPIC_CLIPS_DIR=~/Projects/mesopic-clips
docker compose --profile camera up -d mediamtx     # rtsp://127.0.0.1:8554/<clip-id>
```

The stream is a straight remux, not a re-encode — an accuracy number measured against
different pixels from the ones a human labelled is not measured against the clip.

**The manifest names one file, not one recipe.** `scripts/normalize-clip.sh` is bit-exact
on one machine and not across machines: libx264 defaults its thread count to the core
count, and frame threading changes the bytes. Re-normalising an original elsewhere yields a
structurally identical artefact — same duration, same frame count, same colour tags — with
a different SHA-256, which `resolve_clip` refuses.

Which of the two you fix depends on what the labels are bound to, and that differs by clip:

- **Labelled by watching a file** — `residential-lobby-01` and the `home-hallway-*` clips.
  The times mean what they mean *because* of the artefact somebody scrubbed through, so
  get that file. Re-cutting it and repointing the manifest would silently redefine what
  the labels describe.
- **Labelled without one** — `home-entrance-01`, tallied live on paper while the recording
  happened. There is no artefact the labels are bound to: the times are media seconds on
  the recording's own timeline, and any faithful normalisation of the original carries them
  identically. Here re-normalising and repointing the manifest is the cheaper fix, and it
  is what happened — the SHA-256 names a cut made from the original on the maintainer's
  machine, not the one the labeller happened to produce.

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
| `home-entrance-01` | own_rig | obtained | good_doorway | 33 in, 32 out — markos | yes |
| `home-hallway-oblique-01` | own_rig | obtained | hard | 3 in, 3 out — emil | yes |
| `home-hallway-oblique-02` | own_rig | obtained | hard | 1 in, 1 out — emil | yes |
| `home-kitchen-oblique-01` | own_rig | obtained | hard | n/a — no counting line | yes |
| `residential-lobby-01` | own_rig | obtained | hard | 2 in, 9 out — markos | yes |
| `storefront-oblique-01` | stock | unknown | hard | 3 in, 1 out — draft | no |
| `entrance-headon-01` | stock | unknown | typical | 1 in — draft | no |
| `entrance-headon-loop30-01` | stock | unknown | typical | none | no |

The stock clips are ~19 s of a shop entrance and exist so metric-plugin work has
something concrete to run against. Neither is inside the reference mounting envelope
(both are near eye-level rather than 30–60° from horizontal), and being stock, neither
can ever gate.

### The 2.1 m `home-hallway-*` and `home-kitchen-*` clips: what they are and what they are not

Three own-rig clips of a domestic hallway and kitchen, 3 min 34 s in total, one subject,
a fixed mount, daylight. They are the first footage here that **is** gate-eligible: the
provenance is ours and the only person in frame is the author. That matters, because it
is the half of the problem no stock library can solve.

They are still not the gate clip, for reasons that are about the footage rather than the
paperwork:

- **The mount is 2.1 m**, below the 2.2 m floor for a `typical` scene, so all three are
  classified `hard` on the reference table's own terms. Everything else about them —
  angle inside 30–60°, feet visible throughout, single-file, ≥ 1080p — is `typical` or
  better; the height is the axis that binds. The gate is specified on `good_doorway`, so
  no amount of labelling makes these clips produce it.
- **There are only ~8 crossings in the longest clip.** At that count a single miscount is
  a double-digit error, so the number could not distinguish a passing engine from a
  failing one even if the scene qualified.
- **One subject means no two-abreast, no mutual occlusion, no queue.** Whatever these
  clips can say about `queue_len` is nothing.

What they are good for is the thing that was actually blocked: metric plugins can now be
written against real detections on real footage instead of invented expectations.

**That dedicated recording session has since happened** — 2.5–3.5 m, ≥ 1 hour continuous,
several people walking a known schedule — and it is `home-entrance-01` below. Note that it
shares the `home-` prefix and nothing else: different house, different rig, different
mount, three subjects rather than one. Nothing in this section describes it. See
`../../Mesopic-docs/docs/04-testing/ground-truth-clip-set-design.md` §7 and §8; §8's
hour-grain-vs-minute-grain question was meant to be answered *before* that session and was
not, so it now has to be answered against the footage rather than ahead of it.

> **Gate-eligibility answers permission, not validity.** `gate_eligible()` asks whether we
> may publish from footage of these people at all — provenance and consent, and nothing
> else. Committing these three clips armed a gap that had been inert while every
> gate-eligible clip was hypothetical, so `gate_blockers()` now asks the other question
> beside it: does the clip measure the scene the gate is specified on, and did a human
> stand behind the labels. A clip can pass the first and fail the second — these three do.
> What the code still does not check is clip length or crossing density, so "is this clip
> a *sufficient* basis for this number?" remains a judgement call.

### `residential-lobby-01`: the first long clip, and still not the gate clip

40 minutes of a residential building's entrance hall, own rig, consent obtained. It is the
first clip here long enough to be read at hour grain and the first with more than one
person in frame. It is gate-eligible on the derived rule, and it is still not the gate
clip:

- **The lighting binds it to `hard`, and only the lighting.** The mount is 2.8 m at
  roughly 45°, which is inside the reference envelope on both axes — angle to the nearest
  five degrees or so, not instrumented. Everything about the geometry is what the good
  doorway asks for. A strong backlit doorway is the `hard` column's own named example, and
  a scene is bound by its worst axis, so the class does not move.
- **Eleven crossings in 40 minutes.** Longer than the `home-*` clips without being denser,
  so a single miscount is a ~9 % error. Wide enough to catch a tracker breaking, far too
  wide to detect drift.

That first point is the useful one. This is the first mount here whose geometry the
reference envelope actually wants, so the doorway is a candidate for a real gate session
once the light is dealt with — same rig, even artificial lighting, no re-mounting.

The labels are **not** a draft. `markos` watched the clip end to end against a hand-written
tally made before the truth file existed, and the two agreed, which makes this the first
clip here a human has stood behind. So one refusal remains:

```
scene is hard, but this gate is specified on good_doorway
```

Gating it on `hard` instead — `gate_scene=SceneReference.HARD`, which is what that
parameter exists for — passes with nothing misrepresented. It is a deliberately looser bar
than M1's: the hard scene's band is ≤ 25 % MAPE launch-credible against the good doorway's
≤ 7 %. Combined with eleven crossings, a number from this clip is a regression tripwire and
is never an M1 result.

Two properties of the file a labeller needs to know before opening it:

- **The doorway is blown out white for most of the clip**, so everyone crossing it is a
  silhouette. That is why it is `hard`, and why it is a genuinely useful thing to measure
  against.
- **It is eight DVR segments concatenated out of chronological order.** The clock burnt
  into the top-left jumps backwards at roughly 301, 602, 903, 1204, 1505, 1806, 2107 and
  2408 s, and the last two seconds are an editor's outro card. Media time is unaffected,
  which is the whole reason `t_s` is media time — the burnt-in clock is not a timebase and
  must not be used as one.

### `home-entrance-01`: the first clip inside the good-doorway envelope

Two hours of a domestic entrance, own rig, consent obtained, three subjects. It is the
first clip here whose scene classifies as `good_doorway` — the scene the gate is specified
on — so it is the first one that can produce that number at all rather than a proxy for it.

The mount is **2.6 m at 34°**, measured at the rig with a tape before the camera moved, and
both figures sit inside the reference envelope on both axes. Feet are visible throughout.
Lighting is the axis that bound every earlier clip here to `hard`, and this is the first
one it does not bind:

- **Daylight only** — no artificial light at all, sun out, through an open living-room
  window — and **nothing clips**. Measured on the artefact the manifest names, from exact
  luma histograms of 30 frames spread across the two hours, in the raw code values the
  encoder wrote:

  | | |
  |---|---|
  | Pixels at 255 | **0.0000 %** — the brightest pixel in any sample is 240–245 |
  | Pixels above 95 % of full scale (Y > 242) | **0.000 %** |
  | Pixels in the bottom tenth (Y < 26) | 3.8 % |
  | 90th-percentile luma | **226–232 of 255** (88.6 %–91.0 %), across all 30 samples |

  Nothing clips at the top and nothing crushes at the bottom, and the levels hold flat
  end to end. **A daylight source is not what makes a doorway `hard`; a blown-out one
  is**, and this doorway is not blown out. Contrast `residential-lobby-01`, where the
  doorway is white for most of the clip and everyone crossing it is a silhouette.

  > Measure the **raw Y plane**, not `format=gray`. `gray` is a full-range format, so the
  > scaler expands tv range 16–235 onto 0–255 on the way in and manufactures clipping that
  > is not in the file — it reports 6.1 % of pixels at 255 for a clip whose true maximum
  > is 245. Read the first `w*h` bytes of a `yuv420p` frame instead, or use `signalstats`,
  > which does not convert. The two agree; `gray` disagrees with both.

**Sixty-five crossings, 33 in and 32 out** — denser than anything else here, and still not
dense. The limit is the grain rather than the count: the gate is stated at hour grain, two
hours is **two buckets** (footfall 20 and 13), and a single miscount is 5.0 % or 7.7 % of
its bucket against the good doorway's ≤ 7 %. That is a far better instrument than
`residential-lobby-01`'s eleven crossings and still a coarse one for a pass/fail that close
to the line.

**Eighty-eight of the 120 minutes are empty, and one quiet stretch runs 53 minutes**
(33:43 to 1:26:40). That is real footage of an empty hallway, not a hole in the labelling:
all three subjects are inside when it begins and the first mark after it is an exit, so
occupancy joins up across it. A quiet minute is an observation the engine can get wrong,
which is why `_empty_minutes` keeps every one of them.

**The clip is the first two hours of a 2 h 3 min original**, cut with the fourth argument
to `scripts/normalize-clip.sh`. The tail is footage nobody labelled, and scoring it would
charge the engine for crossings the truth file was never going to contain.

Two things it still cannot do:

- **There is no camera or line for it in `mesopic.yaml`**, so nothing can be scored against
  it yet. The endpoint order that decides the sign of every crossing has to be drawn
  against the frame, not guessed — the same gap the section below describes.
- **`gate_blockers()` returns nothing for it**: own rig, consent obtained, `good_doorway`,
  and a named rater. It is the first clip here that clears every check the code makes,
  which leaves the judgement the code does *not* make as the only thing in the way — is
  two hours and 33 inward crossings a **sufficient** basis for a published number? That is
  not obviously yes, and the code will not stop anyone who decides it is.

#### How `home-entrance-01` was labelled

Rater `markos`, tallied **live on paper while the recording was happening**, then
transcribed. That is a different method from every other clip here and it cuts both ways: a
live observer sees the room rather than a compressed frame and cannot be fooled by the
codec, but they cannot rewind either, and the footage has not been re-watched since.

**Marks are whole seconds, so read them as ±0.5 s.** Exactly one crossing sits on a minute
boundary — the inward mark at 1440.0 s — making it the only label in the file whose
uncertainty spans two minute buckets. At hour grain nothing moves.

The transcription was checked by walking each subject's entries and exits in order: all
three alternate cleanly end to end, and nobody enters a house they are already inside. That
caught one real error — the mark at 24:04 was attributed to the wrong subject, which had
made one subject's exit at 24:18 impossible and another's entry at 26:04 a double — and it
is worth being exact about
what the check proves. It finds contradictions **among the marks that were written down**.
It cannot see a crossing that was never written down at all, and a *pair* of missed
crossings — out and back in, unobserved — leaves occupancy consistent and invisible. The
rater field says a human stands behind these labels; it does not say they were re-watched.

### How the `home-hallway-*` clips were labelled

Rater `emil`, 2026-08-17, watching the normalised 1080p25 clips end to end — not the
camera originals, which are 60 fps and a different length, and not sampled frames.

**Marks are whole seconds, so read the timestamps as ±0.5 s** rather than the ±0.3 s the
stock drafts claim. That is coarser than it looks and still comfortably enough for a
count: the only crossing anywhere near a minute boundary is `-01`'s inward mark at 61.0 s,
which stays in minute 1 across its whole uncertainty range. Nothing here can move buckets.

The labels were cross-checked against an independent background-subtraction sweep of the
same footage, run at 2 Hz against this same line. **All six crossings in `-01` matched**,
median offset 0.4 s and worst 1.2 s (the outward mark at 67 s, the one to re-time first if
anyone ever needs sub-second precision). The sweep also produced eight crossings the rater
did not, *every one of them* between 18 s and 53 s — the interval where the subject is
sitting down, seated, and standing up again. That is blob jitter across the line, and it
is a good illustration of why a truth file is not something a detector can produce for
itself: the machine's extra "crossings" are concentrated exactly where a human sees
somebody sitting still.

`entrance-headon-loop30-01` is `entrance-headon-01` repeated 98 times to reach 30 minutes.
It exists for soak-style runs that want a byte-reproducible file rather than the
indefinite loop `docker/mediamtx.yml` already serves. **It is not an accuracy fixture**:
98 copies of one scene is a single observation repeated, and every loop seam is a hard cut
that breaks tracks and manufactures crossings that are not in the source.

### The stock labels are unverified drafts

`storefront-oblique-01` and `entrance-headon-01` carry `labelled_by: draft-unverified`.
They were derived by stepping through sampled frames rather than by watching the video end
to end, and nobody has checked them since. They are good enough to iterate a metric against
and are **not** good enough to be anybody's reference for a published number — which, being
stock, they can never be anyway.

`score(..., gating=True)` refuses both, so neither can reach a published number by accident.
Verify them in the clicker and re-save under your own name — the rater field is the
accountability mechanism, and promoting a draft is meant to be a human act rather than a
flag someone flips. The judgement calls worth re-checking are the ones a detector will also
find hard:

- **`storefront-oblique-01` @ 0.1–1.9 s** — a man walks right-to-left across the front of
  the store, over the mat, without entering. Recorded as *no crossing*. Whether that is
  right depends entirely on where the `storefront-oblique-door` line is drawn (see below).
- **`storefront-oblique-01` @ 17.0 s** — a man walks *out*. Recorded as `out`.
- Timestamps are ±0.3 s throughout.

### The calls `markos` made in `residential-lobby-01`

Verified end to end against a hand-written tally, so these are recorded decisions rather
than open questions — but they are the ones a detector will get wrong, which is what makes
the clip worth running:

- **@ 1766.5 s and 2327.4 s** — a woman leaves and later returns with a dog. One person,
  one crossing, each way. The dog is not a crossing.
- **@ 365, 449, 527, 601, 1699 and 1805 s** — the camera's own motion overlay fires with
  nobody in frame: the doorway's exposure swings, and two of those are concatenation seams.
  All recorded as *no crossing*, and every one is a place a detector can plausibly invent
  one.
- Timestamps are ±0.3 s.

### A gap this surfaced: the labeller cannot see the line

A truth file records a crossing of `line_id`, but the line's geometry lives in
`mesopic.yaml`, and the clicker does not show it. For a head-on doorway that is harmless —
the threshold is obvious. For an oblique storefront it is not: someone walking along the
pavement passes within a metre of the door, and whether they count depends on where the
line sits. Until the clicker can overlay the configured line, an oblique clip's labels are
only meaningful alongside the config they were made against.

**`mesopic.yaml` in this directory is that config**, committed beside the labels for
exactly this reason. It defines one line, `home-hallway-corridor`, on the hallway camera,
and three zones. Line ids are namespaced to the rig that owns them: `line_id` is a
PRIMARY KEY in the store and unique across a whole config, so two clips sharing one can
never both have geometry here — the config is rejected with `duplicate line_id`, and the
second clip cannot be given a line without renaming the first one's labels. A line is
geometry inside a frame, so it is shared only by clips of the same rig in the same
framing, which is what the two `home-hallway-*` clips are.

`home-entrance-01`'s line, `home-entrance-door`, is **not there yet**, so nothing can be
scored against the gate clip. It is the one thing in that file nobody can derive: which
side of the threshold is "outside" is not in the truth file, and endpoint order alone
decides the sign of all 65 crossings. Draw it against the footage with `mesopic
calibrate`. Read it before labelling anything — particularly the note on endpoint order, which
decides the sign of every crossing and is invisible in the resulting truth file.

The kitchen clip has no counting line on purpose. Nobody transits a threshold in it; the
subject moves between three stations and stands at each. A line drawn across that scene
would be measuring an event the footage does not contain.

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
mesopic truth validate fixtures/clips/*.clip.json fixtures/truth/*.truth.json
```

## Times are media time

`t_s` is seconds from the start of the clip, not a UTC instant — the one place in this
repository where "timestamps are UTC everywhere" does not apply, and it has to be.
Replaying a clip stamps every frame with the wall clock of the replay, so a UTC timestamp
in a truth file would describe the labelling session rather than the footage. Conversion
happens at exactly one boundary, inside `mesopic.truth.score`.

## Adding a clip

1. **Normalise the camera original**, which is never itself a fixture:

   ```
   scripts/normalize-clip.sh ~/Downloads/IMG_1234.MOV my-clip-01 "$MESOPIC_CLIPS_DIR"
   ```

   This produces the 1080p25 SDR BT.709 artefact the manifest hashes, and it exists
   because three things about a camera original will otherwise poison the fixture
   quietly:

   - **A recent iPhone records HLG/BT.2020 high dynamic range.** No RTSP camera ever
     hands the engine that, and fed in untouched it decodes washed-out and flat — an
     accuracy number measured against it would be precise and meaningless. The script
     tone-maps; its exposure constant was measured from the scene, not assumed, and the
     comment there says how to re-measure for a different one.
   - **The container carries where it was shot.** An iPhone `.MOV` has
     `com.apple.quicktime.location.ISO6709` in it — a GPS fix of the room, which for
     own-rig footage is somebody's home — plus an audio track of whatever was said in it.
     Both are stripped. Neither belongs in a file that gets passed around.
   - **The encode has to be reproducible**, or re-making the clip invalidates the labels.
     The script is bit-exact: same input and same ffmpeg give the same SHA-256.

   Then hash it, and never re-encode. `resolve_clip` refuses a clip that does not match
   its manifest, which is the point.

2. Write `clips/<clip_id>.clip.json`. Record the licence **and the date a human read it** —
   a licence nobody checked on a stated day is a licence nobody checked. Measure
   `mount_height_m` and `mount_angle_deg` **at the rig, with a tape, before the camera
   moves**; they are unrecoverable afterwards and they decide which reference scene the
   clip belongs to, and therefore which target it is read against.
3. Be honest about `consent.model_release`. `unknown` is the correct answer far more often
   than it is comfortable, and it costs nothing except the ability to gate on that clip.
4. Label it, and commit the truth file.
5. Run `mesopic truth validate` on both.
