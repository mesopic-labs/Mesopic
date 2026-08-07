# Contributing to Muster

Thanks for your interest in Muster — the open-source video-intelligence engine that
turns the RTSP/ONVIF cameras you already own into business sensors. The engine is
MIT-licensed and yours to run; contributions are welcome, especially:

- **Camera compatibility reports** — a make/model that works (or misbehaves).
- **Integration adapters** — new export channels, NVR/home-automation bridges.
- **Metric-accuracy validation** — MOTA/IDF1/count-MAE results on real footage.
- **Geometry and pipeline correctness** — the crown-jewel counting logic.

The hosted cloud is a separate, optional, paid tier and is **not** part of this
repository. Everything here is the free-forever engine and local dashboard.

---

## Ground rules

- Be excellent to each other. All participation is governed by our
  [Code of Conduct](./CODE_OF_CONDUCT.md).
- **Privacy is a hard invariant, not a feature.** Frames are discarded the instant
  they are processed; only anonymous foot-points and derived metrics are stored. A
  change that stores pixels, sends video off-box, or adds biometric identity in v1
  will be rejected on principle — raise it as an ADR first if you think it's needed.
- Discuss non-trivial changes before you build them (see
  [Proposing a feature](#proposing-a-feature)). Small fixes can go straight to a PR.

---

## Development environment

Muster's engine is **Python-first and CPU-only capable** — a GPU/TPU/NPU is an
optional speed-up, never a requirement — so you can develop and test the entire
correctness-critical core on a laptop with no camera and no accelerator.

**Prerequisites**

- [`uv`](https://docs.astral.sh/uv/) — manages the Python version, the virtualenv, and
  the lockfile. It is the only prerequisite that is not optional.
- Docker — for the container target and the simulated-camera dev stack. Any Docker-
  compatible runtime works; Colima and Docker Desktop are both fine on macOS.

**Setup**

```bash
git clone <repo-url> muster
cd muster
make setup          # uv sync + install the git hooks
make check          # lint, types, boundaries, tests — the same gate CI runs
```

Run the engine against any RTSP stream — a real camera, or the [simulated
one](#hardware-free-rtsp-the-mediamtx-simulation), which needs no camera and no clip:

```bash
make test-stream                      # a synthetic camera on rtsp://127.0.0.1:8554/synthetic
uv run muster run --config ./muster.yaml
```

---

## The SDLC: test-first, always

We follow a test-first SDLC for **every** change, no matter how small. In order:

1. **Design note.** State what you're changing and why. For anything with real
   design surface, write it down before code. For a significant, hard-to-reverse
   decision, that note becomes an [ADR](#architecture-decision-records-adrs).
2. **Failing tests.** Write the tests that describe the desired behaviour and watch
   them fail. Correctness-critical geometry and aggregation are pure functions of
   data — assert exact events against scripted foot-points; no camera required.
3. **Implementation.** Write the minimum code to make the tests pass.
4. **Static analysis.** `ruff` and `mypy --strict` must be clean (see below).
5. **Review.** Open a PR; it merges only behind green CI and review.

Correctness-critical code here is deliberately shaped to be testable without hardware:
geometry and aggregation are pure functions of data, so you assert exact events against
scripted foot-points with no camera, no model, and no database in the loop.

### Running tests

```bash
make test                        # the fast suite (what runs on every commit)
uv run pytest -m privacy         # just the privacy invariants
uv run pytest muster-engine/tests/test_types.py   # a single file
make cov                         # with a coverage report
```

- The base of the pyramid — geometry, aggregation, sampler, store — is **fast,
  deterministic, and hardware-free**. Property tests use `hypothesis`.
- Heavier work is marked: `slow` (golden clips, soak tests) and `integration` (needs a
  broker or a stream). Neither runs in the fast loop; `make test` excludes them.
- Coverage targets the pure-logic core hardest. A green coverage number on unasserted
  branches is a lie — test behaviour, not lines.

#### Hardware-free RTSP: the mediamtx simulation

You do **not** need a camera to test ingest, and you do not need a clip either. We
manufacture a live RTSP stream in software with `mediamtx` + `ffmpeg`, so ingest,
reconnect, and the full pipeline get automated coverage on a CPU-only runner:

```bash
make test-stream                 # serves a camera on :8554; no engine, no clip, no download
```

Two paths are published:

| Path | What it is |
|---|---|
| `rtsp://127.0.0.1:8554/synthetic` | 1080p25 H.264, generated live. **Always available.** Develop against this one. |
| `rtsp://127.0.0.1:8554/sample` | your own clip on loop, if `examples/clips/sample.mp4` exists |

`/synthetic` needs no file on disk, which is deliberate: it carries no licence question,
works offline, and is the same 1080p25 shape the M0 perf gate is specified against. If
you want *byte-identical* frames across runs — comparing perf between commits, say —
`scripts/make-sample-clip.sh` writes a synthetic clip to `examples/clips/` and `/sample`
will serve it.

Check it is up with any RTSP client:

```bash
ffprobe -rtsp_transport tcp rtsp://127.0.0.1:8554/synthetic
docker compose --profile camera down          # stop it
```

**Never point `/sample` at footage of a real space.** Clips are gitignored (`*.mp4`), but
the reason is the point rather than the mechanism: footage of real people is personal
data, and this repository has no business holding any. Labelled footage for accuracy work
is sourced and licensed separately, and never committed.

Real cameras belong to the manual hardware track, never to CI.

---

## Coding conventions

Code must be maintainable by a human first and a machine second. Non-negotiables:

- **Small, cohesive functions — max 40 lines** so a function fits on one screen.
- **Single Responsibility Principle** and a single level of abstraction per function.
- **DRY.** Reuse over copy-paste; extract shared logic.
- **Type aliases for domain values.** Use `NewType`/`TypeAlias` for meaningful
  values (`CameraId`, `ZoneId`, `StorageKey`) rather than bare `str`/`int`. These are
  load-bearing: `mypy` catches a `CameraId`/`ZoneId` mix-up for you.
- **All imports at the top of the file.** No local or lazy imports inside functions.
- **Comments explain *why*, never *what*.** Self-documenting names over prose that
  restates the code. Delete dead code; leave each file better than you found it.
- **Private methods go below public methods** within a class.
- **Limit parameters** — ideally three or fewer; pass a domain object when you need
  more.

### Static analysis (required to merge)

```bash
make check     # ruff check + ruff format --check + mypy --strict + import-linter + pytest
```

All of it must pass with zero findings. `pre-commit` runs the same tools on staged
changes; CI runs them on the full tree and blocks the merge if any fails.

**Suppressions are not a way through.** A blanket `# noqa` or `# type: ignore` will be
asked about in review. If a rule is genuinely wrong for a line, narrow it to that rule
and say why in the same comment.

### Module boundaries are enforced, not suggested

`import-linter` fails the build on architectural violations, not style ones: analytics
may not persist or serve, the detector and tracker may not know about geometry, ingest
hands over frames and nothing else, and **the cloud-sync client has no import path to a
frame**. If your change needs to cross one of those lines, the boundary is probably the
thing that is wrong — raise it before working around it.

---

## Architecture Decision Records (ADRs)

Significant, hard-to-reverse decisions are recorded as ADRs. They are **append-only**:
if a decision changes, add a new ADR that *supersedes* the old one rather than editing
history — the "why we didn't" is as valuable as the "why we did."

Write an ADR when your change touches the compute split, the privacy boundary, the
storage/sync model, licensing, the inference runtime, or anything future-you will
re-litigate at 2 a.m. Follow the standard skeleton:

```
# NNNN. Title
Status: Proposed | Accepted | Superseded by NNNN | Deprecated
Context   — the forces in play, the constraints, the numbers
Decision  — what we are doing, stated unambiguously
Consequences
  Positive — what this buys us
  Negative / tradeoffs — what we knowingly give up
Alternatives considered — what we rejected and the real reason
```

Number the file with the next free integer (`NNNN-kebab-case-title.md`). Open it as
`Proposed` and let review move it to `Accepted`.

---

## Licensing and provenance

Muster is MIT, and it has to stay cleanly MIT — a licence problem in an open-source
project is not fixable after the fact, because every downstream user has already
redistributed the result.

- **Write the code you contribute.** Do not paste code from Stack Overflow, another
  repository, a blog post, or an AI assistant's output that reproduces a specific
  existing implementation. GPL or AGPL code in an MIT repository cannot be un-shipped.
- **If you adapt something, say so in the PR** — the source, its licence, and how much.
  Small, clearly-attributed, permissively-licensed snippets are usually fine; we would
  much rather have the conversation before the merge than after.
- **New dependencies need their licence stated in the PR**, along with maintainer,
  release cadence, and transitive weight. The default install must carry **no AGPL** —
  AGPL-licensed components stay behind an opt-in extra the user chooses deliberately.
- Muster does not ship model weights. A model is fetched at runtime as a separately
  licensed artefact, which keeps its licence separable from this code. Please don't
  commit weights, clips, or fixtures containing real footage.

### Sign-off (DCO)

Contributions are accepted under the [Developer Certificate of
Origin](https://developercertificate.org/) — a short statement that you wrote the patch
or otherwise have the right to submit it under the project's licence. There is no CLA
and you keep your copyright.

Add a sign-off line to each commit with `git commit -s`:

```
Signed-off-by: Your Name <your.email@example.com>
```

---

## Branching, commits, and PRs

We work **trunk-based**: `main` is always releasable.

- **One branch per task**, branched off `main`. Keep it short-lived and focused; a PR
  should do one thing.
- **Squash-merge** into `main`. Because history is squashed, the *PR title* becomes the
  commit — write it as a clear, imperative summary (e.g. `Add MQTT export for live
  occupancy`). Reference the issue/ADR it closes in the body.
- **Green CI is a gate, not a suggestion.** Tests, `ruff`, `mypy --strict`, and the
  import boundaries all pass before merge. No red merges to `main`.
- Keep individual commits on the branch coherent while you work — they'll be squashed,
  but reviewers read them.
- **Never commit a secret.** No tokens, API keys, or RTSP URLs with real credentials, in
  code, tests, fixtures, comments, or commit messages. `gitleaks` runs in the hooks and in
  CI, but it is a safety net, not a filter. If something does land in history, say so
  immediately in the PR — rotation is required, and rewriting history alone is not enough.

---

## Proposing a feature

1. **Search first** — check open [issues](../../issues) and
   [discussions](../../discussions) so you don't duplicate work.
2. **Start a discussion** for anything non-trivial. Describe the problem and the
   business metric it serves before proposing a solution — Muster's scope is
   deliberately the "core six + two adjacencies," and features like
   loss-prevention, face recognition, pose, and ANPR are explicitly out of v1.
3. **If it's architecturally significant, draft an ADR** as part of the proposal.
4. **Then open a PR** following the SDLC above.

Camera-compatibility reports and accuracy-validation data are always welcome via
issues even without a code change — they directly shape the roadmap.

---

Thanks for helping build video-intelligence for the cameras people already own.
