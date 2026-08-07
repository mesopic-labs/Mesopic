## What and why

<!-- The design note: what this changes and, more importantly, why. If the decision is
     architecturally significant, link the ADR that lands with it. -->

Closes #

## Checklist

- [ ] A failing test was written first, and it now passes.
- [ ] `make check` is green locally (ruff, mypy --strict, import-linter, pytest).
- [ ] No blanket `# type: ignore` or `# noqa` — any suppression is narrow and justified inline.
- [ ] No secrets, tokens, RTSP URLs, or customer data in code, tests, fixtures, or commits.
- [ ] No code copied from elsewhere (licence contamination is not fixable after the fact).

## Invariants

- [ ] **No frame, crop, or pixel bytes** are written to disk, a column, a log, or the network.
- [ ] No new dependency — or, if there is one: package, maintainer, downloads, last release,
      transitive count and **licence** are stated below, and `uv.lock` is regenerated.
- [ ] No AGPL enters the default dependency graph (ADR-0013).
- [ ] Every new cloud query carries `WHERE tenant_id = $1`.

## Flagged changes

<!-- REQUIRED if this PR touches .github/workflows/, CODEOWNERS, branch protection,
     release scripts, the Dockerfile, LICENSE, or uv.lock. Say what changed and why. -->

- [ ] This PR touches none of the above.

## How it was verified

<!-- Commands run and what you saw. "Tests pass" is not evidence; the output is. -->
