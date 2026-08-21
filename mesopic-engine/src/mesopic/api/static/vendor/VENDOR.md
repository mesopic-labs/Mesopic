# Vendored front-end assets

Two third-party libraries, committed as files rather than fetched at runtime or installed
by a build step. Both are prescribed by ADR-0007 (server-rendered HTMX + Jinja + uPlot,
no SPA) and neither pulls a transitive dependency.

They are vendored, not linked, because the dashboard has to work on a box with no
outbound path: a CDN `<script>` would turn a dashboard load into an off-box request and
break the airgapped installs this product is built for. `test_the_page_loads_nothing_from_the_internet`
is what keeps that true.

| File | Library | Version | Licence | SHA-256 |
|---|---|---|---|---|
| `htmx.min.js` | [htmx](https://github.com/bigskysoftware/htmx) | 2.0.10 | 0BSD | `71ea67185bfa8c98c39d31717c6fce5d852370fcdfd129db4543774d3145c0de` |
| `uPlot.iife.min.js` | [uPlot](https://github.com/leeoniya/uPlot) | 1.6.32 | MIT | `19c8d4c6ad88929a79f4ae49d6f7161566dfd0ba3d15cc495e974f787eb78f1f` |
| `uPlot.min.css` | uPlot | 1.6.32 | MIT | `df630c6a8d6f8eeaff264b50f73ce5b114f646ffd9a0bb74f049b0a00135fa04` |

Fetched from `unpkg.com/htmx.org@2.0.10/dist/` and `unpkg.com/uplot@1.6.32/dist/`,
unmodified. Licence texts travel with the code, as both licences require:
`LICENSE-htmx.txt`, `LICENSE-uPlot.txt`.

`test_every_vendored_asset_matches_its_recorded_digest` re-computes every hash above on
each run, and `test_every_vendored_asset_is_recorded` fails if a file here is missing
from the table. Vendored code is code nobody reviews a second time; the digest is what
turns a silent substitution — a bad re-download, a stray edit, a compromised mirror —
into a red build.

## Upgrading

Download the new version, replace the file, update its row (version **and** digest), and
re-read the licence in case it changed. The tests fail until the table matches the bytes,
which is the point.
