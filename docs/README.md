# Documentation

This directory is Mesopic's published documentation — start at [index.md](./index.md).
Every page here renders on GitHub as it stands, and `make docs` builds the same set into
a static site under `site/`, styled by the engine's own `hud.css` so the docs and the
dashboard cannot drift into looking like different products.

A page whose content the [README](../README.md) also carries is **included from it**
rather than copied: `<!--include:README.md#Quickstart-->` expands at build time. There is
one copy of the quickstart command, and it is the one a launch visitor reads on GitHub.
Add a page by writing the markdown here and adding it to `NAV` in `tools/docs/build.py` —
a page missing from either side fails the build tests.

## Where new documentation goes

- **How something works** goes in the module docstring, next to the code, where it will
  be read and where it rots visibly.
- **An architecturally significant decision** — irreversible, cross-cutting, or expensive
  to undo — becomes an **ADR**, landing in the same pull request as the code that first
  relies on it. Never make a decision in code without the reasoning trail beside it, and
  never re-decide in code what an ADR already settled. ADRs are append-only: supersede,
  don't rewrite.
- **A rule contributors must follow** goes in [CONTRIBUTING.md](../CONTRIBUTING.md),
  written as prose with its reasoning.
- **A rule that can be enforced mechanically** goes in the linter, the type checker, or
  an import-linter contract — not in a document someone has to remember. `make check` is
  the source of truth for what "correct" means here.
