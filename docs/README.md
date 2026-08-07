# Documentation

This directory is where Muster's published documentation will live — the quickstart,
configuration reference, integration guides, camera compatibility matrix, and the
architecture and decision records that are useful to someone running or extending the
engine. It is built into a static site as part of the launch milestone.

Until then, the documentation that exists is in the [README](../README.md):
what Muster measures, how to run it, how to configure cameras, lines, and zones, and
what it does and does not store about people.

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
- **A rule an AI coding agent must follow** goes in [CLAUDE.md](../CLAUDE.md) — short,
  imperative, and only if it cannot be enforced mechanically. A rule a linter can enforce
  belongs in the linter, not in a document someone has to remember.
