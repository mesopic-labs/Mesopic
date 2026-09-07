"""The AGPL boundary, asserted against the lockfile rather than trusted to a deny-list.

ADR-0013's rule is that the **default install carries no AGPL**: MIT engine, Apache-2.0
detector, MIT runtime. Ultralytics is reachable only across an opt-in boundary — the
`mesopic[ultralytics]` extra — and installing it is the user's own act.

CI's `dependency review` job denies AGPL licences, and on its own it cannot express that
rule. `uv.lock` is one file describing *every* extra, so ultralytics appears in it by
construction; the action reads the lockfile, sees an AGPL package in the diff, and fails a
bump that is entirely legitimate. Allowing those two packages there is the only way to let
the bump through — and an allow-list with nothing behind it would also let ultralytics
migrate into the default dependencies unnoticed, which is the one thing the deny-list was
protecting.

So this is what actually holds the line: the *shape of the graph*, walked from the root
package's own dependencies. The allow-list says "these two may appear in the lockfile";
this says "and only ever behind the extra".

Implements part of ADR-0013.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

LOCKFILE = Path(__file__).resolve().parents[2] / "uv.lock"
"""The workspace lockfile, two levels up: the engine is one package in it."""

ROOT_PACKAGE = "mesopic"

COPYLEFT_PACKAGES = frozenset({"ultralytics", "ultralytics-platform"})
"""The AGPL-licensed packages this repository knowingly locks, and the exact set the CI
job allows. `ultralytics-platform` arrived as a transitive dependency of ultralytics
8.4.137; it is named here so that its arrival was a decision rather than a diff nobody
read."""


def _packages() -> dict[str, dict[str, Any]]:
    locked = tomllib.loads(LOCKFILE.read_text(encoding="utf-8"))
    return {package["name"]: package for package in locked["package"]}


def _default_closure(packages: dict[str, dict[str, Any]]) -> set[str]:
    """Every package a plain `pip install mesopic` would resolve to.

    Walks `dependencies` only. `optional-dependencies` is what an extra adds, and skipping
    it here is the whole point: this closure is the default install, and the extras are
    outside it by definition.
    """
    seen: set[str] = set()
    frontier = [ROOT_PACKAGE]
    while frontier:
        name = frontier.pop()
        if name in seen or name not in packages:
            continue
        seen.add(name)
        frontier.extend(dep["name"] for dep in packages[name].get("dependencies", []))
    return seen


def test_no_copyleft_package_is_in_the_default_install() -> None:
    """ADR-0013's actual promise, as a property of the dependency graph.

    Not "ultralytics is absent from the lockfile" — it is present, and has to be, because
    one lockfile covers every extra. The claim is narrower and stronger: nothing in the
    default resolution reaches it.
    """
    reachable = _default_closure(_packages())

    assert reachable & COPYLEFT_PACKAGES == set(), (
        "an AGPL package is reachable from the default dependencies — "
        "the opt-in boundary of ADR-0013 is broken"
    )


def test_ultralytics_is_reachable_through_its_extra() -> None:
    """The other half, and the reason the first test can never pass vacuously.

    A rename, a removed extra or a typo would make the closure test pass by there being
    nothing to find. This fails in exactly those cases.
    """
    root = _packages()[ROOT_PACKAGE]
    extra = root["optional-dependencies"]["ultralytics"]

    assert [dependency["name"] for dependency in extra] == ["ultralytics"]


def test_the_workflow_allows_exactly_the_packages_named_here() -> None:
    """The list CI allows and the list this file guards must be the same list.

    If a bump introduces a third AGPL package, the `dependency review` job fails on it —
    which is correct — and whoever adds it to the workflow's allow-list has to add it here
    too, where the closure test then proves it stayed behind the boundary.
    """
    workflow = (Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    allowed = set(re.findall(r"pkg:pypi/([A-Za-z0-9._-]+)", workflow))

    assert allowed == COPYLEFT_PACKAGES


def test_the_closure_would_catch_a_migration_into_the_default_install() -> None:
    """Mutation evidence, because the real graph is the passing case.

    A test that only ever sees a correct lockfile cannot distinguish "the boundary holds"
    from "the walk stopped early". This hands the same walk a graph where the extra's
    package has moved into the root's own dependencies, and requires it to notice.
    """
    migrated = {
        ROOT_PACKAGE: {"dependencies": [{"name": "ultralytics"}]},
        "ultralytics": {"dependencies": [{"name": "ultralytics-platform"}]},
        "ultralytics-platform": {},
    }

    assert _default_closure(migrated) & COPYLEFT_PACKAGES == COPYLEFT_PACKAGES
