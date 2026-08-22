"""The docs site build (P5.3).

The site is generated from `docs/*.md` into static HTML styled by the engine's own
`hud.css`, so the darkroom palette is shared rather than forked (P5.7).
"""

from __future__ import annotations

import re
from pathlib import Path

from tools.docs.build import NAV, build_site

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "docs"
TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


def test_every_page_in_the_nav_is_built(tmp_path: Path) -> None:
    """The nav is the contract: a page listed there must exist as HTML afterwards."""
    build_site(source=SOURCE, out=tmp_path)

    built = {path.name for path in tmp_path.glob("*.html")}
    assert built == {f"{page.slug}.html" for page in NAV}


def test_every_internal_link_in_the_built_site_resolves(tmp_path: Path) -> None:
    """A moved section keeps its old `(#anchor)` links, which point at a page that no
    longer holds that heading. Those resolve to nothing and look fine until clicked."""
    build_site(source=SOURCE, out=tmp_path)

    dangling: dict[str, set[str]] = {}
    for built in sorted(tmp_path.glob("*.html")):
        html = built.read_text(encoding="utf-8")
        for href in re.findall(r'href="([^"]+)"', html):
            if href.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = (tmp_path / href.split("#", 1)[0]).resolve()
            if not target.exists():
                dangling.setdefault(href, set()).add(built.name)

    assert not dangling, f"links that go nowhere: {dangling}"


def test_the_quickstart_page_is_transcluded_from_the_readme(tmp_path: Path) -> None:
    """One source for the command a launch visitor actually runs.

    The README keeps the quickstart because that is the path P5.5's external tester
    walked; the site needs it too. Retyping it into `docs/` would leave two copies of a
    `docker run` line with nothing keeping them equal, and the one that rots is the one
    nobody reads while testing.
    """
    source_text = (SOURCE / "quickstart.md").read_text(encoding="utf-8")
    assert "docker run" not in source_text, "the page holds a second copy, not an include"

    build_site(source=SOURCE, out=tmp_path)

    built = (tmp_path / "quickstart.html").read_text(encoding="utf-8")
    assert "ghcr.io/mesopic-labs/mesopic-engine:latest" in built
    assert "MESOPIC_RTSP_URL" in built


def test_the_site_takes_its_tokens_from_the_hud_system(tmp_path: Path) -> None:
    """P5.7: the docs site is the second surface built on `hud.css`, and the one most
    likely to grow a palette of its own. An undefined custom property does not error —
    `var(--gone)` inherits — so drift here renders a plausible wrong colour.
    """
    build_site(source=SOURCE, out=tmp_path)

    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", (tmp_path / "hud.css").read_text()))

    stale: dict[str, set[str]] = {}
    for asset in [TEMPLATES / "docs.css", *sorted(TEMPLATES.glob("*.html"))]:
        text = asset.read_text(encoding="utf-8")
        local = defined | set(re.findall(r"(--[a-z0-9-]+)\s*:", text))
        for token in re.findall(r"var\(\s*(--[a-z0-9-]+)", text):
            if token not in local:
                stale.setdefault(token, set()).add(asset.name)

    assert not stale, f"tokens referenced but never defined in hud.css: {stale}"


def test_every_in_page_anchor_in_the_readme_resolves() -> None:
    """The README is the site's source for its transcluded pages, so a dead anchor here
    ships twice: once on GitHub, and once on whichever page includes that section.
    GitHub silently renders a link to a heading that does not exist.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    headings = {
        re.sub(r"[^a-z0-9\s-]", "", text.strip().lower()).replace(" ", "-")
        for text in re.findall(r"^#{1,6} (.+)$", readme, re.MULTILINE)
    }
    dangling = {a for a in re.findall(r"\]\(#([a-z0-9-]+)\)", readme) if a not in headings}

    assert not dangling, f"README anchors pointing at no heading: {dangling}"
