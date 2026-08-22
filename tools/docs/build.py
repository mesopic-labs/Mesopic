"""Build the Mesopic docs site (P5.3).

A ~150-line static generator rather than a framework: the page set is small, and the
site's whole visual identity is the engine's own `hud.css`, which it copies rather than
forks so the darkroom palette cannot drift away from the dashboard's (P5.7).

Run it with `make docs`. Output is `site/`, which is not tracked.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import markdown
from jinja2 import Environment, FileSystemLoader, StrictUndefined

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HUD_CSS = REPO / "mesopic-engine" / "src" / "mesopic" / "api" / "static" / "hud.css"


@dataclass(frozen=True)
class Page:
    """One page in the site, and its position in the navigation."""

    slug: str
    title: str


# Order is a decision, not directory order: a reader arriving cold should meet the
# product, run it, then configure it, before anything reference-shaped.
NAV: tuple[Page, ...] = (
    Page("index", "Overview"),
    Page("quickstart", "Quickstart"),
    Page("configuration", "Configuration"),
    Page("integrations", "Integrations"),
    Page("architecture", "Architecture"),
    Page("cameras", "Cameras"),
    Page("comparison", "How it compares"),
    Page("roadmap", "Roadmap"),
)


def _renderer() -> markdown.Markdown:
    return markdown.Markdown(extensions=["fenced_code", "tables", "toc", "attr_list"])


_INCLUDE = re.compile(r"<!--include:([^#]+)#(.+?)-->")


def _section(document: str, heading: str) -> str:
    """The body of one `## heading`, up to the next one of the same level."""
    match = re.search(
        rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", document, re.MULTILINE | re.DOTALL
    )
    if match is None:
        message = f"no '## {heading}' section to include"
        raise ValueError(message)
    return re.sub(r"\n-{3,}\s*\Z", "", match.group(1)).strip()


def _read_page(source: Path, page: Page) -> str:
    """Read a page, expanding any `<!--include:FILE#Heading-->` marker it carries.

    A page the README also publishes is included from it rather than duplicated, so the
    two cannot disagree. Everything else is a real file in `docs/`.
    """

    def expand(marker: re.Match[str]) -> str:
        document = (REPO / marker.group(1)).read_text(encoding="utf-8")
        return f"## {marker.group(2)}\n\n{_section(document, marker.group(2))}"

    return _INCLUDE.sub(expand, (source / f"{page.slug}.md").read_text(encoding="utf-8"))


def _retarget_links(html: str) -> str:
    """Rewrite links that are correct on GitHub but wrong in the built site.

    `docs/*.md` is browsable on GitHub as-is, which is what makes the README's pointers
    work before the site is published. Those same paths have to be re-aimed here: a page
    link gains `.html`, and a repo path that climbs out of `docs/` is served from the
    site root instead, because `../` escapes the site entirely.
    """
    html = re.sub(r'href="(?!https?:|mailto:)([^"#]+)\.md(#[^"]*)?"', r'href="\1.html\2"', html)
    return html.replace('href="../examples/', 'href="examples/')


def build_site(*, source: Path, out: Path) -> None:
    """Render every page in `NAV` from `source` into `out`."""
    env = Environment(
        loader=FileSystemLoader(HERE / "templates"),
        autoescape=True,
        undefined=StrictUndefined,
    )
    template = env.get_template("page.html")
    md = _renderer()

    out.mkdir(parents=True, exist_ok=True)
    for page in NAV:
        md.reset()
        body = _retarget_links(md.convert(_read_page(source, page)))
        (out / f"{page.slug}.html").write_text(
            template.render(page=page, nav=NAV, body=body), encoding="utf-8"
        )

    shutil.copyfile(HUD_CSS, out / "hud.css")
    shutil.copyfile(HERE / "templates" / "docs.css", out / "docs.css")
    # The configuration page links the worked example; ship it rather than link off-site.
    shutil.copytree(REPO / "examples", out / "examples", dirs_exist_ok=True)


if __name__ == "__main__":  # pragma: no cover - the `make docs` entry point
    build_site(source=REPO / "docs", out=REPO / "site")
