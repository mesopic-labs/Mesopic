"""The labelling tool obeys the same rule as the product it labels for.

`tools/clicker/index.html` is opened from the filesystem and handed a video of real
people. The product's claim is that footage never leaves the building; a labelling tool
that fetched a font from a CDN would be sending nothing but a request, and would still be
the exact shape of the thing we tell customers we do not do.

So: no network, at all. No external origin, no fetch, no upload, no third-party anything.
The video is read with ``createObjectURL`` and stays in the tab.

Written red-first.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLICKER = REPO_ROOT / "tools" / "clicker" / "index.html"

# Anything that could move bytes off the page, or pull bytes onto it.
FORBIDDEN_CALLS = (
    "fetch(",
    "XMLHttpRequest",
    "WebSocket",
    "navigator.sendBeacon",
    "EventSource",
    "importScripts",
    "import(",
)

# An external origin in any attribute: src, href, a CSS url(), an @import.
EXTERNAL_ORIGIN = re.compile(r"""(?:src|href)\s*=\s*["']?(?:https?:)?//""", re.IGNORECASE)
ANY_REMOTE_URL = re.compile(r"https?://", re.IGNORECASE)


@pytest.fixture(scope="module")
def source() -> str:
    assert CLICKER.is_file(), f"{CLICKER} is missing"
    return CLICKER.read_text(encoding="utf-8")


@pytest.mark.privacy
def test_the_clicker_makes_no_network_calls(source: str) -> None:
    offenders = [call for call in FORBIDDEN_CALLS if call in source]

    assert not offenders, (
        f"the labelling tool can talk to the network: {offenders}. "
        "It is handed footage of real people; it gets no network at all."
    )


@pytest.mark.privacy
def test_the_clicker_loads_nothing_from_an_external_origin(source: str) -> None:
    assert not EXTERNAL_ORIGIN.search(source), (
        "the labelling tool loads an asset from another origin. Inline it: a request for "
        "a font is still a request made from a page holding a customer's footage."
    )


@pytest.mark.privacy
def test_the_clicker_mentions_no_remote_url_at_all(source: str) -> None:
    """Including in a comment. A commented-out endpoint is a suggestion to a future
    reader, and this is not a file that should contain suggestions of that kind."""
    assert not ANY_REMOTE_URL.search(source)


@pytest.mark.privacy
def test_the_clicker_has_no_form_that_could_submit(source: str) -> None:
    assert "<form" not in source.lower()


def test_the_clicker_reads_the_video_locally(source: str) -> None:
    """The positive half: it is genuinely doing the local-object-URL thing, not simply
    failing to do anything."""
    assert "createObjectURL" in source


def test_the_clicker_is_dependency_free(source: str) -> None:
    """One file, no build step. A `package.json` next to it would mean neither."""
    assert not (CLICKER.parent / "package.json").exists()
    assert "<script" in source.lower()
