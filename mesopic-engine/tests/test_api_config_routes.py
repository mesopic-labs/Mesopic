"""`/config` over HTTP: the file on the page, and the page back onto the file.

This is the second state-changing surface the engine exposes and the first one that can
change any key in the site's configuration rather than two of them, so the checks around
it are the ones P3.3 established plus one more:

* **A session is required to *read* it.** Every other view is open on the LAN, and P3.10
  deliberately gated writes only. This page is the exception because `RtspSource` may
  legitimately carry an inline `url:` — the address is the credential — so rendering the
  file to an unauthenticated GET would publish a camera password to anyone who can reach
  the port. ADR-0023.
* **A refusal shows the P2.1 message.** `describe_validation_error` is value-free by
  construction (`include_input=False`), which is what makes it safe to render — and the
  operator cannot fix a document they are only told is "invalid".
* **A save that lost a race is refused, not merged.** `/calibrate` writes the same file.
  An editor opened before a zone was redrawn holds a document without it, and saving that
  would silently undo the drawing.

Red-first for P3.4.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from mesopic.api.app import create_app
from mesopic.api.auth import Credential
from mesopic.config.schema import MesopicConfig
from mesopic.errors import ConfigError
from mesopic.store.store import Store
from mesopic.supervisor.handle import WorkerReport
from mesopic.types import CameraId, CameraState

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "mesopic.yaml"

FRONT_DOOR = CameraId("front-door")
TILL = CameraId("till")

PASSWORD = "correct-horse-battery-staple"  # noqa: S105 - a fixture, not a credential

MARKER = "# a comment the operator wrote and expects to still be here"


class FakeEngine:
    """The composition root's two halves: the file it owns, and the reload behind it.

    Real enough to be worth asserting against — it writes the document to a real file, so
    the stale-digest tests race against the same bytes the route reads.
    """

    def __init__(self, path: Path, config: MesopicConfig) -> None:
        self.path = path
        self.config = config
        self.saved: list[str] = []
        self.refuse: str | None = None

    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    async def save_config(self, document: str) -> MesopicConfig:
        self.saved.append(document)
        if self.refuse is not None:
            raise ConfigError(self.refuse)
        self.path.write_text(document, encoding="utf-8")
        self.config = MesopicConfig.model_validate(yaml.safe_load(document))
        return self.config


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "mesopic.yaml"
    path.write_text(
        f"{MARKER}\n{EXAMPLE_CONFIG.read_text(encoding='utf-8')}",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def config(config_file: Path) -> MesopicConfig:
    parsed: dict[str, Any] = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    return MesopicConfig.model_validate(parsed)


@pytest.fixture
def store(tmp_path: Path, config: MesopicConfig) -> Iterator[Store]:
    with Store(tmp_path / "mesopic.db") as store:
        store.migrate()
        store.apply_config(config)
        yield store


@pytest.fixture
def engine(config_file: Path, config: MesopicConfig) -> FakeEngine:
    return FakeEngine(config_file, config)


def _reports() -> dict[CameraId, WorkerReport]:
    return {
        FRONT_DOOR: WorkerReport(state=CameraState.STREAMING, consecutive_failures=0),
        TILL: WorkerReport(state=CameraState.STREAMING, consecutive_failures=0),
    }


def _app(config: MesopicConfig, store: Store, engine: FakeEngine, **kw: Any) -> Any:
    return create_app(
        config=config,
        store=store,
        camera_reports=_reports,
        config_document=kw.pop("config_document", engine.text),
        save_config=kw.pop("save_config", engine.save_config),
        credential=kw.pop("credential", Credential(PASSWORD)),
        **kw,
    )


@pytest.fixture
async def client(
    config: MesopicConfig, store: Store, engine: FakeEngine
) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=_app(config, store, engine))
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        await client.post("/login", data={"password": PASSWORD})
        yield client


def _squashed(html: str) -> str:
    """Rendered text with its wrapping collapsed, so an assertion is about the sentence
    rather than about where Jinja put a newline."""
    return " ".join(html.split())


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _post(client: httpx.AsyncClient, document: str, digest: str) -> httpx.Response:
    return await client.post("/config", data={"document": document, "digest": digest})


# --- Rendering ---------------------------------------------------------------


async def test_the_page_renders_the_file_on_disk(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    """The file, not a re-serialisation of the parsed config: a page that rendered
    `model_dump` would drop every comment the moment the operator pressed save."""
    response = await client.get("/config")

    assert response.status_code == 200
    assert MARKER in response.text


async def test_the_page_carries_the_digest_of_what_it_rendered(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    response = await client.get("/config")

    assert _digest(engine.text()) in response.text


async def test_the_page_is_never_cached(client: httpx.AsyncClient) -> None:
    """The one page that can hold a camera's password, so it does not go in a disk cache
    — the same rule that keeps the calibration snapshot out of one."""
    response = await client.get("/config")

    assert response.headers["cache-control"] == "no-store"


# --- Who may read it ---------------------------------------------------------


async def test_reading_the_page_requires_a_session(
    config: MesopicConfig, store: Store, engine: FakeEngine
) -> None:
    """The exception to P3.10's write-only gate, and the reason ADR-0023 exists."""
    transport = httpx.ASGITransport(app=_app(config, store, engine))
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as anonymous:
        response = await anonymous.get("/config")

    assert response.status_code == 401
    assert MARKER not in response.text


async def test_the_page_is_unavailable_when_no_credential_is_configured(
    config: MesopicConfig, store: Store, engine: FakeEngine
) -> None:
    """503 rather than 401: no session could ever satisfy it, and sending the operator to
    a login page that cannot work is worse than saying so (ADR-0019)."""
    transport = httpx.ASGITransport(app=_app(config, store, engine, credential=None))
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as anonymous:
        response = await anonymous.get("/config")

    assert response.status_code == 503
    assert MARKER not in response.text


async def test_the_page_is_unavailable_without_a_config_file(
    config: MesopicConfig, store: Store, engine: FakeEngine
) -> None:
    """An engine handed a config object has nothing to render and nowhere to write."""
    transport = httpx.ASGITransport(
        app=_app(config, store, engine, config_document=None, save_config=None)
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        await client.post("/login", data={"password": PASSWORD})
        response = await client.get("/config")

    assert response.status_code == 503


# --- Saving ------------------------------------------------------------------


async def test_a_valid_document_is_saved_and_reloaded(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    edited = engine.text().replace("fps_max: 5.0", "fps_max: 4.0")

    response = await _post(client, edited, _digest(engine.text()))

    assert response.status_code == 200
    assert engine.saved == [edited]
    assert engine.text() == edited


async def test_the_page_reflects_the_saved_document(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    """What comes back is read off the file, so a save that landed differently than the
    browser asked shows what actually landed."""
    edited = engine.text().replace("fps_max: 5.0", "fps_max: 4.0")

    response = await _post(client, edited, _digest(engine.text()))

    assert "fps_max: 4.0" in response.text


async def test_an_invalid_document_is_refused_with_the_key_path(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    before = engine.text()
    engine.refuse = f"config {engine.path} is invalid: budget.fps_min: fps_min (1.0) is above"

    response = await _post(client, before.replace("fps_max: 5.0", "fps_max: 0.5"), _digest(before))

    assert response.status_code == 400
    assert "budget.fps_min" in response.text
    assert engine.text() == before


async def test_a_refused_save_keeps_the_operators_text_on_the_page(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    """Handing back the file instead would discard the edit they are trying to fix."""
    before = engine.text()
    engine.refuse = "config is invalid: budget.fps_min: fps_min (1.0) is above fps_max (0.5)"
    rejected = before.replace("fps_max: 5.0", "fps_max: 0.5")

    response = await _post(client, rejected, _digest(before))

    assert "fps_max: 0.5" in response.text


async def test_a_stale_digest_is_refused(client: httpx.AsyncClient, engine: FakeEngine) -> None:
    """`/calibrate` writes this file too. A save from an editor opened before a zone was
    redrawn would silently undo the drawing."""
    before = engine.text()
    engine.path.write_text(f"{before}\n# drawn in the meantime\n", encoding="utf-8")

    response = await _post(client, before.replace("fps_max: 5.0", "fps_max: 4.0"), _digest(before))

    assert response.status_code == 409
    assert engine.saved == []
    assert "drawn in the meantime" in engine.text()


async def test_a_save_names_the_sections_that_need_a_restart(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    """A source is opened when the worker spawns, so a changed camera is on disk and not
    in effect. Saying nothing would leave the operator watching for a change that cannot
    come until they restart."""
    edited = engine.text().replace('name: "Front door"', 'name: "Side door"')

    response = await _post(client, edited, _digest(engine.text()))

    assert response.status_code == 200
    assert "Not until the next restart: cameras." in _squashed(response.text)


async def test_a_geometry_or_budget_save_promises_no_restart(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    """The whole point of the reload: this one is live, and the page says so plainly."""
    edited = engine.text().replace("fps_max: 5.0", "fps_max: 4.0")

    response = await _post(client, edited, _digest(engine.text()))

    assert "Saved. The running engine has adopted it." in _squashed(response.text)


async def test_a_cross_origin_save_is_refused(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    """A stolen session still cannot be spent from another host's page (ADR-0018 item 8)."""
    response = await client.post(
        "/config",
        data={"document": engine.text(), "digest": _digest(engine.text())},
        headers={"origin": "http://evil.example"},
    )

    assert response.status_code == 403
    assert engine.saved == []


async def test_saving_is_unavailable_without_an_engine_behind_it(
    config: MesopicConfig, store: Store, engine: FakeEngine
) -> None:
    transport = httpx.ASGITransport(
        app=_app(config, store, engine, config_document=None, save_config=None)
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        await client.post("/login", data={"password": PASSWORD})
        response = await _post(client, "site: {}", "0" * 64)

    assert response.status_code == 503


async def test_a_document_larger_than_the_limit_is_refused(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    """Unbounded input is unbounded memory: the rule every entry point here follows.

    413 rather than 400, matching the login route — the body is refused for its size,
    which is knowable before anything parses it or looks at what it holds.
    """
    response = await _post(client, "#" * (512 * 1024), _digest(engine.text()))

    assert response.status_code == 413
    assert engine.saved == []


# --- What a real browser does that httpx does not -----------------------------


async def test_a_document_submitted_with_crlf_lands_with_the_line_endings_it_had(
    client: httpx.AsyncClient, engine: FakeEngine
) -> None:
    """HTML says a textarea's value is normalised to CRLF on submit, and every browser
    does it. Nothing in this suite would ever see it — `httpx` sends exactly the string it
    is given — so without this the first real save silently rewrites every line ending in
    the operator's file, and the diff is the whole document.
    """
    edited = engine.text().replace("fps_max: 5.0", "fps_max: 4.0")

    response = await _post(client, edited.replace("\n", "\r\n"), _digest(engine.text()))

    assert response.status_code == 200
    # Bytes, not `read_text`: text mode translates CRLF back to LF on the way in, which
    # would hide exactly the thing this test exists to catch.
    assert b"\r" not in engine.path.read_bytes()
    assert engine.path.read_bytes().decode("utf-8") == edited


async def test_the_editor_keeps_a_leading_blank_line(
    config: MesopicConfig, store: Store, engine: FakeEngine
) -> None:
    """The other half of the same spec: a browser drops one newline immediately after the
    opening tag, so a document that starts with a blank line loses it on every round trip
    — a slow rewrite of a file nobody asked us to touch."""
    engine.path.write_text(f"\n{engine.text()}", encoding="utf-8")
    transport = httpx.ASGITransport(app=_app(config, store, engine))
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        await client.post("/login", data={"password": PASSWORD})
        response = await client.get("/config")

    body = response.text
    opened = body.index(">", body.index("<textarea")) + 1
    assert body[opened] == "\n", "the tag must be followed by the newline the browser eats"
    assert body[opened + 1] == "\n", "and then by the document's own leading newline"


async def test_a_document_holding_markup_is_escaped_and_survives_the_round_trip(
    config: MesopicConfig, store: Store, engine: FakeEngine
) -> None:
    """A YAML comment can hold anything, including a closing tag.

    Unescaped it would end the textarea early — the operator's document truncated on the
    page and their next save writing the truncation to disk — and run whatever followed
    on the one page that requires their session.
    """
    engine.path.write_text(
        f"# </textarea><script>alert(1)</script>\n{engine.text()}", encoding="utf-8"
    )
    transport = httpx.ASGITransport(app=_app(config, store, engine))
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        await client.post("/login", data={"password": PASSWORD})
        response = await client.get("/config")

    assert "<script>" not in response.text
    assert "&lt;/textarea&gt;&lt;script&gt;" in response.text
