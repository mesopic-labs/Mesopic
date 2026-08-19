"""`/config`: the site's configuration file, on a page, editable.

§13.1 makes `muster.yaml` authoritative — the store's tables are its compiled form and the
supervisor's geometry is compiled from it — so the honest way to edit a site is to edit
that file. This view is that, and deliberately nothing more: a textarea holding the
document, a save that validates before it lands, and a reload that swaps what a running
engine can swap.

Four decisions worth stating, because each of them could reasonably have gone the other
way:

* **The whole file, as text.** A form per section would be friendlier and could not
  express half the schema — sources, zones, a camera being added — so the operator would
  still keep a text editor open, and the two would disagree. Editing the document also
  means comments survive: they are the operator's, and `write_document` never re-renders.
* **This page requires a session to *read*.** Every other view is open on the LAN and
  P3.10 gated writes only. `RtspSource` may legitimately carry an inline `url:`, because
  for a camera the address *is* the credential — so an open `/config` would publish a
  camera password to anyone who can reach the port (ADR-0023).
* **The refusal is shown, not swallowed.** The house rule is that errors reaching a user
  are generic, and this is the one place that would make the surface useless: an operator
  told only "invalid" cannot fix a document. It is safe here because the message comes
  from `describe_validation_error`, which passes `include_input=False` — P2.1 built it
  that way precisely so a config error could be repeated without repeating the config.
* **A save carries a digest of what the page rendered.** `/calibrate` writes this same
  file, so an editor opened before a zone was redrawn holds a document without it. Saving
  that would silently undo the drawing, and silence is the part that matters.

Implements P3.4 (engine-architecture.md §13, §13.1).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from muster.api.auth import WriteGuard
from muster.api.forms import bounded_body, declared_too_large, form_field
from muster.config.changes import sections_needing_restart
from muster.config.schema import MusterConfig
from muster.errors import ConfigError

if TYPE_CHECKING:
    from fastapi.responses import Response
    from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

ConfigDocument = Callable[[], str]
"""Read `muster.yaml`'s text. `None` for an engine built without a file behind it."""

ConfigSaver = Callable[[str], Awaitable[MusterConfig]]
"""Write a whole config and hot-reload it, returning the config that landed.

Returning the *reloaded* config for the same reason `GeometrySaver` does: the file is the
authority, so what the app renders afterwards is what came back off it."""

MAX_BODY_BYTES = 512 * 1024
"""Ceiling on the submitted form, on the wire.

Comfortably larger than any real site — the worked example is under 4 KB — and small
enough that an unbounded body cannot become the box's memory. Measured on the encoded
body rather than on the document, because that is what is actually read: percent-encoding
a YAML file roughly triples its newlines and colons, and a limit that ignored that would
be a limit on something nobody sent."""


def _digest_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _submitted(request: Request) -> tuple[str, str]:
    """The two fields the editor posts, from a body bounded before it is read.

    A missing or repeated field is refused generically rather than explained: only a
    non-browser produces one, and the page it would be explained to does not exist.
    """
    if declared_too_large(request, MAX_BODY_BYTES):
        raise HTTPException(status_code=413, detail="that request was too large")
    body = await bounded_body(request, MAX_BODY_BYTES)
    if body is None:
        raise HTTPException(status_code=413, detail="that request was too large")
    document = form_field(request, body, "document")
    digest = form_field(request, body, "digest")
    if document is None or digest is None:
        raise HTTPException(status_code=400, detail="invalid request")
    return _as_submitted_by_a_browser(document), digest


def _as_submitted_by_a_browser(document: str) -> str:
    """Undo the line-ending normalisation every browser applies to a textarea.

    HTML specifies that a textarea's value is normalised to CRLF on submit, so the
    document coming back is not the document that went out even when nothing was typed.
    Written through unchanged it would rewrite every line ending in the operator's file on
    the first save — a whole-document diff for a one-key edit, and one no test driving the
    app over ASGI would ever see, because a client sends exactly the string it is given.

    The other half of the same spec is in the template: a browser eats one newline
    directly after the opening tag, so the tag is followed by one of its own.
    """
    return document.replace("\r\n", "\n")


def config_router(
    *,
    current: Callable[[], MusterConfig],
    adopt: Callable[[MusterConfig], None],
    templates: Jinja2Templates,
    config_document: ConfigDocument | None,
    save_config: ConfigSaver | None,
    guard: WriteGuard,
    same_origin: Callable[[Request], bool],
) -> APIRouter:
    """The `/config` routes, bound to one app's config cell.

    `same_origin` is passed in rather than imported so both state-changing surfaces
    demonstrably run the same check — there is one implementation, in `calibration`, and
    a second copy of a CSRF floor is how the two drift apart.
    """
    router = APIRouter()

    def _text() -> str:
        """The file's current text, or 503 — the surface is real, the file is not."""
        if config_document is None:
            raise HTTPException(status_code=503, detail="no config file is in use")
        try:
            return config_document()
        except ConfigError:
            # Names the path, which belongs in the log rather than on a page.
            logger.warning("the config file could not be read")
            raise HTTPException(status_code=503, detail="the config could not be read") from None

    def _page(
        request: Request,
        *,
        document: str,
        digest: str,
        status_code: int = 200,
        error: str | None = None,
        saved: bool = False,
        restart: tuple[str, ...] = (),
        stale: bool = False,
    ) -> Response:
        response = templates.TemplateResponse(
            request=request,
            name="config.html",
            context={
                "site_id": current().site.site_id,
                "document": document,
                "digest": digest,
                "error": error,
                "saved": saved,
                "restart": restart,
                "stale": stale,
                "can_save": save_config is not None,
            },
            status_code=status_code,
        )
        # The one page that can hold a camera's password: the same reason the calibration
        # snapshot is `no-store`, applied to text rather than to pixels.
        response.headers["Cache-Control"] = "no-store"
        return response

    @router.get("/config", dependencies=[Depends(guard)])
    async def config_page(request: Request) -> Any:
        text = _text()
        return _page(request, document=text, digest=_digest_of(text))

    @router.post("/config", dependencies=[Depends(guard)])
    async def save(request: Request) -> Any:
        """Validate, write, reload — and say which of it could not take effect yet."""
        if not same_origin(request):
            raise HTTPException(status_code=403, detail="cross-origin request refused")
        if save_config is None:
            raise HTTPException(status_code=503, detail="the engine is not running")
        document, digest = await _submitted(request)

        on_disk = _text()
        if digest != _digest_of(on_disk):
            # Re-armed with the current digest rather than the stale one: pressing save
            # again then replaces what is there, which is a decision the operator is now
            # making rather than one they made before the file moved under them.
            return _page(
                request,
                document=document,
                digest=_digest_of(on_disk),
                status_code=409,
                stale=True,
            )

        before = current()
        try:
            landed = await save_config(document)
        except ConfigError as error:
            return _page(
                request,
                document=document,
                digest=digest,
                status_code=400,
                error=str(error),
            )
        adopt(landed)
        text = _text()
        return _page(
            request,
            document=text,
            digest=_digest_of(text),
            saved=True,
            restart=sections_needing_restart(before, landed),
        )

    return router


__all__ = ["MAX_BODY_BYTES", "ConfigDocument", "ConfigSaver", "config_router"]
