"""Reading an HTML form's body without a dependency, and without an unbounded read.

Deliberately not FastAPI's `Form`, and not Starlette's `request.form()`: both require
`python-multipart` — the second asserts on it before it even looks at the content type —
and what arrives here is a handful of urlencoded fields that `urllib.parse` has handled
since forever. A dependency for that is the wrong trade in an MIT engine.

The size ceiling is the other half. `request.body()` accumulates the whole stream before
anything can measure it, and a chunked request carries no `content-length` for a
pre-check to read, so a limit enforced afterwards is advisory: it refuses bytes it has
already allocated. Reading the stream is what makes it true.

Extracted from P3.10's login route when P3.4 needed the same two guarantees for a much
larger body. One implementation, two callers, one limit each.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import parse_qs

if TYPE_CHECKING:
    from fastapi import Request

FORM_MEDIA_TYPE = "application/x-www-form-urlencoded"


def declared_too_large(request: Request, limit: int) -> bool:
    """Whether the caller announced a body over `limit`, before a byte is read."""
    declared = request.headers.get("content-length")
    if declared is None or not declared.isdigit():
        return False
    return int(declared) > limit


async def bounded_body(request: Request, limit: int) -> bytes | None:
    """At most `limit` bytes, or `None` — read from the stream, not into memory."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def form_field(request: Request, body: bytes, name: str) -> str | None:
    """One field from an already-bounded urlencoded body, or `None`.

    Anything that is not exactly one `name` field is `None` — a repeated field is
    ambiguous, and choosing one of the two values is how a form gets read differently
    from how it was submitted.
    """
    if not _is_form(request):
        return None
    values = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True).get(name, [])
    if len(values) != 1:
        return None
    return values[0]


def _is_form(request: Request) -> bool:
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    return media_type == FORM_MEDIA_TYPE


__all__ = ["FORM_MEDIA_TYPE", "bounded_body", "declared_too_large", "form_field"]
