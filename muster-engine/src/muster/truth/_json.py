"""Reading a ground-truth document off disk, bounded and strictly.

Both artefacts — manifests and truth files — are small JSON documents written by a human
or by the labelling tool, and both are untrusted input in the sense that matters: a
malformed one must fail loudly rather than produce a plausible number. The size limit is
here because reading an arbitrarily large file into memory is the cheapest thing to get
wrong, and it is per-caller because a manifest is a few hundred bytes while an hour of
labelled crossings is not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from muster.errors import TruthError


def read_json(path: Path, max_bytes: int) -> Any:
    """Read a size-bounded JSON document, failing as a ``TruthError`` whatever goes wrong."""
    try:
        size = path.stat().st_size
    except OSError:
        message = f"cannot read {path.name}"
        raise TruthError(message) from None
    if size > max_bytes:
        message = f"{path.name} is larger than a ground-truth document should ever be"
        raise TruthError(message)

    try:
        # `utf-8-sig` strips a byte-order mark if one is there and is plain UTF-8 if it
        # is not. Windows editors and PowerShell add a BOM to files a human saves; the
        # JSON is valid to every other tool, and failing it would send a labeller hunting
        # for a character their editor does not display.
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        message = f"{path.name} is not readable JSON"
        raise TruthError(message) from None


def parse[T: BaseModel](model: type[T], payload: Any, path: Path) -> T:
    """Validate a parsed document, reporting the field that failed and not the document.

    ``include_input=False`` is the load-bearing argument, and matches `config.loader`'s
    handling of the same problem: pydantic attaches the offending value as ``input``, and
    for a validator that runs on the *assembled model* that value is every field at once.
    A truth file holds a rater's name and a manifest's ``provenance.url`` is free text
    somebody pasted into, so a rejection that quotes the document back is a rejection that
    writes both into whatever caught it. The field path and the rule are what a labeller
    needs; the payload is what they already have on disk.
    """
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        problems = error.errors(include_url=False, include_context=False, include_input=False)
        detail = "; ".join(f"{_location(problem['loc'])}: {problem['msg']}" for problem in problems)
        message = f"{path.name} is not a valid {model.__name__}: {detail}"
        raise TruthError(message) from None


def _location(loc: tuple[int | str, ...]) -> str:
    """``crossings.3.direction`` — the key path a human edits, not a pydantic repr."""
    return ".".join(str(part) for part in loc) or "<document>"
