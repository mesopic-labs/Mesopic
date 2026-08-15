"""Read `muster.yaml` from disk into a validated `MusterConfig`.

`yaml.safe_load` only — never `yaml.load`. The path is canonicalised and checked for
containment before it is opened.

Every failure leaves here as a `ConfigError` whose message was **built by this module**,
never borrowed from the underlying exception, and the underlying exception is suppressed
rather than chained. A camera's RTSP URL carries its credentials, and both `pydantic`'s
`input_value=` rendering and `yaml`'s "problem" snippet quote the offending input back —
which any traceback or `logger.exception` call would then print. Reporting only the
*location* of a problem is the one formulation that cannot leak its *value*.

Implements P2.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from muster.config.schema import MusterConfig
from muster.errors import ConfigError

MAX_CONFIG_BYTES = 1 << 20
"""A site description is a page of YAML. Anything larger is a mistake or an attack."""

MAX_REPORTED_PROBLEMS = 10
"""Enough to fix a config in one pass, few enough that the error stays readable."""


def load_config(path: Path) -> MusterConfig:
    """Parse and validate a config file.

    Raises:
        ConfigError: the file is missing, unparseable, or fails validation.
    """
    document = _read_document(path)
    try:
        return MusterConfig.model_validate(document)
    except ValidationError as error:
        raise ConfigError(_describe_validation_error(path, error)) from None


def _read_document(path: Path) -> dict[str, Any]:
    """Canonicalise, read and parse the file, without quoting its contents on failure."""
    resolved = path.expanduser().resolve()
    try:
        size = resolved.stat().st_size
        if size > MAX_CONFIG_BYTES:
            msg = f"config {resolved} is {size} bytes, over the {MAX_CONFIG_BYTES}-byte limit"
            raise ConfigError(msg)
        text = resolved.read_text(encoding="utf-8")
    except OSError as error:
        msg = f"cannot read config {resolved}: {error.strerror}"
        raise ConfigError(msg) from None

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        msg = f"config {resolved} is not valid YAML{_yaml_location(error)}"
        raise ConfigError(msg) from None

    if not isinstance(document, dict):
        msg = f"config {resolved} must be a YAML mapping at the top level"
        raise ConfigError(msg)
    return document


def _yaml_location(error: yaml.YAMLError) -> str:
    """Where the parse failed — never *what* was there."""
    if not isinstance(error, yaml.MarkedYAMLError) or error.problem_mark is None:
        return ""
    mark = error.problem_mark
    return f" at line {mark.line + 1}, column {mark.column + 1}"


def _describe_validation_error(path: Path, error: ValidationError) -> str:
    problems = error.errors(include_url=False, include_context=False, include_input=False)
    reported = problems[:MAX_REPORTED_PROBLEMS]
    detail = "; ".join(f"{_location(problem['loc'])}: {problem['msg']}" for problem in reported)
    if len(problems) > len(reported):
        detail += f"; and {len(problems) - len(reported)} more"
    return f"config {path} is invalid: {detail}"


def _location(loc: tuple[int | str, ...]) -> str:
    """`cameras.0.source.url` — the key path a human edits, not a pydantic repr."""
    return ".".join(str(part) for part in loc) or "<root>"
