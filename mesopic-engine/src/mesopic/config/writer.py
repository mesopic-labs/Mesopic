"""Write `mesopic.yaml`: two keys of it from the canvas, or all of it from the editor.

`write_geometry` replaces the `zones:` and `lines:` blocks and nothing else.
`write_document` replaces the whole file with text an operator typed. They share the two
guards below — validate the candidate, then rename it into place — because those are the
properties that make the engine safe to let near a document a human owns, and a second
copy of either is a second chance to forget one.

§13.1 makes the config file authoritative and the store's `zones`/`lines` tables its
compiled form, so the calibration view cannot save geometry to the store alone — the next
`apply_config` would overwrite it from the file. Saving means editing the file, which
makes the engine a writer of a document a human owns.

That is why the geometry save is a round trip rather than a dump. `yaml.safe_dump` of the
whole document would be four lines of code and would silently delete every comment in the
file, including the ones marking an RTSP URL as an env reference rather than an inline
secret. `ruamel.yaml` preserves what it did not touch; that path touches two keys.

The editor needs none of that. What the operator submitted *is* the document, so their
comments and layout survive by being written back unchanged rather than re-rendered.

The candidate document is validated **before** it can land. Referential integrity is a
property of the whole document — a zone naming a camera that does not exist is only
visible from the top — so the check runs against the rendered result, not against the
fragment that was passed in. A config the engine itself broke is a config that fails at
the next start, on a box nobody is watching.

Implements P3.3 and P3.4 (engine-architecture.md §13, §13.1).
"""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from mesopic.config.loader import describe_validation_error
from mesopic.config.schema import MesopicConfig
from mesopic.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from mesopic.config.schema import LineConfig, NormalizedPoint, ZoneConfig

MAX_LINE_WIDTH = 4096
"""Wide enough that a polygon is never wrapped.

ruamel's default of 80 columns folds a long flow sequence across lines, which is valid
YAML and unreadable geometry — the one thing this module exists to avoid."""


def write_geometry(path: Path, *, zones: Sequence[ZoneConfig], lines: Sequence[LineConfig]) -> None:
    """Save geometry into an existing config, preserving everything around it.

    Raises:
        ConfigError: the file cannot be read, or the result would not load.
    """
    resolved = path.expanduser().resolve()
    document = _load_round_trip(resolved)
    _replace_block(document, "zones", [_zone_block(zone) for zone in zones])
    _replace_block(document, "lines", [_line_block(line) for line in lines])

    rendered = _render(document)
    _reject_an_invalid_result(rendered, resolved)
    _replace_atomically(resolved, rendered)


def read_document(path: Path) -> str:
    """The config file's text, exactly as it is on disk.

    The one place an `OSError` around the config becomes a `ConfigError`, so the route
    rendering the editor has a single failure to handle rather than two.
    """
    resolved = path.expanduser().resolve()
    try:
        return resolved.read_text(encoding="utf-8")
    except OSError as error:
        msg = f"cannot read config {resolved}: {error.strerror}"
        raise ConfigError(msg) from None


def write_document(path: Path, text: str) -> None:
    """Save a whole config an operator edited, or refuse it whole.

    Raises:
        ConfigError: the text is not YAML, would not load as a config, or cannot be
            written. The message names the failing key path and never the value under it.
    """
    resolved = path.expanduser().resolve()
    try:
        candidate = yaml.safe_load(text)
    except yaml.YAMLError:
        # Deliberately not the parser's own message: it quotes the offending line, and
        # the offending line may be the one holding an RTSP URL.
        msg = f"config {resolved} is not valid YAML"
        raise ConfigError(msg) from None
    _reject_an_invalid_config(candidate, resolved)
    _replace_atomically(resolved, text)


def _replace_block(document: Any, key: str, blocks: list[CommentedMap]) -> None:
    """Swap one geometry block, dropping an introducing comment when nothing is left.

    ruamel keeps a comment attached to the key when the value under it is replaced, which
    is exactly what preserves an operator's annotation across a redraw. An **empty**
    replacement is the one case where that is wrong: the comment now introduces items that
    do not exist, and ruamel renders it as the key's own value line with the `[]` dangling
    at column zero. That is not YAML, so the file the engine just wrote fails to parse at
    the next start — on a box nobody is watching.

    Dropping the comment loses an operator's note in that one case, which is the same
    trade ADR-0018 already accepts for a rewritten shape, and strictly better than a
    config that will not load.
    """
    document[key] = _sequence(blocks)
    if not blocks:
        document.ca.items.pop(key, None)


def _round_trip() -> YAML:
    yaml_io = YAML(typ="rt")
    yaml_io.preserve_quotes = True
    yaml_io.width = MAX_LINE_WIDTH
    # Matches the indentation every document we publish already uses, so a save does not
    # reformat the untouched half of the file into a diff nobody asked for.
    yaml_io.indent(mapping=2, sequence=4, offset=2)
    return yaml_io


def _load_round_trip(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        msg = f"cannot read config {path}: {error.strerror}"
        raise ConfigError(msg) from None
    try:
        return _round_trip().load(text)
    except yaml.YAMLError:
        msg = f"config {path} is not valid YAML"
        raise ConfigError(msg) from None


def _render(document: Any) -> str:
    stream = io.StringIO()
    _round_trip().dump(document, stream)
    return stream.getvalue()


def _reject_an_invalid_result(rendered: str, path: Path) -> None:
    """Validate what would land, by the same path a user's file takes."""
    try:
        candidate = yaml.safe_load(rendered)
    except yaml.YAMLError:
        # The engine mangling the document itself, rather than being handed a bad one.
        # Caught here because this is the last point before the rename: without it the
        # parser error escapes the save route as a 500, and the guard whose whole job is
        # "a config the engine broke never lands" would let exactly that through.
        msg = f"the engine produced a config for {path} that is not valid YAML"
        raise ConfigError(msg) from None
    _reject_an_invalid_config(candidate, path)


def _reject_an_invalid_config(candidate: Any, path: Path) -> None:
    """Refuse a parsed document that is not a config, naming where and not what."""
    try:
        MesopicConfig.model_validate(candidate)
    except ValidationError as error:
        raise ConfigError(describe_validation_error(path, error)) from None


def _replace_atomically(path: Path, text: str) -> None:
    """Write beside the target, then rename over it.

    A config half-written by a crashed save is a box that will not start, and the operator
    finds out at the next restart rather than now.
    """
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        msg = f"cannot write config {path}: {error.strerror}"
        raise ConfigError(msg) from None


def _zone_block(zone: ZoneConfig) -> CommentedMap:
    return CommentedMap(
        zone_id=str(zone.zone_id),
        camera_id=str(zone.camera_id),
        role=str(zone.role.value),
        polygon=_sequence((_point(point) for point in zone.polygon), flow=True),
        metrics=_sequence((str(metric) for metric in zone.metrics), flow=True),
    )


def _line_block(line: LineConfig) -> CommentedMap:
    return CommentedMap(
        line_id=str(line.line_id),
        camera_id=str(line.camera_id),
        a=_point(line.a),
        b=_point(line.b),
        positive_dir=str(line.positive_dir.value),
        metrics=_sequence((str(metric) for metric in line.metrics), flow=True),
    )


def _point(point: NormalizedPoint) -> CommentedSeq:
    return _sequence((float(coordinate) for coordinate in point), flow=True)


def _sequence(items: Iterable[Any], *, flow: bool = False) -> CommentedSeq:
    """A sequence node, block by default and flow where §13.1 writes it flow."""
    sequence = CommentedSeq(items)
    sequence.fa.set_flow_style() if flow else sequence.fa.set_block_style()
    return sequence


__all__ = ["read_document", "write_document", "write_geometry"]
