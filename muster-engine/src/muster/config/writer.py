"""Replace the `zones:` and `lines:` blocks of `muster.yaml`, and nothing else.

§13.1 makes the config file authoritative and the store's `zones`/`lines` tables its
compiled form, so the calibration view cannot save geometry to the store alone — the next
`apply_config` would overwrite it from the file. Saving means editing the file, which
makes the engine a writer of a document a human owns.

That is why this is a round trip rather than a dump. `yaml.safe_dump` of the whole
document would be four lines of code and would silently delete every comment in the file,
including the ones marking an RTSP URL as an env reference rather than an inline secret.
`ruamel.yaml` preserves what it did not touch; this module touches two keys.

The candidate document is validated **before** it can land. Referential integrity is a
property of the whole document — a zone naming a camera that does not exist is only
visible from the top — so the check runs against the rendered result, not against the
fragment that was passed in. A config the engine itself broke is a config that fails at
the next start, on a box nobody is watching.

Implements P3.3 (engine-architecture.md §13, §13.1).
"""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from muster.config.loader import describe_validation_error
from muster.config.schema import MusterConfig
from muster.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from muster.config.schema import LineConfig, NormalizedPoint, ZoneConfig

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
    document["zones"] = _sequence(_zone_block(zone) for zone in zones)
    document["lines"] = _sequence(_line_block(line) for line in lines)

    rendered = _render(document)
    _reject_an_invalid_result(rendered, resolved)
    _replace_atomically(resolved, rendered)


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
        MusterConfig.model_validate(yaml.safe_load(rendered))
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


__all__ = ["write_geometry"]
