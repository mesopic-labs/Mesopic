"""`mesopic doctor`: what this box can do, and whether it can see the camera.

The first thing to ask a self-hoster for when an install misbehaves, and — per
[ADR-0013](../../../../Mesopic-docs/docs/01-architecture/adr/0013-permissive-detector-fallback.md)
decision 5 — the surface that makes the detector's licence *visible* rather than buried
in a cache directory. A commercial adopter should be able to see what they are running
without reading the source.

The command has two halves and they fail independently. The host report is always safe to
run: it reads the box and the cache and never touches the network, never downloads a
model, and never opens a socket. The camera preflight is opt-in — pass a URL and it says
whether the stream opens, at what resolution and rate, and, when it does not, what to go
and check.

**An RTSP URL is a credential.** It carries the camera's username and password, so this
module treats every string it can emit as suspect:

* `redact_url` is the only way a URL becomes printable. It keeps the host, port and path
  — which is the whole diagnostic value, and the half a typo lands in — and replaces the
  userinfo and the entire query string with `***`.
* `scrub` is the belt to that braces. Third-party error text is where a URL usually
  escapes (FFmpeg embeds the URL it was handed in its own messages), so anything derived
  from an exception is rewritten before it is stored on a report, and `render` rewrites
  every line again on the way out. Neither trusts the other to have done it.

Implements the ADR-0013 half of P1; the camera preflight is the `probe` half of FR-E10.
"""

from __future__ import annotations

import os
import platform
import re
import socket
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from mesopic import __version__
from mesopic.detector.model_manager import DEFAULT_MODEL, MODELS, ModelManager
from mesopic.detector.runtime import available_runtimes, resolve_runtime
from mesopic.ingest.source import FrameSource
from mesopic.types import DecodedFrame, Runtime

REDACTED = "***"
"""What stands in for anything that might be a credential. Never the empty string: a
reader has to be able to see that something *was* removed."""

UNKNOWN_LICENCE = "unknown"
"""The licence of a model that is not in the registry. Deliberately not blank, and
deliberately not a guess — an unrecognised model is one whose terms we cannot state."""

SAMPLE_FRAMES = 10
"""Frames the preflight decodes before it answers.

Enough for the rate between the first and the last to be a reading rather than a jitter
measurement, and few enough that a 25 fps camera is done in under half a second.
"""

ENGINE_LICENCE = "MIT"
"""This repository's own licence, printed beside the model's because ADR-0013's claim is
about the *chain*: MIT engine + Apache-2.0 detector + MIT runtime, no AGPL anywhere."""

RTSP_DEFAULT_PORT = 554
"""Assumed when the URL names no port, exactly as an RTSP client would assume it."""

_MIN_STANDALONE_SECRET = 6
"""Shortest password removed from text on its own, without the URL around it.

A guard against the cure being worse than the disease. Blind substring removal of a
password of `1` would rewrite "Server returned 401" as "Server returned 4***", destroying
the diagnostic to protect a secret the surrounding text had not actually disclosed. Above
this length a verbatim match is the password rather than a coincidence.
"""

_MIN_FRAMES_FOR_RATE = 2
"""Two timestamps make a spacing; one makes none."""

REACH_TIMEOUT_S = 2.0
"""Explicit, always. The probe runs after a connection has already failed slowly, and a
diagnostic that hangs is one the user kills before it prints anything."""


# --- redaction --------------------------------------------------------------


_URL_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s'\"]+")
"""Anything shaped like a URL, wherever it turns up in text we did not write."""


def redact_url(url: str) -> str:
    """A URL with its credentials removed, or `***` if it cannot be taken apart.

    The host, port and path survive on purpose. They are what the operator needs — "it
    tried *that* address and *that* stream path" is most of a camera diagnosis, and the
    stream path is where the common typo lives. The userinfo is the secret, and so is the
    query string: Dahua-style URLs carry the login in `?user=&password=`, and picking
    which parameters are safe would be guesswork applied to a credential.

    A string that will not parse returns `***` whole. There is no safe partial answer to
    give about a string we could not decompose.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
        has_userinfo = bool(parts.username or parts.password)
    except ValueError:
        # Malformed authority — an unclosed IPv6 bracket, a non-numeric port.
        return REDACTED

    if not parts.scheme or not host:
        return REDACTED

    authority = f"{REDACTED}@{host}" if has_userinfo else host
    if port is not None:
        authority = f"{authority}:{port}"
    query = f"?{REDACTED}" if parts.query else ""
    return f"{parts.scheme}://{authority}{parts.path}{query}"


def scrub(text: str, *, url: str | None = None) -> str:
    """Rewrite every URL in `text`, and remove `url`'s credentials however they are spelled.

    Two passes because they catch different leaks. The pattern pass handles the usual
    case — FFmpeg quoting the whole URL back inside an error string — while the exact
    pass handles a library that printed the credentials on their own, without the scheme
    that the pattern keys on.
    """
    scrubbed = _URL_PATTERN.sub(lambda match: redact_url(match.group()), text)
    for secret in _secrets_in(url):
        scrubbed = scrubbed.replace(secret, REDACTED)
    return scrubbed


def _secrets_in(url: str | None) -> tuple[str, ...]:
    """The literal substrings of `url` that must never survive, longest first."""
    if url is None:
        return ()
    try:
        parts = urlsplit(url)
        username, password = parts.username, parts.password
    except ValueError:
        # Unparseable, so nothing finer than the whole string can be identified.
        return (url,)

    if not username and not password:
        return (url,)

    secrets = [url]
    userinfo = username or ""
    if password:
        userinfo = f"{userinfo}:{password}"
        # The password on its own, for a library that names it without the URL around it.
        # Not the username: it is far likelier to collide with ordinary words in a
        # diagnostic ("user", "admin") and far less damaging when it survives.
        if len(password) >= _MIN_STANDALONE_SECRET:
            secrets.append(password)
    secrets.append(userinfo)
    # Longest first, so a shorter secret never carves a hole out of a longer one and
    # leaves the remainder unmatched.
    return tuple(sorted(secrets, key=len, reverse=True))


# --- the host report --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelReport:
    """The active detector: where its artefact lives and what terms it came under."""

    name: str
    licence: str
    cache_dir: Path
    cached: bool
    size_bytes: int | None


@dataclass(frozen=True, slots=True)
class HostReport:
    """What this box is, and what it can execute."""

    version: str
    platform: str
    machine: str
    python: str
    cpu_count: int | None
    runtimes: tuple[Runtime, ...]
    selected_runtime: Runtime
    selected_explicit: bool
    model: ModelReport


def host_report(
    *,
    cache_dir: Path,
    model_name: str = DEFAULT_MODEL,
    env: Mapping[str, str] | None = None,
) -> HostReport:
    """Describe the box. Reads only — no network, no download, no camera.

    `env` is injectable for the same reason `resolve_runtime` takes it: the runtime
    override is an environment variable, and a test should not have to mutate the
    process to exercise it.
    """
    selection = resolve_runtime(env=env)
    return HostReport(
        version=__version__,
        platform=platform.platform(),
        machine=platform.machine(),
        python=sys.version.split()[0],
        cpu_count=os.cpu_count(),
        runtimes=available_runtimes(),
        selected_runtime=selection.runtime,
        selected_explicit=selection.explicit,
        model=_model_report(cache_dir, model_name, selection.runtime),
    )


def _model_report(cache_dir: Path, model_name: str, runtime: Runtime) -> ModelReport:
    """What the cache holds, and the licence regardless of whether it holds anything.

    The licence is a property of the model this box *would* run, not of the bytes that
    happen to be on disk, so a fresh install that has not downloaded anything yet still
    tells its operator what they are about to accept.
    """
    spec = MODELS.get(model_name)
    if spec is None:
        return ModelReport(
            name=model_name,
            licence=UNKNOWN_LICENCE,
            cache_dir=cache_dir,
            cached=False,
            size_bytes=None,
        )

    artefact = ModelManager(cache_dir).locate(model_name, runtime=runtime)
    return ModelReport(
        name=spec.name,
        licence=spec.licence,
        cache_dir=cache_dir,
        cached=artefact is not None,
        size_bytes=artefact.path.stat().st_size if artefact is not None else None,
    )


# --- the camera preflight ---------------------------------------------------


Reach = Callable[[str, int], bool]
"""Whether a TCP connection to `(host, port)` is accepted. Injected so the preflight's
failure paths are testable without touching a network."""


@dataclass(frozen=True, slots=True)
class CameraReport:
    """One camera's preflight. `target` is redacted at construction, never after.

    `reachable` is three-valued on purpose: `None` means the address could not be probed
    (an unparseable URL), which is not the same claim as "nothing is there".
    """

    target: str
    opened: bool
    reachable: bool | None
    width: int | None
    height: int | None
    fps: float | None
    failure: str | None
    hint: str | None


_NO_FRAMES = "the stream opened and ended without producing a frame"

_HINTS: tuple[tuple[str, str], ...] = (
    # Ordered: the first needle found wins, so the specific status codes come before the
    # generic exception names that carry them.
    ("401", "the camera rejected the credentials — check the username and password"),
    ("Unauthorized", "the camera rejected the credentials — check the username and password"),
    ("PermissionError", "the camera rejected the credentials — check the username and password"),
    (
        "404",
        "the camera answered but has no stream at that path — vendors differ: "
        "/stream1, /Streaming/Channels/101, /cam/realmonitor?channel=1&subtype=0",
    ),
    (
        "NotFound",
        "the camera answered but has no stream at that path — vendors differ: "
        "/stream1, /Streaming/Channels/101, /cam/realmonitor?channel=1&subtype=0",
    ),
    (
        "ConnectionRefused",
        "nothing is listening on that port — check the port number and that the "
        "camera's RTSP service is switched on",
    ),
    (
        "Timeout",
        "the camera did not answer in time — check the IP address, the route to it, "
        "and any firewall in between",
    ),
    (
        "InvalidData",
        "the connection opened but nothing parsed as a video stream — check that this is "
        "really the camera's RTSP port, and that the profile's codec is H.264 or H.265",
    ),
    (
        "stream ended",
        "the camera accepted the connection and then closed it without sending a frame "
        "— check that the profile is enabled and not already at its session limit",
    ),
)

_UNREACHABLE_HINT = (
    "nothing accepted a connection on that port — check the IP address, the port, and "
    "any firewall between this box and the camera"
)

_REACHABLE_HINT = (
    "something answered on that port but the stream would not start — check the username "
    "and password, and the stream path: vendors differ (/stream1, "
    "/Streaming/Channels/101, /cam/realmonitor?channel=1&subtype=0)"
)

_UNPROBED_HINT = (
    "check that the address and port are reachable, that the username and password are "
    "right, and that the stream path is the one this camera uses"
)


def tcp_reachable(host: str, port: int, *, timeout_s: float = REACH_TIMEOUT_S) -> bool:
    """Whether anything accepts a TCP connection at `host:port`.

    The one question about a camera that can be answered without a credential, and the
    one FFmpeg is worst at answering: it collapses a refused connection, a rejected
    password and a missing stream path into the same opaque error class, and
    `RtspFrameSource` keeps only that class name because the message carries the URL.
    Splitting "the box is not there" from "the box is there and said no" turns a dead end
    into a next step.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        # Refused, filtered, unroutable, unresolvable — all of them mean the same thing
        # to the operator: nothing is answering where they pointed us.
        return False


def preflight(
    open_source: Callable[[], FrameSource],
    *,
    url: str,
    sample_frames: int = SAMPLE_FRAMES,
    reach: Reach = tcp_reachable,
) -> CameraReport:
    """Open a camera, decode a few frames, and report what happened — never the URL.

    The factory is passed rather than an open source so that a failure to *connect* is
    inside this function's error handling. That is where most preflights end, and it is
    the path on which a URL would otherwise reach a traceback.

    Nothing raises out of here. A diagnostic that propagates is a diagnostic that prints
    a stack trace, and the frames on that stack hold the URL.
    """
    target = redact_url(url)
    frames: list[DecodedFrame] = []
    failure: Exception | None = None
    source: FrameSource | None = None

    try:
        source = open_source()
        for frame in source.frames():
            frames.append(frame)
            if len(frames) >= sample_frames:
                break
    except Exception as error:  # noqa: BLE001 - see the docstring: nothing may escape here
        failure = error
    finally:
        # A camera enforces a session limit, and a preflight that walked away holding one
        # would cost the next real run its connection.
        if source is not None:
            source.close()

    if frames:
        return _opened(target, frames)
    return _failed(target, failure, url=url, reach=reach)


def _opened(target: str, frames: list[DecodedFrame]) -> CameraReport:
    return CameraReport(
        target=target,
        opened=True,
        # It carried a video stream out of that port. Asking a socket whether the port is
        # open would be asking a question already answered.
        reachable=True,
        width=frames[0].width,
        height=frames[0].height,
        fps=_rate_of(frames),
        failure=None,
        hint=None,
    )


def _failed(target: str, error: Exception | None, *, url: str, reach: Reach) -> CameraReport:
    detail = _NO_FRAMES if error is None else scrub(f"{type(error).__name__}: {error}", url=url)
    reachable = _reachability_of(url, reach)
    return CameraReport(
        target=target,
        opened=False,
        reachable=reachable,
        width=None,
        height=None,
        fps=None,
        failure=detail,
        hint=_hint_for(detail, reachable),
    )


def _reachability_of(url: str, reach: Reach) -> bool | None:
    """Probe the address the URL names, or `None` when it names none we can read."""
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    if not host:
        return None
    return reach(host, RTSP_DEFAULT_PORT if port is None else port)


def _rate_of(frames: list[DecodedFrame]) -> float | None:
    """The stream's own frame rate, read off capture timestamps rather than the clock.

    `DecodedFrame.ts` is the stream's presentation clock anchored to UTC at connect, so
    this is the rate the *camera* is sending at — not the rate the engine would sample at,
    and not a measure of this box's speed. A single frame has a timestamp but no spacing,
    and no rate can honestly be read off it.
    """
    if len(frames) < _MIN_FRAMES_FOR_RATE:
        return None
    span = (frames[-1].ts - frames[0].ts).total_seconds()
    if span <= 0:
        return None
    return (len(frames) - 1) / span


def _hint_for(detail: str, reachable: bool | None) -> str:
    """Turn a failure into something to go and check.

    The error text is tried first and is matched against the *scrubbed* string, so the
    needles are only ever compared with something already safe to print; they are
    exception class names and status codes, never message prose, which FFmpeg rewords
    between versions. When the text says nothing useful — which, on a real camera, is
    most of the time — the TCP probe still narrows it to one of two halves.
    """
    for needle, hint in _HINTS:
        if needle in detail:
            return hint
    if reachable is None:
        return _UNPROBED_HINT
    return _REACHABLE_HINT if reachable else _UNREACHABLE_HINT


# --- rendering --------------------------------------------------------------

_LABEL_WIDTH = 12


def render(host: HostReport, camera: CameraReport | None = None) -> list[str]:
    """The report as lines for a terminal — the last boundary before stdout.

    Every line goes through `scrub` again on the way out. The reports are built by code
    that already redacts, and this does not trust it: one boundary that cannot be
    bypassed is worth more than a rule every future caller has to remember.
    """
    lines = [
        f"mesopic {host.version}  (engine licence: {ENGINE_LICENCE})",
        _field("python", host.python),
        _field("platform", host.platform),
        _field("machine", host.machine),
        _field("cpu cores", "unknown" if host.cpu_count is None else str(host.cpu_count)),
        "",
        "detector",
        _field("runtime", _runtime_summary(host)),
        _field("available", ", ".join(runtime.value for runtime in host.runtimes)),
        _field("model", host.model.name),
        _field("licence", host.model.licence),
        _field("cache", str(host.model.cache_dir)),
        _field("cached", _cache_summary(host.model)),
    ]
    if camera is not None:
        lines += ["", "camera", *_camera_lines(camera)]
    return [scrub(line) for line in lines]


def _field(label: str, value: str) -> str:
    return f"  {label.ljust(_LABEL_WIDTH)}{value}"


def _runtime_summary(host: HostReport) -> str:
    """Name the runtime, whether it was asked for, and whether this box can run it.

    The combination is the point. An operator who set `MESOPIC_DETECTOR_RUNTIME` and got
    something else has a performance mystery with nothing anywhere to explain it
    (ADR-0012), and this is the line that explains it.
    """
    asked = "requested" if host.selected_explicit else "default"
    usable = "available" if host.selected_runtime in host.runtimes else "NOT available on this box"
    return f"{host.selected_runtime.value} ({asked}, {usable})"


def _cache_summary(model: ModelReport) -> str:
    if not model.cached:
        return "no — the first run downloads it"
    size = "" if model.size_bytes is None else f"  ({model.size_bytes / 1_048_576:.1f} MB)"
    return f"yes{size}"


def _camera_lines(camera: CameraReport) -> list[str]:
    lines = [_field("target", camera.target), _field("opened", "yes" if camera.opened else "no")]
    if camera.opened:
        rate = "unknown" if camera.fps is None else f"{camera.fps:.1f} fps"
        lines.append(_field("resolution", _resolution_of(camera)))
        lines.append(_field("stream rate", rate))
    elif camera.reachable is not None:
        lines.append(_field("port open", "yes" if camera.reachable else "no"))
    if camera.failure is not None:
        lines.append(_field("error", camera.failure))
    if camera.hint is not None:
        lines.append(_field("check", camera.hint))
    return lines


def _resolution_of(camera: CameraReport) -> str:
    if camera.width is None or camera.height is None:
        return "unknown"
    return f"{camera.width}x{camera.height}"
