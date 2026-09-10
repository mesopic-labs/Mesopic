"""The `mesopic` command line — the self-hoster's entry point before Docker.

Commands are typed Python functions; Typer derives the interface from the annotations.

    mesopic run        --config /data/mesopic.yaml   # the supervisor, all cameras
    mesopic discover                                # ONVIF probe, prints a config stanza
    mesopic calibrate  --camera front-door          # snapshot + geometry editor
    mesopic export     --metric footfall --since …  # CSV out of the local store
    mesopic doctor                                  # box, accelerators, model licence
    mesopic spike      --rtsp <url>                 # P1: the hard-coded perf spike
    mesopic bench      --rtsp <url> --duration 1800 # P1.7: the M0 gate's soak
    mesopic truth validate fixtures/clips/*.json    # MK.2: check a label or a manifest
    mesopic truth score    --truth … --clip …      # P2.9: the M1 accuracy number
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from mesopic import __version__
from mesopic.bench import (
    BenchRun,
    BenchThresholds,
    describe_environment,
    drive_bench,
    evaluate,
    read_current_rss_bytes,
)
from mesopic.config.loader import load_config
from mesopic.detector.detector import Detector
from mesopic.detector.model_manager import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_CACHE,
    MODEL_CACHE_ENV_VAR,
    ModelManager,
)
from mesopic.detector.onnx_detector import OnnxDetector
from mesopic.doctor import host_report, preflight, render, tcp_reachable
from mesopic.errors import ConfigError, MesopicError, TruthError
from mesopic.ingest.rtsp import RtspFrameSource
from mesopic.ingest.source import FrameSource
from mesopic.runner import DATA_DIR_ENV_VAR, STORE_FILENAME, Engine, store_path
from mesopic.sampler.sampler import FrameSampler
from mesopic.spike import run_spike
from mesopic.store.store import Store
from mesopic.tracker.bytetrack import ByteTrackTracker
from mesopic.tracker.tracker import Tracker
from mesopic.truth import DRAFT_RATER, gate_eligible, load_manifest, load_truth, score
from mesopic.types import CameraId, MetricName, MetricRow, MinuteBucket

DEFAULT_BENCH_FPS = 2.5
"""Above the M0 floor on purpose, so a box that sustains it clears the floor with margin.

The sampler admits frames on the source's own grid, so any one gap between admitted
frames can run up to a source frame long; its due instant steps by whole periods, so the
average holds the target on any camera. The margin is for the gaps, not the average.
"""

DEFAULT_BENCH_DURATION_S = 1800.0
"""The M0 gate's soak length: 30 minutes (implementation-plan P1.7)."""

app = typer.Typer(
    name="mesopic",
    help="Video-intelligence for the cameras you already own. Footage never leaves your box.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Root callback.

    Present so Typer keeps the sub-command structure while `version` is the only command
    registered — without it, a single-command app collapses into a bare `mesopic`.
    """


@app.command()
def version() -> None:
    """Print the engine version."""
    typer.echo(__version__)


# The seams the commands are built from. Separate functions so a test can replace the
# camera, the model cache, the tracker and the port probe without touching any command's
# own logic.


def _open_source(camera_id: CameraId, url: str) -> FrameSource:
    return RtspFrameSource(camera_id, url)


def _reach_port(host: str, port: int) -> bool:
    """`doctor`'s fourth seam: the plain TCP probe behind its "port open" line."""
    return tcp_reachable(host, port)


def _open_detector(model: str, cache_dir: Path) -> Detector:
    artefact = ModelManager(cache_dir).ensure(model)
    return OnnxDetector(artefact.path)


def _open_tracker() -> Tracker:
    return ByteTrackTracker()


def _resolve_url(rtsp: str | None, rtsp_env: str | None) -> str:
    """Take the URL from a flag or an env var, and fail without ever echoing it.

    Both forms exist on purpose. `--rtsp` is what the implementation plan specifies and
    what a quick local run wants; `--rtsp-env` is the form that keeps a camera password
    out of shell history and out of `ps` output on a shared box.
    """
    if (rtsp is None) == (rtsp_env is None):
        message = "pass exactly one of --rtsp or --rtsp-env"
        raise typer.BadParameter(message)

    if rtsp_env is None:
        # The equality check above leaves only this pairing.
        return rtsp if rtsp is not None else ""

    # Naming the variable is safe and is the only actionable part of the error; its
    # value is the secret.
    url = os.environ.get(rtsp_env)
    if not url:
        message = f"environment variable {rtsp_env} is unset or empty"
        raise typer.BadParameter(message)
    return url


@app.command()
def doctor(
    rtsp: Annotated[
        str | None,
        typer.Option(
            "--rtsp",
            help="Optional: an RTSP URL to preflight. Prefer --rtsp-env: this form is "
            "visible in `ps`.",
        ),
    ] = None,
    rtsp_env: Annotated[
        str | None,
        typer.Option("--rtsp-env", help="Name of an env var holding the RTSP URL to preflight."),
    ] = None,
    model: Annotated[
        str, typer.Option("--model", help="Detector model to report on.")
    ] = DEFAULT_MODEL,
) -> None:
    """Report what this box can do: cores, accelerators, model cache and its licence.

    The first thing to ask a user for when a self-hosted install misbehaves, and the
    surface that makes the model's licence visible rather than buried (ADR-0013).

    Given a URL it also preflights one camera — does the stream open, at what resolution
    and rate, and if not, what to go and check. Opt-in, because the report itself must
    stay runnable on a box with no camera and no network: it reads the cache without ever
    filling it.

    The URL is a credential and is never printed. What is printed is the address it was
    aimed at with the userinfo and query string replaced — enough to see a typo in the
    stream path, and nothing anyone could log in with.
    """
    cache_dir = Path(os.environ.get(MODEL_CACHE_ENV_VAR, DEFAULT_MODEL_CACHE))
    host = host_report(cache_dir=cache_dir, model_name=model)

    camera = None
    if rtsp is not None or rtsp_env is not None:
        url = _resolve_url(rtsp, rtsp_env)
        camera = preflight(
            lambda: _open_source(CameraId("doctor"), url), url=url, reach=_reach_port
        )

    for line in render(host, camera):
        typer.echo(line)

    # A preflight that failed is a failed command: `mesopic doctor --rtsp ...` is the
    # thing a setup script runs before it trusts a URL.
    if camera is not None and not camera.opened:
        raise typer.Exit(code=1)


@app.command()
def spike(
    rtsp: Annotated[
        str | None,
        typer.Option("--rtsp", help="RTSP URL. Prefer --rtsp-env: this form is visible in `ps`."),
    ] = None,
    rtsp_env: Annotated[
        str | None,
        typer.Option("--rtsp-env", help="Name of an env var holding the RTSP URL."),
    ] = None,
    fps: Annotated[float, typer.Option("--fps", help="Target sampling rate.")] = 3.0,
    model: Annotated[str, typer.Option("--model", help="Detector model name.")] = DEFAULT_MODEL,
    stats: Annotated[
        bool,
        typer.Option("--stats", help="Also log fps, latency, CPU% and RSS once a second."),
    ] = False,
) -> None:
    """Run one camera through ingest -> sample -> detect -> track, printing JSON lines.

    The P1 perf spike (P1.6), and the thing P1.7's M0 gate is measured against. Not the
    production path: one camera, no config, no store, no supervisor.
    """
    url = _resolve_url(rtsp, rtsp_env)
    cache_dir = Path(os.environ.get(MODEL_CACHE_ENV_VAR, DEFAULT_MODEL_CACHE))

    try:
        source = _open_source(CameraId("spike"), url)
        detector = _open_detector(model, cache_dir)
        tracker = _open_tracker()

        for line in run_spike(
            source=source,
            sampler=FrameSampler(target_fps=fps),
            detector=detector,
            tracker=tracker,
            emit_stats=stats,
        ):
            typer.echo(line)
    except MesopicError as error:
        # Generic to the user, and deliberately *not* chained: a StreamDropped raised
        # from the RTSP layer can carry the URL in its message, and `raise ... from error`
        # would print it in the traceback (CLAUDE.md, ADR-0005).
        typer.echo(f"spike failed: {type(error).__name__}", err=True)
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        # The expected way a 30-minute soak ends.
        raise typer.Exit(code=130) from None


@app.command()
def bench(  # noqa: PLR0917 - each argument is one documented gate threshold (§9), not a blob
    rtsp: Annotated[str | None, typer.Option("--rtsp", help="RTSP URL.")] = None,
    rtsp_env: Annotated[
        str | None,
        typer.Option("--rtsp-env", help="Name of an env var holding the RTSP URL."),
    ] = None,
    fps: Annotated[float, typer.Option("--fps", help="Target sampling rate.")] = DEFAULT_BENCH_FPS,
    model: Annotated[str, typer.Option("--model", help="Detector model name.")] = DEFAULT_MODEL,
    duration: Annotated[
        float,
        typer.Option("--duration", help="Soak length in seconds. The M0 gate wants >= 1800."),
    ] = DEFAULT_BENCH_DURATION_S,
    assert_min_fps: Annotated[
        float | None,
        typer.Option("--assert-min-fps", help="Fail below this sustained effective fps."),
    ] = None,
    assert_max_cpu: Annotated[
        float | None,
        typer.Option("--assert-max-cpu", help="Fail above this fraction of total cores."),
    ] = None,
    assert_no_mem_growth: Annotated[
        bool, typer.Option("--assert-no-mem-growth", help="Fail on second-half RSS growth.")
    ] = False,
    assert_no_throttle: Annotated[
        bool, typer.Option("--assert-no-throttle", help="Fail on any thermal-throttle event.")
    ] = False,
    baseline_hardware: Annotated[
        bool,
        typer.Option(
            "--baseline-hardware",
            help="Mark this run as the N100 baseline of record. Off unless it really is one.",
        ),
    ] = False,
    out: Annotated[Path | None, typer.Option("--out", help="Write the JSON artefact here.")] = None,
) -> None:
    """Soak one camera and report the M0 gate's verdict (P1.7).

    The measurement half of the M0 exit criterion: one 1080p stream through
    decode->detect->track for a fixed duration, with sustained fps, latency, CPU, RSS
    and thermal-throttle events reduced to a pass or a fail per threshold.

    Thresholds are opt-in flags rather than defaults, so a bare `mesopic bench` measures
    and asserts nothing — a gate is something you ask for deliberately.

    Sustained fps is frames over *wall clock*, which includes the second or two spent
    opening the stream. Across the gate's 30 minutes that is noise; across a 5-second
    smoke run it halves the number, so short runs read low and are not evidence of a
    slow pipeline.
    """
    url = _resolve_url(rtsp, rtsp_env)
    cache_dir = Path(os.environ.get(MODEL_CACHE_ENV_VAR, DEFAULT_MODEL_CACHE))

    try:
        # Built once and reused across reconnects: loading an ONNX session per drop
        # would charge the model's startup cost to the fps the gate is read from.
        detector = _open_detector(model, cache_dir)
        tracker = _open_tracker()

        def open_lines() -> Iterator[str]:
            return run_spike(
                source=_open_source(CameraId("bench"), url),
                sampler=FrameSampler(target_fps=fps),
                detector=detector,
                tracker=tracker,
                emit_stats=True,
            )

        run = drive_bench(
            open_lines=open_lines,
            duration_s=duration,
            run=BenchRun(started_monotonic=0.0, rss_reader=read_current_rss_bytes),
        )
    except MesopicError as error:
        # Never chained: a StreamDropped can carry the URL, and the traceback would
        # print it (CLAUDE.md, ADR-0005).
        typer.echo(f"bench failed: {type(error).__name__}", err=True)
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        typer.echo("bench interrupted", err=True)
        raise typer.Exit(code=130) from None

    run.record_rss_source(
        "current-/proc/self/statm" if read_current_rss_bytes() is not None else "peak-ru_maxrss"
    )
    result = evaluate(
        run,
        BenchThresholds(
            min_fps=assert_min_fps,
            max_cpu_fraction=assert_max_cpu,
            no_mem_growth=assert_no_mem_growth,
            no_throttle=assert_no_throttle,
        ),
        baseline_hardware=baseline_hardware,
        environment=describe_environment(),
    )

    artefact = result.to_json()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(artefact + "\n")
    typer.echo(artefact)

    if result.verdicts and not result.passed:
        raise typer.Exit(code=1)


truth_app = typer.Typer(
    name="truth",
    help="Ground-truth clip sets: validate the labels an accuracy number is measured against.",
    no_args_is_help=True,
)
app.add_typer(truth_app, name="truth")

MANIFEST_SUFFIX = ".clip.json"
TRUTH_SUFFIX = ".truth.json"


def _describe_manifest(path: Path) -> str:
    """Summarise a clip manifest, leading with the answer that matters.

    Gate-eligibility is printed for every manifest, in words, because the failure this
    command exists to catch is footage being committed under a provenance nobody checked.
    A reviewer should be able to see "no" without knowing the rule.

    The scene rides alongside it because "yes" answers a narrower question than it looks
    like it answers: eligibility is about consent and provenance, and a clip can clear
    both and still be the wrong scene to read this gate's number off. Own-rig footage at
    a 2.1 m mount is the worked example — eligible, `hard`, and not a good doorway.
    """
    manifest = load_manifest(path)
    verdict = "yes" if gate_eligible(manifest) else "no"
    return (
        f"ok  {manifest.clip_id}  {manifest.duration_s:g}s  "
        f"{manifest.width}x{manifest.height}@{manifest.fps:g}  "
        f"{manifest.provenance.kind}/{manifest.consent.model_release}  "
        f"scene {manifest.scene.reference}  "
        f"gate-eligible: {verdict}"
    )


def _describe_truth(path: Path) -> str:
    truth = load_truth(path)
    count = len(truth.crossings)
    plural = "" if count == 1 else "s"
    # A draft is the one rater name that changes what the file may be used for, so it is
    # spelled out rather than left for the reader to recognise.
    draft = "  (unverified — cannot gate)" if truth.labelled_by == DRAFT_RATER else ""
    return (
        f"ok  {truth.clip_id}  {truth.duration_s:g}s  "
        f"{count} crossing{plural}  by {truth.labelled_by}{draft}"
    )


def _describe(path: Path) -> str:
    """Dispatch on the filename, because the two documents are not interchangeable."""
    if path.name.endswith(MANIFEST_SUFFIX):
        return _describe_manifest(path)
    if path.name.endswith(TRUTH_SUFFIX):
        return _describe_truth(path)
    message = f"unrecognised name: expected *{MANIFEST_SUFFIX} or *{TRUTH_SUFFIX}"
    raise TruthError(message)


@truth_app.command("validate")
def truth_validate(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Clip manifests (*.clip.json) or truth files (*.truth.json)."),
    ],
) -> None:
    """Check ground-truth documents, and say whether each clip may back a released claim.

    Every file is checked before anything exits, so one broken label does not hide the
    next: a labelling session is fixed in one pass, not one error at a time.
    """
    failed = False
    for path in paths:
        try:
            print(f"{path.name}: {_describe(path)}")
        except TruthError as error:
            # The detail is the point here: it names the field that failed in a file the
            # user wrote, and carries nothing sensitive.
            print(f"{path.name}: FAILED  {error}")
            failed = True

    if failed:
        raise typer.Exit(code=1)


SCORE_ROW_LIMIT = 100_000
"""Ceiling on the rows one scoring run will read out of the store.

Not a page size — a tripwire. A truncated read cannot be detected downstream: the missing
minutes look exactly like minutes the engine stayed silent for, which `score` reads as a
claim that nothing happened. Hitting this is refused rather than scored.
"""

STRAY_MARGIN = timedelta(hours=1)
"""How far either side of the clip the store is read.

`score` counts the minutes the engine spoke about that the footage does not span, and a
query clipped to the clip's own span could never return one — the check would be
structurally dead rather than passing. An hour covers both failure modes it exists to
catch, a teardown flush just past the end and a `stream_start` misaligned by a whole
series, without reaching into a different run's history.
"""


def _parse_stream_start(raw: str) -> datetime:
    """Parse the flag, and insist on UTC. `score` enforces the minute boundary itself."""
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        message = f"--stream-start is not an ISO-8601 instant: {raw}"
        raise TruthError(message) from None
    if parsed.tzinfo is None:
        message = "--stream-start must carry a UTC offset, so media time cannot drift with the box"
        raise TruthError(message)
    return parsed.astimezone(UTC)


def _score_database(data_dir: Path | None) -> Path:
    """Where the store is, without a config file to ask — this command reads, never runs."""
    if data_dir is not None:
        return data_dir.expanduser().resolve() / STORE_FILENAME
    from_env = os.environ.get(DATA_DIR_ENV_VAR)
    if from_env:
        return Path(from_env).expanduser().resolve() / STORE_FILENAME
    message = f"no store to read: pass --data-dir or set ${DATA_DIR_ENV_VAR}"
    raise TruthError(message)


def _run_metrics(
    database: Path,
    *,
    camera_id: CameraId | None,
    metric: MetricName,
    stream_start: datetime,
    duration_s: float,
) -> list[MetricRow]:
    """Read one run's rows out of the store — the seam between the engine and `score`."""
    with Store(database) as store:
        rows = store.metrics_between(
            start=stream_start - STRAY_MARGIN,
            end=stream_start + timedelta(seconds=duration_s) + STRAY_MARGIN,
            limit=SCORE_ROW_LIMIT,
            camera_id=camera_id,
            metrics=[metric],
        )
    if len(rows) >= SCORE_ROW_LIMIT:
        message = f"refusing to score a truncated read: hit the {SCORE_ROW_LIMIT}-row ceiling"
        raise TruthError(message)
    return rows


@truth_app.command("score")
def truth_score(  # noqa: PLR0917 - each argument is one flag of the gate run, not a blob
    truth_path: Annotated[
        Path, typer.Option("--truth", help="The truth file (*.truth.json).", show_default=False)
    ],
    clip_path: Annotated[
        Path, typer.Option("--clip", help="The clip manifest (*.clip.json).", show_default=False)
    ],
    stream_start: Annotated[
        str,
        typer.Option(
            "--stream-start",
            help="UTC instant the clip's first frame was ingested, on a minute boundary.",
            show_default=False,
        ),
    ],
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help=f"Where the SQLite store lives. Defaults to ${DATA_DIR_ENV_VAR}.",
            show_default=False,
        ),
    ] = None,
    camera_id: Annotated[
        str | None,
        typer.Option("--camera-id", help="Score one camera's rows.", show_default=False),
    ] = None,
    metric: Annotated[
        MetricName, typer.Option("--metric", help="Which quantity to measure.")
    ] = MetricName.FOOTFALL,
    gating: Annotated[
        bool,
        typer.Option(
            "--gating",
            help="Read this as a published number, and refuse if the clip may not back one.",
        ),
    ] = False,
) -> None:
    """Measure a run the engine left in the store against the labels for the same clip.

    Without `--gating` this measures, which is what metric development wants all day.
    With it, the clip has to be allowed to produce a released figure — and a refusal
    prints the reason instead of a number, because a number nobody may quote is worse
    than no number when it is the plausible-looking one that gets copied.
    """
    try:
        truth = load_truth(truth_path)
        manifest = load_manifest(clip_path)
        started = _parse_stream_start(stream_start)
        rows = _run_metrics(
            _score_database(data_dir),
            camera_id=None if camera_id is None else CameraId(camera_id),
            metric=metric,
            stream_start=started,
            duration_s=truth.duration_s,
        )
        result = score(
            truth,
            manifest,
            rows,
            stream_start=MinuteBucket(started),
            gating=gating,
            metric=metric,
        )
    except MesopicError as error:
        print(f"FAILED  {error}")
        raise typer.Exit(code=1) from None

    print(f"clip: {result.clip_id}  metric: {result.metric}  minutes: {result.minutes}")
    print(f"truth_total: {result.truth_total}  predicted_total: {result.predicted_total:g}")
    print(f"total_error_pct: {result.total_error_pct:.1f}")
    print(f"mape_pct: {result.mape_pct:.1f}  stray_minutes: {result.stray_minutes}")


@app.command()
def run(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to mesopic.yaml.", show_default=False),
    ],
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help=f"Where the SQLite store lives. Defaults to ${DATA_DIR_ENV_VAR}, "
            "then the config file's own directory.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Run the engine: every enabled camera, the store, and the local dashboard.

    The whole box in one process and one event loop (engine-architecture.md §9). Stops on
    SIGINT or SIGTERM, closing the open minute before it goes.
    """
    try:
        config = load_config(config_path)
    except ConfigError as error:
        # `load_config` has already generalised what it says: a validation error can
        # otherwise carry a camera's whole source block, credentials included (P2.1).
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from None

    database = store_path(config_path=config_path, data_dir=data_dir)
    database.parent.mkdir(parents=True, exist_ok=True)
    with Store(database) as store:
        asyncio.run(Engine(config, store, config_path=config_path).run())


if __name__ == "__main__":
    app()
