"""The `muster` command line — the self-hoster's entry point before Docker.

Commands are typed Python functions; Typer derives the interface from the annotations.

    muster run        --config /data/muster.yaml   # the supervisor, all cameras
    muster discover                                # ONVIF probe, prints a config stanza
    muster calibrate  --camera front-door          # snapshot + geometry editor
    muster export     --metric footfall --since …  # CSV out of the local store
    muster doctor                                  # box, accelerators, model licence
    muster spike      --rtsp <url>                 # P1: the hard-coded perf spike
    muster bench      --rtsp <url> --duration 1800 # P1.7: the M0 gate's soak
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import typer

from muster import __version__
from muster.bench import (
    BenchRun,
    BenchThresholds,
    describe_environment,
    drive_bench,
    evaluate,
    read_current_rss_bytes,
)
from muster.detector.detector import Detector
from muster.detector.model_manager import ModelManager
from muster.detector.onnx_detector import OnnxDetector
from muster.errors import MusterError
from muster.ingest.rtsp import RtspFrameSource
from muster.ingest.source import FrameSource
from muster.sampler.sampler import FrameSampler
from muster.spike import run_spike
from muster.tracker.bytetrack import ByteTrackTracker
from muster.tracker.tracker import Tracker
from muster.types import CameraId

DEFAULT_MODEL = "yolox-nano"
"""The Apache-2.0 default (ADR-0013). Provisional until P1.7's N100 table lands."""

DEFAULT_BENCH_FPS = 2.5
"""Above the M0 floor on purpose.

The floor is 2 fps, but capture-time gating admits frames on the source's own grid: at
25 fps a 500 ms gate lands on the 520 ms frame, pinning the effective rate at 1.923 and
failing a `>= 2.0` assertion on arithmetic rather than on capacity. 2.5 divides 25 fps
evenly (400 ms), and a box that sustains it clears the floor with margin.
"""

DEFAULT_BENCH_DURATION_S = 1800.0
"""The M0 gate's soak length: 30 minutes (implementation-plan P1.7)."""

MODEL_CACHE_ENV_VAR = "MUSTER_MODEL_CACHE"
DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "muster" / "models"

app = typer.Typer(
    name="muster",
    help="Video-intelligence for the cameras you already own. Footage never leaves your box.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Root callback.

    Present so Typer keeps the sub-command structure while `version` is the only command
    registered — without it, a single-command app collapses into a bare `muster`.
    """


@app.command()
def version() -> None:
    """Print the engine version."""
    typer.echo(__version__)


@app.command()
def doctor() -> None:
    """Report what this box can do: cores, accelerators, model cache and its licence.

    The first thing to ask a user for when a self-hosted install misbehaves, and the
    surface that makes the model's licence visible rather than buried (ADR-0013).
    """
    raise NotImplementedError


# The three seams the spike is built from. Separate functions so a test can replace the
# camera, the model cache, and the tracker without touching the command's own logic.


def _open_source(camera_id: CameraId, url: str) -> FrameSource:
    return RtspFrameSource(camera_id, url)


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
    except MusterError as error:
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

    Thresholds are opt-in flags rather than defaults, so a bare `muster bench` measures
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
    except MusterError as error:
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


if __name__ == "__main__":
    app()
