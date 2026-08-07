"""The `muster` command line — the self-hoster's entry point before Docker.

Commands are typed Python functions; Typer derives the interface from the annotations.

    muster run        --config /data/muster.yaml   # the supervisor, all cameras
    muster discover                                # ONVIF probe, prints a config stanza
    muster calibrate  --camera front-door          # snapshot + geometry editor
    muster export     --metric footfall --since …  # CSV out of the local store
    muster doctor                                  # box, accelerators, model licence
    muster spike      --rtsp <url>                 # P1: the hard-coded perf spike
"""

from __future__ import annotations

import typer

from muster import __version__

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


if __name__ == "__main__":
    app()
