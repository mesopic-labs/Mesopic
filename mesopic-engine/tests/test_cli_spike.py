"""`mesopic spike` — argument handling and, above all, secret hygiene.

The pipeline itself is tested in `test_spike.py`; here the components are fakes. What
this file is really guarding is the rule that an RTSP URL is a credential: it must not
reach stdout, an error message, or a traceback, on any path.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from mesopic import cli
from mesopic.errors import StreamDropped
from mesopic.types import CameraId, DecodedFrame, Detection, FrameTs, Track, TrackId

RUNNER = CliRunner()

# `rtsp://user:pass@` is the one credential form allowed in this repository — the
# documented placeholder, exempted by name in `.gitleaks.toml`. Any other spelling is a
# leak as far as the scanner is concerned, and it is right to insist. The host is in RFC
# 5737's documentation range, so this URL cannot address a real camera either.
CREDENTIAL = "user:pass"
CAMERA_HOST = "192.0.2.10"
CAMERA_URL = f"rtsp://{CREDENTIAL}@{CAMERA_HOST}:554/Streaming/Channels/101"
T0 = datetime(2026, 8, 11, 12, 0, 0, tzinfo=UTC)


class _FakeSource:
    def __init__(self, count: int = 4) -> None:
        self._count = count
        self.closed = False

    def frames(self) -> Iterator[DecodedFrame]:
        for i in range(self._count):
            yield DecodedFrame(
                camera_id=CameraId("spike"),
                ts=FrameTs(T0 + timedelta(seconds=i)),
                image=np.zeros((4, 4, 3), dtype=np.uint8),
                width=4,
                height=4,
            )
        # A camera stream does not end, it drops — and this is the only exit path a
        # real `mesopic spike` ever takes (ingest/rtsp.py).
        message = "camera 'spike': stream ended"
        raise StreamDropped(message)

    def close(self) -> None:
        self.closed = True


class _FakeDetector:
    def detect(self, _frame: DecodedFrame) -> list[Detection]:
        return [Detection(box=(0, 0, 2, 4), score=0.9)]

    def close(self) -> None:
        return None


class _FakeTracker:
    def update(self, frame: DecodedFrame, _detections: list[Detection]) -> list[Track]:
        return [
            Track(
                camera_id=CameraId("spike"),
                track_id=TrackId(1),
                ts=frame.ts,
                foot_point=(0.5, 0.5),
                score=0.9,
            )
        ]


@pytest.fixture(autouse=True)
def _fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the CLI off the network and off the model cache."""

    def _source(_camera_id: CameraId, _url: str) -> _FakeSource:
        return _FakeSource()

    def _detector(_model: str, _cache_dir: Path) -> _FakeDetector:
        return _FakeDetector()

    monkeypatch.setattr(cli, "_open_source", _source)
    monkeypatch.setattr(cli, "_open_detector", _detector)
    monkeypatch.setattr(cli, "_open_tracker", _FakeTracker)


def _frame_lines(output: str) -> list[str]:
    """Frame lines only — stderr diagnostics and stats lines filtered out."""
    return [
        line for line in output.splitlines() if line.strip().startswith("{") and "stats" not in line
    ]


def test_spike_streams_one_json_line_per_admitted_frame() -> None:
    result = RUNNER.invoke(cli.app, ["spike", "--rtsp", CAMERA_URL, "--fps", "1"])

    lines = _frame_lines(result.output)
    assert len(lines) == 4
    for line in lines:
        assert set(json.loads(line)) == {"ts", "fps", "tracks"}


def test_spike_exits_non_zero_when_the_stream_drops() -> None:
    """P1.7 soaks for 30 minutes; a drop at minute 20 must not look like success."""
    result = RUNNER.invoke(cli.app, ["spike", "--rtsp", CAMERA_URL, "--fps", "1"])

    assert result.exit_code == 1
    assert _frame_lines(result.output), "the frames it did manage still belong on stdout"


def test_spike_never_prints_the_rtsp_url() -> None:
    """It carries the camera's password. It must not reach stdout on the happy path."""
    result = RUNNER.invoke(cli.app, ["spike", "--rtsp", CAMERA_URL, "--fps", "1"])

    assert CREDENTIAL not in result.output
    assert CAMERA_HOST not in result.output


def test_spike_never_leaks_the_rtsp_url_when_the_stream_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure path is where a URL usually escapes — inside the exception text."""

    def _exploding_source(_camera_id: CameraId, url: str) -> _FakeSource:
        message = f"could not connect to {url}"
        raise StreamDropped(message)

    monkeypatch.setattr(cli, "_open_source", _exploding_source)

    result = RUNNER.invoke(cli.app, ["spike", "--rtsp", CAMERA_URL])

    assert result.exit_code != 0
    combined = result.output + str(result.exception or "")
    assert CREDENTIAL not in combined
    assert CAMERA_HOST not in combined


def test_spike_reads_the_url_from_an_env_var_so_it_stays_out_of_shell_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--rtsp` puts a credential in `ps` output and `~/.zsh_history`. Offer the safe form."""
    monkeypatch.setenv("MY_CAMERA_URL", CAMERA_URL)

    result = RUNNER.invoke(cli.app, ["spike", "--rtsp-env", "MY_CAMERA_URL", "--fps", "1"])

    assert CREDENTIAL not in result.output
    assert len(_frame_lines(result.output)) == 4


def test_spike_rejects_an_unset_env_var_by_name_not_by_value() -> None:
    result = RUNNER.invoke(cli.app, ["spike", "--rtsp-env", "DEFINITELY_UNSET_VAR"])

    assert result.exit_code != 0
    assert "DEFINITELY_UNSET_VAR" in result.output


def test_spike_requires_exactly_one_url_source() -> None:
    assert RUNNER.invoke(cli.app, ["spike"]).exit_code != 0
    assert (
        RUNNER.invoke(
            cli.app, ["spike", "--rtsp", CAMERA_URL, "--rtsp-env", "MY_CAMERA_URL"]
        ).exit_code
        != 0
    )


def test_spike_emits_stats_lines_only_when_asked() -> None:
    without = RUNNER.invoke(cli.app, ["spike", "--rtsp", CAMERA_URL, "--fps", "1"])
    assert all("stats" not in line for line in without.output.splitlines())

    with_stats = RUNNER.invoke(cli.app, ["spike", "--rtsp", CAMERA_URL, "--fps", "1", "--stats"])
    payloads = [
        json.loads(line) for line in with_stats.output.splitlines() if line.strip().startswith("{")
    ]
    stats_lines = [p for p in payloads if p.get("stats")]

    assert stats_lines, "--stats must emit at least one stats line"
    assert set(stats_lines[0]) >= {"fps", "mean_latency_ms", "cpu_percent", "rss_bytes"}
