"""`mesopic doctor` — the command, and above all its secret hygiene.

The report itself is tested in `test_doctor.py`. What this file guards is the wiring: that
the host half runs on a box with no camera and no network, that the camera half is opt-in
and decides the exit code, and that an RTSP URL reaches neither stdout, nor stderr, nor a
traceback, on any path through the command.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from mesopic import cli
from mesopic.detector.model_manager import MODEL_CACHE_ENV_VAR
from mesopic.errors import StreamDropped
from mesopic.types import CameraId, DecodedFrame, FrameTs

RUNNER = CliRunner()

# The documented placeholder credential, exempted by name in `.gitleaks.toml`; the host
# is in RFC 5737's documentation range and cannot address a real camera.
CREDENTIAL = "user:pass"
CAMERA_HOST = "192.0.2.10"
CAMERA_URL = f"rtsp://{CREDENTIAL}@{CAMERA_HOST}:554/Streaming/Channels/101"
T0 = datetime(2026, 8, 26, 9, 0, 0, tzinfo=UTC)


class _FakeSource:
    def __init__(self, count: int = 6) -> None:
        self._count = count

    def frames(self) -> Iterator[DecodedFrame]:
        for i in range(self._count):
            yield DecodedFrame(
                camera_id=CameraId("doctor"),
                ts=FrameTs(T0 + timedelta(seconds=i * 0.04)),
                image=np.zeros((720, 1280, 3), dtype=np.uint8),
                width=1280,
                height=720,
            )
        message = "camera 'doctor': stream ended"
        raise StreamDropped(message)

    def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def model_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the model cache somewhere empty, so the report never reads a real one."""
    monkeypatch.setenv(MODEL_CACHE_ENV_VAR, str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _no_port_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the port probe off the network; `test_doctor.py` covers what it decides.

    Left real, every failure test here would spend the probe's timeout dialling an
    address in the documentation range that is unroutable by design.
    """
    monkeypatch.setattr(cli, "_reach_port", lambda _host, _port: False)


@pytest.fixture
def _fake_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    def _source(_camera_id: CameraId, _url: str) -> _FakeSource:
        return _FakeSource()

    monkeypatch.setattr(cli, "_open_source", _source)


def test_doctor_reports_the_box_with_no_camera_and_no_arguments() -> None:
    """The half that always works — the first thing to ask a user to paste into an issue."""
    result = RUNNER.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "cpu cores" in result.output
    assert "ort-cpu" in result.output


def test_doctor_names_the_detector_licence() -> None:
    """ADR-0013 decision 5: the adopter can see the terms without reading the source."""
    result = RUNNER.invoke(cli.app, ["doctor"])

    assert "Apache-2.0" in result.output


def test_doctor_does_not_download_a_model_to_answer_a_question(model_cache: Path) -> None:
    """Reporting on the cache must not fill it: `doctor` runs on a metered link too."""
    result = RUNNER.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert list(model_cache.iterdir()) == []


@pytest.mark.usefixtures("_fake_camera")
def test_doctor_preflights_a_camera_when_it_is_given_one() -> None:
    result = RUNNER.invoke(cli.app, ["doctor", "--rtsp", CAMERA_URL])

    assert result.exit_code == 0
    assert "1280x720" in result.output
    assert "25.0 fps" in result.output


@pytest.mark.usefixtures("_fake_camera")
def test_doctor_never_prints_the_rtsp_url_it_was_given() -> None:
    result = RUNNER.invoke(cli.app, ["doctor", "--rtsp", CAMERA_URL])

    assert CREDENTIAL not in result.output
    assert CAMERA_HOST in result.output, "the address is the diagnosis; the password is not"


def test_doctor_exits_non_zero_when_the_camera_cannot_be_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A preflight that failed must be a failed command — a script has to be able to tell."""

    def _refused(_camera_id: CameraId, _url: str) -> _FakeSource:
        message = "camera 'doctor': ConnectionRefusedError while connecting"
        raise StreamDropped(message)

    monkeypatch.setattr(cli, "_open_source", _refused)

    result = RUNNER.invoke(cli.app, ["doctor", "--rtsp", CAMERA_URL])

    assert result.exit_code == 1
    assert "cpu cores" in result.output, "the host report is what gets pasted into the issue"
    assert "port" in result.output, "a failure has to say what to go and check"


def test_doctor_reports_whether_the_port_answered_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The distinction FFmpeg will not draw: nothing there, versus there and refusing."""
    probed: list[tuple[str, int]] = []

    def _opaque(_camera_id: CameraId, _url: str) -> _FakeSource:
        message = "camera 'doctor': ExitError while connecting"
        raise StreamDropped(message)

    def _probe(host: str, port: int) -> bool:
        probed.append((host, port))
        return True

    monkeypatch.setattr(cli, "_open_source", _opaque)
    monkeypatch.setattr(cli, "_reach_port", _probe)

    result = RUNNER.invoke(cli.app, ["doctor", "--rtsp", CAMERA_URL])

    assert probed == [(CAMERA_HOST, 554)]
    assert "port open   yes" in result.output


def test_doctor_never_leaks_the_url_out_of_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure path is where a URL escapes: FFmpeg puts it in its own error text."""

    def _exploding(_camera_id: CameraId, url: str) -> _FakeSource:
        message = f"could not connect to {url}"
        raise StreamDropped(message)

    monkeypatch.setattr(cli, "_open_source", _exploding)

    result = RUNNER.invoke(cli.app, ["doctor", "--rtsp", CAMERA_URL])

    assert result.exit_code == 1
    combined = result.output + str(result.exception or "")
    assert CREDENTIAL not in combined


def test_doctor_never_leaks_the_url_out_of_an_error_that_is_not_ours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PyAV raises a wide, unstable set; none of it may reach a traceback."""

    def _exploding(_camera_id: CameraId, url: str) -> _FakeSource:
        message = f"unexpected failure on {url}"
        raise RuntimeError(message)

    monkeypatch.setattr(cli, "_open_source", _exploding)

    result = RUNNER.invoke(cli.app, ["doctor", "--rtsp", CAMERA_URL])

    assert result.exit_code == 1
    combined = result.output + str(result.exception or "")
    assert CREDENTIAL not in combined


@pytest.mark.usefixtures("_fake_camera")
def test_doctor_reads_the_url_from_an_env_var_so_it_stays_out_of_shell_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_CAMERA_URL", CAMERA_URL)

    result = RUNNER.invoke(cli.app, ["doctor", "--rtsp-env", "MY_CAMERA_URL"])

    assert result.exit_code == 0
    assert CREDENTIAL not in result.output
    assert "1280x720" in result.output


def test_doctor_rejects_an_unset_env_var_by_name_not_by_value() -> None:
    result = RUNNER.invoke(cli.app, ["doctor", "--rtsp-env", "DEFINITELY_UNSET_VAR"])

    assert result.exit_code != 0
    assert "DEFINITELY_UNSET_VAR" in result.output


def test_doctor_refuses_both_url_forms_at_once() -> None:
    result = RUNNER.invoke(cli.app, ["doctor", "--rtsp", CAMERA_URL, "--rtsp-env", "MY_CAMERA_URL"])

    assert result.exit_code != 0
    assert CREDENTIAL not in result.output
