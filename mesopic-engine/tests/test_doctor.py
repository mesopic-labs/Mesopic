"""`mesopic doctor` — the report itself, and the redaction that guards it.

Two things are under test here. The first is the report: cores, accelerators, and the
model cache with the licence the artefact came under, which is the surface ADR-0013's
decision 5 owes a commercial adopter.

The second is the rule that makes the camera half of the command safe to run: an RTSP
URL carries the camera's password, so *every* string this module can emit is checked for
it — including the ones built out of a third-party error message, which is where a URL
usually escapes.
"""

from __future__ import annotations

import socket
import string
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from mesopic import doctor
from mesopic.detector.model_manager import DEFAULT_MODEL, ModelManager, ModelSpec
from mesopic.errors import StreamDropped
from mesopic.types import CameraId, DecodedFrame, FrameTs, Runtime

# The one credential form this repository allows: the documented placeholder, exempted
# by name in `.gitleaks.toml`. The host is in RFC 5737's documentation range, so this
# URL cannot address a real camera either.
CREDENTIAL = "user:pass"
CAMERA_HOST = "192.0.2.10"
CAMERA_PATH = "/Streaming/Channels/101"
CAMERA_URL = f"rtsp://{CREDENTIAL}@{CAMERA_HOST}:554{CAMERA_PATH}"

T0 = datetime(2026, 8, 26, 9, 0, 0, tzinfo=UTC)


# --- redaction --------------------------------------------------------------


def test_redact_strips_the_credentials_and_keeps_the_target() -> None:
    """The password is the secret; the address is the diagnosis."""
    redacted = doctor.redact_url(CAMERA_URL)

    assert CREDENTIAL not in redacted
    assert redacted == f"rtsp://{doctor.REDACTED}@{CAMERA_HOST}:554{CAMERA_PATH}"


def test_redact_removes_a_credential_carried_in_the_query_string() -> None:
    """Dahua-style URLs put the login in the query, not the userinfo."""
    url = f"rtsp://{CAMERA_HOST}:554/cam/realmonitor?channel=1&user=admin&password=changeme"

    redacted = doctor.redact_url(url)

    assert "changeme" not in redacted
    assert "admin" not in redacted
    assert "/cam/realmonitor" in redacted


def test_redact_gives_up_rather_than_guess_at_something_unparseable() -> None:
    """A string we cannot take apart is a string we cannot prove is safe to print."""
    assert doctor.redact_url("rtsp://[not-an-address:554/x") == doctor.REDACTED
    assert doctor.redact_url("this is not a url") == doctor.REDACTED


def test_scrub_rewrites_a_url_embedded_in_somebody_elses_error_message() -> None:
    """FFmpeg puts the URL it was handed into its own error strings."""
    message = f"Server returned 401 Unauthorized when connecting to {CAMERA_URL}"

    scrubbed = doctor.scrub(message)

    assert CREDENTIAL not in scrubbed
    assert "401 Unauthorized" in scrubbed


def test_scrub_removes_the_known_url_even_spelled_without_a_scheme() -> None:
    """Given the URL, exact-substring removal catches what the pattern would miss."""
    message = f"auth failed for {CREDENTIAL}@{CAMERA_HOST}"

    assert CREDENTIAL not in doctor.scrub(message, url=CAMERA_URL)


# Unreserved URL characters only: anything else has to be percent-encoded before it can
# appear in a URL at all, so a password containing one never reaches us raw.
_PASSWORD = st.text(alphabet=string.ascii_letters + string.digits + "-._~", min_size=8)


@pytest.mark.privacy
@given(password=_PASSWORD)
def test_no_password_survives_redaction(password: str) -> None:
    """The invariant, not one example of it: whatever the password is, it does not print."""
    url = f"rtsp://operator:{password}@{CAMERA_HOST}:554{CAMERA_PATH}"
    # A password that happens to spell part of the address it is embedded in would be
    # "leaked" by the address alone, which is a different claim from this one.
    assume(password not in f"rtsp://operator@{CAMERA_HOST}:554{CAMERA_PATH}")

    assert password not in doctor.redact_url(url)
    assert password not in doctor.scrub(f"failed to open {url}", url=url)


def test_scrub_removes_a_password_quoted_on_its_own() -> None:
    """A library that names the password without the URL around it still leaks it."""
    url = f"rtsp://admin:s3cr3t-lens@{CAMERA_HOST}:554/stream1"

    scrubbed = doctor.scrub("authentication failed for password s3cr3t-lens", url=url)

    assert "s3cr3t-lens" not in scrubbed


def test_scrub_does_not_shred_a_diagnostic_around_a_very_short_password() -> None:
    """Blind substring removal on a password of "1" would rewrite "401" as well."""
    url = f"rtsp://admin:1@{CAMERA_HOST}:554/stream1"

    assert doctor.scrub("Server returned 401 Unauthorized", url=url) == (
        "Server returned 401 Unauthorized"
    )


def test_scrub_leaves_ordinary_diagnostics_intact() -> None:
    assert doctor.scrub("TimeoutError while connecting") == "TimeoutError while connecting"


# --- the host report --------------------------------------------------------


def _build_cache_entry(cache_dir: Path, payload: bytes) -> None:
    """Put a cached artefact where the manager keeps them, without a download.

    Through `ModelManager` rather than by writing the filename here: the cache key is
    the manager's business (ADR-0012), and a test that spelled it out would keep passing
    after a change that stopped `doctor` finding anything.
    """

    def _download(_spec: ModelSpec, dest: Path) -> None:
        dest.write_bytes(payload)

    def _publish(src: Path, dest: Path) -> None:
        dest.write_bytes(src.read_bytes())

    ModelManager(cache_dir, download=_download, transform=_publish).ensure(DEFAULT_MODEL)


def test_host_report_names_the_model_licence_even_when_nothing_is_cached(
    tmp_path: Path,
) -> None:
    """The licence is a property of the model we would fetch, not of the cache."""
    report = doctor.host_report(cache_dir=tmp_path)

    assert report.model.licence == "Apache-2.0"
    assert report.model.cached is False
    assert report.model.size_bytes is None


def test_host_report_reports_a_cached_artefact_and_its_size(tmp_path: Path) -> None:
    _build_cache_entry(tmp_path, b"x" * 2048)

    report = doctor.host_report(cache_dir=tmp_path)

    assert report.model.cached is True
    assert report.model.size_bytes == 2048


def test_host_report_says_so_when_the_model_is_not_one_we_know(tmp_path: Path) -> None:
    report = doctor.host_report(cache_dir=tmp_path, model_name="no-such-model")

    assert report.model.licence == doctor.UNKNOWN_LICENCE
    assert report.model.cached is False


def test_host_report_lists_the_runtimes_this_box_can_execute(tmp_path: Path) -> None:
    """ORT-CPU is the guaranteed path (ADR-0012); an accelerator's absence is normal."""
    report = doctor.host_report(cache_dir=tmp_path, env={})

    assert Runtime.ORT_CPU in report.runtimes
    assert report.selected_runtime is Runtime.ORT_CPU
    assert report.cpu_count is None or report.cpu_count >= 1


def test_host_report_says_when_an_operator_asked_for_a_runtime(tmp_path: Path) -> None:
    """An explicit request this box cannot honour is the performance mystery (ADR-0012)."""
    report = doctor.host_report(cache_dir=tmp_path, env={"MESOPIC_DETECTOR_RUNTIME": "openvino"})

    assert report.selected_runtime is Runtime.OPENVINO
    assert report.selected_explicit is True


# --- the camera preflight ---------------------------------------------------


class _FakeSource:
    """A camera that yields `count` frames at `period_s`, then drops like a real one."""

    def __init__(self, count: int = 8, period_s: float = 0.04) -> None:
        self._count = count
        self._period_s = period_s
        self.closed = False

    def frames(self) -> Iterator[DecodedFrame]:
        for i in range(self._count):
            yield DecodedFrame(
                camera_id=CameraId("doctor"),
                ts=FrameTs(T0 + timedelta(seconds=i * self._period_s)),
                image=np.zeros((720, 1280, 3), dtype=np.uint8),
                width=1280,
                height=720,
            )
        message = "camera 'doctor': stream ended"
        raise StreamDropped(message)

    def close(self) -> None:
        self.closed = True


def test_preflight_reports_the_resolution_and_the_streams_own_rate() -> None:
    source = _FakeSource(count=8, period_s=0.04)

    report = doctor.preflight(lambda: source, url=CAMERA_URL, sample_frames=5)

    assert report.opened is True
    assert (report.width, report.height) == (1280, 720)
    assert report.fps == pytest.approx(25.0)
    assert report.failure is None


def test_preflight_releases_the_camera_socket_when_it_is_done() -> None:
    """A camera has a session limit; a preflight that held one would cost a real run."""
    source = _FakeSource()

    doctor.preflight(lambda: source, url=CAMERA_URL, sample_frames=3)

    assert source.closed is True


def test_preflight_names_the_target_it_tried_without_the_credentials() -> None:
    report = doctor.preflight(_FakeSource, url=CAMERA_URL, sample_frames=3)

    assert CREDENTIAL not in report.target
    assert CAMERA_HOST in report.target


def _unreachable(_host: str, _port: int) -> bool:
    return False


def _reachable(_host: str, _port: int) -> bool:
    return True


def test_preflight_explains_a_refused_connection_in_words() -> None:
    def _refused() -> _FakeSource:
        message = "camera 'doctor': ConnectionRefusedError while connecting"
        raise StreamDropped(message)

    report = doctor.preflight(_refused, url=CAMERA_URL, reach=_unreachable)

    assert report.opened is False
    assert report.hint is not None
    assert "port" in report.hint


def test_preflight_explains_rejected_credentials_without_repeating_them() -> None:
    def _unauthorized() -> _FakeSource:
        message = f"Server returned 401 Unauthorized for {CAMERA_URL}"
        raise StreamDropped(message)

    report = doctor.preflight(_unauthorized, url=CAMERA_URL, reach=_reachable)

    assert report.opened is False
    assert report.hint is not None
    assert "password" in report.hint
    assert report.failure is not None
    assert CREDENTIAL not in report.failure


def test_preflight_probes_the_port_when_ffmpeg_will_not_say_what_went_wrong() -> None:
    """FFmpeg collapses most network failures into one opaque error class.

    A refused connection and a rejected password arrive as the same `ExitError`, and
    `RtspFrameSource` keeps only the class name — deliberately, because the message
    carries the URL. So the useful distinction is drawn here instead, by a plain TCP
    connection to the address, which needs no credentials to be informative.
    """

    def _opaque() -> _FakeSource:
        message = "camera 'doctor': ExitError while connecting"
        raise StreamDropped(message)

    report = doctor.preflight(_opaque, url=CAMERA_URL, reach=_unreachable)

    assert report.reachable is False
    assert report.hint is not None
    assert "firewall" in report.hint


def test_preflight_distinguishes_a_camera_that_answers_from_a_stream_that_will_not_start() -> None:
    def _opaque() -> _FakeSource:
        message = "camera 'doctor': ExitError while connecting"
        raise StreamDropped(message)

    report = doctor.preflight(_opaque, url=CAMERA_URL, reach=_reachable)

    assert report.reachable is True
    assert report.hint is not None
    assert "password" in report.hint
    assert "path" in report.hint


def test_preflight_does_not_probe_a_port_the_stream_already_opened_on() -> None:
    probes: list[tuple[str, int]] = []

    def _recording(host: str, port: int) -> bool:
        probes.append((host, port))
        return True

    report = doctor.preflight(_FakeSource, url=CAMERA_URL, sample_frames=3, reach=_recording)

    assert probes == [], "it opened a stream on that port; there is nothing left to ask"
    assert report.reachable is True


def test_tcp_reachable_sees_a_socket_that_is_listening() -> None:
    with socket.create_server(("127.0.0.1", 0)) as server:
        host, port = server.getsockname()[:2]

        assert doctor.tcp_reachable(host, port) is True


def test_tcp_reachable_reports_a_port_with_nothing_behind_it() -> None:
    with socket.create_server(("127.0.0.1", 0)) as server:
        host, port = server.getsockname()[:2]

    assert doctor.tcp_reachable(host, port, timeout_s=0.5) is False


def test_preflight_survives_an_error_that_is_not_one_of_ours() -> None:
    """A diagnostic that raises is a diagnostic that prints a traceback holding the URL."""

    def _exploding() -> _FakeSource:
        message = f"could not connect to {CAMERA_URL}"
        raise RuntimeError(message)

    report = doctor.preflight(_exploding, url=CAMERA_URL, reach=_unreachable)

    assert report.opened is False
    assert report.failure is not None
    assert CREDENTIAL not in report.failure


def test_preflight_reports_a_stream_that_opens_and_then_yields_nothing() -> None:
    """Wedged firmware: the socket is fine and no frame ever arrives."""
    report = doctor.preflight(lambda: _FakeSource(count=0), url=CAMERA_URL, reach=_unreachable)

    assert report.opened is False
    assert report.width is None
    assert report.failure is not None


def test_preflight_reports_a_rate_of_none_when_one_frame_is_all_it_got() -> None:
    """One frame has a timestamp but no spacing, and no rate can be read off it."""
    report = doctor.preflight(lambda: _FakeSource(count=1), url=CAMERA_URL, sample_frames=1)

    assert report.opened is True
    assert report.fps is None
    assert (report.width, report.height) == (1280, 720)


# --- rendering --------------------------------------------------------------


def test_render_puts_the_licence_where_a_reader_will_see_it(tmp_path: Path) -> None:
    lines = doctor.render(doctor.host_report(cache_dir=tmp_path))

    assert any("Apache-2.0" in line for line in lines)


def test_render_never_lets_a_url_through_even_if_a_report_carried_one(
    tmp_path: Path,
) -> None:
    """The last boundary before stdout, so it does not trust its own inputs."""
    camera = doctor.CameraReport(
        target=CAMERA_URL,
        opened=False,
        reachable=None,
        width=None,
        height=None,
        fps=None,
        failure=f"exploded on {CAMERA_URL}",
        hint=None,
    )

    rendered = "\n".join(doctor.render(doctor.host_report(cache_dir=tmp_path), camera))

    assert CREDENTIAL not in rendered
