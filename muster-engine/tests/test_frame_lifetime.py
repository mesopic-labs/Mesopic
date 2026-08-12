"""P1.8: run the pipeline and prove no frame byte reached the filesystem.

`test_privacy_invariants.py` is the *static* half of this guarantee — it greps the
source for pixel-shaped columns and image-writing calls. Static checks only see the
call sites someone spelled the way we predicted. This is the *dynamic* half: it arms
the real filesystem boundary, drives real frames through the real pipeline, and
records every write the process actually attempts.

Two assertions, and they fail for different reasons on purpose:

* **The pipeline writes nothing at all.** True today, and the blunt instrument that
  catches a debug `cv2.imwrite` before it reaches review.
* **No write carries frame pixels.** The durable one. P2.5 gives the engine a store
  and P2.6 an aggregator, at which point writing to disk becomes legitimate and the
  first assertion has to relax — this one never does, because the frame buffer is
  tainted with a recognisable byte pattern and every payload is searched for it.

The recorder is deliberately public: P2.9's M1 gate asserts this same property with
the store and aggregator in the loop, and must not reimplement the boundary.
"""

from __future__ import annotations

import builtins
import io
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pytest

from muster.errors import StreamDropped
from muster.sampler.sampler import FrameSampler
from muster.spike import run_spike
from muster.tracker.bytetrack import ByteTrackTracker
from muster.types import CameraId, DecodedFrame, Detection, FrameTs

CAMERA = CameraId("privacy-cam")
T0 = datetime(2026, 8, 12, 9, 0, 0, tzinfo=UTC)

TAINT = b"MUSTER-FRAME-PIX"
"""The byte pattern the frame buffer is filled with.

Sixteen bytes, so a coincidental match in unrelated output is not a realistic worry,
and it survives any transformation that copies pixels around without resampling them —
a crop, a reshape, a raw `tofile`, a memoryview. It does *not* survive JPEG encoding,
which is why the "wrote nothing at all" assertion carries its own weight rather than
being redundant.
"""

# Modes that open a file for writing. `+` counts: `r+` writes to an existing file.
_WRITE_MODE_CHARS = frozenset("wax+")


@dataclass(frozen=True, slots=True)
class WriteAttempt:
    """One attempt to put bytes on disk, whatever primitive was used."""

    primitive: str
    target: str
    payload: bytes

    def is_tainted(self) -> bool:
        """Whether this write carried frame pixels — in the payload, or on disk.

        The payload is not always visible. `ndarray.tofile` opens the file through
        `builtins.open` but writes through it from C, so the bytes never pass any
        Python-level `write`; the same is true of anything else that hands a file
        object to an extension module. Checking what the target actually contains
        closes that gap for every primitive that opens a path we can see, which is
        why this is not merely a substring test on `payload`.
        """
        return TAINT in self.payload or self._target_file_contains_taint()

    def _target_file_contains_taint(self) -> bool:
        try:
            return TAINT in Path(self.target).read_bytes()
        except (OSError, ValueError):
            # A file descriptor rather than a path, a file already unlinked, or one we
            # may not read. Payload capture is the only evidence available for those.
            return False


class DiskWriteRecorder:
    """Records every filesystem write the process attempts while it is armed.

    The patched set was chosen by testing what each patch actually intercepts on
    CPython 3.12, not by reading source:

    * `io.open` catches every `pathlib` write — `Path.write_bytes`, `Path.write_text`
      and `Path.open` all route through it, and none of them touch `builtins.open`.
    * `builtins.open` catches direct `open()`, and today also catches `ndarray.tofile`
      and `np.save`, which go through it internally.
    * `os.open`/`os.write` catch the file-descriptor path, which bypasses both.
    * `cv2.imwrite` writes from C and is invisible to all of the above, so it is
      patched by name — it is also the single most likely way a frame ever lands on
      disk in this codebase.

    `numpy.ndarray.tofile` cannot be patched (extension types reject attribute
    assignment) and is covered only via `builtins.open`. That is an implementation
    detail of numpy rather than a promise, so `test_the_recorder_catches_*` pins it:
    if a future numpy writes from C instead, this suite fails loudly rather than
    going quietly blind.
    """

    def __init__(self) -> None:
        self.attempts: list[WriteAttempt] = []

    def tainted(self) -> list[WriteAttempt]:
        """The attempts that carried frame pixels."""
        return [attempt for attempt in self.attempts if attempt.is_tainted()]

    def _record(self, primitive: str, target: object, payload: bytes = b"") -> None:
        self.attempts.append(WriteAttempt(primitive=primitive, target=str(target), payload=payload))

    def arm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch the write primitives. Undone by monkeypatch at test teardown."""
        self._arm_open(monkeypatch, builtins, "open")
        self._arm_open(monkeypatch, io, "open")
        self._arm_os(monkeypatch)
        self._arm_cv2(monkeypatch)

    def _arm_open(self, monkeypatch: pytest.MonkeyPatch, module: object, name: str) -> None:
        real: Callable[..., Any] = getattr(module, name)
        label = f"{getattr(module, '__name__', module)}.{name}"

        def spy(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            handle = real(file, mode, *args, **kwargs)
            if not _WRITE_MODE_CHARS.isdisjoint(mode):
                self._record(label, file)
                return _RecordingHandle(handle, lambda data: self._record(label, file, data))
            return handle

        monkeypatch.setattr(module, name, spy)

    def _arm_os(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_open, real_write = os.open, os.write
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT

        def spy_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            if flags & write_flags:
                self._record("os.open", path)
            return real_open(path, flags, *args, **kwargs)

        def spy_write(fd: int, data: Any) -> int:
            self._record("os.write", fd, bytes(data))
            return real_write(fd, data)

        monkeypatch.setattr(os, "open", spy_open)
        monkeypatch.setattr(os, "write", spy_write)

    def _arm_cv2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cv2 = pytest.importorskip("cv2", reason="the OpenCV write path only exists if cv2 does")
        real = cv2.imwrite

        def spy(filename: Any, img: Any, *args: Any, **kwargs: Any) -> Any:
            self._record("cv2.imwrite", filename, np.asarray(img).tobytes())
            return real(filename, img, *args, **kwargs)

        monkeypatch.setattr(cv2, "imwrite", spy)


class _RecordingHandle:
    """A file object that reports what is written through it, then writes it."""

    def __init__(self, wrapped: Any, on_write: Callable[[bytes], None]) -> None:
        self._wrapped = wrapped
        self._on_write = on_write

    def write(self, data: Any) -> Any:
        self._on_write(_as_bytes(data))
        return self._wrapped.write(data)

    def writelines(self, lines: Iterable[Any]) -> None:
        chunks = list(lines)
        for chunk in chunks:
            self._on_write(_as_bytes(chunk))
        self._wrapped.writelines(chunks)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def __enter__(self) -> _RecordingHandle:
        self._wrapped.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Any:
        return self._wrapped.__exit__(exc_type, exc, tb)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._wrapped)


def _as_bytes(data: Any) -> bytes:
    """Whatever was written, as bytes we can search — text streams included."""
    if isinstance(data, str):
        return data.encode("utf-8", "replace")
    return bytes(data)


@pytest.fixture
def disk_writes(monkeypatch: pytest.MonkeyPatch) -> DiskWriteRecorder:
    """An armed recorder. Nothing reaches the filesystem unobserved while it lives."""
    recorder = DiskWriteRecorder()
    recorder.arm(monkeypatch)
    return recorder


def _tainted_frame(offset_s: float, *, size: int = 8) -> DecodedFrame:
    """A frame whose raw buffer is the taint pattern, repeated.

    Built by tiling rather than `np.full` so the taint survives as a contiguous byte
    run: a single repeated byte value would match almost anything.
    """
    magic = np.frombuffer(TAINT, dtype=np.uint8)
    pixels = size * size * 3
    tiled = np.tile(magic, pixels // magic.size + 1)[:pixels]
    return DecodedFrame(
        camera_id=CAMERA,
        ts=FrameTs(T0 + timedelta(seconds=offset_s)),
        image=tiled.reshape((size, size, 3)).copy(),
        width=size,
        height=size,
    )


class _TaintedSource:
    """A finite source that ends in the drop a real camera ends in."""

    def __init__(self, frames: list[DecodedFrame]) -> None:
        self._frames = frames
        self.closed = False

    def frames(self) -> Iterator[DecodedFrame]:
        yield from self._frames
        message = f"camera {CAMERA!r}: stream ended"
        raise StreamDropped(message)

    def close(self) -> None:
        self.closed = True


class _BoxDetector:
    """Returns a box over the tainted pixels, so the tracker has work to do."""

    def detect(self, frame: DecodedFrame) -> list[Detection]:
        return [Detection(box=(0, 0, frame.width // 2, frame.height), score=0.9)]

    def close(self) -> None:
        return None


def _drain_a_spike_run(frames: list[DecodedFrame]) -> list[str]:
    """One camera through the real sampler and the real tracker, to the stream drop.

    The detector is a fake because the real one needs downloaded weights, which would
    make this test a network test. That is the one seam, and the static half of the
    suite covers the detector package independently.
    """
    lines: list[str] = []
    runner = run_spike(
        source=_TaintedSource(frames),
        sampler=FrameSampler(target_fps=2.0),
        detector=_BoxDetector(),
        tracker=ByteTrackTracker(),
    )
    with pytest.raises(StreamDropped):
        lines.extend(runner)
    return lines


# --- The invariant ----------------------------------------------------------


@pytest.mark.privacy
def test_a_spike_run_writes_nothing_to_disk(disk_writes: DiskWriteRecorder) -> None:
    """A frame is a local variable in the worker loop. The loop touches no file.

    When P2.5 lands a store this assertion has to name the store's file as the one
    legitimate write, not be deleted — the day it is deleted is the day a frame dump
    stops being noticed.
    """
    lines = _drain_a_spike_run([_tainted_frame(i / 10.0) for i in range(20)])

    assert lines, "a run that produced no output would pass this test vacuously"
    assert disk_writes.attempts == [], (
        f"the pipeline touched the filesystem: {disk_writes.attempts}"
    )


@pytest.mark.privacy
def test_frame_pixels_never_reach_the_filesystem(disk_writes: DiskWriteRecorder) -> None:
    """The invariant that has to survive the engine legitimately owning a database."""
    _drain_a_spike_run([_tainted_frame(i / 10.0) for i in range(20)])

    assert disk_writes.tainted() == [], (
        f"frame pixels were written to disk: {disk_writes.tainted()}"
    )


@pytest.mark.privacy
def test_frame_pixels_do_not_reach_the_filesystem_when_the_consumer_stops_early(
    disk_writes: DiskWriteRecorder,
) -> None:
    """`muster spike | head -1` unwinds the loop through `GeneratorExit`.

    That is a different code path — the `finally` in `run_spike` — and an exception
    handler is exactly where a "dump the frame for debugging" line gets added.
    """
    source = _TaintedSource([_tainted_frame(i / 10.0) for i in range(20)])
    runner = run_spike(
        source=source,
        sampler=FrameSampler(target_fps=100.0),
        detector=_BoxDetector(),
        tracker=ByteTrackTracker(),
    )
    next(runner)
    runner.close()

    assert source.closed, "the socket must be released on the early-exit path"
    assert disk_writes.attempts == [], (
        f"the early-exit path touched the filesystem: {disk_writes.attempts}"
    )


@pytest.mark.privacy
def test_the_emitted_lines_carry_no_frame_pixels(disk_writes: DiskWriteRecorder) -> None:
    """Stdout is not disk, but it is where a leaked buffer would surface first."""
    lines = _drain_a_spike_run([_tainted_frame(i / 10.0) for i in range(20)])

    for line in lines:
        assert TAINT not in line.encode("utf-8", "replace")


# --- The recorder itself ----------------------------------------------------
#
# A guard nobody has watched catch anything is decoration. These are the tests that
# make the four above mean something: each drives one write primitive and asserts the
# recorder saw it. If a Python or numpy upgrade moves a write off the patched path,
# these fail here rather than silently disarming the invariant tests.


def _leak_via_write_bytes(path: Path) -> None:
    path.write_bytes(TAINT)


def _leak_via_write_text(path: Path) -> None:
    path.write_text(TAINT.decode())


def _leak_via_path_open(path: Path) -> None:
    with path.open("wb") as handle:
        handle.write(TAINT)


def _leak_via_builtin_open(path: Path) -> None:
    # PTH123 wants Path.open() here, but `builtins.open` is the primitive under test:
    # rewriting it would route through `io.open` and stop testing this patch at all.
    with open(path, "wb") as handle:  # noqa: PTH123
        handle.write(TAINT)


def _leak_via_tofile(path: Path) -> None:
    np.frombuffer(TAINT, dtype=np.uint8).tofile(str(path))


def _leak_via_np_save(path: Path) -> None:
    np.save(str(path), np.frombuffer(TAINT, dtype=np.uint8))


def _leak_via_os_write(path: Path) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT)
    try:
        os.write(fd, TAINT)
    finally:
        os.close(fd)


@pytest.mark.privacy
@pytest.mark.parametrize(
    ("primitive", "write"),
    [
        ("Path.write_bytes", _leak_via_write_bytes),
        ("Path.write_text", _leak_via_write_text),
        ("Path.open", _leak_via_path_open),
        ("builtins.open", _leak_via_builtin_open),
        ("ndarray.tofile", _leak_via_tofile),
        ("np.save", _leak_via_np_save),
        ("os.write", _leak_via_os_write),
    ],
)
def test_the_recorder_catches_a_frame_written_by(
    primitive: str,
    write: Callable[[Path], None],
    disk_writes: DiskWriteRecorder,
    tmp_path: Path,
) -> None:
    write(tmp_path / "leaked.bin")

    assert disk_writes.tainted(), f"{primitive} escaped the recorder"


@pytest.mark.privacy
def test_the_recorder_catches_a_frame_written_by_cv2_imwrite(
    disk_writes: DiskWriteRecorder, tmp_path: Path
) -> None:
    """The most plausible real leak: a debug `imwrite` left in a worker loop."""
    cv2 = pytest.importorskip("cv2")

    cv2.imwrite(str(tmp_path / "leaked.png"), _tainted_frame(0.0).image)

    assert disk_writes.tainted(), "cv2.imwrite escaped the recorder"


@pytest.mark.privacy
def test_the_recorder_ignores_reads(disk_writes: DiskWriteRecorder, tmp_path: Path) -> None:
    """A recorder that fires on reads would make the invariant tests meaningless."""
    source = tmp_path / "readable.bin"
    Path(source).write_bytes(TAINT)
    disk_writes.attempts.clear()

    assert source.read_bytes() == TAINT
    assert disk_writes.attempts == []
