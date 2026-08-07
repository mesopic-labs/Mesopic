"""The model cache: content-addressed, atomic, and licence-recording.

These tests never touch the network. The download and quantize steps are injected, so
what is under test here is the *lifecycle* — cache hit/miss, atomicity, content
addressing, licence metadata — rather than any particular model's arithmetic. The real
fetch-and-quantize path is exercised by a `slow` test that does hit the network.

Atomicity is the criterion that matters most (P1.4): the appliance cannot be SSH'd into
to clean up a half-written model, so a process killed mid-quantize must leave the cache
either empty or complete, never partial.
"""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort
import pytest

from muster.detector.model_manager import MODELS, ModelManager, ModelSpec
from muster.errors import ModelError


def _fake_download(spec: ModelSpec, dest: Path) -> None:
    dest.write_bytes(b"fp32:" + spec.name.encode())


def _fake_quantize(src: Path, dest: Path) -> None:
    dest.write_bytes(src.read_bytes().replace(b"fp32:", b"int8:"))


def test_ensure_builds_and_caches_when_absent(tmp_path: Path) -> None:
    downloaded: list[str] = []

    def counting_download(spec: ModelSpec, dest: Path) -> None:
        downloaded.append(spec.name)
        _fake_download(spec, dest)

    manager = ModelManager(tmp_path, download=counting_download, quantize=_fake_quantize)

    artefact = manager.ensure("yolox-nano")

    assert artefact.path.exists()
    assert artefact.path.read_bytes() == b"int8:yolox-nano"
    assert artefact.model_name == "yolox-nano"
    assert artefact.quantization == "int8"
    assert artefact.licence == "Apache-2.0"
    assert downloaded == ["yolox-nano"]


def test_ensure_is_idempotent_and_does_not_refetch(tmp_path: Path) -> None:
    downloads = 0

    def counting_download(spec: ModelSpec, dest: Path) -> None:
        nonlocal downloads
        downloads += 1
        _fake_download(spec, dest)

    manager = ModelManager(tmp_path, download=counting_download, quantize=_fake_quantize)

    first = manager.ensure("yolox-nano")
    second = manager.ensure("yolox-nano")

    assert first.path == second.path
    assert downloads == 1


def test_failed_build_leaves_no_artefact_and_no_scratch(tmp_path: Path) -> None:
    """Killed mid-quantize must leave the cache empty, never partial.

    The appliance has no operator to clear a half-written model by hand, so a partial
    artefact is not a transient annoyance — it is a box that never detects again.
    """

    def exploding_quantize(src: Path, dest: Path) -> None:
        dest.write_bytes(b"half-written")
        message = "killed mid-quantize"
        raise RuntimeError(message)

    manager = ModelManager(tmp_path, download=_fake_download, quantize=exploding_quantize)

    with pytest.raises(RuntimeError):
        manager.ensure("yolox-nano")

    assert list(tmp_path.iterdir()) == []


def test_truncated_cache_entry_is_rebuilt(tmp_path: Path) -> None:
    """A zero-byte cache entry means an earlier run died at the wrong moment.

    ``exists()`` is not the same question as ``is usable``, and returning a truncated
    model hands the inference runtime a file it will fail to load on every frame.
    """
    manager = ModelManager(tmp_path, download=_fake_download, quantize=_fake_quantize)
    artefact = manager.ensure("yolox-nano")
    artefact.path.write_bytes(b"")

    rebuilt = manager.ensure("yolox-nano")

    assert rebuilt.path.read_bytes() == b"int8:yolox-nano"


def test_unknown_model_is_rejected_by_name(tmp_path: Path) -> None:
    manager = ModelManager(tmp_path, download=_fake_download, quantize=_fake_quantize)

    with pytest.raises(ModelError, match="unknown model"):
        manager.ensure("definitely-not-a-model")


def test_model_without_verified_checksum_refuses_to_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unpinned model must fail loudly rather than fetch whatever is at the URL.

    An unverified model is arbitrary code the inference runtime will execute, so a
    missing digest is a hard stop, not a warning.
    """
    monkeypatch.setitem(
        MODELS,
        "unpinned",
        ModelSpec(name="unpinned", url="", sha256="", licence="Apache-2.0", input_size=416),
    )
    manager = ModelManager(tmp_path, quantize=_fake_quantize)

    with pytest.raises(ModelError, match="no verified download URL or checksum"):
        manager.ensure("unpinned")


def test_registered_models_are_pinned_to_a_verified_artefact() -> None:
    """Every shipped model carries a real URL and a full-length digest.

    This is what stops an empty or hand-waved registry entry reaching a user, where it
    would surface as a first-run failure on a box nobody can log into.
    """
    for spec in MODELS.values():
        assert spec.url.startswith("https://"), spec.name
        assert len(spec.sha256) == 64, spec.name
        assert set(spec.sha256) <= set("0123456789abcdef"), spec.name
        assert spec.input_size > 0, spec.name


@pytest.mark.slow
def test_real_fetch_and_quantize_produces_a_loadable_int8_model(tmp_path: Path) -> None:
    """The unfaked path: download the pinned weight, quantize it, load it.

    Everything above injects the download and quantize steps, which proves the cache
    lifecycle but not that either real step works. This one does the actual thing, so a
    broken URL, a stale digest, or a quantizer that emits an unloadable graph is caught
    here rather than on a user's first run.
    """
    manager = ModelManager(tmp_path)
    artefact = manager.ensure("yolox-nano")

    assert artefact.path.stat().st_size > 0
    # Quantization must actually shrink it; INT8 weights are ~a quarter of FP32.
    assert artefact.path.stat().st_size < 3_659_407

    session = ort.InferenceSession(str(artefact.path), providers=["CPUExecutionProvider"])
    (model_input,) = session.get_inputs()
    assert model_input.shape == [1, 3, 416, 416]


def test_every_registered_model_is_permissively_licensed() -> None:
    """ADR-0013: no AGPL anywhere in the default install.

    Apache-2.0 is a hard filter applied before the N100 benchmark, not after, so a
    model added later cannot quietly reintroduce copyleft into the default path.
    """
    assert {spec.licence for spec in MODELS.values()} <= {"Apache-2.0", "MIT", "BSD-3-Clause"}
