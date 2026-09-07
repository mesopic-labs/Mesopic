"""The model cache: content-addressed, atomic, and licence-recording.

These tests never touch the network. The download and transform steps are injected, so
what is under test here is the *lifecycle* — cache hit/miss, atomicity, content
addressing, licence metadata — rather than any particular model's arithmetic. The real
fetch path is exercised by `slow` tests that do hit the network.

Atomicity is the criterion that matters most (P1.4): the appliance cannot be SSH'd into
to clean up a half-written model, so a killed build must leave the cache either empty or
complete, never partial.

One `slow` test here guards a different failure entirely: that the published artefact
still *behaves* like the weights that were downloaded. The pipeline used to apply dynamic
INT8 quantization, which silently reduced this graph to noise and shipped a detector that
found nobody. Every test above passed throughout, because a fake transform and a loadable
file both looked fine.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import onnxruntime as ort
import pytest

from mesopic.detector.model_manager import MODELS, ModelManager, ModelSpec
from mesopic.errors import ModelError
from mesopic.types import Runtime

_PINNED_FP32_BYTES = 3_659_407
"""Size of the pinned yolox-nano release asset, as published unchanged."""


def _fake_download(spec: ModelSpec, dest: Path) -> None:
    dest.write_bytes(b"fp32:" + spec.name.encode())


def _fake_transform(src: Path, dest: Path) -> None:
    dest.write_bytes(src.read_bytes().replace(b"fp32:", b"built:"))


def test_ensure_builds_and_caches_when_absent(tmp_path: Path) -> None:
    downloaded: list[str] = []

    def counting_download(spec: ModelSpec, dest: Path) -> None:
        downloaded.append(spec.name)
        _fake_download(spec, dest)

    manager = ModelManager(tmp_path, download=counting_download, transform=_fake_transform)

    artefact = manager.ensure("yolox-nano")

    assert artefact.path.exists()
    assert artefact.path.read_bytes() == b"built:yolox-nano"
    assert artefact.model_name == "yolox-nano"
    assert artefact.quantization == "fp32"
    assert artefact.licence == "Apache-2.0"
    assert downloaded == ["yolox-nano"]


def test_ensure_is_idempotent_and_does_not_refetch(tmp_path: Path) -> None:
    downloads = 0

    def counting_download(spec: ModelSpec, dest: Path) -> None:
        nonlocal downloads
        downloads += 1
        _fake_download(spec, dest)

    manager = ModelManager(tmp_path, download=counting_download, transform=_fake_transform)

    first = manager.ensure("yolox-nano")
    second = manager.ensure("yolox-nano")

    assert first.path == second.path
    assert downloads == 1


def test_failed_build_leaves_no_artefact_and_no_scratch(tmp_path: Path) -> None:
    """A killed build must leave the cache empty, never partial.

    The appliance has no operator to clear a half-written model by hand, so a partial
    artefact is not a transient annoyance — it is a box that never detects again.
    """

    def exploding_transform(src: Path, dest: Path) -> None:
        dest.write_bytes(b"half-written")
        message = "killed mid-build"
        raise RuntimeError(message)

    manager = ModelManager(tmp_path, download=_fake_download, transform=exploding_transform)

    with pytest.raises(RuntimeError):
        manager.ensure("yolox-nano")

    assert list(tmp_path.iterdir()) == []


def test_truncated_cache_entry_is_rebuilt(tmp_path: Path) -> None:
    """A zero-byte cache entry means an earlier run died at the wrong moment.

    ``exists()`` is not the same question as ``is usable``, and returning a truncated
    model hands the inference runtime a file it will fail to load on every frame.
    """
    manager = ModelManager(tmp_path, download=_fake_download, transform=_fake_transform)
    artefact = manager.ensure("yolox-nano")
    artefact.path.write_bytes(b"")

    rebuilt = manager.ensure("yolox-nano")

    assert rebuilt.path.read_bytes() == b"built:yolox-nano"


def test_unknown_model_is_rejected_by_name(tmp_path: Path) -> None:
    manager = ModelManager(tmp_path, download=_fake_download, transform=_fake_transform)

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
    manager = ModelManager(tmp_path, transform=_fake_transform)

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
def test_real_fetch_produces_a_loadable_graph(tmp_path: Path) -> None:
    """The unfaked path: download the pinned weight and load it.

    Everything above injects the download and transform steps, which proves the cache
    lifecycle but not that either real step works. This one does the actual thing, so a
    broken URL, a stale digest, or a transform that emits an unloadable graph is caught
    here rather than on a user's first run.
    """
    manager = ModelManager(tmp_path)
    artefact = manager.ensure("yolox-nano")

    assert artefact.path.stat().st_size == _PINNED_FP32_BYTES

    session = ort.InferenceSession(str(artefact.path), providers=["CPUExecutionProvider"])
    (model_input,) = session.get_inputs()
    assert model_input.shape == [1, 3, 416, 416]


@pytest.mark.slow
def test_published_artefact_is_the_bytes_whose_digest_was_verified(tmp_path: Path) -> None:
    """What we publish must be what we verified — the guard this cache did not have.

    The pipeline used to quantize the downloaded graph before publishing it, and that
    step reduced the detector to noise while every other test stayed green: the file
    existed, loaded, and had the right input shape. Nothing compared the artefact to the
    weights it came from.

    So this asserts the artefact's digest *is* the pinned digest. While the transform is
    the identity that is exactly the contract; the moment someone reintroduces a
    transform this test fails and makes them prove the new artefact's behaviour on real
    imagery rather than assuming it, which is the forcing function that was missing.
    """
    manager = ModelManager(tmp_path)
    artefact = manager.ensure("yolox-nano")

    digest = hashlib.sha256(artefact.path.read_bytes()).hexdigest()
    assert digest == MODELS["yolox-nano"].sha256


def test_artefact_path_carries_the_runtime_derived_tag(tmp_path: Path) -> None:
    """The cache filename is a contract: it is what makes a warm cache a cache hit.

    It also strands the poisoned `int8` entries the old pipeline left on disk, instead
    of loading one and detecting nothing.
    """
    manager = ModelManager(tmp_path, download=_fake_download, transform=_fake_transform)

    artefact = manager.ensure("yolox-nano")

    assert artefact.path.name == "yolox-nano.fp32.op12.onnx"


def test_switching_runtime_reuses_the_cached_artefact(tmp_path: Path) -> None:
    """The graph is the same bytes whichever runtime loads it (ADR-0012).

    Re-fetching it on a runtime switch would spend a first-run download on a metered
    connection to produce a byte-identical file.
    """
    downloads = 0

    def counting_download(spec: ModelSpec, dest: Path) -> None:
        nonlocal downloads
        downloads += 1
        _fake_download(spec, dest)

    manager = ModelManager(tmp_path, download=counting_download, transform=_fake_transform)

    on_cpu = manager.ensure("yolox-nano", runtime=Runtime.ORT_CPU)
    on_openvino = manager.ensure("yolox-nano", runtime=Runtime.OPENVINO)

    assert on_cpu.path == on_openvino.path
    assert downloads == 1


def test_every_registered_model_is_permissively_licensed() -> None:
    """ADR-0013: no AGPL anywhere in the default install.

    Apache-2.0 is a hard filter applied before the N100 benchmark, not after, so a
    model added later cannot quietly reintroduce copyleft into the default path.
    """
    assert {spec.licence for spec in MODELS.values()} <= {"Apache-2.0", "MIT", "BSD-3-Clause"}


def test_locate_finds_nothing_before_the_model_has_been_built(tmp_path: Path) -> None:
    """`mesopic doctor` asks what is on disk; asking must not start a download."""
    manager = ModelManager(tmp_path, download=_fake_download, transform=_fake_transform)

    assert manager.locate("yolox-nano") is None


def test_locate_returns_the_cached_artefact_without_fetching(tmp_path: Path) -> None:
    downloads = 0

    def counting_download(spec: ModelSpec, dest: Path) -> None:
        nonlocal downloads
        downloads += 1
        _fake_download(spec, dest)

    manager = ModelManager(tmp_path, download=counting_download, transform=_fake_transform)
    built = manager.ensure("yolox-nano")

    located = manager.locate("yolox-nano")

    assert located is not None
    assert located.path == built.path
    assert located.licence == "Apache-2.0"
    assert downloads == 1


def test_locate_ignores_a_half_written_entry(tmp_path: Path) -> None:
    """A zero-byte file is what a killed build leaves; it is not a cached model."""
    manager = ModelManager(tmp_path, download=_fake_download, transform=_fake_transform)
    (tmp_path / "yolox-nano.fp32.op12.onnx").touch()

    assert manager.locate("yolox-nano") is None


def test_locate_refuses_a_model_it_does_not_know(tmp_path: Path) -> None:
    manager = ModelManager(tmp_path, download=_fake_download, transform=_fake_transform)

    with pytest.raises(ModelError):
        manager.locate("no-such-model")
