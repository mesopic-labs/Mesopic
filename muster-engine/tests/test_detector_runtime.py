"""Runtime selection: which native runtime executes the detector's graph (ADR-0012).

None of these tests need an accelerator, and that is the point. The probe is separated
from the *decision*, so what runtime an operator asked for is testable on any box and
whether it can be honoured is the only part that touches hardware.

The load-bearing test here is `test_auto_selection_does_not_pick_an_accelerator`: the ADR
gates promoting an accelerator to the default on a measured win, and that measurement does
not exist yet (P1.7). Auto-selection resolving to anything but ORT-CPU is a decision that
must go through the ADR, not through this module.
"""

from __future__ import annotations

import sys

import pytest

from muster.detector.runtime import (
    DEFAULT_RUNTIME,
    RUNTIME_ENV_VAR,
    artefact_tag,
    available_runtimes,
    is_available,
    resolve_runtime,
)
from muster.errors import ModelError
from muster.types import Runtime


def test_runtime_values_are_stable_wire_strings() -> None:
    """These strings are a contract, not an implementation detail.

    An operator types them into `MUSTER_DETECTOR_RUNTIME` and P2.1 will accept them in
    `muster.yaml`, so renaming one silently breaks a deployment that was working.
    """
    assert Runtime.ORT_CPU.value == "ort-cpu"
    assert Runtime.OPENVINO.value == "openvino"


def test_ort_cpu_and_openvino_share_an_artefact_tag() -> None:
    """Switching runtime must not re-download and re-quantize identical bytes.

    The quantized ONNX graph is the same file whichever runtime loads it. A cache key
    that separated them would cost a first-run download on a metered connection to
    produce a byte-identical result.
    """
    assert artefact_tag(Runtime.ORT_CPU) == artefact_tag(Runtime.OPENVINO)


def test_every_runtime_has_an_artefact_tag() -> None:
    """A runtime with no tag is a `KeyError` on first run, on a box nobody can log into."""
    for runtime in Runtime:
        assert artefact_tag(runtime)


def test_ort_cpu_is_always_available() -> None:
    """`onnxruntime` is a hard dependency: if it were missing we could not have started."""
    assert is_available(Runtime.ORT_CPU) is True


def test_available_runtimes_always_offers_the_guaranteed_path() -> None:
    assert Runtime.ORT_CPU in available_runtimes()


def test_auto_selection_is_the_guaranteed_path() -> None:
    selection = resolve_runtime(env={})

    assert selection.runtime is DEFAULT_RUNTIME
    assert selection.runtime is Runtime.ORT_CPU
    assert selection.explicit is False


def test_auto_selection_does_not_pick_an_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even with every accelerator available, auto-selection stays on the guaranteed path.

    ADR-0012 gates promoting an accelerator to default on a measured win at equal
    accuracy, and the perf table that would show one is owed by P1.7. Until it exists,
    opportunistic selection would mean the M0 gate is measured on a path that is not the
    one most boxes will run.
    """
    monkeypatch.setattr("muster.detector.runtime.is_available", lambda _runtime: True)

    selection = resolve_runtime(env={})

    assert selection.runtime is Runtime.ORT_CPU
    assert selection.explicit is False


def test_explicit_argument_wins_over_the_environment() -> None:
    selection = resolve_runtime(Runtime.ORT_CPU, env={RUNTIME_ENV_VAR: "openvino"})

    assert selection.runtime is Runtime.ORT_CPU
    assert selection.explicit is True


def test_environment_variable_selects_a_runtime() -> None:
    selection = resolve_runtime(env={RUNTIME_ENV_VAR: "openvino"})

    assert selection.runtime is Runtime.OPENVINO
    assert selection.explicit is True


def test_environment_variable_is_case_and_whitespace_insensitive() -> None:
    """An operator writing a compose file should not be punished for a capital letter."""
    selection = resolve_runtime(env={RUNTIME_ENV_VAR: "  OpenVINO \n"})

    assert selection.runtime is Runtime.OPENVINO
    assert selection.explicit is True


def test_empty_environment_variable_is_treated_as_unset() -> None:
    """An exported-but-empty variable is how a shell says "no value".

    Reading it as a request for a runtime named `""` would turn an unset default into a
    startup crash.
    """
    selection = resolve_runtime(env={RUNTIME_ENV_VAR: "   "})

    assert selection.runtime is DEFAULT_RUNTIME
    assert selection.explicit is False


def test_unknown_runtime_name_is_rejected_loudly() -> None:
    """An env var is a trust boundary: parse it into the typed value or reject it.

    Falling back silently on a typo hands the operator a box that is slower than they
    asked for with nothing anywhere saying why.
    """
    with pytest.raises(ModelError, match="tensorrt"):
        resolve_runtime(env={RUNTIME_ENV_VAR: "tensorrt"})


def test_rejected_runtime_name_lists_what_would_have_worked() -> None:
    with pytest.raises(ModelError, match="ort-cpu"):
        resolve_runtime(env={RUNTIME_ENV_VAR: "definitely-not-a-runtime"})


def test_probe_never_propagates_an_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken accelerator install must read as "absent", never as a crash.

    Accelerators are pure upside (ADR-0003); a probe that raises would make the presence
    of a half-installed optional package fatal to a box that never needed it.
    """
    monkeypatch.setitem(sys.modules, "openvino", None)

    assert is_available(Runtime.OPENVINO) is False


def test_unavailable_accelerator_is_absent_from_available_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "openvino", None)

    assert Runtime.OPENVINO not in available_runtimes()
