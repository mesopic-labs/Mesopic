# PLAN — ADR-0012: inference-runtime selection

> **Working document.** Delete this file (or move it to `Muster-docs`) before the
> repository goes public — see `CLAUDE.md`, "This repository is private, and is written as
> if it were not". It is not user-facing documentation.

**Task:** close the ADR-0012 debt. `P1.4` shipped the detector but left ADR-0012 at
`Status: Proposed`, and left the code claiming a runtime probe that does not exist.

**Read before starting:**

- `../Muster-docs/docs/01-architecture/adr/0012-inference-runtime-selection.md` — the ADR
  being resolved. Read the whole thing.
- `../Muster-docs/docs/01-architecture/engine-architecture.md` §6 — the detector's
  responsibilities and the model-lifecycle diagram.
- `CLAUDE.md` in this repo — the hard invariants. Nothing below may break one.

**Do not** copy any text, numbers, or rationale out of `Muster-docs` into this repository,
into a commit message, or into a PR description. Cite the file path instead. `Muster-docs`
is private and stays that way.

---

## 1. What exists today

| File | State |
|---|---|
| `muster-engine/src/muster/detector/onnx_detector.py` | Working. `_open_session` hardcodes `providers=["CPUExecutionProvider"]`. Its module docstring already *claims* OpenVINO/Coral/CUDA "are probed at load time" — currently false. |
| `muster-engine/src/muster/detector/model_manager.py` | Working. `_artefact_path` hardcodes the runtime segment as the literal `.ort.` in the filename. |
| `muster-engine/src/muster/detector/detector.py` | The `Detector` protocol. **Not changed by this plan.** |
| `muster-engine/src/muster/types.py` | Shared vocabulary. Holds every `StrEnum` the pipeline passes between modules. |
| `muster-engine/pyproject.toml` | Already declares an `openvino = ["openvino>=2024.4"]` extra. Nothing imports it — the extra is currently dead. |
| `pyproject.toml` (workspace root) | Holds ruff / mypy / pytest / import-linter config. |

**The gap between the ADR and the code:** the ADR decides that runtime selection is
opportunistic and load-time with a fallback to ORT-CPU, that the cache is content-addressed
by `(model, quantization, opset, runtime)`, and that OpenVINO is an opt-in accelerator
path. None of that is implemented. The runtime is a hardcoded string in two places.

---

## 2. Design decisions — all of them, already made

You are implementing these. Do not re-decide any of them, and do not invent an alternative
if one looks nicer while you are in the file.

**D1. OpenVINO is reached through the native `openvino` package, not through ONNX
Runtime's `OpenVINOExecutionProvider`.** The EP route requires the `onnxruntime-openvino`
wheel, which provides the same `onnxruntime` module as the plain `onnxruntime` wheel this
project already depends on; installing both is a broken environment. The `openvino` extra
already declared in `muster-engine/pyproject.toml` is the native package. Use it.

**D2. OpenVINO plugs in as an `InferenceSession` adapter, not as a second detector class.**
`OnnxDetector` already accepts an injected `session: InferenceSession | None`, where
`InferenceSession` is a two-method Protocol (`get_inputs`, `run`). That Protocol exists
precisely for this — read its docstring at `onnx_detector.py:34`. Write an adapter that
satisfies it. **`OnnxDetector.detect` is not modified.** There is no abstract base class, no
shared mixin, and no second copy of the letterbox→run→decode flow.

**D3. Auto-selection resolves to ORT-CPU, always, for now.** OpenVINO is reachable only by
explicit request. The ADR gates promoting an accelerator to default on a measured win in
the perf table, and that table does not exist yet (it is owed by `P1.7`). Auto-selecting an
unmeasured runtime would also mean the M0 gate gets measured on a path that is not the
guaranteed one. The auto policy lives in exactly one function so that flipping it later,
once the numbers exist, is a one-line change.

**D4. An explicitly requested runtime that is unavailable is a hard error; an
auto-selected one silently falls back.** An operator who asked for OpenVINO and silently
got CPU has an unexplainable performance mystery. An operator who asked for nothing gets
the guaranteed path. Both behaviours are pinned by tests.

**D5. The artefact cache key is derived from the runtime, but runtimes that consume
identical bytes share one entry.** The quantized ONNX file is byte-identical whether ORT or
OpenVINO loads it, so keying them apart would force a pointless re-download and re-quantize
when a user flips runtime. A `Runtime.artefact_tag` maps runtime → the format of the bytes;
`ORT_CPU` and `OPENVINO` both map to `"int8"`. A future runtime that needs a genuinely
different artefact (a Coral `.tflite`, a TensorRT engine) returns a different tag and
therefore gets its own cache entry, with no clobbering. That is the ADR's intent honoured
without inventing a distinction that does not exist in the bytes.

**D6. Runtime selection is a constructor argument plus one env var. It does not touch
`muster.yaml`.** `muster-engine/src/muster/config/schema.py` is still an empty stub owned by
`P2.1`. Adding a `detector.runtime` key to `examples/muster.yaml` now would document a key
no loader reads — exactly the drift `muster-engine/tests/test_documented_config.py` exists
to catch. The env var `MUSTER_DETECTOR_RUNTIME` is the operator-facing surface until `P2.1`
wires the YAML key through to the same `resolve_runtime()` call.

**D7. No logging framework.** The engine has no logger and this plan does not introduce
one. Operator visibility into which runtime was selected comes from `available_runtimes()`
and `resolve_runtime()` being callable, which `muster doctor` (P1.6) and `/healthz` (P3.1)
will render. Do not add `import logging` anywhere.

**D8. `Runtime` lives in `muster/types.py`, not in the detector package.** It is crossing
module boundaries: `P2.1` config will parse it and `P3.1` `/healthz` will report it.
`muster/types.py` is where the shared vocabulary lives and where every other `StrEnum` in
the engine already is.

---

## 3. Ground rules for every step

- **Test first.** Write the failing test, watch it fail for the stated reason, then write
  the smallest implementation that passes. This is the repo's SDLC, not a suggestion.
- **`make check` must be green at the end of every step**, not only at the end of the plan.
  It runs `ruff check`, `ruff format --check`, `mypy --strict`, `lint-imports`, and
  `pytest -m "not slow"`.
- **No blanket `# noqa` or `# type: ignore`.** Where this plan specifies a `noqa`, it names
  the exact rule and the justification to write inline. Do not add others.
- **Comments explain why, never what.** Match the surrounding files — read
  `onnx_detector.py` and `model_manager.py` before writing a line and copy their register.
- Imports at the top of the file. The **one** exception is a heavy native runtime import
  inside a function, which both existing detector files already do with an inline reason —
  follow that pattern exactly (`onnx_detector.py:110`, `model_manager.py:132`), including
  the `# noqa: PLC0415` and the reason comment.
- Branch name: `feat/adr-0012-runtime-selection`. Never commit to `main`.
- Conventional-commit subjects, one commit per step, e.g. `feat(detector): add the Runtime
  vocabulary and the artefact tag`.

---

## Step 0 — Preflight

**Do:**

```bash
git checkout -b feat/adr-0012-runtime-selection
make check
```

**Expected:** all five gates pass on a clean `main`. If anything is red before you start,
stop and report it — do not begin work on a red baseline.

---

## Step 1 — The `Runtime` vocabulary

**File to edit:** `muster-engine/src/muster/types.py`

Add to the `# --- Enumerations ---` section, immediately after `CameraState` (keep the
section's existing ordering style — enums are grouped, one blank-line-separated block each):

```python
class Runtime(StrEnum):
    """Which native runtime executes the detector's graph (ADR-0012).

    ``ORT_CPU`` is the guaranteed path: it is a hard dependency, it runs on Intel, AMD and
    ARM, and it is what the N100 perf gate is measured against. Everything else is an
    accelerator — selected when explicitly asked for, never required.
    """

    ORT_CPU = "ort-cpu"
    OPENVINO = "openvino"
```

**Do not** add Coral or CUDA members. They are deferred by the ADR; an enum member with no
implementation behind it is a lie the type checker will happily propagate.

**Test — new file `muster-engine/tests/test_detector_runtime.py`:**

Model this file's docstring and structure on `muster-engine/tests/test_model_manager.py`.

```python
def test_runtime_values_are_stable_wire_strings() -> None:
```
Assert `Runtime.ORT_CPU == "ort-cpu"` and `Runtime.OPENVINO == "openvino"`. These strings
are what an operator types into `MUSTER_DETECTOR_RUNTIME` and what `P2.1` will accept in
YAML, so they are a contract, not an implementation detail.

**Verify:**

```bash
uv run pytest muster-engine/tests/test_detector_runtime.py -v
make check
```

**Expected:** one test passes; `make check` green.

---

## Step 2 — The runtime module: probe, tag, resolve

**File to create:** `muster-engine/src/muster/detector/runtime.py`

Module docstring must state: ORT-CPU is the guaranteed path; accelerators are selected,
never required; auto-selection deliberately does not pick an accelerator until the perf
table exists (cite `P1.7`, not a number). End with `Implements ADR-0012.`

**Module constants:**

```python
DEFAULT_RUNTIME: Final = Runtime.ORT_CPU
RUNTIME_ENV_VAR: Final = "MUSTER_DETECTOR_RUNTIME"
```

**Public API — exactly these four names, in this order:**

```python
@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    """Which runtime to use, and whether the operator asked for it by name.

    The distinction is the whole fallback policy: an explicit request that cannot be
    honoured is an error, an automatic one degrades to the guaranteed path.
    """

    runtime: Runtime
    explicit: bool


def artefact_tag(runtime: Runtime) -> str: ...


def is_available(runtime: Runtime) -> bool: ...


def available_runtimes() -> tuple[Runtime, ...]: ...


def resolve_runtime(
    requested: Runtime | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> RuntimeSelection: ...
```

**Behaviour, precisely:**

`artefact_tag` — returns `"int8"` for both `ORT_CPU` and `OPENVINO`. Implement it as a
dict lookup keyed by `Runtime` (`_ARTEFACT_TAGS: Final[dict[Runtime, str]]`), not an
`if`/`elif` chain, so adding a runtime that needs its own artefact is a one-line data
change. The docstring carries decision **D5**'s reasoning.

`is_available` — `ORT_CPU` returns `True` unconditionally (`onnxruntime` is a hard
dependency; if it is missing the process could not have started). `OPENVINO` returns `True`
only if the `openvino` package imports **and** `openvino.Core().available_devices` is
non-empty. A package that imports but sees no device is not an available runtime.

The probe must never propagate an exception — the absence of an accelerator is a normal
condition, not an error. Wrap the import-and-query in:

```python
    except Exception:  # noqa: BLE001 - a probe must never propagate; no accelerator is not an error
        return False
```

This is the only `noqa` this plan authorises. The `openvino` import goes inside the
function with `# noqa: PLC0415` and a reason comment, matching
`onnx_detector.py:110` — the whole point is that a box without the extra never pays to
try.

`available_runtimes` — returns every `Runtime` for which `is_available` is `True`,
in `Runtime` declaration order. This is what `muster doctor` will render.

`resolve_runtime` — the single decision point. In order:

1. If `requested is not None`, that is the selection with `explicit=True`.
2. Otherwise read `RUNTIME_ENV_VAR` from `env` (defaulting to `os.environ` when `env` is
   `None`). If set and non-empty after `.strip()`, parse it case-insensitively
   (`.strip().lower()`) into a `Runtime`. On success: `explicit=True`. On an unparseable
   value: raise `ModelError` naming the bad value and listing the valid ones, sorted.
3. Otherwise return `RuntimeSelection(DEFAULT_RUNTIME, explicit=False)`.

`resolve_runtime` **does not probe**. It decides *what was asked for*; whether it can be
honoured is Step 4's job. Keeping those separate is what makes both testable without
hardware.

The `env` parameter exists so tests inject a mapping instead of mutating the process
environment. Type it `Mapping[str, str] | None` and import `Mapping` from
`collections.abc` under `TYPE_CHECKING`, matching how `onnx_detector.py` guards its
`NDArray` import.

**Tests — append to `muster-engine/tests/test_detector_runtime.py`:**

| Test name | Asserts |
|---|---|
| `test_ort_cpu_and_openvino_share_an_artefact_tag` | `artefact_tag(ORT_CPU) == artefact_tag(OPENVINO)`. Docstring: switching runtime must not re-download or re-quantize identical bytes. |
| `test_ort_cpu_is_always_available` | `is_available(Runtime.ORT_CPU) is True`. |
| `test_auto_selection_is_the_guaranteed_path` | `resolve_runtime(env={})` returns `ORT_CPU`, `explicit=False`. |
| `test_auto_selection_does_not_pick_an_accelerator` | With `env={}`, the result is `ORT_CPU` **even if** `is_available(OPENVINO)` is `True`. Monkeypatch `is_available` to return `True` for everything. This test is the executable form of decision **D3** — it is the one that must fail loudly if someone later makes auto-selection opportunistic without also updating the ADR. |
| `test_explicit_argument_wins_over_the_environment` | `resolve_runtime(Runtime.ORT_CPU, env={RUNTIME_ENV_VAR: "openvino"})` returns `ORT_CPU`, `explicit=True`. |
| `test_environment_variable_selects_a_runtime` | `env={RUNTIME_ENV_VAR: "openvino"}` returns `OPENVINO`, `explicit=True`. |
| `test_environment_variable_is_case_and_whitespace_insensitive` | `"  OpenVINO \n"` resolves to `OPENVINO`. An operator typing into a compose file should not be punished for a capital letter. |
| `test_empty_environment_variable_is_treated_as_unset` | `env={RUNTIME_ENV_VAR: "   "}` returns the default, `explicit=False`. An exported-but-empty variable is how a shell script says "unset", not a request for a runtime named `""`. |
| `test_unknown_runtime_name_is_rejected_loudly` | `env={RUNTIME_ENV_VAR: "tensorrt"}` raises `ModelError`; use `pytest.raises(ModelError, match="tensorrt")`. Docstring: an env var is a trust boundary — parse it into the typed value or reject it, never sanitise and continue. |
| `test_probe_never_propagates_an_exception` | Monkeypatch the openvino import path to raise, assert `is_available(Runtime.OPENVINO) is False`. Simplest injection: `monkeypatch.setitem(sys.modules, "openvino", None)` so the attribute access inside the probe raises. |

**Verify:**

```bash
uv run pytest muster-engine/tests/test_detector_runtime.py -v
make check
```

**Expected:** all listed tests pass; `make check` green. Note `filterwarnings = ["error"]`
is on in `pyproject.toml` — if importing `openvino` emits a warning on your machine, the
probe's `except` must still swallow it into `False`, not let a warning-as-error escape.

---

## Step 3 — Key the model cache by runtime

**File to edit:** `muster-engine/src/muster/detector/model_manager.py`

Change two methods. Nothing else in the file moves.

```python
    def ensure(self, model_name: str, *, runtime: Runtime = DEFAULT_RUNTIME) -> ModelArtefact:
```

```python
    def _artefact_path(self, spec: ModelSpec, runtime: Runtime) -> Path:
        return self._cache_dir / f"{spec.name}.{artefact_tag(runtime)}.op{OPSET}.onnx"
```

Import `Runtime` from `muster.types` and `DEFAULT_RUNTIME, artefact_tag` from
`muster.detector.runtime`, both at the top of the file.

Update `_artefact_path`'s existing docstring: it currently explains keying on runtime and
opset. Keep that reasoning and add why two runtimes may legitimately share an entry
(decision **D5**). Do not delete the existing sentences — they are still correct.

`ModelArtefact` is **not** changed. It describes the bytes on disk, and the bytes are
runtime-agnostic. Which runtime will execute them is the detector's business, not the
cache's.

**Note the filename change:** `yolox-nano.int8.op12.ort.onnx` becomes
`yolox-nano.int8.op12.onnx`. Any existing warm cache re-downloads and re-quantizes once.
That is acceptable — the engine is at `0.0.1` and unreleased. **Do not write migration
code and do not delete the orphaned old file.** Silently deleting from a user's cache
directory is not a thing this engine does.

**Tests — append to `muster-engine/tests/test_model_manager.py`**, reusing the existing
`_fake_download` / `_fake_quantize` helpers already in that file:

| Test name | Asserts |
|---|---|
| `test_artefact_path_carries_the_runtime_derived_tag` | `manager.ensure("yolox-nano").path.name == "yolox-nano.int8.op12.onnx"`. Pin the literal name — it is the cache contract. |
| `test_switching_runtime_reuses_the_cached_artefact` | `ensure("yolox-nano", runtime=ORT_CPU)` then `ensure("yolox-nano", runtime=OPENVINO)`; assert both return the same `path` and that the injected download ran exactly once. Docstring: the quantized graph is the same bytes for both runtimes; re-fetching it would be pure waste on a metered connection. |

**Verify:**

```bash
uv run pytest muster-engine/tests/test_model_manager.py -v
make check
```

**Expected:** the two new tests pass and all nine pre-existing tests in that file still
pass unchanged. If you had to edit an existing test's assertions beyond the artefact
filename, you changed more than this step authorises — revert and re-read.

---

## Step 4 — Honour the selected runtime when opening a session

**File to edit:** `muster-engine/src/muster/detector/onnx_detector.py`

**4a. `OnnxDetector.__init__` gains one keyword-only parameter**, inserted before
`session`:

```text
        runtime: Runtime | None = None,
```

It is forwarded to `_open_session` and is **ignored when `session` is not None** — an
injected session is already a runtime, and a test that passes both is a test with a
contradiction in it. Say that in the parameter's docstring line.

**4b. Add a read-only property**, placed immediately after the existing `input_size`
property so the two accessors sit together:

```python
    @property
    def runtime(self) -> Runtime:
        """Which runtime is actually executing the graph.

        What ``muster doctor`` and ``/healthz`` report, and the answer to "why is this box
        slower than that one".
        """
```

Store the backing value on `self._runtime` in `__init__`. When a session is injected,
`self._runtime` is `DEFAULT_RUNTIME` — the injected session is a test double and claiming
anything else would be a fiction.

**4c. `_open_session` signature becomes:**

```python
def _open_session(
    model_path: Path,
    *,
    intra_op_threads: int | None,
    selection: RuntimeSelection,
) -> InferenceSession: ...
```

Its logic, in order:

1. If `selection.runtime is Runtime.ORT_CPU`, open the ORT session exactly as today. This
   path is unchanged, including the existing `except Exception ... raise ModelError` block.
2. Otherwise, check `is_available(selection.runtime)`:
   - Available → build the adapter from Step 5 and return it.
   - Not available **and** `selection.explicit` → raise `ModelError` naming the requested
     runtime and saying how to install it (the `openvino` extra) — do not name a version
     number, the extra owns that.
   - Not available and **not** `selection.explicit` → fall back to the ORT-CPU branch.
     (With decision **D3** this branch is currently unreachable from `resolve_runtime`, but
     it is the fallback the ADR requires and it becomes live the moment auto-selection is
     allowed to pick an accelerator. Write it, and pin it with the test below by calling
     `_open_session` directly with a hand-built `RuntimeSelection`.)
3. If the accelerator is available but its session construction raises, apply the same
   rule: explicit → `ModelError`; automatic → fall back to ORT-CPU.

**Update the module docstring.** Its second paragraph currently claims Coral and
CUDA/TensorRT are probed at load time. They are not, and this plan does not add them.
Rewrite it to say what is true: ORT-CPU is the default and always works; OpenVINO is
available as an explicit opt-in; other accelerators are deferred by ADR-0012 and would
arrive behind this same `InferenceSession` seam. A docstring that overstates the code is
worse than no docstring.

**Tests — append to `muster-engine/tests/test_onnx_detector.py`**, reusing its existing
`FakeSession` helper:

| Test name | Asserts |
|---|---|
| `test_injected_session_reports_the_default_runtime` | `OnnxDetector(Path("unused.onnx"), session=FakeSession(...)).runtime is Runtime.ORT_CPU`. |
| `test_explicit_unavailable_runtime_is_an_error_not_a_silent_downgrade` | Monkeypatch `is_available` to `False`; call `_open_session` with `RuntimeSelection(Runtime.OPENVINO, explicit=True)`; expect `pytest.raises(ModelError, match="openvino")`. Docstring: an operator who asked for an accelerator and silently got CPU has an unexplainable perf mystery. |
| `test_automatic_unavailable_runtime_falls_back_to_cpu` | Same monkeypatch, `explicit=False`, and monkeypatch the ORT branch so no real model file is needed — assert the CPU path was taken rather than an exception raised. Docstring: accelerators are pure upside; their absence is never fatal. |

**Verify:**

```bash
uv run pytest muster-engine/tests/test_onnx_detector.py -v
make check
```

**Expected:** the three new tests pass; the eight pre-existing tests in that file pass
unchanged. `test_real_quantized_model_runs_on_a_1080p_frame` is `slow` and excluded from
`make check` — run it once explicitly at the end of this step:

```bash
uv run pytest muster-engine/tests/test_onnx_detector.py -m slow -v
```

**Expected:** it downloads the pinned weight, quantizes, loads, and passes. This is the
proof that the cache-filename change in Step 3 did not break the real path.

---

## Step 5 — The OpenVINO adapter

**File to create:** `muster-engine/src/muster/detector/openvino_session.py`

Module docstring: this is an adapter, not a detector. It presents an OpenVINO compiled
model through the same two-method `InferenceSession` Protocol that ONNX Runtime satisfies,
which is why the detector's `detect` path is identical on both runtimes. Note that the
model file is the same quantized ONNX in both cases (decision **D5**). End with
`Implements ADR-0012.`

**Public API:**

```python
@dataclass(frozen=True, slots=True)
class _InputInfo:
    """The two attributes `OnnxDetector` reads off a graph input."""

    name: str
    shape: list[int | str]


class OpenVinoSession:
    """An OpenVINO compiled model wearing ONNX Runtime's session interface."""

    def __init__(self, model_path: Path, *, num_threads: int | None = None) -> None: ...

    def get_inputs(self) -> list[Any]: ...

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]: ...
```

**Behaviour:**

- `__init__` imports `openvino` inside the function body with `# noqa: PLC0415` and a
  reason comment (same pattern as `onnx_detector.py:110`), builds a `Core`, and compiles
  the model for the `"CPU"` device. Compiling for `"CPU"` — not `"AUTO"` — is deliberate:
  `AUTO` can migrate the model to an iGPU mid-run, which would change the numerics under a
  running perf measurement. iGPU selection is a future decision, not this one.
- When `num_threads is not None`, pass it as OpenVINO's `INFERENCE_NUM_THREADS` config
  property at compile time. This is the OpenVINO equivalent of ORT's
  `intra_op_num_threads`, and it exists for the same reason — read the comment at
  `onnx_detector.py:114`: every camera is its own process and an unbounded thread pool per
  worker means workers fighting for the same cores.
- Any exception during compile is wrapped in `ModelError` with the model's filename,
  mirroring `_open_session`'s existing error handling verbatim in shape.
- `get_inputs()` returns one `_InputInfo` per model input. Take the name from OpenVINO's
  input node and the shape from its partial shape. **For each dimension: emit `int(dim)` if
  the dimension is static, otherwise emit the string `"?"`.** Do not raise here. The
  existing `_square_input_size` guard in `onnx_detector.py:128` already rejects a
  non-integer axis with a clear `ModelError`, and routing through it means both runtimes
  produce the identical error for the identical problem.
- `run()` ignores `output_names` (the caller passes `None`; `OnnxDetector.detect` always
  does) and returns `[array_for_output_0]`. `decode_yolox_output` reads `outputs[0]` and
  the pinned model has exactly one output. Return a `numpy` array, not an OpenVINO tensor
  wrapper — the postprocess module does array arithmetic on it.

**Wire it into Step 4:** `_open_session`'s OpenVINO branch constructs
`OpenVinoSession(model_path, num_threads=intra_op_threads)`. Import it at the top of
`onnx_detector.py` — it is a first-party module and costs nothing to import; only the
`openvino` package itself is lazy, and that is inside `OpenVinoSession.__init__`.

**Tests — new file `muster-engine/tests/test_openvino_session.py`:**

`openvino` is not installed in the dev environment and will not be on CI. Every fast test
here drives a **fake `Core`**, injected by monkeypatching `sys.modules["openvino"]` with a
`SimpleNamespace` exposing a `Core` class — the same technique
`test_onnx_detector.py`'s `FakeSession` uses, one level further out. Build the fake to
return a compiled model whose input reports name `"images"` and shape `[1, 3, 416, 416]`.

| Test name | Asserts |
|---|---|
| `test_get_inputs_reports_the_graph_name_and_shape` | Returns one input, `name == "images"`, `shape == [1, 3, 416, 416]`. |
| `test_a_dynamic_axis_is_surfaced_as_non_integer` | A fake whose second spatial dim is dynamic yields `"?"` in that slot, and feeding that session to `OnnxDetector` raises `ModelError` with "not a fixed NCHW square". Docstring: both runtimes must fail the same way on the same bad graph. |
| `test_run_returns_the_first_output_as_an_array` | `run(None, {"images": tensor})` returns a one-element list whose member is an `np.ndarray`. |
| `test_num_threads_is_passed_to_the_compiler` | The fake `Core` records its `compile_model` config; assert `INFERENCE_NUM_THREADS` is present when `num_threads=2` and absent when `None`. |
| `test_compile_failure_becomes_a_model_error` | Fake `Core.compile_model` raises; expect `pytest.raises(ModelError)` naming the file. |

Then one real test, `@pytest.mark.slow`, guarded so it does not fail on a box without the
extra:

```python
@pytest.mark.slow
def test_real_openvino_session_runs_the_pinned_model(tmp_path: Path) -> None:
    pytest.importorskip("openvino")
```

It fetches via `ModelManager(tmp_path).ensure("yolox-nano", runtime=Runtime.OPENVINO)`,
builds an `OnnxDetector` with an `OpenVinoSession`, runs one 1080p zero frame, and asserts
`detections == []` and `detector.input_size == 416` — mirroring
`test_real_quantized_model_runs_on_a_1080p_frame` exactly. A flat-zero frame producing no
people is a sanity check on the confidence gate, not an accuracy claim.

**Verify:**

```bash
uv run pytest muster-engine/tests/test_openvino_session.py -v
make check
```

**Expected:** the five fast tests pass; `make check` green. The slow test is skipped
locally (no `openvino` installed) — that is the correct outcome, not a failure.

---

## Step 6 — Toolchain: let mypy see the new import

**File to edit:** `pyproject.toml` (workspace root)

`openvino` is an optional extra and will not be installed when `mypy --strict` runs, so it
must be declared alongside the other unstubbed third-party packages. Add `"openvino.*"` to
the **existing** override block whose comment reads "Third-party packages without stubs.
Add here with a reason, never inline" — the one already listing `av.*`, `cv2.*`,
`onnxruntime.*`, `paho.*`. Extend that block's comment with the reason: `openvino` is an
opt-in extra (ADR-0012) and is absent from the default dev environment, so mypy has no
implementation to resolve.

Do **not** create a second override block, and do **not** add `openvino` to the required
`dependencies` list. It is an extra. Making it a hard dependency would break decision
**D1** and put a vendor runtime in every install.

**Verify:**

```bash
uv run mypy muster-engine/src muster-engine/tests
make check
```

**Expected:** clean. If mypy now reports `warn_unused_ignores` on anything you wrote,
remove the ignore rather than widening the override — `warn_unused_ignores` and
`ignore-without-code` are both on.

---

## Step 7 — Resolve the ADR

This step edits the **sibling private repository**, `../Muster-docs/`. It is a separate
commit in a separate repo; it is not part of this repo's PR diff. Reference it in the PR
description by path only — never paste its contents.

**File:** `../Muster-docs/docs/01-architecture/adr/0012-inference-runtime-selection.md`

1. Change `Status: Proposed` to `Status: Accepted (2026-08-11)`, matching the format ADR-0014
   already uses on its own status line.
2. The final bullet under **Consequences → Negative / tradeoffs** currently ends by saying
   the ADR is Proposed precisely because the benchmark number is still owed. That sentence
   is now the thing that has changed and it must be rewritten honestly, not deleted:
   the decision is accepted because the *portable default* does not depend on the missing
   measurement — auto-selection resolves to ORT-CPU regardless, and the accelerator is
   reachable only by explicit request. The perf table's job is narrowed from "decides the
   default" to "decides whether auto-selection may ever promote OpenVINO", and that
   promotion remains a future change to one function. Keep the rest of the bullet.
3. Add a short **Implementation** section at the end recording what shipped and where:
   `muster/types.py` (`Runtime`), `muster/detector/runtime.py` (probe, tag, resolution),
   `muster/detector/openvino_session.py` (the adapter), and the artefact-tag sharing
   decision. Cite module paths, no code.

**File:** `../Muster-docs/docs/01-architecture/adr/INDEX.md`

Change the Status cell in the 0012 row (line ~57) from `Proposed` to `Accepted`. **Change
nothing else in that file** — the surrounding prose about open ADRs also names 0011 and
0013, which are outside this task, and editing it risks stating something false about them.

**Verify:**

```bash
grep -n "^Status" ../Muster-docs/docs/01-architecture/adr/0012-inference-runtime-selection.md
grep -n "0012" ../Muster-docs/docs/01-architecture/adr/INDEX.md
```

**Expected:** the ADR reads `Status: Accepted (2026-08-11)`; the INDEX row reads
`Accepted`.

---

## Step 8 — Final gate

```bash
make check
uv run pytest -m slow -v
```

**Expected:** `make check` green. The slow suite downloads the pinned weight and exercises
the real ORT path; the OpenVINO slow test skips for want of the extra. Report the skip
explicitly — do not describe the suite as fully passing when one test was skipped.

Then open a PR. In the description: what changed, that ADR-0012 moved to Accepted in
`Muster-docs` (path only, no quotes), and the one-time cache-filename change from Step 3
so a reviewer with a warm cache is not surprised by a re-download. Do not self-approve and
do not merge — all generated code is a proposal.

---

## 4. Edge cases and the required handling

| # | Case | Required behaviour |
|---|---|---|
| 1 | `MUSTER_DETECTOR_RUNTIME` unset | Auto → `ORT_CPU`, `explicit=False`. |
| 2 | Set to an unknown name (`tensorrt`, a typo) | `ModelError` naming the bad value and listing valid ones. Never fall back silently — a typo that silently yields the default is a support ticket nobody can diagnose. |
| 3 | Set with odd case/whitespace (`" OpenVINO\n"`) | Normalise with `.strip().lower()`, resolve to `OPENVINO`. |
| 4 | Set but empty or whitespace-only | Treated as unset. That is how a shell says "no value", not a request for a runtime named `""`. |
| 5 | `openvino` requested explicitly, package not installed | `ModelError` telling the operator to install the `openvino` extra. No version number in the message. |
| 6 | `openvino` installed, `available_devices` empty | `is_available` → `False`. An importable package that sees no hardware is not an available runtime. |
| 7 | The probe itself raises (broken install, driver error, import-time warning under `filterwarnings=error`) | Swallowed to `False` via the single authorised `# noqa: BLE001`. A probe must never take down the engine. |
| 8 | `openvino` available and auto-selection is running | Still resolves to `ORT_CPU` (decision **D3**), pinned by `test_auto_selection_does_not_pick_an_accelerator`. |
| 9 | Accelerator available but its session fails to compile | Explicit → `ModelError`. Automatic → fall back to ORT-CPU. |
| 10 | Existing warm cache with the old `.ort.` filename | One re-download and re-quantize. No migration code. The orphaned file is left in place — this engine does not delete from a user's cache directory. |
| 11 | Caller passes `runtime=OPENVINO` to `OnnxDetector` but `runtime=ORT_CPU` to `ModelManager.ensure` | Harmless today: both map to artefact tag `"int8"`, so it is the same file. `test_switching_runtime_reuses_the_cached_artefact` is what keeps it harmless. |
| 12 | Both `runtime=` and `session=` passed to `OnnxDetector` | `session` wins, `runtime` is ignored, `self._runtime` is `DEFAULT_RUNTIME`. Document it on the parameter; do not raise. |
| 13 | OpenVINO graph with a dynamic input axis | Adapter emits `"?"` for that dimension; the existing `_square_input_size` guard raises `ModelError`. Identical failure on both runtimes. |
| 14 | `close()` on a detector backed by `OpenVinoSession` | `OnnxDetector.close()` drops the reference; OpenVINO frees on garbage collection. No explicit release call, no `__del__`. |

---

## 5. Explicitly out of scope

Do not do these, even though they are adjacent and will look tempting:

- **The `detector.runtime` key in `muster.yaml` and `config/schema.py`.** That is `P2.1`.
  Adding a documented key no loader reads is the exact drift
  `tests/test_documented_config.py` exists to catch (decision **D6**).
- **`muster doctor` and `/healthz`.** `cli.py:42`'s `doctor` and `api/health.py` stay
  untouched. This plan gives them `available_runtimes()` and `resolve_runtime()` to call;
  wiring is `P1.6` and `P3.1`.
- **Coral, CUDA, TensorRT, or an iGPU device.** Deferred by the ADR. Do not add enum
  members for runtimes with no implementation.
- **Benchmarking either runtime.** The perf harness is `P1.7` and it runs on the N100, not
  in CI and not on a dev laptop. Do not add timing assertions.
- **Per-runtime accuracy tests.** ADR-0012 correctly notes INT8 numerics can differ between
  runtimes and that accuracy tests must run per active runtime. That belongs with the
  labelled ground-truth set at `P2.9`, which does not exist yet.
- **Touching `.github/workflows/`, `CODEOWNERS`, or release scripts.** Not needed here. If
  you believe one is needed, stop and ask.
- **Adding any dependency.** The `openvino` extra already exists. Nothing new goes into
  `dependencies`, and `uv.lock` should not change in this PR.

---

## 6. Invariant check before you open the PR

Confirm each, by inspection of your own diff:

1. No code path writes frame, crop, or bbox-pixel bytes anywhere. The new modules handle
   tensors in memory only. `tests/test_privacy_invariants.py` rglobs `src/` so the new
   files are swept automatically — but check by eye too.
2. No secret, token, hostname, or RTSP URL appears in any new file, test, or commit message.
3. No AGPL enters the default install. `openvino` is Apache-2.0 and is an extra, not a
   dependency.
4. `lint-imports` is green — the detector package still imports nothing from `analytics`,
   `store`, `aggregator`, or `api`.
5. No blanket `# noqa` or `# type: ignore`. The only `noqa`s in the diff are the two
   `PLC0415` lazy-import markers and the single authorised `BLE001` on the probe, each with
   an inline reason.
