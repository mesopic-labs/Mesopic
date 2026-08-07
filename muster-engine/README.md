# muster-engine

The MIT-licensed, on-device Muster engine. It does the vision: RTSP/ONVIF in,
foot-points and metrics out, frames discarded the instant they are processed.

CPU-only capable by design — an Intel N100-class mini-PC with no GPU is the baseline, and
every accelerator (OpenVINO, Coral, CUDA/TensorRT) is opportunistic upside, never a
requirement.

Build and test from the repository root, not from here — the toolchain config and the
lockfile live one level up:

```bash
make setup     # uv sync + git hooks
make check     # lint, types, module boundaries, tests
```

See the repository [README](../README.md) for the quickstart and configuration.

## Licence

MIT — see [LICENSE](./LICENSE). Model weights are **not** bundled: they are fetched at
runtime as separately-licensed artefacts, which is what keeps the model's licence
separable from this code (ADR-0008, ADR-0013). The default install carries no AGPL
anywhere; `muster[ultralytics]` is opt-in and AGPL-3.0.
