#!/usr/bin/env bash
# ADR-0008 / ADR-0013: the engine image ships NO model weights. A model is a
# separately-licensed artefact fetched at runtime into /data — which is exactly what
# keeps the model's licence separable from this MIT image. If a weight is ever baked in,
# the image starts carrying that licence, and we would not notice by reading a diff.
#
# Usage: scripts/assert-no-model-weights.sh [image-tag]

set -euo pipefail

IMAGE="${1:-muster-engine:dev}"

# Two things make a file a real weight rather than a false positive:
#
#   1. It lives in our code or the runtime data directory, not inside a third-party
#      package. `onnxruntime` ships three tiny .onnx fixtures of its own (sigmoid,
#      mul_1, logreg_iris) for its test suite; they are part of an MIT dependency and
#      are not a detector.
#   2. Or it is large. A person detector is megabytes; a test fixture is bytes. Anything
#      over 1 MiB with a model extension is a weight no matter where it is hiding.
#
# Either condition alone fails the build.
readonly SIZE_LIMIT_BYTES=1048576

# No `|| true` here, deliberately. If docker cannot run the image, this check must fail
# loudly — a security assertion that reports PASS when it did not actually run is worse
# than no assertion at all.
found=$(docker run --rm --entrypoint sh "$IMAGE" -c "
  find / -xdev -type f \\( \
      -name '*.onnx' -o -name '*.pt' -o -name '*.pth' \
      -o -name '*.tflite' -o -name '*.engine' -o -name '*.safetensors' \\) \
    -printf '%s\t%p\n' 2>/dev/null \
  | awk -F'\t' '\$1 > $SIZE_LIMIT_BYTES || \$2 !~ /site-packages/'
")

if [ -n "$found" ]; then
  echo "FAIL: model weights are baked into ${IMAGE}."
  echo "The model must be fetched at runtime as a separately-licensed artefact (ADR-0008)."
  echo
  printf '%s\n' "$found"
  exit 1
fi

echo "PASS: ${IMAGE} carries no model weights."
