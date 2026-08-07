#!/usr/bin/env bash
#
# Generate a synthetic sample clip for the local RTSP rig.
#
# The `/synthetic` path in docker/mediamtx.yml generates its stream live and needs no
# file at all — reach for that first. This script exists for the case where you want the
# *same bytes every run*: perf comparisons across commits, and any test that would
# otherwise be comparing against a stream that is different each time it starts.
#
# Deliberately synthetic. A clip of a real space is personal data, and a clip off the
# internet is a licence question this repo does not need. Real labelled footage for
# accuracy work is MK.2's job and lives outside the repo.
#
# Usage:  scripts/make-sample-clip.sh [seconds]

set -euo pipefail

SECONDS_LONG="${1:-30}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/examples/clips"
OUT="${OUT_DIR}/sample.mp4"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "error: ffmpeg not found on PATH." >&2
  echo "       Use the '/synthetic' RTSP path instead — it needs no local ffmpeg:" >&2
  echo "         make test-stream" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

# 1080p25 is the shape the M0 perf gate is specified against, so the dev clip matches it.
# -g 50 (keyframe every 2s) keeps seek and loop restarts cheap. No audio: the engine
# never decodes it, and carrying it would only slow the decode path under test.
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "testsrc2=size=1920x1080:rate=25:duration=${SECONDS_LONG}" \
  -c:v libx264 -preset veryfast -pix_fmt yuv420p -g 50 -an \
  "${OUT}"

echo "wrote ${OUT} (${SECONDS_LONG}s, 1080p25)"
echo "serve it with:  make test-stream   ->  rtsp://127.0.0.1:8554/sample"
