#!/usr/bin/env bash
#
# Turn a screen recording of the demo into the launch GIF (P5.5).
#
# The M3 exit criterion puts hard numbers on this asset: under 5 MB, at most 60 seconds,
# autoplaying inline on GitHub and on the landing page. A GIF that misses either is a
# dead asset in an HN or Reddit skim, and "it looked about right" is not a check. So the
# numbers are asserted here rather than eyeballed, and the script fails loudly instead of
# writing a file that would be found wanting after it shipped.
#
# Capture is deliberately not automated. Screen recording on macOS needs a TCC grant that
# a script cannot ask for, and the framing — terminal, one command, then the dashboard —
# is a judgement call that changes every time the dashboard does. Record with Cmd-Shift-5
# (or any recorder), then hand the file to this script.
#
# Usage:  scripts/record-demo-gif.sh <recording.mov> [output.gif]
#
# What to record, in one take:
#   1. a terminal with the one docker command, typed and run
#   2. the dashboard coming up at http://localhost:8080
#   3. counts moving — footfall, line-crossings, occupancy, dwell, the zone heatmap
#
# Bring the stack up with `make gif` first; that seeds docker/gif.yaml, which is the
# site this GIF is defined against.
#
# HOW LONG THE SHOT CAN ACTUALLY BE. "Under 60 seconds" is the ceiling in the plan, not a
# target the format can reach: 20 s of full-frame camera footage measured 5.23 MB here at
# 720px/10fps, already over budget on its own. GIF pays per changed pixel, so what fits is
# governed by how much of the frame moves, not by the clock:
#
#   dashboard filling the frame, camera as an inset   ~40-60 s fits
#   split roughly half and half                       ~30 s fits
#   full-frame camera footage                         ~15-20 s fits
#
# Shoot the dashboard, not the video. The HUD's flat fills cost almost nothing per frame
# and they are what carries the value prop anyway; the camera pane is evidence, not the
# subject. If the ladder below bottoms out, the shot is too busy — that is the diagnosis.

set -euo pipefail

readonly MAX_BYTES=$((5 * 1024 * 1024))
readonly MAX_SECONDS=60

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
readonly SRC="${1:-}"
readonly OUT="${2:-${REPO_ROOT}/docs/assets/demo.gif}"

# Width, fps and palette size, cheapest-looking first. A GIF's size is dominated by how
# many distinct frames it holds and how many colours each needs, so the ladder trades
# resolution before it trades smoothness: a 12 fps dashboard still reads as live, while a
# 640px one stops showing what the numbers say.
readonly LADDER=(
  "960 15 256"
  "960 12 256"
  "800 12 192"
  "800 10 128"
  "720 10 128"
  "640 10 96"
)

die() { echo "error: $*" >&2; exit 1; }

[[ -n "${SRC}" ]] || die "usage: scripts/record-demo-gif.sh <recording.mov> [output.gif]"
[[ -f "${SRC}" ]] || die "no such recording: ${SRC}"
command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not found on PATH"
command -v ffprobe >/dev/null 2>&1 || die "ffprobe not found on PATH"

duration="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${SRC}")"
# Whole seconds, rounded up: 60.4 s is over the line, and bash cannot compare floats.
# awk rather than bc — bash is not shipped anywhere that awk is missing, and bc is.
duration_s="$(awk -v d="${duration}" 'BEGIN { printf "%d", (d == int(d) ? d : int(d) + 1) }')"

if (( duration_s > MAX_SECONDS )); then
  die "recording is ${duration_s}s, over the ${MAX_SECONDS}s limit.
       Re-record shorter, or trim it first:
         ffmpeg -i '${SRC}' -t ${MAX_SECONDS} -c copy trimmed.mov"
fi

mkdir -p "$(dirname "${OUT}")"
workdir="$(mktemp -d -t mesopic-gif)"
trap 'rm -rf "${workdir}"' EXIT
palette="${workdir}/palette.png"

echo "source: ${duration_s}s, budget ${MAX_BYTES} bytes"

for rung in "${LADDER[@]}"; do
  read -r width fps colors <<<"${rung}"

  # Two passes, because a GIF gets one 256-colour table for the whole animation and
  # ffmpeg's default table is generic. Generating it from these actual frames is the
  # difference between a legible dashboard and a dithered mess.
  ffmpeg -v error -y -i "${SRC}" \
    -vf "fps=${fps},scale=${width}:-1:flags=lanczos,palettegen=max_colors=${colors}:stats_mode=diff" \
    "${palette}"

  ffmpeg -v error -y -i "${SRC}" -i "${palette}" \
    -lavfi "fps=${fps},scale=${width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
    -loop 0 "${OUT}"

  bytes="$(wc -c <"${OUT}" | tr -d ' ')"
  printf '  %spx %sfps %s colours -> %s bytes\n' "${width}" "${fps}" "${colors}" "${bytes}"

  if (( bytes <= MAX_BYTES )); then
    printf '\nwrote %s\n' "${OUT}"
    printf '  %s bytes (%s%% of the 5 MB budget), %ss, %spx @ %sfps\n' \
      "${bytes}" "$(( bytes * 100 / MAX_BYTES ))" "${duration_s}" "${width}" "${fps}"
    exit 0
  fi
done

rm -f "${OUT}"
die "every rung of the ladder came out over ${MAX_BYTES} bytes.
       The recording is too long or too busy for a GIF this size. Shorten it, hold the
       camera view still for longer, or cut the number of on-screen transitions."
