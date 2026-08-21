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
# target the format can reach. GIF pays per *changed* pixel, so what fits is governed by
# how much of the frame moves, not by the clock. Measured here:
#
#   dashboard only, tightly cropped     ~30-40 s fits
#   dashboard with a camera inset       ~20-30 s fits
#   full-frame camera footage           ~15-20 s fits   (19.8 s = 5.23 MB at 720px/10fps)
#
# Shoot the dashboard, not the video. Three things blow the budget far faster than length,
# and all three are framing mistakes rather than encoding ones:
#
#   1. DESKTOP WALLPAPER IN THE SHOT. A gradient needs most of the colour table and
#      dithers into noise that will not compress. Record a tight region, or pass --crop.
#   2. RECORDING SMALLER THAN THE LADDER'S WIDTH. Upscaling invents pixels and
#      interpolation noise; a 784px-wide capture came out *larger* at 960px than at 660px.
#      The ladder now clamps to the source width, but record big enough to begin with.
#   3. A BUSY CURSOR. Every frame it moves in is a frame that cannot be differenced.
#
# Usage:  scripts/record-demo-gif.sh [--crop W:H:X:Y] <recording.mov> [output.gif]
#
# --crop takes ffmpeg's geometry and is the fix for wallpaper or window chrome you did not
# mean to capture, without reshooting. To find the numbers, export a frame and read the
# panel's bounds off it:
#
#   ffmpeg -ss 5 -i recording.mov -frames:v 1 frame.png

set -euo pipefail

readonly MAX_BYTES=$((5 * 1024 * 1024))
readonly MAX_SECONDS=60

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT

CROP=""
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --crop) CROP="${2:-}"; shift 2 ;;
    *) echo "error: unknown option $1" >&2; exit 1 ;;
  esac
done
readonly CROP
readonly SRC="${1:-}"
readonly OUT="${2:-${REPO_ROOT}/docs/assets/demo.gif}"

# Width, fps and palette size, most-generous first. The ladder trades resolution before it
# trades smoothness: a 10 fps dashboard still reads as live, while a 400px one stops
# showing what the numbers say — and the numbers are the entire point of the asset.
readonly LADDER=(
  "960 15 256"
  "960 12 256"
  "800 12 192"
  "800 10 128"
  "720 10 128"
  "660 10 128"
  "560 10 128"
  "480 10 96"
  "480 8 96"
  "400 8 64"
)

# No dithering. Dither trades file size for smooth gradients, and this asset is a flat dark
# HUD where there are no gradients worth paying for — it measured ~30% smaller with it off,
# on identical footage.
readonly DITHER="none"

# A light temporal denoise before quantizing. The source is h264, so its "unchanged"
# regions are not actually unchanged: codec noise varies every pixel every frame and
# defeats the frame-differencing GIF relies on. Worth ~13% and costs nothing visible.
readonly DENOISE="hqdn3d=6:6:12:12"

die() { echo "error: $*" >&2; exit 1; }

[[ -n "${SRC}" ]] || die "usage: scripts/record-demo-gif.sh <recording.mov> [output.gif]"
[[ -f "${SRC}" ]] || die "no such recording: ${SRC}"
command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not found on PATH"
command -v ffprobe >/dev/null 2>&1 || die "ffprobe not found on PATH"

src_width="$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "${SRC}" | head -1)"
if [[ -n "${CROP}" ]]; then
  # ffmpeg geometry is W:H:X:Y — the crop's width is what the scaler actually receives.
  src_width="${CROP%%:*}"
fi
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

echo "source: ${duration_s}s, ${src_width}px wide${CROP:+ (cropped ${CROP})}, budget ${MAX_BYTES} bytes"

# The filter chain shared by both passes. Crop first so the denoise and the palette only
# ever see pixels that survive into the GIF.
chain_prefix="${CROP:+crop=${CROP},}${DENOISE},"

tried=""
for rung in "${LADDER[@]}"; do
  read -r width fps colors <<<"${rung}"

  # Never upscale. A rung wider than the source spends bytes on interpolated pixels that
  # carry no information, and reliably comes out LARGER than the rung below it.
  if (( width > src_width )); then
    width="${src_width}"
  fi
  # Clamping collapses the top rungs onto one width; skip the duplicates it creates.
  if [[ " ${tried} " == *" ${width}x${fps}x${colors} "* ]]; then
    continue
  fi
  tried="${tried} ${width}x${fps}x${colors}"

  # Two passes, because a GIF gets one 256-colour table for the whole animation and
  # ffmpeg's default table is generic. Generating it from these actual frames is the
  # difference between a legible dashboard and a dithered mess.
  ffmpeg -v error -y -i "${SRC}" \
    -vf "${chain_prefix}fps=${fps},scale=${width}:-1:flags=lanczos,palettegen=max_colors=${colors}:stats_mode=diff" \
    "${palette}"

  ffmpeg -v error -y -i "${SRC}" -i "${palette}" \
    -lavfi "${chain_prefix}fps=${fps},scale=${width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=${DITHER}:diff_mode=rectangle" \
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

       Check the framing before you shorten the take — it is usually the cause:
         * Is desktop wallpaper or window chrome in the shot? Crop it out:
             ffmpeg -ss 5 -i '${SRC}' -frames:v 1 frame.png
             scripts/record-demo-gif.sh --crop W:H:X:Y '${SRC}'
         * Is a camera pane filling most of the frame? Make it an inset.
         * Was the cursor moving throughout? Park it.

       If the framing is already tight, the take is genuinely too long."
