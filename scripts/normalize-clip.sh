#!/usr/bin/env bash
#
# Turn a camera-original recording into a ground-truth clip fixture.
#
# A phone or camera original is not a fixture. It carries the wrong resolution, the wrong
# frame rate, an audio track, and — on any recent iPhone — HLG/BT.2020 high dynamic range
# that no RTSP camera ever hands the engine. It also carries the GPS coordinates of the
# place it was shot, which for own-rig footage is somebody's home.
#
# This script produces the byte-exact artefact a manifest hashes: 1080p25 SDR BT.709,
# no audio, no metadata. Run it once, hash the output, and never re-encode — a clip that
# does not match its manifest's SHA-256 is refused by `resolve_clip`, which is the point.
#
# Usage:  scripts/normalize-clip.sh <input> <clip_id> <output_dir> [duration_s]
# Example: scripts/normalize-clip.sh ~/Downloads/IMG_5286.MOV home-hallway-oblique-01 ~/clips
#
# A fourth argument keeps only the first <duration_s> seconds. A recording runs past the
# end of the session somebody labelled, and the tail is footage no human watched: scored
# as-is it charges the engine for crossings the truth file was never going to contain.
# `check_pairing` refuses a truth file whose duration disagrees with its manifest by more
# than a millisecond, so the cut has to happen here, once, before the hash. Omit the
# argument and the whole input is encoded, byte-for-byte as before.

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <input> <clip_id> <output_dir> [duration_s]" >&2
  exit 2
fi

IN="$1"
CLIP_ID="$2"
OUT_DIR="$3"
DURATION="${4:-}"
OUT="${OUT_DIR}/${CLIP_ID}.mp4"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "error: ffmpeg not found on PATH." >&2
  exit 1
fi
if [[ ! -f "${IN}" ]]; then
  echo "error: no such input: ${IN}" >&2
  exit 1
fi
if [[ ! "${CLIP_ID}" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
  # Same pattern the manifest schema enforces. Failing here beats failing after a
  # ten-minute encode.
  echo "error: clip_id must be a lowercase slug: ${CLIP_ID}" >&2
  exit 1
fi
if [[ -n "${DURATION}" && ! "${DURATION}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "error: duration_s must be a positive number of seconds: ${DURATION}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

# `-t` goes on the *output*, so it counts frames the filter chain has already decimated
# rather than frames of the source. At 25 fps that makes the length exact — 7200 s is
# 180000 frames and the container says 7200.0 — where trimming the input would land a
# frame either side of it and fail the manifest's millisecond tolerance.
TRIM=()
if [[ -n "${DURATION}" ]]; then
  echo "keeping the first ${DURATION}s of the input"
  TRIM=(-t "${DURATION}")
fi

# --- HLG -> SDR ---------------------------------------------------------------------
#
# Done by hand because this ffmpeg has neither zimg (`zscale`) nor libplacebo, so the
# usual `zscale=t=linear,tonemap=...` chain is unavailable and the `colorspace` filter
# does not accept arib-std-b67 as an input transfer. The three steps below are that
# chain, written out:
#
#   1. Inverse HLG OETF (BT.2100), taking the signal back to scene-linear light.
#   2. BT.2020 -> BT.709 primaries, as a 3x3 on *linear* RGB, which is the only place
#      a primaries conversion is correct.
#   3. BT.709 OETF, back to a displayable SDR signal.
#
# EXPOSURE is where scene-linear is scaled so that something sensible lands on SDR white.
# 1.72 maps a 90% HLG signal to 1.0. It was chosen by measuring the source rather than
# assuming: these clips put their white walls around 87% and their peak just over 100%,
# so the textbook "diffuse white sits at 75%" would blow the walls — and the doorway
# edge with them — to flat white. Re-measure before reusing this on a different scene:
#
#   ffprobe -f lavfi -i "movie=IN[out];[out]fps=1/20,signalstats" \
#           -show_entries frame_tags=lavfi.signalstats.YHIGH -of csv=p=0
#
EXPOSURE="1.72"

readonly HLG_INVERSE_OETF="if(lte(val/maxval,0.5), pow(val/maxval,2)/3, (exp((val/maxval-0.55991073)/0.17883277)+0.28466892)/12)"
readonly TO_LINEAR="clip((${HLG_INVERSE_OETF})*${EXPOSURE},0,1)*maxval"
readonly BT2020_TO_BT709="rr=1.6605:rg=-0.5876:rb=-0.0728:gr=-0.1246:gg=1.1329:gb=-0.0083:br=-0.0182:bg=-0.1006:bb=1.1187"
readonly BT709_OETF="clip(if(lt(val/maxval,0.018), 4.5*val/maxval, 1.099*pow(val/maxval,0.45)-0.099),0,1)*maxval"

TRANSFER="$(ffprobe -v error -select_streams v:0 \
  -show_entries stream=color_transfer -of default=nw=1:nk=1 "${IN}")"

# Decimate and downscale first: both are cheap, and doing them before the per-pixel
# colour work cuts it by roughly 4x in frames and 4x in pixels. Scaling happens in a
# nonlinear space either way, which is what every video pipeline does.
CHAIN="fps=25,scale=1920:1080:flags=lanczos"
if [[ "${TRANSFER}" == "arib-std-b67" ]]; then
  echo "source is HLG (${TRANSFER}); tone-mapping to SDR BT.709 at exposure ${EXPOSURE}"
  CHAIN="${CHAIN},format=gbrp16le"
  CHAIN="${CHAIN},lut=r='${TO_LINEAR}':g='${TO_LINEAR}':b='${TO_LINEAR}'"
  CHAIN="${CHAIN},colorchannelmixer=${BT2020_TO_BT709}"
  CHAIN="${CHAIN},lut=r='${BT709_OETF}':g='${BT709_OETF}':b='${BT709_OETF}'"
else
  echo "source transfer is ${TRANSFER:-unspecified}; no tone-map applied"
fi
CHAIN="${CHAIN},format=yuv420p"
# `lut` and `colorchannelmixer` rewrite pixels and leave frame properties alone, so
# without this the output is BT.709 data still tagged HLG/BT.2020 — and a decoder that
# honours the tag applies the HLG curve a second time. The `-color_*` output options are
# not enough on their own: frame properties win over the encoder context.
CHAIN="${CHAIN},setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"

# -map 0:v            video only. The engine never decodes audio, and a recording of
#                     someone's home has a conversation on it.
# -map_metadata -1    drops the container tags, which on an iPhone original include
#   -map_chapters -1  com.apple.quicktime.location.ISO6709 — the GPS fix of the room.
# -bitexact           keeps the encoder from stamping its version into the file, so the
#                     same input and the same ffmpeg produce the same bytes and the
#                     manifest SHA-256 is reproducible.
# -g 50               keyframe every 2s at 25fps, matching the rest of the rig.
ffmpeg -hide_banner -loglevel error -y \
  -i "${IN}" \
  -map 0:v:0 -map_metadata -1 -map_chapters -1 -an -sn -dn \
  ${TRIM[@]+"${TRIM[@]}"} \
  -vf "${CHAIN}" \
  -c:v libx264 -preset slow -crf 20 -g 50 -pix_fmt yuv420p \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
  -bitexact -fflags +bitexact -flags +bitexact \
  -movflags +faststart \
  "${OUT}"

echo "wrote ${OUT}"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,color_transfer \
  -show_entries format=duration -of default=nw=1 "${OUT}"
echo "sha256: $(shasum -a 256 "${OUT}" | cut -d' ' -f1)"
