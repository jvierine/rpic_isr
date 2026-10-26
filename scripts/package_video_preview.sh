#!/usr/bin/env bash
# Mux the deterministic WebCodecs Annex-B stream; do not invent/interpolate frames.
set -euo pipefail
input="${1:?Usage: package_video_preview.sh INPUT.h264 OUTPUT.mp4}"
output="${2:?Usage: package_video_preview.sh INPUT.h264 OUTPUT.mp4}"
test ! -e "$output"
ffmpeg -hide_banner -fflags +genpts -r 30 -i "$input" -c:v copy \
  -movflags +faststart \
  -metadata title='Collective Thomson Scattering: Electrons, Ions and ISR' \
  -metadata comment='Actual PIC recordings rendered with Rust/Wasm/Rustplotlib. Existing low-density and high-density diagnostic recordings; see accompanying description for limitations. Source: https://github.com/jvierine/rpic_isr' \
  "$output"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,avg_frame_rate,duration \
  -of json "$output"
