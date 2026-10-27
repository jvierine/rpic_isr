#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for run in low high; do
 input="results/youtube/fullscreen-$run.h264"
 output="results/youtube/picisr-$run-preview.mp4"
 test -s "$input"; test ! -e "$output"
 ffmpeg -hide_banner -loglevel warning -fflags +genpts -r 60 -i "$input" \
   -c:v copy -movflags +faststart -metadata title="Collective Thomson scattering: $run density" "$output"
 ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,avg_frame_rate,duration -of json "$output"
done
