#!/usr/bin/env bash
set -euo pipefail
binary=${1:?binary}; out=${2:?new directory}; mkdir "$out"
for device in 0 1; do
 (
  dt=.02; stride=1000; if [ "$device" = 1 ]; then dt=.01; stride=2000; fi
  export RPIC_WEB_CAPTURE="$out/capture$device.h5" RPIC_WEB_FRAMES=128 RPIC_WEB_CROP=1 RPIC_WEB_POOL=16384 RPIC_LIVE_PATH="$out/live$device.h5"
  timeout --kill-after=5s 300s "$binary" --backend cuda --device "$device" --cells 256 --ppc 7 --bragg-mode 1 \
   --density 111639834702.32187 --te 2000 --ti 1000 --bx-t .000001869630096 --by-t .00003567491076 \
   --hot-fraction .01 --hot-energy-ev 20 --dt "$dt" --burn-ms 0 --duration-ms .03 \
   --sample-us .01 --window 256 --diagnostic-stride "$stride" --seed 9401 --output "$out/spectrum$device.h5" >"$out/run$device.log" 2>&1
 ) &
done
wait
python scripts/check_oblique.py "$out/spectrum0.h5"
python scripts/check_oblique.py "$out/spectrum1.h5"
