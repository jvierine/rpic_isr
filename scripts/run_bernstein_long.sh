#!/usr/bin/env bash
# Longer unthermostatted perpendicular-B run, using the validated Boris binary.
set -euo pipefail
binary=${1:?Path to validated rpic_isr_2d binary}
out=${2:?New output directory}
mkdir "$out"
export RPIC_WEB_CAPTURE="$out/capture.h5" RPIC_WEB_FRAMES=4096 RPIC_WEB_CROP=1 RPIC_WEB_POOL=16384
exec timeout --signal=TERM --kill-after=5s 7190s "$binary" \
 --backend cuda --device 0 --cells 256 --ppc 7 --bragg-mode 1 \
 --density 111639834702.32187 --te 2000 --ti 1000 --bz-t 0.00003572386758 \
 --dt .06 --burn-ms .5 --duration-ms 2 --sample-us .01 --window 16384 \
 --diagnostic-stride 1000 --seed 9302 --output "$out/spectrum.h5"
