#!/usr/bin/env bash
set -euo pipefail
binary=${1:?binary}; out=${2:?new directory}; mkdir "$out"
for backend in cpu cuda; do
 "$binary" --backend "$backend" --device 0 --cells 64 --ppc 4 --bragg-mode 1 \
 --density 111639834702.32187 --te 2000 --ti 1000 \
 --bx-t .000001869630096 --by-t .00003567491076 --hot-fraction .1 \
 --ring-energy-ev 10 --ring-width .1 --ring-turn-on-us .2 \
 --dt .01 --burn-ms 0 --duration-ms .002 --sample-us .005 --window 64 \
 --diagnostic-stride 100 --seed 9500 --output "$out/$backend.h5"
done
python scripts/check_oblique.py "$out/cpu.h5" "$out/cuda.h5"
