#!/usr/bin/env bash
set -euo pipefail
binary=${1:?rpic_isr_2d}; checker=${2:?check_2d_backend}; out=${3:?New directory}
mkdir "$out"
for backend in cpu cuda; do
 "$binary" --backend "$backend" --cells 64 --ppc 4 --bragg-mode 1 \
 --density 111639834702.32187 --bz-t .00003572386758 --dt .06 \
 --burn-ms 0 --duration-ms .001 --sample-us .005 --window 64 \
 --diagnostic-stride 10 --seed 9300 --output "$out/$backend.h5"
done
"$checker" "$out/cpu.h5" "$out/cuda.h5"
