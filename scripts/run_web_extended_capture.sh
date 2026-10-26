#!/usr/bin/env bash
# Fourfold movie length and voltage record; old captures are preserved.
set -euo pipefail
cd "$(dirname "$0")/.."
kind=${1:?low or high required};device=${2:?GPU device required}
case "$kind" in low) cells=128;density=1e10;seed=8200;; high) cells=512;density=1e11;seed=8300;; *) exit 1;; esac
export LD_LIBRARY_PATH=/home/juha/miniconda3/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib:/home/juha/miniconda3/lib/python3.9/site-packages/nvidia/cufft/lib
mkdir -p results/web_extended
capture="results/web_extended/${kind}_capture.h5"
output="results/web_extended/${kind}_spectrum.h5"
test ! -e "$capture";test ! -e "$capture.tmp";test ! -e "$output"
nohup env RPIC_WEB_CAPTURE="$capture" RPIC_WEB_FRAMES=4096 RPIC_WEB_CROP=1 target/release/rpic_isr_2d \
 --backend cuda --device "$device" --cells "$cells" --ppc 8 --bragg-mode 2 \
 --density "$density" --dt .06 --burn-ms 10 --duration-ms 16 --sample-us 1 \
 --window 1024 --seed "$seed" --diagnostic-stride 10000 --output "$output" \
 > "results/web_extended/${kind}.log" 2>&1 < /dev/null &
echo "Started $kind extended capture on GPU $device: PID $!"
