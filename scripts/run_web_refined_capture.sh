#!/usr/bin/env bash
# Full-length spatial-refinement capture; preserve old captures and public runs.
set -euo pipefail
cd "$(dirname "$0")/.."
cells=${1:?mesh cells required};device=${2:?GPU device required}
case "$cells" in 256|512) ;; *) echo 'Use 256 or 512 cells' >&2;exit 1;; esac
export LD_LIBRARY_PATH=/home/juha/miniconda3/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib:/home/juha/miniconda3/lib/python3.9/site-packages/nvidia/cufft/lib
mkdir -p results/web_resolution
capture="results/web_resolution/capture_${cells}.h5"
output="results/web_resolution/spectrum_${cells}.h5"
test ! -e "$capture";test ! -e "$capture.tmp";test ! -e "$output"
nohup env RPIC_WEB_CAPTURE="$capture" target/release/rpic_isr_2d \
 --backend cuda --device "$device" --cells "$cells" --ppc 8 --bragg-mode 2 \
 --density 1e11 --dt .06 --burn-ms 10 --duration-ms 4 --sample-us 1 \
 --window 1024 --seed 8300 --diagnostic-stride 10000 --output "$output" \
 > "results/web_resolution/capture_${cells}.log" 2>&1 < /dev/null &
echo "Started refined ${cells}x${cells} capture on GPU $device: PID $!"
