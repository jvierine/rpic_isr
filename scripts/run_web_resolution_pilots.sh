#!/usr/bin/env bash
# Matched physical parameters; refinement changes only mesh and macroparticle count.
set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/home/juha/miniconda3/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib:/home/juha/miniconda3/lib/python3.9/site-packages/nvidia/cufft/lib
mkdir -p results/web_resolution
for device in 0 1; do
  cells=$((256 * (device + 1)))
  output="results/web_resolution/pilot_${cells}.h5"
  test ! -e "$output"
  nohup target/release/rpic_isr_2d --backend cuda --device "$device" \
    --cells "$cells" --ppc 8 --bragg-mode 2 --density 1e11 --dt .06 \
    --burn-ms .05 --duration-ms .05 --sample-us .5 --window 16 --seed 8300 \
    --diagnostic-stride 1000 --output "$output" \
    > "results/web_resolution/pilot_${cells}.log" 2>&1 < /dev/null &
  echo "Started ${cells}x${cells} pilot on GPU $device: PID $!"
done
