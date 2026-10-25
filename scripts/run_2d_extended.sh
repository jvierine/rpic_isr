#!/usr/bin/env bash
# Add 48 independent realizations to the original 16: 24 per GPU.
# Invoke with (device, first_seed) = (0,6000) and (1,7000).
set -euo pipefail
cd "$(dirname "$0")/.."
device="${1:?Device number required}"
seed="${2:?First seed required}"
out="${3:-results/2d_extended}"
mkdir -p "$out"
target/release/rpic_isr_2d --backend cuda --device "$device" \
  --cells 128 --bragg-mode 2 --ppc 8 --dt 0.06 \
  --burn-ms 10 --duration-ms 8 --window 512 \
  --realizations 24 --gpu-batch-size 8 --seed "$seed" \
  --diagnostic-stride 2000 --throttle-ms 20 \
  --checkpoint-dir "$out/device${device}" --output "$out/gpu${device}.h5"
