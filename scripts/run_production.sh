#!/usr/bin/env bash
# 64 independent realizations, 32 on each device (invoke once per device).
# Same physical particles per species as the pilot, twice finer mesh and dt.
set -euo pipefail
cd "$(dirname "$0")/.."
device="${1:?Device number required}"
seed="${2:?First seed required}"
out="${3:-results/production}"
mkdir -p "$out"
target/release/rpic_isr --backend cuda --device "$device" --cells 128 \
    --bragg-mode 2 --ppc 256 --dt 0.06 --duration-ms 8 --burn-ms 10 \
    --window 512 --realizations 32 --gpu-batch-size 8 --seed "$seed" \
    --throttle-ms 20 --checkpoint-dir "$out/device${device}" \
    --output "$out/gpu${device}.h5"
