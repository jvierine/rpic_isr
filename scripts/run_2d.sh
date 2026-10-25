#!/usr/bin/env bash
# Independent 2D2V ensembles; invoke once on each GPU with disjoint seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
device="${1:?Device number required}"
seed="${2:?First seed required}"
out="${3:-results/2d_production}"
mkdir -p "$out"
target/release/rpic_isr_2d --backend cuda --device "$device" \
  --cells 128 --bragg-mode 2 --ppc 8 --dt 0.06 \
  --burn-ms 10 --duration-ms 8 --window 512 \
  --realizations 8 --gpu-batch-size 8 --seed "$seed" \
  --diagnostic-stride 2000 --throttle-ms 20 \
  --checkpoint-dir "$out/device${device}" --output "$out/gpu${device}.h5"
