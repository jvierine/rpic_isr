#!/usr/bin/env bash
# Matched initial particles, three plasma ages. Compare to seeds 200..203
# in the original 1 ms burn-in ensemble. Do not add these repeated seeds to it.
set -euo pipefail
cd "$(dirname "$0")/.."
device="${1:?GPU device required}"
burn="${2:?Burn-in in milliseconds required}"
out="${3:-results/burn_study}"
mkdir -p "$out"
for seed in 200 201 202 203; do
    output="$out/burn${burn}_seed${seed}.h5"
    if test -s "$output"; then continue; fi
    target/release/rpic_isr --backend cuda --device "$device" --cells 64 \
        --bragg-mode 2 --ppc 512 --duration-ms 8 --burn-ms "$burn" \
        --window 512 --seed "$seed" --throttle-ms 20 --output "$output" \
        > "$out/burn${burn}_seed${seed}.log" 2>&1
done
