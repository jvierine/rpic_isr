#!/usr/bin/env bash
# Run independent ensembles, not a distributed tiny FFT.
# Set LD_LIBRARY_PATH for the installed CUDA 12 NVRTC/cuFFT libraries first.
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-results/dual_gpu}"
mkdir -p "$out"
cargo build --release --features cuda --bin rpic_isr
target/release/rpic_isr --backend cuda --device 0 --cells 64 --bragg-mode 2 \
    --ppc 512 --duration-ms 8 --burn-ms 1 --window 512 --realizations 4 \
    --seed 200 --throttle-ms 20 --output "$out/gpu0.h5" > "$out/gpu0.log" 2>&1 &
p0=$!
target/release/rpic_isr --backend cuda --device 1 --cells 64 --bragg-mode 2 \
    --ppc 512 --duration-ms 8 --burn-ms 1 --window 512 --realizations 4 \
    --seed 300 --throttle-ms 20 --output "$out/gpu1.h5" > "$out/gpu1.log" 2>&1 &
p1=$!
status=0
wait "$p0" || status=$?
wait "$p1" || status=$?
exit "$status"
