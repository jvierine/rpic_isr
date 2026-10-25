#!/usr/bin/env bash
# Matched seeds from production, but a 20 ms rather than 10 ms burn-in.
# WAIT_FOR_PID optionally defers this single job until a GPU worker exits.
set -euo pipefail
cd "$(dirname "$0")/.."
if test -n "${WAIT_FOR_PID:-}"; then
    while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do sleep 10; done
fi
target/release/rpic_isr --backend cuda --device 0 --cells 128 --bragg-mode 2 \
    --ppc 256 --dt 0.06 --duration-ms 8 --burn-ms 20 --window 512 \
    --realizations 8 --gpu-batch-size 8 --seed 1000 --throttle-ms 20 \
    --checkpoint-dir results/burn20/device0 --output results/burn20.h5
