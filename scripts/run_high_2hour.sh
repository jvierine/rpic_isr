#!/usr/bin/env bash
# High-resolution two-Bragg-wavelength O+/electron run. Preserve all older data.
# Run from the source checkout with the CUDA runtime libraries on LD_LIBRARY_PATH.
set -euo pipefail
cd "$(dirname "$0")/.."
root="${RPIC_RUN_ROOT:-/scratch/data/juha/rpic_isr/results}"
mkdir -p "$root"
run="$(mktemp -d "$root/high512-2hour-XXXXXX")"
export RPIC_WEB_CAPTURE="$run/capture.h5" RPIC_WEB_FRAMES=4096 RPIC_WEB_CROP=1
export RPIC_TILE_INTERVAL=0 RPIC_PUSH_VARIANT=packed
unset RPIC_PROFILE || true
printf 'Run directory: %s\n' "$run"
set +e
# Graceful deadline at 115 min; external cap includes compilation/startup/output.
# SIGKILL fallback is still inside the user's two-hour maximum.
timeout --signal=TERM --kill-after=5s 7190s target/release/rpic_dual \
  --devices 0 --thermostat after-equilibrium \
  --cells 512 --ppc 8 --bragg-mode 2 --density 1e11 \
  --te 2000 --ti 1000 --ion-mass-amu 16 --radar-mhz 440.2 --dt .06 \
  --burn-ms 1 --duration-ms 3 --sample-us .02 --window 32768 \
  --equilibrium-window-ms .25 --equilibrium-tolerance .005 \
  --thermostat-tau-ms 2 --thermostat-settle-ms .5 \
  --max-energy-drift-percent 1 --max-wall-seconds 6900 \
  --diagnostic-stride 10000 --seed 9300 \
  --live-diagnostics "$run/live.h5" --output "$run/spectrum.h5" \
  > "$run/run.log" 2>&1
status=$?
printf 'Exit status: %s\n' "$status"
printf 'Run directory: %s\n' "$run"
exit "$status"
