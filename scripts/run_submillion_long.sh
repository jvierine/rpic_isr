#!/usr/bin/env bash
# 917504 total macroparticles; one Bragg period; full 16 ms signal.
# The smaller periodic box retains the ISR k but changes finite-box modes.
set -euo pipefail
cd "$(dirname "$0")/.."
root="${RPIC_RUN_ROOT:-/scratch/data/juha/rpic_isr/results}"
mkdir -p "$root"
run="$(mktemp -d "$root/submillion-long-XXXXXX")"
export RPIC_WEB_CAPTURE="$run/capture.h5" RPIC_WEB_FRAMES=4096 RPIC_WEB_CROP=1
# Dense enough global replay pool for >1000 actual tracks inside the camera.
export RPIC_WEB_POOL=16384 RPIC_TILE_INTERVAL=0 RPIC_PUSH_VARIANT=packed
unset RPIC_PROFILE || true
printf 'Run directory: %s\n' "$run"
set +e
timeout --signal=TERM --kill-after=5s 7190s target/release/rpic_dual \
 --devices 0 --thermostat after-equilibrium --cells 256 --ppc 7 --bragg-mode 1 \
 --density 1e11 --te 2000 --ti 1000 --ion-mass-amu 16 --radar-mhz 440.2 --dt .06 \
 --burn-ms 1 --duration-ms 16 --sample-us .02 --window 32768 \
 --equilibrium-window-ms .25 --equilibrium-tolerance .005 \
 --thermostat-tau-ms 2 --thermostat-settle-ms .5 \
 --max-energy-drift-percent 1 --max-wall-seconds 6900 \
 --diagnostic-stride 10000 --seed 9500 \
 --live-diagnostics "$run/live.h5" --output "$run/spectrum.h5" \
 > "$run/run.log" 2>&1
status=$?
printf 'Exit status: %s\nRun directory: %s\n' "$status" "$run"
exit "$status"
