#!/usr/bin/env bash
# Matched physical plasma, 524288 TOTAL particles; preserve every pilot.
set -euo pipefail
cd "$(dirname "$0")/.."
root="${RPIC_RUN_ROOT:-/scratch/data/juha/rpic_isr/results}"
mkdir -p "$root"
trial="$(mktemp -d "$root/half-million-pilots-XXXXXX")"
printf 'Pilot directory: %s\n' "$trial"
unset RPIC_WEB_CAPTURE RPIC_PROFILE RPIC_WEB_POOL || true
export RPIC_TILE_INTERVAL=0 RPIC_PUSH_VARIANT=packed
for spec in 256:4:.06 512:1:.06 256:4:.03 512:1:.03 512:1:.015; do
 IFS=: read -r cells ppc dt <<< "$spec"
 name="n${cells}-ppc${ppc}-dt${dt}"
 timeout --signal=TERM --kill-after=5s 600s target/release/rpic_dual \
  --devices 0 --thermostat off --cells "$cells" --ppc "$ppc" \
  --bragg-mode 2 --density 1e11 --te 2000 --ti 1000 --ion-mass-amu 16 \
  --radar-mhz 440.2 --dt "$dt" --burn-ms 0 --duration-ms .1 \
  --sample-us .02 --window 1024 --diagnostic-stride 1000 --seed 9400 \
  --max-energy-drift-percent 1 --max-wall-seconds 580 \
  --output "$trial/$name.h5" > "$trial/$name.log" 2>&1
 printf 'Finished %s\n' "$name"
done
