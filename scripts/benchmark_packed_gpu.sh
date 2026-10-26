#!/usr/bin/env bash
# Matched, sequential GPU trials. Each campaign gets a fresh directory; no CSV.
set -euo pipefail
cd "$(dirname "$0")/.."
cells="${1:-512}"
duration="${2:-.01}"
root="${RPIC_BENCH_ROOT:-/scratch/data/juha/rpic_isr/results}"
mkdir -p "$root"
campaign="$(mktemp -d "$root/packed-${cells}-XXXXXX")"
for variant in single-direct:0:0:packed single-tile1:0:1:packed single-tile4:0:4:packed single-tile16:0:16:packed single-split:0:0:split single-scaled:0:0:scaled dual-direct:0,1:0:packed; do
  IFS=: read -r name devices interval push <<< "$variant"
  RPIC_PROFILE=1 RPIC_TILE_INTERVAL="$interval" RPIC_PUSH_VARIANT="$push" target/release/rpic_dual \
    --devices "$devices" --thermostat off --cells "$cells" --ppc 8 \
    --bragg-mode 2 --density 1e11 --te 2000 --ti 1000 --dt .06 \
    --burn-ms 0 --duration-ms "$duration" --sample-us .0033632544 \
    --window 16 --diagnostic-stride 1000 --seed 9100 \
    --output "$campaign/$name.h5" 2>&1 | tee "$campaign/$name.log"
done
printf 'Benchmark directory: %s\n' "$campaign"
