#!/usr/bin/env bash
# Matched one-GPU runs, simultaneous on separate devices; no thermostat in record.
set -euo pipefail
binary=${1:?Validated binary}; out=${2:?New directory}
mkdir "$out"
for name in tail control; do
 mkdir "$out/$name"
done
run() {
 local name=$1 device=$2 fraction=$3
 export RPIC_WEB_CAPTURE="$out/$name/capture.h5" RPIC_WEB_FRAMES=4096 RPIC_WEB_CROP=1 RPIC_WEB_POOL=16384
 export RPIC_LIVE_PATH="$out/$name/live.h5"
 timeout --signal=TERM --kill-after=5s 7190s "$binary" \
 --backend cuda --device "$device" --cells 256 --ppc 7 --bragg-mode 1 \
 --density 111639834702.32187 --te 2000 --ti 1000 \
 --bx-t 0.000001869630096 --by-t 0.00003567491076 \
 --hot-fraction "$fraction" --hot-energy-ev 20 --core-thermostat-us 0 \
 --dt .01 --burn-ms .5 --duration-ms 2 --sample-us .01 --window 16384 \
 --diagnostic-stride 10000 --seed 9401 --output "$out/$name/spectrum.h5"
}
run tail 0 .01 >"$out/tail/run.log" 2>&1 & tail_pid=$!
run control 1 0 >"$out/control/run.log" 2>&1 & control_pid=$!
printf '%s\n' "$tail_pid" >"$out/tail/launcher.pid"
printf '%s\n' "$control_pid" >"$out/control/launcher.pid"
set +e
wait "$tail_pid"; tail_exit=$?; printf '%s\n' "$tail_exit" >"$out/tail/exit_code"
wait "$control_pid"; control_exit=$?; printf '%s\n' "$control_exit" >"$out/control/exit_code"
exit $((tail_exit||control_exit))
