#!/usr/bin/env bash
# Record 20 us thermal baseline then 100 us after one-time ring injection.
set -euo pipefail
binary=${1:?binary};out=${2:?new directory};fraction=${3:-.1};mkdir "$out"
run() {
 local name=$1 device=$2 fraction=$3
 mkdir "$out/$name"
 export RPIC_WEB_CAPTURE="$out/$name/capture.h5" RPIC_WEB_FRAMES=4096 RPIC_WEB_CROP=1 RPIC_WEB_POOL=16384 RPIC_LIVE_PATH="$out/$name/live.h5"
 timeout --signal=TERM --kill-after=5s 7190s "$binary" \
 --backend cuda --device "$device" --cells 256 --ppc 7 --bragg-mode 1 \
 --density 111639834702.32187 --te 2000 --ti 1000 --bx-t .000001869630096 --by-t .00003567491076 \
 --hot-fraction "$fraction" --ring-energy-ev 10 --ring-width .1 --ring-turn-on-us 20 \
 --core-thermostat-us 0 --dt .01 --burn-ms 0 --duration-ms .12 --sample-us .005 --window 1024 \
 --diagnostic-stride 200 --seed 9501 --output "$out/$name/spectrum.h5"
}
run ring 0 "$fraction" >"$out/ring.log" 2>&1 & a=$!
run control 1 0 >"$out/control.log" 2>&1 & b=$!
printf '%s\n' "$a" "$b" >"$out/launcher_pids"
set +e
wait "$a"; x=$?;printf '%s\n' "$x" >"$out/ring_exit"
wait "$b"; y=$?;printf '%s\n' "$y" >"$out/control_exit"
exit $((x||y))
