#!/bin/bash
# Push + run the tinygrad-generated HMX kernel on the phone: ./run.sh setup | ./run.sh run M K N [iters] | ./run.sh clean
# (a --conv build also pushes its case dir; run it with CASE=case)
# Only under the host phone lock (PHONE_LOCK_OWNER=<branch> ~/.cache/android-phone/phone-run ./run.sh ...); check the DSP after
# new variants with scripts/android/hmx_probe/run.sh health.
set -euo pipefail
S="${ANDROID_SERIAL:-239dbd8f}"; D="${D:-/data/local/tmp/codex-android-tinygrad-hmx}"
B="${OUT:-$(cd "$(dirname "$0")" && pwd)/build}"; A=(adb -s "$S")
case "$1" in
  setup) "${A[@]}" shell "mkdir -p $D" && "${A[@]}" push "$B/tg_hmx_client" "$B/tg_hmx_rpc.so" $D/ >/dev/null && "${A[@]}" shell "chmod 755 $D/tg_hmx_client" && \
         { [ ! -d "$B/case" ] || "${A[@]}" push "$B/case" $D/ >/dev/null; } ;;
  run) shift; "${A[@]}" shell "cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D I8=${I8:-0} RQ=${RQ:-0} ZY=${ZY:-131} LO=${LO:-0} ${CASE:+CASE=$CASE} timeout ${RUN_TIMEOUT:-60} ./tg_hmx_client \
         'file:///tg_hmx_rpc.so?tg_hmx_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' $*; echo exit=\$?" ;;
  clean) "${A[@]}" shell "rm -rf $D" ;;
esac
