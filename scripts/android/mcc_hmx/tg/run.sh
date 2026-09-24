#!/bin/bash
# Push + run a captured bundle's replay skel on the phone, with the msda health check:
#   ./run.sh <bundle> [iters] [ref float32 file]      (only under the host phone lock)
set -euo pipefail
S="${ANDROID_SERIAL:-239dbd8f}"; D="${D:-/data/local/tmp/codex-android-mcc-tg}"
B="$(cd "$1" && pwd)"; A=(adb -s "$S")
MSDA_BUILD="${MSDA_BUILD:-$HOME/.cache/msda_generic_build}"; MSDA_CASE="${MSDA_CASE:-$HOME/.cache/msda_generic_cases/bevformer_tsa}"
"${A[@]}" shell "rm -rf $D && mkdir -p $D/b"
"${A[@]}" push "$B/build/replay_client" "$B/build/replay_rpc.so" "$MSDA_BUILD/msda_client" "$MSDA_BUILD/msda_rpc.so" $D/ >/dev/null
"${A[@]}" push "$MSDA_CASE" "$D/case" >/dev/null
for f in "$B"/r*.bin "$B"/regions.txt "$B"/calls.txt "$B"/out_bytes.txt "$B"/k*.c; do "${A[@]}" push "$f" "$D/b/" >/dev/null; done
REF=""; [ -n "${3:-}" ] && { "${A[@]}" push "$3" "$D/ref.bin" >/dev/null; REF=ref.bin; }
"${A[@]}" shell "chmod 755 $D/replay_client $D/msda_client && cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D timeout ${RUN_TIMEOUT:-120} \
  ./replay_client 'file:///replay_rpc.so?replay_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' b ${2:-5} $REF; echo exit=\$?"
sleep 2
"${A[@]}" shell "cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D timeout 60 ./msda_client \
  'file:///msda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' 3 0 4 case 2>&1 | tail -1"
