#!/bin/bash
# Push + run a dsp_graph skel on the phone (only under the host phone lock:
# PHONE_LOCK_OWNER=<branch> ~/.cache/android-phone/phone-run ./run_graph.sh ...):
#   ./run_graph.sh <outdir> <input.bin> <expected.bin> [iters]
# outdir: qdq_net.py --skel's (tg_hmx_rpc.so, client, blob.bin). D = the phone directory.
set -euo pipefail
S="${ANDROID_SERIAL:-239dbd8f}"; D="${D:-/data/local/tmp/codex-android-tinygrad-graph}"; A=(adb -s "$S")
O="$1"; "${A[@]}" shell "mkdir -p $D/case"
"${A[@]}" push "$O/tg_hmx_rpc.so" "$O/client" $D/ >/dev/null
"${A[@]}" push "$2" $D/case/a.bin >/dev/null; "${A[@]}" push "$O/blob.bin" $D/case/b.bin >/dev/null; "${A[@]}" push "$3" $D/case/ref.bin >/dev/null
"${A[@]}" shell "cd $D && chmod 755 client && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D timeout ${RUN_TIMEOUT:-300} ./client \
  'file:///tg_hmx_rpc.so?tg_hmx_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' case ${4:-10}; echo exit=\$?"
