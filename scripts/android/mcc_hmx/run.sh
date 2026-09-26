#!/bin/bash
# Push + run the MCC decoder-block skel on the phone, with the known-good msda health check:
#   ./run.sh setup <ref.py export dir> | ./run.sh run <Q> <blocks> [iters] | ./run.sh health
# Run only under the host phone lock: PHONE_LOCK_OWNER=<branch> ~/.cache/android-phone/phone-run ./run.sh ...
set -euo pipefail
S="${ANDROID_SERIAL:-239dbd8f}"
D="${D:-/data/local/tmp/codex-android-mcc-hmx}"
B="${OUT:-$(cd "$(dirname "$0")" && pwd)/build}"
MSDA_BUILD="${MSDA_BUILD:-$HOME/.cache/msda_generic_build}"
MSDA_CASE="${MSDA_CASE:-$HOME/.cache/msda_generic_cases/bevformer_tsa}"
A=(adb -s "$S")
case "$1" in
  setup)
    "${A[@]}" shell "mkdir -p $D/data"
    "${A[@]}" push "$B/mcc_hmx_client" "$B/mcc_hmx_rpc.so" "$MSDA_BUILD/msda_client" "$MSDA_BUILD/msda_rpc.so" $D/ >/dev/null
    "${A[@]}" push "$MSDA_CASE" "$D/case" >/dev/null
    for f in "$2"/x0.bin "$2"/blk*.bin "$2"/ref_out*.bin "$2"/head.bin "$2"/kv.bin "$2"/xyz.bin "$2"/ref_occ.bin "$2"/ref_rgb.bin; do "${A[@]}" push "$f" "$D/data/" >/dev/null; done
    "${A[@]}" shell "chmod 755 $D/mcc_hmx_client $D/msda_client" ;;
  run)
    shift
    "${A[@]}" shell "cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D timeout ${RUN_TIMEOUT:-120} ./mcc_hmx_client \
      'file:///mcc_hmx_rpc.so?mcc_hmx_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' data $*; echo exit=\$?" ;;
  health)
    sleep 2
    "${A[@]}" shell "cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D timeout 60 ./msda_client \
      'file:///msda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' 3 0 4 case 2>&1 | tail -1; echo exit=\$?" ;;
  clean) "${A[@]}" shell "rm -rf $D" ;;
esac
