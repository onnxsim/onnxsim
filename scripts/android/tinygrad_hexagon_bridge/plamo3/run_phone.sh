#!/bin/bash
# Run the PLaMo 3 chunk programs on the attached phone: pushes onnx-remote-hexagon-worker, starts it, forwards its port, runs
# run_phone.py under the shared phone lock, then stops the worker. The compiled chunks come from the tghx directory.
#   run_phone.sh WORKER CHECKPOINT CHUNK_DIR TGHX_DIR [run_phone.py args...]
set -eo pipefail
worker=$1; ckpt=$2; chunks=$3; tghx=$4; shift 4
here=$(cd "$(dirname "$0")" && pwd)
serial="${DEVICE_SERIAL:-239dbd8f}"; port="${RUNNER_PORT:-39520}"; lock="${PHONE_LOCK:-$HOME/.cache/android-phone/phone-run}"
R=/data/local/tmp/onnx-remote-hexagon
py="${PYTHON:-python3}"
args=$(printf "%q " "$@")
job="set -e
adb -s $serial shell mkdir -p $R
adb -s $serial push -q '$worker' $R/onnx-remote-hexagon-worker >/dev/null
adb -s $serial shell 'kill \$(pidof onnx-remote-hexagon-worker) 2>/dev/null; chmod 755 $R/onnx-remote-hexagon-worker' || true
adb -s $serial shell \"cd $R && exec ./onnx-remote-hexagon-worker --port $port --cache-dir $R/cache\" > worker.log 2>&1 &
adb -s $serial forward tcp:$port tcp:$port >/dev/null
sleep 2
status=0
'$py' '$here/run_phone.py' '$ckpt' '$chunks' '$tghx' --port $port $args || status=\$?
adb -s $serial shell 'kill \$(pidof onnx-remote-hexagon-worker) 2>/dev/null' || true
adb -s $serial forward --remove tcp:$port || true
exit \$status"
if [ -x "$lock" ]; then PHONE_LOCK_OWNER="${PHONE_LOCK_OWNER:-plamo3-hexagon}" "$lock" bash -c "$job"; else bash -c "$job"; fi
