#!/bin/bash
# Time every model listed in <modeldir>/manifest.json (L=lo and L=hi) plus tiny_L0.onnx on the
# phone's HTP, strict (no CPU fallback), via ../qnn_shell/qnn_run with an EP-context cache.
# Pushes the QNN/ORT libs once to its own device dir, then one model at a time.
#   ./run_ceiling.sh <modeldir> [iters] [perf_mode]    -> <modeldir>/raw/<model>.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MD="$(cd "$1" && pwd)"
ITERS="${2:-15}"
PERF="${3:-burst}"
DEV="${DEVICE_SERIAL:-239dbd8f}"
R="${REMOTE_DIR:-/data/local/tmp/qnn_ceiling}"
QS="$HERE/../qnn_shell"
NDK_CXX="${NDK_CXX:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++}"
[ -d "$QS/libs" ] || "$QS/fetch_libs.sh"
"$NDK_CXX" -O2 -std=c++17 -static-libstdc++ -I "$QS/headers" -o "$QS/qnn_run" "$QS/qnn_run.cpp" -L "$QS/libs" -lonnxruntime
adb -s "$DEV" shell "mkdir -p $R"
adb -s "$DEV" push -q "$QS/qnn_run" "$QS"/libs/* $R/
mkdir -p "$MD/raw"
models=$(python3 -c "import json,sys;m=json.load(open('$MD/manifest.json'));print(' '.join(f\"{e['tag']}_L{L}\" for e in m for L in (e['lo'],e['hi'])))")
for name in tiny_L0 $models ${EXTRA_MODELS:-}; do
  log="$MD/raw/$name.$PERF.log"
  [ -s "$log" ] && grep -q '^PASS' "$log" && { echo "skip $name (done)"; continue; }
  adb -s "$DEV" push -q "$MD/$name.onnx" $R/
  adb -s "$DEV" shell "cd $R && rm -f ctx_$name.onnx && QNN_PERF=$PERF LD_LIBRARY_PATH=$R \
    ADSP_LIBRARY_PATH='$R;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' \
    ./qnn_run $name.onnx - htp $ITERS o_$name ctx_$name.onnx" > "$log" 2>&1 || true
  adb -s "$DEV" shell "cd $R && rm -f $name.onnx ctx_$name.onnx o_${name}_*.bin"
  echo "$name: $(grep -E '^(PASS|FAIL)' "$log" | head -1)"
done
