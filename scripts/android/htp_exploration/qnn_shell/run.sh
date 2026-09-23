#!/bin/bash
# Build qnn_run, push it + the bundled QNN libs, and run a model on the HTP from a plain adb shell.
#   ./run.sh <model.onnx> <input.bin|-> <mode: cpu|htp|htp-fallback> <iters> [ctx.onnx]
# Env passed through to the device: QNN_PERF (e.g. burst), ORT_THREADS, ORT_LOG, QNN_EXTRA.
set -euo pipefail
cd "$(dirname "$0")"
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
NDK_CXX="${NDK_CXX:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++}"
R=/data/local/tmp/qnn
[ -d libs ] || ./fetch_libs.sh
"$NDK_CXX" -O2 -std=c++17 -static-libstdc++ -I headers -o qnn_run qnn_run.cpp -L libs -lonnxruntime
adb -s "$DEVICE_SERIAL" shell "mkdir -p $R"
adb -s "$DEVICE_SERIAL" push -q qnn_run libs/* "$1" $R/
[ "$2" = "-" ] || adb -s "$DEVICE_SERIAL" push -q "$2" $R/
IN=$([ "$2" = "-" ] && echo - || basename "$2")
ENVS=""
for v in QNN_PERF ORT_THREADS ORT_LOG QNN_EXTRA; do [ -n "${!v:-}" ] && ENVS="$ENVS $v=${!v}"; done
# ADSP_LIBRARY_PATH puts our own (unsigned) libQnnHtpV69Skel.so first; the vendor's copy of the
# QNN libs is an older build that the 2.50 plugin can't use (segfaults, see qnn_shell_findings.md).
adb -s "$DEVICE_SERIAL" shell "cd $R && $ENVS LD_LIBRARY_PATH=$R \
  ADSP_LIBRARY_PATH='$R;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' \
  ./qnn_run $(basename "$1") $IN $3 $4 out_$3 ${5:-}"
