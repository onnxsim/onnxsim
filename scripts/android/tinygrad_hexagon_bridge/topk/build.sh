#!/bin/bash
# Build + run the TopK kernel on the phone's CDSP through a dedicated, TVM-free FastRPC skel (same
# pattern as ../roialign_fast/build.sh). Needs the flat data from gen_topk_test_data.py in $DATA.
# Also builds/runs the phone-CPU ONNX Runtime baseline if $ORT_AAR (an extracted
# onnxruntime-android AAR: headers/ + jni/arm64-v8a/libonnxruntime.so) is set.
set -euo pipefail

# the hand-written kernel headers live in the tinygrad fork (test/external/dsp/hand/); this
# links them in from the pinned revision if they are not present yet
_d="$(dirname "$0")"; while [ ! -f "$_d/fetch_hand_kernels.sh" ] && [ "$_d" != / ]; do _d="$(dirname "$_d")"; done
[ -f "$_d/fetch_hand_kernels.sh" ] && "$_d/fetch_hand_kernels.sh" >/dev/null
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}" "${DATA:?directory with calls.txt and callN_*.bin}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
HEX_ARCH="${HEX_ARCH:-v73}"
REPS="${REPS:-21}"
TURBO="${TURBO:-0}"
OUT="${OUT:-$(mktemp -d)}"
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "$OUT"
cp "$SRC"/topk_rpc.idl "$SRC"/topk_impl.c "$SRC"/topk_kernel.h "$SRC"/topk_client.c .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" topk_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o topk_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -Wall \
  "${INC[@]}" "${QURT_INC[@]}" -o impl.o topk_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o topk_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o topk_client topk_client.c topk_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
D=/data/local/tmp/topk_hvx
adb -s "$DEVICE_SERIAL" shell "mkdir -p $D"
adb -s "$DEVICE_SERIAL" push topk_client topk_rpc.so "$DATA"/calls.txt "$DATA"/call*_*.bin $D/ >/dev/null
adb -s "$DEVICE_SERIAL" shell "chmod 755 $D/topk_client && cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D \
  ./topk_client 'file:///topk_rpc.so?topk_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' $REPS $TURBO"
if [ -n "${ORT_AAR:-}" ]; then
  "$NDK_CLANG" -O2 -I "$ORT_AAR/headers" -o ort_topk_bench "$SRC/ort_topk_bench.c" -L "$ORT_AAR/jni/arm64-v8a" -lonnxruntime
  adb -s "$DEVICE_SERIAL" push ort_topk_bench "$ORT_AAR/jni/arm64-v8a/libonnxruntime.so" "$DATA"/call*.onnx $D/ >/dev/null
  adb -s "$DEVICE_SERIAL" shell "chmod 755 $D/ort_topk_bench && cd $D && LD_LIBRARY_PATH=. ./ort_topk_bench"
fi
