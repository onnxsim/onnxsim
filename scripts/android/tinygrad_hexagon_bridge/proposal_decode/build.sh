#!/bin/bash
# Build + run the proposal-decode kernel on the phone's CDSP through its own TVM-free FastRPC skel,
# and ONNX Runtime's CPU EP on the phone for exactly the same rest.onnx nodes.
# Needs $DATA from capture_proposal_decode.py + pd_host_check (levels.txt, lN_*.bin) and
# make_pd_ort_model.py (pd_region.onnx, pd_region_io.txt).
#   ORT_AAR_DIR: an extracted stock onnxruntime-android AAR (headers/, jni/arm64-v8a/libonnxruntime.so)
set -euo pipefail

# the hand-written kernel headers live in the tinygrad fork (test/external/dsp/hand/); this
# links them in from the pinned revision if they are not present yet
_d="$(dirname "$0")"; while [ ! -f "$_d/fetch_hand_kernels.sh" ] && [ "$_d" != / ]; do _d="$(dirname "$_d")"; done
[ -f "$_d/fetch_hand_kernels.sh" ] && "$_d/fetch_hand_kernels.sh" >/dev/null
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}" "${DATA:?directory with levels.txt and lN_*.bin}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
HEX_ARCH="${HEX_ARCH:-v73}"
REPS="${REPS:-15}"
TURBO="${TURBO:-0}"
ORT_AAR_DIR="${ORT_AAR_DIR:-}"
OUT="${OUT:-$(mktemp -d)}"
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "$OUT"
cp "$SRC"/pd_rpc.idl "$SRC"/pd_impl.c "$SRC"/pd_kernel.h "$SRC"/pd_client.c "$SRC"/ort_pd_bench.c .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" pd_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o pd_rpc_skel.c
# -ffp-contract=off: the graph does mul then add as separate fp32 ops; fusing them into an FMA
# would change rounding and break bit-exactness with ONNX Runtime.
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -ffp-contract=off -mcpu=hexagon"$HEX_ARCH" -Wall \
  "${INC[@]}" "${QURT_INC[@]}" -o impl.o pd_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o pd_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o pd_client pd_client.c pd_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
D=/data/local/tmp/proposal_decode
adb -s "$DEVICE_SERIAL" shell "mkdir -p $D"
adb -s "$DEVICE_SERIAL" push pd_client pd_rpc.so "$DATA"/levels.txt "$DATA"/l*_anchors.bin "$DATA"/l*_deltas.bin \
  "$DATA"/l*_nchw_q.bin "$DATA"/l*_idx.bin "$DATA"/l*_ref.bin $D/ >/dev/null
adb -s "$DEVICE_SERIAL" shell "chmod 755 $D/pd_client && cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D \
  ./pd_client 'file:///pd_rpc.so?pd_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' $REPS $TURBO"
if [ -n "$ORT_AAR_DIR" ]; then
  "$NDK_CLANG" -O2 -I "$ORT_AAR_DIR/headers" -o ort_pd_bench ort_pd_bench.c -L "$ORT_AAR_DIR/jni/arm64-v8a" -lonnxruntime
  adb -s "$DEVICE_SERIAL" push ort_pd_bench "$ORT_AAR_DIR/jni/arm64-v8a/libonnxruntime.so" "$DATA"/pd_region.onnx \
    "$DATA"/pd_region_io.txt "$DATA"/l*_idx64.bin $D/ >/dev/null
  adb -s "$DEVICE_SERIAL" shell "chmod 755 $D/ort_pd_bench && cd $D && LD_LIBRARY_PATH=$D ./ort_pd_bench"
fi
