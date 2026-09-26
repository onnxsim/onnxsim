#!/bin/bash
# Build + run the NMS kernels on the phone's CDSP through a dedicated, TVM-free FastRPC skel (same
# pattern as ../roialign_fast/build.sh). Needs gen_nms_test_data.py's output in $DATA.
set -euo pipefail

# the hand-written kernel headers live in the tinygrad fork (test/external/dsp/hand/); this
# links them in from the pinned revision if they are not present yet
_d="$(dirname "$0")"; while [ ! -f "$_d/fetch_hand_kernels.sh" ] && [ "$_d" != / ]; do _d="$(dirname "$_d")"; done
[ -f "$_d/fetch_hand_kernels.sh" ] && "$_d/fetch_hand_kernels.sh" >/dev/null
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}" "${DATA:?directory with {level,class}_{calls.txt,boxes,scores,ref,thr}.bin}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
HEX_ARCH="${HEX_ARCH:-v73}"
REPS="${REPS:-7}"
TURBO="${TURBO:-0}"
OUT="${OUT:-$(mktemp -d)}"
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "$OUT"
cp "$SRC"/nms_rpc.idl "$SRC"/nms_impl.c "$SRC"/nms_kernel.h "$SRC"/nms_client.c .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" nms_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o nms_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -ffp-contract=off -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -Wall \
  "${INC[@]}" "${QURT_INC[@]}" -o impl.o nms_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o nms_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o nms_client nms_client.c nms_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
D=/data/local/tmp/nms_hvx
adb -s "$DEVICE_SERIAL" shell "mkdir -p $D"
adb -s "$DEVICE_SERIAL" push nms_client nms_rpc.so "$DATA"/*_calls.txt "$DATA"/*_boxes.bin "$DATA"/*_scores.bin \
  "$DATA"/*_ref.bin "$DATA"/*_thr.bin $D/ >/dev/null
adb -s "$DEVICE_SERIAL" shell "chmod 755 $D/nms_client && cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D \
  ./nms_client 'file:///nms_rpc.so?nms_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' $REPS $TURBO"
