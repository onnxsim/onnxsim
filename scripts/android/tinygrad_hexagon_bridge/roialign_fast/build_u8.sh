#!/bin/bash
# Build + run the merged uint8 RoiAlign kernel (roialign_u8_kernel.h) on the phone's CDSP through its
# own TVM-free FastRPC skel (roialign_u8_rpc.idl; the fp32 roialign_rpc skel is left as is for A/B).
# $DATA is one capture_merged_io.py image directory (meta.txt, l*_u8.bin, {box,mask}_{rois,rows}*.bin,
# {box,mask}_ref_u8.bin). CONFIGS: comma list of flags (threads | 256*prefetch | 512*sort).
set -euo pipefail

# the hand-written kernel headers live in the tinygrad fork (test/external/dsp/hand/); this
# links them in from the pinned revision if they are not present yet
_d="$(dirname "$0")"; while [ ! -f "$_d/fetch_hand_kernels.sh" ] && [ "$_d" != / ]; do _d="$(dirname "$_d")"; done
[ -f "$_d/fetch_hand_kernels.sh" ] && "$_d/fetch_hand_kernels.sh" >/dev/null
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}" "${DATA:?capture_merged_io.py image directory}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
HEX_ARCH="${HEX_ARCH:-v69}"  # the test phone is SM8475 (Hexagon V69)
REPS="${REPS:-5}"
TURBO="${TURBO:-0}"
CONFIGS="${CONFIGS:-1,4,6,260,262,516,518,772,774}"
OUT="${OUT:-$(mktemp -d)}"
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "$OUT"
cp "$SRC"/roialign_u8_rpc.idl "$SRC"/roialign_u8_impl.c "$SRC"/roialign_u8_kernel.h "$SRC"/roialign_u8_client.c .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" roialign_u8_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o roialign_u8_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -Wall \
  "${INC[@]}" "${QURT_INC[@]}" -o impl.o roialign_u8_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o roialign_u8_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o roialign_u8_client roialign_u8_client.c roialign_u8_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc -lm
D=/data/local/tmp/roialign_u8
adb -s "$DEVICE_SERIAL" shell "mkdir -p $D"
adb -s "$DEVICE_SERIAL" push roialign_u8_client roialign_u8_rpc.so "$DATA"/meta.txt "$DATA"/l*_u8.bin "$DATA"/*_rois*.bin "$DATA"/*_rows*.bin "$DATA"/*_ref_u8.bin $D/ >/dev/null
adb -s "$DEVICE_SERIAL" shell "chmod 755 $D/roialign_u8_client && cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D \
  ${CLIENT_ENV:-} ./roialign_u8_client 'file:///roialign_u8_rpc.so?roialign_u8_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' $REPS $TURBO $CONFIGS"
