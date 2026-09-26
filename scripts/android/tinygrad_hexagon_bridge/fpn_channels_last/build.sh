#!/bin/bash
# Build + run fpn_client on the phone's CDSP through the TVM-free fpn_rpc FastRPC skel.
#   DATA:  gen_fpn_test_data.py output (lvl*_*.bin, levels.txt) plus fpn_out_kernels.c from
#          ../hex_conv3x3_fpnout_kernel.py --out $DATA/fpn_out_kernels.c
#   RDATA: ../roialign_fast data dir (calls.txt, callN_rois.bin, callN_ref.bin)
set -euo pipefail

# the hand-written kernel headers live in the tinygrad fork (test/external/dsp/hand/); this
# links them in from the pinned revision if they are not present yet
_d="$(dirname "$0")"; while [ ! -f "$_d/fetch_hand_kernels.sh" ] && [ "$_d" != / ]; do _d="$(dirname "$_d")"; done
[ -f "$_d/fetch_hand_kernels.sh" ] && "$_d/fetch_hand_kernels.sh" >/dev/null
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}" "${DATA:?}" "${RDATA:?}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
HEX_ARCH="${HEX_ARCH:-v73}"
REPS="${REPS:-5}"
CONV_REPS="${CONV_REPS:-3}"
TURBO="${TURBO:-0}"
OUT="${OUT:-$(mktemp -d)}"
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "$OUT"
cp "$SRC"/fpn_rpc.idl "$SRC"/fpn_impl.c "$SRC"/fpn_client.c "$SRC"/layout_kernels.h "$SRC"/../roialign_fast/roialign_kernel.h "$DATA"/fpn_out_kernels.c .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" fpn_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o fpn_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -Wall \
  -Wno-unused-function "${INC[@]}" "${QURT_INC[@]}" -o impl.o fpn_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o fpn_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o fpn_client fpn_client.c fpn_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc -lm
D=/data/local/tmp/fpn_channels_last
adb -s "$DEVICE_SERIAL" shell "mkdir -p $D"
adb -s "$DEVICE_SERIAL" push fpn_client fpn_rpc.so "$DATA"/levels.txt "$DATA"/lvl*_*.bin "$RDATA"/calls.txt \
  "$RDATA"/call*_rois.bin "$RDATA"/call*_ref.bin $D/ >/dev/null
adb -s "$DEVICE_SERIAL" shell "chmod 755 $D/fpn_client && cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D \
  ./fpn_client 'file:///fpn_rpc.so?fpn_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' $REPS $CONV_REPS $TURBO"
