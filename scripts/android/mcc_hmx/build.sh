#!/bin/bash
# Build the MCC decoder-block skel (mcc_hmx_rpc.so) + client with the Hexagon SDK's qaic/headers and a Hexagon
# toolchain that knows -mhmx (the SDK's own 19.0.04, or the login-free Hexagon_open_access 19.0.02).
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
HEX_ARCH="${HEX_ARCH:-v69}"
SRC="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$SRC/build}"
mkdir -p "$OUT" && cd "$OUT"
cp "$SRC"/mcc_hmx_rpc.idl "$SRC"/mcc_hmx_impl.c "$SRC"/mcc_hmx_client.c "$SRC"/mcc_block.h .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" mcc_hmx_rpc.idl
INC=(-I . -I "$SRC" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o mcc_hmx_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -mhmx -Wall \
  "${INC[@]}" "${QURT_INC[@]}" -o impl.o mcc_hmx_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o mcc_hmx_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 -c "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o mcc_hmx_stub.o mcc_hmx_rpc_stub.c  # for other clients (the demo app)
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o mcc_hmx_client mcc_hmx_client.c -lm mcc_hmx_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
echo "built $OUT/mcc_hmx_rpc.so $OUT/mcc_hmx_client"
