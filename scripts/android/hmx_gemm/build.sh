#!/bin/bash
# Build the HMX GEMM skel (hmx_gemm_rpc.so) + client with the Hexagon SDK's qaic/headers and a Hexagon
# toolchain that knows -mhmx (the SDK's own 19.0.04, or the login-free Hexagon_open_access 19.0.02).
set -euo pipefail

# the hand-written kernel headers live in the tinygrad fork (test/external/dsp/hand/); this
# links them in from the pinned revision if they are not present yet
_d="$(dirname "$0")"; while [ ! -f "$_d/fetch_hand_kernels.sh" ] && [ "$_d" != / ]; do _d="$(dirname "$_d")"; done
[ -f "$_d/fetch_hand_kernels.sh" ] && "$_d/fetch_hand_kernels.sh" >/dev/null
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
HEX_ARCH="${HEX_ARCH:-v69}"
SRC="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$SRC/build}"
mkdir -p "$OUT" && cd "$OUT"
cp "$SRC"/hmx_gemm_rpc.idl "$SRC"/hmx_gemm_impl.c "$SRC"/hmx_gemm_client.c "$SRC"/hmx_gemm_llm_client.c "$SRC"/hmx_gemm_u8_client.c "$SRC"/hmx_gemm_layers_client.c "$SRC"/hmx_qconv_client.c "$SRC"/hmx_gemm.h "$SRC"/hmx_qconv.h "$SRC"/hmx_qconv3.h "$SRC"/qc_case.h "$SRC"/runner/rn_model.h "$SRC"/runner/rn_exec.h "$SRC"/runner/rn_load.h "$SRC"/runner/hmx_runner_client.c "$SRC"/hmx_gemm_u8.h "$SRC"/hmx_block.h "$SRC"/hmx_runtime.h .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" hmx_gemm_rpc.idl
INC=(-I . -I "$SRC/runner" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o hmx_gemm_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -mhmx -Wall \
  "${INC[@]}" "${QURT_INC[@]}" -o impl.o hmx_gemm_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o hmx_gemm_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o hmx_gemm_client hmx_gemm_client.c -lm hmx_gemm_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o hmx_gemm_llm_client hmx_gemm_llm_client.c -lm hmx_gemm_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o hmx_gemm_u8_client hmx_gemm_u8_client.c -lm hmx_gemm_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
for c in hmx_gemm_layers_client hmx_qconv_client hmx_runner_client; do
  "$NDK_CLANG" -O2 -ffp-contract=off "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o $c $c.c -lm hmx_gemm_rpc_stub.c \
    -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
done
echo "built $OUT/hmx_gemm_rpc.so $OUT/hmx_gemm_client $OUT/hmx_gemm_llm_client $OUT/hmx_gemm_u8_client $OUT/hmx_gemm_layers_client $OUT/hmx_qconv_client $OUT/hmx_runner_client"
