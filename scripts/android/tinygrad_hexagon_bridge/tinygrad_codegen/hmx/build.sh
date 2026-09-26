#!/bin/bash
# Build the tinygrad-generated HMX kernel skel (tg_hmx_rpc.so) + client for one shape:
#   HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... TINYGRAD=<tinygrad hvx-hmx checkout> ./build.sh M K N [--i8 | --rq [--relu] | --conv H W C N S [--relu]]
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}" "${TINYGRAD:?}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
PY="${PY:-python3}"; HEX_ARCH=v69
SRC="$(cd "$(dirname "$0")" && pwd)"
# hmx_runtime.h moved to the tinygrad fork (test/external/dsp/hand/hmx/), where it sits next to the
# kernels that need it. It used to be read from scripts/android/hmx_gemm/, which no longer has it
# since the hand-kernel consolidation (#1978). TINYGRAD is that checkout; fall back to the other
# build.sh siblings' convention if it is unset.
HMX_RT="$TINYGRAD/test/external/dsp/hand/hmx/hmx_runtime.h"
[ -f "$HMX_RT" ] || HMX_RT="$SRC/../../../hmx_gemm/hmx_runtime.h"
[ -f "$HMX_RT" ] || { echo "cannot find hmx_runtime.h (looked in $HMX_RT); see HEXAGON_SDK.md" >&2; exit 1; }
OUT="${OUT:-$SRC/build}"; mkdir -p "$OUT"
HMX=1 DEV=DSP MOCKDSP=1 TC=1 TC_OPT=1 HVX_ARCH=v69 CC="${CC:-clang-19}" PYTHONPATH="$TINYGRAD" "$PY" "$SRC/gen_kernel.py" "$1" "$2" "$3" "$OUT" "${@:4}"
cd "$OUT"
cp "$SRC"/tg_hmx_rpc.idl "$SRC"/tg_hmx_impl.c "$SRC"/tg_hmx_client.c "$HMX_RT" .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" tg_hmx_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon$HEX_ARCH "${INC[@]}" -o skel.o tg_hmx_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon$HEX_ARCH -mhvx=$HEX_ARCH -mhvx-length=128b -mhmx \
  "${INC[@]}" "${QURT_INC[@]}" -o impl.o tg_hmx_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o tg_hmx_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.a" "$LIBPATH/pic/libgcc.so"  # .a: the memcpy helper the compiler emits for the 2 KB tile values
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o tg_hmx_client tg_hmx_client.c tg_hmx_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
echo "built $OUT/tg_hmx_rpc.so $OUT/tg_hmx_client"
