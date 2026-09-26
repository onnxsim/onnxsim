#!/bin/bash
# Build the replay skel (replay_rpc.so) + client for one captured bundle (mcc_tg.py check --no-run --save <bundle>):
#   HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... NDK_CLANG=... ./build.sh <bundle>   -> <bundle>/build/
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
HEX_ARCH=v69
SRC="$(cd "$(dirname "$0")" && pwd)"
BUNDLE="$(cd "$1" && pwd)"
OUT="$BUNDLE/build"; mkdir -p "$OUT"
python3 "$SRC/emit.py" skel "$BUNDLE"
cd "$OUT"
cp "$SRC"/replay_rpc.idl "$SRC"/replay_impl.c "$SRC"/replay_client.c "$BUNDLE"/replay_gen.h .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" replay_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
CC=("$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon$HEX_ARCH -mhvx=$HEX_ARCH -mhvx-length=128b -mhmx)
objs=()
for k in "$BUNDLE"/k*.c; do
  o="$(basename "${k%.c}").o"
  [ "$o" -nt "$k" ] || "${CC[@]}" -Wno-everything -o "$o" "$k"
  objs+=("$o")
done
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon$HEX_ARCH "${INC[@]}" -o skel.o replay_rpc_skel.c
"${CC[@]}" "${INC[@]}" "${QURT_INC[@]}" -o impl.o replay_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o replay_rpc.so skel.o impl.o "${objs[@]}" "$LIBPATH/pic/libgcc.a" "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 "${INC[@]}" -o replay_client replay_client.c replay_rpc_stub.c -lm \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
echo "built $OUT/replay_rpc.so $OUT/replay_client"
