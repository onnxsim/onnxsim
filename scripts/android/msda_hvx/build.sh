#!/bin/bash
# Build the generic MSDA skel (msda_rpc.idl / msda_impl.c / msda_kernel.h) + its phone bench client
# into $OUT (also msda_stub.o, for other clients to link), then -- unless BUILD_ONLY=1 -- push them with
# msda_ref.py case directories to a phone directory of your own and run the bench.
#   CASES: case dirs, FLAGS: comma list (threads | 256 * (queries per job / 16)), REPS, TURBO,
#   D: phone directory (default /data/local/tmp/msda_hvx-$USER). Run the phone part under the
#   host's phone lock (e.g. PHONE_LOCK_OWNER=<branch> ~/.cache/android-phone/phone-run ./build.sh).
set -euo pipefail

# the hand-written kernel headers live in the tinygrad fork (test/external/dsp/hand/); this
# links them in from the pinned revision if they are not present yet
_d="$(dirname "$0")"; while [ ! -f "$_d/fetch_hand_kernels.sh" ] && [ "$_d" != / ]; do _d="$(dirname "$_d")"; done
[ -f "$_d/fetch_hand_kernels.sh" ] && "$_d/fetch_hand_kernels.sh" >/dev/null
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
DEVICE_SERIAL="${DEVICE_SERIAL:-${ANDROID_SERIAL:-239dbd8f}}"
HEX_ARCH="${HEX_ARCH:-v69}"  # the test phone is SM8475 (Hexagon V69)
REPS="${REPS:-10}" TURBO="${TURBO:-0}" FLAGS="${FLAGS:-1,4}"
SRC="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$SRC/build}"
mkdir -p "$OUT" && cd "$OUT"
cp "$SRC"/msda_rpc.idl "$SRC"/msda_impl.c "$SRC"/msda_kernel.h "$SRC"/msda_shape.h "$SRC"/msda_io.h "$SRC"/msda_client.c .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" msda_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o msda_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -Wall \
  "${INC[@]}" "${QURT_INC[@]}" -o impl.o msda_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o msda_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 -c "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o msda_stub.o msda_rpc_stub.c
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o msda_client msda_client.c msda_stub.o \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc -lm
[ -n "${BUILD_ONLY:-}" ] && exit 0
: "${CASES:?msda_ref.py case dirs}"
D="${D:-/data/local/tmp/msda_hvx-${USER:-user}}"
adb -s "$DEVICE_SERIAL" shell "rm -rf $D && mkdir -p $D"
adb -s "$DEVICE_SERIAL" push msda_client msda_rpc.so $D/ >/dev/null
names=()
for c in $CASES; do
  n=$(basename "$c"); names+=("$n")
  adb -s "$DEVICE_SERIAL" push "$c" "$D/$n" >/dev/null
done
adb -s "$DEVICE_SERIAL" shell "chmod 755 $D/msda_client && cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D \
  ./msda_client 'file:///msda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' $REPS $TURBO $FLAGS ${names[*]}"
