#!/bin/bash
# Build the Mask R-CNN demo APK.
#   HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... ANDROID_HOME=~/android-sdk ./build_app.sh
# gradle 8.7 needs a JDK <= 21 (JAVA_HOME; JDK 25 fails with "Unsupported class file major version 69").
# Heavy (gradle, ~1.7 GB peak); on a shared machine run it capped, e.g.
#   systemd-run --user --wait --collect --pipe -p MemoryMax=12G -p MemorySwapMax=0 ./build_app.sh
# 1. ORT + QNN EP + Qualcomm QNN runtime libs (Maven Central) via ../htp_exploration/qnn_shell/fetch_libs.sh
# 2. the three Hexagon FastRPC skels + their ARM stubs, built by ../e2e_pipeline/build.sh (BUILD_ONLY)
# 3. native/maskrcnn_engine.cpp (includes ../e2e_pipeline/e2e_run.cpp) -> libmaskrcnn_demo.so,
#    native/{yolo,sam,sr}_engine.cpp -> lib{yolo,sam,sr}_demo.so, native/mcc_engine.cpp (+ the ../mcc_hmx
#    decoder skel) -> libmcc_demo.so, libmcc_hmx_rpc.so, native/rtdetr_engine.cpp (+ the ../msda_hvx
#    skel) -> librtdetr_demo.so, libmsda_rpc.so, native/game_engine.cpp (NSS + NFRU, OpenCL) -> libgame_demo.so
# 4. everything into app/src/main/jniLibs/arm64-v8a, then gradle assembleDebug (offline).
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
ANDROID_HOME="${ANDROID_HOME:-$HOME/android-sdk}"
NDK="${NDK:-$ANDROID_HOME/ndk/27.2.12479018/toolchains/llvm/prebuilt/linux-x86_64/bin}"
HERE="$(cd "$(dirname "$0")" && pwd)"
QS="$HERE/../htp_exploration/qnn_shell"
E2E="$HERE/../e2e_pipeline"
B="$HERE/native/out"
J="$HERE/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$B" "$J"
[ -f "$QS/libs/libQnnHtp.so" ] || "$QS/fetch_libs.sh"
# skels + stubs (e2e build.sh builds its own driver too; OUT/IMGS are unused with BUILD_ONLY)
BUILD="$B/e2e" BUILD_ONLY=1 OUT=/nonexistent IMGS=/nonexistent NDK="$NDK" "$E2E/build.sh"
INC=(-I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc"
     -I "$B/e2e/rpn_fused" -I "$B/e2e/roi" -I "$B/e2e/roiu8")
"$NDK/aarch64-linux-android29-clang++" -O2 -std=c++17 -shared -fPIC -static-libstdc++ -I "$QS/headers" "${INC[@]}" \
  -o "$J/libmaskrcnn_demo.so" "$HERE/native/maskrcnn_engine.cpp" \
  "$B/e2e/rpn_glue.o" "$B/e2e/rpn_stub.o" "$B/e2e/roi_stub.o" "$B/e2e/roiu8_stub.o" \
  -L "$QS/libs" -lonnxruntime -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc \
  -ljnigraphics -llog -Wl,--no-undefined
# YOLO, SAM and super-resolution modes: one engine library each (ORT + QNN EP only, no DSP skels)
for e in yolo sam sr; do
  "$NDK/aarch64-linux-android29-clang++" -O2 -std=c++17 -shared -fPIC -static-libstdc++ -I "$QS/headers" \
    -o "$J/lib${e}_demo.so" "$HERE/native/${e}_engine.cpp" -L "$QS/libs" -lonnxruntime -ljnigraphics -llog \
    -Wl,--no-undefined
done
# MCC 3D mode's DSP decoder (../mcc_hmx): its skel + stub, relinked into libmcc_demo.so (the loop above
# built the QNN-only variant; this one replaces it)
OUT="$B/mcc_hmx" NDK_CLANG="$NDK/aarch64-linux-android29-clang" "$HERE/../mcc_hmx/build.sh" >/dev/null
"$NDK/aarch64-linux-android29-clang++" -O2 -std=c++17 -shared -fPIC -static-libstdc++ -I "$QS/headers" -I "$B/mcc_hmx" -I "$HERE/../mcc_hmx" \
  "${INC[@]}" -o "$J/libmcc_demo.so" "$HERE/native/mcc_engine.cpp" "$B/mcc_hmx/mcc_hmx_stub.o" -L "$QS/libs" -lonnxruntime \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc -ljnigraphics -llog -Wl,--no-undefined
cp "$B/mcc_hmx/mcc_hmx_rpc.so" "$J/libmcc_hmx_rpc.so"
# Game-upscaling mode: NSS + NFRU (../vision_models/{nss,nfru}); their OpenCL kernel sources are compiled
# in as strings (game_cl.h), the vendor libOpenCL.so is dlopen'ed at run time (../vision_models/nss/cl_dl.h;
# CL_HEADERS: the Khronos OpenCL headers, default /usr/include)
VM="$HERE/../vision_models"
python3 - "$VM/nss/nss_kernels.cl" "$VM/nfru/nfru_kernels.cl" > "$B/game_cl.h" <<'PY'
import sys
for name, path in (("NSS_CL", sys.argv[1]), ("NFRU_CL", sys.argv[2])):
    src = open(path).read()
    assert ")CLSRC\"" not in src
    print(f'static const char {name}[] = R"CLSRC({src})CLSRC";')
PY
"$NDK/aarch64-linux-android29-clang++" -O2 -std=c++17 -shared -fPIC -static-libstdc++ -I "$QS/headers" -I "$B" \
  -I "$VM/nss" -I "${CL_HEADERS:-/usr/include}" -o "$J/libgame_demo.so" "$HERE/native/game_engine.cpp" \
  -L "$QS/libs" -lonnxruntime -ljnigraphics -llog -ldl -Wl,--no-undefined
# RT-DETR mode: the MSDA skel + stub (../msda_hvx, BUILD_ONLY) and the engine, which #includes
# ../vision_models/rtdetr/msda_hvx/dec_run.cpp
OUT="$B/msda" BUILD_ONLY=1 NDK_CLANG="$NDK/aarch64-linux-android29-clang" "$HERE/../msda_hvx/build.sh"
"$NDK/aarch64-linux-android29-clang++" -O2 -std=c++17 -shared -fPIC -static-libstdc++ -I "$QS/headers" \
  -I "$B/msda" -I "$HERE/../msda_hvx" "${INC[@]}" -o "$J/librtdetr_demo.so" "$HERE/native/rtdetr_engine.cpp" \
  "$B/msda/msda_stub.o" -L "$QS/libs" -lonnxruntime -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" \
  -lcdsprpc -ljnigraphics -llog -Wl,--no-undefined
cp "$B/msda/msda_rpc.so" "$J/libmsda_rpc.so"
cp "$QS"/libs/*.so "$J/"
cp "$B/e2e/rpn_fused/rpn_rpc.so" "$J/librpn_rpc.so"      # jniLibs must be lib*.so to be extracted
cp "$B/e2e/roi/roialign_rpc.so" "$J/libroialign_rpc.so"
cp "$B/e2e/roiu8/roialign_u8_rpc.so" "$J/libroialign_u8_rpc.so"
printf 'sdk.dir=%s\n' "$ANDROID_HOME" > "$HERE/local.properties"
GRADLE="${GRADLE:-$(ls -d "$HOME"/.gradle/wrapper/dists/gradle-*-bin/*/gradle-*/bin/gradle 2>/dev/null | tail -1)}"
ANDROID_HOME="$ANDROID_HOME" "${GRADLE:-gradle}" --offline --no-daemon --max-workers=4 -q -p "$HERE" assembleDebug
ls -la "$HERE/app/build/outputs/apk/debug/"*.apk
