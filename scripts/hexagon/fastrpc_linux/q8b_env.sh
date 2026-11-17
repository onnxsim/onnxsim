# Source this file on the Q8B after installing the SDK under ~/opt.
export HEXAGON_SDK_ROOT="$HOME/opt/hexagon-sdk/Hexagon_SDK/6.4.0.2"
export HEXAGON_TOOLCHAIN="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04/Tools"
export HEXAGON_TOOLS="$HEXAGON_TOOLCHAIN"
export X86_SYSROOT="$HOME/opt/x86-sysroot"
export QEMU_X86=/usr/bin/qemu-x86_64
export QEMU_LD_PREFIX="$X86_SYSROOT"
export HEX_ARCH=v68
export HEXAGON_ARCH_VERSION=68
export HEXAGON_MLIR_ROOT="$HOME/opt/hexagon-mlir"
export HEXKL_ROOT="$HOME/opt/hexkl/hexkl_addon"

# SDK tools are x86-64 executables; keep their library path local to QEMU.
hexagon_clang() {
  LD_LIBRARY_PATH="$HEXAGON_TOOLCHAIN/lib" "$QEMU_X86" -L "$X86_SYSROOT" \
    "$HEXAGON_TOOLCHAIN/bin/hexagon-clang" "$@"
}
hexagon_qaic() {
  "$QEMU_X86" -L "$X86_SYSROOT" "$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" "$@"
}
