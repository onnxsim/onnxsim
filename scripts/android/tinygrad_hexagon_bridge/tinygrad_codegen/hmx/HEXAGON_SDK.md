# Where the Hexagon SDK actually is

```bash
export HEXAGON_SDK_ROOT=/mnt/data/cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2   # 6.0 GB, build 1, extracted Oct 2025
export HEXAGON_TOOLCHAIN=$HOME/.cache/hexagon-oa-19/Tools
export NDK_CLANG=/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang
```

**No download, no account, no install.** It is already on this machine.

## The thing worth not repeating

It is *not* under `/` in any usual place and *not* under `~/.cache` - it lives on the
separate `/mnt/data` volume inside a TVM cache tree. A `find / -xdev` will miss it, and so
will searching `/opt/qcom` (which has the open-access *toolchain*, a different product).

What convinced me it was the right one: rebuilding the whole graph skel from scratch with it
reproduced the earlier session's `tg_hmx_rpc.h`, `tg_hmx_rpc_skel.c`, `tg_hmx_rpc_stub.c` and
the skel object at **byte-identical MD5s**. Same SDK, same inputs, same outputs.

`qaic` here is the SDK's copy and runs natively:

```
$ $HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic --help
qaic - Qaic's Another Idl Compiler
```

## What it provides that the open-access toolchain does not

| path | what |
|---|---|
| `ipc/fastrpc/qaic/Ubuntu/qaic` | the IDL compiler (stub + skel + header generation) |
| `incs`, `incs/stddef` | `AEEStdDef.idl`, `remote.h` |
| `rtos/qurt/computev69/include/{qurt,posix}` | **QuRT** - `qurt.h`, `qurt_hvx_lock` |
| `ipc/fastrpc/rpcmem/inc/rpcmem.h` | `dmahandle` / rpcmem |
| `ipc/fastrpc/remote/ship/android_aarch64` | `libcdsprpc.so` for the AP-side client |

The QuRT row is the one that matters. The open-access toolchain is cross-compilation-only and
carries **no** `qurt.h` and no `HAP_power.h` / `HAP_compute_res.h` - and those are exactly what
`hmx_runtime.h` includes. So a Hexagon SDK is genuinely required to build a skel; it just
happens to be here.

## The NDK

The AP-side client needs an aarch64 NDK clang plus `-lcdsprpc` from the SDK's
`remote/ship/android_aarch64`. `aarch64-linux-android29-clang` is in the system NDK.

## About qaic being open source

`qaic` is published separately at https://github.com/qualcomm/QAIC (BSD-3, builds with
GHC 9.10.3 + Cabal) because "the IDL version supported in the SDK is tuned to describe
specifically the interface between the application processor and the Hexagon DSPs". That is
worth knowing for reproducibility, but it does **not** replace the SDK: the QuRT and HAP
headers still have to come from somewhere, and the only source is the SDK. Building qaic from
source would get us the IDL step and nothing else.

hexagon-mlir is not a route either - it is an AI-compiler stack for the Hexagon **NPU**
(Triton/PyTorch), and its own build documentation lists the Hexagon SDK as an input, so it
adds an MLIR/LLIR build on top rather than removing the dependency.
