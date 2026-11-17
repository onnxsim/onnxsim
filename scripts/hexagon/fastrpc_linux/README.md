# Hexagon FastRPC on Linux: reproduction and elementwise extension

`build.sh` rebuilds a small FastRPC service for the cDSP on a Linux aarch64 host
(Radxa Dragon Q8B / SC8280XP) and runs its host test on the device. The
interface is `ew_rpc.idl`:

| method | purpose |
| --- | --- |
| `run_add(a, b)` | smoke test: scalar add on the DSP |
| `elementwise(op, a, b, c)` | float32 elementwise over byte payloads. `op` 0..7 = relu, add, sub, mul, max, min, abs, neg; unary ops ignore `b` |

`ew_client.c` checks every op against the same expression on the host, bit for
bit, for sizes 1, 7, 1024 and 65537, and checks that bad op codes and length
mismatches are rejected.

## Real model: MNIST-8 on the cDSP

`onnx_dsp_run.py` runs an ONNX graph node by node. Conv, MaxPool, MatMul, Add, and Relu
go to the DSP. Reshape and broadcasting stay on the host. Every DSP node output is compared
with onnxruntime on the same graph, and the final output is compared with onnxruntime on
the original model.

```sh
./build.sh                                            # builds out/ and runs ew_client
mkdir -p models && curl -fsSL -o models/mnist-8.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx
ADSP_LIBRARY_PATH=$PWD/out python onnx_dsp_run.py models/mnist-8.onnx
```

The driver needs `numpy`, `onnx`, and `onnxruntime`. Add `--simplified` to run an onnxsim
output on the DSP instead of the original graph.

Passing run (`mnist-8`, random input, seed 0): all 12 DSP nodes match onnxruntime exactly
(max |err| = 0), and the final output matches with the same argmax.

## Run

```sh
./build.sh                 # builds into ./out and runs ew_client on the cDSP
RUN_ON_DEVICE=0 ./build.sh # build only
```

## Prerequisites

### Q8B installation (2026-11-17)

The Radxa Dragon Q8B was validated with the official SDK 6.4.0.2 and its
Hexagon Tools 19.0.04 under `~/opt/hexagon-sdk/Hexagon_SDK/6.4.0.2`.
Copy `q8b_env.sh` to `~/opt/hexagon-env.sh`, then source it before working:

```sh
source ~/opt/hexagon-env.sh
hexagon_clang --version
cd ~/build/onnxsim/scripts/hexagon/fastrpc_linux
bash build.sh
```

Validation passed on the physical cDSP: `run_add(40, 2) = 42`, all eight
elementwise operations at lengths 1, 7, 1024 and 65537 were bit-exact, and
both invalid-op and mismatched-length checks were rejected as expected.
The build log is `~/opt/hexagon-validation/fastrpc-build.log`.
The existing tinygrad-generated matmul also rebuilt and passed 20/20 DSP
trials within `1e-5` (worst absolute error `4.77e-7`); see
`~/opt/hexagon-validation/tinygrad-matmul.log`.

The SDK came from Qualcomm's official download URL:
`https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/6.4.0.2/Hexagon_SDK_lnx.zip`.
The GET download worked even though a HEAD request returned HTTP 403.
`unzip -tq` verified the archive; `unzip -q` preserved its symlinks and
executable permissions.

QEMU user emulation and binfmt support run the SDK's x86-64 executables on
the ARM64 board. The Ubuntu Noble amd64 sysroot was created with:

```sh
sudo apt-get install qemu-user qemu-user-binfmt debootstrap patchelf unzip ninja-build cmake
sudo debootstrap --arch=amd64 --foreign --variant=minbase \
  --include=libatomic1,libstdc++6,libgcc-s1 noble ~/opt/x86-sysroot \
  http://archive.ubuntu.com/ubuntu
sudo chroot ~/opt/x86-sysroot /debootstrap/debootstrap --second-stage
```

Native LLVM/Clang 21 and `mlir-21-tools` are also installed. Native Clang
successfully emits a Hexagon v68 object (`clang-21 --target=hexagon
-mcpu=hexagonv68`); `mlir-opt-21` runs on ARM64.

Qualcomm's **hexagon-mlir sources**, at commit
`9ec15e80d9d53aa7fb97654da00bfd7047a77ead`, are prepared under
`~/opt/hexagon-mlir`, with the pinned Triton and triton-shared checkouts and
Qualcomm patches. This is a source checkout, not an installed Hexagon-MLIR
compiler. Its current build scripts assume x86-64 and default to v75.
Upstream documents v73/v75/v79/v81 as tested targets. HexKL 1.0 beta 2 is
extracted under `~/opt/hexkl/hexkl_addon`, but contains no v68 DSP library;
using the current Hexagon-MLIR runtime on this Q8B requires a port.
The SDK's v68 C/FastRPC workflow remains the path for tinygrad-generated
kernels on this board.

Sources: [Hexagon-MLIR build instructions](https://github.com/qualcomm/hexagon-mlir/blob/9ec15e80d9d53aa7fb97654da00bfd7047a77ead/docs/user-guide.md),
[build script](https://github.com/qualcomm/hexagon-mlir/blob/9ec15e80d9d53aa7fb97654da00bfd7047a77ead/scripts/build_hexagon_mlir.sh).

- Hexagon SDK 6.4.0.2 extracted (`HEXAGON_SDK_ROOT`). The SDK zip extracted with
  Python's `zipfile` loses symlinks and execute bits; restore them from the zip
  metadata before running the tools.
- Hexagon Tools 19.0.04 inside the SDK (`HEXAGON_TOOLCHAIN`).
- `qemu-user` (provides `qemu-x86_64`) and an amd64 root at `X86_SYSROOT`,
  built with `debootstrap --arch=amd64`. Install `libatomic1` into it, since
  `hexagon-clang` needs `libatomic.so.1`.
- The cDSP skel is unsigned. `ew_client` enables unsigned modules for the cDSP
  domain with `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)`. That is
  fine for development, not for production.

## Notes

- `sequence<int32>` parameters reach the skel as byte lengths, so the skel checks
  `pLen == 4 * count`.
- Conv `auto_pad` (SAME_UPPER/SAME_LOWER) is resolved on the host into explicit pads, since
  MNIST uses it.

- The DSP side uses scalar float. The output is compared bit for bit, so any
  difference from host IEEE arithmetic shows up as a failure.
- Output length is fixed by the caller (`rout` sequence capacity). The DSP
  requires `len(c) == len(a)`.
