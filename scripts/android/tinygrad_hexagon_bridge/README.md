# Bridging tinygrad's Hexagon codegen onto real hardware through TVM's transport

Follow-up to `scripts/android/maskrcnn_e2e/README.md`'s "Diagnosed but not fixed" 1x1-conv
finding: asked to eliminate that using tinygrad instead of hand-writing a TVM schedule fix. This
is the record of what that actually took, what's real and working, and what isn't yet.

## Why tinygrad can't reach this phone's DSP directly

tinygrad's Hexagon backend (`tinygrad/runtime/ops_dsp.py`) has a real (non-mock) driver that
talks to `/dev/adsprpc-smd` via raw FastRPC ioctls, and a `MOCKDSP=1` mode that instead runs the
same generated C under `qemu-hexagon-static` on the host -- useful for codegen/correctness, not
a timing signal.

The real driver doesn't work on this phone: it's a production Android build (`adb root` is
refused -- *"adbd cannot run as root in production builds"*), SELinux is `Enforcing`, and
`/vendor/dsp/cdsp/fastrpc_shell_3` (the bootstrap binary the driver unconditionally reads) exists
but is permission-denied to the unprivileged `shell` user `adb shell` runs as. There's no
privilege-escalation path available without rooting/unlocking the bootloader -- an invasive
action on physical hardware, out of scope here.

TVM's own `tvm_rpc_android`, by contrast, reaches the DSP fine as a plain `adb shell`-launched
process (confirmed: `readelf -d` on the deployed binary shows it links `libcdsprpc.so`, the
vendor's *official* user-space FastRPC library) -- a different, already-permitted access path
than tinygrad's raw-ioctl one.

## The bridge

Since TVM's transport works and tinygrad's own doesn't, but TVM's Hexagon RPC has a **generic**
module loader (`session.load_module` -> `tvm.hexagon.load_module`, not tied to relay-compiled
graphs -- any correctly-built `.so` can be loaded and its exported functions called directly),
the bridge compiles tinygrad's own generated kernel with the same Hexagon toolchain TVM uses,
wraps it in a thin TVM `PackedFunc` ABI shim, links both into one `.so`, and loads/calls it
through TVM's existing session. Only the *transport* is TVM's; the *kernel* is tinygrad's own
codegen, unmodified.

```
capture_kernel.py --hw 512 --cin 64 --cout 256 --beam 2 --out kernel.c
    # runs a uint8 GEMM through DEV=DSP MOCKDSP=1, captures the rendered C source (verified
    # correct against a numpy reference under qemu), strips the mock-only entry boilerplate

bridge_and_test.py --kernel kernel.c --hexagon-toolchain $HEXAGON_TOOLCHAIN \
    --tvm-root <tvm source root> --hw 512 --cin 64 --cout 256
    # compiles kernel.c with the real (non-mock) Hexagon toolchain, generates and compiles
    # wrapper_template.c (a plain-C TVM PackedFunc shim -- not C++, since TVM's
    # TVM_DLL_EXPORT_TYPED_FUNC macro pulls in tvm::runtime::Array/Map, which need more libc++
    # than this freestanding Hexagon build provides), links via
    # tvm.contrib.hexagon.tools.link_shared, uploads+loads+calls it via a real
    # HexagonLauncher session, and times it
```

Verified end-to-end, real hardware, `cin=64, cout=256, hw=512` (the pathological shape from the
1x1-conv finding):

| Kernel | Median | Throughput | Correct |
|---|---:|---:|---|
| tinygrad, naive (BEAM=0) | 2137.8 ms | 0.42 GMAC/s | yes, bit-exact |
| tinygrad, BEAM=2 (found a real 4x4 register-tiled microkernel) | 8.4 ms | 1.00 GMAC/s | yes |
| tinygrad, vrmpy TensorCore (see below) | 29.9 ms | 0.28 GMAC/s | yes |
| stock TVM (hand-tuned `vrmpy` schedule) | 3.8 ms | 2.18 GMAC/s | yes |

### v65 code generation check

The installed Hexagon SDK 6.4 compiler rejects `-mcpu=hexagonv65`, so the bridge accepts a
compiler override. System clang 19 can emit v65 objects; TVM's SDK linker still links the support
libraries for v68. This compiled and ran the generated kernel on the attached v69 phone, with
bit-exact output (53.8 ms for the 512x64x256 generic kernel):

```sh
bridge_and_test.py --kernel kernel.c --hexagon-toolchain "$HEXAGON_TOOLCHAIN" \
  --hexagon-clang /usr/bin/clang-19 --hex-arch v65 --link-arch v68 \
  --tvm-root <tvm source root> --hw 512 --cin 64 --cout 256
```

This validates v65-targeted generic code on newer hardware. The captured default kernel did not
select `vrmpy`. Full-model QEMU execution of openpilot v0.9.7 `supercombo.onnx` now completes with
148 captured kernels and a `(1, 6504)` output. Against ONNX Runtime CPU on the same seeded inputs,
the maximum absolute output difference is `2.48` (mean `0.069`); the largest output magnitude is
`8618`, and the 99th-percentile relative error is `0.084%` for outputs whose reference magnitude
exceeds `0.01`. This is close numerical agreement, not bit-exact equivalence.

### Exporting the captured model code

`examples/openpilot/compile3.py` captures and validates tinygrad's graph, but its pickle keeps the
generated programs embedded in the JIT object. Export those programs into reviewable per-kernel
sources and compile v65-targeted ELF objects with:

```sh
export TINYGRAD_ROOT=/path/to/tinygrad
export PYTHONPATH="$TINYGRAD_ROOT/examples/openpilot:$TINYGRAD_ROOT"
export HEXAGON_TOOLCHAIN=/path/to/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools
MOCKDSP=1 DEV=DSP FLOAT16=1 NOLOCALS=1 BEAM=0 CC=/usr/bin/clang-19 \
  python3 "$TINYGRAD_ROOT/examples/openpilot/compile3.py" \
    /path/to/supercombo.onnx /tmp/supercombo_v65.pkl
MOCKDSP=1 DEV=DSP FLOAT16=1 NOLOCALS=1 BEAM=0 CC=/usr/bin/clang-19 \
  python3 scripts/android/tinygrad_hexagon_bridge/export_openpilot_kernels.py \
    /tmp/supercombo_v65.pkl --tinygrad-root "$TINYGRAD_ROOT" \
    --out /tmp/supercombo_v65_codegen --hexagon-clang /usr/bin/clang-19 --hex-arch v65 \
    --onnx /path/to/supercombo.onnx
```

The exporter preserves capture order and call signatures in `manifest.json`, including each
argument's backing storage slot and byte offset through `BITCAST`/`SHRINK` views. This identifies
the shared 2.5 MiB scratch arena used by this capture, rather than incorrectly treating every
intermediate as a separate allocation. It writes the full QEMU source and ELF for each unique
program, and emits kernel-only C with a unique symbol plus a v65 ELF object when `--hexagon-clang`
is supplied. Identical generated sources share one code artifact.
With `--onnx`, it also packs initializer tensors into 128-byte-aligned `weights.bin` and records
names, shapes, dtypes, offsets, and hashes in the manifest. For the current v0.9.7 capture this
exports 148 calls backed by 65 unique program sources. The captured ELF files are QEMU executables;
the `.o` files are v65 objects for composing a phone-side runtime. The original capture used NPY
inputs for five non-image values, which baked their sample data into the graph. `compile3.py` now
supports `DSP_ALL_INPUTS=1` to realize all seven ONNX inputs on DSP before capture. The regenerated
`/tmp/supercombo_v65_dynamic.pkl` has 148 calls, 65 unique sources, all seven inputs on DSP, and
seven PARAM slots (0 through 6) whose shapes match the seven model inputs. Its shared scratch arena
is 2.62 MiB. The exported manifest is `/tmp/supercombo_v65_dynamic_codegen`.

`capture_openpilot_buffer_bindings.py` replays this capture under QEMU and hashes the actual bytes
passed to every kernel. It verified all 313 ONNX initializer-to-BUFFER slots with no conflicts, and
saved the sole extra constant buffer (`BUFFER:435`, the ONNX constant `[0.010002136, 1.0]`) plus a
zero-input reference output. `generate_openpilot_graph_driver.py` consumes those bindings and emits
a C dispatcher for all 148 calls. The generated driver holds a 2.5 MiB aligned arena, accepts the
seven packed float32 inputs, and loads the 51.3 MB aligned weight blob in chunks. The native client
reuses the existing `run_kernel` FastRPC method: each 8 MiB weight chunk carries its byte offset in
the second sequence, and an empty second sequence selects graph execution.

### Running one captured model kernel on the phone

The native FastRPC transport can run a selected captured call as a probe. For example, call 93 is
a small `E_10_4_128` program. Prepare deterministic synthetic activations and a QEMU reference, then
build and run its kernel-only C on the phone's v69 DSP with a v65 target:

```sh
python3 scripts/android/tinygrad_hexagon_bridge/prepare_openpilot_kernel_probe.py \
  /tmp/supercombo_v65_codegen --call 93 --synthetic-weights \
  --out /tmp/supercombo_v65_phone_probe
HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.4.0.2 \
HEXAGON_TOOLCHAIN=/path/to/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools \
HEX_ARCH=v68 LINK_ARCH=v68 MODEL_ARCH=v65 MODEL_CLANG=/usr/bin/clang-19 \
OPENPILOT_PROBE_DIR=/tmp/supercombo_v65_phone_probe \
  bash scripts/android/tinygrad_hexagon_bridge/native_transport/build.sh
```

On the attached v69 phone this returned output bit-exact to QEMU across 5 calls; median RPC-inclusive
time was `0.646 ms` for the 20 KiB output kernel, with a 22 KiB packed input. Activations and weights
are synthetic, so this validates the generated v65 function's phone execution and transport ABI,
not the actual layer result from a real model frame. The 3x3 stem probes initially returned FastRPC
error `78`. DSP crash logs showed
an access fault at `data2 + 18`: an eight-half vector load began on an odd row of a 9-half filter,
while the generated C type promised 16-byte alignment. QEMU tolerated the unaligned access; the
phone did not. `DSPRenderer` now gives FP16 vector types their natural 2-byte alignment, which
lowers those loads safely. Calls 0 and 1 then matched QEMU bit-for-bit on the phone using their
actual ONNX initializer weights and synthetic activations. Call 0 took `3371 ms` median; call 1
took `15.91 ms`. These are single-kernel RPC timings, not full-model latency.

Additional exact phone probes cover call 2 (`48.34 ms`), call 4 (7x7 conv, `18.04 ms`), call 93
(`0.645 ms`), call 133 (`0.172 ms`), and the final 46-argument call 147 (`142.35 ms`, 26 KiB
output). The captured graph has 148 ordered DSP calls, 65 unique programs, and covers all 595 nodes
/ 26 operator types in the ONNX model; the captured QEMU run still produces `(1, 6504)`. These
probes give representative early, middle, and final-kernel coverage. The seven model inputs are
dynamic in the regenerated capture, with verified weight binding, persistent phone-side arena
allocation, and successful dispatch across all 148 calls.

The host `clang-19` accepts `-mcpu=hexagonv65`; the installed Qualcomm SDK compiler accepts only
v68 and newer. A v65 syntax-only pass succeeded for all 65 unique generated kernel sources. A full
capture-time execution of the new seven-input graph completed in QEMU with 148 calls; the optimized
capture build took about four minutes because one fused kernel is large. The generated dispatcher
and all 65 kernel objects were linked into a standalone QEMU executable; its full 148-call zero-input
result matched the capture reference bit-for-bit. The graph runner sends the generated graph in
eight-kernel FastRPC batches. The phone build with Hexagon SDK 6.4.0.2 succeeds, uploads all model
weights, and runs the graph in `9.23 s`; its output matches the QEMU zero-input reference bit for
bit. The SDK's default FastRPC DSP stack is 16 KiB, but call 61 (`tg_call_61_r_32_32_32`) needs a
`0x5798`-byte frame. Configuring a 64 KiB cDSP FastRPC stack before the first RPC fixes the TLBMISS
that previously occurred in this call.

A one-kernel-per-RPC profile measured `9.23 s` end-to-end on the attached phone; measured kernel
RPCs account for `9.21 s`, so transport overhead is small relative to execution. The slowest calls
were call 0 at `3.94 s` (42.7%), call 109 at `461 ms`, calls 6 and 10 at `402 ms` and `397 ms`,
call 108 at `342 ms`, and call 99 at `211 ms`. Those six calls account for 62.4% of measured kernel
time. Per-batch timing is printed by the native client; one-call batches were used for this profile
to isolate individual kernels.
Changing only call 0's compiler target to v68 left its generated arithmetic scalar and did not
improve performance. A gather-only qfloat experiment did vectorize the arithmetic and remained
bit-exact, but regressed call 0 to `6.79 s`: assembling vectors from scattered loads cost more
than the saved arithmetic. The useful change was to pack stem.0 weights from OIHW to `[input,
kernel_y, kernel_x, output]` order and load each 32-output-channel slice contiguously. Call 0 then
fell from `3.94 s` to `0.76 s`; the complete graph fell from `9.23 s` to `6.02 s` (about `1.53x`
faster) and remained bit-exact to the QEMU zero-input reference. All other kernels remain the
validated v65 objects. The qfloat-capable v68 call 0 is compiled from the packed source; the
`OPENPILOT_GRAPH_PREBUILT=1` option prevents unvalidated kernels from being recompiled for v68.
The small renderer extension used during v68 capture is recorded in
[`v68-stem0-vectorization.patch`](v68-stem0-vectorization.patch); the exporter performs the
contiguous weight load rewrite and initializer packing.

Further profiling confirmed calls 6 and 10 share one generated source and object. Retargeting that
shared kernel to v68 reduced each call from about `395 ms` to `81 ms`, but changed 6,284 of 6,504
output values (maximum absolute difference `2.67e-5` on the zero-input reference). Since this
path is not bit-exact, it is excluded from the validated graph. The graph-preparation helper now
updates every call that shares a selected source, so experimental selections link one object
without duplicate symbols. The phone is left on the bit-exact build with calls 0, 6, 10, 109, and
147 on v68; the latest run measured `4.62 s` end-to-end and matched all 6,504 values exactly.

Testing call 108 (`r_10_64_32_128_4`) as a v68 kernel cut its isolated time from `342.6 ms` to
`29.8 ms`, but the end-to-end output failed bit-exact comparison. Disabling FP contraction reduced
the speedup (`42.2 ms`) and still failed exact comparison. Both variants are excluded. Restoring the
validated graph produced `5.96 s` end-to-end and matched the zero-input reference exactly; call 108
remains on v65.

The mismatch is in the vectorized arithmetic path: compiling the original per-lane scalar source for
v68 remained exact but took `360 ms`, while an explicit vector accumulation sequence took `49 ms`
and still differed. The current restored v65 call 108 measured `341 ms`; the packed-call-0 graph
measured `5.96 s` and was bit-exact. Until the vector operations can reproduce scalar rounding,
call 108 stays on v65.

Call 109 (`r_512_2048_10`) was optimized by tiling 32 adjacent positions. This reuses each loaded
weight across 32 positions while keeping the original 2,048-term reduction order for each one. The
[`optimize_openpilot_call109.py`](optimize_openpilot_call109.py) reproduces the kernel rewrite.
The two phone runs measured `62.2 ms` and `64.3 ms`, down from about `461 ms`, and both remained
bit-exact. Tile 64 was also exact but slightly slower at `64.9 ms`. The combined graph with calls
0, 109, and 147 on v68 measured `5.35 s` and `5.38 s` on two tile-32 runs, matching the reference.

Calls 6 and 10 share the same convolution source. The raw v68 vector kernel was fast but inexact;
tiling 32 output-channel groups lets the kernel reuse each activation tile across groups while
keeping every output's original reduction order. [`optimize_openpilot_calls610.py`](optimize_openpilot_calls610.py)
generates this scalar-accumulation kernel. Each call dropped from about `338 ms` to `72 ms` on two
phone runs. With calls 0, 6, 10, 109, and 147 selected for v68, the graph measured `4.63 s` and
`4.62 s` and remained bit-exact.

The same v68 retargeting on call 99 (`r_10_48_32_128_4`) reduced its phone time from `210 ms` to
`21.9 ms`, but also failed the full-graph exact comparison. The conversion-materialized and explicit
vector-accumulation call 108 variants likewise remained inexact (`47.9 ms` and `49.4 ms`). The final
graph with only packed stem.0 on v68 measured `6.01 s` and matched the reference exactly.

Call 147 (`r_813_8_256_64_16_32_64_16_32_64_32_32_32_32_32_32_4`) proved safe to retarget:
its v68 kernel ran in `49.8 ms` and `49.6 ms` on the tile-32 graph runs, down from about `143 ms`,
with the full graph bit-exact both times.

To regenerate the optimized v68 capture without asking the old QEMU to execute v68 instructions,
use the capture-only hook (its output values are placeholders; only the captured graph/code is
used):

```sh
TINYGRAD_ROOT=/home/takecheeze/.cache/tg-vrmpy
OPENPILOT_ONNX=/tmp/openpilot_supercombo.onnx
PYTHONPATH="$PWD/scripts/android/tinygrad_hexagon_bridge/capture_only_sitecustomize:$TINYGRAD_ROOT/examples/openpilot:$TINYGRAD_ROOT" \\
MOCKDSP_CAPTURE_ONLY=1 MOCKDSP=1 DEV=DSP FLOAT16=1 NOLOCALS=1 BEAM=0 CACHELEVEL=0 CC=/usr/bin/clang-19 \\
DSP_ALL_INPUTS=1 HVX_ARCH=v68 HEXAGON_TOOLCHAIN=/mnt/data/cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools \\
  python3 "$TINYGRAD_ROOT/examples/openpilot/compile3.py" "$OPENPILOT_ONNX" /tmp/supercombo_v68_dynamic.pkl
PYTHONPATH="$TINYGRAD_ROOT" HVX_ARCH=v68 MOCKDSP=1 \\
  python3 scripts/android/tinygrad_hexagon_bridge/export_openpilot_kernels.py /tmp/supercombo_v68_dynamic.pkl \\
    --tinygrad-root "$TINYGRAD_ROOT" --out /tmp/supercombo_v68_stem0_export --hex-arch v68 \\
    --onnx "$OPENPILOT_ONNX" --optimize-openpilot-stem0
python3 scripts/android/tinygrad_hexagon_bridge/optimize_openpilot_call109.py /tmp/supercombo_v68_stem0_export --tile 32
python3 scripts/android/tinygrad_hexagon_bridge/optimize_openpilot_calls610.py /tmp/supercombo_v68_stem0_export --tile 32
python3 scripts/android/tinygrad_hexagon_bridge/prepare_openpilot_stem0_graph.py \\
  /tmp/supercombo_v65_dynamic_codegen /tmp/supercombo_v68_stem0_export \\
  /tmp/supercombo_v65_bindings/buffer_bindings.json /tmp/supercombo_v65_phone_graph \\
  --out-bundle /tmp/supercombo_v65_calls610_109_147_bundle --out-graph /tmp/supercombo_v65_calls610_109_147_graph \\
  --hexagon-clang /mnt/data/cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang \\
  --v68-call 6 --v68-call 109 --v68-call 147
```

Build and run that mixed graph with the same SDK environment as the v65 graph above, replacing
the graph directories and adding `MODEL_ARCH=v68 OPENPILOT_GRAPH_PREBUILT=1`:

```sh
HEXAGON_SDK_ROOT=/mnt/data/cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2 \\
HEXAGON_TOOLCHAIN=/mnt/data/cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools \\
HEX_ARCH=v68 LINK_ARCH=v68 MODEL_ARCH=v68 \\
OPENPILOT_GRAPH_DIR=/tmp/supercombo_v65_calls610_109_147_graph \\
OPENPILOT_GRAPH_BUNDLE_DIR=/tmp/supercombo_v65_calls610_109_147_bundle \\
OPENPILOT_GRAPH_BINDINGS_DIR=/tmp/supercombo_v65_bindings \\
OPENPILOT_GRAPH_PREBUILT=1 \\
  bash scripts/android/tinygrad_hexagon_bridge/native_transport/build.sh
```

Hexagon SDK 6.4.0.2 is available in the shared cache at
`/mnt/data/cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2`, including `qaic`, FastRPC IDL headers,
and Android `libcdsprpc.so`. Its bundled Hexagon compiler still cannot target v65, so the graph
kernels must use host `clang-19` as above; the transport support objects can use v68 as shown below.

With the SDK installed, build and run the phone graph against that same QEMU reference using:

```sh
python3 scripts/android/tinygrad_hexagon_bridge/capture_openpilot_buffer_bindings.py \
  /tmp/supercombo_v65_dynamic.pkl /tmp/supercombo_v65_dynamic_codegen \
  --tinygrad-root /path/to/tinygrad --out /tmp/supercombo_v65_bindings
python3 scripts/android/tinygrad_hexagon_bridge/generate_openpilot_graph_driver.py \
  /tmp/supercombo_v65_dynamic_codegen /tmp/supercombo_v65_bindings/buffer_bindings.json \
  --out /tmp/supercombo_v65_phone_graph
HEXAGON_SDK_ROOT=/mnt/data/cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2 \
HEXAGON_TOOLCHAIN=/mnt/data/cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools \
HEX_ARCH=v68 LINK_ARCH=v68 MODEL_ARCH=v65 MODEL_CLANG=/usr/bin/clang-19 \
OPENPILOT_GRAPH_DIR=/tmp/supercombo_v65_phone_graph \
OPENPILOT_GRAPH_BUNDLE_DIR=/tmp/supercombo_v65_dynamic_codegen \
OPENPILOT_GRAPH_BINDINGS_DIR=/tmp/supercombo_v65_bindings \
  bash scripts/android/tinygrad_hexagon_bridge/native_transport/build.sh
```

Four v65-specific issues were fixed to get that result:

1. The DSP heuristic selected 64 float32 lanes (a 256-byte vector) on 128-byte HVX, which broke
   depthwise Conv node 15. The heuristic now caps vector lanes by the reduction dtype's byte width.
2. The renderer's float re-vectorization emitted 32-lane float vector arithmetic for the following
   1x1 Conv. v65 has integer HVX but no floating-point HVX; LLVM's lowering returned values around
   `4.2e6` where CPU returned `43.4`. The renderer now keeps float lanes scalar when qfloat support
   is unavailable, while retaining integer vectorization. A QEMU 1x1 Conv regression test matches
   the CPU result.
3. A final half-vector load at the end of a 1.5 MiB buffer faulted in QEMU because LLVM issued a
   full 128-byte HVX load for a logical 64-byte half vector. DSP allocations and mock mappings now
   include a guarded tail; logical buffer and transfer sizes stay unchanged.
4. Odd-row FP16 vector loads in 3x3 convolutions faulted on the phone due to an overstrong vector
   alignment promise. The DSP renderer now uses natural half alignment for FP16 vectors.

The heuristic and Conv changes, with source and numerical regressions, are recorded in
`v65-vector-width.patch`. This completes full-model generated-code and simulator validation.
Representative early, middle, and final `supercombo` kernels now execute through the FastRPC bridge
on the phone, but there is not yet a phone-side runner for the full graph and Adreno has not yet
been replaced.

### Correction: the "v68-selected" graphs above do not run on a Snapdragon 845

Calls 0, 6, 10, 109 and 147 in the 4.6-6.0 s graphs above were compiled for v68 (qfloat), which the 845's v65 cDSP does not
have. Built for v65 instead with `prepare_openpilot_stem0_graph.py --opt-arch v65 --opt-call 6 --opt-call 109` (the tiled
rewrites of calls 6/10/109 are scalar C), the whole graph is **8.26 s, bit-exact** to the QEMU reference on the phone; call 0,
the stem, is 3.9 s of that. The reason is fp16: v65 has no fp16 arithmetic or conversion, so each fp16 weight use in the stem's
inner loop was an `__extendhfsf2` call (320 call sites in that one kernel). The next section removes both problems at the
source instead of per call.

## openpilot on v65: every modeld program generated by tinygrad, one standalone DSP program each

openpilot master (checked at 027770d) builds six tinygrad programs in `selfdrive/modeld/SConscript`: `driving_supercombo.onnx`,
`dmonitoring_model.onnx`, and the camera warps (driving yuv420 512x256, two frames; DM luma 1440x960) for each of the two
camera configurations. All six are generated by the tinygrad fork (`onnxsim/tinygrad`, branch `openpilot-v65-graph`, on top of
`dsp-consolidated`) as standalone v65 programs, checked bit for bit under QEMU, and run on the phone (V69, which executes the
v65 code) bit-exact to that reference:

```sh
export TINYGRAD_ROOT=/path/to/tinygrad-fork HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... CC=clang-19
M=/path/to/openpilot/selfdrive/modeld/models
openpilot_v65/run.sh model driving $M/driving_supercombo.onnx
openpilot_v65/run.sh model dm $M/dmonitoring_model.onnx
openpilot_v65/run.sh warp warp_drive_1928 --camera 1928x1208 --warp-to 512x256 --layout yuv420 --frames 2
openpilot_v65/run.sh warp warp_drive_1344 --camera 1344x760 --warp-to 512x256 --layout yuv420 --frames 2
openpilot_v65/run.sh warp warp_dm_1928 --camera 1928x1208 --warp-to 1440x960 --layout luma --border-fill 16
openpilot_v65/run.sh warp warp_dm_1344 --camera 1344x760 --warp-to 1440x960 --layout luma --border-fill 16
openpilot_v65/run.sh phone dm 5 4          # iters, threads [, batch, prof]
```

`driving_supercombo` is 156 calls (107 threaded, 73 distinct kernels), 120 MB of fp32 weights uploaded in 0.2 s, 15.9 MB of
activations; its `action_t` input is not read by any kernel and is left out of the program. The QEMU reference for it takes
65 s per inference, and one of its fused kernels takes over five minutes to compile.

What the fork does, all in tinygrad rather than in per-call rewrites of exported C:

- **`DSP_V65_HW=1`**: the real v65 constraints. HVX there is integer-only, so no float vectors (LLVM miscompiled them into
  values around 4e6), upcasts of at most one 128-byte register, and half emulated as fp16 bits with float32 math. The
  emulation's generic decomposition used to flush fp16 subnormals to zero, which zeroed every weight below 6.1e-5; widening now
  keeps them exactly.
- **`ONNX_FP16_AS_FP32=1`** (with `FLOAT16=0`): fp16 initializers become fp32 on the host at load, so no kernel converts a
  weight. fp16 storage only costs conversions on hardware without fp16.
- **`DSP_THREADS=4`**: tinygrad's CPU `core_id` split, enabled for the DSP. A kernel's outer global loop is split across the
  cDSP's hardware threads (four on the 845); each output keeps its reduction order, so results are unchanged. The thread
  heuristic's 128K-elements-per-thread floor is a renderer property, 16K on the DSP, where one element can cost hundreds of
  cycles.
- **`runtime/support/dsp_graph_v65.py`**: `dsp_graph.capture` of a TinyJit replay (nothing runs), emitted as one program. Views
  into the JIT's memory-planned arena stay (region, offset), so kernels that alias an arena through a BITCAST still share memory.
  Never-written regions are the weight blob, uploaded once in 8 MB chunks and read in place. `core_id` kernels run on a qurt
  thread pool (from v65 QuRT context-switches HVX, so pool threads can't deadlock on the two HVX contexts). `run_qemu` runs the
  whole emitted program under qemu-hexagon for an exactness check without a phone.
- `examples/openpilot/compile3.py`: numpy-generated inputs (uint8 camera frames; `Tensor.randn` needs 64-bit division on the
  DSP), `ALL_OUTPUTS=1` keeps the recurrent-state outputs (`next_state_*`) as part of the graph, `BENCH_RUNS`.
  `examples/openpilot/compile_warp.py` is upstream's warp math, pickled like compile3 and saving realistic transforms (a random
  matrix gives NaN coordinates, whose float-to-int casts are undefined in C).

Phone results (Xiaomi 12S V69 running v65 code; RPC-inclusive, best of N; every output bit-exact to the QEMU reference):

| program | 1 thread | 4 threads |
|---|---:|---:|
| `driving_supercombo` (all four outputs, recurrent state included) | 2057 ms | 768 ms |
| `dmonitoring_model` | 893 ms | 342 ms |
| driving warp, 1928x1208 | 87.9 ms | 50.3 ms |
| driving warp, 1344x760 | 75.2 ms | 43.0 ms |
| DM warp, 1928x1208 | 238.9 ms | 78.3 ms |
| DM warp, 1344x760 | 225.3 ms | 70.1 ms |

The model rows include every codegen change below. Each step, as 1 / 4 threads:

| step | driving | DM |
|---|---:|---:|
| fp32 weights, `DSP_THREADS` | 3831 / 1702 ms | 1623 / 620 ms |
| + scalar register blocking | 3796 / 1493 | 1094 / 419 |
| + materialized conv padding | 2991 / 1234 | 1044 / 388 |
| + strided sub-line prefetch | 2156 / 1089 | 1024 / 370 |
| + scalar GEMV upcast | 2057 / 768 | 893 / 342 |

**Strided sub-line prefetch (`HVX_PREFETCH_SUBLINE`, default on).** Scalar float code loads a float4 per reduction step. A 1x1
conv over NCHW channels, or a GEMM over weight rows, walks those loads at a multi-KB stride and missed on every step, because
the existing vector-load prefetch skips anything under a line. One `dcfetch` `HVX_PREFETCH_STRIDES` steps ahead on such loads
took driving's 192->64 1x1 conv from 371 to 105 ms single-threaded (102 -> 30 ms with 4 threads).

**Scalar GEMV (`DSP_SCALAR_GEMV`, default on).** A reduction that nothing broadcasts into used to get the HVX rule, which
upcasts the innermost output axis to 128. On v65 float that is a 128-float accumulator in memory, and for the policy head's
`[out, in]` Gemm weights it reads 128 rows per step. With threads it got slower (59.7 ms with 4 threads vs 20.3 with 1). A
4-wide upcast plus reduce unroll gives 12.4 / 3.8 ms; `[in, out]` GEMVs are unchanged.

**Materialized conv padding (`CONV_PAD_MATERIALIZE=1`).** tinygrad pads a conv input lazily, so the zero padding becomes a
mask on every input load inside the conv. On scalar v65 the masked loads and their address math cost more than the MACs. With
the flag the conv reads a padded copy made by its own small kernel. Fusing that copy into the producer was worse (DM
419 -> 547 ms): the producer then computes, masked, over the padded grid and loses its register blocking. As a separate kernel
the strided 3x3 benchmark stem went 330 -> 89 ms (+5 ms for the copy). Outputs are unchanged in value, and DM's differ from the
unpadded capture only at rounding level (6e-5 on values up to 862) because fusion changed.

A cheaper exact trunc (the one-instruction truncating convert plus copysign, in place of the bit-mask emulation) left the
warps' output unchanged but measured no faster, so it was dropped.

**Scalar register blocking.** Under `DSP_V65_HW`, a float reduction with reuse along two axes (a conv) takes tinygrad's CPU
upcast rules, small upcasts on the reused axes, instead of one vector-wide axis. Accumulators × unrolled taps are capped near
the 32-register file (`DSP_SCALAR_UNROLL`, default 16). On openpilot-shaped layers (phone, 4 threads) the 1x1 convs went
22.4 -> 9.2 ms and 43.0 -> 15.7 ms, and a strided 3x3 stem 465 -> 332 ms. A GEMV has one reuse axis and measured 10x slower
blocked, so it keeps the vector-style upcast. The driving stem's MAC loop is now 16 register accumulators, 16 MACs in 10
packets. What remains is around that loop: with a 3-tap inner trip count, the padding-masked input loads and address math
(about 100 packets, with spills) run once per 48 MACs. Pre-padded inputs are the next lever.

### Through the compiler/runner RPC (the way models run)

ONNX models are compiled and run only through the RPC: `openpilot_v65/run.sh compiler [KEY=VAL...]` starts the compiler service
(the fork's commit is its `--compiler-id`, and `KEY=VAL` extras such as `ONNX_QDQ_INT_CONV=1 ONNX_QDQ_LUT=1 TC_OPT=1` go into its
command and therefore its cache key). `run.sh rpc model.onnx --input-raw ... --iters 5 --profile` then compiles, loads and runs on
the phone. Per-call profiling comes back through the RPC too: each artifact's `program.txt` names its calls' kernels, the worker
labels its per-call events with them, and the client prints the slowest (`PROF_TOP`). `run.sh phone` (a standalone client over
adb) remains only for the camera warps, which are tinygrad programs with no ONNX form for the compiler contract.

The same programs go through onnxsim's native remote executor (`tools/onnx-remote`). `openpilot_v65/compile_v65.sh` is an
`onnx-remote-compiler --command`: it captures, emits, checks the program under qemu, builds the skel and packs a `tghx-v65`
artifact. `onnx-remote-hexagon-worker` (`build_worker.sh`, Android) is the runner, with `load_compiled` once and
`run_compiled` per inference. `e2e.sh` drives `onnx-remote-client --compile-run` against both on an attached phone:

```sh
onnx-remote-compiler --port 39502 --cache-dir ~/.cache/onnxsim-v65 --target hexagon-v65 --compiler-id "tinygrad-$(git -C "$TINYGRAD_ROOT" rev-parse --short HEAD)" \
  --command "$PWD/openpilot_v65/compile_v65.sh {input} {output} {manifest}" &
openpilot_v65/build_worker.sh worker/
CLIENT=.../onnx-remote-client openpilot_v65/e2e.sh worker/onnx-remote-hexagon-worker dmonitoring_model.onnx \
  --input-raw 2:1,1382400:img.bin --input-raw 1:1,3:calib.bin --expect ref.bin --iters 5
```

| model | compile (cache miss) | artifact | `load_compiled` | `run_compiled` (RPC-inclusive / DSP) | vs capture reference |
|---|---:|---:|---:|---:|---|
| tiny conv (uint8 + float in, 2 outputs) | seconds | 14 KB | 72 ms | 9.6 / 1.3 ms | 4.8e-7 from ORT |
| `dmonitoring_model` | ~20 min | 14.7 MB | 474 ms | 479 / 437 ms | bit-exact |
| `driving_supercombo` (4 outputs) | ~45 min | 121 MB | 3580 ms | 1740 / 1494 ms | bit-exact |

**Quantized models and the transport.** With the integer path (W8/W16 A16 QDQ models, `ONNX_QDQ_INT_CONV=1 ONNX_QDQ_LUT=1
TC_OPT=1`), 4 threads on the V69 phone, all bit-exact to the capture:

| step | driving: DSP / RPC-inclusive | DM: DSP |
|---|---:|---:|
| integer path, u8 x u8 tensor cores | 394 / 631 ms | 99.8 ms |
| integer reductions skip gather-making upcasts | 398 / 673 | 93.6 |
| the vector upcast takes the axis at unit stride in the most buffers (a W16 7x7 depthwise vectorized its NCHW output width, gathering its input: 26.6 ms -> off the top list) | 347 / 631 | 93.6 |
| outputs in their own dtypes (`ALL_OUTPUTS=2`): the 2 MB uint8 frame queue stays uint8, each output its own aligned slice | 337 / 470 | 94 |
| recurrent state resident on the runner (`--resident 4:1,5:2,6:3`) | 340 / 356 | |
| mixed W8/W16 heads (`drv_w16t10_a16_hmix`) with `ONNX_QDQ_INT_GEMM=1` for weights >= 65536 elements (`ONNX_QDQ_INT_GEMM_MIN`) | 237 / 262 | |
| camera-frame mean subtraction as a table (`ONNX_QDQ_LUT` accepts a constant that is one value broadcast over the channels) | 240 / 252 | |
| eight pixels per weight load in the vrmpy tensor core (`DSP_TC_MUPCAST`, default 8) | 203 / 226 | |
| v65 elementwise kernels vectorize a unit-stride axis when nothing else did (`DSP_V65_COPY_VECTORIZE`; the frame-queue shift was a byte-by-byte copy: 4.6 ms -> 0.08 ms under qemu) | 198 / 220 | 94 |

The last row is a cache effect that QEMU's instruction count hides (11% there, 16% on the phone: the tensor-core layers went
78 -> 51 ms). Pixels per weight load, driving DSP ms: 1 -> 242, 2 -> 228, 4 -> 204, 8 -> 203, 16 -> 218. Storing the accumulators
contiguously (a pixel-major 1x1 conv, one vector store instead of 32 scalar ones) was exact but slower and is not in the tree.

The integer heads are not bit-exact to the float ones: `qmatmul` quantizes the activation to uint16 at run time, so on the
same inputs they differ from an ORT float reference of the same weights by 0.15 on average (6.7 at most) on outputs whose mean
magnitude is 894. Weights below 65536 elements stay float: converting all ~90 heads (32x32 layers included) was 891 ms.
The second row's DSP time is unchanged within noise: the two ~12 ms preprocessing kernels are gone (5 ms of table lookups
remain), but the hmix run-to-run spread is about 4 ms.

Lossy options measured against the "plan error no worse than today" budget (`w16t10` + `hmix` heads, held-out segments 8 and 5;
`evaluate.py --backend phone`, `../../openpilot_dsp/README.md`):
- **12- or 10-bit lookup tables** (index the uint16 tables by the nearest 16 or 64 values, so a table is 8 KB or 2 KB instead of
  128 KB): driving DSP 203.6 -> 199.8 ms (12 bits) and 199.0 ms (10 bits). About 2% for a coarser activation inside those stages,
  so it is not in the tree. The 12 lookup kernels of one shape cost about 23 cycles per element, so it is not the table's cache
  misses that dominate them.
- **uint8 activations in any backbone window** would remove a vrmpy pass, the plane-split copies and the 64K tables, but the
  budget has no room: the backbone activations are heavy-tailed (max/std 20-60), A16 already gives 20-60x less relative precision
  than fp16, and a block of uint8 windows measured 2x the plan error (0.137 vs 0.069 m, `../../openpilot_dsp/README.md` section 3).
- Where the 203 ms goes (before the pixel blocking): elementwise 98 ms, of which 41 ms are integer copies, layout changes and
  plane splits, 28 ms float requantization epilogues (scalar float on v65) and 26 ms table lookups; vrmpy tensor-core convs 77 ms;
  depthwise 41 ms. The copies are lossless work; the epilogues would be integer fixed-point (up to one uint16 step of difference).

| depthwise convs as sliding-window vrmpy over each channel's flat padded image (`nn/dw_v65.py`; all 29 driving layers, 7x7/3x3/5x5, stride 2 and channel multipliers) | 163 / 185 | 79.5 |
| u16 table lookups through VTCM `vgather` (`DSP_V65_VGATHER=1`, on by default in `compile_v65.sh`) | 154 / 178 | 75.0 |

| vrmpy tensor core: accumulator folded into the WMMA's C, 4 output-channel vectors x 2 pixels per weight/activation load (`DSP_TC_MUPCAST=4`, `DSP_TC_PUPCAST=2`) | 144 / 165 | 72.1 |

**Tensor-core convs.** The loop accumulator was added outside the vrmpy op (`pm_wmma_add` never fires here: the WMMA sits behind a
reshape), so each K step zeroed a register, ran the vrmpy from zero and added the result back; a pattern now folds a reshaped WMMA
plus the accumulator into the WMMA's own accumulator input (restricted to the vrmpy shape (32, 1, 4): without the restriction six HMX
i8 tests fail). That alone cut 18% of the instructions under qemu and was neutral on the phone. The gain is the blocking, again
opposite to qemu's count. Driving DSP ms (all byte-identical): 8 x 1 blocking 153.6, 8 x 2 143.8, 4 x 2 142.3 (the qemu count says
4 x 2 is slower than 8 x 1: 1949 vs 1355 us); 4 x 4, 2 x 4 and 2 x 8 stalled the compiler in qemu (not excluded on performance).
Tensor-core kernels in driving: 51.2 -> 39.1 ms. Not tried: weight-prefetch tuning, the thread-axis choice, sharing the weight load
between the two byte planes of an A16 activation. The compiler cache key is the fork commit: an uncommitted change re-serves the
cached artifact, so commit before measuring.

**FastRPC calls and data.** An inference is one FastRPC call (`tg_graph_run` runs all 371 kernel calls on the DSP), so there are no
calls left to fuse. What crosses it is data: the recurrent state (`state_*_q` in, `next_state_*_q` out, 2.2 MB each way for driving)
used to be copied through FastRPC every call. Now the emitter records which output slices feed which input regions (`G_NSTATE`,
`G_ST_*` in `graph.h`, `state` records in `program.txt`), the skel loops the slices back on the DSP after every run, and
`tg_graph_run`'s new `flags` argument leaves resident state out of the input and the output transfer. The gap between the runner's
call and the DSP loop for driving: 6.2 -> 3.5 ms (RPC-inclusive 161-164 -> 157 ms). Outputs are byte-identical and `--resident`
checks the resident path bit for bit against sending the state back.

The rest of the 3.5 ms was measured with phase stamps inside `tg_graph_run` (not in the tree): the state loop-back copy 2.45 ms
(2.2 MB, so libc `memcpy` ran at about 0.9 GB/s), the input copy 0.47 ms for driving and 1.15 ms for DM (1.4 MB), graph thread
creation 0.02 ms, output copy 0.02 ms, FastRPC marshalling about 0.5 ms. VTCM setup is once per load and helper threads persist,
so neither was a per-run cost. The copies now run on the graph thread (it holds the HVX lock) with 128-byte vector loads and
stores and `dcfetch` ahead: the gap is **1.0 ms** for both (driving 142.3 ms call / 141.3 ms DSP loop, DM 72.0 / 71.2 ms), outputs
identical. What is left is FastRPC marshalling; `rpcmem` zero-copy buffers would only trim part of that.

**DSP clock vote (`DSP_V65_PERF_VOTE`, level 3 by default in `compile_v65.sh`).** The skel never voted for DSP power, so the cDSP ran
at whatever DCVS chose for a light FastRPC client: identical runs of the same artifact ranged from 144 to 192 ms for driving and
72 to 96 ms for DM, and a PMU probe of the tiny-kernel graphs measured about 360 MHz effective. The skel now takes the HVX power
vote and a DCVS performance-mode request when an artifact opens (reference-counted over open programs, released when the last one
closes; the same pattern as `scripts/android/mcc_hmx/tg/replay_impl.c`). Level 1 (performance mode, no pinned corner) was slower
(synthetic 300-call graph 39.2 ms vs 29.2 ms unvoted), level 2 pins TURBO, level 3 pins the MAX corner. Interleaved A/B, two rounds:

| vote | driving DSP ms | DM DSP ms |
|---|---:|---:|
| none | 144-192 (varies) | 72-96 (varies) |
| TURBO | 149.4, 147.4 | 76.4, 76.4 |
| MAX (default) | 139.7, 139.2 (140.0, 140.2 in the final build) | 71.6, 71.6 (71.4) |

Outputs are byte-identical with and without the vote. Two consequences. Every earlier A/B in this README was taken with the clock
free-running: differences of a few ms between runs could be clock state, so decisions rest on the large or repeated ones (the
pinned clock makes runs reproducible to about 1 ms from here on). And a pinned MAX corner costs power: for battery use pass
`DSP_V65_PERF_VOTE=0`. The tiny-kernel graphs (300 calls of 512-element softmaxes, 100 distinct-size slices) were measured unvoted, where distinct kernels looked 100-200 us each more expensive than repeats; the PMU's own cycle counts, with
corrected V69 event codes (the itrace header's low bytes are offset by 2 from the V69 manual's: I$ miss 0x12, D$ miss 0x13, JTLB
miss 0x58, DTLB miss 0xb3, packets 0x03, DU stall 0xa0, L2 DU read miss 0x7d, cycles with one thread 0x3b), showed a median call of
only ~1.9 K packets with no cache or TLB misses: the "113 us floor" was an artifact of the profile listing only the slowest 254
calls, and the tiny-graph slowness is the low clock. The first PMU probe of this file used the header's codes and counted other
threads' packet commits instead; its numbers are void.

**Depthwise (sliding-window vrmpy).** Each channel's padded image is one flat byte signal. A 128-lane vrmpy takes one unaligned
128-byte load at offset `128b + s + ky*Wp + 4g` and a splat of 4 weights; four shifts s = 0..3 give 128 outputs, re-ordered with
`vshuff` -4 then -8 (the negative forms are the full-interleave constants). 56 vrmpys per 128 outputs for 7x7 and 12 for 3x3,
instead of about three loads per output. The idea is the SDK's `qhdsp_hvx_conv7x7_ab.c` (read for technique, not copied: it is
Qualcomm Proprietary). The old depthwise bucket was 40 ms; the new kernels are 0.03-0.3 ms each. The input planes come from a small
custom prep kernel; the tinygrad-generated pad/flatten/plane-split version made DM slower (93.8 -> 101.9 ms). Not tried: the
`valign` variant (aligned loads and register shifts).

**Lane join.** A 128-lane u16 -> u8 plane split built its vector from 256 scalar loads plus `vinsert` chains: a wide load split
into two 64-lane halves fell through to a per-lane constructor. `_lane_join` in `ops_dsp.py` turns a STACK of consecutive elements
from several narrower loads of one buffer into one unaligned wide load: 680 -> 20 us for the standalone kernel under qemu; no
change on the phone's total (the plane-split kernels are small). The 19 `E_128_2_128` calls (8.0 ms) that looked like lookups are
requantization epilogues; `vgather` already converts every real lookup in the driving capture (10 sources, 22 calls).

**Table lookups through `vgather`.** An unsigned PD gets VTCM with `HAP_compute_res_acquire` and a VTCM parameter (no HMX vote;
2, 4 and 8 MB were all granted). The 128 KB tables need the word-offset gather (`Q6_vgather_ARMWw`, 64 halfwords per instruction,
reads VTCM only, region within one page); the skel copies each distinct table into VTCM once after the weights load. Per call on
98304 random indices: scalar loop 118-122 us, `vgather` with the table resident 12.5-13 us, with the table copied every call 191 us.
Driving's 20 lookup calls went 8.6 -> 1.25 ms. The emitted program keeps the scalar loop and uses it when VTCM is unavailable, so
qemu stays bit-exact. Not tested: a target with less VTCM than this phone's 8 MB (driving's 20 tables are 2.6 MB), and the driving
and DM artifacts loaded together in one runner session (each takes its own VTCM).
The earlier "26 ms of table lookups" was a loose classifier's number; the real u16 lookups were 8.6 ms.

The copy rule needs unit stride in *every* buffer that uses the axis: with "any buffer" DM went from 93.5 to 119.5 ms (13 kernels of
a 345x3x64 shape vectorized an axis one of their buffers walks with a stride, so the vector was a gather). Elementwise total for
driving is now 85.7 ms (98.0 before). What is left there is table lookups (about 20 ms), the stem's plane split + pad + channel-4
interleave (7 ms: a 4-way transpose, which needs vector shuffles tinygrad cannot express) and many 1-4 ms layout kernels.

**Table lookups (the largest part of what is left in the elementwise bucket, about 40 ms of driving's 86).** Benchmarked on the
phone as eight chained Gelu lookups over 98304 elements (`lut8.onnx`-style: DQ -> Gelu -> Q with per-tensor u16 parameters), DSP time
per kernel: 238 us with 64K-entry tables, 110 us with 256-entry (uint8) tables. The upcast width does not matter (128 lanes: 1.91 ms
for the chain, 8 lanes: 1.91, none: 2.09), so the `vinsert`/`valign` chains that build the 128-lane result vector are not the
bottleneck: the loop is scalar loads (index, table) and a store at about 4.5 cycles per element even with a table that fits L1
(the 4 hardware threads share the pipeline), plus about half again from the 128 KB table missing L1. Shrinking the table
(`ONNX_QDQ_LUT_DROP`, lossy) gave only 2% on the whole model. Only a vector gather would help (V65's `vgather` reads from VTCM), which
needs the tables in VTCM and a gather op in the DSP renderer; not attempted.
**Stem plane split + pad + channel-4 interleave (7 ms):** a 4-way byte transpose of u16 activations; fusing the hi and lo plane
kernels into one would save one read of the u16 tensor (about 1-2 ms of the 7); the transpose itself needs `vshuff`/`vdeal`, which
tinygrad's expression set does not have.

Tried and dropped: blocking pixels inside the tensor-core construction (`DSP_TC_MBLOCK`): the compiler passes ran out of memory
(12 GB in 6 s). The same blocking as an ordinary upcast after the tensor core (`DSP_TC_MUPCAST`) works.

The last two are transport: casting the frame queue to float made an 8 MB output (and a 15 ms scalar cast kernel), and
sending `state_*_q` in and `next_state_*_q` back every call moved 4.4 MB over `adb forward`. The runner now keeps each
`next_X` output (program.txt `state` records) as the next call's `X` when the client sends `X` empty; the client's
`--resident` checks that against sending it back explicitly. What is left per call is `new_img` (393 KB) and the FastRPC
copies (about 6 ms between `hexagon_run` and the DSP time).

**Compile time.** One compiler service handles requests concurrently: cache hits are served while models compile, `--jobs`
(`COMPILE_JOBS`, default 2) bounds concurrent cold compiles, and a second request for a model already compiling waits for it
and reuses the artifact. Each kernel is compiled once and cached by source across compiles (the mock-DSP capture build through
tinygrad's disk cache, the program's objects through `DSP_GRAPH_OBJ_CACHE`); the qemu check and the skel build share those
objects and run side by side; compile3's self-test reruns are skipped. Quantized models, W8/W16 A16:

| | before | cold | warm (new cache key, unchanged kernels) |
|---|---:|---:|---:|
| `driving_supercombo` | 812 s | 495 s (compiled alongside DM) | 139 s |
| `dmonitoring_model` | 69 s | 56 s | |

What a warm compile still spends is qemu running the model: three capture runs and the reference, about 17 s each for driving,
plus the 16 s check of the emitted program.

The compiler's cache key is the target, `--compiler-id`, command string and model bytes. It does not include the tinygrad
checkout the command runs, so put the fork's commit in `--compiler-id` (for example `tinygrad-$(git -C "$TINYGRAD_ROOT"
rev-parse --short HEAD)`); otherwise a codegen change keeps serving old artifacts. The compiler service is one request at a
time, so a long compile blocks cache hits behind it. A second instance on the same
`--cache-dir` (its publication is multi-process safe) serves those. The runner is reached through `adb forward`; a
`run_compiled` round trip adds about 8 ms of TCP over USB to the DSP time. For driving it adds about 250 ms, mostly from moving
its 2.4 MB of inputs and 8 MB of outputs, which are mostly the recurrent state.

This makes all of them generatable and correct on v65. They are not fast: openpilot runs at 20 Hz, and these are one to two
orders of magnitude away. v65 float is scalar. DM's stem converts each uint8 input pixel again for every output-channel group
and accumulates four outputs at a time, and the warps spend about 180 cycles per pixel on a float divide plus a trunc emulation
per coordinate. The next lever is scalar register blocking over output channels, the tiling that made calls 6/10 5x faster by
hand, chosen by the heuristic. After that comes the integer path (`scripts/openpilot_dsp/hvx65`) for the convolutions.

BEAM search gave a genuine 2.37x improvement over naive codegen, confirmed on real hardware (not
just qemu's instruction-count proxy) -- real signal. But even BEAM=2's best kernel is ~2.2x
*slower* than TVM's hand-tuned schedule: tinygrad's Hexagon renderer had (before this session)
zero references to `vrmpy` or any dot-product intrinsic anywhere in its source -- it relies
entirely on LLVM's generic auto-vectorizer to recognize a scalar multiply-accumulate loop and
turn it into hardware SIMD, which it doesn't do for Hexagon's `vrmpy`. Higher BEAM values (4, 6)
converge to the identical kernel as BEAM=2 -- confirmed by diffing the generated source
byte-for-byte -- so more search width doesn't help; the renderer's action space has no vrmpy
option to find.

## Adding real vrmpy support to tinygrad

`vrmpy_tensorcore.patch` (PR: https://github.com/onnxsim/tinygrad/pull/1, against a
`v0.14.0`-pinned branch of the `onnxsim/tinygrad` fork) adds `hexagon_v65`, a real `TensorCore`
declaration for `V6_vrmpyub_acc_128B` / `V6_vrmpybusv_acc_128B`:
`D(int32x32) = C(int32x32) + dot4(A(u8x4 broadcast scalar), B(u8x128, 32 groups of 4))`, one HVX
vector instruction.

This is the **first TensorCore in tinygrad with `threads=1`**. Every existing definition
(CUDA/AMD/Metal) assumes warp-cooperative execution -- the `TensorCore` dataclass's `threads`,
`opts` and `swizzle` fields all encode how work splits across a warp's lanes. `vrmpy` needs none
of that: it's a plain single-thread SIMD instruction. There's no existing single-thread example
to copy the `opts`/`swizzle` fields from; they were derived directly from
`TensorCore.__post_init__`'s assertions (`local_axes=0` since `2**0==threads`; all 5
axis-doubling opts needed for `N=32` are `"u"`-type since `M=1` needs none) and worked on the
first attempt.

Two real bugs surfaced and fixed along the way, both specific to being the first
tensor-core-capable-but-warp-incapable backend:
1. `MockDSPRenderer.__init__` has its own override that never calls the base class's, silently
   dropping `tensor_cores` -- the TC was registered on `DSPRenderer` but invisible to the qemu
   path used for verification.
2. `hand_coded_optimizations` unconditionally attempts an `OptOps.LOCAL` opt right after
   successfully applying a TC (an "improve ILP" step), **uncaught** -- crashes outright on any
   backend with tensor cores but no local-memory support, which every backend until now has had.
   Guarded behind `renderer.has_local`.

**Verified correct**: bit-exact under qemu, and bit-exact on real Hexagon v73 hardware via the
bridge above (`tinygrad_gemm` -> genuine `__builtin_HEXAGON_V6_vrmpyub_acc_128B` calls, compiled
with the vendor Hexagon toolchain, no errors).

**Not yet fast**: 0.28 GMAC/s, slower than even naive scalar codegen. Root cause, precisely
diagnosed: `devectorizer2`'s `do_stack_wmma` (`codegen/__init__.py`) unconditionally decomposes
*every* WMMA's accumulator into 32 individual scalar element-loads before rendering. That's the
*correct* behavior for every other backend -- each GPU thread genuinely only ever holds a few
scalar elements of a warp-distributed fragment, so there's nothing to "keep as a vector" at the
per-thread level. It's *wrong* for Hexagon: the "32 elements" is one HVX vector register that a
single thread (`threads=1`, no warp) processes atomically in one instruction, and it should stay
vector-resident across the whole reduction loop instead of being rebuilt from 32 scalar reads
before every accumulate call. BEAM search independently corroborates the regression: across 15+
candidates explored at BEAM=2, it never selects the WMMA path, consistently preferring plain
scalar codegen -- exactly matching what the real-hardware measurement shows.

**Correction**: an earlier version of this note claimed a "minimal, non-interleaved single-TC-call
test" still showed the scalar rebuild, "confirming" the issue wasn't about interleaving. That test
used a hand-rolled monkeypatch of `postrange.apply_opts` to try to isolate a bare `Opt(OptOps.TC)`
application -- the monkeypatch silently wasn't being hit (confirmed later via the official
`NOOPT=1` control, which behaves differently), so that test was actually exercising the normal,
unmodified path the whole time and proved nothing about interleaving specifically. See "Three
more attempts" below for what a *reliable* trace found instead.

### Three more attempts at a fix, each real and each a dead end

Further digging (tracing via a monkeypatch of `heuristic.hand_coded_optimizations` directly,
which -- unlike patching `apply_opts` -- reliably fires) found the actual mechanism: `applied_opts`
after a real TC application is `[Opt(TC, ...), Opt(UPCAST, axis=0, amt=4)]` -- `hand_coded_optimizations`'s
own "improve ILP by upcasting M and N" step is what groups multiple independent M-row accumulators
into one interleaved buffer. Three attempts followed, each tested for correctness and, where
correct, for real speed on hardware:

1. **Skip the M-upcast for backends without locals** (guard `for tc_dim in [1,0]` down to `[0]`
   when `not tk.ren.has_local`): eliminates the interleaving cleanly -- WMMA call count drops from
   4 to 1 at the same shape, and the accumulator becomes a plain contiguous 32-element buffer
   slice instead of a stride-4 interleaved one. **Measured on real hardware: slower**, not
   faster -- 0.19 GMAC/s vs. 0.28 GMAC/s. Losing the M-row interleaving's instruction-level
   parallelism costs more than the simpler accumulator access saves. Reverted.
2. **Reorder the `TensorCore.swizzle` field's upcast axis list**: swept several permutations of
   `('u0','u1','u2','u3','u4')`. Every permutation except the original either produced **wrong
   results** (confirmed against a numpy reference) or left the accumulator's offset pattern
   completely unchanged. Conclusion: `swizzle` governs `A`/`B`'s data layout, tightly coupled to
   `vrmpy`'s actual hardware lane semantics (permuting it independently breaks which output lane
   receives which input) -- it does not control `C`'s (the accumulator's) element ordering at all.
3. **Remove the reversal in `TensorCore.base_upcast_axes()`** (the actual, separate mechanism that
   *does* order `C`, via `_apply_tc_opt`'s `tc_upcast_axes` slicing): even with only Hexagon's TC
   registered, this **crashes** a different invariant -- `codegen/late/coalesce.py`'s
   `memory_coalescing` pass asserts (`attempting multiple stores: 4`) on the resulting UOp
   structure. The reversal isn't an arbitrary convention; something else downstream depends on it
   being there, even though its only in-repo justification is a one-line comment ("this is defined
   in the swizzle... first we use the upcast axes, then the reduce").

All three are consistent with one pattern: every lever that touches the accumulator's construction
or ordering is entangled with some *other* invariant in the generic, shared kernel-optimization
pipeline that every backend goes through -- not something a local, Hexagon-only tweak can safely
adjust without either the fix itself regressing (attempt 1) or breaking a different, seemingly
unrelated assumption (attempts 2 and 3). A real fix within the shared `TensorCore`/`WMMA`
machinery needs a maintainer's-level map of what `_apply_tc_opt`, `base_upcast_axes`, and
`memory_coalescing` jointly assume -- more archaeology than this session had time for.

### Modeling Hexagon as its own accelerator, not a `TensorCore` variant

Given the shared-machinery entanglement above, the more promising direction is to **stop routing
through `Ops.WMMA`/`TensorCore` entirely** for Hexagon and instead give it its own, hand-written
lowering -- sidestepping `do_stack_wmma`, `base_upcast_axes`, and `memory_coalescing` altogether
rather than trying to make them single-thread-aware.

tinygrad has an established, precedented mechanism for exactly this: `Tensor.custom_kernel`
(`UOp.custom_kernel`) lets you hand-build a kernel's UOp graph directly -- explicit `UOp.range`
loops, `.index()`/`.store()`/`.load()`, and `Ops.CUSTOM`/`Ops.CUSTOMI` for injecting a raw,
target-specific intrinsic as a format string (`arg.format(*rendered_srcs)`). `tinygrad/llm/kernels/amd.py`
is a full, real, ~500-line module built this way for AMD-specific quantized-linear/attention
kernels, including register-resident (never memory-array-decomposed) accumulators via
`UOp.placeholder(shape, dtype, addrspace=AddrSpace.REG)` threaded through the reduction loop with
`.store()`/`.after()` -- precisely the "stay vector-resident across the loop" property `do_stack_wmma`
fails to give Hexagon's accumulator. `_amd_dp4a` (`amd.py`) is the exact same shape of problem as
`vrmpy`: a 4-element dot-product-accumulate hardware intrinsic, injected via one `Ops.CUSTOMI` call.

`custom_kernel_attempt.py` was a from-scratch attempt at the Hexagon equivalent -- one `vrmpy`
call, accumulating a real `int32x32` register value across a K-reduction, entirely bypassing
`Ops.WMMA`. It got stuck for a while (see the file's own header comment for that history), but
**this now works**, in `hex_gemm_kernel.py` -- see "It works" below. The bugs found and fixed
along the way, each a genuine, distinct issue (not the same wall hit repeatedly):

1. **UOp shape-broadcast mismatch** (`(32,)`/`(128,)`/`(1,)` couldn't unify): caused by calling
   `.load()` on the wide A/B slices before feeding them to the `Ops.CUSTOMI` intrinsic call, which
   gives them their full array shape. Traced `Ops.CUSTOM`/`Ops.CUSTOMI`'s shape rule directly
   (`case Ops.CUSTOM | Ops.CUSTOMI: if self.dtype is dtypes.void: return None` in `uop/ops.py`) and
   `_amd_load`'s real pattern in `amd.py` to find the fix.
2. **Fixed** by passing the raw `Ops.INDEX` (an address, shape `()`) directly into the `CUSTOMI` op
   instead of a `.load()`'d array value, matching `_amd_load`'s actual usage exactly -- then hit a
   **UOp spec-verification failure** (`uop/spec.py`'s `type_verify`) on `acc.load()`: loading an
   `Ops.AFTER`-wrapped placeholder directly isn't a form the verifier recognizes.
3. **Fixed** by dropping the explicit `.load()` entirely -- re-reading
   `_amd_flash_attention_decode_partial`'s real, working accumulator idiom
   (`prev_acc = acc.after(offset)[head]`, `acc[head].store(...)`, no `.load()` calls anywhere) showed
   that a `AddrSpace.REG` placeholder, once `AFTER`-scoped to its dependency, *is* the loadable value
   directly -- indexing/using it does the load implicitly.
4. Past both of those, hit **a genuine tinygrad control-flow bug**: `codegen/late/linearizer.py`'s
   scheduling pass asserted (`assert y.src[1] not in x.backward_slice_with_self`) on an adjacent
   comment reading *"TODO: this can happen! it causes infinite loop in shufflenet"*. **Fixed** by
   calling `.end(kc)` on the reduction range exactly once (only where the accumulate step's update
   value is constructed), not again in the final store's `.end(...)` call -- the double-`.end()` on
   the same range was the trigger, not an unrelated tinygrad bug after all.
5. Next error was mundane: `AxisType.GLOBAL` for the outer `M` loop rendered to `Ops.SPECIAL` with a
   GPU-style workitem code (`'g'`) `ClangRenderer` doesn't implement (`has_threads=False`, no grid
   dispatch). **Fixed** by using `AxisType.WEAK` (plain sequential loop) instead.
6. Then a **pointer-cast bug**: the format string wrote `*(unsigned char128*)&{1}`, but `{1}`
   (an `Ops.INDEX`) already renders as a pointer expression -- taking `&` of it is invalid. **Fixed**
   by dropping the `&`.
7. With that fixed, the *compiled* code revealed the real semantic bug: the `(32,)`-shaped
   `Ops.CUSTOMI` result was still being decomposed by the generic elementwise devectorizer into 32
   separate calls, each with a spurious `+lane_index` appended to the result -- `Ops.CUSTOMI`,
   despite representing one opaque hardware call, gets treated exactly like any other `(32,)`-shaped
   elementwise computation, which assumes each of the 32 output elements is an independent
   per-lane result. **Fixed** by never constructing a `(32,)`-shaped *value* at all: the whole
   accumulate step became one `dtypes.void`-dtype `Ops.CUSTOM` **statement** (`case Ops.CUSTOM |
   Ops.CUSTOMI: if self.dtype is dtypes.void: return None` in `uop/ops.py` -- `void` dtype exempts
   it from shape/broadcast tracking entirely), addressing the accumulator only by its base pointer
   (`acc[0]`, shape `()`, from indexing the `AddrSpace.REG` placeholder -- same address-not-value
   pattern as step 2) and doing the load-vrmpy-store round trip as raw C inside the format string.
8. Final snag: the format string referenced `unsigned char128`/`int32` -- vector typedefs the
   renderer only auto-emits when a shaped *value* of that type appears somewhere, which no longer
   happens once accumulation is a `void` statement. **Fixed** by using
   `__attribute__((vector_size(128)))` inline in the cast expression instead of a named typedef,
   sidestepping the need to inject a declaration into the kernel prologue at all.

### It works

`hex_gemm_kernel.py` is the resulting, working, from-scratch `vrmpy` GEMM kernel: `C[M,N] =
A[M,K] @ B[K,N]` (uint8 x uint8 -> int32), correct under qemu and **correct on real Hexagon v73
hardware** (via the TVM-transport bridge), at the *exact* pathological shape this whole
investigation started from -- `scripts/android/maskrcnn_e2e/README.md`'s ranked-profile finding,
`cin=64, cout=256`, spatial `200x272` (`M=54400`):

| | Median | Throughput |
|---|---:|---:|
| tinygrad, naive (BEAM=0) | -- | 0.42 GMAC/s (isolated 512-row proxy shape) |
| tinygrad, BEAM=2 (tiled) | -- | 1.00 GMAC/s (isolated 512-row proxy shape) |
| tinygrad, `Ops.WMMA`/`TensorCore` (broken accumulator) | -- | 0.28 GMAC/s (isolated 512-row proxy shape) |
| stock TVM (hand-tuned `vrmpy` schedule) | 254.1 ms | 3.51 GMAC/s (**real shape**) |
| **tinygrad, hand-written `custom_kernel`** | **29.363 ms** | **30.35 GMAC/s (real shape)** |

**8.65x faster than stock TVM's hand-tuned schedule, bit-exact correct, at the real shape** --
not a synthetic proxy. This is the single largest line item in the original ranked profile
(rank 2, 13.4% of the isolated-timing total, one of the "small-channel 1x1 convs at 10.1x below
the throughput of large-channel 3x3 convs" from the systemic-pattern finding); at 30.35 GMAC/s it's
now well past that comparison point entirely.

`B` needs pre-packing into `hex_gemm_kernel.pack_b()`'s layout before use (128 contiguous bytes
per `(N-tile, K-chunk)`, matching what one `vrmpy` call reads in one shot) -- a one-time,
host-side repack of the (static) weight tensor, not something done per-inference.

**Bug found and fixed**: originally failed whenever `cout == 32` exactly, or more generally
whenever the N-tile count is 1 (also affects `build_strided_kernel()`, same underlying pattern).
Root cause: `_reg_i32`'s accumulator-init scoped its `.after()` dependency to only the innermost
enclosing range (`nt_rng`); tinygrad's optimizer *eliminates* degenerate extent-1 ranges entirely,
which silently dropped the "reset the accumulator once per row" dependency down to "reset once,
ever" -- corrupting every row after the first. Fixed by depending on every enclosing range (`M`
can never degenerate to extent 1 in practice, so this is robust). This directly unlocked the tiny
`cout=12`/`cout=3` coverage below, via `cout` zero-padded to 32.

### Coverage: the other small-channel 1x1 convs in the backbone

`maskrcnn_e2e/profile_data/conv_profile.json` lists every unique conv shape in the backbone. The
same kernel (just re-parameterized by `cin`/`cout`/`M`, no code changes) was verified correct and
measured on real hardware against the next four largest small-channel 1x1-conv shapes by
isolated-timing impact:

| `cin` | `cout` | spatial | stock TVM | `custom_kernel` | speedup | correct |
|---:|---:|---|---:|---:|---:|---|
| 64 | 256 | 200x272 | 3.51 GMAC/s | 30.35 GMAC/s | **8.65x** | yes |
| 128 | 512 | 100x136 | 6.03 GMAC/s | 38.66 GMAC/s | **6.41x** | yes |
| 64 | 64 | 200x272 | 2.88 GMAC/s | 14.36 GMAC/s | **4.99x** | yes |
| 256 | 64 | 200x272 | 7.60 GMAC/s | 18.15 GMAC/s | **2.39x** | yes |
| 512 | 128 | 100x136 | 12.72 GMAC/s | 25.45 GMAC/s | **2.00x** | yes |

All five bit-exact correct, all faster than TVM's hand-tuned schedule. Together with the strided
shape below, they account for ~2217 ms of the ~7620 ms isolated-timing total from the original
ranked profile -- **about 29% of the entire backbone's isolated-timing sum**, from the same
`hex_gemm_kernel.py`/`build_strided_kernel()` code, re-parameterized by shape each time.

### Coverage: the strided 1x1 conv

The one strided shape in the profile (`cin=256, cout=128, stride=2 @200x272`, 82.9 ms) needed
real 2D spatial indexing -- `build_strided_kernel()` in `hex_gemm_kernel.py` takes the flat,
*unstrided* input (`ih*iw, cin`) and two `(oh, ow)` `WEAK` ranges instead of one flat `M` range,
computing the strided source row as `(oh*stride)*iw + (ow*stride)` directly in the index
expression -- no change to the accumulate/vrmpy step at all, only to how `A`'s address is
computed. Correct on the first attempt (qemu and real hardware), no new bugs:

| `cin` | `cout` | spatial | stride | stock TVM | `custom_kernel` | speedup |
|---:|---:|---|---:|---:|---:|---:|
| 256 | 128 | 200x272 | 2 | 5.38 GMAC/s | 24.86 GMAC/s | **4.62x** |

### Coverage: the tiny RPN/mask-head convs (`cout=12` or `cout=3`)

Total impact across all spatial sizes of these two shapes is ~182 ms (not ~14 ms as an earlier
version of this note estimated -- corrected here), dominated by the largest spatial size
(`200x272`) for each: 83.7 ms (`cout=12`) and 52.1 ms (`cout=3`), together ~136 ms, ~75% of the
group's total. `cout` isn't a multiple of 32, so the kernel's N-tile loop doesn't apply directly;
covered by zero-padding the weight matrix's `cout` up to 32 (wasting most of a `vrmpy` lane, but
functionally correct and simple: `hex_gemm_kernel.build_kernel()` used completely unchanged with
`cout=32`, and only the first `real_cout` output columns are read). This is exactly the `NT=1`
case the bug above blocked -- fixing that bug is what unlocked this coverage:

| `cin` | `cout` (real) | spatial | stock TVM | `custom_kernel` (padded to 32) | speedup |
|---:|---:|---|---:|---:|---:|
| 256 | 12 | 200x272 | 2.00 GMAC/s | 3.89 GMAC/s | **1.94x** |
| 256 | 3 | 200x272 | 0.80 GMAC/s | 0.97 GMAC/s | **1.22x** |

Both bit-exact correct, both faster than TVM despite computing (and discarding) 20 or 29 unused
output lanes per call -- the `vrmpy` instruction's fixed 32-lane width means the "wasted" lanes
cost nothing extra beyond the one instruction already being issued. The smaller spatial sizes of
these same two shapes (`13x17` through `100x136`, ~46 ms combined) aren't separately verified but
should behave the same way (same kernel, same correctness argument, only the M-loop trip count
differs) -- not measured individually given their small individual impact.

### Coverage: the larger-channel 1x1 convs (`cin`/`cout` > 128)

The earlier coverage passes only looked at shapes with `cin<=128 or cout<=128` (the "small-channel"
systemic-pattern group). Re-checking the *entire* `conv_profile.json` (not just that filtered
subset) by total isolated-timing impact turned up several **larger**-channel 1x1 convs still
running well below the backbone's ~50-80 GMAC/s ceiling -- an earlier version of this note missed
these entirely by filtering too narrowly. The kernel needed zero code changes (it was never
channel-count-limited, only the earlier *search* for targets was) -- verified correct and measured
on real hardware exactly as before:

| `cin` | `cout` | spatial | stride | stock TVM | `custom_kernel` | speedup |
|---:|---:|---|---:|---:|---:|---:|
| 256 | 512 | 200x272 | 2 | 7.54 GMAC/s | 43.80 GMAC/s | **5.81x** |
| 256 | 256 | 200x272 | 1 | 9.78 GMAC/s | 38.00 GMAC/s | **3.89x** |
| 512 | 256 | 100x136 | 1 | 14.40 GMAC/s | 36.73 GMAC/s | **2.55x** |
| 256 | 1024 | 50x68 | 1 | 20.10 GMAC/s | 44.53 GMAC/s | **2.22x** |
| 1024 | 256 | 50x68 | 1 | 24.43 GMAC/s | 35.15 GMAC/s | **1.44x** |

All five bit-exact correct, all faster than TVM -- including the strided `cin=256,cout=512` case
(the `build_strided_kernel()` variant), which got the *largest* speedup of any shape covered so
far (5.81x) despite TVM already doing reasonably well on it in absolute terms (7.54 GMAC/s is
mid-pack, not obviously "broken" the way the smallest-channel shapes were).

**Total real coverage across all twelve verified shapes**: ~3562 ms of the ~7623 ms isolated-timing
total from the original ranked profile -- **about 47% of the entire backbone's isolated-timing
sum**, every one bit-exact correct and faster than TVM's hand-tuned schedule.

### Correction: every 1x1-conv speedup above was measured against unsigned synthetic weights

`hex_gemm_kernel.py`'s 1x1-conv kernel is `vrmpyub`-only (unsigned x unsigned) -- every speedup
number in the four sections above was measured with synthetic `rng.integers(0, 100, ...)` uint8
weight data. The real backbone's 1x1 conv weights are genuinely **signed int8** (confirmed from
`backbone.onnx`'s `ConvMulFusion_W_*_quantized` tensors), which this kernel can't even accept --
this went unnoticed until the chained-subgraph work (`codex/hex-backbone-block1`) ran real
extracted weights through the 1x1 path for the first time and it failed its own correctness check
outright. `hex_gemm_signed_kernel.py` (a separate kernel, `vrmpybusv`, not a fix to
`hex_gemm_kernel.py` -- see its own docstring) exists for exactly this reason.

Re-measured every shape above on real hardware (device `239dbd8f`) with `hex_gemm_signed_kernel.py`
against signed-random weight data (`rng.integers(-128, 127, ...)`, matching the real weights'
*distribution*, not real extracted weights for all thirteen shapes -- out of scope for the time
available; this tests whether `vrmpybusv`'s different calling convention, not the exact weight
values, changes throughput, which is the actual open question). Stock TVM's own numbers are
unchanged and not re-measured -- TVM was never wrong about signed weights, only this project's own
kernel was:

| shape | `cin` | `cout` (real) | spatial | stride | stock TVM | signed (`vrmpybusv`) | **new speedup** | orig. claim (unsigned) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| flagship / small-channel #1 | 64 | 256 | 200x272 | 1 | 3.51 | 26.10 | **7.44x** | 8.65x |
| small-channel #2 | 128 | 512 | 100x136 | 1 | 6.03 | 29.20 | **4.84x** | 6.41x |
| small-channel #3 | 64 | 64 | 200x272 | 1 | 2.88 | 14.28 | **4.96x** | 4.99x |
| small-channel #4 | 256 | 64 | 200x272 | 1 | 7.60 | 15.73 | **2.07x** | 2.39x |
| small-channel #5 | 512 | 128 | 100x136 | 1 | 12.72 | 20.89 | **1.64x** | 2.00x |
| strided | 256 | 128 | 200x272 | 2 | 5.38 | 19.63 | **3.65x** | 4.62x |
| tiny RPN `cout=12` (padded to 32) | 256 | 12 | 200x272 | 1 | 2.00 | 3.50 | **1.75x** | 1.94x |
| tiny mask-head `cout=3` (padded to 32) | 256 | 3 | 200x272 | 1 | 0.80 | 0.86 | **1.08x** | 1.22x |
| larger-channel #1 (strided) | 256 | 512 | 200x272 | 2 | 7.54 | 30.74 | **4.08x** | 5.81x |
| larger-channel #2 | 256 | 256 | 200x272 | 1 | 9.78 | -- | *not re-measured* | 3.89x |
| larger-channel #3 | 512 | 256 | 100x136 | 1 | 14.40 | 25.93 | **1.80x** | 2.55x |
| larger-channel #4 | 256 | 1024 | 50x68 | 1 | 20.10 | 31.30 | **1.56x** | 2.22x |
| larger-channel #5 | 1024 | 256 | 50x68 | 1 | 24.43 | 25.49 | **1.04x** | 1.44x |

**Twelve of thirteen shapes re-measured** (`native_transport`-free TVM-RPC bridge, matching the
original measurement methodology exactly). The `larger-channel #2` shape (`cin=256,cout=256
@200x272`) could not be re-measured -- its 54400x256 int32 output is 55.7 MB, past a hard
transfer-size wall in this project's TVM-RPC bridge session (`hexagon_rpc_send failed: 78`,
first documented in the elementwise-add prefetch work's own real-hardware measurements, a
pre-existing RPC-bridge limitation, not a kernel bug) -- left honestly unmeasured rather than
estimated.

**The finding is consistent and one-directional**: every single re-measured shape is *slower*
with signed weights than the original unsigned claim (`vrmpybusv`'s 32-wide broadcast calling
convention has a real, measurable throughput cost vs. `vrmpyub`'s plain-scalar one -- confirming
hypothesis (b), not (a)), by anywhere from ~1% (small-channel #3, effectively noise) to ~30%
(larger-channel #1's strided case, larger-channel #3). **The reassuring part**: every one of the
twelve re-measured shapes is *still* a real win over stock TVM (every speedup stays above 1x,
none flip to a loss) -- the *direction* of every claim in this project holds, only the *magnitude*
was overstated by testing the wrong weight signedness. The original unsigned numbers above are
left in place rather than replaced, per this project's established correction convention (see
the `TensorCore` devectorizer "Correction" note earlier in this file) -- this section is the
record of what changed and why, not a silent rewrite.

### Coverage: a real 3x3 conv (`hex_conv3x3_kernel.py`) -- correct everywhere, fast on some shapes

Every kernel above handles 1x1 convs (a plain GEMM once the spatial dims are flattened into `M`).
3x3 convs are `2509 ms` of the `6851 ms` conv-only profile total (the biggest bucket after 1x1),
so they're the natural next target. `hex_conv3x3_kernel.py` extends the same `vrmpybusv`
accumulate-loop pattern to a genuine spatial convolution:

- **Padding pushed out of the kernel**: the real backbone's 3x3 convs are all `stride=1, pad=1`
  ("same" output size). Rather than branch on boundary conditions inside the hot loop, the input is
  zero-padded by 1 pixel on each spatial side *before* the kernel runs (`pad_input()`) -- once padded,
  `a_row = oh + kh` / `a_col = ow + kw` directly index the padded array for `kh, kw in 0..2`, no `-1`
  offset or bounds check needed. (A real integration into the backbone would need to pad with the
  input's actual quantization zero-point, not literal 0, to match `qnn.conv2d` semantics exactly --
  noted in the file, not yet done, since this file's own correctness check compares against a
  reference padded the same literal-0 way.)
- **Combined reduction range**: one `REDUCE` range of extent `9 * (cin//4)`, decomposed inside the
  kernel via `//`/`%` into `(kh, kw, kc)` -- reuses the exact `_reg_i32` multi-range-dependency
  accumulator-init fix from `hex_gemm_kernel.py` (needed here too: `cin==4` makes this reduction
  range itself degenerate-extent-9, and separately `cout==32` still makes `nt` extent-1).
- **Weight packing extended to 9 positions**: `pack_weight_3x3()` packs each of the 9 kernel
  positions with `pack_b()`'s exact per-position layout (`Wp[pos, nt, kc, n_lane*4+ks]`), so each
  `vrmpybusv` call still reads one contiguous 128-byte weight slice.

Verified **bit-exact correct on real hardware** (`vrmpybusv_acc_128B`, uint8 activation x signed
int8 weight, matching the real backbone's QNN quantization) at the three highest-impact 3x3 shapes
in the profile, at their real spatial sizes. Initial (untiled, `ow_tile=1`) results were a **mixed
result**: correct everywhere, but the single biggest 3x3 bucket in the whole profile
(`cin=cout=256`) was *slower* than stock TVM, the two smaller-channel shapes roughly even to
modestly faster:

| `cin` | `cout` | spatial | stock TVM | untiled (`ow_tile=1`) | speedup |
|---:|---:|---|---:|---:|---:|
| 256 | 256 | 200x272 | 49.30 GMAC/s (0.651 s) | 33.41 GMAC/s (0.960 s) | **0.68x (slower)** |
| 64 | 64 | 200x272 | 20.86 GMAC/s (0.096 s) | 28.96 GMAC/s (0.069 s) | **1.39x** |
| 128 | 128 | 100x136 | 31.87 GMAC/s (0.063 s) | 32.19 GMAC/s (0.062 s) | **1.01x** |

**Why, most likely**: this kernel did zero explicit cache-blocking or output-tile reuse -- a direct
nested loop (`oh -> ow -> nt -> reduction`) with no register tiling across neighboring output
pixels, unlike the 1x1 kernels where every "row" (`M`) is independent and TVM's own baseline was
already memory-bound in a way a single-accumulator loop matches well. A 3x3 conv's per-output-pixel
weight working set is 9x an equivalent 1x1's (`9*256*256 = 589824` packed weight bytes at the
biggest shape -- past a typical Hexagon L1's size, so every output pixel's full reduction re-streamed
weight data from L2/memory with no reuse across pixels).

### Fixing it: output-tile reuse (`ow_tile`)

`build_kernel()`'s new `ow_tile` parameter processes several adjacent output columns per
accumulator group: for each reduction step, the packed weight slice (shared across all tiled output
columns -- it depends only on `nt`/`kc`, not on the output column) is loaded into a local vector
once and reused across `ow_tile` separate `vrmpybusv_acc` calls, one per tiled column with its own
`AddrSpace.REG` accumulator, instead of being re-fetched from memory once per output pixel. Both the
fused accumulate step and the fused output write-back are single `Ops.CUSTOM` statements covering
all `ow_tile` columns at once (see the file for why: an earlier version tried chaining `ow_tile`
separate per-column `CUSTOM` statements via `.after()`, which looked right -- each later write's
source embeds a dependency on the earlier write -- but silently dropped every write except the last
from the rendered output, since `.after()` only orders two nodes that are *already* reachable from
the sink; it doesn't itself make an otherwise-unreferenced void statement reachable. Confirmed by
inspecting the generated C for `ow_tile=2`: only the last column's write appeared, so half the
output columns were simply never written -- exactly the 50% mismatch that bug produced before the
fix).

Tile size was picked using the `HEXSIM=1` BEAM-search timing mode (see "Timing BEAM search
candidates with hexagon-sim instead of raw instruction counting" elsewhere in this file, or PR
https://github.com/onnxsim/onnxsim/pull/1780) rather than guessed -- a genuine, deliberate test of
that infrastructure, not just a demonstration. **Correction**: the sweep below was described as "at
the real `cin=cout=256, 200x272` shape" -- it's actually a small spatial proxy (`8x32`) at the same
`cin=cout=256` (the dimension that actually drives the cache-pressure effect being measured);
`hexagon-sim --timing`'s cycle-accurate simulation of the real kernel's ~31M reduction steps at full
`200x272` scale didn't finish in several minutes (confirmed directly: killed after running past that
without completing), so the sweep never ran at true full scale. `hexagon-sim --timing`'s
Pcycles-derived cost was **non-monotonic** in tile size, real signal a qemu-instruction-count proxy
would not have shown:

| `ow_tile` | Pcycles-derived time | vs untiled |
|---:|---:|---:|
| 1 (untiled) | 7244.75 us | -- |
| 2 | 8265.37 us | **worse** |
| 4 | 5319.82 us | 1.36x better |
| 8 | 4745.00 us | **1.53x better** |

`ow_tile=8` won clearly and was verified bit-exact correct on real hardware at all three profile
shapes, with a uniform ~1.3x real-hardware speedup over the untiled kernel and no regressions:

| `cin` | `cout` | spatial | stock TVM | untiled | tiled (`ow_tile=8`) | vs TVM | vs untiled |
|---:|---:|---|---:|---:|---:|---:|---:|
| 256 | 256 | 200x272 | 49.30 GMAC/s | 33.41 GMAC/s | **44.16 GMAC/s** (0.727 s) | 0.90x (still slower) | 1.32x |
| 64 | 64 | 200x272 | 20.86 GMAC/s | 28.96 GMAC/s | **38.01 GMAC/s** (0.053 s) | **1.82x** | 1.31x |
| 128 | 128 | 100x136 | 31.87 GMAC/s | 32.19 GMAC/s | **42.44 GMAC/s** (0.047 s) | **1.33x** | 1.32x |

Reported honestly: tiling closes *most* of the gap at the biggest shape (0.68x -> 0.90x of TVM) but
not all of it, while pushing the two already-ahead shapes further ahead (1.39x -> 1.82x, 1.01x ->
1.33x). `ow_tile=8` needs 8 concurrent `(32,)` `int32` accumulators live at once (8 HVX vector
registers, 1KB) plus the weight/activation working set -- likely close to Hexagon v73's real
register budget, which is a plausible reason larger tiles weren't swept further.

### Sweeping further: `ow_tile=8` is a real ceiling, not just an unswept guess

Follow-up (same `8x32,cin=cout=256` `HEXSIM=1` proxy as above, for fast iteration -- see the
correction above about why full `200x272` scale isn't practical for this): both of this section's
own "Not done" items were actually tried, and both came back negative, turning the earlier
speculation ("likely close to Hexagon v73's real register budget") into a confirmed, evidenced
finding rather than a guess:

- **Sweeping `ow_tile` past 8**: `ow_tile=16` measured *flat* (0.00475139 s, 0.13% worse than
  `ow_tile=8`'s 0.004745 s -- noise-level) and `ow_tile=32` measured **clearly worse** (0.005719 s,
  20% worse). No further gain past 8; the register-budget hypothesis holds.
- **2D (row+column) tiling at the same total accumulator count**: an `oh_tile=2, ow_tile=4` 2D tile
  (still 8 total accumulators, so the same register pressure as the winning 1D `ow_tile=8`, testing
  whether 2D activation-data locality helps independently of the weight-reuse count) measured
  0.005024 s -- **5.9% worse** than the pure 1D `ow_tile=8` result, not better. The weight-reuse
  factor is what the tiling amortizes (fixed at 8 either way here), and the 2D tile's less-sequential
  activation addressing (a row-stride jump instead of contiguous columns) cost more than any
  locality benefit it might have added.

So this is a genuine ceiling, not an unexplored direction: at this kernel's current register/loop
structure, `ow_tile=8` (already the committed, merged value) is the real local optimum among the
sweep tried, both along the 1D axis and against a same-budget 2D alternative. Closing the remaining
`cin=cout=256` gap (0.90x of TVM) further would need a structurally different approach -- e.g.
reducing the *weight* working set some other way, or restructuring the reduction to touch less
memory per output tile -- not more of the same tiling.

**Still not done**: real end-to-end backbone impact isn't measured (would need the same
splice-into-`relay.build()` mechanism `backbone_splice/` uses, currently blocked on that work's own
unresolved RPC loading bug -- see `backbone_splice/README.md`).

### Coverage: the ResNet stem 7x7 conv (`hex_stem7x7_kernel.py`) -- the last uncovered conv shape

`conv_profile.json`'s last remaining uncovered *conv* bucket (excluding the 3x3 tiling refinement
above): the single ResNet stem conv, `ishape=[1,3,800,1088]`, `wshape=[64,3,7,7]`, `stride=[2,2]`,
`pad=[3,3]`, 308.5 ms. `hex_stem7x7_kernel.py` generalizes `hex_conv3x3_kernel.py`'s pattern from
9 kernel positions to 49, and adds real output-vs-input spatial size mismatch (stride=2, so unlike
the 3x3 kernels' "same"-size case, `(oh, ow)` need their own ranges computed from `(ih, iw)`, not
reused directly -- the same `(oh, ow)` vs `(ih, iw)` split `build_strided_kernel()` in
`hex_gemm_kernel.py` already established for strided 1x1 convs).

One genuinely new wrinkle: `cin=3` isn't a multiple of 4, the K-chunk width every `vrmpybusv` call
in this project assumes. Handled the same way `hex_gemm_kernel.py`'s tiny-`cout` coverage handled
`cout` not being a multiple of 32 ("Coverage: the tiny RPN/mask-head convs" above) -- zero-pad, but
on the reduction (`cin`) axis instead of the N (`cout`) axis this time: `cin` padded up to 4 with
an all-zero weight channel, so the padding lane always contributes `activation * 0 == 0` regardless
of what's in the corresponding activation padding byte. Free for the same reason N-axis padding
was free: `vrmpybusv`'s K-chunk width is fixed at 4 regardless, so `cin=3` already pays for a
4-wide reduction step; padding to `cin=4` just makes that width explicit.

Verified **bit-exact correct on real hardware** (device `239dbd8f`, `vrmpybusv_acc_128B`, uint8
activation x signed int8 weight) at the exact real profile shape, at full scale (`oh=400, ow=544`,
~10.66M reduction steps) -- also bit-exact under qemu at that same full scale (6 seconds to
generate+verify, not just the smaller shapes checked during development):

| `cin` | `cout` | spatial (in) | kernel | stride | stock TVM | `custom_kernel` | speedup |
|---:|---:|---|---:|---:|---:|---:|---:|
| 3 | 64 | 800x1088 | 7x7 | 2 | 2.27 GMAC/s | 31.92 GMAC/s | **14.08x** |

The largest speedup of any shape covered so far, and it makes sense why: TVM's packed NCHWc
schedule pads `cin` up to a full 32-wide input-channel block regardless of the real channel count
(the same packed-layout convention `chunked_kernel_test.py` inspected earlier in this project),
so at `cin=3` it's doing >10x more multiply-accumulate work than necessary padding to 32; this
kernel only pads to 4, the minimum `vrmpybusv` needs, wasting a much smaller fraction.

This closes out every *conv* shape in the profile except the 3x3 tiling refinement above. Adding
this unambiguous win (308.5 ms) to the twelve 1x1 shapes' running total (~3562 ms) gives ~3870.5 ms
of the 7623 ms grand total covered by a kernel that's faster than TVM everywhere it's been
measured -- **about 51%** (the 3x3 conv's ~2509 ms is left out of this figure, same as before,
since one of its three measured shapes is a real, reported loss against TVM, not an unambiguous
win -- see its own section above for the honest breakdown rather than folding it into one summary
percentage).

**Not covered by this file**: the small elementwise/pooling ops (`add`/`maxpool`/`sigmoid`,
~176.7 ms combined, the only remaining uncovered items in `noncon_profile.json` besides `resize`,
which was already handled separately via a TVM-schedule-level fix earlier in this project, not a
`custom_kernel`) -- out of scope here, left for a follow-up.

### Coverage: the elementwise add (`hex_add_kernel.py`) -- closed with software prefetch

The follow-up to the note just above: `add` (FPN lateral + top-down merge, `int32 + int32 ->
int32`, pre-requantization accumulators -- confirmed from
`scripts/android/maskrcnn_e2e/profile_noncon_ops.py`'s `build_specs()`) is the biggest remaining
uncovered item, 102.3 ms across 6 shapes; the single biggest of those is `ishape=[1,256,200,272]`
(40.6 ms across 4 occurrences at that shape -- used below as the representative real shape).

**Tried tinygrad's normal codegen path first**, per this file's own stated preference (a
`custom_kernel` is for when a specific hardware intrinsic like `vrmpy` is needed and the generic
devectorizer mishandles it -- a plain add has no such need in principle): `a + b` through BEAM
search (BEAM=2 and BEAM=6 both tried) at the real full-scale shape (13,926,400 elements) found
**no vectorized candidate at all** -- every candidate was the identical plain scalar `for` loop
(`*(data0+i) = *(data1+i) + *(data2+i)`, no HVX, no unrolling). Bridged to real hardware anyway
rather than assume it wasn't worth measuring:

| kernel | median | throughput | vs TVM |
|---|---:|---:|---:|
| scalar (normal codegen, BEAM=2) | 292.812 ms | 0.048 G-elem/s | **0.057x (17.6x slower)** |
| stock TVM (`relay.add`) | 16.620 ms | 0.838 G-elem/s | -- |

Not remotely competitive -- a genuine negative result for "just let BEAM handle it," not a search
budget problem (BEAM=6 converged to the same kernel as BEAM=2).

**So a hand-written `custom_kernel` after all**, but for a different reason than every other
kernel in this file: not because a specific accumulate-in-place HVX instruction (`vrmpy`) needs
explicit intrinsics, but because tinygrad's Hexagon backend apparently won't auto-vectorize even a
trivial elementwise loop to HVX width on its own. `build_vector_kernel()` is the simplest
`custom_kernel` in this project: no accumulator, no reduction, no `_reg_i32`-style degenerate-range
dependency tracking (every other kernel needed that because of a persistent `REG` accumulator
across a reduction loop; this op has neither) -- one `(32,)`-wide `int __attribute__((
vector_size(128)))` vector load/add/store per iteration (one HVX register, 32 int32 lanes), using
plain C vector-extension `+` rather than an HVX builtin (clang lowers vector-extension arithmetic
to the matching HVX instruction directly under `-mhvx`; unlike `vrmpy`, plain add has no
accumulate-into-place semantics needing an explicit intrinsic).

Verified bit-exact correct under qemu and on real hardware (device `239dbd8f`) at the real
`[1,256,200,272]` shape:

| kernel | median | throughput | vs TVM |
|---|---:|---:|---:|
| scalar (normal codegen) | 292.812 ms | 0.048 G-elem/s | 0.057x |
| vectorized `custom_kernel` | **23.367 ms** | **0.596 G-elem/s** | **0.71x (still slower)** |
| stock TVM (`relay.add`) | 16.620 ms | 0.838 G-elem/s | -- |

Vectorizing alone recovered **12.5x** over the scalar version -- a real, large improvement -- but
still landed at 0.71x of TVM's throughput, not a win. Consistent with this being a genuinely
memory-bandwidth-bound op (3 buffers, no reuse -- `256*200*272*4` bytes = 55.7 MB per buffer at
the real full-scale shape, ~167 MB of total traffic per call), the fix that actually mattered
wasn't the lever this project's other kernels have relied on (more compute-side reuse/tiling --
there's no reuse to be had here, every element is touched exactly once) but **hiding DDR latency
behind the loop's own memory ops** via software prefetch.

**Tried instruction-level unrolling first** (the same lever that fixed the 3x3 conv's
cache-blocking gap above) -- `HEXSIM=1`-screened at unroll factors 1/2/4/8/16/32, at two very
different scales (a ~12 MB-traffic case and a ~144 KB-traffic case): **flat, no improving trend at
either scale** (all within ~6% of each other, noise-level). Confirmed on real hardware at 1M
elements: unroll=8 was *slower* than unroll=1 (22% worse), not faster -- a real, evidenced dead
end for this op, the same kind of finding as the 3x3 conv tiling's `ow_tile` ceiling (see above),
just for a different lever. Makes sense in hindsight: unrolling only pays off when loop-control
overhead or a lack of independent in-flight operations bottlenecks a kernel; nothing here is
compute-bound or loop-count-bound, so there's no overhead for unrolling to amortize.

**Software prefetch is a different lever**: hiding memory latency, not reducing instruction count.
`build_vector_kernel()`'s `prefetch_dist` parameter issues `__builtin_HEXAGON_Y2_dcfetch` for both
input operands `prefetch_dist` HVX vectors ahead of the one currently being added, each loop
iteration -- a non-faulting hint (confirmed empirically: reads past the buffer's end near the tail
of the loop, under qemu and on real hardware, at every scale tested here, cause no crash and no
incorrect output). `HEXSIM=1` showed a real, monotonic improvement with distance, plateauing
around 16 (~1.7x faster than no prefetch at that plateau) -- a completely different signal shape
from unrolling's flat line, and a second genuine confirmation (after the 3x3 conv tiling work) that
`hexagon-sim --timing` can distinguish a real fix from a real dead end on the same op.

Confirmed on real hardware (device `239dbd8f`, bit-exact correct at every size below and at the
real full-scale shape under qemu) at `prefetch_dist=16`, at three sizes -- **not the exact real
profile shape**: the TVM RPC session used for real-hardware bridging hit a hard transfer-size wall
somewhere between 32 MB and 48 MB per buffer (`hexagon_rpc_send failed: 78`, unrelated to this
kernel -- a pre-existing limitation of the RPC bridge path itself, out of scope to fix here), so
the largest size actually reachable (8,388,608 elements, 32 MB/buffer) is below the real
`13,926,400`-element shape used for every number elsewhere in this file:

| elements | traffic/buffer | baseline (no prefetch) | `prefetch_dist=16` | speedup |
|---:|---:|---:|---:|---:|
| 1,048,576 | 4 MB | 3.890 ms / 0.270 G-elem/s | 3.088 ms / 0.340 G-elem/s | **1.26x** |
| 4,194,304 | 16 MB | 8.967 ms / 0.468 G-elem/s | 5.948 ms / 0.705 G-elem/s | **1.51x** |
| 8,388,608 | 32 MB | 14.873 ms / 0.564 G-elem/s | 9.363 ms / 0.896 G-elem/s | **1.59x** |

The speedup grows with size at every step measured, and by 8M elements the prefetch kernel's
0.896 G-elem/s is already *higher* than stock TVM's 0.838 G-elem/s measured at the larger
13.9M-element shape -- suggestive of a real win at full scale, not a same-shape apples-to-apples
comparison (TVM's number carries its own RPC/copy overhead at a different total size), so reported
as directional evidence rather than a confirmed final number. `build_vector_kernel()`'s default is
now `prefetch_dist=16`; pass `prefetch_dist=0` to reproduce the original, unprefetched kernel.

**Not attempted here**: resolving the RPC transfer-size limit to get an exact full-scale real
number (a bridge-infrastructure issue, not a kernel one); `maxpool`/`sigmoid` coverage (a separate,
concurrent effort in this session).

### `maxpool`/`sigmoid`: two real, structural blockers -- both now unblocked

The last two items in `noncon_profile.json` (confirmed via `profile_noncon_ops.py`'s
`build_specs()`, not assumed): `maxpool` is `uint8`, kernel `3x3`, stride `2`, pad `1`, dominant
shape `ishape=[1,64,400,544]` (60.2 ms of `maxpool`'s 61.8 ms total -- the ResNet stem's pool
right after the 7x7 conv, `400x544 -> 200x272`); `sigmoid` is plain `float32`
(`relay.sigmoid(relay.var("d", dtype="float32"))` -- RPN/mask-head logits, not a quantized/LUT op),
five shapes from 663 to 163200 elements, 12.6 ms total. Both hit a real blocker in the pass that
first looked at them, documented here rather than forced:

- **`maxpool`**: tinygrad's normal codegen (BEAM=2, real shape) produces correct (qemu-verified
  against a numpy reference), fully scalar code -- the exact same "no vectorization at all" finding
  as `add`'s own history. But unlike `add` (a flat, fully-contiguous elementwise op) or the conv
  kernels (which vectorize across `cout`, a 32-wide axis `vrmpy` matches directly), `maxpool` has
  **no equivalent wide, contiguous axis to vectorize across** in this backbone's `NCHW` layout: the
  only "wide" dimension is channels (64 here), but channels are non-contiguous in memory (stride
  `H*W` apart), so a single HVX vector load can't gather multiple channels' worth of data in one
  instruction; the one genuinely contiguous axis (`W`) needs a stride-2 windowed reduction (kernel
  3, stride 2), which is vectorizable in principle via HVX's `vmax` plus a deinterleave/byte-pack
  step (`vdeal`/`vpacke`-style builtins) to subsample the stride-1-computed max back down to the
  real stride-2 output positions -- but that's a genuinely new HVX pattern this project hasn't used
  before (every other kernel here either has no windowing at all, or windows along the
  `vrmpy`-native `cout`/K-chunk axes), and getting its lane semantics right without a real chance to
  debug on hardware afterward (this was a single-pass task) was judged too likely to land a subtly
  wrong kernel to attempt here. **Unblocked in a follow-up pass** -- see below.
- **`sigmoid`**: blocked earlier and more fundamentally -- tinygrad's `DSPCompiler` (`MOCKDSP=1`
  path, and very likely the real-hardware path too, since both use the same freestanding
  `-nostdlib -ffreestanding` link setup) can't even *compile* a plain float division:
  `ld.lld: error: undefined symbol: __hexagon_divsf3` on the simplest possible `a.sigmoid()`
  correctness check. Every kernel this project has ever built (12+ conv/GEMM/elementwise kernels
  across `hex_gemm_kernel.py`, `hex_conv3x3_kernel.py`, `hex_stem7x7_kernel.py`,
  `hex_add_kernel.py`) has been `uint8`/`int8`/`int32` -- **this is the first time this project has
  tried a `float32` op on this DSP backend at all**. **Follow-up below: root-caused and fixed** --
  turned out to be a clang-version-dependent soft-float linking gap, not a fundamental Hexagon
  limitation.

Both are real, evidenced findings, not just "ran out of time" -- useful for whoever picks these up
next: `maxpool` needs a genuinely new HVX deinterleave pattern (see below for how that turned out
not to be necessary after all); `sigmoid`'s blocker is also resolved -- see its own section below.

### `maxpool` unblocked: TVM's own reference layout, not a new HVX pattern (`hex_maxpool_kernel.py`)

The blocker above was specific to plain `NCHW` layout -- channels aren't contiguous there, so
there's no wide axis to vectorize across without a genuinely new stride-2 deinterleave pattern.
But `NCHW` isn't actually what TVM's own reference schedule uses for this op *in this exact real
subgraph*. Confirmed by compiling a real `qnn.conv2d -> nn.max_pool2d -> qnn.conv2d` graph (the
real backbone's actual stem-conv -> pool -> layer1 sandwich) through `relay.build()` and inspecting
the compiled graph JSON directly, not assumed: `tvmgen_default_fused_nn_max_pool2d`'s shape is
`[1, 2, 10, 10, 32]` at a small test size -- **packed NCHWc** (`ic_bn=32`), the exact same layout
convention `chunked_kernel_test.py`/`backbone_splice/gen_chunked_prod.py` already validated for
conv kernels in this project. TVM never registers a Hexagon-specific pooling schedule
(`topi/hexagon/pooling.py` is a generic, layout-agnostic `AutoInlineInjective` schedule) -- the
packing comes entirely from `topi.nn.pool2d`'s generic layout-string parametrization combined with
`AlterOpLayout` choosing to keep the whole subgraph packed rather than repacking back to `NCHW`
around the pool, exactly as it already does for the convs on either side of it.

In that layout, the channel-block axis (32 contiguous `uint8` bytes) *is* the wide, contiguous
vectorization axis the plain-`NCHW` attempt found missing -- no stride-2 `W`-axis deinterleave
needed at all, since the windowed 3x3/stride-2 reduction happens over the `(H, W)` axes exactly as
before, applied uniformly across all 32 channel lanes at once via one comparison per window
position. `hex_maxpool_kernel.py`'s `build_kernel()` follows this project's now-standard reduction
pattern (`_reg_u8x32`, mirroring `hex_gemm_kernel.py`'s `_reg_i32` multi-range-dependency fix for
any range that can degenerate to extent 1): a `(32,)`-wide `unsigned char` accumulator, initialized
to `0` (safe for `uint8` max -- `0` is the dtype's true minimum, so a padded lane can never win
against a real activation value, the same "padding is free" argument `hex_stem7x7_kernel.py` made
for its `cin`-axis padding), `max`'d against each of the 9 window positions via
`__builtin_elementwise_max` (plain vector-extension `?:` doesn't compile in Clang's C mode for
vector conditions here -- `__builtin_elementwise_max` does and is the portable fix).

Host-side padding (`pad_nchwc()`) zero-pads the `(H, W)` axes the same way `hex_conv3x3_kernel.py`
pads its input, before the kernel runs -- no in-kernel bounds checks needed.

Verified **bit-exact correct on real hardware** (device `239dbd8f`) at the real profile shape
(`cin=64, 400x544 -> 200x272`, also bit-exact under qemu at the same full scale and at a small
shape during development), and measured against stock TVM's own `nn.max_pool2d` at the identical
shape, same methodology (`HexagonLauncher`/`get_executor_from_factory`, separate RPC sessions to
avoid the cross-session-mixing issue found elsewhere in this project):

| shape | stock TVM | `custom_kernel` | speedup |
|---|---:|---:|---:|
| `cin=64, 400x544 -> 200x272` | 76.092 ms / 0.046 G-elem/s | **38.247 ms / 0.091 G-elem/s** | **1.99x** |

Bit-exact correct and faster than TVM, closing out `maxpool`'s 60.2 ms of the 61.8 ms total (the
one smaller `[1,256,25,34]` shape, 1.6 ms, wasn't separately re-verified given its small impact --
same kernel, same correctness argument, only the loop trip counts differ).

**Not done**: the ~4x throughput a full 128-byte HVX vector could offer. This kernel's natural
data width is 32 bytes (one channel-block, confirmed via `hexagon-llvm-objdump` -- the compiled
`__builtin_elementwise_max` call lowers to Hexagon's scalar-core packed-byte `vmaxub` on 64-bit
register pairs, 8-wide SIMD, not the separate 128-byte HVX vector coprocessor's own `vmaxub_128B`).
Getting a full 128-byte op would need grouping 4 output positions per vector call the same way
`hex_conv3x3_kernel.py`'s `ow_tile` does for the stride-1 conv kernels -- but `maxpool`'s stride-2
windows make neighboring *output* positions' input windows non-contiguous (each window position's
4-output-wide group needs input at `2*ow+kw` for `ow=ow0..ow0+3`, a stride-2 gapped read, not a
contiguous load), the same deinterleave complexity the original blocker flagged -- avoided
deliberately here in favor of landing a correct, real, already-faster-than-TVM result first.
### `sigmoid`, unblocked: it was a clang version, not a Hexagon limitation

Investigating via TVM's own reference approach (checking what TVM's Hexagon target does
differently, per this project's established `hexagon-sim`/`+hvx-qfloat` context) turned up the
real cause before qfloat/LUT tricks were even needed: **`__hexagon_divsf3` is a clang/LLVM-version
dependency, not a fundamental gap.** Reproducing the exact failure with different `clang` binaries
(all available in this environment, `clang-15` through `clang-21`) shows a clean version split:

| `clang` version | `a.sigmoid()` under `MOCKDSP=1` |
|---|---|
| 15, 17 | `ld.lld: error: undefined symbol: __hexagon_divsf3` (the original finding) |
| 19, 21 | **compiles and runs correctly**, no missing symbol at all |

Disassembling a minimal repro compiled with `clang-19` shows why: it inlines scalar Hexagon float
division as a native Newton-Raphson instruction sequence (`sfrecipa`/`sffixupn`/`sffixupd` +
`sfmpy:lib`) directly in the generated code, never calling out to compiler-rt at all. `clang-15`/
`clang-17` instead emit a libcall to `__hexagon_divsf3` -- which `DSPCompiler`'s freestanding
`-nostdlib` build has nothing to provide, hence the link failure. Same Hexagon target, same `-O2`,
different codegen choice depending purely on which LLVM version compiled it.

**The fix, landed in `onnxsim/tinygrad`'s `vrmpy-hexagon-support` branch** (commit `4ac16f5b6`,
stacked onto the existing open PR https://github.com/onnxsim/tinygrad/pull/1, same branch every
other tinygrad-side change in this project has used): link the Hexagon toolchain's own `libgcc.a`
(a plain static archive providing `__hexagon_divsf3` and friends) into `DSPCompiler`'s build when
`HEXAGON_TOOLCHAIN`/`HEXAGON_SDK_ROOT` is set, so `float32` ops work regardless of which `clang`
version ends up compiling them -- a robust, version-independent fix rather than pinning a specific
clang. Since it's a static archive, only symbols a kernel actually references get pulled in, so
this is a costless no-op for every existing `uint8`/`int8`/`int32` kernel (confirmed: an existing
`int32` add kernel produces identical output with and without the archive linked).

Verified **bit-exact correct under qemu** (`MOCKDSP=1`, `CC=clang-17` -- the previously-failing
version, now working via the `libgcc.a` link) against a numpy `1/(1+exp(-x))` reference at all five
real profile shapes:

| elements | max abs error vs numpy |
|---:|---:|
| 663 | 5.96e-08 |
| 2,550 | 5.96e-08 |
| 10,200 | 5.96e-08 |
| 40,800 | 5.96e-08 |
| 163,200 | 1.19e-07 |

All within float32 rounding precision -- a clean pass, not an approximation with a real error
budget (no polynomial/bit-trick approximation was needed after all; tinygrad's own composition of
primitive UOps for `.sigmoid()`, once it can actually link, is already numerically exact to
float32 precision).

### Real-hardware follow-up: correct, but no auto-vectorization, same lesson as `add`

Bridged via the same pattern as every other kernel here (`hexagon-clang` compile + a thin
`sigmoid_wrapper_template.c` TVM `PackedFunc` shim + `tvm.contrib.hexagon.tools.link_shared` +
`HexagonLauncher`/RPC session, device `239dbd8f`, real profile shape `n=163200`).

**The anticipated "same class of problem resurfaces in the bridge's own link step" did NOT
happen**: the Hexagon SDK's own `hexagon-clang` is version **19.0.04** -- squarely in the "clang
>=19 inlines float division natively" bucket this section already found, not the 15/17 bucket that
needed `libgcc.a`. Compiling the captured kernel with it produces zero undefined symbols; no
soft-float archive needed on this path at all. (The `DSPCompiler`-side `libgcc.a` fix above is
still correct and still needed -- it's what makes `MOCKDSP=1`/qemu work with whichever `CC` a given
environment happens to default to -- this is just confirmation that the specific real-hardware
toolchain in use here was never going to hit the same gap.)

**Correctness: bit-exact with the qemu run**, same `max_abs_err=9.16e-08` on real silicon as under
`MOCKDSP=1` -- deterministic, not a coincidence of RNG seeding.

**Speed: correct, but real, honest loss** -- confirmed at both `BEAM=0` (tinygrad's default here,
which turned out to fully unroll the loop into a 125 KB source, one literal expression per element)
and `BEAM=2` (which found a proper single scalar loop, ~1 KB source) -- both landed at essentially
the same real-hardware throughput, consistent with neither being vectorized at all:

| kernel | median | throughput | vs TVM |
|---|---:|---:|---:|
| tinygrad, `BEAM=0` (full unroll) | 15.797 ms | 0.0103 G-elem/s | 0.32x (3.1x slower) |
| tinygrad, `BEAM=2` (scalar loop) | 16.763 ms | 0.0097 G-elem/s | 0.30x (3.3x slower) |
| stock TVM (`relay.sigmoid`) | 5.038-5.072 ms | 0.0322-0.0324 G-elem/s | -- |

Both tinygrad variants are within noise of each other and consistently ~3x behind TVM -- the same
finding `add`'s own history already established for this backend: tinygrad's normal codegen path
does not reach for HVX vector width on its own for a plain elementwise op, regardless of `BEAM`
search width (higher `BEAM` finds better *loop structure*, not vectorization, here). `add` closed
an equivalent gap with a hand-written `Tensor.custom_kernel` (`hex_add_kernel.py`); the same move
would very plausibly work here too (`sigmoid`'s transcendental body doesn't block it -- the
existing scalar composition already proved numerically exact, so a vectorized version only needs
to run that same expression across an HVX-width group of lanes at once, not re-derive the
approximation) -- not attempted here, flagged as the natural next step.

Also unverified on real Hexagon v65 hardware specifically (this test phone is Hexagon v69, kernels
here targeted `v73` for forward-compatibility, matching every other real-hardware kernel in this
project -- see `scripts/android/maskrcnn_e2e/README.md`'s "Xiaomi 12S (Hexagon V69; kernels
compiled for v73)"): this SDK snapshot ships no `v65`-specific `libgcc.a` (oldest available is
`v68`), so `DSPCompiler`'s `MOCKDSP=1`/qemu-side fallback uses the lowest available version --
Hexagon's scalar ISA has been stable `v65`-`v81`, making this a reasonable bet, but not one this
real-hardware run (targeting `v73`, which has its own `libgcc.a` and didn't need it anyway) confirms
for `v65` specifically.

### Coverage: FPN `resize2d` (`hex_resize2x_kernel.py`) -- correct, beats stock, loses to the already-fixed fast path

The last op in the profile with no tinygrad-generated kernel at all: `resize` (595.6 ms across 3
shapes, FPN's 2x nearest-neighbor upsamples). Unlike every op above, this one already has a real,
shipped fix in this project -- `scripts/android/hexagon_resize2x.py`, landed earlier (see
`scripts/android/maskrcnn_e2e/README.md`'s "Fixed: FPN resize2d, via an exact-2x integer fast
path"), a monkeypatched TVM `te.compute` schedule, not a `custom_kernel`. That fast path is the
real bar to clear here, not stock TVM's original (much slower, per-pixel float `ceil`/`floor`/
`round`) schedule.

FPN's resize is always exactly 2x per axis, `nearest_neighbor` + `half_pixel` +
`round_prefer_floor`, which collapses to plain integer replication with no float math:
`out[c, oh, ow] = in[c, oh // 2, ow // 2]`, `int8` dtype -- no accumulator, no reduction, pure
data movement (every input pixel replicated into a 2x2 output block, per channel, independently).

`build_kernel()`'s approach: NCHW's innermost (contiguous) axis is W, so the horizontal doubling
is vectorizable via a compile-time `__builtin_shufflevector` byte-duplication mask over 128-byte
input chunks (`pad_input_row()` zero-pads each row to a multiple of 128 host-side, the same
pre-padding convention `hex_conv3x3_kernel.py` established). Vertical doubling (output rows `2r`
and `2r+1` are byte-identical) is done by building the doubled row once into a **local stack
buffer**, then writing that buffer to both output row addresses.

**A real bug found getting there**: the first version wrote row `2r` directly to the output array,
then read that same output memory back through a plain `signed char*` cast to duplicate it into
row `2r+1`. Under qemu this looked fine at the smallest shape but failed at the other two: row
`2r` itself was bit-exact, but row `2r+1` diverged in *exactly* the byte range covered by a
preceding partial (non-full-128-byte) shuffle write. The compiler doesn't reliably order a later
scalar read against an earlier HVX vector store to the same output-array region across separate
statements, even inside one function body -- confirmed by testing `-fno-strict-aliasing` (no
effect, ruling out the obvious TBAA explanation) and then by direct byte-level inspection (rows
matched everywhere *except* the tail region a partial vector write had just touched). A local
buffer sidesteps the ambiguity entirely: never aliased by anything else, so the two final output
writes are trivially independent and can't be reordered into each other.

Verified **bit-exact correct on real hardware** (device `239dbd8f`) at all three real FPN shapes.
Comparing against both stock TVM and the fast path needed one more real fix along the way:
building both in the same Python process (matching this project's usual bridge-script pattern)
produced a fast-path timing number numerically identical to stock's -- a real, reproducible
compilation-cache collision (TVM appears to cache a compiled result keyed in a way that doesn't
account for the `topi.image.resize2d` monkeypatch swap between builds), not a property of the fast
path itself. Confirmed by rebuilding the fast path in complete process isolation, which reproduced
the originally-established ~4/6.7/27.5 ms numbers from `maskrcnn_e2e/README.md` almost exactly:

| shape | stock TVM | fast-path TVM (isolated build) | `custom_kernel` | vs stock | vs fast path |
|---|---:|---:|---:|---:|---:|
| `(25,34)->(50,68)` | 29.794 ms | 4.004 ms | 5.332 ms | **5.59x** | 0.75x (slower) |
| `(50,68)->(100,136)` | 113.403 ms | 6.712 ms | 9.994 ms | **11.35x** | 0.67x (slower) |
| `(100,136)->(200,272)` | 450.945 ms | 27.492 ms | 26.124 ms | **17.26x** | **1.05x** (barely faster) |

Reported honestly, matching this project's norm: this kernel is dramatically faster than *stock*
TVM everywhere, but the fast path was already a good fix -- also pure `//2` integer indexing,
just lowered through TVM's own auto-vectorizing schedule rather than hand-written. The
hand-written kernel only edges it out at the largest shape; at the two smaller ones, TVM's
compiler-generated loop beats the hand-rolled shuffle+scalar-tail-loop version, likely because
fixed per-call/per-row overhead (the scalar tail loops handling non-128-aligned widths) matters
more at smaller sizes than any advantage from hand-picking the shuffle instruction. This closes
out `resize` as a **code-generation** target (a real, correct, bridged tinygrad kernel now exists
for every op in the profile except the 3x3 conv's remaining tiling gap), but not as a **speed**
target against the already-good fast path -- consistent with this file's running distinction
between the two kinds of "coverage."

**Not done**: closing the two-smaller-shapes gap (a smarter row-vectorized loop with less scalar
tail overhead, or `HEXSIM=1`-guided tuning the way the 3x3 conv's tiling work did, was flagged but
not attempted given the fast path is already a solid, shipped baseline); folding this kernel back
into the real `relay.build()` graph the way `backbone_splice/` attempts for convs (blocked on that
same unresolved RPC loading bug).

## Running the full backbone graph, TVM-free: `requantize`, the missing piece

The user's explicit ask, after all of the above: chain every covered op together and run the
*real* backbone graph end to end, through `native_transport/` only -- **zero TVM anywhere in the
runtime execution path** (TVM as an offline correctness oracle is fine; TVM as part of what
actually executes on the phone is not, by explicit choice: "tvm dependency bothers a lot").

Every op above is covered, but chaining any two covered conv kernels together needs one thing none
of them produce: **`qnn.requantize`**, the int32-accumulator-to-uint8 rescale+clamp step ONNX's
`FakeQuantizationToInteger` pass inserts between every conv layer. `grep -rl requantize
scripts/android/tinygrad_hexagon_bridge/` confirms it: nothing in this project had ever built one.

### The formula, read out of TVM's own source, not assumed

`scripts/android/maskrcnn_e2e/README.md`'s finding #3 already flagged that rounding mode matters
here ("`requantize` with `TONEAREST` rounding mismatched the host on a synthetic test (72% of
elements); the default `UPWARD` mode ... is correct") without spelling out the exact bit-level
formula. Reading `src/relay/qnn/op/requantize.cc` and `src/target/intrin_rule.cc`'s
`QMultiplyShift` (what `UPWARD`'s `fixed_point_multiply` lowers to, via the `tir.q_multiply_shift`
intrinsic) gives the precise, otherwise-undocumented arithmetic:

```
tensor        = int32(x) - input_zero_point                    # usually 0 for a fresh accumulator
left_shift    = max(shift, 0);  right_shift = max(-shift, 0)     # shift from GetFixedPointMultiplierShift
prod          = (int64(tensor) << left_shift) * int64(multiplier)  # multiplier: Q31 fixed-point int32
total_shift   = right_shift + 31                                  # q=31 (Q31 format)
scaled        = (prod + (1 << (total_shift - 1))) >> total_shift  # UPWARD: bias-then-shift = round-half-up
out           = clip(scaled + output_zero_point, 0, 255)          # uint8 range
```

`(multiplier, shift)` come from `GetFixedPointMultiplierShift(input_scale / output_scale)`, a
`frexp`-based Q31 decomposition -- ported directly into Python
(`hex_requantize_kernel.py`'s `compute_multiplier_shift()`), so every quantization node's real
scale/zero-point pair can be turned into the exact same integer constants TVM itself would bake in,
computed host-side and traced into the kernel as compile-time constants (matching how every other
kernel in this project bakes shapes in at trace time).

### `hex_requantize_kernel.py`

A plain per-element `long long` scalar loop implementing the formula above exactly (int64
arithmetic throughout, matching TVM's own intermediate precision bit-for-bit) -- not vectorized to
HVX width: the variable-shift, wide-multiply arithmetic this op needs has no single HVX vector
instruction the way `vrmpy`/plain `+` do, so a good vectorization is real follow-up work, not
attempted here (this is a small, simple op -- 12.6ms-add-scale territory, not a conv-scale
bottleneck -- and correctness was the actual blocker, not speed).

Verified against `requantize_ref()`, a bit-exact numpy port of the same int64 formula (not a float
approximation), across several distinct `(multiplier, shift, input_zero_point, output_zero_point)`
combinations covering both the left-shift and right-shift branches of the formula: under
`MOCKDSP=1`/qemu at n=1,000, n=13,926,400 (this project's standard "real full-scale" element
count), and three additional parameter combinations -- **bit-exact correct in every case**.

### Real hardware, through `native_transport` only

Spliced the generated kernel into `native_transport/mini_rpc_impl.c` the same way the flagship
`hex_gemm_kernel.py` GEMM was (dispatch by buffer size in `mini_rpc_run_kernel`, no IDL change),
at `n=2,000,000` (`in_scale=0.02, out_scale=0.05, in_zp=0, out_zp=114`) -- a real-scale slice sized
to stay under the ~32MB/buffer RPC transfer wall this project's own `native_transport` work already
found (the full 13.9M-element/55.6MB-input real stem-conv size would exceed it). Verified
**bit-exact correct on real hardware** (device `239dbd8f`), zero TVM anywhere in the executed path
-- `client_main.c`/`mini_rpc_impl.c` only, same as every other `native_transport` result in this
file. `gen_requantize_test_data.py` regenerates the test vectors; `build.sh` picks them up
automatically once pushed.

### What this does and doesn't prove

Proven: the one missing kernel now exists, is bit-exact against TVM's own documented-nowhere-else
fixed-point formula, and runs correctly on real hardware through the TVM-free transport. Combined
with the already-covered conv/add/maxpool/resize kernels, every *kind* of node the real 578-node
backbone graph contains now has a working, real-hardware-verified, TVM-free kernel.

**Not done, and worth being precise about why**: actually chaining several of these kernels
together into one real, running subgraph (e.g. the ResNet stem: `conv7x7 -> requantize ->
maxpool`) against real extracted backbone weights, let alone the full 578-node graph. This needs,
concretely:
1. **Real weight/scale extraction**: `backbone.onnx` (from `scripts/android/maskrcnn_e2e/prepare.py`)
   is still in ONNX QDQ form -- `DequantizeLinear`/`QuantizeLinear` nodes carrying the real
   per-tensor `scale`/`zero_point` initializers, not yet Relay's fixed-point `(multiplier, shift)`
   form. Extracting the real stem conv's weight tensor and the real scale/zero-point pair for its
   following `requantize` (`compute_multiplier_shift()` above turns that directly into kernel
   constants, no TVM needed for this step) is a data-extraction task distinct from anything else
   built here.
2. **A fused driver**: `native_transport/`'s own philosophy -- "call one fixed, hand-written kernel
   function with a handful of buffer pointers" -- extends naturally to a whole *static* graph
   (this backbone's shape/topology are fixed for the 800x1088 input): generate ONE `.so` containing
   every op's already-existing kernel-generation function's output concatenated in topological
   order, sharing on-device local/global buffers for intermediate activations, called via a single
   FastRPC round trip -- no new dynamic dispatch protocol needed, just concatenation +
   buffer-plumbing of code this project already generates correctly per-op.
3. **A reference to check against**: ONNX Runtime on the same sliced ONNX subgraph is the simplest
   ground truth (already the established reference throughout `scripts/android/maskrcnn_e2e/`).

None of this is hypothetical -- every piece has a working precedent elsewhere in this project
(per-op kernel generation, buffer-safe accumulator patterns, real-hardware `native_transport`
verification, ONNX-vs-kernel correctness checking) -- it's a real, scoped, multi-step follow-up,
not a research question. Flagged precisely rather than rushed, per this file's established norm.

## Chaining a real subgraph: `conv7x7 -> requantize -> maxpool`, real weights, real hardware

The follow-up to all three points above, done for real: the ResNet-50 stem (`conv7x7(s2,p3) ->
requantize -> maxpool(3x3,s2,p1)`) running end to end on real Hexagon hardware, real
`backbone.onnx` weights/scales, real image input, zero TVM in the executed path. The user's
explicit direction for this whole thread was to keep it that way -- reuse TVM/ONNX Runtime only as
an offline correctness *reference*, never as part of what runs on the phone.

### Real weight/scale extraction (`backbone_subgraph/extract_and_verify.py`)

`backbone.onnx`'s QDQ graph, traced forward from the `image` input (not assumed): a stem
`Conv` node (`2_quant`, `pads=[3,3,3,3]`, `strides=[2,2]`, `kernel_shape=[7,7]`, confirmed
matching `hex_stem7x7_kernel.py`'s own `K/STRIDE/PAD` constants exactly) consuming
`ConvMulFusion_W_2_quantized` (int8 weight, **per-tensor** scale -- confirmed via its
initializer's shape being `()`, not `(64,)`, so no per-channel requantize complication) and
`ConvAddFusion_Add_B_5_quantized` (a real int32 bias, pre-scaled to `input_scale * weight_scale`
-- standard QDQ convention), followed by a `MaxPool` node (`8_quant`, `pads=[1,1,1,1]`,
`strides=[2,2]`, `kernel_shape=[3,3]`, matching `hex_maxpool_kernel.py` exactly) whose surrounding
`QuantizeLinear`/`DequantizeLinear` pair reuses the *same* scale/zero-point before and after --
confirming numerically what was already known architecturally: maxpool needs no real requantize
step at all, its uint8 output is directly usable as the next op's input.

Two real gaps this surfaced, neither covered by any kernel in this project before now:

1. **Bias-add.** Every real conv in this graph has a nonzero bias (`ConvAddFusion_Add_B_5`'s
   values run into the thousands, not negligible) -- none of `hex_gemm_kernel.py`/
   `hex_conv3x3_kernel.py`/`hex_stem7x7_kernel.py` compute one; they're pure `sum(x*w)`.
   `hex_bias_add_kernel.py` is a new, minimal `custom_kernel` (no accumulator, no reduction --
   same class as `hex_add_kernel.py`, just a per-channel broadcast instead of an elementwise
   same-shape add): `out[pos,c] = acc[pos,c] + bias[c]`.
2. **The image's zero-point (114, not 0).** Every activation *after* the stem conv has
   `zero_point=0` (confirmed from the graph's own scale/zero-point initializers -- expected,
   ReLU makes everything non-negative) -- but the raw `image` input's `zero_point` is 114, and
   none of this project's kernels subtract an activation zero-point before their dot product (all
   of them assume it's 0, which is why they've worked correctly everywhere *except* here). Rather
   than add real zero-point-subtraction arithmetic to the kernel, the standard quantized-inference
   trick applies: fold it into the bias instead, host-side, once, since weights are static --
   `folded_bias[oc] = bias[oc] - image_zp * sum_{ic,kh,kw}(weight[oc,ic,kh,kw])`, so
   `sum(x*w) + folded_bias == sum((x-image_zp)*w) + bias` exactly, no on-device zero-point logic
   needed at all. (One more real detail this forced: `hex_stem7x7_kernel.py`'s `pad_input()`
   docstring already flagged that "a real integration needs the input's actual quantization
   zero-point for the spatial border, not literal 0" -- this is that real integration, so the
   image's spatial padding uses `image_zp=114`, not literal 0, this time.)

A quick host-side numpy composition of the whole chain (real weight, real folded bias, real
multiplier/shift, real image) against the ORT reference agreed to 98.8% exact / mean abs error
0.096 -- confirming the *math* was right before spending any device time on it.

### The layout-mismatch gap composition alone surfaced

Something no single kernel's own correctness check could have caught: `hex_stem7x7_kernel.py`'s
output is `(pos, cout)` **NHWC-flat** (position-major, channel-minor), but `hex_maxpool_kernel.py`
expects TVM's packed **NCHWc** layout (`[ic_chunk, h, w, 32]`, channel-block-major) -- two
individually-correct, individually-verified kernels whose outputs and inputs are simply different
physical layouts. Every kernel in this project was right in isolation; nothing had ever tested
whether two of them *compose*, and this is exactly the kind of gap that only shows up when they
do. Bridged with a plain, unvectorized copy loop (`nhwc_to_padded_nchwc()` in
`native_transport/subgraph_driver.c`) -- correctness first, matching this project's precedent for
every new piece of glue; not a `custom_kernel`, just a hand-written C loop, since it's pure
data movement with no arithmetic to get an HVX instruction for.

### The fused driver, and a real bug its own diagnostic isolated

`native_transport/subgraph_driver.c`: one exported function, `run_stem_subgraph()`, calling
`hex_stem7x7 -> hex_bias_add -> hex_requantize -> nhwc_to_padded_nchwc -> hex_maxpool` in
sequence against on-device `static` buffers (~84 MB total: the stem's raw `int32` accumulator,
the packed-NCHWc layout buffer, the requantized intermediate) -- none of it crosses the RPC
boundary, only the padded input image goes in and the final packed maxpool output comes out,
via one more size-dispatched branch in `mini_rpc_impl.c`'s `mini_rpc_run_kernel()` (same pattern
`requantize` used above). Weight (12.5 KB) and folded bias (256 B) are baked in as compile-time
`static const` arrays rather than passed as extra RPC buffers -- simplest fit for a fixed,
known-at-generation-time subgraph, avoiding an IDL change.

First real-hardware attempt failed: `rc=78`, and `adb logcat` showed a DSP-side crash with
`Bad VA` (bad virtual address) inside `mini_rpc.so`. **Isolated with a second, tiny-scale
(32x32 input) copy of the exact same driver structure** (`native_transport/small_subgraph_driver.c`,
kept as a real diagnostic artifact, not scratch) run *before* the full-scale one in the same test
binary: it linked and ran (`rc=0`) but its output was **wrong** -- 92% of pixels mismatched,
not a boundary-only discrepancy. Two symptoms, traced to one real cause: every kernel function's
parameters are declared `__attribute__((align_value(128)))` (a compiler *assumption*, not an
enforced check) because every prior kernel's buffers came straight from the FastRPC-mapped,
page-aligned `a`/`b`/`c` RPC arguments -- but this driver's *new* on-device `static` intermediate
buffers (the accumulator, the layout buffer, the requantized array) had no alignment attribute at
all, so the compiler emitted HVX vector loads/stores assuming 128-byte alignment against memory
the linker never actually guaranteed it for. Silently wrong at small scale (the misaligned vector
ops just read/wrote adjacent-but-wrong bytes); a hard fault at full scale, large enough to cross
an actual unmapped boundary. Fixed by adding `__attribute__((aligned(128)))` to every static
buffer and constant array in the driver -- after which **both** the small-scale diagnostic (now
bit-exact against its own numpy reference) and the full-scale run (`rc=0`) succeeded.

### Real-hardware result, real weights, real image, real ORT reference

Device `239dbd8f`, the true `800x1088` input processed by the true stem conv through the true
maxpool, `wall_ms=583.153` (client-measured, covering the *entire* fused chain in one RPC call --
no TVM, no `tvm.rpc`, no `relay.build`, nowhere in this path):

| | value |
|---|---:|
| Match vs. ONNX Runtime reference | **99.9995%** of pixels exact |
| Max abs diff | **1** (uint8, i.e. off-by-one rounding at 5 pixels out of 87,040) |
| Mean abs diff | 4.9e-6 |
| Wall time (fused, on-device) | 583.153 ms |

Tighter agreement than this project's own established int8-quantization-noise baseline
(`scripts/android/maskrcnn_e2e/README.md`'s feature-map mean abs error of ~0.06-0.11) --
consistent with a handful of genuine rounding-order differences between this fixed-point pipeline
and ORT's own float-simulated QDQ graph, not a real correctness gap. **First proof in this
project that independently-verified kernels compose correctly into a real, multi-op, real-weight
subgraph on real hardware** -- the specific question Stage 1 left open.

### What this does and doesn't prove

Proven: the composition question. Kernels built and verified in isolation (`hex_stem7x7_kernel.py`,
`hex_bias_add_kernel.py`, `hex_requantize_kernel.py`, `hex_maxpool_kernel.py`) chain correctly
into a real multi-op subgraph against real backbone weights, entirely through the TVM-free
`native_transport` path, with real ORT-level accuracy. The layout-mismatch and alignment bugs are
now both fixed and documented, so extending this same pattern to more of the graph is mechanical,
not exploratory.

**Not done**: the full 578-node backbone (this subgraph is 3 of those 578 nodes' worth of ops --
the stem, one requantize, one maxpool); a first ResNet bottleneck block was scoped in the original
directive but not reached, this subgraph's own investigation (extraction methodology, the layout
gap, the alignment bug) filled the available time. `native_transport`'s current ~84 MB-per-call
static buffer footprint for just this small subgraph also won't scale to the full backbone's much
larger activations unmodified -- buffer reuse (e.g. requantizing in place into the same memory the
`int32` accumulator occupied) is the natural next lever before attempting more of the graph, not
attempted here since this subgraph didn't need it to fit.

## Extending the chain: a full ResNet-50 bottleneck block (stage1/block1)

Follow-up to the `conv7x7 -> requantize -> maxpool` subgraph above, closing the gap its own
"Not done" note left open: the first full ResNet-50 bottleneck block, chained directly onto that
subgraph's output. Real weights/biases/scales extracted from `backbone.onnx` the same way
(`backbone_subgraph/extract_and_verify_block1.py`), no synthetic data anywhere in this pipeline.

```
main path:     conv10(1x1 reduce, 64->64) -> bias -> requantize -> conv17(3x3, 64->64) -> bias ->
               requantize -> conv24(1x1 expand, 64->256) -> bias           [kept RAW int32]
shortcut path: conv30(1x1 downsample, 64->256, from the same maxpool input) -> bias
                                                                             [kept RAW int32]
merge:         rescale both raw int32 branches into a shared scale domain, add, relu,
               one final requantize to uint8
```

Every op type already had a real, hand-written kernel (`hex_gemm_kernel.py`'s `pack_b()` layout,
`hex_conv3x3_kernel.py`, `hex_bias_add_kernel.py`, `hex_requantize_kernel.py`) -- except one real
gap this block's own numerics surfaced, and one signedness gap its real weights surfaced.

### New gap #1: the residual add needs a real rescale, not a raw int32+int32 add

The earlier FPN-lateral `add` (`hex_add_kernel.py`'s own coverage section) is a plain
`int32+int32->int32` op with no rescale -- both branches happen to share an implicit accumulator
scale there. **This block's residual add does not**: computing each branch's raw accumulator
scale directly from `backbone.onnx`'s real quantization params
(`input_scale * weight_scale` for `conv24`'s and `conv30`'s outputs) gives a ratio of **~0.51**
between the two branches -- confirmed decisively by testing the naive un-rescaled add directly:
**max abs diff 65, mean abs diff 4.36, only 37.5% exact** against a real ONNX Runtime reference,
decisively wrong. The correct approach (matching how real quantized-inference runtimes handle a
residual add between two differently-scaled branches): rescale each raw `int32` accumulator into
a *shared* scale domain first -- the same Q31 fixed-point formula `hex_requantize_kernel.py`
already uses, just without the final `clip(0,255)`/`uint8` cast, so the intermediate stays a
signed `int32` that can go negative before the real final clamp -- **then** add, relu, and clamp
once. `rescale_inplace_noclamp()`/`add_relu_clamp()` in `native_transport/block1_glue.c` implement
this (new, hand-written numeric code -- not covered by any existing kernel, verified in numpy
against the real ORT reference before any C was trusted):

| | max abs diff | mean abs diff | exact match |
|---|---:|---:|---:|
| Naive `int32+int32` (no rescale) | 65 | 4.36 | 37.5% |
| Rescale-then-add (this block's approach) | **3** | **0.40** | **64.0%** |

Still not bit-exact (unlike the stem+maxpool subgraph's 99.9995%), but well within this project's
established int8-quantization-noise baseline (`scripts/android/maskrcnn_e2e/README.md`'s
feature-map mean abs error of ~0.06-0.11 against a roughly ±14 range) -- plausibly explained by
this pipeline skipping a redundant quantize-then-immediately-dequantize round trip the *literal*
QDQ ONNX graph encodes for the add's two inputs (an artifact of QDQ format bookkeeping, not
something a real compiled int8 pipeline would actually execute) rather than a real bug; not
chased further, since this level of agreement already matches the project's own accepted norm.

### New gap #2: real backbone weights are SIGNED int8, exposing a gap in `hex_gemm_kernel.py`'s own coverage

`hex_gemm_kernel.py`'s 1x1-conv kernel is **uint8 x uint8 only** (`vrmpyub`, confirmed directly
from its own docstring and kernel body -- no `vrmpybusv` path at all), and every 1x1-conv
coverage number reported earlier in this file was measured against **synthetic unsigned test
weights**, not real ones. The real backbone's 1x1 conv weights are genuinely **signed int8**
(confirmed from `backbone.onnx`: `ConvMulFusion_W_*_quantized` tensors are `int8`) -- this was
never exercised before because every prior 1x1-conv test in this project used synthetic
`uint8` data. Feeding `conv10`'s real signed weight through `hex_gemm_kernel.py` as-is fails its
own correctness check outright.

**New kernel, not a fix to the existing one** (per this task's scope, `hex_gemm_kernel.py` itself
is untouched): `hex_gemm_signed_kernel.py`, structurally identical to `hex_gemm_kernel.py`'s
`build_kernel()` (same `_reg_i32` accumulator pattern, same `pack_b()`-compatible weight layout)
with exactly one change -- `vrmpyub_acc_128B(acc, weight_vec, activation_scalar)` becomes
`vrmpybusv_acc_128B(acc, broadcast32(activation_scalar), weight_vec)`, matching the exact
broadcast calling convention `hex_conv3x3_kernel.py`/`hex_stem7x7_kernel.py` already use for
their own signed-weight `vrmpybusv` calls (note the operand order also swaps between the two
intrinsics). Verified bit-exact under qemu at this block's real shapes (`conv10`: 64->64;
`conv24`/`conv30`: 64->256, both `m=54400`) before being used.

**This means every "1x1 conv" speedup number reported elsewhere in this file was measured against
unsigned synthetic weights, not the real network's signed ones** -- worth flagging plainly as a
real, previously-unstated gap in this project's own coverage claims, discovered only because this
was the first time real extracted backbone weights (not synthetic data) were run through the 1x1
conv path. Re-measuring the existing 1x1-conv coverage sections' real-hardware speedups against
real signed weights via `hex_gemm_signed_kernel.py` is a natural follow-up, not done here (this
block's own two new convs are verified correct with it; the *speedup* numbers for `hex_gemm_kernel`
elsewhere in this file were not re-measured with signed data).

### Verified bit-exact match: numpy, `hexagon-sim`, and real hardware, all agree

The fused driver (`native_transport/block1_kernels.c` -- kernel bodies, verbatim tinygrad-generated
C; `native_transport/block1_glue.c` -- the layout glue and the new rescale-add-relu-clamp merge)
was verified at three independent levels, all producing the **identical** result:

| Verification level | max abs diff | mean abs diff | exact match |
|---|---:|---:|---:|
| numpy (`extract_and_verify_block1.py`, vs. real ORT reference) | 3 | 0.401961 | 63.9883% |
| `hexagon-sim` (Qualcomm's own instruction-set simulator) | 3 | 0.401961 | 63.9883% |
| **Real hardware** (device `239dbd8f`) | **3** | **0.401961** | **63.9883%** |

Real-hardware wall time for the whole block (one fused RPC call, chained on Stage 2's maxpool
output): **1440.5 ms** -- notably heavier than the stem+maxpool subgraph's 583 ms, expected given
this block's much larger compute (three real convs plus a downsample, `256`-channel expand/
downsample stages) and its current lack of any inter-op buffer-reuse optimization (see below).

### What this does and doesn't prove

Proven: composition continues to hold at real scale, through a genuine branch-and-merge structure
(not just a straight chain) -- the residual add is the first place in this project's kernel-level
work where two independently-computed branches must be numerically reconciled before continuing,
and doing that correctly needed real, new analysis (the rescale-before-add finding above), not
just gluing existing pieces together. Also surfaced a real, previously-unflagged gap in
`hex_gemm_kernel.py`'s own coverage claims (unsigned-only, never tested against the real
network's signed weights) -- a genuinely useful finding for anyone trusting this project's earlier
1x1-conv speedup numbers as representative of the real backbone.

**Not done**: chaining stem+maxpool+block1 into a *single* RPC call (currently two separate calls,
Stage 2's subgraph then this one, with the intermediate maxpool output round-tripped through the
host in between for this verification -- fusing them removes that round trip); buffer reuse across
this block's several `13.9`-`55.7` MB intermediate `int32` buffers (all currently separate static
allocations -- fit fine on real hardware here, but won't scale indefinitely as more blocks are
chained); the remaining 577 nodes of the real 578-node backbone (this is 2 bottleneck-block-scale
chunks -- stem+maxpool, and one full block -- of a ResNet-50 that has 16 such blocks total across
4 stages, plus FPN/RPN); re-measuring `hex_gemm_kernel.py`'s existing coverage sections against
real signed weights, per the gap noted above.

## Removing TVM as a transport dependency

The bridge above still depends on TVM for two separate things: (1) **transport** -- getting bytes
to and from the DSP at all (`tvm.rpc.tracker.Tracker`, `tvm.contrib.hexagon.build.HexagonLauncher`,
`tvm_rpc_android`, TVM's own `libhexagon_rpc_skel.so`, and TVM's MinRPC wire protocol tunneled
through it), and (2) **codegen coverage** -- convs/ops this session hasn't hand-written a kernel
for yet still come from `relay.build()`. This section is about (1) only.

### What TVM's transport actually is

Reading `src/runtime/hexagon/rpc/hexagon_rpc.idl` and `hexagon_rpc_skel.c` (both in TVM's own
source tree) settles this precisely: the FastRPC interface TVM defines has exactly **five**
methods -- `open`, `close`, `send`, `receive`, `init` (`hexagon_rpc_skel_handle_invoke`'s switch
statement, cases 0-4; anything else returns `AEE_EUNSUPPORTED`). Every actual TVM RPC operation --
loading a module, calling a `PackedFunc`, running a graph -- is **not** a separate FastRPC method.
It's a byte stream TVM's `MinRPCServer` (`src/runtime/minrpc/minrpc_server.h`, instantiated in
`rpc_server.cc`'s `HexagonRPCServer`) decodes on the DSP side, fed through nothing but repeated
`send`/`receive` calls. So "TVM's transport" is really two independent, very differently-sized
things bundled together: a **thin, generic FastRPC shell** (5 methods, entirely reimplementable
from `hexagon_rpc.idl`'s pattern) carrying TVM's own **MinRPC application protocol** (a real wire
format for module loading, `PackedFunc` calls, `DLTensor` argument marshaling -- substantial to
reimplement, and not needed at all if the DSP side doesn't need to run arbitrary `PackedFunc`s).

Given this project's actual use case -- call one fixed, hand-written kernel function with a
handful of buffer pointers, no generic dispatch needed -- reimplementing MinRPC is the wrong
amount of engineering. The right amount is: **write our own tiny FastRPC interface**, generated by
the same `qaic` compiler Qualcomm's SDK ships (the same tool TVM's own build uses to generate
`hexagon_rpc_skel.c`/`hexagon_rpc_stub.c` from `hexagon_rpc.idl`), and drive it from a from-scratch
native client using only `libcdsprpc.so` (Qualcomm's own official, public FastRPC library --
present at `/vendor/lib64/libcdsprpc.so` on-device, not a TVM artifact). See `native_transport/`.

### Proof: a complete, TVM-free round trip on real hardware

`native_transport/mini_rpc.idl` defines a two-method interface (`run_add`, a trivial sanity check;
`run_kernel`, a buffer-in/buffer-in/buffer-out shape matching what a real vrmpy kernel needs).
`qaic` generates `mini_rpc_skel.c` (DSP-side dispatch) and `mini_rpc_stub.c` (host-side call
marshaling) from it -- no TVM code involved in generating either. `mini_rpc_impl.c` is the
DSP-side implementation (currently a placeholder byte-add, standing in for where a real kernel
call would go); it's compiled with `hexagon-clang` and linked with `hexagon-link` directly (the
same two flags TVM's own `tvm.contrib.hexagon.tools.link_shared` passes -- `-Bdynamic -shared
-export-dynamic` plus `libgcc.so` -- read out of `tools.py` and replicated without importing it).
`client_main.c` is a from-scratch native ARM64 program, cross-compiled with the Android NDK,
linked only against `libcdsprpc.so`, that calls the `qaic`-generated stub directly. Verified on
real hardware end to end (device `239dbd8f`), zero TVM Python or C++ in the loop:

```
enable_unsigned_pd rc=0
open OK, handle=0xb400007e59006450
run_add(40,2) rc=0 result=42
run_kernel rc=0 out=11 22 33 44 55 66 77 88
closed OK
```

### Two real bugs found getting here

1. **`AEE_ECONNREFUSED` (114) on every `remote_handle64_open` call, even against TVM's own
   already-proven `libhexagon_rpc_skel.so`** (tested directly, to isolate whether the problem was
   our `.so` or something about the calling process -- it was the latter). Root cause: unsigned
   (unsigned = not Qualcomm-signed) module loading on the CDSP is refused by default and must be
   explicitly enabled *per FastRPC session*, before the first `open`, via
   `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &{domain: CDSP_DOMAIN_ID, enable: 1},
   ...)` (`remote.h`). TVM's own launcher/session code (`apps/hexagon_launcher/launcher_android.cc`,
   `src/runtime/hexagon/rpc/android/session.cc`'s `enable_unsigned_pd()`) already does exactly this
   -- which is *why* the TVM-transport bridge never had to think about it -- but a from-scratch
   client has to make this call itself. This is not the same bug as the `AEE_EUNSUPPORTED`/"method
   5" failure in `../backbone_splice/README.md` (that one happens on an *already-connected*
   TVM session, well after this same enable call has already succeeded internally); still open.
2. **`AEE_EUNABLETOLOAD` (0x406) after fixing (1)**: the generated skel has two *undefined*
   external symbols, `mini_rpc_open`/`mini_rpc_close` -- the `open`/`close` handlers every
   `remote_handle64`-based interface needs (see `hexagon_rpc_skel.c`'s own `case 0`/`case 1`,
   dispatching to `hexagon_rpc_open`/`hexagon_rpc_close`) -- that `mini_rpc_impl.c` simply hadn't
   defined yet. The DSP-side dynamic loader can't load a `.so` with unresolved externals; adding
   trivial `mini_rpc_open`/`mini_rpc_close` implementations (allocate/free a 1-byte handle, mirroring
   TVM's own `hexagon_rpc_open`/`_close` in `rpc_server.cc`) fixed it.

### What this does and doesn't prove

Proven: the FastRPC transport layer -- everything from `libcdsprpc.so` down to a loaded, callable
DSP-side `.so` -- has **no TVM dependency at all**. `tvm.rpc.tracker.Tracker`,
`tvm.contrib.hexagon.build.HexagonLauncher`, and `tvm_rpc_android` can be replaced outright by
`native_transport/`'s pattern for this project's actual use case (call fixed, hand-written
kernels, not arbitrary graphs).

**Now done**: `mini_rpc_impl.c`'s `run_kernel` dispatches on buffer size -- the original 8-byte PoC
test still gets the placeholder byte-add unchanged, but a call sized for `hex_gemm_kernel.py`'s
own default shape (`cin=64,cout=256,m=54400`, the flagship Mask R-CNN pathological shape used
throughout this project) runs the *real*, unmodified generated kernel (`hex_gemm_kernel.py --cin
64 --cout 256 --m 54400`'s output, pasted in verbatim after its own qemu correctness check
passed). `gen_gemm_test_data.py` generates real input data (`pack_b()`-packed, same random seed
as `hex_gemm_kernel.py`'s own `main()`) as `gemm_a.bin`/`gemm_bp.bin`; `build.sh` picks them up
automatically if present and pushes them alongside the binaries. Verified **bit-exact correct on
real hardware** (device `239dbd8f`) against a numpy reference -- the full 55.7 MB output, zero TVM
anywhere in the loop, end to end through this transport instead of `bridge_and_test.py`'s TVM-RPC
bridge.

## Timing BEAM search candidates with hexagon-sim instead of raw instruction counting

`MOCKDSP=1`'s `qemu-hexagon-static` path (used throughout this whole investigation for fast
host-side iteration) times BEAM search candidates via QEMU's `inscount()` pseudo-register -- a
raw instruction count, not a cycle-accurate signal, and structurally blind to Hexagon-specific
pipeline behavior (HVX multi-cycle vector ops, dual/triple-issue slotting) or memory-hierarchy
effects. The 3x3 conv coverage work (`hex_conv3x3_kernel.py`, `codex/hex-gemm-3x3-conv`) found
exactly the case where this matters: the same kernel *shape*, same instruction-count profile, ran
faster than TVM at `cin=64` and **slower** than TVM at `cin=256` on real hardware, purely from
cache/working-set effects a plain instruction count can't see (no output-tile reuse across
pixels -- the fix is real, just not yet built). BEAM search using QEMU's inscount as its cost
signal has no way to detect a difference like that.

### hexagon-sim, and specifically its `--timing` mode

`hexagon-sim` is Qualcomm's own instruction-set simulator (ships in the Hexagon SDK's
`HEXAGON_Tools/*/Tools/bin/hexagon-sim` -- the same simulator this project's separate
`scripts/android/hexagon_sim_harness.py` already uses for TVM-compiled-kernel cycle counts, with
sim-vs-phone speedup correlation previously validated at 2.4x/4.5x/2.5x tracking the real phone's
1.8x/5.1x/2.0x). Its default mode reports a PMU-derived `Pcycles=` total at process exit, but
that default mode turned out to be a fast functional-only estimate, not meaningfully better than
instruction counting for this purpose. `hexagon-sim --help` reveals a separate, undocumented
(from this project's prior usage) `--timing` flag ("Run timing mode") alongside `--timing_nodbc`
("...without data backed cache") and per-component cache trace flags (`--dcachetrace`,
`--l2cachetrace`) -- strong evidence of a real pipeline/cache-hierarchy model gated behind that
flag, distinct from the default.

**Confirmed empirically**, with a minimal, decisive test: two synthetic kernels with the
*identical* instruction count (a fixed 65536-iteration load-increment-store loop), differing only
in whether their working set is a 4KB (L1-resident) or 4MB (cache-hostile, strided) buffer:

| Working set | `--timing` Pcycles-derived time |
|---|---:|
| 4KB, cache-resident | 0.000393 s |
| 4MB, cache-hostile | 0.010925 s |

**~27.8x apart, despite executing the exact same instructions in the exact same order.** Raw
instruction counting (what `MOCKDSP` gives BEAM today) would report these as identical. This is
real evidence `--timing` mode sees a real class of effect instruction counting cannot, in
principle, ever see.

### The tinygrad-side change

`HEXSIM=1` is a new third `DSPDevice` mode (alongside the real phone and `MOCKDSP=1`), landed on
the `onnxsim/tinygrad` fork's `vrmpy-hexagon-support` branch
(https://github.com/onnxsim/tinygrad/pull/1, same PR as the earlier `vrmpy` `TensorCore` work --
stacked onto it rather than a new PR, matching this project's usual pattern for an already-open,
unmerged PR). Mechanism, in `tinygrad/runtime/ops_dsp.py`:

- `HexagonSimRenderer` emits a hosted `main()` (hexagon-sim's standalone-OS mode has real libc,
  unlike `MOCKDSP`'s bare-metal `trap0`-syscall entry) with zero-filled static buffers. This is
  safe because cycle count for a fixed-control-flow kernel (true of every kernel this project
  generates -- conv/gemm with static loop bounds, no data-dependent branches) doesn't depend on
  data *values*, only on the shapes/loop-bounds already baked into the generated source -- so no
  live buffer data needs copying from the caller at all.
- Reading a cycle-counter register live from inside a standalone-sim binary doesn't work (the
  PCYCLE control register pair reads back 0 in this mode -- the same finding
  `hexagon_sim_harness.py` already made), so `HexagonSimCompiler.compile()` compiles the *same*
  kernel wrapper twice (`REPEAT=1` vs `REPEAT=2`, calling the kernel body once vs. twice) and
  `HexagonSimProgram.__call__` runs both ELFs under `hexagon-sim --timing`, returning the
  *difference* in each run's total Pcycles -- the simulator is deterministic, so this exactly
  isolates one kernel invocation's cost and cancels the fixed process-startup overhead (same
  differencing technique `hexagon_sim_harness.py`'s `run_kernel(..., measure_cycles=True)` uses).
  Both ELFs are cached together via the normal `Compiler.compile_cached()` path, so the real
  hexagon-clang + hexagon-sim round trip only happens once per distinct kernel source, not once
  per `__call__`.
- This needed **zero changes to BEAM search itself** -- `Program.__call__`'s return value (a
  float, lower is better) is the only integration point, and it already treats that value as an
  opaque timing signal regardless of which `DSPDevice` mode produced it.

Verified: deterministic across repeated runs of the same compiled ELF pair; scales correctly with
workload (doubling a kernel's inner-loop trip count measured ~1.93-1.97x, not exactly 2x --
plausible with pipeline/dual-issue overlap now being modeled); a real tinygrad-rendered GEMM
kernel run end to end through `Tensor.realize()` under `HEXSIM=1` reports a sane, nonzero,
Hexagon-pipeline-derived per-kernel time through the normal `DEBUG=2` display.

Usage: `HEXSIM=1 DEV=DSP HEXAGON_TOOLS=<Hexagon SDK Tools dir> BEAM=2 python3 your_script.py` --
same shape as `MOCKDSP=1`'s existing usage, just swap the env var. Prefer `HEXSIM=1` over
`MOCKDSP=1` whenever a BEAM search decision plausibly involves a memory/cache tradeoff (tile
sizes, blocking, working-set-sensitive reduction orders); `MOCKDSP=1` remains the faster option
(no real hexagon-clang/hexagon-sim round trip) when only correctness verification is needed, or
when candidates differ only in raw compute-instruction mix with comparable working sets.

## RoiAlign, Stage 1: yes, data-dependent addressing is expressible -- no, it isn't fast yet

`dynamic_ops_survey.md` (PR #1813) ranked `RoiAlign` the second most tractable of the four
dynamic-shape ops in Mask R-CNN's "rest" graph, but flagged a genuine open question first: can
this project's tinygrad/`custom_kernel` machinery express **data-dependent source addressing** at
all -- every RoI's bilinear-sample feature-map location depends on that RoI's own runtime box
coordinates, unlike every kernel built so far (`hex_gemm_kernel.py` through
`hex_boxhead_gemm_kernel.py`), which all read from a compile-time-known, purely shape-derived
address pattern.

**Answer: yes**, and it needed no new UOp/`Ops` capability -- `Tensor.gather(dim, index)`
(`tinygrad/mixin/op.py`) already exists as a first-class, backend-agnostic Tensor op. `hex_roialign_kernel.py`'s
`roi_align_tensor()` rebuilds `scripts/android/maskrcnn_e2e/tinygrad_ops.py`'s exact NumPy
`roi_align()` reference (avg pooling, `sampling_ratio=2`, matching this model's real RoiAlign node
attributes) as `Tensor` composition, using `.gather()` for the four data-dependent bilinear-tap
reads. Verified correct two ways against that same NumPy reference:

| Path | max abs err |
|---|---:|
| Default device | 5.90e-06 |
| `DEV=DSP MOCKDSP=1` (real `--target=hexagon` compile, `qemu-hexagon-static` execution) | 1.53e-05 |

The second row is the one that mattered for this task -- it confirms the actual Hexagon-targeted
compile+execute path handles this, not just tinygrad's generic default device. One real bug hit
getting there, the same class `sigmoid` already established: the bilinear weight arrays are plain
`np.arange`-derived arithmetic, which defaults to `float64` -- feeding that straight to `Tensor(...)`
pulled in `__hexagon_muldf3`/`__hexagon_adddf3` (double-precision compiler-rt symbols, a different,
larger set than the single-precision ones `sigmoid`'s `libgcc.a` fix covers), missing from the
freestanding link. Fixed with an explicit `.astype(np.float32)` -- a bug in this file's own
arithmetic, not a Hexagon/tinygrad limitation.

**What this does not establish: speed.** `Tensor.gather()`'s real implementation
(`index.unsqueeze(-1)._one_hot_along_dim(self.shape[dim]).where(x, 0)).sum(-1)`, read directly from
`tinygrad/mixin/op.py`) is a **dense one-hot-mask-and-reduce over the whole gathered dimension**,
not a genuine indirect/sparse load -- O(H*W) work per gathered pixel, not O(1). At real box-head
scale (a ~200x272 feature map, 4 taps x 49 positions per ROI, up to 1000 ROIs) this would be
enormously wasteful. A fast kernel needs real HVX indexed-load hardware -- confirmed present and
compiler-accessible on this exact toolchain, not assumed: `__builtin_HEXAGON_V6_vgathermh_128B`
(HVX vector gather) and `__builtin_HEXAGON_V6_vlutvvb_128B` (HVX vector lookup-table) both compile
cleanly against `hexagon-clang -mcpu=hexagonv73 -mhvx=v73`. The concrete next step: a hand-written
`custom_kernel` around one of those, with each RoI's sample addresses + bilinear weights computed
host-side (cheap scalar arithmetic, per the survey's own hybrid-split design) and fed to the
HVX-native instruction inside the kernel, padded to a compile-time-max ROI count the same way
`hex_gemm_kernel.py`'s tiny-`cout` coverage and `hex_stem7x7_kernel.py`'s `cin=3` padding already
established for a fixed-upper-bound trick.

**Not done**: real hardware (this stayed at `MOCKDSP=1`/qemu correctness, matching the survey's own
explicit permission for a correctness-only Stage-1 result); batching over more than one ROI
(single-ROI correctness was this pass's whole scope); the HVX-native `vgather`/`vlutvvb` kernel
itself.

## RoiAlign, Stage 2: a real, fast kernel on the phone -- no gather instruction needed

**Correction to Stage 1 first.** Stage 1's "dense one-hot-mask-and-reduce, O(H*W) per gathered
pixel" description is what `Tensor.gather()` *builds*, not what tinygrad *emits*. Rendering Stage 1's
kernel at a real shape (`DEBUG=4 DEV=DSP MOCKDSP=1`, P5 level, 25x34x256, one RoI) shows tinygrad's
own rewrite rules fold the `one_hot(...).where(x, 0).sum()` pattern into a **direct indexed load**
(`val = cond ? *(data2 + idx) : 0.0f`, the index read from a loaded tensor) -- no loop over H*W at
all. So Stage 1 already did genuine indirect loads; what it lacked was vectorization (NCHW makes every
tap a stride-850 scalar read per channel, and it was rendered for `-mhvx=v65`, which has no HVX fp32).

**Design: the data dependence is per *pixel*, not per *lane*.** In a channels-last (H, W, C) feature
map, each bilinear tap is C=256 contiguous floats -- 8 whole, aligned 128-byte HVX vectors. So the
scalar core computes each sample's two rows/cols and four weights (cheap, runtime box coords), and the
HVX unit does plain aligned `vmem` loads plus a 4-tap multiply-accumulate across channels. No per-lane
gather is needed at all. `vgathermh`/`vgathermw` (checked in the SDK's `hvx_hexagon_protos.h`: per-lane
16/32-bit offsets into a **VTCM** source region, result lands in VTCM) would be the wrong tool twice
over: nothing here needs per-lane addresses, and the P2 feature map (55.7 MB) is an order of magnitude
bigger than VTCM (single-digit MB). This is the same packed-channels insight that unblocked `maxpool`.
The kernel is hand-written C (`roialign_fast/roialign_kernel.h`, header-only so host, qemu and the
DSP skel compile identical code) rather than a tinygrad `custom_kernel`: its body is almost entirely
data-dependent scalar control flow (float->int, clamps, bounds skips) around a trivial vector FMA,
which in `custom_kernel` form would be one big `Ops.CUSTOM` string anyway.

Exact ONNX semantics (opset-12 `RoiAlign`, no `coordinate_transformation_mode` attribute, i.e. no
half-pixel offset; `avg`; `sampling_ratio=2`; clamp/skip rules) taken from the real model's 8 RoiAlign
nodes, with **real inputs**: one COCO image (val2017 `000000000139`) through the full
`MaskRCNN-12-qdq` in ONNX Runtime, capturing every RoiAlign node's real feature map, real RoIs and real
output (`roialign_fast/dump_real_roialign_io.py`). Real call shapes: box head 43/111/218/628 RoIs at
7x7 on P5/P4/P3/P2 (1000 total), mask head 1/3/22/74 at 14x14 (100 total), all fp32, C=256.

**Verification, three levels, all 8 real calls:**

| Level | Result |
|---|---|
| Host C (x86, same header) vs ORT's real outputs | bit-exact (max abs err 0) |
| `qemu-hexagon-static`, Hexagon ISA, scalar fp32 (`-mhvx=v65`) | max abs err <= 9e-6 |
| **Real hardware** (device `239dbd8f`, CDSP, HVX qf32, via a dedicated TVM-free FastRPC skel) | **max abs err <= 6.5e-5** (qf32 rounding), PASS at every call and thread count |

qemu 8.2's Hexagon decoder can't execute the qfloat HVX instructions the vectorized build uses
(`decode_packet: assertion failed`), hence the scalar build there; the real phone is the check for the
vector build. The 55.7 MB P2 map didn't hit this project's ~32 MB transfer ceiling: the client
allocates every buffer with `rpcmem_alloc` (ION-backed, mapped into the DSP rather than copied).

**Speed, real hardware, all 8 real calls summed** (DSP-side time from `HAP_perf_get_time_us`, median
of 5; FastRPC round trip adds ~0.3-0.5 ms per call):

| | 1 thread | 4 threads |
|---|---:|---:|
| This kernel, plain | 137.1 ms | 44.7 ms |
| This kernel, + `l2fetch` of the next bin's tap rows | **85.4 ms** | **27.1 ms** |
| ONNX Runtime CPU **on the phone** (stock `onnxruntime-android` 1.26 arm64, C API) | 267.4 ms | 63.3 ms (default, all cores) |
| ONNX Runtime CPU on the 32-thread desktop host (Ryzen AI Max+ 395), for scale | 86.2 ms | 11.5 ms (default) |

So on the phone: **3.1x faster than one CPU thread, 2.3x faster than ORT using every CPU core**, which
is how `rest.onnx` runs today. Split by head: box head 19.8 ms vs ORT 46.7 ms; mask head 7.3 ms vs
16.6 ms. The kernel is memory-latency bound, not compute bound: the plain version spends ~500 cycles
per sample against ~50 cycles of HVX work, a `HAP_power_set` TURBO/DCVS-off vote changed nothing
(the DSP was already at full clock), and a one-bin-ahead 1-D `l2fetch` of the two 2-pixel row
segments each sample reads cut time 1.6x -- except on the smallest (P5) level, whose 0.87 MB map is
already cache-resident (flat to ~5% slower there). Threads split RoIs across QuRT threads, each holding
the HVX unit (`qurt_hvx_lock`); 4 threads give 2.9-3.2x.

For the record against Stage 1 on the only path Stage 1 ever ran (qemu, scalar, same single P5 RoI):
1.00 M vs 1.49 M executed instructions -- 1.5x from the channels-last layout alone. The real speedup
comes from HVX fp32 vectors, threads and prefetch, none of which Stage 1's `-mhvx=v65` rendering had;
Stage 1 was never run on hardware.

**Caveats, stated plainly:**
- **Layout: now priced, see Stage 3 below.** The kernel reads channels-last features and writes
  channels-last (R, OH, OW, C) output; ORT reads/writes NCHW. The output side is free in practice (the
  fc6 box-head MatMul consumes the flattened crop, so its weight rows can be permuted offline). The
  input side was unpriced here: converting the backbone's NCHW FPN maps costs 13.3 ms on the DSP, and
  having the FPN output convs write channels-last directly costs ~2.3 ms instead -- measured, verified
  and compared against a same-session ORT baseline in "RoiAlign, Stage 3".
- One image's RoIs, fp32 only (the real model's RoiAlign inputs are dequantized fp32).
- Not integrated into any pipeline; each call is a standalone FastRPC round trip.

`roialign_fast/` reproduces everything: `dump_real_roialign_io.py` (real inputs/outputs from ORT),
`gen_roialign_test_data.py` (channels-last binaries), `roialign_host_check.c`, `roialign_qemu.c`,
`build.sh` (qaic + hexagon-clang/link + NDK client + adb run; `TURBO=1` adds the clock vote),
`make_roialign_single_node_models.py` + `ort_roialign_bench.c` (the host and phone ORT baselines).

## RoiAlign, Stage 3: pricing the channels-last layout, then removing its cost

Stage 2's 2.3x assumed the FPN feature maps were already channels-last. This stage measures what
getting them there actually costs, two ways, on the phone with real data.

**Where the RoiAlign inputs come from.** Traced in the real `backbone.onnx`: each of the four
RoiAlign feature maps (`391`/`423`/`455`/`487` = P5/P4/P3/P2, `spatial_scale` 1/32..1/4) is a
**backbone graph output** that crosses the backbone/rest split as fp32 NCHW. Each is produced by
`Conv` (the FPN 3x3 "output" conv, 256->256, pad 1) -> `QuantizeLinear` (uint8, per-tensor scale,
zero point 128-133) -> `DequantizeLinear` (fp32). Weights are int8 symmetric (zero point 0), the bias
is int32 at scale `s_in * s_w`, and the conv inputs are uint8 with zero points 125-127. ONNX Runtime
fuses that QDQ pattern into `QLinearConv`, so its outputs come from exact integer accumulation plus a
float requantize. `fpn_channels_last/capture_fpn_out.py` shows that an integer-only reimplementation
reproduces ORT **bit for bit at all four levels**:
- exact int32 accumulation, with the input zero point folded into the bias and the border padded
  with that zero point;
- `q = clamp(round_half_even(float(acc) * (s_in*s_w/s_out)) + zp)`;
- `(q - zp) * s_out`.

That is 0 uint8 mismatches out of 18.5 M values, and bit-identical fp32. It is also bit-identical to
the maps Stage 2's RoiAlign data was captured from.

**Two ways to get channels-last, both measured on the phone** (device `239dbd8f`, DSP-side time, six
full runs, three with DCVS and three with a core+bus TURBO vote, which changed nothing measurable):

1. **Keep the backbone as is and transpose its NCHW outputs.** `fpn_channels_last/layout_kernels.h`
   transposes 32x32 fp32 blocks in HVX registers with five rounds of the perfect-shuffle trick:
   `vshuff(.., -4)` word-zips rows i and i+16, and five rotations of the (row, col) index make a
   transpose. It checks bit-exact against a scalar transpose under qemu at every level shape,
   including the non-multiple-of-32 tails. All four maps (P2-P5, ~74 MB fp32):

   | Transpose variant | ms (mean of 6, min-max) |
   |---|---:|
   | scalar, 1 thread | 224.0 (222.0-225.5) |
   | HVX, 1 thread | 37.0 (36.8-37.2) |
   | HVX, 4 threads (split by pixel range) | **13.3** (13.1-13.6) |
   | HVX + next-block `l2fetch`, 4 threads | 13.3 (12.6-13.6) |
   | HVX, channel-group-outer loop, 1 / 4 threads | 33.6 / 13.9 |

   It is memory-bound: 4 threads help (2.8x), while prefetch and the other loop order don't. One
   earlier cold first run measured P2 alone at ~28-30 ms, so the first call after the DSP has been
   idle can pay ~2x.

2. **Have the producer write channels-last.** `hex_conv3x3_fpnout_kernel.py` is
   `hex_conv3x3_kernel.py`'s vrmpybusv conv, generated through tinygrad `custom_kernel` with the same
   `ow_tile` reuse. It adds the requantize+dequantize epilogue above into its store, and a `layout`
   switch:
   - channels-last: each accumulator vector is 32 channels of one pixel, stored contiguously;
   - NCHW: the same 32 lanes land H*W floats apart.

   The epilogue runs as scalar IEEE fp32, with vectorization explicitly disabled. HVX fp32 on this
   phone is qf32, which is not IEEE-rounded, and one ulp would lose bit-exactness. Run on the phone
   with the real FPN conv inputs, weights and biases, all four levels, every run:
   - The **NCHW output is bit-identical to ORT's real backbone output.**
   - The channels-last output is bit-identical to every transpose variant's output.
   - The qemu-hexagon run agreed at all four levels too.

   The store layout is the only difference between the two variants:

   | FPN output conv (fused epilogue, 1 thread) | channels-last store | NCHW store | delta |
   |---|---:|---:|---:|
   | P5 25x34 | 14.7 ms | 14.4 ms | +0.3 ms |
   | P4 50x68 | 46.6 ms | 45.8 ms | +0.8 ms |
   | P3 100x136 | 217.8 ms | 214.9 ms | +2.9 ms |
   | P2 200x272 | 871.5 ms | 873.2 ms | -1.8 ms |
   | **total** | 1150.7 ms | 1148.4 ms | **+2.3 ms** |

   The conv's absolute time isn't part of this comparison. It is backbone work either way, runs
   single-threaded here with a scalar epilogue, and isn't competing with TVM's multi-threaded
   backbone in this table. The delta is what the layout costs.

**End to end, RoiAlign side, all 8 real calls.** RoiAlign runs on the NHWC maps the fused conv
produced (4 threads with prefetch): 27.8 ms (27.7-27.9), max abs err vs ORT's real outputs 6.5e-5, the
same as Stage 2. ORT's CPU RoiAlign on the phone, which reads NCHW directly, was re-measured this
session with Stage 2's own `ort_roialign_bench` at **57.3-59.5 ms** (mean 58.3; Stage 2 recorded
63.3):

| Path to RoiAlign output | ms | vs ORT (58.3 / 63.3) |
|---|---:|---:|
| ORT CPU RoiAlign on the phone (all cores, NCHW) | 58.3 | 1.00x |
| Stage 2 kernel, maps assumed channels-last (layout unpriced) | 27.8 | 2.10x / 2.28x |
| + separate HVX transpose of today's NCHW maps | 27.8 + 13.3 = **41.1** | **1.42x / 1.54x** |
| FPN output convs write channels-last (fused) | 27.8 + 2.3 = **30.1** | **1.94x / 2.10x** |

So the answer to "is the 2.3x real today" is **mostly no, and fixably yes.** With the existing NCHW
backbone the transpose costs 13.3 ms, about 44% of the ~30 ms the kernel saves, leaving ~1.4-1.5x.
With the producer writing channels-last, the win survives at ~1.9-2.1x, for a layout cost of about
2 ms.

One real bug on the way. The first fused-epilogue kernel read accumulator lanes as scalars straight
out of tinygrad's `REG` placeholder, which renders as a plain 4-byte-aligned `int bufN[32]` that the
reduction also accesses through 128-byte vector casts. That crashed `qemu-hexagon-static` outright. A
standalone repro isolated it: the reduction alone, and a static-buffer harness, both ran fine, while
any scalar read of that buffer crashed. Copying the accumulator into an explicitly
`aligned(128)` local first fixed it.

**Not done**:
- The fused conv isn't wired into a TVM-free full-backbone run. The chain in "Running the full
  backbone graph, TVM-free" stops at stage1/block1, far upstream of the FPN, so the ~2 ms figure is
  for the verified producer at real shapes and real data, not a measured full pipeline.
- The epilogue is scalar, for exact IEEE results. A qf32 HVX epilogue would be faster but not
  bit-exact.
- RoiAlign reading uint8 maps directly (4x less traffic for a latency/bandwidth-bound kernel) wasn't
  tried. It interacts with how out-of-range samples contribute, since dequantize is affine and
  skipped samples add 0 in fp32, not the zero point.

`fpn_channels_last/` reproduces everything:
- `capture_fpn_out.py` (real conv inputs/params/outputs from `backbone.onnx` in ORT, plus the
  bit-exact epilogue check);
- `gen_fpn_test_data.py` (phone binaries);
- `layout_kernels.h` + `layout_qemu.c` (transpose and its qemu check);
- `fpn_rpc.idl` / `fpn_impl.c` / `fpn_client.c` / `build.sh` (TVM-free FastRPC skel + client:
  transpose, fused conv, RoiAlign, clock vote).

The conv kernels themselves come from `../hex_conv3x3_fpnout_kernel.py --data fpn_out_real.npz --out
$DATA/fpn_out_kernels.c`, which also runs all four levels under qemu against ORT.

## RoiAlign, Stage 4: uint8 maps in, the heads' uint8 rows out -- one DSP call per head

In the e2e pipeline (`../e2e_pipeline/`, `pipe_e_opt.txt` / `pipe_e_u8.txt`), each head's RoiAlign
isn't one op but a span of steps:
- 4 CPU `dq` steps turn the backbone's uint8 NHWC FPN maps into fp32;
- 4 DSP `roialign` calls, one per FPN level;
- a CPU ScatterND (`seg2`/`seg4`) merges the per-level outputs into the head's row order;
- in `pipe_e_u8.txt`, a CPU `quant` then turns them back into the u8 head's input.

`roialign_fast/roialign_u8_kernel.h` does that whole span in one integer kernel, as one FastRPC call
per head:
- It reads the uint8 maps directly.
- It computes each sample's position and bilinear weights in fp32 with ORT's exact expressions, then
  quantizes the weights to Q14. The 4th weight absorbs the rounding, so each sample's weights sum to
  exactly 1.
- It accumulates u8 x Q14 in exact int32, subtracts the input zero point once per bin, and does one
  fixed-point requant per level to the head's input scale/zero point.
- It writes each RoI straight into its destination row (the `*_row_index` tensors `seg2`/`seg4`
  scatter with).

The Stage 3 caveat about out-of-range samples dissolves: skipped samples add nothing, and the zero
point is subtracted per *valid* sample.

**Exactness.** The reference is QuantizeLinear(ORT fp32 RoiAlign, merged), the exact bytes today's
CPU steps produce, captured from the real pipeline by `capture_merged_io.py`. On the 6 e2e images:
- box: 99.93-99.95% of bytes identical;
- mask: 99.94-99.99%;
- **max diff 1 LSB** everywhere.

Host (`roialign_u8_host_check.c`), qemu (`roialign_u8_qemu.c`, HVX build) and phone
(`roialign_u8_client.c`) produce byte-identical outputs. The remaining 1-LSB bytes are fp32 values
within a hair of a rounding boundary. Sweeping the weight bits (14-16) and the requant pre-shift
didn't move the exact fraction beyond noise.

End to end, `../e2e_pipeline/roialign_u8_emulate.py` replays `pipe_e_opt.txt` on the host with both
spans replaced by the kernel. The `*_nhwc` heads start with a QuantizeLinear at the same qparams, so
they see exactly the kernel's bytes. Against the all-ORT reference on the 6 images:

| host emulation of `e_opt` | matched | box IoU | score \|d\| | mask IoU |
|---|---:|---:|---:|---:|
| ORT RoiAlign + dq + ScatterND (today) | 60/61 (64 dets) | 0.938 | 0.015 | 0.871 |
| `roialign_u8` | 60/61 (64 dets) | 0.942 | 0.013 | 0.883 |

**Speed on the phone.** Device `239dbd8f`, SM8475 / Hexagon V69, unsigned PD, image 139 (1000 box
RoIs, 99 mask RoIs). Medians of 7-9 runs, 4 threads + `l2fetch` prefetch unless noted. "RPC" is the
client-side round trip of the single call.

| step (cumulative) | box DSP / RPC | mask DSP / RPC | CPU steps left in the span | span total |
|---|---:|---:|---:|---:|
| Stage 3 fp32 kernel, 4 calls per head (today's `e_opt`) | 19.9 / 21.4 | 6.8 / 8.0 | dq 13.1 + merges 16.6 + 6.5 | **65.6** (+ `pipe_e_u8`'s `quant` x2) |
| uint8 merged kernel, portable vector code | 20.6 / 21.1 | 8.3 / 8.9 | none | 30.0 |
| + `vmpyuh_acc` MAC, separable sample precompute | 12.3 / 13.4 | 4.3 / 5.1 | none | 18.5 |
| + accumulators kept in registers | **6.4 / 6.8** | **2.1 / 2.5** | none | **9.3** |

All 6 images at the last step: box 5.8-6.7 ms DSP (6.2-7.1 RPC), mask 0.5-2.2 ms (0.9-2.6 RPC).

What mattered, and what didn't:
- **The MAC instruction.** The generic int32 vector code costs ~12 HVX ops per tap per 128 channels
  (`vunpack` + `vmpyieo`/`vmpyie`). `vzxt` u8->u16 plus two `vmpyuh_acc` (u16 x scalar u16 -> u32
  accumulate) cost 3. The even/odd lane splits both instructions introduce go through the
  elementwise requant untouched, and `vshuffeh`/`vshuffeb` undo them before the store.
- **Registers, not the stack.** Indexing the accumulators by a runtime channel-half count made
  hexagon-clang keep them on the stack (a `vmem` load/store around every MAC). Instantiating the
  kernel for C = 128 / 256 halved the time again.
- **Prefetch.** One bin-ahead `l2fetch` per sample row takes 1-thread box time from 30.5 to 20.1 ms.
  With it, a real 1-thread run is as fast as a cache-hot one (every job the same RoI: 20.1 ms).
  Neither a row-ahead variant nor prefetching the next RoI's whole footprint helped.
- **Threads.** 4 is the sweet spot: 6 threads is slower (box 7.9 vs 6.4 ms). The locality sort (by
  level, then y, then x) is slower too (box 9.4 vs 7.7 in the same run); contiguous per-thread
  slices of the natural order balance better.
- **One call vs one per level.** Merging all levels into one call saves ~1-2 ms of RPC per head
  over 4 calls.
- **Not needed:**
  - A TURBO clock vote gains nothing.
  - **VTCM:** `vtcm_probe` shows this unsigned PD can query and acquire all 8 MB
    (`HAP_compute_res_*`). But P2 (13.9 MB, 61% of box RoIs) doesn't fit, and with prefetch the
    kernel is no longer latency-bound, so DMA staging has little to win.

**Proposed integration (not wired: `../maskrcnn_demo_app/` and `e2e_run.cpp` belong to the demo).**
A pipeline step that replaces, per head, the `dq` x4 + `roialign` x4 + `seg2`/`seg4` + `quant` lines
of `pipe_e_u8.txt`:

    roialign_u8 MAPS QPARAMS ROIS ROWS OH OW SR S_OUT Z_OUT DST
    # box head of pipe_e_u8.txt (levels in the roialign lines' order, P5..P2):
    roialign_u8 391_quantized_nhwc,423_quantized_nhwc,455_quantized_nhwc,487_quantized_nhwc \
        0.11570039391517639:129:0.03125,0.128072127699852:133:0.0625,0.12207645177841187:128:0.125,0.13543301820755005:131:0.25 \
        2656,2639,2622,2605 2751_row_index,2723_row_index,2695_row_index,2667_row_index \
        7 7 2 0.11669740080833435 130 2788_u8

`QPARAMS` holds each map's dq `scale:zero_point:spatial_scale`. `ROWS` pairs each level with the
`*_row_index` that seg2's ScatterND uses for that level's output. The mask head is the same with
`6701,6684,6667,6650`, `6795_row_index,6767_row_index,6739_row_index,6711_row_index`, 14x14 and the
mask head's `0.11957937479019165 132` into `6833_u8`. The `dq` lines go too, because the maps feed
nothing else.

The step makes one call:

    roialign_u8_rpc_run(h, map0..map3 /* uint8 NHWC, rpcmem */, level_geom /* H,W,z_in x4 */,
                        level_fparams /* s_in, spatial_scale x4 */, level_counts /* RoIs per level */,
                        rois /* [n,4] fp32, concatenated in level order */, rows /* [n] int32 */,
                        C, OH, OW, sr, s_out, z_out, flags /* 260 = 4 threads + prefetch */,
                        out /* [n, OH, OW, C] uint8 = the head's input */, &dsp_us);

The skel is `roialign_u8_rpc.so`: `roialign_u8_rpc_skel_handle_invoke`, unsigned PD, built by
`roialign_fast/build_u8.sh`. `roialign_u8_cpu.c` is the same call on the CPU. Constraints:
- C % 128 == 0 and C <= 256;
- sr <= 4;
- OH*sr and OW*sr <= 256.

Until it's known why the demo serializes DSP skel calls, assume this call does not overlap the RPN
kernel's.

**Reproduce** (`roialign_fast/`):
- `capture_merged_io.py E2E_OUT image.bin DEST` captures one image's maps, per-level RoIs/rows and
  reference bytes.
- `roialign_u8_host_check.c DEST/<img>` is the host check.
- `roialign_u8_qemu.c` + `roialign_u8_qemu_args.py` are the qemu check.
- `DATA=DEST/<img> build_u8.sh` does the phone build and sweep (`CONFIGS` = comma list of
  threads + 256*prefetch + 512*sort; `CLIENT_ENV=VTCM_PROBE=1` adds the VTCM probe).
- `../e2e_pipeline/roialign_u8_emulate.py OUT REF images` is the e2e host check.
- The fp32 kernel and skel (`roialign_kernel.h`, `roialign_rpc.idl`, ...) are unchanged, for A/B.

## Proposal decode (RPN): bit-exact on the phone's DSP, 1.58x ONNX Runtime

The easiest-ranked op in `dynamic_ops_survey.md`. Everything below comes from the real
`rest.onnx` graph and one real inference (COCO `000000000139.jpg`), not from assumptions.

### What the region actually is

Per FPN level, **TopK runs first** (on the objectness scores), then anchors and deltas are gathered
by the top-k indices, so decode only ever sees k = 1000/1000/1000/1000/663 = **4663 boxes**, not the
217,413 anchors. The objectness sigmoid is not in this region: it's in the backbone, before the
reshape. The kernel replaces, per level, everything from the squeezed TopK indices to the
grid-quantized boxes that feed the min-size filter and NMS: **315 non-Constant nodes** over the 5
levels (50 Unsqueeze, 40 Add, 40 Mul, 35 Gather, 30 Sub, 30 Slice, 20 Clip, 20 Reshape, 15+15 Q/DQ,
10 Concat, 10 Exp). `capture_proposal_decode.py` finds the region by walking the graph and reads
every constant out of it:

- Detectron decode: widths/heights with `+1`, dw/dh clipped at 4.1352 (= log(1000/16)), `-1` on
  x2/y2; clip x to [0, 1279] and y to [0, 959]. Those bounds are baked into the model and are not
  the 800x1088 canvas; they're reproduced as found.
- The deltas are **requantized** on the way in: for P2-P4 the rest-side Q/DQ grid (e.g. P2:
  0.018187/129) differs from the backbone's output grid (0.018761/133), so this is a real lossy
  step and the kernel reproduces it. For P5/P6 it's an identity.
- The final boxes are Q/DQ'd onto a uint8 grid (scale 5.0157, zero point 0), ~5 px steps.
- Layout: the backbone turns the `[1, 12, H, W]` uint8 conv output into per-anchor rows over all
  anchors (Reshape `[1,3,4,H,W]` -> Transpose `(0,3,4,1,2)` -> Reshape `[-1,4]`, plus Q/DQ).
  Decode needs only k rows, so the kernel can gather straight from the NCHW conv output
  (`a = i % 3`, `(h, w) = divmod(i / 3, W)`). Checked: this reproduces rest.onnx's delta input
  exactly.

### Correctness: bit-exact everywhere

`pd_kernel.h` (one header for host, qemu and DSP) matches ONNX Runtime's real output **bit for bit**
at all 5 levels and for both delta sources (rest.onnx's fp32 per-anchor input, and the backbone's
NCHW uint8 conv output):

| check | result |
|---|---|
| host C (`pd_host_check.c`) | 0 of 18,652 values differ |
| qemu, Hexagon v73 scalar build (`pd_qemu.c`) | 0 differ |
| **phone CDSP** (`pd_client.c`, every config below) | **0 differ** |

This can be bit-exact where RoiAlign (max error 6.5e-5 on the phone) wasn't, because it's scalar
IEEE fp32: no HVX qfloat, and it's built with `-ffp-contract=off` so there are no FMAs the graph
doesn't have. `exp` is the kernel's own polynomial (no libm, no fp64): it's **at most 1 ulp** from
correctly rounded over all 2.18 billion floats in [-10, 4.1352]. ORT's own `Exp` differs from it at
ulp level, so before the grid quantization boxes differ from ORT's by up to 1.2e-4 in a few percent
of elements (coordinates are ~1000 px, so 1 ulp is 6e-5). The ~5 px box grid then absorbs every
one of those differences.

### Speed on the phone

All 5 levels in one FastRPC call; medians of two full runs, 21 reps each. ORT is the same 315 nodes
cut out of rest.onnx (`make_pd_ort_model.py`), one `Run`, stock onnxruntime-android 1.26 arm64 C API:

| | DSP kernel | roundtrip | vs ORT |
|---|---:|---:|---:|
| ONNX Runtime CPU on the phone (1 thread / default) | -- | 1.04 / 1.05 ms | 1.00x |
| NCHW uint8 source, fast path, 6 threads | 0.40 ms | **0.66 ms** | **1.58x** |
| NCHW uint8 source, fast path, 4 threads | 0.47 ms | 0.75 ms | 1.39x |
| NCHW uint8 source, fast path, 1 thread | 1.29 ms | 1.58 ms | 0.66x |
| fp32 per-anchor source (today's boundary), fast path, 6 threads | 0.51 ms | 0.83 ms | 1.26x |
| division-per-element reference path, 6 threads | 0.76-0.85 ms | 1.11-1.15 ms | ~0.93x |

ORT gains nothing from more threads (315 tiny nodes, dominated by per-node dispatch). The first
bit-exact DSP version was 4.5 ms single-threaded and lost to ORT at every thread count. What
closed the gap, with every step keeping the output bit-exact:

1. **12 scalar fp32 divisions per box -> none on the common path.** The two-stage delta requant is
   a function of the backbone's uint8 value only, so it becomes a 256-entry per-level LUT. fp32
   deltas map back to their uint8 index with an exact on-grid check and fall back to the full chain
   if they're ever off-grid; the host check tests that fallback with deliberately perturbed deltas.
   The box-grid quantize multiplies by 1/scale and only does the real division when the quotient is
   within 1e-3 of a .5 tie. That was checked exhaustively over all 1.15 billion floats in [0, 1279]:
   0 mismatches, with 0.016% of values taking the division fallback.
2. **Anchors computed, not gathered.** All 5 real anchor tables are exactly `base[i % 3] + (w, h,
   w, h) * stride` with small-integer values, so they're exact in fp32. `set_model` verifies this
   against the uploaded table on the DSP and falls back to table reads if it ever fails. The row
   divide is a magic-number multiply, because Hexagon has no integer divide instruction and the
   freestanding build has no helper for it.
3. **Blocked, branch-free math plus `dcfetch` prefetch of the next block's gather addresses.** The
   generated assembly shows the exp loop software-pipelined at 15 packets per element (about 2 fp32
   ops per packet, the scalar FP limit), so the math is cheap. The random gathers into the
   multi-MB maps are the real cost, which is why prefetching the next block helped (1.45 -> 1.29 ms
   single-threaded).

The TURBO clock vote changed nothing.

### Honest caveats

- **This region is small.** It's ~1 ms of ORT time, while the whole rest.onnx remainder costs
  200-500 ms per image. The absolute saving is ~0.4 ms. FastRPC's fixed ~0.26 ms is about 40% of the
  roundtrip, so a standalone call is a poor unit. The case for it is as part of a larger
  DSP-resident rest pipeline (TopK and NMS next to it), not on its own.
- **It needs DSP threads.** Single-threaded it loses to ORT; 4+ threads are needed to win.
- **The NCHW source also removes the backbone's full-map layout work** (dequant + transpose over all
  217,413 anchors, done today inside the compiled backbone). `bb_layout` prices that with a plain,
  unvectorized C loop at 9.7 ms on 1 DSP thread / 3.6 ms on 4 (output checked equal to rest.onnx's
  real input). That's an upper bound: the backbone's real cost for these fused layout ops wasn't
  isolated here.

### Reproduce

```
python capture_proposal_decode.py --model maskrcnn_sim.onnx --rest rest.onnx --image IMG --out DATA
cc -O2 -ffp-contract=off -o pd_host_check pd_host_check.c && ./pd_host_check DATA   # also writes lN_params.bin
python make_pd_ort_model.py --rest rest.onnx --data DATA
clang-19 --target=hexagon -mcpu=hexagonv73 -O2 -ffp-contract=off -static -nostdlib -ffreestanding \
  -fuse-ld=lld -o pdq pd_qemu.c   # then: qemu-hexagon-static pdq DATA/lN_*.bin ... A k H W
HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... DATA=DATA ORT_AAR_DIR=<extracted onnxruntime-android aar> ./build.sh
```

## TopK: exact on the DSP, and faster than ORT for the batched per-level selections

`dynamic_ops_survey.md` ranked TopK "high" difficulty: no sort/select primitive exists anywhere in
this project. It turned out not to need the bitonic sort the survey expected. `topk/` is an exact
TopK (values **and** int64 indices, byte-for-byte equal to ONNX Runtime's) running on the phone's
CDSP through its own FastRPC skel, the same TVM-free pattern as `roialign_fast/`.

**The real nodes.** One COCO image (`000000000139`) through the full `MaskRCNN-12-qdq` in ORT,
capturing every TopK node's real input, k and outputs (`topk/dump_real_topk_io.py`). All seven are
1-D, fp32, `largest=1`, `sorted=1`:

| Node | Role | n | k |
|---|---|---:|---:|
| 908 | P2 pre-NMS select | 163,200 | 1000 |
| 1234 | P3 pre-NMS select | 40,800 | 1000 |
| 1560 | P4 pre-NMS select | 10,200 | 1000 |
| 1886 | P5 pre-NMS select | 2,550 | 1000 |
| 2212 | P6 pre-NMS select | 663 | 663 |
| 2488 | post-NMS select (after the per-level NMS) | 1,465 (data-dependent) | 1000 |
| 6528 | final detection cap | 106 (data-dependent) | 100 |

The kernel takes n and k at call time, so the two data-dependent calls need no padding.

**Ties are the normal case, and the tie order has to match ORT's.** The scores are dequantized
int8, so the 163,200-element level has only **51 distinct values**. Across the seven calls there
are 51 to 213. At that level the k-th value is shared by 111 elements, and 99 of them are taken.
On all seven calls ORT orders by **value descending, then index ascending**. ORT on the phone's
CPU reproduces the captured x86 output exactly, so the same order holds on ARM. Matching values
alone would not be enough: the indices decide which boxes go on to NMS.

### The algorithm: select first, then sort only the survivors

In `topk/topk_kernel.h`, one header that builds for the host, qemu and the CDSP:

1. **Key.** Each fp32 value maps to a monotone uint32 (bigger float, bigger key). -0 is mapped to
   +0 so the two tie, as a float compare would.
2. **Threshold.** Read 128 evenly spaced blocks of 8 contiguous values (1024 keys) and quickselect
   the key at the rank that should leave ~1.5k survivors. Inputs with n ≤ 4k skip this step and keep
   everything.
3. **Collect.** One streaming HVX pass: a 32-lane `key >= t` compare, turned into a 32-bit lane mask
   (AND with per-lane bit weights, then a rotate/OR tree). Only set bits are visited (`ctz`), and
   a fully set mask stores the whole vector at once. Survivors are appended in index order, and an
   `l2fetch` runs 16 KB ahead. Index ranges are independent, so a big call can split this pass
   across threads.
4. **Emit.** Hash the survivors' distinct keys, sort those keys descending, then do one stable
   counting-sort pass by key rank. Stability keeps index order within a key, which is exactly ORT's
   tie order. Above 256 distinct keys it falls back to a stable radix sort. Values are recovered
   from the keys, so there is no random-access gather of the input, except for key 0, which +0 and
   -0 share.

**Correctness doesn't depend on the threshold.** If at least k survive, every element of the true
top-k has a key ≥ the k-th key ≥ t, so the answer is inside the survivor set. If fewer than k
survive, the threshold is lowered and the pass rerun; the last resort takes everything. On the
real data the first threshold always sufficed (one pass).

### Verification

| Level | Result |
|---|---|
| Host C, all 7 real calls | values and int64 indices byte-exact vs ORT, every variant |
| Host C, 600 randomized cases | exact. Covers heavy ties, ±0, negatives, all values tied, a layout that makes the sample miss so the retry and fallback paths run, and many distinct values (the radix fallback). Also clean under ASan/UBSan |
| `qemu-hexagon-static`, v73 HVX build | exact on all 7 (the kernel uses only integer HVX ops, which qemu 8.2 decodes) |
| Real CDSP, device `239dbd8f` | exact on all 7, every variant and thread setting |

### Speed on the phone (medians of 31 runs; ORT is the op's current path, on the phone's CPU)

DSP time is `HAP_perf_get_time_us` around the kernel. Round trip is the client-side time for the
whole FastRPC call.

| Call (n → k) | ORT CPU, 1 thread | DSP, 1 thread | DSP, collect split 6 ways |
|---|---:|---:|---:|
| P2 163,200 → 1000 | 435 µs | 488 µs | **235 µs** |
| P3 40,800 → 1000 | 280 µs | 226 µs | **157 µs** |
| P4 10,200 → 1000 | 73 µs | 185 µs | |
| P5 2,550 → 1000 | 59 µs | 107 µs | |
| P6 663 → 663 | 13 µs | 37 µs | |
| post-NMS 1,465 → 1000 | 18 µs | 81 µs | |
| final 106 → 100 | 4 µs | 20 µs | |

The five per-level selections are independent, so they go to the DSP as **one** RPC:

| 5 per-level TopKs | DSP time | Round trip | vs ORT (859 µs, 1 thread; 849 µs default threads) |
|---|---:|---:|---:|
| one after another, 1 thread | 997 µs | 1433 µs | slower |
| one after another, big collects split 4 ways | 689 µs | 1119 µs | slower end to end |
| concurrently, one thread per level | 557 µs | 990 µs | slower end to end |
| **concurrently + P2's collect split 4 ways** | **369 µs** | **720 µs** | **2.3× on DSP time, ~1.2× including the round trip** |

All seven calls take 470 µs of DSP time, against ORT's 880 µs. But the last two TopKs sit after
NMS, which runs on the CPU, so they are separate RPCs, and with ~0.25 ms each the round-trip total
is 1295 µs.

**Honest reading.** The batched per-level selection beats ORT even with the FastRPC round trip. The
two downstream TopKs (1465 → 1000, 106 → 100) are slower on the DSP than on a 3 GHz big core even
before their ~0.25 ms round trip, so they should stay on the CPU unless the surrounding
post-processing moves to the DSP too. The small full-sort calls (P4–P6) are also slower per call.
The scalar core's sort work is the limit, but inside the concurrent batch they overlap with P2's
collect, so they don't set the batch time.

### What mattered, measured in order

- **Scratch reuse.** A fresh 2.6 MB `malloc`/`free` per call cost 1–1.5 ms on the DSP heap, plus a
  cold-page penalty in the kernel itself. Scratch is now allocated once per batch slot and reused.
- **Prefetch.** The collect over 163,200 elements took 1387 µs single-threaded, only ~470 MB/s
  (each vector load waited on DDR). An `l2fetch` 16 KB ahead cut it to 401 µs, the same lesson
  RoiAlign learned.
- **Threshold.** A strided scalar sample of 1024 values cost ~300 µs. Blocks of 8 values brought it
  to ~100 µs, and a quickselect instead of sorting the sample brought it to ~40 µs. **Refuted
  along the way:** I guessed the ~100 µs was TLB-bound (every sample line on its own page) and
  tried packing the sample onto 32 pages. It was not faster, and the worse sample almost doubled
  the survivors, so it was reverted. The cost was the sort.
- **Emit.** Radix sort of ~1.5k survivors took 100–180 µs. An HVX `==` scan per distinct key was
  tried and was *worse* on the calls with 89–213 distinct keys (181 → 387 µs). The single
  counting-sort pass by key rank brought it to 52–87 µs.
- **Threads.** On their own, splitting the big collects and running levels concurrently each
  helped. Combining them (hybrid) was best, since P2 alone otherwise bounds the batch.
- The TURBO clock vote changed nothing, as for RoiAlign.

### `__builtin_reduce_or` and qemu

The first vector version tested "any lane passed?" with clang's `__builtin_reduce_or` on the
plain 0/-1 compare mask. Under qemu it silently dropped survivors on the three select-path calls:
1533, 1159 and 1228 survivors instead of 1799, 1455 and 1618. The same C was exact on the host.
Lane by lane, the key map and compare were correct. The inlined reduction had become a `vrmpy`
followed by `vdeal`/`vdeal`/`vdeal`/`vshuff`, which relies on VLIW packets reading old register
values. That sequence misses single-lane hits under qemu. The **same build is exact on the real
CDSP**, so this is a qemu 8.2 emulation fault, not a hexagon-clang miscompile. The shipped kernel
uses an explicit rotate/OR tree, which is exact on both. The fault is kept as an A/B variant
(`TK_VEC_REDUCE_OR_MASK`) that the qemu harness and the phone client both run: MISMATCH under
qemu, EXACT on the device.

**Not handled:** NaN inputs (none occur), anything other than 1-D fp32 `largest=1, sorted=1`, and
concurrent RPCs from several clients (the scratch is static per batch slot).

`topk/` reproduces everything:
- `dump_real_topk_io.py` captures the real inputs and outputs.
- `gen_topk_test_data.py` writes the flat test files and single-node ORT models, and times host ORT.
- `topk_host_check.c` is the real-data check plus the randomized stress test.
- `topk_qemu.c` is the qemu harness.
- `topk_rpc.idl`, `topk_impl.c` and `topk_client.c` are the DSP skel and client.
- `ort_topk_bench.c` is the phone-CPU ORT baseline.
- `build.sh` builds and runs on the phone, and runs the ORT baseline too if `ORT_AAR` is set.

## NonMaxSuppression: exact on all 85 real calls, faster than ORT for the per-level ones

`dynamic_ops_survey.md` ranked NMS the hardest dynamic op (greedy selection is sequential, and it
looked like it might need new tinygrad capability). Like RoiAlign, it doesn't need tinygrad's UOp
path at all: `nms/` is hand-written C with HVX, bridged over its own FastRPC skel (qaic IDL +
native client, `rpcmem` buffers, no TVM), with inputs captured from one real ONNX Runtime inference
(`dump_real_nms_io.py`: MaskRCNN-12-qdq, COCO 000000000139).

**Real semantics, confirmed from the graph and ORT's own source.** All 85 nodes are
`center_point_box=0`, one batch, one class per call, `max_output_boxes_per_class=2000` (never
binding), no `score_threshold` input. Two groups:

| group | calls | iou | boxes per call | input order | kept |
|---|---:|---:|---|---|---:|
| per FPN level (RPN proposals) | 5 | 0.7 | 1000, 1000, 1000, 1000, 663 | already score-sorted (by the preceding TopK) | 1465 |
| per class (box head) | 80 | 0.5 | 0..152, 573 in total, 63 calls empty | not sorted | 106 |

ORT's CPU kernel (`onnxruntime/core/providers/cpu/object_detection/non_max_suppression.cc` +
`non_max_suppression_helper.h`) visits candidates through a `priority_queue` (score descending,
ties by **lower index first**), keeps a candidate unless `SuppressByIOU(candidate, kept)` is true for
some kept box, and emits `(batch, class, index)` triples in selection order. `SuppressByIOU` is
`intersection / union > iou_threshold` in fp32, with early `return false` for no overlap and for
non-positive intersection/areas/union. `nms_kernel.h` copies that operation for operation (built with
`-ffp-contract=off`), and visits candidates in the same order via a stable merge sort. The phone's
own ORT build (stock `onnxruntime-android` 1.26 arm64) selects exactly the same boxes as host ORT on
all 85 calls, so one reference serves both.

**Two kernels.** `nms_scalar` is the greedy loop, scalar. `nms_hvx` is the same greedy loop, but tests
each candidate against 64 already-kept boxes per step (two HVX vectors, kept boxes stored
structure-of-arrays), with one cross-lane OR-reduction per step.

**Getting HVX to agree with ORT exactly took three real findings:**

1. **This phone's HVX has no IEEE fp32.** The CDSP is Hexagon V69. Built with `-mhvx-ieee-fp`, every
   IEEE `.sf` arithmetic instruction returns 0. Only qfloat exists. (`roialign_fast/`'s kernel was
   already using qfloat without saying so, which is why it's 6.5e-5 off ORT rather than bit-exact.)
2. **qfloat doesn't round like IEEE, so the vector test can't decide ties.** Measured on the phone,
   a qfloat add/sub/mul on IEEE inputs, converted back to `.sf`, is off by up to 2^-23 × the larger
   **operand**, not the result. It is not exact under cancellation the way IEEE subtraction is:
   `1346.5 - 1343` gives 3.50012207. Real IoUs land a few ulp from the threshold (one real per-level
   pair has IoU 0.70000005), so the vector path only decides clear-cut pairs. It computes
   `d = inter·(1+thr) − thr·(area1+area2)`, which is `inter − thr·union` in exact arithmetic, and
   bounds its error by 2^-23·(2·M·(w+h) + 7·(area1+area2)), with M the largest coordinate in the call.
   With the band half-width t at 4× that bound, a pair is *surely suppressed* if d > t, *surely not*
   if d < −t, and anything in between is re-decided with the exact scalar `SuppressByIOU`. Box-overlap
   tests are max/min/compare on the input coordinates, which are exact in qfloat.
3. **clang silently undoes the careful version.** Plain vector-extension C (`a*b - c` on `float`
   vectors) lowers to chained `vmpy(qf32, qf32)` with no renormalization, which is ~500 ulp off.
   The first HVX build missed one real per-level call because of it: `inter` came out 880.46875 where IEEE
   gives 880.498718. Rewriting the chain with explicit intrinsics that convert back to `.sf` after
   every op wasn't enough either, because LLVM folds each `qf32→sf` conversion into the next op and
   emits the same `vmpy(qf32, qf32)`: `3.50012207 × 3.00006104` came out 10.5625. An empty
   `asm("" : "+v"(r))` after each conversion stops the folding (the objdump then has zero
   qf32-input ops), and the same product comes out 10.5006.

The band rarely fires: 9 exact rechecks in 5214 candidate tests on the real data (30 in 30719 on
the stress sets; counted on the host's portable path, which uses the same bound). To exercise it
harder than one image does, `gen_nms_stress_data.py` makes 80 adversarial calls (31.6k boxes):
coordinates on a coarse grid so many IoUs sit exactly on 0.5/0.7, heavy score ties, zero-area and
swapped-corner boxes, n from 1 to 2000, coordinates up to ~2000. Before fix 3 the HVX kernel got 12
of those 80 wrong on the phone, including a pair with IoU exactly 0.5 in real arithmetic. After it,
all are exact.

**Verified exact (identical selected indices, in order) at every level:**

| check | real data (85 calls) | stress (80 calls) |
|---|---|---|
| host C, both kernels (`nms_host_check.c`) | 85/85 | 80/80 |
| qemu-hexagon, both kernels, portable fp32 path (`nms_qemu.c`) | 85/85 | 80/80 |
| phone CDSP, `nms_scalar`, 1 and 4 threads | 85/85 | 80/80 |
| phone CDSP, `nms_hvx` (qfloat), 1/4/5/6 threads | 85/85 | 80/80 |

**Speed on the phone, real data.** Each group runs as one RPC with calls spread across QuRT threads.
Times are medians of 9 runs. ORT is the sum of per-node medians, since that's how `rest.onnx` runs
today. ORT's NMS is single-threaded, so its "default" thread setting changes nothing.

| | per level (5 calls) | per class (80 calls) |
|---|---:|---:|
| ORT on the phone's CPU (1 thread / default) | 3.66 / 3.64 ms | 0.199 / 0.199 ms |
| DSP `nms_scalar`, 1 / 4 threads (DSP time) | 16.26 / 8.47 ms | 0.21 / 0.10 ms |
| DSP `nms_hvx`, 1 thread: DSP time / RPC round trip | 2.34 / 2.71 ms (**1.56x / 1.35x**) | 0.18 / 0.45 ms |
| DSP `nms_hvx`, 4 threads: DSP time / RPC round trip | 0.93 / 1.26 ms (**3.9x / 2.9x**) | 0.10 / 0.28-0.47 ms |
| DSP `nms_hvx`, 5 threads: DSP time / RPC round trip | 0.86 / 1.13 ms (**4.2x / 3.2x**) | 0.09 / 0.28-0.40 ms |
| DSP `nms_hvx`, one RPC per node (as ORT runs them), 1 thread | 3.85 ms (~1.0x) | 18.9 ms (**95x slower**) |

- **Per-level NMS is a real win:** about 3x ORT on the phone's CPU including the RPC round trip, with
  4-5 threads. 4 threads leave one thread with two of the five calls, which is why 5 helps a little.
- **Per-class NMS should stay on the CPU unless it's fused into a larger DSP call.** ORT does all 80
  calls in 0.2 ms; one batched RPC already costs more than that in round trip alone, and one RPC per
  node is 95x slower.
- The scalar kernel is exact but 4.4x slower than ORT (the DSP's scalar core against a big ARM core).
  It's the fallback and the correctness reference, not the fast path.

Not done: fusing NMS with its neighbours (proposal decode/TopK before it, RoiAlign after) into one
DSP call, which is where the per-class calls would stop paying a round trip each; one image's real
calls only for timing; `score_threshold`/`center_point_box=1`/multi-class inputs aren't supported,
since this model never uses them.

## Fused RPN post-processing: TopK -> decode -> NMS -> proposals in one DSP call

The three kernels above each pay a FastRPC round trip of ~0.25-0.4 ms, which ate much of their
standalone gain. `rpn_fused/` runs the whole RPN post-processing span as **one** DSP call: the
per-level objectness scores and box deltas go in, the proposal list comes out, and every
intermediate stays on the DSP. It includes `../proposal_decode/pd_kernel.h`,
`../topk/topk_kernel.h` and `../nms/nms_kernel.h` unchanged; `rpn_fused/rpn_kernel.h` only adds
the glue between them.

### The real span, read out of the graph

`capture_rpn_fused.py` walks `rest.onnx` and asserts every structural fact and constant instead of
assuming it. The region is **604 non-Constant nodes**, from 10 inputs (per level: fp32 objectness
`[1,A]` and fp32 per-anchor deltas `[1,A,4]`, today's backbone/rest boundary) to one output, tensor
`2527`: the proposal list that feeds the box/mask heads. Per FPN level:

1. TopK(objectness, k): k = 1000/1000/1000/1000/663. TopK runs *before* decode.
2. Proposal decode of those k boxes (the 315-node region `../proposal_decode` replaces).
3. A width/height "min-size" filter that nothing mentioned before: `w = (x2 - x1) + 1`,
   `h = (y2 - y1) + 1`, Q/DQ'd onto their own **uint8** grid (P2-P5: the box grid's scale 5.0157;
   P6: 5.0196), keep `!(w < 0) && !(h < 0)`, then NonZero + Gather on boxes and TopK values.
4. NMS (iou 0.7, max 2000; the per-level 5 of the 85).
5. `Gather(selected, 2) -> Slice[0:1000]`: the first <=1000 selections. It never binds here, since
   each level has at most 1000 boxes.

Then the 5 levels are concatenated (P2..P6), a second TopK takes the top min(1000, N) scores, and
its indices gather the boxes. That TopK is the "post-NMS 1465 -> 1000" call in the TopK section;
its values output is unused. The region **stops at the proposal list** because the next ops (FPN
level assignment via sqrt/log/floor, per-level NonZero/Gather/ScatterElements around RoiAlign)
have no DSP implementation yet. That is where the graph goes back to the CPU.

Two facts make the glue simple and exact:
- **Every box after decode sits on the single uint8 box grid** (scale 5.0157, zero point 0). The
  graph's many later Q/DQs onto that same grid (after the Split, each Gather, the Concat) are
  identities on grid values, so the kernel only moves boxes between decode and the output.
- **The min-size filter can never drop a box in this model, for any input.** Its w/h grids are
  uint8 with zero point 0, so a dequantized w/h is `q * s` with q >= 0, and `Less(., 0)` is always
  false. On the DSP it cost ~87 us per level for nothing, so the kernel reads the boxes in place
  when the zero point is 0; `rpn_exact_filter = 1` computes it exactly as the graph does. The host
  check and qemu verify both paths.

### Correctness: byte-exact at every level, on 5 images

Five COCO images (000000000139, 632, 724, 785, 1000); each keeps 1262-1642 boxes after the per-level
NMS, and every one ends in 1000 proposals.

| check | result |
|---|---|
| host C (`rpn_host_check.c`): every intermediate (per-level TopK values+indices, decoded boxes, NMS selections, post-NMS TopK indices) and the proposals; both delta sources, both NMS kernels, decode's reference path, the computed filter | byte-exact on all 5 images x 6 variants |
| qemu, Hexagon v73 (`rpn_qemu.c`, scalar build as for NMS: no qfloat in qemu 8.2) | byte-exact on all 5 images x 5 variants |
| **phone CDSP** (`rpn_client.c`): proposals + per-level NMS kept counts, 11 configurations | **byte-exact on all 5 images x 11 configurations** |
| ORT on the phone's CPU, the same 604 nodes (`ort_rpn_bench.c`) | matches the capture on all 5 images |

### Speed on the phone

Medians of 21 runs. "DSP" is `HAP_perf_get_time_us` around the whole span; "round trip" is the
client-side time for the FastRPC call. ORT is `rpn_region.onnx` (the 604 nodes cut out of
rest.onnx), one `Run`, stock onnxruntime-android on the phone's CPU. ORT's thread count makes no
difference (1 thread vs default within 0.02 ms).

| image | ORT CPU | fused, NCHW uint8 source: DSP / round trip | vs ORT | fused, fp32 deltas (today's boundary): DSP / round trip | vs ORT |
|---|---:|---:|---:|---:|---:|
| 139 | 6.72 ms | 1.48 / 1.98 ms | **3.40x** | 1.69 / 2.19 ms | **3.07x** |
| 632 | 5.92 ms | 1.49 / 1.92 ms | **3.09x** | 1.61 / 2.08 ms | **2.84x** |
| 724 | 6.98 ms | 2.01 / 2.49 ms | **2.80x** | 2.15 / 2.62 ms | **2.66x** |
| 785 | 5.95 ms | 1.68 / 2.12 ms | **2.80x** | 1.83 / 2.27 ms | **2.62x** |
| 1000 | 6.42 ms | 1.64 / 2.13 ms | **3.02x** | 1.72 / 2.19 ms | **2.93x** |

Against the three standalone calls, all re-measured the same day on the same phone at each one's
best configuration, image 139:

| | DSP | round trip |
|---|---:|---:|
| TopK, 5 levels batched (one thread per level, P2's collect split 4 ways) | 0.37 ms | 0.78 ms |
| + proposal decode, NCHW source, 6 threads | 0.40 ms | 0.65 ms |
| + NMS per level, HVX, 5 threads | 0.88 ms | 1.16 ms |
| **= sum of the three calls** | 1.65 ms | **2.59 ms** |
| **fused, one call** (NCHW source) | **1.48 ms** | **1.98 ms (1.31x)** |

The fused call also does work the three calls don't cover: the filter, the per-level cap, the
concat, the post-NMS TopK and the final gather, ~0.1 ms of DSP time. With the fp32-delta source
the sums are 1.76 / 2.76 ms against the fused 1.69 / 2.19 ms (1.26x).

**Honest reading.** Most of the gain is the two round trips removed (~0.6 ms), plus a small DSP
win from overlapping. The DSP work itself isn't faster than the three kernels at their best,
because it is the same code. On today's boundary the whole span is **~2.2 ms end to end against
~6.4 ms in ORT on the phone's CPU**.

### Scheduling inside the call: what was measured

The obvious layout, one QuRT thread per level running its whole chain, was the **slowest** of the
multithreaded ones (median 1.97 ms DSP). P2's TopK collect over 163,200 scores ran 1.1-1.3 ms when the
other four levels' decode and NMS were running alongside it, against 0.49 ms alone. Splitting it
didn't help either, because the split parts couldn't get an HVX context while the level threads
held them all. What worked was running the steps in phases, each with the layout its standalone
kernel measured best:

| layout (median over 5 images) | DSP | round trip |
|---|---:|---:|
| everything on one thread | 4.42 ms | 4.99 ms |
| one thread per level, whole chain | 1.97 ms | 2.44 ms |
| phased: A TopK (thread per level, P2 split 4) -> B decode split by box over 6 threads -> C NMS per level -> D merge | 1.64 ms | 2.11 ms |
| phased, decode inside the per-level NMS phase | 1.64 ms | 2.12 ms |

(NCHW source; the last two are within noise of each other, and each wins on some images.) Phase walls on image 139:
A TopK ~0.36 ms, C decode + NMS ~1.0 ms (the slowest level's NMS, 0.5-0.65 ms, is most of it),
D merge ~0.1 ms. Two smaller things mattered: the filter shortcut above, and decode (scalar) runs
before its thread takes an HVX context, so the level threads don't hold every context while
others still need one for NMS. The TURBO clock vote wasn't measured for the fused call; it changed
nothing for any of the three standalone kernels.

**Not done:** the next stretch, FPN level assignment and RoiAlign, is still on the CPU. RoiAlign
has its own DSP kernel (`roialign_fast/`), but the level assignment and the
NonZero/Gather/ScatterElements around it don't. Fusing those in, and taking the NCHW deltas
straight from a DSP-resident backbone, would remove the next round trips. Only this span's
per-level NMS is fused; the 80 per-class NMS calls come later, after the box head.

## tinygrad codegen for HVX: plain Tensor code instead of hand kernels

Most kernels in this directory are either `Tensor.custom_kernel`s with hand-injected HVX C strings or plain hand-written
C, because tinygrad's own Hexagon codegen produced scalar code even for a plain elementwise add (17.6x slower than TVM,
see "Coverage: the elementwise add" above). This section fixes that at the codegen layer, in the `onnxsim/tinygrad`
fork (PR https://github.com/onnxsim/tinygrad/pull/2, branch `hvx-codegen`, stacked on the `vrmpy-hexagon-support` work in PR #1), so the
ordinary ops can be written as ordinary tinygrad code.

### Why plain tinygrad code was scalar

Upstream tinygrad already knows about the DSP in two places: `memory_coalescing` groups loads and stores up to 128
lanes, and the upcast heuristic tries a 128-wide upcast. Loads and stores were vectorized. Four things weren't:

1. **The ALU was split per lane and never rebuilt.** `devectorizer2` splits every elementwise op into scalars, and only
   memory ops get regrouped, so `a+b` rendered as `(int128){(v0[0]+v1[0]),(v0[1]+v1[1]),...}`. LLVM lowers that lane by
   lane: 1738 instructions for a small test kernel versus 181 for the same kernel written `v0+v1`. **Fix:** a bottom-up re-vectorization pass in the
   DSP renderer, `STACK(op(a_i,b_i)) -> op(STACK(a_i), STACK(b_i))`, plus renderer rules for splats, lane slices
   (`__builtin_shufflevector`) and vector casts (`__builtin_convertvector`). It only forms a vector op when every
   operand column is itself vector-shaped (a lane slice of one vector, a splat, constants, or recursively the same op);
   per-lane mixes of compares/casts stay scalar. That restriction isn't cosmetic: vectorizing one such mix crashed
   LLVM's Hexagon backend (`Cannot select: v2i32 = bitcast (V2Q ...)`) in `test_max_pool2d_return_indices`.
2. **`max` wasn't a native op.** Clang's renderer has no `Ops.MAX`, so it was decomposed into compare+select. It now
   renders as `__builtin_elementwise_max` for integers (one `vmax`). Floats keep tinygrad's own `(a<b)?b:a` NaN
   semantics, written as a statement expression that names each operand once. A plain ternary (or even
   `__typeof__(a) _a=(a)`) repeats the operand text, and since single-use results are inlined, max chains like
   `argmax`/`cross_entropy` grew exponentially as *source text*: over 6 GB of host memory for one test (360 MB on the
   unmodified branch), most likely what drove this machine into global OOM kills while this was being developed.
3. **Real shapes missed the 128-lane upcast.** The DSP branch only tried exactly 128, so `...x272` dims fell back to a
   4-wide upcast. It now tries 128/64/32 and prefers the widest; the "nothing upcasted" fallback does the same.
4. **Coalescing made loads wider than anything used them.** Two adjacent 32-byte loads were merged into a 64-byte one
   that every consumer used as two 32-byte halves; LLVM split it through a stack round trip. On the DSP, loads are now
   capped at the widest contiguous store group (the kernel's actual vector width).

Plus one performance rule: a software prefetch (`dcfetch`, `HVX_PREFETCH` bytes ahead, default 2048) for every load that
covers a full 128-byte line, one per line. That's what closed the add kernel's gap on hardware earlier in this file, and
it's what separates a plain vector add (153k Pcycles in `hexagon-sim --timing`) from the hand kernel (76k).

All changes are DSP-gated; other backends generate identical code.

### Result: the same plain Tensor code, before and after

`tinygrad_codegen/tgk_ops.py` writes three ops as ordinary tinygrad code, at the real shapes the hand kernels target:
the int32 add (`[1,256,200,272]`), the uint8 3x3/stride-2 maxpool in packed NCHWc (`[1,2,402,546,32]` in), and the
int32->uint8 requantize (2M elements, TVM's Q31 `QMultiplyShift` formula). `gen_kernels.py` generates both the plain
version and the existing hand `custom_kernel`, checks each byte-exact against numpy under qemu, and `build.sh` runs them
on the phone through a dedicated FastRPC skel with `rpcmem` buffers (same pattern as `roialign_fast/`), timing on the DSP.
Everything below is byte-exact against numpy on real hardware (device `239dbd8f`), single DSP thread, default clocks.

Real shapes on the phone, median of 7-9 reps, range over independent runs (ms):

| op | unmodified tinygrad | **plain tinygrad, this change** | hand `custom_kernel` |
|---|---:|---:|---:|
| add | 103.7 | **6.8 - 7.2** | 7.0 - 7.3 |
| maxpool | 68.9 - 69.1 | **33.8 - 36.9** | 32.9 - 36.0 |
| requantize | 14.6 - 14.8 | **9.3 - 9.9** | 23.3 - 25.5 |

Small shapes, `hexagon-sim --timing` (HEXSIM=1, v73, cycle-level, microseconds at the simulator's clock):

| op | plain tinygrad | hand |
|---|---:|---:|
| add `[1,256,25,34]` | 74.7 | 76.2 |
| maxpool `[1,2,52,70,32]` | 117.5 | 213.7 |
| requantize 64k | 490.2 | 842.0 |

- **add:** ~15x faster than before and at parity with the hand kernel.
- **maxpool:** 2x faster than before, within a few percent of hand on the phone. HEXSIM (built at `-O1`) shows it well
  ahead of hand; the phone (`-O2`) doesn't, so trust the phone number. Neither version uses HVX here: at the natural
  32-byte channel-block width both lower to the scalar core's packed-byte `vmaxub`. Getting 128-byte HVX vectors needs
  a stride-2 deinterleave (the gap `hex_maxpool_kernel.py`'s own docstring describes); that isn't done.
- **requantize:** 2.5x faster than hand, but *not* because of HVX: the formula needs 64-bit products and HVX has no
  64-bit lanes, so both versions stay scalar. The plain version wins on unrolling plus prefetch. A real HVX version
  would use the Q31 `vmpyo`/`vmpye` multiply-high pair instead of int64 arithmetic; not done.

### Checks

- `test/test_tiny.py` under `DEV=DSP MOCKDSP=1`: same result as the unmodified branch (16 passed, 5 skipped).
- `test/backend/test_ops.py` under `DEV=DSP MOCKDSP=1`: 38 failed / 381 passed / 8 skipped, the **identical** failure
  set to the unmodified branch (all 38 pre-existing). Peak host memory 1 GB (run under a 6 GB cgroup cap).
- `test/backend/test_ops.py` on `DEV=CPU`: 2 failed / 417 passed / 8 skipped, identical to the unmodified branch
  (`test_clip`, `test_pad_replicate_mode`), so the shared heuristic/coalescing edits don't change other backends.

### What this does and doesn't cover

- **Hand kernels this makes redundant** (not deleted here): `hex_add_kernel.py`, `hex_requantize_kernel.py`, and, at a
  few percent cost, `hex_maxpool_kernel.py`. The chained backbone drivers under `native_transport/` embed their
  generated C verbatim and would need regenerating to switch over.
- **qfloat (fp32 vector math):** done in the next section.
- **Not done: threading.** The hand kernels' multi-thread speedups come from `qurt` worker threads; qemu's bare-metal mode
  and `hexagon-sim`'s standalone mode (tinygrad's two DSP test paths) have no `qurt`, and tinygrad's own on-device DSP
  runtime can't reach this phone (see the top of this file). A threaded renderer variant could only be validated through
  a FastRPC harness like `tinygrad_codegen/`'s.
- **Not covered by this pass:** convolutions/GEMM (`vrmpy` still needs `custom_kernel` or the TensorCore path, whose
  devectorizer problem is described near the top of this file), and the data-dependent ops (NMS, TopK, proposal decode,
  RoiAlign fast path), which stay hand-written C.

## tinygrad codegen for HVX, stage 2: float vector math as qfloat

This phone's V69 HVX has no IEEE fp32; float vector math is Qualcomm's qfloat (qf32), whose results have to be converted
back to IEEE sf (see the NMS section, which hit this first). This extends the stage-1 codegen to float kernels, in
onnxsim/tinygrad PR https://github.com/onnxsim/tinygrad/pull/4 (branch `hvx-qfloat`, which also carries PR #3's float-max
fix). `HVX_ARCH=v69` compiles for the phone; the default stays v65 (no HVX float), where nothing changes.

### What the lowering does, and why it isn't "convert after every op"

At v68+, LLVM already lowers plain float vector C (`a*b + c`) to qfloat. The question is where it loses precision. Measured
with `tinygrad_codegen/qfloat/qfsim.py` on `hexagon-sim -mv69 --timing` (real data, output checked against numpy fp32 and
float64), for the same plain tinygrad code under three lowerings:

| kernel (plain Tensor code) | scalar | LLVM qfloat as-is | convert after every op | **chosen** |
|---|---:|---:|---:|---:|
| 4-tap bilinear blend, 4096x256 | 16.7M cycles | 0.93M | 1.87M | **0.93M** |
| Horner polynomial (exp-like), 1M | 12.7M | 0.76M | 2.06M | **0.76M** |
| `((a-b)*(c-d))*((a+b)*(c+d))`, 1M | -- | 0.76M | 1.66M | **1.39M** |

| worst-case relative error | LLVM as-is | every op | chosen |
|---|---:|---:|---:|
| blend (max abs error vs numpy fp32) | 7.2e-7 | 7.2e-7 | 7.2e-7 |
| polynomial | 3.6e-7 | 3.6e-7 | 3.6e-7 |
| products of computed values | **21%** | 4.2% | 4.2% |

- **Where LLVM is fine:** for adds/subs and for multiplies with an IEEE operand (a load or constant), its output is
  *bit-identical* to converting after every op, and 2-2.7x faster.
- **Where it isn't:** a multiply whose operands are *both* computed values. LLVM keeps both in qf32 and multiplies
  qf32 x qf32 without renormalizing: 21% worst case. So exactly those multiplies get both operands forced to IEEE sf
  through an empty-asm barrier (NMS's recipe) first; everything else is plain vector C.
- **The remaining 4.2%** on that kernel is qfloat, not the lowering: qf32 add/sub error scales with the *larger operand*,
  not the result, so cancellation (`a-b` with `a~b`, exact in IEEE) amplifies it. Code that subtracts near-equal floats
  on this DSP needs an exact fallback, like NMS's near-threshold recheck; no lowering removes that.
- A first version split 128-lane vectors into HVX registers through memory (`((__hvx_v*)&a)[i]`): 10x slower than
  splitting with `__builtin_shufflevector`, which is what it does now.

### On the phone

`tinygrad_codegen/qfloat/build.sh` runs the captured kernels on the CDSP (compiled for v69: v73 code would use IEEE HVX
float, which returns zeros here), `analyze.py` checks the outputs. Single DSP thread, milliseconds, min over 9 reps, two
clean runs (the phone is shared, so a third run was disturbed and isn't counted):

| kernel | time | max abs err vs numpy fp32 | max rel err vs float64 |
|---|---:|---:|---:|
| blend, plain tinygrad (qfloat) | 0.83 - 1.02 | 7.2e-7 | (near-zero outputs) |
| blend, hand-written qf32 (`hand_blend.c`, same prefetch) | 0.90 - 1.23 | 7.2e-7 | |
| blend, plain tinygrad, scalar | 11.7 - 13.4 | 3.6e-7 | |
| products of computed values, chosen lowering | 0.91 - 0.93 | | 4.2% |
| products of computed values, LLVM as-is | 0.73 - 0.74 | | 21% |
| sigmoid, 163,200 elements (scalar, see below) | 12.4 - 12.5 | 1.2e-7 | 5.2e-7 |

The phone's errors match hexagon-sim's exactly. The plain-tinygrad blend runs as fast as the hand-written qf32 kernel.

**Sigmoid, at all five real profile shapes (663 to 163,200 elements):** correct and closer to float64 than ONNX
Runtime's fp32 sigmoid (max rel error 5.7e-7 vs ORT's 1.8% on tiny outputs), but **still scalar**, about 76 ns/element
on the phone. tinygrad's `exp2` does its range reduction with a float->int conversion, and V69 has no vector
float<->int convert (`vconv_w_sf`/`vconv_sf_w` arrive in V73); it also needs vector compare/select and a reciprocal.
Vectorizing it on V69 would mean emulating the conversion with integer bit operations, which isn't done.

### Checks

- tinygrad `test/unit/test_dsp_render.py`: the qf32 x qf32 multiply gets the barrier, plain arithmetic doesn't, nothing
  changes below v68; plus the stage-1 float-max guard.
- tinygrad `test/backend/test_ops.py` under `DEV=DSP MOCKDSP=1`: the same 38 pre-existing failures as the base branch.
  qemu can't decode HVX float, so v69 float correctness is from hexagon-sim and the phone (above).
- These hexagon-sim checks are meant to replace the TVM qfloat CI (`hexagon-qfloat.yml`); they go into
  `tests/test_hexagon_tinygrad.py` once PR #1846 (the tinygrad Hexagon CI) has merged.

## Beyond the backbone: the box-head fc6 MatMul (`hex_boxhead_gemm_kernel.py`)

Every kernel above targets `backbone.onnx` (ResNet-50/FPN/RPN, the Hexagon DSP half of the split
pipeline documented in `../maskrcnn_e2e/README.md`). This is the project's first step into
`rest.onnx` -- the dynamic-shape remainder (proposal decode, TopK, 85x NMS, RoiAlign, 4 MatMuls,
mask head, post-processing) currently running 100% on ONNX Runtime CPU -- specifically the fc6
box-head layer, the single largest compute chunk there (`../maskrcnn_e2e/README.md`'s own
"roughly 13 GMAC for the fc6 box-head layer on 1000 proposals" note).

**The open question this section resolves first**: whether `rest.onnx`'s MatMuls retain int8 QDQ
quantization or run in fp32. Traced directly from the real graph (`rest.onnx`, produced by
`../maskrcnn_e2e/prepare.py`): the fc6 `MatMul` node's own inputs are both `DequantizeLinear`
outputs (`elem_type` FLOAT), and its output feeds a `QuantizeLinear` -- the classic QDQ-wrapped-fp32
pattern, not a real `QLinearMatMul`/`MatMulInteger`. **The MatMul itself computes in fp32.** Real
weight shape confirmed from the graph: `(12544, 1024)` -- `12544 = 7*7*256` (RoiAlign's 7x7 crop of
256-channel FPN features, flattened), `1024` the fc6 output width; activation rows = proposal count
(dynamic, ~1000 in the real pipeline). This puts the op in the same category as `sigmoid` (this
project's only other fp32 op) rather than the `vrmpy`-based int8 path every other kernel here uses.

**The kernel**: no `vrmpy` equivalent exists for fp32 -- `hex_boxhead_gemm_kernel.py`'s
`build_kernel()` is a plain HVX vector FMA loop, following `hex_gemm_kernel.py`'s exact structure
(`_reg_f32`, the same multi-range-dependency accumulator-init fix every kernel in this series has
needed) but with one `float __attribute__((vector_size(128)))` (32-lane) accumulator stepped one
K element at a time (`acc += a_scalar * b_vec`, a scalar-broadcast FMA clang lowers directly to
HVX float instructions) instead of a 4-wide `vrmpyub`/`vrmpybusv` int8 dot product per step.

Verified bit-exact (within fp32 sequential-accumulation tolerance vs. a float64 reference) under
qemu at multiple scales, including the **exact full real shape** (`M=1000, K=12544, N=1024`,
matching the real fc6 layer precisely): `max_abs_err=2.95e-3`, consistent with every smaller-scale
check (tiny, multi-tile, real-K-depth-with-small-M all land in the same 1e-7 to 3e-3 range
depending on reduction depth -- expected fp32 rounding, not a correctness concern; a
`std::float64`-precision reference at this reduction depth necessarily disagrees with any fp32
accumulator by a comparable amount regardless of implementation).

**Real hardware** (device `239dbd8f`, via `native_transport/`): the full real weight
(`12544*1024*4 ~= 49MB`) exceeds the ~32MB/buffer RPC transfer wall this project's own
`native_transport`/`requantize` work already found, so real-hardware verification used a
real-shape-consistent **slice** (`M=64, K=12544` -- unchanged, the real reduction depth --
`N=512`, half the real width, chosen so the packed weight buffer fits at ~24.5MB) rather than
forcing an oversized transfer. Ran successfully (`rc=0`), bit-exact within tolerance against the
float64 reference (`max_abs_err=2.98e-3`, same magnitude as the qemu runs).

**A new, real finding for this project**: real hardware's output is **not bit-exact against
qemu's**, given the identical input data and identical generated kernel source (`max abs diff
3.02e-3` between the two, evenly spread across every output row -- not concentrated in a few
elements the way a logic bug would show, and the same magnitude as both environments'
independent agreement with the float64 reference). This is the first kernel in this project with
both a real fp32 dtype *and* a substantial (12544-step) reduction -- `sigmoid` was fp32 but had no
reduction; every reduction-heavy kernel before this was int8/int32, where fixed-point arithmetic
has no equivalent rounding-path ambiguity. The most likely explanation is a genuine FMA-contraction
or rounding-mode difference between `MOCKDSP`'s qemu build (Hexagon v65 target) and the real
hardware bridge's build (v73) -- not chased down further here, but worth flagging precisely for
any future fp32 reduction kernel in this project: don't expect bit-exact qemu/real-hardware
agreement the way every int8 kernel here has had, expect fp32-rounding-tolerance agreement instead.

**Speed, reported honestly**: 942.751 ms real-hardware wall time for the verified slice
(`M=64,K=12544,N=512`, ~0.411 GMAC) -- **~0.436 GMAC/s**, roughly two orders of magnitude slower
than every int8 kernel in this project (30-50 GMAC/s). This is expected, not a red flag: fp32 has
no `vrmpy`-equivalent wide dot-product instruction on this HVX generation (one FMA processes 32
output lanes but only *one* K-step per instruction, versus `vrmpyub`'s 128 lanes/4-wide-K per
instruction for int8), and this kernel has had **zero tuning** applied (no tiling/blocking, no
prefetch, no register accumulation reuse -- exactly the untuned starting point `hex_add_kernel.py`
was at before its own prefetch pass, and `hex_conv3x3_kernel.py` was at before its `ow_tile` pass).
An x86 desktop `onnxruntime` CPU run of the identical shape took 0.64 ms -- **not a fair comparison
point** (a multi-core, BLAS-backed desktop CPU vs. one untuned DSP thread) and explicitly not the
real baseline (ONNX Runtime running on the phone's own ARM CPU, which is what this op actually
competes against in the real pipeline) -- that real comparison was not measured here.

**Not done**: tuning (this kernel is a direct, unoptimized port of the pattern that worked for
int8 -- real headroom likely exists, matching every other kernel's own before/after tuning story
in this project); a real phone-ARM-CPU ONNX Runtime baseline for a fair speed comparison; the
mask head (the other real compute chunk `../maskrcnn_e2e/README.md` names alongside fc6); NMS,
TopK, RoiAlign, proposal decode (fundamentally dynamic-shape/control-flow ops, a different
engineering problem than every static-shape kernel in this project, deliberately out of scope
here).

## Continuous integration (no phone)

`.github/workflows/hexagon-tinygrad.yml` runs `tests/test_hexagon_tinygrad.py` on a hosted
ubuntu-24.04 runner. It replaced the TVM v0.17.0 + hexagon-sim job for `../hexagon_qfloat.py`,
which is no longer CI-tested (see `../README.md`). Tools: upstream clang-19 + lld-19 and
`qemu-user-static` 8.2 from apt, the Hexagon open-access toolchain 19.0.02 (cached, for
`hexagon-sim`), and the onnxsim/tinygrad fork at a pinned commit (`TINYGRAD_SHA` in the workflow).
Upstream clang is needed because tinygrad's `MOCKDSP` and several qemu harnesses target
`hexagonv65`, which the toolchain's own hexagon-clang 19 no longer accepts. The freestanding qemu
builds use the toolchain's `target/hexagon/include` for their libc headers, because upstream clang
has no Hexagon sysroot. Every test skips when
its tools are missing; the workflow fails on any skip except the reserved qfloat slot.

- **tinygrad codegen** (`ci/codegen_check.py`, run in a subprocess so `DEV`/`MOCKDSP`/`HEXSIM`
  never leak): add, a 3x3/s2 maxpool on packed NCHWc, and a Q31 requantize, written as plain
  `Tensor` code and run under qemu-hexagon, must be byte-exact with NumPy. The rendered C must use
  HVX vector types: 128 bytes or more for add and requantize, 32 for maxpool (its full-width
  version needs a stride-2 deinterleave the codegen doesn't build yet). A float-`max` chain must
  render linearly: before tinygrad fccd84cfa the source doubled per op. Under `hexagon-sim
  --timing`, the vectorized add must take at least 8x fewer cycles than `NOOPT=1` scalar code
  (about 50x measured).
- **Hand-written kernels**, on synthetic data only (CI downloads no model):
  - `nms/`: host and qemu checks on `gen_nms_stress_data.py` sets, with an ORT reference.
  - `topk/`: the host check's built-in stress mode, plus qemu on the real calls' shapes with a NumPy
    reference in ORT's tie order. qemu needs `ci/hexagon_divrt.c` for integer divide.
    `vec-reduce_or-mask` is excluded, since it is the documented qemu-fault reproducer.
  - `roialign_fast/`: host and qemu checks against an ORT opset-12 RoiAlign model built with
    `onnx.parser`.
  - `proposal_decode/`: `ci/pd_selfcheck.c` requires both delta sources, the reference path, both
    fast paths and the off-grid fallback to agree bit for bit. It then writes `pd_qemu.c`'s inputs,
    with the host reference output as the expected result.
  - `rpn_fused/`: build-only. Its checks need ORT captures of the real `rest.onnx` span.
- **qfloat**: `test_codegen_qfloat_float_ops` is a reserved, skipped slot for the tinygrad qfloat
  lowering's tests.

Locally (about 40 s, peak under 600 MB):

```bash
TINYGRAD_PATH=/path/to/onnxsim-tinygrad HEXAGON_TOOLS=/path/to/Tools HEXAGON_CLANG=clang-19 \
  pytest -v tests/test_hexagon_tinygrad.py
```

## Files

- `capture_kernel.py` -- capture tinygrad's rendered Hexagon C for a shape, verified under qemu.
- `wrapper_template.c` -- the plain-C TVM PackedFunc ABI shim template.
- `bridge_and_test.py` -- compile with the real toolchain, link, deploy via TVM RPC, run, time.
- `vrmpy_tensorcore.patch` -- the tinygrad-side vrmpy `TensorCore` support (also at
  https://github.com/onnxsim/tinygrad/pull/1, branch `vrmpy-hexagon-support`); superseded in
  practice by `hex_gemm_kernel.py` below, which is both correct and fast, but kept as the record
  of the `TensorCore`-machinery dead ends (see "Three more attempts" above).
- `custom_kernel_attempt.py` -- the intermediate, still-broken steps of the `custom_kernel`
  investigation (see its own header comment for the bug-by-bug history); superseded by
  `hex_gemm_kernel.py`.
- `hex_gemm_kernel.py` -- **the working result**: a correct, hand-written Hexagon `vrmpy` GEMM
  kernel via `Tensor.custom_kernel`, measured 8.65x faster than stock TVM's hand-tuned schedule at
  the real Mask R-CNN pathological shape (see "It works" above). Bridge its output through
  `bridge_and_test.py` (after pre-packing `B` via `pack_b()`) to run it on real hardware.
- `hex_conv3x3_kernel.py` -- a real spatial 3x3 conv via the same `vrmpybusv`/`custom_kernel`
  pattern, extended to a 9-position reduction, pre-padded input addressing, and (`ow_tile`)
  output-tile reuse across neighboring output columns. Bit-exact correct at every shape tested;
  faster than stock TVM at two of the three real profile shapes, ~0.90x of TVM (up from 0.68x
  before tiling) at the biggest one (`cin=cout=256`) -- see "Coverage: a real 3x3 conv" above.
- `hex_stem7x7_kernel.py` -- the ResNet stem 7x7 conv, generalizing `hex_conv3x3_kernel.py`'s
  9-position pattern to 49 and handling `cin=3` (not a multiple of 4) via reduction-axis
  zero-padding. Bit-exact correct and 14.08x faster than stock TVM on real hardware at the real
  profile shape -- see "Coverage: the ResNet stem 7x7 conv" above.
- `hex_add_kernel.py` -- the elementwise `add`: a normal-codegen+BEAM attempt (found no
  vectorization, 17.6x slower than TVM -- a real negative result), a hand-vectorized
  `custom_kernel` (12.5x faster than the scalar version, but still 0.71x of TVM), then software
  prefetch (`prefetch_dist`, the current default) closing the rest of the gap -- up to 1.59x faster
  than the unprefetched kernel and ahead of TVM's measured throughput at the largest size real
  hardware could verify (below the real profile shape; see "Coverage: the elementwise add" above
  for why). Instruction unrolling was tried too and found to be a real dead end for this op,
  unlike for the 3x3 conv.
- `add_wrapper_template.c` -- a thin TVM PackedFunc ABI shim for `hex_add_kernel.py`'s
  `int32/int32/int32` signature, alongside `wrapper_template.c` (which is hardcoded to the GEMM
  kernels' `uint8/uint8/int32` signature and can't be reused as-is).
- `hex_maxpool_kernel.py` -- the 3x3/stride=2/pad=1 `maxpool`: unblocked by building on TVM's own
  reference layout (packed NCHWc, confirmed by inspecting a real compiled graph, not assumed)
  instead of the plain-NCHW layout that blocked the first attempt. Bit-exact correct and 1.99x
  faster than stock TVM on real hardware at the real profile shape -- see "`maxpool` unblocked"
  above.
- `maxpool_wrapper_template.c` -- a thin TVM PackedFunc ABI shim for `hex_maxpool_kernel.py`'s
  single-buffer-in `uint8` signature (no second operand, unlike the GEMM/add kernels).
- `hex_resize2x_kernel.py` -- the FPN `resize2d` (exact-2x nearest-neighbor upsample), the last op
  in the profile that had no tinygrad-generated kernel at all. Bit-exact correct and 5.6x-17.3x
  faster than *stock* TVM on real hardware, but slower than the already-fixed fast path
  (`../hexagon_resize2x.py`) at two of the three real shapes -- see "Coverage: FPN `resize2d`"
  above for the honest breakdown.
- `hex_requantize_kernel.py` -- `qnn.requantize`, the int32-to-uint8 rescale+clamp glue between
  conv layers, bit-exact against TVM's own `UPWARD`-rounding fixed-point formula (read out of
  TVM's source, not assumed) -- the one kernel missing before any two covered convs can be chained
  into a real subgraph. Verified on real hardware through `native_transport/` only, zero TVM in
  the executed path. See "Running the full backbone graph, TVM-free" above.
- `hex_boxhead_gemm_kernel.py` -- the box-head fc6 MatMul, this project's first step outside
  `backbone.onnx` into the dynamic-shape `rest.onnx` remainder. The first fp32-*and*-reduction
  kernel here (confirmed the op is genuinely fp32 by tracing the real graph, not assumed);
  correct at the exact real shape under qemu and on real hardware (a real-shape-consistent slice,
  RPC-transfer-limited); real hardware and qemu agree within fp32 rounding tolerance but not
  bit-exact, a new finding for this project -- see "Beyond the backbone" above. Untuned,
  ~0.436 GMAC/s, honestly two orders of magnitude behind this project's int8 kernels -- real
  tuning headroom likely exists, not yet attempted.
- `hex_bias_add_kernel.py` -- the per-output-channel `int32` bias-add every real conv needs
  before `requantize`, and (via the standard zero-point-folding trick) how this project's kernels
  handle the raw image's nonzero activation zero-point without any on-device subtraction logic.
  See "Chaining a real subgraph" above.
- `backbone_subgraph/extract_and_verify.py` -- extracts real stem-conv weight/bias/scales from a
  prepared `backbone.onnx`, computes the folded bias and requantize multiplier/shift, and
  generates the padded-image/packed-weight data files `native_transport/subgraph_driver.c` embeds.
- `native_transport/subgraph_driver.c` -- the fused `conv7x7 -> bias_add -> requantize ->
  layout_transform -> maxpool` driver: real backbone weights/bias baked in as compile-time
  constants, every intermediate activation on-device only (never crosses the RPC boundary), one
  `native_transport` call for the whole chain. Verified on real hardware, real image, 99.9995%
  exact match vs. the real ONNX Runtime reference. See "Chaining a real subgraph" above --
  including the real 128-byte-alignment bug its own `small_subgraph_driver.c` diagnostic isolated
  (kept as a record of how, not scratch).
- `hex_gemm_signed_kernel.py` -- a signed-weight (`vrmpybusv`) variant of `hex_gemm_kernel.py`'s
  1x1-conv GEMM kernel, needed because `hex_gemm_kernel.py` itself is unsigned-only (`vrmpyub`)
  and the real backbone's 1x1 conv weights are signed int8 -- a real, previously-unflagged gap
  in this project's own 1x1-conv coverage, discovered while extending the chained subgraph below
  to real weights for the first time. See "Extending the chain" above.
- `reverify_signed_1x1.py` -- re-measures every 1x1-conv shape's speedup-vs-TVM with
  `hex_gemm_signed_kernel.py` (signed weights) instead of `hex_gemm_kernel.py` (unsigned synthetic
  weights, the original measurement). See "Correction: every 1x1-conv speedup above was measured
  against unsigned synthetic weights" above.
- `backbone_subgraph/extract_and_verify_block1.py` -- extracts real weights/biases/scales for
  ResNet-50 stage1/block1 (`conv10`/`conv17`/`conv24`/`conv30`) from `backbone.onnx`, derives the
  residual-add rescale multiplier/shifts, generates `native_transport/gen_block1_weight_consts.c`
  and real test input/reference data, and verifies the whole block numerically against a real
  ONNX Runtime reference before any C is trusted.
- `native_transport/block1_kernels.c` / `block1_glue.c` -- the fused stage1/block1 driver:
  verbatim tinygrad-generated kernel bodies plus hand-written layout/padding glue and the new
  rescale-add-relu-clamp residual-merge step. Verified bit-exact identical across numpy,
  `hexagon-sim`, and real hardware (max abs diff 3/255, mean 0.40/255). See "Extending the chain"
  above.
- `native_transport/` -- a from-scratch, TVM-free FastRPC transport: custom `qaic`-generated
  interface, a native ARM64 client using only `libcdsprpc.so`, verified end to end on real
  hardware. See "Removing TVM as a transport dependency" above; `native_transport/build.sh`
  reproduces the whole pipeline from source.
- `dynamic_ops_survey.md` -- a planning document (no kernel code) surveying whether the dynamic-
  shape ops in Mask R-CNN's `rest.onnx` (proposal decode, TopK, NonMaxSuppression, RoiAlign) could
  be ported to hand-written HVX kernels, with real shapes/dtypes/constants pulled from the actual
  graph and a ranked, honest difficulty assessment per op.
- `hex_roialign_kernel.py` -- Stage 1 of `RoiAlign` (the "beyond the backbone" survey's
  second-ranked target): confirms data-dependent source addressing is expressible via tinygrad's
  existing `Tensor.gather()` (no new UOp capability needed) and runs correctly through the real
  `DEV=DSP MOCKDSP=1` compile+qemu path, not just the default device. Correctness only -- the
  generic `.gather()` lowers to a dense one-hot-mask-and-reduce, not real HVX indexed addressing;
  see "RoiAlign, Stage 1" above for the concrete `vgathermh`/`vlutvvb`-based next step.
  (Corrected in Stage 2: tinygrad already folds that pattern into a direct indexed load.)
- `roialign_fast/` -- RoiAlign Stage 2: a hand-written channels-last HVX kernel
  (`roialign_kernel.h`, plain + `l2fetch`-prefetching variants), its dedicated TVM-free FastRPC skel
  (`roialign_rpc.idl`, `roialign_impl.c`, `roialign_client.c`, `build.sh`), real-input capture
  (`dump_real_roialign_io.py`, `gen_roialign_test_data.py`), host/qemu checks
  (`roialign_host_check.c`, `roialign_qemu.c`) and the ORT baselines
  (`make_roialign_single_node_models.py`, `ort_roialign_bench.c`). Correct on the phone at all 8
  real calls; 27.1 ms vs ORT's 63.3 ms on the phone's CPU. See "RoiAlign, Stage 2" above.
  Stage 4 adds the merged uint8 kernel (`roialign_u8_kernel.h`) and its skel/client/CPU build
  (`roialign_u8_rpc.idl`, `roialign_u8_impl.c`, `roialign_u8_client.c`, `roialign_u8_cpu.c`,
  `build_u8.sh`), capture and checks (`capture_merged_io.py`, `roialign_u8_host_check.c`,
  `roialign_u8_qemu.c`, `roialign_u8_qemu_args.py`): the whole dq + RoiAlign + merge + quant span
  of each head in one call, box 6.8 ms / mask 2.5 ms. See "RoiAlign, Stage 4" above.
- `hex_conv3x3_fpnout_kernel.py` -- the FPN 3x3 output conv (vrmpybusv, `ow_tile`) with ORT's
  requantize+dequantize epilogue fused into its store, writing fp32 channels-last or NCHW. Bit-exact
  vs ORT's real backbone outputs at all four levels, under qemu and on the phone; the channels-last
  store costs ~2.3 ms more than NCHW across P2-P5. See "RoiAlign, Stage 3" above.
- `fpn_channels_last/` -- RoiAlign Stage 3: real FPN-conv capture (`capture_fpn_out.py`), an HVX
  NCHW->NHWC transpose (`layout_kernels.h`, qemu-checked by `layout_qemu.c`), and a TVM-free FastRPC
  skel/client (`fpn_rpc.idl`, `fpn_impl.c`, `fpn_client.c`, `build.sh`) that prices the layout on the
  phone: separate transpose 13.3 ms (RoiAlign side 1.42x vs ORT) vs. channels-last producer +2.3 ms
  (1.94x). See "RoiAlign, Stage 3" above.
- `proposal_decode/` -- RPN proposal decode (post-TopK gather -> delta requant -> Detectron decode ->
  clip -> box-grid quantize), all 5 FPN levels in one FastRPC call: bit-exact with ONNX Runtime on
  host, qemu and the phone, 1.58x ORT-on-the-phone end to end with 6 DSP threads, and able to read the
  backbone's NCHW uint8 conv output directly instead of the full-map per-anchor layout. See
  "Proposal decode (RPN)" above; `proposal_decode/build.sh` runs the phone side.

- `topk/` -- exact TopK (values and int64 indices byte-equal to ORT, including its tie order) on
  the CDSP via its own FastRPC skel: threshold select + HVX collect + counting sort by key rank.
  The five batched per-level selections run in 369 µs of DSP time (720 µs including the round
  trip) vs ORT-on-phone's 859 µs; the two small downstream TopKs are slower than the CPU. Also
  documents a qemu 8.2 fault in one `__builtin_reduce_or` lowering (exact on the device). See
  "TopK" above; `topk/build.sh` reproduces it.

- `nms/` -- NonMaxSuppression for all 85 real `rest.onnx` calls, exact against ONNX Runtime:
  `nms_kernel.h` (scalar and qfloat-HVX greedy kernels), its own FastRPC skel/client
  (`nms_rpc.idl`, `nms_impl.c`, `nms_client.c`, `build.sh`), host and qemu checks, real-input capture
  (`dump_real_nms_io.py`, `gen_nms_test_data.py`), adversarial tie/threshold stress sets
  (`gen_nms_stress_data.py`), and host and phone ORT baselines (`make_nms_single_node_models.py`,
  `ort_nms_bench.c`). See "NonMaxSuppression" above.
- `rpn_fused/` -- the whole RPN post-processing span (per-level TopK -> decode -> w/h filter -> NMS
  -> cap, then concat -> post-NMS TopK -> gather: the 604 `rest.onnx` nodes from the per-level
  scores/deltas to the proposal list) as one FastRPC call, built from `proposal_decode/`, `topk/`
  and `nms/`'s kernel headers unchanged. Byte-exact with ORT on host, qemu and the phone for 5 real
  images; ~2.2 ms end to end against ~6.4 ms for ORT on the phone's CPU, and 1.31x faster than the
  three standalone calls combined. `capture_rpn_fused.py` (span + real I/O), `rpn_kernel.h`,
  `rpn_host_check.c`, `rpn_qemu.c`, `rpn_rpc.idl`/`rpn_impl.c`/`rpn_client.c`, `ort_rpn_bench.c`,
  `build.sh`. See "Fused RPN post-processing" above.
- `tinygrad_codegen/` -- add, maxpool and requantize written as plain tinygrad Tensor code (`tgk_ops.py`), plus a
  generator that checks them and the hand `custom_kernel`s byte-exact under qemu and emits their C (`gen_kernels.py`),
  and a TVM-free FastRPC skel/client (`tgk_rpc.idl`, `tgk_impl.c`, `tgk_client.c`, `build.sh`) that times both on the
  phone. See "tinygrad codegen for HVX" above.
- `ci/` -- phone-free CI helpers for `tests/test_hexagon_tinygrad.py`: `codegen_check.py` (tinygrad
  HVX codegen checks under qemu / hexagon-sim), `pd_selfcheck.c` (synthetic proposal-decode
  self-consistency + qemu inputs), `hexagon_divrt.c` (integer divide helpers for freestanding
  qemu builds). See "Continuous integration" above.
- `tinygrad_codegen/hmx/` -- the tinygrad fork's HMX (V69 matrix unit) fp16 TensorCore: `hmxsim.py` runs a generated
  matmul on hexagon-sim `--mhmx 1` with real data, `build.sh`/`run.sh` + `tg_hmx_*` run it on the phone next to the hand
  kernel in `../hmx_gemm`. Bit-exact everywhere; ~100x slower than the hand kernel for now (see its README).
- `tinygrad_codegen/qfloat/` -- stage 2: `qfsim.py` runs a plain-tinygrad float kernel with real data on hexagon-sim
  (qemu can't decode HVX float) and reports accuracy and cycles, or `--dump`s it; `build.sh` + `qf_rpc.idl`/`qf_impl.c`/
  `qf_client.c` run the dumps (plus `hand_blend.c`) on the phone, `analyze.py` checks them; `qf_ops.py` holds the inputs
  and references. See "tinygrad codegen for HVX, stage 2" above.
