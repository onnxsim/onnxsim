# Big-endian status

onnxsim works on big-endian hosts. The test suite produces the same result on
s390x as on x86_64, and the C++ unit tests pass on both.

This was not true before: constant folding was disabled on big endian and failed
quietly, so `simplify()` returned a model that was semantically valid but barely
simplified. This document records what was wrong, what changed, and how to
re-check it.

## Why byte order matters here

ONNX fixes the byte order of tensor payloads. From `onnx.in.proto`, on
`TensorProto.raw_data`:

> When this raw_data field is used to store tensor value, elements MUST be
> stored in as fixed-width, little-endian order.

So `raw_data` is little-endian on *every* host. DLPack, by contrast, carries no
byte-order field — a `DLTensor` is host order by convention. The two layouts
coincide on a little-endian machine and differ on a big-endian one, so the
conversion has to happen at every `TensorProto` <-> `DLPack` crossing. Code that
`memcpy`s or `reinterpret_cast`s `raw_data` into native scalars is correct only
by accident of running on a little-endian CPU.

## How it is measured

s390x, under `qemu-s390x-static` on an x86_64 host, against an **amd64 control**
running the same commit, the same test suite and the same package versions — so
byte order is the only variable. See `scripts/cross/README.md` for the
procedure; both runs go through `scripts/cross/run_s390x_tests.sh`.

Environment for both runs: CPython 3.12.3, numpy 1.26.4, onnx 1.23.0 (the
vendored version), no onnxruntime (so the reference-evaluator fallback is in
use).

```
              before the fix                     after the fix
little endian  3 failed, 99 passed, 21 skipped    3 failed, 99 passed, 21 skipped
big endian     6 failed, 95 passed, 22 skipped    3 failed, 99 passed, 21 skipped
```

The three remaining failures are the same on both architectures and are
environmental, not byte-order related: `test_simplify_with_unavailable_provider_raises`,
`test_fuse_conv_bn_into_conv` and `test_fuse_convtranspose_bn` all need
onnxruntime, which has no s390x build.

The three that used to fail only on big endian
(`test_simplify_without_onnxruntime`, `test_defer_constant_expand_folds_small_consumer`,
`test_defer_constantofshape_folds_small_consumer`) now pass, as does
`test_profiling.py:134`, which used to *self-skip* with "constant folding did
not run the ONNX Runtime executor here" — hence 21 skips rather than 22.

(These counts are from when the fix above landed, against the full test
suite of the time. The job's Python suite has since been narrowed to the
core-simplify `CORE_TESTS` allowlist described under "Coverage" below, so a
fresh run reports much smaller pass/skip totals — the same regressions this
document describes would still be caught, since every file in that table is
part of the allowlist.)

## What was wrong

### 1. The DLPack bridge refused to run (the actual breakage)

`BuildFromProto` in `onnxsim/dlpack_bridge.h` threw
`"dlpack bridge: only little endian is supported"`. Every constant fold goes
through it. `RunOps` catches per-op failures, logs a warning and moves on —
right for an op the executor genuinely cannot run, but here it turned a total
loss of constant folding into a line on stderr while `simplify()` still returned
`check_ok=True`. The damage cascaded: once one fold was skipped, later folds
referenced initializers that were never produced (`no initializer _v_7`), and
dependent optimizations stopped happening too. Passes like `fuse_mul_into_conv`
emit the new weights as a small subgraph and rely on the folder to evaluate it,
so on big endian they left `Reshape`/`Unsqueeze`/`Mul` nodes behind and an
unscaled `W`.

Worth stating plainly: this was **never silent numerical corruption**. The
big-endian output stayed mathematically equivalent to the input, and `check_n`
(which runs both models and compares) would have caught it otherwise. The bug
was a silent loss of optimization.

The fix converts instead of refusing. `SwapElementBytes` in `dlpack_dtype.h`
reverses each element in place, and the bridge applies it on the way in
(`raw_data` -> host) and on the way out (host -> `raw_data`), guarded by
`if constexpr (kRawDataIsHostOrder)` so a little-endian build is byte-for-byte
what it was — the input direction keeps its zero copy and the output direction
its single copy. On big endian the input direction gives up the zero copy,
which is the cost of being correct there.

### 2. `raw_data` read in host order in `IntTensorToSymTensor`

`onnxsim/onnxsim.cpp` `memcpy`d `raw_data` straight into host `int64_t`/`int32_t`
to seed symbolic shape inference, under a comment claiming the data was read
little-endian. It now decodes byte-wise through `ReadLittleEndian<T>()`.

No test or targeted probe was ever found where this changed onnxsim's output —
ONNX's own shape inference appears to resolve the cases probed before the
symbolic seeds matter. It is fixed because the code was wrong as written, and
because leaving it while fixing (1) would have converted a loud failure into a
quiet wrong answer.

## Not a problem, contrary to an earlier version of this document

`onnx/common/tensor.h`'s `Tensor::data<T>()` does `reinterpret_cast` `raw_data`
in host order, and `ir_pb_converter.cc` copies `raw_data` in without swapping.
An earlier revision of this file inferred from that the optimizer passes
inherit the assumption. **They do not.** Nothing in onnxsim or onnx-optimizer
calls `Tensor::data<T>()` — the passes read tensor values through
onnx-optimizer's `ParseTensorData` (`onnxoptimizer/passes/tensor_util.cc`),
which explicitly byte-swaps on big-endian hosts:

```cpp
/*onnx is little endian serialized always-tweak byte order if needed*/
if (!is_processor_little_endian()) { ... }
```

`Tensor::data<T>()` remains a trap for future code, but it is not on any path
onnxsim uses today.

Separately, and still true: **onnx < 1.16 actively corrupts models on big
endian.** `numpy_helper.to_array()` called `convert_endian(tensor)`, byte-swapping
the proto *in place*, so merely reading an initializer rewrote its `raw_data`
into spec-violating big-endian order. Fixed upstream well before the vendored
1.23 (which swaps into a local instead), but relevant to anyone pairing onnxsim
with an old onnx.

## Coverage

`onnxsim/dlpack_dtype_test.cpp` covers `SwapElementBytes` and asserts that
`kRawDataIsHostOrder` agrees with the CPU it is running on. It is
dependency-free, so it cross-compiles and runs under qemu directly:

```sh
cmake --build .cross-build-s390x/onnxsim-build --target dlpack_dtype_test
qemu-s390x-static -L /rootfs-s390x .cross-build-s390x/onnxsim-build/dlpack_dtype_test
```

The checks assert on bytes rather than decoded scalars, so they mean the same
thing on either architecture. `sym_expr_test`, `model_metrics_test`,
`sym_value_eval_test` and `sym_shape_infer_test` also pass on s390x.

`.github/workflows/big-endian.yml` runs the whole thing: it bootstraps the
rootfs, cross-builds, runs the C++ tests through CTest (with qemu as
`CMAKE_CROSSCOMPILING_EMULATOR`) and then the Python suite. It is weekly, on
demand, and on pull requests touching the harness or the files that read and
write `raw_data` — a full run is ~40 minutes cold, which is too much for every
PR. Widen the `paths` filter to `onnxsim/**` to make it stricter.

The Python suite itself only runs an explicit allowlist of core-simplify test
files (`CORE_TESTS` in `run_s390x_tests.sh`) — the CLI/Python API surface, the
default-on and opt-in graph rewrite/fusion passes, shape inference and
contrib-op schema registration, the function/custom rewriter engine, model
checking/info/memory-planning/backend dispatch, profiling, and the
raw_data-adjacent external-data loading path. It deliberately excludes the
much larger set of quantization-algorithm, pruning-algorithm,
hardware-backend-export/compat and LLM/GGUF-reconstruction test files: those
operate on plain numpy arrays and Python floats and never touch
`TensorProto.raw_data` or the DLPack bridge, so they cannot tell little-endian
and big-endian apart, and running them under qemu would only add emulation
cost and unrelated flakiness without adding byte-order coverage. A new test
for one of the core areas above belongs in `CORE_TESTS`; a new test for an
algorithm/backend/export feature does not need to be added there at all.

## Known unrelated failure in the no-onnxruntime configuration

`test_fuse_conv_bn_into_conv`, `test_fuse_convtranspose_bn` and
`test_fuse_conv_with_bias_bn_into_conv` fail onnxsim's own `check_n`
equivalence check (max diff order ~1-2.4, far past float noise) whenever
onnxruntime is absent and the reference evaluator runs instead. This is **not**
byte-order related — they fail identically on x86_64 under the same conditions,
and the original two failed before the byte-order fixes landed;
`test_fuse_conv_with_bias_bn_into_conv` is a later addition exercising the
same fuse_bn_into_conv pass with a pre-existing Conv bias, and hits the exact
same reference-evaluator disagreement. onnxruntime has no s390x build, so the
big-endian job cannot avoid that configuration; it deselects these by name in
`run_s390x_tests.sh` rather than tolerating failures generally, so a real
big-endian regression still turns the job red.

Worth chasing separately: either BN fusion is subtly wrong and ORT's tolerance
hides it, or the reference evaluator disagrees with ORT on
`BatchNormalization`. Whichever it is, it affects everyone running onnxsim
without onnxruntime, which `pyproject.toml` lists as an optional dependency.
