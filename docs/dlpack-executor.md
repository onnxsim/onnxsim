# DLPack tensor exchange at the constant-folding executor boundary

## Why

onnxsim's constant folder repeatedly builds a throwaway sub-model for each
"fold group" and asks a `ModelExecutor` to evaluate it. Historically that
boundary exchanged tensors as `onnx::TensorProto`:

```cpp
// old
virtual std::vector<onnx::TensorProto> _Run(
    const onnx::ModelProto& model,
    const std::vector<onnx::TensorProto>& inputs) const = 0;
```

That representation forces a protobuf materialization on both sides of every
executor call. The worst offender was the output path (`TensorToTensorProto`),
which appended results **element by element** with `add_float_data(dptr[i])`,
reallocating the repeated field as it grew — far more expensive than a memcpy.
The input path copied too (`TensorProtoToTensor` allocated an ORT buffer and
memcpy'd into it, after copying each initializer into the feed vector).

Two goals motivated moving the boundary to DLPack:

1. **Avoid the TensorProto tax** on the hot folding path — borrow buffers in,
   move ORT's own output buffers out.
2. **Embeddability.** onnxsim can be dropped into another ONNX-based compiler or
   runtime stack (a different ORT build, IREE, TVM, a hardware vendor runtime)
   by having that host implement *one* executor callback that speaks a standard
   tensor ABI, without ever touching `onnx::TensorProto` or the vendored ORT.

DLPack (`DLManagedTensor`) is that ABI. It is a small, stable C struct — data
pointer, device, dtype, shape, strides, deleter — usable identically in native
builds, WebAssembly, and across an FFI. Nothing about it is Python-specific.

## The boundary

```cpp
// onnxsim.h
struct ModelExecutor {
  virtual ~ModelExecutor() = default;
  virtual std::vector<DLManagedTensorPtr> Run(
      const onnx::ModelProto& model,
      const std::vector<const DLManagedTensor*>& inputs) const = 0;
};
```

- **Positional, not named.** `inputs[i]` feeds `model.graph().input(i)`; the
  result has one tensor per `model.graph().output()`, in order. DLPack has no
  name field, and `RunOps` already relied on positional order, so nothing is
  lost.
- **`DLManagedTensorPtr`** is `std::unique_ptr<DLManagedTensor,
  DLManagedTensorDeleter>`; releasing it invokes the tensor's own DLPack
  `deleter` exactly once. Ownership is therefore RAII on the C++ side and an
  explicit deleter contract at the C ABI.

### Ownership / lifetime contract

- **Inputs are borrowed** for the duration of the call. The executor must not
  free them or retain them past return. onnxsim keeps the backing buffers
  (initializer `raw_data`) alive across the call.
- **Outputs are freshly owned** by the caller. Each carries a `deleter` that
  releases whatever the producer attached — a borrowed-buffer no-op, an
  `Ort::Value`, or a host allocation.
- All tensors at this boundary are **CPU (`kDLCPU`), contiguous, and
  little-endian**.

### What stays `TensorProto`

The **model** still crosses as a serialized `ModelProto` — ONNX initializers are
inherently protobuf, and the sub-model is tiny. Only the **runtime feeds and
fetches** become DLPack. Likewise, folded results are ultimately baked back into
the model as `raw_data` initializers; that final write is the one unavoidable
copy (`ToTensorProto`, a single `set_raw_data`).

## dtype mapping

`dlpack_dtype.h` maps the stable ONNX dtype wire numbers to `DLDataType` and
back, bijectively, with `lanes == 1`. It depends only on `dlpack.h` (not the
onnx headers), so it is unit-tested standalone (`dlpack_dtype_test`).

| ONNX dtype | DLDataType (code, bits) | bytes |
|------------|--------------------------|-------|
| FLOAT16    | kDLFloat, 16             | 2     |
| FLOAT      | kDLFloat, 32             | 4     |
| DOUBLE     | kDLFloat, 64             | 8     |
| BFLOAT16   | kDLBfloat, 16            | 2     |
| INT8       | kDLInt, 8                | 1     |
| INT16      | kDLInt, 16               | 2     |
| INT32      | kDLInt, 32               | 4     |
| INT64      | kDLInt, 64               | 8     |
| UINT8      | kDLUInt, 8               | 1     |
| UINT16     | kDLUInt, 16              | 2     |
| UINT32     | kDLUInt, 32              | 4     |
| UINT64     | kDLUInt, 64              | 8     |
| BOOL       | kDLBool, 8               | 1     |

Rejected at this boundary (no contiguous little-endian layout the folder
exchanges): STRING, COMPLEX64/128, FLOAT8\*, INT4/UINT4/FLOAT4, UNDEFINED. This
matches — and slightly extends (adds FLOAT16/BFLOAT16/UINT32) — the dtype set the
old TensorProto↔ORT converters handled.

## Adapters

One boundary, several adapters (`dlpack_bridge.h` holds the conversions):

| Adapter | Where | Input | Output | Copies |
|---------|-------|-------|--------|--------|
| `CppModelExecutor` | onnxsim.cpp (built-in ORT) | `BorrowAsOrtValue` wraps the feed buffer via ORT's borrowing `CreateTensor` — **zero copy** | `FromOrtValue` moves ORT's own output buffer into the managed tensor — **zero copy** | none at the boundary |
| `CApiModelExecutor` | capi/onnxsim_c_api.cpp | host receives borrowed `DLManagedTensor*` | host returns owned `DLManagedTensor*`, released via their deleters | host's choice |
| `PyModelExecutor` | cpp2py_export.cc | `ToTensorProto` → bytes | bytes → `FromTensorProtoOwning` | protobuf round trip (see below) |
| `XnnpackModelExecutor` | xnnpack_executor.cpp (`ONNXSIM_BUILTIN_XNNPACK`) | feed pointers passed straight to `xnn_setup_runtime_v2` as `xnn_external_value`s — **zero copy** | XNNPACK writes into an executor-allocated `std::vector<float>` (it has no ORT-style "hand back the session's buffer" mode) — **one copy at the boundary** (allocation, not a memcpy) | one allocation per output |
| `RemoteModelExecutor` | remote_executor.cpp (`ONNXSIM_BUILTIN_REMOTE_EXECUTOR`) | serialized over the dependency-free remote transport | newly owned CPU float32 buffers | one network serialization/copy per input and output |

### Native remote executor

`GetRemoteModelExecutor()` is an opt-in C++ executor. It sends each
constant-folding submodel as a serialized `ModelProto` together with its
contiguous CPU float32 inputs. The endpoint returns output tensors, which are
wrapped as owned DLPack tensors for the normal folding path. The default ORT
executor is unchanged.

Enable it with:

```sh
cmake -S . -B build \
  -DONNXSIM_REMOTE_TRANSPORT=ON \
  -DONNXSIM_BUILTIN_REMOTE_EXECUTOR=ON
```

Then include `onnxsim/remote_executor.h` and pass
`*GetRemoteModelExecutor({"device-host", 39501})` to the C++ simplifier API.
The worker is intentionally a separate process: the same client-side
executor can talk to a native ONNX Runtime worker, an accelerator compiler, or
the optional AXCL worker. The current first implementation supports float32
CPU tensors; adding other dtypes is a protocol/backend extension rather than
an ABI change to `ModelExecutor`.

### XNNPACK: an explicitly-partial backend, not a drop-in ORT replacement

`GetXnnpackModelExecutor()` (`onnxsim/xnnpack_executor.h` declares it, guarded
by `ONNXSIM_HAS_XNNPACK`) runs fold groups through Google's
[XNNPACK](https://github.com/google/XNNPACK) instead of ONNX Runtime, going
through XNNPACK's own graph-level **Subgraph API**
(`xnn_create_subgraph`/`xnn_define_*`/`xnn_create_runtime_v4`) rather than its
per-operator kernel API. `onnxsim/onnx_to_xnnpack_subgraph.{h,cpp}` is the
ONNX-ModelProto-to-`xnn_subgraph_t` lowering; `onnxsim/xnnpack_executor.cpp`
is the `ModelExecutor` adapter that creates a runtime from the result, feeds
it the call's `inputs`, invokes it, and wraps the outputs as
`DLManagedTensor`s.

Unlike `CppModelExecutor`, which delegates to ORT and so handles essentially
any ONNX op, this lowering supports only a small, explicit op set — see
`onnx_to_xnnpack_subgraph.h`'s `kSupportedOps`. `Lower()` throws
`std::runtime_error` naming the reason for anything outside that: an
unsupported op, an unsupported tensor dtype, `Gemm` with `alpha != 1` or
`transA != 0`, a `Reshape` target shape that is itself produced by another
node in the fold group rather than being a constant or a feed, and so on.
Notably absent is `Conv`: XNNPACK's convolution Node is NHWC-native while
ONNX's is NCHW, and getting that layout conversion (activations *and* the
OIHW→OHWI weight transpose) right needs more verification than this first
cut has had — left as follow-up work rather than shipped un-verified.

Most of the op set is fp32 only: `Add`/`Sub`/`Mul`/`Div`, `Relu`, `Sigmoid`,
`Gemm`/`MatMul` (2D operands only), `Reshape` (static target shape only).
`QuantizeLinear`, `DequantizeLinear`, and `QLinearMatMul` additionally
support standard ONNX int8/uint8 quantization, mapped onto XNNPACK's own
quantized datatypes (`qint8`/`quint8`/`qcint8`) — `QuantizeLinear`/
`DequantizeLinear` lower to an `xnn_unary_convert` Node bridging an fp32
Value and a quantized one; `QLinearMatMul` (no float in between; a's/b's/
y's own quantized Values feed `xnn_define_fully_connected` directly, the
same Node Gemm/MatMul use) supports `b` quantized per-tensor or per-column.
Scope is deliberately narrower here too: per-tensor only for
`QuantizeLinear`/`DequantizeLinear` and `QLinearMatMul`'s output, symmetric
(`zero_point == 0`) only for per-channel quantization (XNNPACK's per-channel
datatypes don't have a per-channel zero-point parameter at the API surface
this lowering uses), and no `com.microsoft` contrib ops (`QGemm`,
`QLinearConv`, `QLinearAdd`, ...) — only standard ONNX.

One correctness trap worth flagging for anyone extending this further:
`xnn_define_channelwise_quantized_tensor_value` stores its `scale` array as
a bare pointer (read again whenever a runtime is created from the
subgraph), not a copy — unlike `dims`, which it does copy. A per-channel
scale array therefore needs the same "outlive the subgraph and any runtime
built from it" lifetime as tensor data does (see
`LoweredSubgraph::owned_scale_arrays`, parallel to `owned_tensors`); get
this wrong and the failure mode is not a crash, it's silent all-zero
output, caught only by an actual numeric test
(`xnnpack_executor_test.cpp`'s per-channel `QLinearMatMul` case), not a
build or a null-pointer check.

This makes `GetXnnpackModelExecutor()` an explicitly opt-in *alternative*
executor (pass it to `Simplify` instead of `GetBuiltinModelExecutor()` when
you specifically want XNNPACK — e.g. to test XNNPACK embeddability the same
way `tests/test_tinygrad_integration.py` tests that backend), not a general-purpose replacement: swapping it in for a
model using an unsupported op fails constant folding outright rather than
falling back. `onnxsim/xnnpack_executor_test.cpp` exercises it end to end
(each supported op, a two-node chain, quantize/dequantize round trips,
per-tensor and per-channel `QLinearMatMul`, and the unsupported-op error
path) against independently-computed expected values.

Build it with `-DONNXSIM_BUILTIN_XNNPACK=ON` (see `cmake/build_xnnpack.cmake`,
which `FetchContent`s XNNPACK — pinned by commit, since XNNPACK has no tagged
releases — the same way ORT's from-source build fetches its own `cpuinfo` and
`pthreadpool` dependencies). It composes with `ONNXSIM_BUILTIN_ORT`
independently: a build can have either, both, or neither, and the caller
picks which `GetXxxModelExecutor()` to hand `Simplify`. Not supported for the
Emscripten/WebAssembly build.

The Python adapter still pays a `TensorProto` round trip because the Python side
(onnxruntime's Python API, onnx's reference evaluator) speaks `TensorProto`, not
DLPack. Python is not the zero-copy target; a future dlpack-native Python
executor could bypass it via `__dlpack__`.

### C ABI executor callback (the embeddability seam)

`onnxsim_simplify_with_executor` accepts:

```c
typedef int (*OnnxsimExecuteFn)(
    void* user_data, const void* model_data, size_t model_size,
    const DLManagedTensor* const* inputs, size_t num_inputs,
    DLManagedTensor*** out_outputs, size_t* out_num_outputs);
typedef void (*OnnxsimExecuteFreeFn)(
    void* user_data, DLManagedTensor** outputs, size_t num_outputs);
```

A NULL `execute_fn` falls back to the built-in executor, so the new entry point
is a drop-in superset of `onnxsim_simplify`. The host implements one function in
terms of its own tensors; onnxsim adopts each returned tensor (releasing it via
its DLPack deleter) and then calls `OnnxsimExecuteFreeFn` to release the array
container. The Rust `-sys` crate can add a matching declaration to expose this
(not done in this prototype).

## JsModelExecutor / onnxruntime-web: the separate-heap caveat

`JsModelExecutor` (added in #555, `scripts/convertmodel/js_model_executor.cpp`,
and ported to this boundary here) runs constant folding in onnxruntime-web. This
is the case DLPack helps most — **but with a hard constraint**: onnxsim-wasm and
onnxruntime-web are *separate WASM modules with separate linear memories*. A
`DLManagedTensor.data` pointer minted in onnxsim's heap is meaningless inside
ort-web's heap, so true zero-copy across that boundary is impossible without a
shared `SharedArrayBuffer` memory. Per fold group the realistic path is:

1. onnxsim → JS: `emscripten::typed_memory_view` over onnxsim's heap at the
   tensor's `data` offset → a JS typed-array **view, zero copy on onnxsim's
   side**.
2. JS → ort-web: `new ort.Tensor(dtype, view, dims)` copies into ORT's heap on
   feed — **one copy, unavoidable across modules**.
3. `session.run(...)`.
4. ort-web → JS: read back with `preferredOutputLocation: 'cpu'` — **one copy
   out**.
5. JS → onnxsim: `.set()` into a buffer onnxsim `_malloc`'d and exposed as a
   view — **one copy in**.

So DLPack does **not** remove the memory-domain copies across the ort-web
boundary — nothing can, across two modules. What it removes is the **protobuf
serialize/parse** on every tensor (copy *plus* varint encode/decode). Net
change: `protobuf-encode + 2 domain copies + protobuf-decode` → `2 domain
copies`.

Consequences for the design:

- The **built-in in-wasm `CppModelExecutor` is the zero-copy ideal** (same heap
  → `CreateTensor` borrows directly). `JsModelExecutor` deliberately trades that
  away for what ort-web offers (WebGPU EP, kernels the vendored ORT lacks). The
  same `DLManagedTensor` descriptor serves both; only the deleter differs
  (borrow vs. heap free).
- **WebGPU:** an on-device ort-web output has no portable CPU DLPack pointer.
  First cut: force `preferredOutputLocation: 'cpu'` so outputs cross as
  `kDLCPU`. (`kDLWebGPU` exists in `dlpack.h` but the C++ folder cannot consume
  a GPU buffer.)
- **Batch a whole fold group in one embind crossing** (an array of descriptors),
  not one call per tensor — embind calls have real overhead, and `RunOps`
  already groups nodes per session.

## Files

- `third_party/dlpack/dlpack.h` — vendored DLPack ABI header (include root
  `third_party/`, so `#include "dlpack/dlpack.h"`).
- `onnxsim/dlpack_dtype.h` — pure ONNX↔DLPack dtype mapping (+ `dlpack_dtype_test.cpp`).
- `onnxsim/dlpack_bridge.h` — `TensorProto`↔`DLManagedTensor` and
  `Ort::Value`↔`DLManagedTensor` converters.
- `onnxsim/onnxsim.h` — `ModelExecutor::Run` (DLPack) + `DLManagedTensorPtr`.
- `onnxsim/onnxsim.cpp` — `CppModelExecutor` adapter; `RunOps` bridge; old
  element-wise converters removed.
- `onnxsim/cpp2py_export.cc` — `PyModelExecutor` adapts DLPack ↔ bytes.
- `onnxsim/capi/onnxsim_c_api.{h,cpp}` — `OnnxsimExecuteFn` +
  `onnxsim_simplify_with_executor`.
- `tests/test_tvm_integration.py` (+ `.github/workflows/backend-integration.yml`) —
  regression test for the TVM embeddability claim above: feeds onnxsim's
  simplified output into Apache TVM's Relax ONNX importer and checks it still
  compiles and computes the same result.
- `tests/test_halide_integration.py` (+ `.github/workflows/backend-integration.yml`) —
  the same embeddability claim exercised against
  [Halide](https://halide-lang.org/). Halide has no ready-made ONNX frontend,
  so the test ships a small ONNX-subset-to-Halide lowering and checks that
  onnxsim's simplified output still lowers, compiles, and computes the same
  result as onnx's reference evaluator.
- `tests/test_tinygrad_integration.py` (+ `.github/workflows/backend-integration.yml`) —
  the same embeddability claim exercised against
  [tinygrad](https://github.com/tinygrad/tinygrad)'s own `OnnxRunner`
  (`tinygrad.nn.onnx.OnnxRunner`), which imports and eagerly executes an ONNX
  `ModelProto` directly -- no separate compile step, and no vendor hardware
  needed (it runs on tinygrad's own default backend, typically CPU/CLANG on an
  ordinary CI runner). This feeds onnxsim's simplified
  output straight into the target's native ONNX frontend and checks it still
  runs and computes the same result.

## Rust binding

The Rust crates expose the seam:

- `onnxsim-sys`: the DLPack structs (`DLManagedTensor` et al.), `OnnxsimExecuteFn`
  / `OnnxsimExecuteFreeFn`, and `onnxsim_simplify_with_executor`, mirroring the C
  header one-to-one.
- `onnxsim` (safe): `simplify_with_executor(model, options, |submodel, inputs|
  -> Result<Vec<Tensor>, E>)`. Inputs arrive as borrowed `TensorRef`s (dtype +
  shape + little-endian bytes); outputs are owned `Tensor`s. Trampolines convert
  to/from `DLManagedTensor` with an RAII deleter, guard the FFI boundary with
  `catch_unwind`, and validate output byte lengths. The DLManagedTensor
  construction/teardown round trip is unit-tested and verified leak/UAF-free
  under AddressSanitizer.

The `JsModelExecutor`'s `Run` now takes borrowed `DLManagedTensor*` feeds
(reading dtype/shape/bytes directly, names recovered positionally from
`graph().input()`) and returns owning managed tensors built from the runner's
output bytes via `FromTensorProtoOwning`. See `docs/wasm_ort_web.md` for the
onnxruntime-web build variant and its Asyncify sync-C++/async-JS bridge.

`JsModelExecutor` batches a whole fold group's tensors into a single embind
crossing per direction — one concatenated byte blob plus one flat
`[dtype, ndim, dims...]` metadata array — instead of building a JS object per
tensor. The per-fold-group data copies across the two separate heaps remain
(they are unavoidable), but the number of embind round trips is now O(1) rather
than O(tensors × fields). See `docs/wasm_ort_web.md`.

## Follow-ups

- Optional dlpack-native Python executor via `__dlpack__` to drop the Python
  adapter's protobuf round trip.
