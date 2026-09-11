# Ruby + cumo integration sample

An integration sample showing onnxsim consumed from **Ruby only** -- no
Python, no Ruby protobuf gem, no `onnx`-alike Ruby gem -- using
[cumo](https://github.com/sonots/cumo) (a GPU/CUDA-backed `NArray`,
API-compatible with [`Numo::NArray`](https://github.com/ruby-numo/numo-narray))
for the tensor side.

It exercises four onnxsim features from Ruby via
[`onnxsim/capi/onnxsim_c_api.h`](../../onnxsim/capi/onnxsim_c_api.h), the same
C ABI the [Rust bindings](../../rust/README.md) use:

1. **Parse** the sample model from ONNX's textual IR syntax
   (`onnxsim_parse_model_text`) -- see "Why the text syntax?" below.
2. **Simplify** it (`onnxsim_simplify_path`) -- constant folding collapses a
   foldable `Add` into a new initializer and drops the node.
3. **Diff** the before/after op counts (`onnxsim_model_info_diff`) -- the same
   report the `onnxsim` CLI prints.
4. **Export** the simplified model to a standalone `.safetensors` archive
   (`onnxsim_export_safetensors`, see [the main README's "Safetensors / GGUF
   archives" section](../../README.md#safetensors--gguf-archives)) and read
   its tensors back with a plain-Ruby reader -- the archive is the standard
   safetensors format (an 8-byte header length, a JSON header, then raw
   bytes), so no protobuf parsing is needed to get at the tensor data.

The folded constant is then loaded into a `Cumo::NArray` and used to run the
simplified graph's one remaining node (`y = x + folded_c`) on the GPU via
cumo. That result is checked against two independent references: a real
ONNX Runtime session (via the
[`onnxruntime`](https://github.com/ankane/onnxruntime-ruby) gem) run on both
the unsimplified *and* the simplified model -- the same "does it still
compute the same result" claim this repo's other backend integrations make
(see [`docs/dlpack-executor.md`](../../docs/dlpack-executor.md)'s TVM/Halide/
nncase/tinygrad tests), now exercised from Ruby with a real ORT.

This ORT is intentionally a separate story from onnxsim_c's own: the
`onnxruntime` gem vendors its own prebuilt ONNX Runtime binary (see
"Prerequisites" below), so running the model needs no native build at all --
only *simplifying* it does, since onnxsim_c embeds ONNX Runtime as its own
constant-folding backend at C++ compile time.

## Layout

| File                       | Role                                                              |
| --------------------------- | ------------------------------------------------------------------ |
| `onnxsim_capi.rb`           | FFI binding to `onnxsim_c`.                                        |
| `safetensors_reader.rb`     | Plain-Ruby `.safetensors` reader (stdlib `json` only).             |
| `build_sample_model.rb`     | The tiny test model this sample simplifies, as ONNX text syntax.   |
| `simplify_and_run.rb`       | The end-to-end sample; run this one.                               |

(No `lib/` subdirectory: the top-level `.gitignore`'s Python-oriented `lib/`
entry would otherwise hide a Ruby `lib/` here too.)

### Why the text syntax?

`build_sample_model.rb` writes the test model in [ONNX's textual IR
syntax](https://onnx.ai/onnx/repo-docs/Syntax.html) -- the same format
`onnx.parser.parse_model` reads in Python, and what this repo's own tests
prefer over `onnx.helper.make_node`/`make_graph`/`make_model` chains (see the
top-level [`CLAUDE.md`](../../CLAUDE.md)) -- rather than building the graph
field by field:

```
<
  ir_version: 8,
  opset_import: ["" : 13]
>
ruby_cumo_sample (float[4] x) => (float[4] y)
<float[4] const_a = {1.0, 2.0, 3.0, 4.0}, float[4] const_b = {10.0, 20.0, 30.0, 30.0}>
{
  folded_c = Add(const_a, const_b)
  y = Add(x, folded_c)
}
```

Ruby has no `onnx.parser` of its own, so this only works because onnxsim's C
API now exposes one: `onnxsim_parse_model_text` (added alongside this
sample) wraps onnx's `OnnxParser::Parse<ModelProto>` -- the exact same parser
Python's `onnx.parser.parse_model` calls, since onnx (and its parser) is
always built regardless of `ONNXSIM_BUILTIN_ORT` (see the top-level
`CLAUDE.md`) -- and hands back a serialized `ModelProto`, no protobuf library
needed on the Ruby side. It's a small, generally useful addition to the
shared C ABI: any binding with no protobuf tooling of its own -- this
sample, but the same call is there for the [Rust bindings](../../rust/README.md)
too, not just wired up on that side yet -- can get the same readable model
construction Python's tests already have. Point `simplify_and_run.rb` at
your own `.onnx` file instead (see "Running" below) if you'd rather simplify
something real.

## Prerequisites

1. **The `onnxsim_c` shared library**, built with the C API enabled (this
   compiles the full onnxsim stack, including ONNX Runtime as a
   constant-folding backend -- see the [top-level `CLAUDE.md`](../../CLAUDE.md)
   for why that's unlike the Python wheel build):

   ```sh
   git submodule update --init --recursive
   cmake -B build -DONNXSIM_C_API=ON -DONNXSIM_BUILTIN_ORT=ON -DONNXSIM_PREBUILT_ORT=ON
   cmake --build build --target onnxsim_c
   ```

   `-DONNXSIM_PREBUILT_ORT=ON` links a released ONNX Runtime build instead of
   compiling it from source (much faster the first time); drop it to build ORT
   from source instead. See [`rust/README.md`'s "Building the native
   library"](../../rust/README.md#building-the-native-library) for the full
   set of options -- the library this sample needs is the same one.

2. **Ruby gems**: `bundle install` (see `Gemfile`) -- `ffi` for the C API
   binding, [`onnxruntime`](https://github.com/ankane/onnxruntime-ruby) for
   the independent reference check, and [`cumo`](https://github.com/sonots/cumo)
   for the GPU-backed `NArray`.

   `onnxruntime` needs nothing extra to install on Linux (x86-64/arm64) or
   Windows -- it vendors a prebuilt ONNX Runtime CPU binary
   (`OnnxRuntime.ffi_lib`, overridable) and "just works". On macOS it needs
   `brew install onnxruntime` (Intel) or nothing (Apple Silicon, also
   vendored); see its README for GPU execution providers
   (`CUDAExecutionProvider`/`CoreMLExecutionProvider`), which need a
   separately-downloaded GPU build pointed at via `OnnxRuntime.ffi_lib =`.
   This is a completely different copy of ONNX Runtime from the one
   `-DONNXSIM_BUILTIN_ORT=ON` links into `onnxsim_c` above -- no relation
   between the two beyond both being ONNX Runtime.

   cumo needs an NVIDIA GPU (Compute Capability 3.5+), CUDA 11.0+, and
   optionally cuDNN 8.0+ to install and run -- it has no CPU-only mode. On a
   machine without one, `simplify_and_run.rb` falls back to
   [`Numo::NArray`](https://github.com/ruby-numo/numo-narray) (`gem install
   numo-narray`), cumo's CPU-only, constructor-for-constructor-compatible
   counterpart, so the whole pipeline (onnxruntime included) can still be
   exercised end to end without GPU hardware -- swap in a CUDA machine with
   `cumo` installed to run the real GPU path with no code changes.

## Running

```sh
ONNXSIM_LIB_DIR=../../build bundle exec ruby simplify_and_run.rb
```

`ONNXSIM_LIB_DIR` (same variable name and `:`-separated-directories
convention the Rust bindings use) points at the directory holding
`libonnxsim_c.so` (`.dylib` on macOS); `ONNXSIM_LIB_PATH` names the library
file itself directly if you'd rather be exact. Expected output looks like:

```
onnxruntime reference (unsimplified model): [111.0, 222.0, 333.0, 434.0]

simplifying /tmp/.../sample_model.onnx -> /tmp/.../sample_model.simplified.onnx

+------------------+----------------+------------------+
|                  | Original Model | Simplified Model |
+------------------+----------------+------------------+
| Add              | 2              | 1 *              |
| Constant         | 2              | 1 *              |
| Model Size       | 180.0B         | 118.0B *         |
| Initializers     | 2              | 1 *              |
| MACs             | 0.0            | 0.0              |
| FLOPs            | 0.0            | 0.0              |
| Memory Access    | 96.0B          | 48.0B *          |
| Memory Footprint | 80.0B          | 48.0B *          |
| Compute Density  | 0.00 FLOP/Byte | 0.00 FLOP/Byte   |
+------------------+----------------+------------------+
onnxruntime result (simplified model): [111.0, 222.0, 333.0, 434.0]
folded initializer "folded_c": dtype=F32 shape=[4]
cumo result (x + folded_c): [111.0, 222.0, 333.0, 434.0]
OK: onnxsim_c, onnxruntime and cumo all agree
```

(Captured from an actual run against a locally built `onnxsim_c` -- with the
`Cumo::NArray`-line coming from the `Numo::NArray` CPU fallback, since that
run had no GPU.)

To simplify your own model instead of the built-in sample, pass its path:

```sh
ONNXSIM_LIB_DIR=../../build bundle exec ruby simplify_and_run.rb /path/to/model.onnx
```

(In that case the `folded_c`/`x`-shape assumptions in `simplify_and_run.rb`
are specific to the built-in sample model -- read through the script before
pointing it at an arbitrary model.)

## Limitations

- The safetensors-to-`Cumo::NArray` dtype mapping
  (`SAFETENSORS_DTYPE_TO_CUMO` in `simplify_and_run.rb`) covers the plain
  integer and `F32`/`F64` float dtypes. `F16`/`BF16` have no native
  `Numo`/`Cumo` element type and are left unmapped.
- The three-way result comparison in `simplify_and_run.rb` uses plain `==`/
  `!=`, which is fine for the built-in sample model's exactly-representable
  float32 values but not a general floating-point comparison; a model with
  results that only agree up to rounding would need a tolerance instead.
