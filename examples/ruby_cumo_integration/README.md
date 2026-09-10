# Ruby + cumo integration sample

An integration sample showing onnxsim consumed from **Ruby only** -- no
Python, no Ruby protobuf gem, no `onnx`-alike Ruby gem -- using
[cumo](https://github.com/sonots/cumo) (a GPU/CUDA-backed `NArray`,
API-compatible with [`Numo::NArray`](https://github.com/ruby-numo/numo-narray))
for the tensor side.

It exercises three onnxsim features from Ruby via
[`onnxsim/capi/onnxsim_c_api.h`](../../onnxsim/capi/onnxsim_c_api.h), the same
C ABI the [Rust bindings](../../rust/README.md) use:

1. **Simplify** a small ONNX model (`onnxsim_simplify_path`) -- constant
   folding collapses a foldable `Add` into a new initializer and drops the
   node.
2. **Diff** the before/after op counts (`onnxsim_model_info_diff`) -- the same
   report the `onnxsim` CLI prints.
3. **Export** the simplified model to a standalone `.safetensors` archive
   (`onnxsim_export_safetensors`, see [the main README's "Safetensors / GGUF
   archives" section](../../README.md#safetensors--gguf-archives)) and read
   its tensors back with a plain-Ruby reader -- the archive is the standard
   safetensors format (an 8-byte header length, a JSON header, then raw
   bytes), so no protobuf parsing is needed to get at the tensor data.

The folded constant is then loaded into a `Cumo::NArray` and used to run the
simplified graph's one remaining node (`y = x + folded_c`) on the GPU via
cumo, checking the result against what the *unsimplified* graph would have
produced.

## Layout

| File                       | Role                                                              |
| --------------------------- | ------------------------------------------------------------------ |
| `onnx_pb_writer.rb`         | Minimal, dependency-free ONNX protobuf writer (see below).         |
| `onnxsim_capi.rb`           | FFI binding to `onnxsim_c`.                                        |
| `safetensors_reader.rb`     | Plain-Ruby `.safetensors` reader (stdlib `json` only).             |
| `build_sample_model.rb`     | Builds the tiny test model this sample simplifies.                 |
| `simplify_and_run.rb`       | The end-to-end sample; run this one.                               |

(No `lib/` subdirectory: the top-level `.gitignore`'s Python-oriented `lib/`
entry would otherwise hide a Ruby `lib/` here too.)

### Why a hand-rolled ONNX writer?

Building even a tiny test model needs *some* way to produce a serialized
`ModelProto`. Rather than reach for Python or a Ruby protobuf gem, `onnx_pb_writer.rb` hand-encodes
the handful of ONNX messages this sample needs (`ModelProto`/`GraphProto`/
`NodeProto`/`TensorProto`/`ValueInfoProto`) directly against protobuf's wire
format (tag + varint / length-delimited value) and onnx.proto3's stable field
numbers -- keeping the whole sample, model included, to Ruby and cumo. Point
`simplify_and_run.rb` at your own `.onnx` file instead (see "Running" below)
if you'd rather simplify something real.

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
   binding, and [`cumo`](https://github.com/sonots/cumo) for the GPU-backed
   `NArray`.

   cumo needs an NVIDIA GPU (Compute Capability 3.5+), CUDA 11.0+, and
   optionally cuDNN 8.0+ to install and run -- it has no CPU-only mode. On a
   machine without one, `simplify_and_run.rb` falls back to
   [`Numo::NArray`](https://github.com/ruby-numo/numo-narray) (`gem install
   numo-narray`), cumo's CPU-only, constructor-for-constructor-compatible
   counterpart, so steps 1-3 above (and the arithmetic in step 4) can still be
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
simplifying /tmp/.../sample_model.onnx -> /tmp/.../sample_model.simplified.onnx

                ┌ The Difference ┐
...op-count / size table...

folded initializer "folded_c": dtype=F32 shape=[4]
cumo result (x + folded_c): [111.0, 222.0, 333.0, 434.0]
OK: matches the unsimplified graph's reference output
```

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
- `onnx_pb_writer.rb` only encodes what `build_sample_model.rb` needs
  (float32 tensors, `Add` nodes, no attributes) -- not a general ONNX writer.
