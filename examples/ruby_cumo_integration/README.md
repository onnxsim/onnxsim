# Ruby + cumo integration sample

An integration sample showing onnxsim consumed from **Ruby only** -- no
Python, no Ruby protobuf gem, no `onnx`-alike Ruby gem -- using
[cumo](https://github.com/sonots/cumo) (a GPU/CUDA-backed `NArray`,
API-compatible with [`Numo::NArray`](https://github.com/ruby-numo/numo-narray))
for the tensor side. Two runnable scripts:

- **`simplify_and_run.rb`** -- simplify a small hand-written model and
  cross-check it against a real ONNX Runtime session and cumo (see "Simplify
  example" below).
- **`train_and_export.rb`** -- train a tiny model with cumo doing the
  gradient descent, then export and simplify it (see "Training example"
  below).

## Layout

| File                       | Role                                                              |
| --------------------------- | ------------------------------------------------------------------ |
| `onnxsim_capi.rb`           | FFI binding to `onnxsim_c`.                                        |
| `safetensors_reader.rb`     | Plain-Ruby `.safetensors` reader (stdlib `json` only).             |
| `cumo_compat.rb`            | Requires cumo, falling back to Numo with no GPU (see below).       |
| `build_sample_model.rb`     | The tiny test model `simplify_and_run.rb` simplifies, as ONNX text.|
| `simplify_and_run.rb`       | The simplify example; run this one.                                |
| `train_and_export.rb`       | The training example; run this one.                                |

(No `lib/` subdirectory: the top-level `.gitignore`'s Python-oriented `lib/`
entry would otherwise hide a Ruby `lib/` here too.)

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
   set of options -- the library both scripts need is the same one.

2. **Ruby gems** (see `Gemfile`): `ffi` for the C API binding,
   [`onnxruntime`](https://github.com/ankane/onnxruntime-ruby) for the
   independent reference check, [`numo-narray`](https://github.com/ruby-numo/numo-narray)
   as cumo's CPU fallback, and [`cumo`](https://github.com/sonots/cumo)
   itself (in its own `:cuda` Bundler group, since it needs a GPU just to
   install -- see below).

   On a machine with an NVIDIA GPU, CUDA 11.0+ and optionally cuDNN 8.0+
   (what cumo itself needs -- it has no CPU-only mode):

   ```sh
   bundle install
   ```

   On a machine without one -- including CI -- skip the `:cuda` group so
   `bundle install` doesn't fail trying to compile cumo's native extension:

   ```sh
   bundle config set --local without cuda
   bundle install
   ```

   Either way `cumo_compat.rb` picks the right one at runtime: `Cumo` if the
   gem installed and a GPU is actually there, `Numo::NArray` (cumo's
   constructor-for-constructor-compatible counterpart) otherwise -- so both
   scripts (onnxruntime included) run end to end either way, with no code
   changes.

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

## Simplify example

`simplify_and_run.rb` exercises four onnxsim features from Ruby via
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
`onnxruntime` gem vendors its own prebuilt ONNX Runtime binary, so running
the model needs no native build at all -- only *simplifying* it does, since
onnxsim_c embeds ONNX Runtime as its own constant-folding backend at C++
compile time.

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
construction Python's tests already have. Both scripts use it:
`train_and_export.rb` (below) interpolates its *learned* weights into the
same syntax. Point `simplify_and_run.rb` at your own `.onnx` file instead
(see "Running" below) if you'd rather simplify something real.

### Running

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
`cumo result` line coming from the `Numo::NArray` CPU fallback, since that
run had no GPU.)

To simplify your own model instead of the built-in sample, pass its path:

```sh
ONNXSIM_LIB_DIR=../../build bundle exec ruby simplify_and_run.rb /path/to/model.onnx
```

(In that case the `folded_c`/`x`-shape assumptions in `simplify_and_run.rb`
are specific to the built-in sample model -- read through the script before
pointing it at an arbitrary model.)

## Training example

`train_and_export.rb` trains a one-input linear regression, `y = w*x + b`,
against noise-free synthetic data (`y = 3*x + 2`) by hand-rolled gradient
descent -- forward pass, MSE loss gradient, weight update, all as a handful
of elementwise `Cumo::NArray` ops run every epoch. This is plain cumo code;
onnxsim has no autodiff/training-graph feature of its own to drive from
Ruby (unlike, say, the [dlpack-executor embeddability
seam](../../docs/dlpack-executor.md)) -- it only enters once training is
done and there is a model to build, simplify, and export:

1. Interpolate the learned `w`/`b` into the same ONNX text syntax
   `simplify_and_run.rb` uses (see "Why the text syntax?" above) and parse it
   via `onnxsim_parse_model_text`.
2. Simplify it (`onnxsim_simplify_path`) and print the before/after report
   (`onnxsim_model_info_diff`).
3. Run the simplified, trained model through a real ONNX Runtime session and
   check it against cumo's own forward pass with the same learned weights --
   the same cross-engine check `simplify_and_run.rb` makes, now on a model
   this script trained itself.

### Running

```sh
ONNXSIM_LIB_DIR=../../build bundle exec ruby train_and_export.rb
```

Writes the trained, simplified model to `trained_linear.onnx` in this
directory by default (pass a path as the first argument to write elsewhere;
it's `.gitignore`d here as a run artifact, not a fixture). Expected output:

```
epoch 0: loss=1421.5 w=0.4515 b=0.0335
epoch 10000: loss=0.00723 w=3.0129 b=1.8234
epoch 20000: loss=7.2e-05 w=3.0013 b=1.9824
epoch 30000: loss=1.0e-06 w=3.0001 b=1.9982
trained on cumo: w=3.000013 (true 3.0), b=1.999825 (true 2.0)

+------------------+----------------+------------------+
|                  | Original Model | Simplified Model |
+------------------+----------------+------------------+
| Add              | 1              | 1                |
| Constant         | 2              | 2                |
| Mul              | 1              | 1                |
...
+------------------+----------------+------------------+
onnxruntime prediction (simplified, trained model): [4.9998, 7.9999, ...]
cumo prediction:                                     [4.9998, 7.9999, ...]
OK: onnxruntime and cumo agree on the trained model (max diff 0.0)
wrote /path/to/trained_linear.onnx (832 bytes)
```

(Also an actual run, `Numo::NArray` fallback again -- the op-count table is
mostly unchanged here since this model has nothing left to constant-fold;
what it does show is onnxsim's own metadata/encoding normalization, which is
why "Model Size" isn't marked `*` the way `simplify_and_run.rb`'s reclaimed
bytes are.)

## CI

[`.github/workflows/ruby-cumo-integration.yml`](../../.github/workflows/ruby-cumo-integration.yml)
builds `onnxsim_c` and runs both scripts on every PR that touches this
directory or `onnxsim/capi/onnxsim_c_api.{h,cpp}`, plus a weekly schedule
(catches breakage from a new `onnxruntime`/`cumo`/`numo-narray` gem release)
and `workflow_dispatch`. The runner has no GPU, so it installs gems with
`bundle config set --local without cuda` -- exactly the `Numo::NArray`
fallback path described above, not a lighter stand-in for it.

## Limitations

- The safetensors-to-`Cumo::NArray` dtype mapping
  (`SAFETENSORS_DTYPE_TO_CUMO` in `simplify_and_run.rb`) covers the plain
  integer and `F32`/`F64` float dtypes. `F16`/`BF16` have no native
  `Numo`/`Cumo` element type and are left unmapped.
- `simplify_and_run.rb`'s three-way result comparison uses plain `==`/`!=`,
  which is fine for the built-in sample model's exactly-representable
  float32 values but not a general floating-point comparison.
  `train_and_export.rb` needs (and uses) an actual tolerance instead, since
  its weights are *learned*, not exactly-representable literals -- ORT's and
  cumo's float32 `Mul`+`Add` can differ by a ULP or two.
- `train_and_export.rb`'s gradient descent is deliberately the simplest
  possible case (one feature, no bias term tricks, fixed learning rate/epoch
  count tuned for this exact synthetic dataset) -- a starting point to adapt,
  not a general training loop.
