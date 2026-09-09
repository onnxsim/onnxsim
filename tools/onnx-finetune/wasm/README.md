# onnx-finetune-wasm

Same training loop as `../src/main.cpp` (the native CLI), compiled to
WebAssembly and exposed to JS via Embind instead of argv, so it can run
fine-tuning entirely client-side in a browser tab.

## Status

**Build-verified, including WebGPU (JSEP).** `emcmake cmake` + `cmake --build`
against ORT v1.19.2 with `onnxruntime_ENABLE_TRAINING_APIS=ON` and
`onnxruntime_USE_JSEP=ON` (this file's `ONNX_FINETUNE_WASM_WEBGPU` default)
produces a working `onnx-finetune-wasm.js`/`.wasm` (932 build steps,
~11.5 MB `.wasm`). One upstream bug needed a patch to get there --
`patches/mlas-wasm-ort-enforce.patch` (applies to the ONNX Runtime source
checkout, not this repo): `onnxruntime/core/mlas/lib/q4_dq.cpp` (int4
block-quantization kernels, unrelated to anything this tool uses) calls
`ORT_ENFORCE`/`ORT_THROW` unconditionally, but those aren't defined under
`BUILD_MLAS_NO_ONNXRUNTIME`, which the wasm build always sets. Apply it to
your `onnxruntime` checkout before configuring:

```sh
cd /path/to/onnxruntime
git apply /path/to/onnxsim/tools/onnx-finetune/wasm/patches/mlas-wasm-ort-enforce.patch
```

Traced (and confirmed by the successful build) that the training+wasm CMake
wiring itself needed no changes: with `onnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB=ON`,
`bundle_static_library()` in `onnxruntime_webassembly.cmake` produces a real
linkable `add_library(... STATIC IMPORTED)` target, and
`orttraining/training_api/*.cc` is globbed straight into `onnxruntime_session`
(already one of the bundled libraries) by `onnxruntime_session.cmake` -- so
training symbols end up in the archive with no separate training library
needed on the link line.

Build time (4 cores, `-O3`, ORT + onnx + protobuf + abseil all from source):
25m35s cold, 6m41s warm (95.8% ccache hit rate) *if the build directory name
matches between runs* -- generated headers get included via
`-I<build-dir>/onnxruntime-build`, so a differently-named build directory
(e.g. a fresh temp dir per CI run) busts most of ccache's hash and only
saves ~7%. Reuse the same directory name (wipe contents, keep the path) to
get the real benefit; `CCACHE_BASEDIR` does not fix this specific case since
the differing path segment is the directory name itself, not just its
absolute prefix.

**Runtime: partially exercised, currently flaky.** Loaded the compiled
module in Node (bundled with emsdk, not a browser) and ran a real training
loop against the same toy artifacts the native CLI's example uses. First
attempt: the full 20-epoch loop ran and the loss converged
(5.06 -> ~0.00001, matching the native CLI almost exactly) -- genuine
confirmation the training loop executes correctly in wasm -- but
`exportModel()` then failed with an undecodable raw exception pointer
instead of a catchable JS error. Adding `DISABLE_EXCEPTION_CATCHING=0` to
`onnx-finetune-wasm`'s own link flags looked like the fix (that setting
lives on `onnxruntime_webassembly`'s LINK_FLAGS in
`onnxruntime_webassembly.cmake`, which is silently ignored once that target
is a bundled `STATIC IMPORTED` library rather than the final linked module)
-- but `build.ninja` shows the flag was already present globally via ORT's
own wasm CXX flags both before and after that change, so it wasn't actually
the fix. Two rebuilds since then (with the flag change, then again with
constructor/TrainStep try/catch diagnostics added) both fail *immediately*,
before any training step runs at all -- a regression from the first
successful run, cause not yet identified. Leading hypothesis, unconfirmed:
JSEP/WebGPU EP init probing `navigator.gpu` during session construction,
which doesn't exist in Node (only real browsers) -- would need a JSEP-off
comparison build to confirm (a ~25min rebuild, not a quick check, since
`-DUSE_JSEP=1` is baked into many cached object files). Whether it runs
correctly in an actual browser (`example/`, where `navigator.gpu` is real)
is untested.

## Design

Everything that can avoid touching a virtual filesystem does:
`CheckpointState::LoadCheckpointFromBuffer` and the `std::vector<uint8_t>`
overload of the `TrainingSession` constructor take the four artifact files
directly as bytes, so the JS side just needs `fetch()` + `ArrayBuffer` --
no `FS.writeFile` staging. The one exception is `ExportModelForInferencing`,
which only has a path-based signature upstream in this ORT version; that
single write goes through Emscripten's in-memory MEMFS (never a real disk)
and gets read straight back into a `Uint8Array` to hand to JS.

`FinetuneSession` (Embind class, `src/onnx_finetune_wasm.cpp`) mirrors the
native CLI's loop one-for-one:

| native CLI | wasm binding |
|---|---|
| load artifacts from `--artifacts-dir` | `new Module.FinetuneSession(checkpointBytes, trainingModelBytes, evalModelBytes, optimizerModelBytes)` |
| `--lr` | `session.setLearningRate(lr)` |
| `TrainStep` + loss print | `session.trainStep(inputFloat32Array, targetFloat32Array, batch, inputDim, targetDim)` -> loss |
| `OptimizerStep` | `session.optimizerStep()` |
| `LazyResetGrad` | `session.lazyResetGrad()` |
| `ExportModelForInferencing` + `--output-names` | `session.exportModel(['name1', ...])` -> `Uint8Array` |
| `--teacher-model` (distillation mode) | 5-argument constructor overload: `new Module.FinetuneSession(checkpointBytes, trainingModelBytes, evalModelBytes, optimizerModelBytes, teacherModelBytes)` |
| soft/hard loss breakdown | `session.lastSubLosses()` -> `{soft, hard}` after `trainStep()` (`undefined` outside distillation mode) |

Batch construction, shuffling, and the epoch loop live in JS
(`example/app.js`) rather than C++, same division of responsibility as the
native CLI (C++ owns the training step, the caller owns the data loop).

Distillation mode (see `../README.md`'s "Knowledge distillation" section for
the full CLI-side recipe) needs no change to `trainStep`'s own signature:
`target` still means whatever the training graph's `labels` input expects
(int64 class indices, marshaled as a `Float32Array` of whole numbers -- see
`onnx_finetune_wasm.cpp`'s `trainStep` for why: Embind typed-array bindings
don't make mixed-dtype-per-call convenient, so the float->int64 conversion
happens in C++ instead of asking every caller to build an `Int32Array`/
`BigInt64Array` themselves). `FinetuneSession` internally runs a second,
plain inference session against the teacher on each step's `input` to
produce `teacher_logits` and feeds all three tensors to `TrainStep`, exactly
mirroring `main.cpp`'s `--teacher-model` handling.

## Building

Requires `emcmake`/`emcc` (from emsdk) and a host `protoc` binary (protobuf's
codegen runs on the host during a cross build, same requirement as
`../../build_wasm.sh` has for onnxsim itself):

```sh
# from an onnxruntime source checkout with training support (same one used
# for the native build, see ../README.md)
PROTOC=$(which protoc)  # or build one, see ../../build_wasm.sh

emcmake cmake -B build \
  -DORT_SOURCE_DIR=/path/to/onnxruntime \
  -DONNX_CUSTOM_PROTOC_EXECUTABLE=$PROTOC \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -t onnx-finetune-wasm
```

Output: `build/onnx-finetune-wasm.js` + `build/onnx-finetune-wasm.wasm`.
Copy both into `example/` (or adjust the `<script src=...>` path in
`example/index.html`) to run the demo.

## Trying the example

Generate the same toy artifacts the native CLI's example uses, alongside
`example/`:

```sh
cd example
python3 ../../scripts/make_toy_model.py -o toy_model.onnx
python3 ../../scripts/make_synthetic_data.py --num-samples 2048
PYTHONPATH=/path/to/onnxruntime/build/Release/build/lib \
  python3 ../../scripts/generate_artifacts.py toy_model.onnx -o artifacts
```

Then serve `example/` with any static file server (needs no special
headers -- this build has wasm threads off) and open `index.html`. It
should show the same loss curve as the native CLI's toy example (roughly
5 -> under 0.001 over 20 epochs) and offer a `finetuned.onnx` download at
the end.

## Memory

wasm32's linear memory is capped at 4 GiB (see the `MAXIMUM_MEMORY` comment
in `../../CMakeLists.txt` for onnxsim's own reasoning on this same limit).
Measurements against the native build earlier in this project's history put
full AdamW fine-tuning at roughly 4x a model's raw fp32 weight size in RSS --
so this comfortably covers small-to-medium models (a few hundred million
parameters) but rules out anything approaching billion-parameter scale
in-browser.
