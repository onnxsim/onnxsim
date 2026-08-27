# onnx-deploy

A standalone C++ tool (own `CMakeLists.txt`, no dependency on onnxsim's own
build -- see the repo's `CLAUDE.md`) that glues together the multiple `.onnx`
files `optimum-onnx` exports for one model into a single autoregressive
generation pipeline, entirely in C++/Python -- no `optimum`, no `torch`, no
Python required at all for the C++ side.

**Status: built and CI-verified** (`.github/workflows/onnx-deploy.yml`) for
the native CLI, the C ABI, the Python extension, and the WASM module -- see
"Verifying the flow" and the "WASM" section below for exactly what CI proves
and how to reproduce it locally.

## The question this answers

*Is there a C++ library in this repo that glues together the multiple `.onnx`
files `optimum-onnx` exports for one model, so they can be deployed as a
single autoregressive-generation pipeline -- and can it be built as a
dynamic library / WASM module where the actual ONNX Runtime binary is
swapped at runtime instead of baked in at build time?*

Not before this directory. That gluing previously only existed in Python,
via `optimum.onnxruntime`'s `ORTModelForSeq2SeqLM` / `ORTModelForCausalLM`
(see the "Transformers export" section of the top-level `README.md` and
`tests/test_optimum_export_deploy.py`, which drives exactly that Python
class end-to-end against onnxsim's output). onnxsim's own C++ core
(`onnxsim/`) is a graph simplifier -- it takes one `onnx::ModelProto` in and
produces a simplified one out. Its only use of the ONNX Runtime C++ API is
`onnxsim/dlpack_bridge.h`'s single-model, single-call `ModelExecutor::Run()`,
used internally for constant folding during simplification (see
`docs/dlpack-executor.md`) -- it has no notion of multiple persistent
sessions, a KV-cache loop, or a runtime-swappable ONNX Runtime binary.

## What optimum-onnx actually exports

`optimum.exporters.onnx.main_export(..., task="text2text-generation-with-past",
no_post_process=True)` writes a *directory*, not one file:

```
encoder_model.onnx             # seq2seq only (T5, BART, Whisper, ...)
decoder_model.onnx             # first decode step -- no KV-cache inputs
decoder_with_past_model.onnx   # every step after -- KV-cache in AND out
config.json, generation_config.json, tokenizer files, ...
```

Decoder-only causal LMs (GPT-2, Llama, ...) export the same
`decoder_model.onnx` / `decoder_with_past_model.onnx` pair with no
`encoder_model.onnx`.

By default, recent `optimum-onnx` merges the two decoder files into one
`decoder_model_merged.onnx` with a top-level `If` node switching branches on
a boolean `use_cache_branch` input. **This tool deliberately targets the
plain three-file split instead** (`no_post_process=True`), for the same
reason `tests/test_optimum_export_deploy.py` does: as of this writing,
onnxsim simplifying the merged file's `If` branches produces a model that
fails at runtime with an ONNX Runtime broadcast error in cross-attention --
see that test's docstring. Supporting the merged shape later is a matter of
adding one more input (`use_cache_branch`, a length-1 bool tensor) to
`KvCachePipeline::RunDecoderStep` and pointing both "sessions" at the same
`Ort::Session` -- not done here.

## Design

### Layer 1: the KV-cache glue (`include/onnx_deploy/kv_cache_pipeline.h`)

Every decode step after the first, ONNX Runtime hands back cache tensors
named `present.{i}.key` / `present.{i}.value` (plus, for seq2seq models,
`present.{i}.encoder.key` / `.value` for cross-attention alongside
`present.{i}.decoder.key` / `.value` for self-attention). The *next* call
needs those same tensors fed back in as `past_key_values.{i}.key` /
`.value`. `KvCachePipeline::HarvestPresentIntoCache` does that rename by
string substitution alone -- no shapes, dtypes, or layer/head counts are
ever hardcoded, so the same code drives any architecture that follows the
naming convention. Because `Ort::Value` is move-only and owns (or borrows)
its own buffer, the rename is a pointer/ownership move, not a tensor copy.

A subtlety this actually needs to get right (and the toy model below is
specifically built to catch a regression in): some architectures' cache
entries (T5-style cross-attention) are computed once at step 0 and never
re-output by `decoder_with_past_model.onnx` afterward. `RunDecoderStep`
therefore always *borrows* a view of a cache entry for a `Run()` call
(`detail::BorrowView`) rather than moving it out -- an entry that isn't
refreshed by that call's outputs stays owned in the cache, valid for the
next step too.

Two sessions, one host-side loop: encoder once (seq2seq only), then
`decoder_model.onnx` for step 0, then `decoder_with_past_model.onnx` in a
loop with the cache fed back each time, greedy-argmax over `logits` each
step, until `eos_token_id` or `max_new_tokens`.

### Layer 2: the swappable-libort C ABI (`onnx_deploy_c_api.h` / `.cpp`)

`kv_cache_pipeline.h` builds against ONNX Runtime's C++ API with
`ORT_API_MANUAL_INIT` defined, which means the ORT function table
(`Ort::Global<void>::api_`) is **not** resolved at static-init time by a
linked `OrtGetApiBase()` call -- nothing in this header, or in
`onnx_deploy_c_api.cpp`, references any symbol from libonnxruntime at link
time at all. `onnx_deploy_load_ort(libort_path)` resolves the real thing at
*runtime*: `dlopen`/`LoadLibrary` the given `libonnxruntime.so`/`.dylib`/
`.dll`, `dlsym`/`GetProcAddress` its `OrtGetApiBase` export, call
`GetApi(ORT_API_VERSION)`, and hand the resulting `OrtApi*` to
`Ort::InitApi()`. This is ONNX Runtime's own documented mechanism for
"custom operator libraries that are not linked to onnxruntime" (see the
comment above `Ort::InitApi(const OrtApi*)` in `onnxruntime_cxx_api.h`),
applied to the whole pipeline instead of just a custom op.

The payoff: `libonnx_deploy_c.so` builds and links with **zero** dependency
on any specific ONNX Runtime binary -- `ldd` on it shows only libstdc++/
libgcc_s/libc/libm, confirmed in CI (see below). The same compiled artifact
can be pointed at a CPU build, a GPU/EP-specific build, or a newer/older
version, by passing a different path to `onnx_deploy_load_ort` -- no
recompile. (ORT's C API is itself only forward-compatible in the
"newer .so serves an older API version request" direction, not the other
way around -- see "Verifying the flow" below.)

The ABI's shape mirrors onnxsim's own C ABI conventions
(`onnxsim/capi/onnxsim_c_api.h`): every fallible call returns an
`OnnxDeployStatus`, takes a nullable `char** out_error` for a freshly
`malloc`'d message on failure, and no C++ exception ever crosses the
`extern "C"` boundary.

### Layer 3: consumers

- **`src/main.cpp`** (`onnx-deploy` CLI) -- a thin consumer of the C ABI
  above, included specifically so building/running it is also an
  executable smoke test of the ABI itself, not just of the C++ core it
  wraps: `onnx-deploy --libort PATH <export_dir> <id1,id2,...>
  [--max-new-tokens N] [--eos-token-id N] [--decoder-start-token-id N]`. No
  tokenizer -- pipe in ids you got from `AutoTokenizer` separately, the same
  "simplest possible format" choice `tools/onnx-finetune` makes for its raw
  float32 training data.
- **`python/onnx_deploy_py.cc`** -- a compiled [nanobind](https://github.com/wjakob/nanobind)
  extension over the same C ABI (not over `kv_cache_pipeline.h` directly, so
  it has no ONNX Runtime header dependency of its own), following the same
  nanobind convention as onnxsim's own `onnxsim/cpp2py_export.cc`. See "Why
  a Python extension at all" below.
- **`wasm/src/onnx_deploy_wasm.cpp`** -- a from-scratch WASM port of the same
  algorithm (not the C ABI or `kv_cache_pipeline.h` reused verbatim -- see
  "WASM" below for why), reached from JS through one Asyncify-awaited
  `generate()` call, with ONNX Runtime itself swapped in via
  [onnxruntime-web](https://github.com/microsoft/onnxruntime) instead of a
  native `libonnxruntime`.

## Why a Python extension at all, when `optimum.onnxruntime` already does this in Python

`optimum.onnxruntime.ORTModelForSeq2SeqLM`/`ORTModelForCausalLM` is pure
Python calling `onnxruntime`'s Python bindings: the encoder/decoder session
objects, the KV-cache dict, and the `generate()` loop are all live Python
objects and bytecode, importable, `inspect`-able, and monkeypatchable at
runtime, and the `.onnx` files it reads sit on disk as plain files next to
it. A compiled extension like `onnx_deploy_py` moves the loop, the
cache-tensor threading, and the session lifetime entirely into compiled
machine code reachable from Python only through the four calls
`onnx_deploy_py` exports (`load_ort`, `Pipeline`, `.generate`,
`.is_seq2seq`) -- there's no Python-level loop to monkeypatch, no
`ORTModelForSeq2SeqLM.generate` to trace through with `inspect`/`dis`, and
nothing about the decode loop's logic visible to `pip show`/source-reading
the way `optimum`'s own `.py` files are. In that narrow sense, yes: a
compiled extension is more opaque than the pure-Python glue, and if the
goal is specifically "make the *generation loop* harder to casually read or
patch," this is a real step up from `optimum`.

Two things it deliberately does **not** do, worth being clear about before
reaching for it as an "obfuscation" tool:

- **It doesn't protect the model weights themselves.** The `.onnx` files
  (and their external-data weight files) still sit on disk, in plain ONNX
  format, next to the extension -- `onnx_deploy_py.Pipeline("some_dir")`
  loads them exactly the same way `optimum` would. Anyone with the
  directory can open them with `onnx.load()`/Netron regardless of which
  loop code drives inference. If the actual goal is hiding *weights*, that
  needs something this tool doesn't have: e.g. weights baked into the
  extension as encrypted/obfuscated data and decrypted only into
  ONNX Runtime's in-memory buffers at load time -- a meaningfully bigger
  feature, not implemented here.
- **It's not a security boundary against a motivated reverse engineer.**
  Compiled C++ is slower and more annoying to read than Python, not opaque
  to it -- a `.so` full of `Ort::Session`/`std::map<std::string, Ort::Value>`
  calls disassembles and Ghidra/IDA-analyzes just fine, and every ABI call
  is a stable, documented, exported symbol by design (that's what makes it
  usable from Python/Rust/Go/etc. in the first place). Treat this as
  "raises the floor of casual inspection," not "DRM."

So: more flexible than the pure-Python glue for keeping the *loop logic*
out of easily-read/patched Python, genuinely (that's `onnx_deploy_py`,
added here) -- but not a substitute for actually protecting model weights
if that's the real requirement.

## What is deliberately out of scope

- **Tokenization.** `Generate()`/`.generate()` take and return `int64_t`
  token ids, not strings. See e.g.
  [`mlc-ai/tokenizers-cpp`](https://github.com/mlc-ai/tokenizers-cpp) as the
  natural thing to link in front of this.
- **Sampling strategies.** Greedy-only (`ArgmaxLastToken`). Top-k/top-p/
  temperature sampling is a small, independent change to that one function.
- **`config.json` parsing.** Generation parameters are passed in explicitly
  (`GenerationConfig`/CLI flags/Python kwargs) rather than read from
  `generation_config.json`, so there is no JSON dependency anywhere in this
  tool -- everything about tensor shapes and layer count comes from the
  loaded `Ort::Session`s' own input/output metadata.
- **Batch size > 1** and **beam search**.
- **The merged `decoder_model_merged.onnx` shape** (see above).
- **Weight obfuscation/encryption** (see previous section).

## Execution providers (accelerators)

By default every layer runs on CPU (ORT's built-in CPU EP natively, the
`wasm`/CPU backend in onnxruntime-web). Native, native dynamic library, and
WASM also support selecting an accelerator, threaded through the same
swappable-libort design as everything else:

- **Native, CUDA**: `onnx_deploy_create_ex(model_dir, "cuda", cuda_device_id, ...)`
  (`onnx-deploy --execution-provider cuda --cuda-device-id N`; Python
  `Pipeline(model_dir, execution_provider="cuda", cuda_device_id=N)`) calls
  `Ort::SessionOptions::AppendExecutionProvider_CUDA`. This needs the
  `--libort`/`load_ort()` target to actually be a CUDA-enabled ONNX Runtime
  build (the plain CPU release tarballs this README's examples use are not;
  a `-gpu-` release or a from-source `--use_cuda` build is) *and* a
  CUDA-capable GPU/driver present at runtime -- neither is required at
  **build** time by this library, same as the CPU case. If either is
  missing, session creation fails with `ONNX_DEPLOY_ERROR` and ORT's own
  error message (e.g. naming a missing `libonnxruntime_providers_*.so`, or
  a CUDA driver error) -- not a crash.
- **Native, WebGPU**: `onnx_deploy_create_ex(model_dir, "webgpu", 0, ...)`
  (`--execution-provider webgpu`; Python `execution_provider="webgpu"`)
  calls the generic `Ort::SessionOptions::AppendExecutionProvider("WebGPU", {})`
  -- there's no dedicated `AppendExecutionProvider_WebGPU` typed helper the
  way there is for CUDA, and no provider-options keys are documented for it
  in `onnxruntime_c_api.h` as of this writing, so none are exposed beyond
  selecting it by name. This is ORT's own **native** WebGPU EP (built on
  [Dawn](https://dawn.googlesource.com/dawn), running against a real GPU via
  Vulkan/Metal/D3D12) -- a genuinely different implementation from
  onnxruntime-web's browser WebGPU EP the WASM target below uses, despite
  the identical-looking name. As of ORT 1.23.0 (the latest release as of
  this writing), confirmed empirically: the plain prebuilt release tarballs
  (CPU *or* GPU) do **not** include it -- `AppendExecutionProvider("WebGPU", {})`
  throws `"WebGPU execution provider is not supported in this build"`
  against them; a from-source build with `--use_webgpu` is needed. Against
  an even older ORT (1.18.1/1.19.2, which don't yet recognize the name in
  this generic mechanism at all) it throws `"Unknown provider name..."`
  instead -- either way, a clean `ONNX_DEPLOY_ERROR`, not a crash.
- **WASM, WebGPU**: `Module.generate(..., executionProviders)` where
  `executionProviders` is a JS array like `["webgpu"]` (empty/omitted =
  onnxruntime-web's own default). Requires the host to actually expose
  `navigator.gpu` (a real WebGPU-capable browser, or a WebGPU-enabled Node
  build -- plain Node does not have this). If it doesn't,
  `ort.InferenceSession.create` fails and `generate()`'s returned Promise
  rejects cleanly -- see the next paragraph for why that took more than
  "just let the exception propagate."

**What's actually verified vs. just implemented:** this sandbox has no GPU,
no CUDA toolkit, no WebGPU-enabled ORT build, and plain Node has no
`navigator.gpu` -- so none of the three accelerated paths above has been
run against real hardware. What *is* verified in CI
(`.github/workflows/onnx-deploy.yml`, see "Verifying the flow" below): the
CPU path is unaffected (regression-tested), and requesting
`cuda`/native-`webgpu`/wasm-`webgpu` where unavailable fails cleanly with a
clear error rather than crashing -- a real, previously-broken behavior for
the WASM case that testing this (not just implementing it) caught: an
unhandled rejection crossing the WASM Asyncify boundary (see
`CreateSession`/`RunSession` in `wasm/src/onnx_deploy_wasm.cpp`) was
observed to crash the whole process instead of rejecting `generate()`'s
Promise, no matter how carefully it was try/caught on either the JS or C++
side of that specific boundary. The fix:
`Module.onnxDeployCreateSession`/`onnxDeployRunSession` never reject --
they resolve to `{ __onnxDeployError: message }` on failure
(`wasm/test/ort_web_runtime.mjs`), which `onnx_deploy_wasm.cpp` checks for
immediately after every `.await()` and turns into an ordinary,
synchronously-thrown-and-caught C++ exception, entirely outside any
Asyncify unwind. `wasm/test/run_test.mjs`'s WebGPU-unavailable case is
exactly this, and is what actually caught the original crash.

## Building

```sh
# ORT_HOME only needs to point at *some* ONNX Runtime distribution's headers
# (any recent release works -- the ORT C API is stable). The libonnxruntime
# actually run against is chosen separately, at runtime, via --libort /
# onnx_deploy_load_ort() / onnx_deploy_py.load_ort() -- see below.
cmake -B build -DORT_HOME=/path/to/onnxruntime-linux-x64-1.19.2
cmake --build build
```

Add `-DONNX_DEPLOY_PYTHON=ON` (needs `pip install nanobind`) to also build
`onnx_deploy_py` under `build/python/`.

## Verifying the flow

CI (`.github/workflows/onnx-deploy.yml`) does exactly this, from a clean
checkout, on every change under `tools/onnx-deploy/`:

1. Downloads two different real ONNX Runtime releases (1.18.1 and 1.19.2).
2. Configures and builds `onnx_deploy_c`/`onnx-deploy`/`onnx_deploy_py`
   against **only the older release's headers** -- and asserts
   `ldd build/libonnx_deploy_c.so` shows no `libonnxruntime` dependency.
3. Generates a tiny hand-built seq2seq export with
   `scripts/make_toy_seq2seq.py` (`onnx` package only -- no
   torch/transformers/optimum). It is not a real language model: see that
   script's docstring for the exact math, chosen so the correct output
   sequence is fully hand-computable (`compute_expected_ids()`) and so a
   broken KV-cache handoff -- including the "cache entry not re-output every
   step" subtlety above -- changes the output instead of just not crashing.
4. Runs `onnx-deploy` against the toy export with `--libort` pointed at the
   **older** release, asserts the exact expected token sequence.
5. **Swaps `--libort` to the newer release, no rebuild, on the same
   compiled binary**, and asserts the identical expected sequence again --
   this is the actual "swappable at runtime" claim, exercised for real, not
   just a design note.
6. Repeats the same swap test through `onnx_deploy_py` (fresh Python
   process per ORT build).
7. Generates a second toy export, `scripts/make_toy_int8_kv_decoder.py`
   (decoder-only, no encoder): its `past_key_values.0.key`/`present.0.key`
   are INT8 from the start -- the same graph shape
   `onnxsim.quantize_kv_cache` produces (see
   `docs/kv-cache-quantization.md` in the main package) -- and its `Concat`
   genuinely grows the cache every step rather than overwriting a single
   slot, so `logits` at each step is a function of the *whole* cache
   history, not just the latest token. Runs it against both ORT releases
   the same way as steps 4-5, proving `detail::BorrowView`'s INT8 case
   (`kv_cache_pipeline.h`) threads a quantized cache through `Generate()`
   correctly across many steps, not just fp32/int64.
8. Checks that a bad model directory and a bad `--libort` path both fail
   cleanly (`ONNX_DEPLOY_ERROR` / exit 1 with a message), not a crash.
9. Checks that `--execution-provider cuda`, `--execution-provider webgpu`
   (ORT's native EP -- no GPU/CUDA- or WebGPU-enabled ORT build on this
   runner), and an unknown provider name all fail cleanly the same way --
   see "Execution providers" above for what this does and doesn't prove.
   The `wasm-verify` job's `run_test.mjs` run includes the equivalent
   WebGPU-unavailable check for onnxruntime-web's (different) browser
   WebGPU EP.

To reproduce locally: download any two ONNX Runtime releases from
<https://github.com/microsoft/onnxruntime/releases> (`onnxruntime-linux-x64-*.tgz`
et al.), then run the same `cmake`/`make_toy_seq2seq.py`/`onnx-deploy`
sequence the workflow does.

## WASM

This repo's existing WASM/ONNX Runtime bridge (`JsModelExecutor`,
`scripts/convertmodel/js_model_executor.cpp`, see `docs/wasm_ort_web.md`) is
an Asyncify bridge for exactly **one** awaited JS call per `Run()` -- built
for onnxsim's single constant-fold call, not a multi-step decode loop.
`wasm/src/onnx_deploy_wasm.cpp` is a fresh implementation for the
"persistent sessions across many awaited calls" case that bridge doesn't
cover: it holds onnxruntime-web `InferenceSession`s (encoder, decoder,
decoder-with-past) alive JS-side across the whole `generate()` loop, the
same `docs/wasm_ort_web.md` already flags per-call `InferenceSession.create`
as a likely bottleneck for exactly this reason.

**Why a separate implementation instead of reusing `kv_cache_pipeline.h` or
the C ABI:** both are built on `Ort::Value`/`Ort::Session`, ORT's native C++
types -- there is no native ORT at all on the WASM side of this bridge
(every tensor computation happens in JS/onnxruntime-web), so there is
nothing for those types to wrap. `onnx_deploy_wasm.cpp` re-implements the
same two design decisions instead (`present.*` -> `past_key_values.*`
renamed by string substitution alone; a cache entry not re-output by a call
stays valid for the next one) against a plain `WasmTensor` struct
(`{dtype, shape, data}`, `data` always `std::vector<double>`) that crosses
the JS boundary as a plain object.

**"Swappable libort" in WASM terms** means the JS host chooses which
onnxruntime-web build/version/execution-provider implements
`Module.onnxDeployRunSession` -- see `wasm/test/ort_web_runtime.mjs` for the
reference implementation -- mirroring the native side's `dlopen(libort_path)`
swap at the JS/wasm boundary instead of the OS loader. There is zero ORT C/
C++ code or dependency in `wasm/CMakeLists.txt`/`onnx_deploy_wasm.cpp` at
all -- unlike `tools/onnx-finetune/wasm` (which statically compiles ONNX
Runtime from source into its module), this target doesn't link ORT into
wasm in any form.

**Scope note:** this exposes one async entry point, `generate()` (create
sessions, run the whole decode loop, return), rather than a reusable
class wrapping multiple `generate()` calls without re-creating sessions
each time -- enough to prove the persistent-session-across-many-awaited-
calls mechanism works end-to-end (this repo's only prior Asyncify bridge is
single-call), not a full session-lifecycle API. That's a natural follow-up.

Build and test locally (mirrors the `wasm-verify` CI job):

```sh
# Install Emscripten once (any recent version; CI pins the same one
# static.yml uses for onnxsim's own wasm build):
git clone https://github.com/emscripten-core/emsdk.git
./emsdk/emsdk install 5.0.0 && ./emsdk/emsdk activate 5.0.0
source ./emsdk/emsdk_env.sh

cd tools/onnx-deploy/wasm
emcmake cmake -B build && cmake --build build

python3 ../scripts/make_toy_seq2seq.py -o /tmp/toy_seq2seq --vocab-size 7 --encoder-ids 3,4 --decoder-start-token-id 0
cd test && npm install && node run_test.mjs /tmp/toy_seq2seq
# -> generate(max_new_tokens=8, eos=-1): [0, 1, 2, 3, 4, 5, 6, 0]
#    generate(max_new_tokens=20, eos_token_id=6): [0, 1, 2, 3, 4, 5, 6]
#    generate(execution_providers=["wasm"]): [0, 1, 2, 3, 4, 5, 6, 0]
#    generate(execution_providers=["webgpu"]) correctly rejected (no navigator.gpu here): ...
#    OK: onnx_deploy_wasm matches the native pipeline's expected output.
```
