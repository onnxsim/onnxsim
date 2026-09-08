# RKNN3 (Rockchip next-gen) LLM integration check

Verifies that `onnxsim`'s output still converts and runs through
`rknn.api.RKNN.load_llm()` -- **RKNN3-Toolkit**'s dedicated LLM/VLM
conversion entry point, from
[`airockchip/rknn3-toolkit`](https://github.com/airockchip/rknn3-toolkit),
Rockchip's *next-generation* SDK for the RK1820/RK1828/RK3572 NPU line.

## Not the same package as `scripts/rknn`

RKNN3-Toolkit is a **separate, incompatible** package from `rknn-toolkit2`
(the one `scripts/rknn` covers) -- the repo's own README says so explicitly
("RKNN3-Toolkit is not compatible with RKNN-Toolkit and RKNN-Toolkit2") --
despite both exposing the identical `from rknn.api import RKNN` import path.
`rknn3_backend.py` checks `hasattr(RKNN, "load_llm")` specifically so this
harness fails loudly rather than silently running against the wrong package
if both ever end up importable in the same environment.

## Two LLM stacks, only one of which touches ONNX

Rockchip actually ships two separate ways to put an LLM on one of its NPUs:

1. **RKLLM** (`airockchip/rknn-llm`, for RK3588/RK3576/RK3562/RV1126B --
   the older, more mature stack): `rkllm.api.RKLLM.load_huggingface()` /
   `load_gguf()` load straight from a Hugging Face checkpoint or GGUF file.
   **No ONNX anywhere in this path** (`rkllm-toolkit`'s own
   `requirements.txt` doesn't even list the `onnx` package) -- onnxsim has
   no hook here.
2. **RKNN3-Toolkit's `load_llm()`** (this harness): takes an **ONNX file**
   as its `model` argument, plus a `.config.pkl` sidecar -- confirmed
   directly against `rknn3-toolkit/examples/qwen2_5/test.py`, the real
   upstream example. So, unlike RKLLM, onnxsim genuinely sits in this
   pipeline the same way it already sits ahead of `rknn-toolkit2`/`rknn-llm`
   for CV models: `scripts/rknn3/llm_export.py` produces that ONNX (a
   trimmed, verified port of `airockchip/rknn3-model-zoo`'s own
   `py_utils/export_llm_helper.py` reference implementation), and this
   harness runs onnxsim's `simplify()` on it before handing it to
   `load_llm()`.

## Real vs. static, and the fidelity tier

Like `scripts/rknn` (and unlike `scripts/axera`'s static op-list heuristic --
Pulsar2 has no pip package at all), this runs the **real** RKNN3-Toolkit
converter and its **PC simulator** (`init_runtime()` with no `target=`
argument) -- verified directly: `load_llm()` + `build(do_quantization=False)`
+ the simulator ran a real, tiny (`hidden_size=8`, 2 layers)
`Qwen2ForCausalLM`-architecture checkpoint
([`yujiepan/qwen2.5-tiny-random`](https://huggingface.co/yujiepan/qwen2.5-tiny-random))
end to end and produced real `(1, 1, vocab_size)` logits. As with
`scripts/rknn`, that's a *functional* simulator, not real RK1820/RK1828/
RK3572 silicon -- no NPU coprocessor board or `rknn3_transfer_proxy` link was
used anywhere in this harness. It also always computes in `float16`, not
`float32` (`rknn.config()`'s `float_dtype` parameter only accepts
`'float16'`), and always quantizes the LM head to an integer dtype
regardless of `do_quantization` -- see "Four real, verified findings" below
for what that means for this harness's comparison tolerance.

## Four real, verified findings

All four reproduced directly while building this harness, on a plain
x86-64 Linux host with `rknn-toolkit==1.1.0` (the RKNN3 one) and no RK
device:

1. **`load_llm()` strips the embedding lookup.** Even though the exported
   ONNX declares `input_ids` (int64 token ids) as its first input,
   `load_llm()`'s own log says so explicitly: *"The gather index 'input_ids'
   is from model input, but 'auto' found in vocab, treat it as embedding!"*
   -- `rknn.inference()` then expects precomputed float `input_embeds`
   instead, looked up host-side from a separately exported `.embed.bin`
   float16 weight file. `rknn3_backend.run()` does this lookup itself.
2. **The upstream ONNX-export reference code breaks under `torch>=2.9`.**
   `rknn3-model-zoo`'s `causal_llm_to_onnx` calls plain
   `torch.onnx.export(..., dynamic_axes=...)` with no `dynamo=` argument --
   correct for the legacy TorchScript exporter that used to be the default.
   Reproduced against `torch==2.14.0`: that call now raises deep inside
   `torch/onnx/_internal/exporter/_dynamic_shapes.py`
   (`ValueError: treespec.unflatten(leaves): ...`), because PyTorch's new
   `torch.export`-based exporter is the default starting in 2.9 and does not
   accept a plain `dynamic_axes` dict the legacy exporter did.
   `llm_export.export_causal_lm_to_onnx()` passes `dynamo=False` explicitly.
3. **`scripts/common` collides with RKNN3-Toolkit's own internal `common`
   package.** Every other `scripts/<vendor>` harness in this repo imports
   `scripts/common/ep_numerics.py` via `sys.path.insert(0, scripts_dir)` +
   `from common.ep_numerics import compare` -- reproduced directly: doing
   that anywhere before `rknn.load_llm()` runs makes it fail with
   `ModuleNotFoundError: No module named 'common.rknpu_profiler'`, because
   RKNN3-Toolkit's C-extension code does its own bare `import common...`
   internally, and Python resolves the name to the already-cached (and
   already fully-imported) onnxsim package instead once anything has put
   `scripts/` on `sys.path`, even transiently. `rknn3_backend.py` loads
   `ep_numerics.py` by file path (`importlib.util.spec_from_file_location`)
   instead, never registering anything under the bare name `common` in
   `sys.modules` -- see `rknn3_backend._load_module()`'s docstring.
4. **The PC simulator's `float16`-only compute path makes a legitimate,
   large simplification land one rounding step off -- not an onnxsim bug.**
   onnxsim's own `simplify()` shrinks this checkpoint's export from 702 to
   266 nodes (constant-folding `Shape`/`Gather`/`Concat` chains and the
   like) and its own correctness check (a separate, FP32 comparison)
   passes. But `rknn.config()`'s `float_dtype` only ever accepts
   `'float16'` (there is no `float32` option -- confirmed reading
   `rknn.api.rknn`'s source), and comparing the original vs. simplified
   graph's prefill logits through `load_llm()` + the PC simulator gave a
   small, exactly reproducible `max_abs_diff` of `0.015625` = `2**-6` on
   logits up to magnitude ~18 -- precisely one `float16` ULP at that
   magnitude. `rknn3_backend.compare_logits()`'s tolerance
   (`rtol=3e-2, atol=5e-2`, looser than `scripts/rknn`'s CNN-tuned
   `rtol=1e-2, atol=1e-3`) is set with margin above that single-ULP noise
   floor, not loosened to mask a real regression -- see its docstring.
   Also tried and found *not* the cause: `load_llm()` separately always
   quantizes the LM head to `w6a16` (6-bit weights) regardless of
   `build(do_quantization=False)` -- `rknn3_backend._llm_config()`
   overrides that to the least-lossy `w16a16`, a reasonable precision
   improvement on general principle, but the measured diff was bit-for-bit
   identical with and without it, so the head's own quantization is not
   what this specific divergence traces to.

## What it checks

Same original-vs-simplified framing as `scripts/rknn`, for the one fixed
checkpoint:

1. Export the checkpoint to ONNX + `.config.pkl` + `.embed.bin`.
2. `simplify` the ONNX with onnxsim.
3. Convert (`load_llm` + `build(do_quantization=False)`) and run
   (`init_runtime()` + a one-token prefill `inference()`) the **original**
   ONNX. If that already fails, RKNN3-Toolkit doesn't support this graph ->
   `unsupported`, **not** a failure.
4. Convert and run the **simplified** ONNX the same way (reusing the same
   `.config.pkl`/`.embed.bin` -- neither depends on the ONNX graph itself).
   If the original worked but the simplified doesn't -> `rknn3_regression`.
5. Compare the two prefill logits. Divergence beyond tolerance ->
   `rknn3_regression`.

`do_quantization=False` throughout: this checks graph-compile and
single-token-prefill numerics, not GRQ/W4A16 quantization accuracy (a
separate, calibration-dataset-dependent concern).

## Files

| file | purpose |
| --- | --- |
| `rknn3_backend.py` | wraps `rknn.api.RKNN.load_llm()`: config/build/init_runtime/inference against the PC simulator, the embedding-lookup workaround, and the `sys.modules["common"]` collision fix. Degrades gracefully (`RKNN3_AVAILABLE`) when RKNN3-Toolkit is absent or is actually `rknn-toolkit2` under the same import path. |
| `llm_export.py` | HF checkpoint -> ONNX + `.config.pkl` + `.embed.bin`, a trimmed and `torch>=2.9`-fixed port of `rknn3-model-zoo`'s reference export code. |
| `worker.py` | runs the check for the one fixed checkpoint in an isolated subprocess (the converter can abort at the C-extension level), printing one `__RESULT__<json>` line. |
| `run_rknn3_compat.py` | drives the check, writes a CSV, and exits non-zero on a regression. Entry point for CI. |

## Running locally

Requires an x86-64 Linux host and RKNN3-Toolkit's wheel, which is **not on
PyPI** -- download it from
[`airockchip/rknn3-toolkit`'s `rknn3-toolkit/packages/`](https://github.com/airockchip/rknn3-toolkit/tree/main/rknn3-toolkit/packages)
(cp310/cp312 Linux/macOS wheels; no cp311 wheel is published, unlike
`rknn-toolkit2`'s cp310/cp311/cp312 set):

```bash
pip install -r requirements_cp312-1.1.0.txt   # from that packages/ directory
pip install rknn3_toolkit-1.1.0-cp312-cp312-manylinux2014_x86_64.whl
pip install torch transformers jinja2 accelerate   # the ONNX-export side
pip install .                                       # or an onnxsim wheel

python scripts/rknn3/run_rknn3_compat.py --output rknn3-compat.csv
```

The in-tree smoke test `tests/test_rknn3_compat.py` reuses this harness and
is skipped automatically when RKNN3-Toolkit isn't installed.

## Extending

This harness intentionally covers only the plain, single-segment causal-LM
export path (`llm_export.py` is a trimmed port -- see its docstring for what
it leaves out: Qwen3.5's segment-wise export, the Qwen3-ASR audio-embedding
variant, and VLM vision-tower export, all of which `rknn3-model-zoo` also
covers but this harness's `Qwen2ForCausalLM` checkpoint doesn't exercise).
