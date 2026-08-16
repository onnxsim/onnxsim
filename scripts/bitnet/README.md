# BitNet → onnxruntime converter

[BitNet b1.58](https://github.com/microsoft/BitNet) is Microsoft's ternary-weight
LLM family: every projection weight is one of `{-s, 0, +s}` for a single
per-tensor scale, and activations are quantized to int8 ("W1.58A8"). The models
are trained that way, so the ternary structure is exact, not an approximation
applied after the fact.

An ONNX export of a BitNet model does **not** preserve any of that. Ternary
weights come out as dense float32 initializers feeding ordinary `MatMul` nodes,
so the graph is 16× larger than the information in it and runs on the generic
float kernel — none of BitNet's advantage survives the export.

This directory converts such an export into onnxruntime's native low-bit
kernels. No custom ops and no BitNet runtime are involved: the output is a plain
`.onnx` file that stock `onnxruntime` executes.

| file | what it is |
| --- | --- |
| `bitnet_ort.py` | the converter — ternary detection and graph rewriting (onnx + numpy only) |
| `bitnet_model.py` | a reference BitNet b1.58 decoder in PyTorch, for producing an export offline |
| `convert_bitnet.py` | the CLI driver: export/load → onnxsim → rewrite → onnxsim → verify |

## Running

```bash
pip install onnx onnxruntime onnxsim
python convert_bitnet.py bitnet.onnx -o bitnet.ort.onnx

# No BitNet checkpoint at hand? Build, export and convert the reference decoder.
pip install torch
python convert_bitnet.py --demo -o demo.onnx --benchmark 30
```

The CLI prints a JSON report: node counts, weight compression, the largest
relative logit error against the float model, and argmax agreement.

## What the conversion does

Weights are detected **structurally** — each `MatMul` initializer is tested for
ternary values — so the converter works on any BitNet export regardless of how it
was produced, and leaves genuinely float layers (the LM head, attention's
activation×activation matmuls) alone.

### `--mode nbits` (default)

Rewrites each ternary `MatMul` to `com.microsoft::MatMulNBits` with `bits=2`.
Ternary maps exactly onto 2-bit codes `{0,1,2}` with zero-point `1`, so the
weight *representation* is lossless — a `--accuracy-level 1` conversion
reproduces the float model to ~1e-7.

Two details decide whether this is fast or catastrophically slow, and both are
easy to get wrong:

* **`accuracy_level=4`** (int8 compute, the default here). onnxruntime's 2-bit
  float compute path is unvectorized; at `accuracy_level=1` it runs ~35× *slower*
  than a plain float `MatMul`. Level 4 hits the MLAS blocked int8 kernel — and is
  also exactly BitNet's own W1.58A8 arithmetic, so it is the faithful choice as
  well as the fast one.
* **`block_size=128`**. onnxruntime accepts 16 and 256 but has no 2-bit int8
  kernel for them, and silently falls back to that same slow path. Auto-selection
  never picks them.

Measured on one `2560×6912` projection (the released 2B-4T FFN shape),
single-threaded, one token, against a float32 `MatMul`:

| block_size | 16 | 32 | 64 | **128** | 256 |
| --- | --- | --- | --- | --- | --- |
| speedup vs float | 0.02× | 2.7× | 5.7× | **8.2×** | 0.03× |

### `--mode int8`

Emits BitNet's W1.58A8 arithmetic explicitly: per-token absmax int8 activation
quantization feeding `MatMulInteger`, rescaled by `a_scale * w_scale`. The matmul
is exact integer arithmetic.

This mode uses **only standard ONNX operators** — no `com.microsoft` domain — so
it targets runtimes without contrib ops. On onnxruntime's CPU provider it is
*slower* than float (the quantization chain is not fused into the integer
kernel), so prefer `nbits` when onnxruntime is the target.

## Measured results

The reference decoder at `--demo-hidden 512 --demo-layers 4` (28 BitLinear
layers), onnxruntime 1.28 CPU, single-threaded:

| mode | weights | model | seq=1 (decode) | seq=64 (prefill) | argmax agreement |
| --- | --- | --- | --- | --- | --- |
| `nbits` | 14.1× smaller | 9.9× smaller | **3.8× faster** | 0.86× | 100% / 98.4% |
| `int8` | 4.0× smaller | 3.6× smaller | 0.16× | 0.18× | 100% / 96.9% |

Both modes quantize activations to int8, so logits move by ~1–2%; the
predictions do not. The speedup is a decode-time win: at one token per step the
projections are memory-bound and 2-bit weights are 16× less memory to move,
while a 64-token prefill is compute-bound and the advantage disappears.

Weight compression is 14.1× rather than the full 16× because MatMulNBits stores
a float32 scale per 128-element block. BitNet's scale is per-tensor, so those
scales are redundant — an ONNX-level cost, not a BitNet one.

On the released `microsoft/bitnet-b1.58-2B-4T` the same arithmetic gives ~2.08 B
ternary parameters: 8.3 GB as float32, ~590 MB after conversion. Note the tied
128256×2560 embedding table (1.3 GB in float32) is left full precision, as in the
released model, and then dominates the file — quantizing it is a separate
concern from BitLinear.

## Why onnxsim is in the pipeline

`convert_bitnet.py` runs `onnxsim.simplify` on both sides of the rewrite.

The first call folds the export's rotary and shape arithmetic — 421 → 192 nodes
on the reference decoder — before any of the low-bit work starts. It can also
decide whether there is anything to convert at all. `F.linear` is
`x @ weight.T`, since torch stores weights as `[out, in]` and ONNX `MatMul` wants
`[in, out]`, so every BitLinear starts as `Transpose(weight) → MatMul`, and
ternary detection needs a plain initializer. torch's own constant folding
normally collapses that transpose, but when an exporter does not
(`do_constant_folding=False`, and some non-torch exporters), **onnxsim is the
difference between 0 convertible layers and all 14** — measured on the reference
decoder, and pinned by `test_simplify_exposes_transposed_weights`.

The second call confirms onnxsim still simplifies a graph containing
contrib-domain `MatMulNBits` without breaking it, and it cleans up the
quantization arithmetic that `--mode int8` introduces.

## Regression test

`tests/test_bitnet.py` is the automated counterpart. It builds the reference
decoder offline (random weights, no checkpoint download), exports it, and pins
the whole pipeline: ternary detection and 2-bit packing round-trip exactly, both
modes run in onnxruntime and preserve the predictions, block-size selection stays
on the kernels that actually have an int8 path, and onnxsim simplifies the
converted graph with its numerical check passing.

`torch` is only needed for the export leg — the converter itself is onnx+numpy —
so the packing and detection cases run even without it. To run the file locally:

```bash
pip install torch onnxruntime
pip install --force-reinstall --no-deps .   # the onnxsim under test
pytest tests/test_bitnet.py -v
```

## Limitations

- Only `MatMul` nodes whose weight is a top-level graph initializer are
  rewritten; weights inside `If`/`Loop` subgraphs are left alone.
- float32 graphs only — a float16 export is not rewritten.
- Packed BitNet checkpoints must be dequantized to `{-s, 0, +s}` float32 before
  export; the converter reads ternary *values*, not BitNet's own packing.
