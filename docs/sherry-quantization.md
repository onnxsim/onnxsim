# Sherry: 1.25-bit 3:4 sparse ternary quantization (`apply_sherry_quantization`)

## What this is

`onnxsim.apply_sherry_quantization` is a **data-free, weight-only** PTQ pass
that rewrites every matched layer's constant float32 weight into Sherry's
**3:4 sparse ternary** format: within every contiguous block of 4 weights,
exactly 3 are non-zero (`±1`) and exactly 1 is zero, with a single scaling
factor `α` per **output channel**.

The rule is the paper's own closed-form greedy solver for
`min ||W - Tα||²` under that 3:4 constraint, which it calls
**"Sparse-AbsMean"**:

1. **Prune** -- in each 4-weight block, zero the element with the smallest
   `|w|`.
2. **Ternarize** -- the surviving three become `sign(w)`.
3. **Scale** -- `α` is the mean `|w|` over that output channel's
   **non-pruned** weights (*not* over all of its weights: averaging the
   pruned zeros in would shrink `α` by ~`3/4` and under-reconstruct every
   surviving weight -- the single easiest way to get this rule wrong).

No calibration data, no search, no training: `W` alone determines the
result, exactly like `onnxsim.apply_gguf_ternary_quantization` (BitNet
b1.58's plain absmean ternary rule) and unlike the Hessian/activation-driven
`onnxsim.apply_gptq`/`onnxsim.apply_awq` family.

```
Before:
  Y = MatMul(X, W)                 # W constant, [K, N], float32

After (identical graph -- only the initializer changes):
  Y = MatMul(X, W_sherry)          # W_sherry constant, [K, N], float32,
                                   # every column holding {-α_col, 0, +α_col}
                                   # with one zero in every 4 elements
```

### Why "3:4" and why 1.25 bits

- A block of 4 is power-of-two SIMD-aligned.
- 25% sparsity stays in a safe margin, rather than the aggressive
  unstructured sparsity that usually costs accuracy.
- The bit arithmetic closes exactly: `C(4, 3) * 2^(3-1) = 16` distinct
  magnitude/sparsity patterns, which plus one sign bit saturate a 5-bit
  index -- **5 bits per 4 weights = 1.25 bits/weight**.

## Where this comes from

[Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained
Sparsification](https://arxiv.org/abs/2601.07892) (Huang, Wu, Hu, Yu, Yang,
Zhu, Liu, Wu), from Tencent's
[AngelSlim](https://github.com/Tencent/AngelSlim) (`sherry` branch,
`Sherry/`). onnxsim ports the *algorithm*, not any framework's code -- the
same rationale as `onnxsim/billm.py`/`onnxsim/gptq.py`.

**What is and is not reproduced here** (see `onnxsim/sherry.py`'s own
docstring for the full statement):

- The quantization rule is **the paper's own published Sparse-AbsMean
  strategy**, not something onnxsim derived or tuned. `tests/test_sherry.py`
  verifies it with real numpy arithmetic against a fully hand-computed
  example -- which element each 4-block prunes, the resulting signs, and
  the exact `α` -- rather than against a recalled constant.
- **Sherry's other half, "Arenas" (Annealing Residual Synapse), is not
  implemented and is out of scope.** Arenas is a QAT mechanism: it adds a
  decaying full-precision residual (`Y = X(Tα) + λ_t(XW)`, with `λ_t`
  annealed to zero by the end of training) to counter gradient
  homogenization and weight trapping *while the network trains*. onnxsim is
  stateless PTQ graph rewriting with no training loop -- see
  `docs/nncf-comparison-future-work.md`'s "Explicitly out of scope" section
  on QAT. **The paper's reported accuracy numbers come from Sherry _with_
  Arenas under QAT**, so this pass should not be expected to reproduce
  them; it implements the quantization scheme only.
- ONNX has no sub-INT4 tensor type and no 5-bit packed format, so -- like
  `onnxsim.apply_gguf_ternary_quantization`,
  `onnxsim.apply_iq4_nl_quantization` and
  `onnxsim.apply_gguf_q4_k_quantization` -- the result is a **float32
  quantize-dequantize round trip folded into a new initializer**, not the
  literal 5-bit packed layout. **The 1.25 bits/weight is a property of the
  format, not of what this pass writes**: the output model is the same size
  as the float32 model it came from. Its value is simulating the format's
  numerics (accuracy studies, downstream tooling that packs the weights
  itself), not compressing the file.

## `α` granularity and weight layout

The paper's scale is per-channel, and "channel" here means what every other
weight-only quantizer in this repo means by it -- the **output channel**.
Weights are normalized to `[N, K]` (output channel first), exactly as
`onnxsim/awq.py`, `onnxsim/pb_llm.py` and `onnxsim/bwa_ptq.py` do:

| Layer | Weight layout | Normalization |
| --- | --- | --- |
| `MatMul(X, W)`, `Gemm(..., transB=0)` | `[K, N]` | `W.T` |
| `Gemm(..., transB=1)` | `[N, K]` | as-is |
| `Conv` | `[N, C_in, kH, kW]` | `reshape(N, -1)` |

The 4-element blocks therefore run **along the reduction axis inside one
output channel row** -- the same "a group runs along K inside one row"
convention `onnxsim.quantize_weight_only_int4` already uses. A consequence
worth stating, because it is what makes the layout handling verifiable: a
`transB=1` `Gemm` and the equivalent untransposed `MatMul` quantize to
*exactly transposed* results, which `tests/test_sherry.py` asserts with
`assert_array_equal`, not a tolerance.

### Ragged rows

A channel row whose length is not a multiple of 4 is zero-padded up to one,
and the padding is made inert -- which takes care, since a min-magnitude
prune and a mean-based scale are each sensitive to it differently:

- A padded zero has the smallest possible magnitude, so it is the slot the
  block's zero lands on. That is deliberate: in the real packed format a
  ragged tail *is* padded out to a whole 4-block and the pad occupies the
  zero slot, so every **real** weight in a ragged final block survives.
  Pruning the smallest *real* element instead would destroy a real weight
  purely because the tensor length is not a multiple of 4 (and would zero a
  1-element tail outright).
- Padded zeros never enter the `α` mean. Unlike a max-based scale (which is
  invariant to them), a mean-based one would be silently diluted -- the same
  care `onnxsim/gguf_ternary_quant.py` takes with its own absmean scale.

`tests/test_sherry.py` checks the result against an independent,
loop-written reference implementation that has no padding at all, for every
row length from 1 to 17.

One deliberate asymmetry: a **real** `0.0` weight is a weight, so it is
either its block's pruned element or a kept element contributing `|0|` to
the mean; a pad is neither. And since `sign(0) = 0`, a kept real zero stays
`0.0` rather than being forced to `+α` (zero is its own best ternary code) --
the only way a block ends up with two zeros.

## Scope

Handled:
- `MatMul(X, W)` / `Gemm(X, W[, B], transA=0, alpha=1, beta=1)` with `W` a
  constant 2-D float32 initializer. `transB` may be 0 or 1.
- `Conv` with a constant float32 weight of any rank >= 2 (opt out with
  `include_conv=False`).
- Any opset: no new nodes are introduced at all, only a replacement
  initializer.

Left untouched (safe no-op, the node passes through as-is):
- Non-constant weights (e.g. a graph input, or a `Cast`ed initializer).
- Non-float32 weights.
- Non-2-D `MatMul`/`Gemm` weights -- a `[B, K, N]` batched weight has no
  unambiguous output channel axis.
- Weight names listed in `skip_names`.

This is an aggressive, lossy quantizer by construction -- 1.25 bits/weight
is the whole point. Expect real accuracy loss relative to the INT4 family;
and note that against an *unconstrained* per-channel absmean ternary
quantizer (free to place its zeros anywhere), the fixed one-zero-per-4
structure is measurably *slightly worse* in pure reconstruction error
(~8% higher MSE in `tests/test_sherry.py`'s own measurement). That is the
price paid for a packable, hardware-friendly layout, not a modelling win --
what it does beat comfortably is a naive whole-tensor global-absmean
ternarization (~28% lower MSE there).

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.apply_sherry_quantization(
    model,
    include_conv=True,        # also quantize Conv weights
    skip_names={"lm_head.weight"},  # e.g. keep the LM head in float32
)
onnx.save(quantized, "model.sherry.onnx")
```

The round-trip transform is also available directly, for one array:

```python
import numpy as np
import onnxsim

# One channel: a single alpha over the whole flattened array.
onnxsim.quantize_dequantize_sherry(np.array([0.5, -2.0, 3.0, 1.0]))
# -> array([0., -2., 2., 2.])   # 0.5 pruned; alpha = (2 + 3 + 1) / 3 = 2
```

`tests/test_sherry.py` covers: the hand-computed example above (both as a
raw array and as a `MatMul` weight's two output channels); exactly one zero
per 4-block with every non-zero equal to `±α` for its channel; `α` being the
kept mean rather than the whole-channel mean; the `Gemm transB=1` /
`MatMul` transpose equivalence; per-filter `Conv` handling and
`include_conv=False`; padding inertness against an independent reference;
`skip_names` and the non-constant/non-float32/non-2-D no-ops; both
reconstruction-quality comparisons above, measured over 20 seeds each; and
an `onnxruntime` round trip producing finite output at a sane relative L2.
