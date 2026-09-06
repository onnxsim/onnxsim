# NVFP4: NVIDIA two-level FP4 quantization (`quantize_weight_only_nvfp4`)

## What this is

`onnxsim.quantize_weight_only_nvfp4` quantizes a layer's weight block-wise
onto NVIDIA's NVFP4 format: each element is the same 4-bit floating-point
value `onnxsim.mx_quantization`'s MXFP4 already uses (E2M1: 1 sign, 2
exponent, 1 mantissa bit -- `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`), but each
16-element block's scale is an **E4M3** FP8 value (1 sign, 4 exponent, 3
mantissa bits -- not restricted to a power of two the way MXFP4's E8M0
scale is), and the whole tensor additionally shares one FP32
**global scale** that keeps every block's own E4M3 scale inside what E4M3
can represent.

```
Before:
  Y = MatMul(X, W) [+ bias]          -- W constant, [K, N], float32

After:
  Codebook: initializer, float32 [16]     -- E2M1's own 16 fixed values
  Wq: initializer, uint8 [K, N]           -- codebook index per element
  Ws: initializer, float32 [K/16, N]      -- E4M3 block scale * global scale
  Whatever_hat = Reshape(Mul(Reshape(Gather(Codebook, Wq)), Ws), ...)
  Y = MatMul(X, Whatever_hat) [+ bias]
```

## Where this comes from

NVIDIA, ["Pretraining Large Language Models with
NVFP4"](https://arxiv.org/abs/2509.25149) (2025); the same two-level
scaling scheme NVIDIA's own Model-Optimizer implements
(`modelopt/torch/quantization/qtensor/nvfp4_tensor.py`) and Transformer
Engine documents as its `NVFP4` recipe.

Read `docs/mxfp4.md` first. NVFP4 shares MXFP4's element format (E2M1) and
codebook exactly, but makes two different choices for the scale:

- **Block size 16, not 32** -- a finer grain, since the extra scale
  precision below makes a smaller block worthwhile.
- **A 3-bit-mantissa (E4M3) block scale, not a mantissa-less power of
  two.** MXFP4's E8M0 scale can only shrink/grow a block by an exact
  factor of 2, wasting up to half a binade of a block's own dynamic range
  between the codebook's coarse steps and the block's actual magnitude.
  E4M3's extra mantissa bits fit a block's own scale far more tightly, at
  the cost of a real multiply instead of an exponent add.

E4M3 itself still has a bounded range (`448.0` max), so NVFP4 adds a
second, **per-tensor FP32 scale** so the largest block's own scale still
fits inside it:

```
global_scale       = amax(tensor) / (6.0 * 448.0)     # 6.0 = E2M1's own max magnitude
raw_block_scale_i  = amax(block_i) / (global_scale * 6.0)
block_scale_i       = round_to_nearest_e4m3(raw_block_scale_i)
dequant(code, i)    = E2M1_CODEBOOK[code] * block_scale_i * global_scale
```

Since `amax(block_i) <= amax(tensor)`, `raw_block_scale_i` never exceeds
`448.0`, so it never needs clamping before the E4M3 rounding step -- that
rounding is where NVFP4's accuracy actually comes from relative to a
naive float32-scaled block.

ONNX has no native E4M3 tensor type wired into `Gather`-based dequant the
way this module needs, so -- following the exact approach
`onnxsim.mx_quantization` already uses for E8M0 -- this module stores
`block_scale_i * global_scale`, already E4M3-rounded, as one plain float32
value per `(output channel, block)` group, rather than the on-disk E4M3
byte plus a separate FP32 scalar. Reconstruction is numerically identical
to a real two-level NVFP4 dequantize; only the two-field on-disk *bit*
layout isn't reproduced.

## Scope

Handled:

- `MatMul(X, W)` with `W` a constant 2-D float32 tensor whose reduction
  dimension `K` is divisible by `block_size`, or `Gemm(X, W)` with
  `transA=0`, `alpha=1`, `beta=1`. `transB` may be 0 or 1.
- Any opset -- the codebook lookup is built entirely from ordinary
  `Gather`/`Reshape`/`Mul`/`Cast` (opset 11+), no opset-21
  `DequantizeLinear` `block_size` attribute needed.
- `skip_names` to leave specific matched weights unquantized.

Left untouched (safe no-op, node passes through as-is):

- Non-constant weights, non-2-D weights, or a reduction dimension not
  divisible by `block_size`.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.quantize_weight_only_nvfp4(model)  # block_size=16, NVFP4's own default
onnx.save(quantized, "model.nvfp4.onnx")
```

Needs no calibration data: the codebook, the per-block E4M3 scale, and the
per-tensor global scale all come entirely from the weight's own values.

## Relationship to `quantize_weight_only_mxfp4` / `quantize_weight_only_if4`

All three (`mx_quantization`, `if4_quantization`, `nvfp4_quantization`)
quantize onto a 4-bit codebook with a per-block scale via the same
`Gather`/`Reshape`/`Mul` graph shape; they differ only in how the scale
(and, for IF4, the codebook itself) is chosen:

| | block size | scale format | second-level scale |
|---|---|---|---|
| MXFP4 | 32 | E8M0 (power of two) | none |
| NVFP4 | 16 | E4M3 | one FP32 scalar per tensor |
| IF4 | 16 | E8M0 (power of two) | none (adds a per-block *format* choice between E2M1/INT4 instead) |

They compose the same way onnxsim's other weight-only schemes do: each
pass only rewrites the `MatMul`/`Gemm` weights it's given, so running one
after another only affects layers the first one skipped (e.g. via
`skip_names`, or because a different block size didn't evenly divide `K`).
