# MXFP4: OCP Microscaling quantization (`quantize_weight_only_mxfp4`)

## What this is

`onnxsim.quantize_weight_only_mxfp4` quantizes a layer's weight block-wise
onto the OCP Microscaling (MX) format's own MXFP4 element type: each
32-element block shares one **power-of-two** scale (no mantissa bits at
all -- the actual definition of "microscaling"), and each element within
the block is a 4-bit floating-point value (E2M1: 1 sign, 2 exponent, 1
mantissa bit), evaluating to one of a small, fixed set of magnitudes:
`{0, 0.5, 1, 1.5, 2, 3, 4, 6}`.

```
Before:
  Y = MatMul(X, W) [+ bias]          -- W constant, [K, N], float32

After:
  Codebook: initializer, float32 [16]     -- E2M1's own 16 fixed values
  Wq: initializer, uint8 [K, N]           -- codebook index per element
  Ws: initializer, float32 [K/32, N]      -- power-of-two scale per block
  Whatever_hat = Reshape(Mul(Reshape(Gather(Codebook, Wq)), Ws), ...)
  Y = MatMul(X, Whatever_hat) [+ bias]
```

## Where this comes from

The [OCP Microscaling (MX) Formats Specification](https://arxiv.org/abs/2310.10537)
(Rouhani et al., 2023; standardized by the Open Compute Project) makes one
specific, narrow choice that every other block-wise scheme in onnxsim
doesn't: the per-block scale must be a **pure power of two**, stored as an
8-bit exponent field with zero mantissa bits ("E8M0"), rather than an
arbitrary float32 fit to the block's data. Every INT4 scheme already in
onnxsim (`quantize_weight_only_int4` and everything built on it) and
`onnxsim.nf4` use an ordinary float32 scale instead. The payoff of MX's
narrower choice: applying a power-of-two scale is an exponent add, not a
real multiply -- the reason MX formats are landing in hardware natively
(NVIDIA Blackwell, AMD CDNA3/MI300, Intel Gaudi3), where a per-block
float32 scale still needs an actual multiplier.

MXFP4 pairs that power-of-two block scale with E2M1 elements. Unlike
`onnxsim.nf4`'s own codebook (16 values fit to a standard normal
distribution's quantile points, chosen because neural network weights are
typically close to normally distributed), MXFP4's codebook is not
data-derived at all -- it is exactly what a 4-bit IEEE-754-style
floating-point format's own bit patterns evaluate to, fixed by the format
definition, identical for every tensor and every block.

ONNX has no native MX tensor type, so -- following the exact same
approach `onnxsim.nf4` already uses for its own non-uniform codebook, since
neither has a standard affine `DequantizeLinear` representation -- this
module builds the dequantization out of ordinary ONNX ops any opset-11+
runtime already supports: `Gather` the per-element code out of a 16-entry
constant codebook, then `Mul` by the per-block power-of-two scale.
Reconstruction is numerically exact to what a real MXFP4 dequantize
produces; only the on-disk *bit* representation (E8M0's packed exponent
byte, E2M1's packed nibble) isn't reproduced -- the same simplification
`onnxsim.nf4` already makes for its own codes.

## Scope

Handled:

- `MatMul(X, W)` with `W` a constant 2-D float32 tensor whose reduction
  dimension `K` is divisible by `block_size`, or `Gemm(X, W)` with
  `transA=0`, `alpha=1`, `beta=1`. `transB` may be 0 or 1.
- Any opset -- unlike onnxsim's affine INT4 schemes, this needs no
  opset-21 `DequantizeLinear` `block_size` attribute, since the codebook
  lookup is built entirely from ordinary `Gather`/`Reshape`/`Mul`/`Cast`
  (opset 11+).
- `skip_names` to leave specific matched weights unquantized.

Left untouched (safe no-op, node passes through as-is):

- Non-constant weights, non-2-D weights, or a reduction dimension not
  divisible by `block_size`.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.quantize_weight_only_mxfp4(model)  # block_size=32, the OCP MX default
onnx.save(quantized, "model.mxfp4.onnx")
```

Needs no calibration data: both the codebook and the per-block
power-of-two scale come entirely from the weight's own values.

## See also

`docs/focus-fp4.md` -- `onnxsim.quantize_weight_only_mxfp4_focus`, a port of
Tencent AngelSlim's FOCUS (CRS/DGS) that emits the *identical* format this
page describes -- same initializer names, shapes, dtypes and the same
bit-identical power-of-two scale -- but picks each block's E2M1 codes with a
quantization scale decoupled from the stored dequantization scale. Only the
code bytes differ, so nothing downstream can tell the two apart.
