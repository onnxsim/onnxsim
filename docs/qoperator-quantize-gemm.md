# QOperator general Gemm quantization (`quantize_qoperator_gemm`)

## What this is

`onnxsim.quantize_qoperator_gemm` is a self-contained C++ graph rewrite
(`onnxsim/passes/qoperator_quantize_gemm.h`) that statically
(calibration-based) quantizes a `Gemm` node whose weight `B` is a constant
2-D float32 tensor into ONNX Runtime's **`com.microsoft`** contrib op
`QGemm` -- the fully-general analogue of `quantize_qoperator`'s
`QLinearMatMul` rewrite, which only handles "vanilla" Gemm (`transA=0`,
`alpha=1`) because `QLinearMatMul` (a standard ONNX op) has no
transpose/scale attributes of its own. `QGemm` keeps `transA`/`transB`/
`alpha` as attributes, so this pass handles **any** `transA`, `transB`, or
`alpha` value `quantize_qoperator` cannot.

```
Before:
  Y = Gemm(A, B, C, transA=ta, transB=tb, alpha=al, beta=1.0)
      -- A: runtime float32; B: constant 2-D float32; C: constant 1-D
         float32 of length N (optional)

After:
  Aq = QuantizeLinear(A, As, Azp)                       -- As/Azp: CALIBRATED
  Yq = QGemm(Aq, As, Azp, Bq, Bs, Bzp, Cq, Ys, Yzp,
             transA=ta, transB=tb, alpha=al)            -- true int8 compute
  Y  = DequantizeLinear(Yq, Ys, Yzp)                    -- Ys/Yzp: CALIBRATED
```

## Weight and bias quantization

`B` is quantized per output channel (INT8, symmetric) **in its own storage
layout** -- since `QGemm` keeps `transB` as its own attribute, there is no
need to physically transpose `B` into a fixed `[K, N]` layout first, unlike
`QLinearMatMul`'s rewrite.

`C` (when present) is quantized ahead of time into INT32 with zero point 0
and a per-column scale of `alpha * a_scale * b_scale[n]` -- `QGemm`'s own
documented bias convention -- and accumulated directly in the quantized
compute. This is a meaningful difference from `quantize_qoperator`'s
vanilla-Gemm handling, which has no bias input on `QLinearMatMul` at all and
so adds the bias back in float *after* dequantizing.

## Scope

Handled:
- A `Gemm` node whose weight `B` is a constant 2-D float32 tensor, with any
  `transA`/`transB`/`alpha` value.
- A bias `C`, if present, that is a constant 1-D float32 tensor of exactly
  `N` elements (the common per-column-bias case) with `beta == 1`.

Left untouched (safe no-op, node passes through as-is):
- A non-constant, non-float32, or non-2-D `B`.
- A `C` of any other shape (e.g. 2-D, or a scalar) -- outside this pass's
  handled scope.
- A non-default `beta` when `C` is present: `QGemm` has no `beta` attribute
  of its own to carry a different value through (its documented bias-scale
  formula implicitly assumes `beta = 1`), the same restriction
  `quantize_matmul_common.h`'s `MatchMatMulLike` already applies to the
  vanilla-Gemm path, for the same underlying reason.
- A node whose activation and/or output tensor has no calibrated range.

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --qoperator-quantize-gemm
```

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_qoperator_gemm(model)
onnx.save(model, "model.quant.onnx")

sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_qoperator_quantize_gemm.py` runs this simplify -> quantize ->
deploy sequence on small `Gemm` models -- including a `transA=1, transB=1,
alpha=2.5` case with a quantized bias, the exact combination
`quantize_qoperator`'s `QLinearMatMul` path cannot handle -- executing both
the float and quantized graphs through `onnxruntime.InferenceSession`.
