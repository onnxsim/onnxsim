# Dynamic INT8 quantization, fused dequantize (`quantize_dynamic_matmul_integer_to_float`)

## What this is

`onnxsim.quantize_dynamic_matmul_integer_to_float` is a single,
self-contained C++ graph rewrite
(`onnxsim/passes/dynamic_quantize_matmul_integer_to_float.h`) applying
**exactly the same dynamic quantization scheme**
[dynamic-quantization.md](dynamic-quantization.md)'s `quantize_dynamic`
does -- same matching rules, same per-output-channel symmetric INT8 weight
quantization from static values, same runtime `DynamicQuantizeLinear`
activation quantization -- but the dequantize step is a single ONNX
Runtime **`com.microsoft`** contrib op, `MatMulIntegerToFloat`, instead of
`quantize_dynamic`'s three-to-four separate standard-ONNX nodes:

```
Before:
  Y = MatMul(X, W)                                   # W constant, [K, N], float32

After (quantize_dynamic, for comparison):
  Xq, Xs, Xzp = DynamicQuantizeLinear(X)
  Wq, Ws      = <as above>
  Acc         = MatMulInteger(Xq, Wq, Xzp)            # int32
  Y           = Cast<float>(Acc) * (Xs * Ws)          # Cast + 2x Mul

After (this pass):
  Xq, Xs, Xzp = DynamicQuantizeLinear(X)
  Wq, Ws      = <as above>
  Y = MatMulIntegerToFloat(Xq, Wq, Xs, Ws, Xzp[, Bias])  # one fused op
```

`MatMulIntegerToFloat`'s own schema dequantizes directly to float **and**
adds an optional bias in the same op, so a `Gemm` bias needs no separate
`Add` node either -- this pass needs only two nodes total
(`DynamicQuantizeLinear` plus the one contrib op) versus `quantize_dynamic`'s
four to five.

## The tradeoff: portability

Unlike `quantize_dynamic`'s output, which is 100% standard ONNX, this
pass's result needs a `com.microsoft`-aware runtime (ONNX Runtime itself,
or another runtime importing the same contrib schemas) to execute --
`MatMulIntegerToFloat` is an ONNX Runtime contrib op. If portability to a
non-ONNX-Runtime engine matters more than the smaller node count, use
`quantize_dynamic` instead; both apply identical weight/activation
quantization math, so the numerical result is the same either way.

## Scope

Identical to `quantize_dynamic`'s (see
[dynamic-quantization.md](dynamic-quantization.md#scope) and its
Accumulator-overflow guard section, which apply unchanged here -- the same
`IsSafeInt32ReductionDepth` check guards this pass too, since
`MatMulIntegerToFloat`'s kernel still accumulates in int32 internally even
though its final output is float).

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --dynamic-quantize-matmul-integer-to-float
```

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_dynamic_matmul_integer_to_float(model)
onnx.save(model, "model.quant.onnx")

sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_dynamic_quantize_matmul_integer_to_float.py` mirrors
`test_dynamic_quantize_matmul.py`'s cases -- plain `MatMul`, a `transB=1`
`Gemm` with a bias (verifying the bias lands as `MatMulIntegerToFloat`'s
7th input directly, not a separate `Add`), the no-bias empty-placeholder
input, and the accumulator-overflow skip -- executing both the float and
quantized graphs through `onnxruntime.InferenceSession`.
