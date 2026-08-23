# QOperator where quantization (`quantize_qoperator_where`)

## What this is

`onnxsim.quantize_qoperator_where` is a self-contained C++ graph rewrite
(`onnxsim/passes/qoperator_quantize_where.h`) that statically
(calibration-based) quantizes every `Where` node whose two data operands are
both non-constant float32 tensors into ONNX Runtime's **`com.microsoft`**
contrib op `QLinearWhere` -- the ternary-select analogue of
`quantize_qoperator_elementwise`'s `QLinearAdd`/`QLinearMul` rewrite (see
`docs/qoperator-quantize-elementwise.md` for why these are contrib, not
standard, ONNX ops).

```
Before:
  Z = Where(Cond, A, B)   -- Cond: bool; A, B: both runtime float32

After:
  Aq = QuantizeLinear(A, As, Azp)   -- As/Azp: CALIBRATED
  Bq = QuantizeLinear(B, Bs, Bzp)   -- Bs/Bzp: CALIBRATED
  Zq = QLinearWhere(Cond, Aq, As, Azp, Bq, Bs, Bzp, Zs, Zzp)
       -- true int8 compute; Cond passes through unquantized (it's bool)
  Z  = DequantizeLinear(Zq, Zs, Zzp)   -- Zs/Zzp: CALIBRATED
```

Like `quantize_qoperator_elementwise`'s `QLinearAdd`/`QLinearMul`,
`QLinearWhere` has no "weight" role -- both data operands are calibrated as
activations, on top of the output's own calibrated range (QOperator format
computes directly in int8, so the output must be quantized too). The
condition operand is never quantized: `QLinearWhere`'s schema takes it as a
plain `tensor(bool)`, passed straight through unchanged.
`list_qoperator_where_quantizable_tensors` reports both data operands' and
the node's output's tensor names for each qualifying node; `calibrate()`'s
`extra_tensor_names` parameter is how they get folded into the same
calibration run -- `quantize_qoperator_where` (the Python wrapper in
`onnxsim/calibration.py`) does this automatically.

## Why a constant operand is left alone

Same reasoning as `quantize_qoperator_elementwise`: a `Where` data operand
that is a constant tensor is better quantized from its own static values
than force-fed through the runtime calibration harness as if it varied at
inference time. A `Where` with *either* data operand constant is left
untouched entirely, rather than partially rewritten.

## Scope

Handled:
- A `Where` node whose two data operands (inputs 1 and 2 -- the condition
  is always boolean and never a quantization candidate) are both
  non-constant float32 tensors.

Left untouched (safe no-op, node passes through as-is):
- Either data operand constant or non-float32, or a node whose operands
  and/or output tensor has no calibrated range.
- A node consuming *another* rewritten node's output in the same
  quantization call -- the same pre-existing QOperator-family
  characteristic `qoperator_quantize_elementwise.h` documents.

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --qoperator-quantize-where
```

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_qoperator_where(model)
onnx.save(model, "model.quant.onnx")

sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_qoperator_quantize_where.py` runs this simplify -> quantize ->
deploy sequence on small `Where` models (including a broadcasting-shapes
variant), executing both the float and quantized graphs through
`onnxruntime.InferenceSession`.
