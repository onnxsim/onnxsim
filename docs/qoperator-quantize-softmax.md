# QOperator softmax quantization (`quantize_qoperator_softmax`)

## What this is

`onnxsim.quantize_qoperator_softmax` is a self-contained C++ graph rewrite
(`onnxsim/passes/qoperator_quantize_softmax.h`) that statically
(calibration-based) quantizes every standalone `Softmax` node whose input is
a float32 tensor into ONNX Runtime's **`com.microsoft`** contrib op
`QLinearSoftmax` -- the reduction-axis analogue of
`quantize_qoperator_activation`'s `QLinearSigmoid`/`QLinearLeakyRelu` rewrite
(see `docs/qoperator-quantize-elementwise.md` for why these are contrib, not
standard, ONNX ops).

```
Before:
  Y = Softmax(X, axis=ax)     X: runtime float32 tensor

After:
  Xq = QuantizeLinear(X, Xs, Xzp)                          -- CALIBRATED
  Yq = QLinearSoftmax(Xq, Xs, Xzp, Ys, Yzp,
                      axis=ax, opset=default_domain_opset) -- true int8 compute
  Y  = DequantizeLinear(Yq, Ys, Yzp)                       -- CALIBRATED
```

Like `quantize_qoperator_activation`'s `QLinearSigmoid`/`QLinearLeakyRelu`,
there is only ever one operand here, so only its own calibrated range is
needed on top of the output's -- `QLinearSoftmax` computes directly in int8,
so the output must be quantized too.
`list_qoperator_softmax_quantizable_tensors` reports the input's and the
node's output's tensor names for each qualifying node; `calibrate()`'s
`extra_tensor_names` parameter is how they get folded into the same
calibration run -- `quantize_qoperator_softmax` (the Python wrapper in
`onnxsim/calibration.py`) does this automatically.

## Why the `opset` attribute matters

Standard ONNX `Softmax` has **two incompatible axis semantics** across
opset versions:

- **Pre-opset-13**: flattens the tensor into a 2-D matrix at `axis` (all
  dims before `axis` become the row, all dims from `axis` on become the
  column) and reduces over that single trailing dimension.
- **Opset-13+**: reduces over `axis` in place, same rank in and out --
  the far more intuitive, now-standard behavior.

`QLinearSoftmax`'s `opset` attribute tells ONNX Runtime's kernel which of
these two behaviors to replicate. Rather than guess or hardcode a version,
this pass reads the **model's own** declared default-domain (`""`/
`"ai.onnx"`) opset from its opset imports and threads that value through
verbatim, so the rewritten node reproduces the exact semantics the original
`Softmax` node already had -- correct regardless of which opset the input
model targets. A model with no resolvable default-domain opset import is
left untouched entirely: there is no safe default to guess, so no `Softmax`
node in it is reported as quantizable either.

## Scope

Handled:
- A standalone `Softmax` node with exactly 1 float32 input, in a model with
  a resolvable default-domain opset import. The `axis` attribute (default
  `-1`) is carried over unchanged.

Left untouched (safe no-op, node passes through as-is):
- A non-float32 input, or a node whose input and/or output tensor has no
  calibrated range.
- Any `Softmax` node in a model with no resolvable default-domain opset
  import.
- A node consuming *another* rewritten node's output in the same
  quantization call -- the same pre-existing QOperator-family
  characteristic `qoperator_quantize_elementwise.h` documents.

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --qoperator-quantize-softmax
```

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_qoperator_softmax(model)
onnx.save(model, "model.quant.onnx")

sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_qoperator_quantize_softmax.py` runs this simplify -> quantize ->
deploy sequence on small `Softmax` models -- including one built on opset 11
specifically to prove the `opset` attribute is threaded through correctly
and ONNX Runtime's pre-opset-13 flattened-reduction kernel path produces the
same result as the unquantized opset-11 graph -- executing both the float
and quantized graphs through `onnxruntime.InferenceSession`.
