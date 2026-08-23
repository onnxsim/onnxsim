# QOperator pooling quantization (`quantize_qoperator_pool`)

## What this is

`onnxsim.quantize_qoperator_pool` is a self-contained C++ graph rewrite
(`onnxsim/passes/qoperator_quantize_pool.h`) that statically
(calibration-based) quantizes every standalone `AveragePool` or
`GlobalAveragePool` node whose input is a float32 tensor into ONNX
Runtime's **`com.microsoft`** contrib ops `QLinearAveragePool`/
`QLinearGlobalAveragePool` -- the pooling analogue of
`quantize_qoperator_activation`'s `QLinearSigmoid`/`QLinearLeakyRelu`
rewrite (see `docs/qoperator-quantize-elementwise.md` for why these are
contrib, not standard, ONNX ops).

```
Before (illustrated for AveragePool; GlobalAveragePool is identical but for
the op/QLinear* name and having no attributes of its own to carry over):
  Y = AveragePool(X, kernel_shape=k, pads=p, ...)   X: runtime float32

After:
  Xq = QuantizeLinear(X, Xs, Xzp)                            -- CALIBRATED
  Yq = QLinearAveragePool(Xq, Xs, Xzp, Ys, Yzp,
                          kernel_shape=k, pads=p, ...,
                          channels_last=0)                   -- true int8 compute
  Y  = DequantizeLinear(Yq, Ys, Yzp)                         -- CALIBRATED
```

Like `quantize_qoperator_activation`'s `QLinearSigmoid`/`QLinearLeakyRelu`,
there is only ever one operand here, so only its own calibrated range is
needed on top of the output's -- these ops compute directly in int8, so the
output must be quantized too. `list_qoperator_pool_quantizable_tensors`
reports the input's and the node's output's tensor names for each
qualifying node; `calibrate()`'s `extra_tensor_names` parameter is how they
get folded into the same calibration run -- `quantize_qoperator_pool` (the
Python wrapper in `onnxsim/calibration.py`) does this automatically.

## Attributes are carried over verbatim

`QLinearAveragePool`'s schema mirrors standard ONNX `AveragePool`'s own
attribute set exactly (`kernel_shape`, `pads`, `strides`, `ceil_mode`,
`count_include_pad`, `auto_pad`), so this pass copies every attribute the
original node has onto the rewritten node unchanged via `copyAttributes` --
no per-attribute translation needed. `GlobalAveragePool` has no attributes
of its own, so this is a no-op for that case. Both contrib ops additionally
take a `channels_last` attribute standard ONNX pooling doesn't have; since
onnxsim only ever produces NCHW-layout graphs, this pass always explicitly
sets it to `0`.

## Why a `dilations` attribute is left alone

Standard ONNX `AveragePool` gained an optional `dilations` attribute in
opset 19. ONNX Runtime's `QLinearAveragePool` kernel does not accept it --
`"Unrecognized attribute: dilations for operator QLinearAveragePool"` is a
real, observed ONNX Runtime error for this exact combination. Rather than
risk emitting a node ONNX Runtime refuses to load, an `AveragePool` node
carrying a `dilations` attribute (regardless of its value) is left
untouched entirely.

## Scope

Handled:
- A standalone `AveragePool` or `GlobalAveragePool` node with exactly 1
  float32 input, and (for `AveragePool`) no `dilations` attribute.

Left untouched (safe no-op, node passes through as-is):
- A non-float32 input, or a node whose input and/or output tensor has no
  calibrated range.
- An `AveragePool` node with a `dilations` attribute set.
- A node consuming *another* rewritten node's output in the same
  quantization call -- the same pre-existing QOperator-family
  characteristic `qoperator_quantize_elementwise.h` documents.

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --qoperator-quantize-pool
```

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_qoperator_pool(model)
onnx.save(model, "model.quant.onnx")

sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_qoperator_quantize_pool.py` runs this simplify -> quantize ->
deploy sequence on small `AveragePool` (including kernel_shape/strides and
padding/count_include_pad variants) and `GlobalAveragePool` models,
executing both the float and quantized graphs through
`onnxruntime.InferenceSession`.
