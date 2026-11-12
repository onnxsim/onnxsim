"""Regression tests for the quantize_dynamic / quantize_static fixes.

* quantize_dynamic on a model whose weights are external-data initializers that
  were never loaded used to SEGV inside the native weight reader. It must raise.
* quantize_dynamic must not leave the replaced float weights in the graph.
* quantize_static below opset 13 used to return the model unquantized without a
  word. It must raise, and at opset 13 it must actually insert QDQ nodes.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import TensorProto, parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _matmul_model(weight, opset=13):
    k, n = weight.shape
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": {opset}]
        >
        agraph (float[1, {k}] x) => (float[1, {n}] y)
        {{
            y = MatMul(x, W)
        }}
        """
    )
    model.graph.initializer.extend([onnx.numpy_helper.from_array(weight, "W")])
    return model


def _external_unloaded(name, dims):
    t = TensorProto()
    t.name = name
    t.data_type = TensorProto.FLOAT
    t.dims.extend(dims)
    t.data_location = TensorProto.EXTERNAL
    entry = t.external_data.add()
    entry.key = "location"
    entry.value = "weights.bin"
    return t


def test_quantize_dynamic_unloaded_external_data_raises_not_segfaults():
    model = _matmul_model(np.zeros((16, 16), np.float32))
    # Swap the materialized weight for an external-data reference with no bytes,
    # which is what a model looks like after save_as_external_data mutated it.
    del model.graph.initializer[:]
    model.graph.initializer.extend([_external_unloaded("W", [16, 16])])
    with pytest.raises(RuntimeError, match="external data"):
        onnxsim.quantize_dynamic(model)


def test_quantize_dynamic_drops_replaced_float_weights_and_stays_close():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 16)).astype(np.float32)
    model = _matmul_model(w)
    q = onnxsim.quantize_dynamic(model)

    used = {i for n in q.graph.node for i in n.input}
    float_inits = [
        i.name for i in q.graph.initializer if i.data_type == TensorProto.FLOAT
    ]
    assert "W" not in float_inits, "replaced float weight must be pruned"
    assert all(name in used for name in float_inits), "no dead float initializers left"

    x = rng.standard_normal((1, 32)).astype(np.float32)
    ref = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
        None, {"x": x}
    )[0]
    out = ort.InferenceSession(q.SerializeToString(), providers=["CPUExecutionProvider"]).run(
        None, {"x": x}
    )[0]
    assert np.max(np.abs(ref - out)) < 0.05 * np.max(np.abs(ref))


def test_quantize_static_rejects_opset_below_13():
    rng = np.random.default_rng(1)
    model = _matmul_model(rng.standard_normal((8, 8)).astype(np.float32), opset=11)
    calib = [{"x": rng.standard_normal((1, 8)).astype(np.float32)} for _ in range(4)]
    with pytest.raises(ValueError, match="opset >= 13"):
        onnxsim.quantize_static(model, calibration_data=calib)


def test_quantize_static_inserts_qdq_at_opset_13():
    rng = np.random.default_rng(2)
    model = _matmul_model(rng.standard_normal((8, 8)).astype(np.float32), opset=13)
    calib = [{"x": rng.standard_normal((1, 8)).astype(np.float32)} for _ in range(4)]
    q = onnxsim.quantize_static(model, calibration_data=calib)
    q = q[0] if isinstance(q, tuple) else q
    ops = [n.op_type for n in q.graph.node]
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops
