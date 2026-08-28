"""Tests for ``onnxsim.quantize_bf16`` (the ``quantize_bf16`` C++ pass).

Like ``test_quantize_fp16.py``, models are built with multiple chained ops
(not one MatMul/Conv in isolation), since quantize_bf16 is a whole-graph
transform rather than a per-node pattern match.

Unlike quantize_fp16's tests, these do NOT execute a bfloat16 MatMul/Relu/Add
through onnxruntime's CPUExecutionProvider: as of onnxruntime 1.29, CPU EP
has no bfloat16 compute kernels for those ops at all (only ``Cast``/
``Identity`` do) -- a real, current runtime-support gap, not a bug in this
pass (the produced model is valid ONNX; see the ``onnx.checker`` calls
below, and ``docs/bf16-quantization.md``'s note on this). So instead:
- Numeric correctness of the conversion itself is checked directly, by
  decoding the produced bfloat16 initializer back to float32 (via
  ``ml_dtypes.bfloat16``, the numpy extension dtype ``onnx.numpy_helper``
  uses for BFLOAT16 tensors) and comparing against the original float32
  weight.
- One test does execute a real quantized graph through
  ``onnxruntime.InferenceSession``, using ``Identity`` (which CPU EP does
  support in bfloat16) as the compute op, to prove the boundary Cast
  wiring this pass builds actually loads and runs under a real engine, not
  just passes the checker.
"""

import collections

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.shape_inference
import pytest
from onnx import parser

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")
ml_dtypes = pytest.importorskip("ml_dtypes")


def _model(body, initializer=(), opset=13, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _initializer_by_name(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return init
    raise KeyError(name)


def _node_input_initializer(model, op_type, input_index):
    # quantize_bf16 replaces a converted weight with a *new* initializer
    # (leaving the old float32 one orphaned in the model until a follow-up
    # simplify() call prunes it -- documented behavior, matching every other
    # onnxsim quantize_* pass), so the initializer actually feeding a node
    # must be looked up by that node's *current* input name, not by
    # iterating/indexing graph.initializer blindly.
    node = next(n for n in model.graph.node if n.op_type == op_type)
    return _initializer_by_name(model, node.input[input_index])


def _to_bf16_rounded_float32(array):
    # The reference conversion this pass's C++ FloatToBFloat16Bits should
    # match: round each float32 value to its nearest bfloat16 and back.
    return array.astype(ml_dtypes.bfloat16).astype(np.float32)


def _two_matmul_model():
    rng = np.random.default_rng(0)
    k, n1, n2 = 16, 12, 8
    w1 = _f32(rng.standard_normal((k, n1)) * 0.5, "W1")
    w2 = _f32(rng.standard_normal((n1, n2)) * 0.5, "W2")
    model = _model(
        f"""
        g (float[4,{k}] X) => (float[4,{n2}] Y)
        {{
          H = MatMul(X, W1)
          Hr = Relu(H)
          Y = MatMul(Hr, W2)
        }}
        """,
        initializer=[w1, w2],
    )
    return model, rng, k, n2


def test_quantize_bf16_keep_io_types():
    model, rng, k, n2 = _two_matmul_model()

    quant = onnxsim.quantize_bf16(model)
    onnx.checker.check_model(quant)

    ops = _op_counts(quant)
    assert ops["Cast"] == 2  # one boundary cast for X in, one for Y out
    assert ops["MatMul"] == 2
    assert ops["Relu"] == 1

    # The model's own declared I/O stays float32.
    assert quant.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert quant.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    # But the weight actually feeding the (first) MatMul is now bfloat16 (the
    # old float32 initializer is left orphaned in the model -- see
    # _node_input_initializer's comment).
    w1_init = _node_input_initializer(quant, "MatMul", 1)
    assert w1_init.data_type == onnx.TensorProto.BFLOAT16

    w1_orig = onnx.numpy_helper.to_array(
        next(i for i in model.graph.initializer if i.name == "W1")
    )
    w1_bf16 = onnx.numpy_helper.to_array(w1_init).astype(np.float32)
    expected = _to_bf16_rounded_float32(w1_orig)
    np.testing.assert_array_equal(w1_bf16, expected)


def test_quantize_bf16_no_keep_io_types():
    model, rng, k, n2 = _two_matmul_model()

    quant = onnxsim.quantize_bf16(model, keep_io_types=False)
    onnx.checker.check_model(quant)

    ops = _op_counts(quant)
    # No boundary casts needed -- the graph's own I/O is redeclared bfloat16.
    assert ops["Cast"] == 0
    assert quant.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
    assert quant.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16


def test_quantize_bf16_converts_constant_node():
    # A Constant node's embedded value is a float32 "inline initializer" --
    # FetchConstantTensor covers it the same way as a true graph
    # initializer, so it should be converted too. The weight is built as a
    # real Constant node (kept on onnx.helper rather than a parsed
    # initializer) because that's specifically what this test exercises.
    rng = np.random.default_rng(1)
    k, n = 8, 4
    w = rng.standard_normal((k, n)).astype(np.float32)
    const_node = onnx.helper.make_node(
        "Constant",
        [],
        ["W"],
        value=onnx.numpy_helper.from_array(w, "W"),
    )
    model = _model(
        f"""
        g (float[3,{k}] X) => (float[3,{n}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """
    )
    model.graph.node.insert(0, const_node)

    quant = onnxsim.quantize_bf16(model)
    onnx.checker.check_model(quant)

    w_init = _node_input_initializer(quant, "MatMul", 1)
    assert w_init.data_type == onnx.TensorProto.BFLOAT16
    w_bf16 = onnx.numpy_helper.to_array(w_init).astype(np.float32)
    np.testing.assert_array_equal(w_bf16, _to_bf16_rounded_float32(w))


def test_quantize_bf16_no_clamping_needed_for_large_values():
    # bfloat16 keeps float32's full 8-bit exponent range, unlike float16
    # (which must clamp values beyond +-65504) -- a large-magnitude weight
    # stays finite, and NaN stays NaN, with no special-casing in the pass.
    w = np.array([[1.0e10, -1.0e10, 3.0, float("nan")]], dtype=np.float32)
    weight = _f32(w, "W")
    model = _model(
        """
        g (float[2,1] X) => (float[2,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[weight],
    )

    quant = onnxsim.quantize_bf16(model)
    onnx.checker.check_model(quant)

    w_init = _node_input_initializer(quant, "MatMul", 1)
    w_bf16 = onnx.numpy_helper.to_array(w_init).astype(np.float32)
    assert np.all(np.isfinite(w_bf16[0, :3]))
    assert np.isnan(w_bf16[0, 3])
    np.testing.assert_array_equal(w_bf16, _to_bf16_rounded_float32(w))


def test_quantize_bf16_skips_optional_input_default_initializer():
    # An initializer whose name is also a graph input (the ONNX "optional
    # input with a default value" convention) is left alone entirely -- see
    # quantize_bf16.h's doc comment.
    w = _f32(np.random.randn(4, 2).astype(np.float32), "W")
    model = _model(
        """
        g (float[3,4] X, float[4,2] W) => (float[3,2] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[w],
    )

    quant = onnxsim.quantize_bf16(model)
    onnx.checker.check_model(quant)
    w_init = quant.graph.initializer[0]
    assert w_init.data_type == onnx.TensorProto.FLOAT  # untouched


def test_quantize_bf16_clears_stale_value_info_on_already_shape_inferred_model():
    # Same regression as test_quantize_fp16.py's identically-named test: a
    # model that already went through shape inference has its interior
    # activations' value_info pre-populated float32, and quantize_bf16 (like
    # quantize_fp16) doesn't re-run shape inference itself, so it must not
    # leave that now-wrong float32 declaration in place -- a real bug found
    # via a real torchvision model. No onnxruntime.InferenceSession run here
    # (CPU EP has no bfloat16 MatMul/Relu kernel, see this file's module
    # docstring) -- the value_info correctness itself is the check.
    model, rng, k, n2 = _two_matmul_model()
    model = onnx.shape_inference.infer_shapes(model)
    h_before = next(vi for vi in model.graph.value_info if vi.name == "H")
    assert h_before.type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    quant = onnxsim.quantize_bf16(model)
    onnx.checker.check_model(quant)

    h_after = next((vi for vi in quant.graph.value_info if vi.name == "H"), None)
    if h_after is not None:
        assert h_after.type.tensor_type.elem_type != onnx.TensorProto.FLOAT


def test_quantize_bf16_boundary_casts_execute():
    # Proves the boundary Cast wiring this pass builds is not just
    # checker-valid but actually loads and runs under a real engine.
    # Identity is used as the compute op in the middle since (as of
    # onnxruntime 1.29) CPUExecutionProvider has a bfloat16 Cast/Identity
    # kernel but no bfloat16 MatMul/Relu/Add kernel -- see this file's
    # module docstring.
    model = _model(
        """
        g (float[2,3] X) => (float[2,3] Y)
        {
          Y = Identity(X)
        }
        """
    )

    quant = onnxsim.quantize_bf16(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Cast"] == 2
    assert ops["Identity"] == 1

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    (out,) = _run(quant, {"X": x})
    np.testing.assert_array_equal(out, x)
