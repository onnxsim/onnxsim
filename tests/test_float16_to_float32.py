"""Tests for the ``float16_to_float32`` C++ pass
(onnxsim/passes/float16_to_float32.h).

Retypes an all-float16 graph to float32 -- the onnxsim-core counterpart of
``scripts/axera/legalize.py``'s ``float16_to_float32`` rule (see
``tests/test_axera_legalize.py``'s own
``test_float16_graph_is_retyped_everywhere_it_matters``, which this file's
cases mirror at the C++-pass level). Usable from any binding via
``extra_optimizers=["float16_to_float32"]``.

Models are built with ``onnx.parser`` per CLAUDE.md's convention, except for
the float16 tensor literals themselves -- the text format has no float16
literal syntax, so those are built with ``numpy_helper.from_array`` and
attached as initializers after parsing, also per CLAUDE.md.

``onnxsim.simplify``'s own ``check_n`` correctness check compares the
*converted* model against itself with the *same* feed dict -- no use here,
since this pass changes the graph's own declared input/output dtype (float16
-> float32), so the two models need differently-typed feeds. Numeric
equivalence is instead checked by running both models through ONNX Runtime
directly, each with its own correctly-typed input.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, numpy_helper, parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=17, ir_version=10):
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


def _f16(array, name):
    return numpy_helper.from_array(array.astype(np.float16), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _chained_model():
    """`y = (x + w) * w` in float16 throughout -- a couple of chained ops,
    like quantize_fp16's own tests use, rather than one op in isolation."""
    w = np.array([[1.5, -2.0, 0.25], [3.0, 0.5, -1.25]], np.float16)
    return _model(
        """
        g (float16[2,3] x) => (float16[2,3] y)
        {
          s = Add(x, w)
          y = Mul(s, w)
        }
        """,
        initializer=[_f16(w, "w")],
    )


def test_float16_graph_becomes_float32_and_computes_close_to_the_same_thing():
    model = _chained_model()
    sim_model, ok = onnxsim.simplify(model, extra_optimizers=["float16_to_float32"])
    assert ok
    assert sim_model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert sim_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    for init in sim_model.graph.initializer:
        assert init.data_type != TensorProto.FLOAT16
    onnx.checker.check_model(sim_model)

    x = np.random.RandomState(0).randn(2, 3).astype(np.float16)
    (before,) = _run(model, {"x": x})
    (after,) = _run(sim_model, {"x": x.astype(np.float32)})
    # float16 has ~3 decimal digits of precision, and the converted graph
    # computes in true float32 (a different rounding path than the
    # original's float16 kernels), so this is a looser tolerance than an
    # exact-conversion pass would need -- the same reasoning
    # test_quantize_fp16.py's _assert_close uses for its own precision
    # change, direction reversed.
    assert np.allclose(
        before.astype(np.float64), after.astype(np.float64), rtol=0.05, atol=0.05
    ), np.abs(before.astype(np.float64) - after.astype(np.float64)).max()


def test_float16_is_converted_wherever_it_is_stored():
    """Constants live in three places -- initializers, a `Constant` node's
    `value` attribute, and a `Cast`'s `to` attribute -- and converting only
    one leaves a graph mixing precisions, which onnxruntime rejects
    outright at load time (the same failure mode
    ``test_axera_legalize.py``'s Python-level version of this rule exists
    to prevent)."""
    two = _f16(np.array(2.0, np.float16), "two")
    model = _model(
        """
        g (float16[4] x) => (float16[4] y16)
        {
          c = Constant<value = float16[1] {3}>()
          d = Mul(x, c)
          e = Add(d, two)
          y = Div(e, two)
          y16 = Cast<to = 10>(y)
        }
        """,
        initializer=[two],
    )
    sim_model, ok = onnxsim.simplify(model, extra_optimizers=["float16_to_float32"])
    assert ok

    assert all(
        init.data_type != TensorProto.FLOAT16 for init in sim_model.graph.initializer
    )
    for node in sim_model.graph.node:
        for attr in node.attribute:
            if attr.name == "value":
                assert attr.t.data_type != TensorProto.FLOAT16
            if node.op_type == "Cast" and attr.name == "to":
                assert attr.i != TensorProto.FLOAT16
    for value in list(sim_model.graph.input) + list(sim_model.graph.output):
        assert value.type.tensor_type.elem_type != TensorProto.FLOAT16

    # Loads under onnxruntime, which the half-converted graph would not.
    ort.InferenceSession(
        sim_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def test_float32_graph_is_left_alone():
    model = _model(
        """
        g (float[2,3] x) => (float[2,3] y)
        { y = Add(x, x) }
        """
    )
    sim_model, ok = onnxsim.simplify(model, extra_optimizers=["float16_to_float32"])
    assert ok
    assert sim_model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT


def test_disabled_by_default():
    """`float16_to_float32` is `PassType::Other`, so a plain `simplify()`
    call (no `extra_optimizers`) must leave a float16 graph alone."""
    model = _chained_model()
    sim_model, ok = onnxsim.simplify(model)
    assert ok
    assert sim_model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT16
