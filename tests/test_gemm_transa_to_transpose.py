"""Tests for the ``gemm_transA_to_transpose`` C++ pass
(onnxsim/passes/gemm_transa_to_transpose.h).

A `Gemm` with `transA = 1` gets an explicit `Transpose` on `A` instead, with
`transA` cleared -- the onnxsim-core counterpart of
``scripts/axelera/legalize.py``'s ``gemm_transA_to_transpose`` rule (that
file's docstring records ``transA == 0`` as a real vendor compiler
requirement). Usable from any binding via
``extra_optimizers=["gemm_transA_to_transpose"]``.

Models are built with ``onnx.parser`` per CLAUDE.md's convention.
Numeric equivalence comes from ``onnxsim.simplify``'s own ``check_n``.
"""

import numpy as np
import onnx
from onnx import numpy_helper, parser

import onnxsim


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
    onnx.checker.check_model(model)
    return model


def _find(model, op_type):
    return next(n for n in model.graph.node if n.op_type == op_type)


def _attr_int(node, name, default=0):
    a = next((a for a in node.attribute if a.name == name), None)
    return a.i if a is not None else default


def test_gemm_transA_becomes_an_explicit_transpose_and_computes_the_same_thing():
    b = numpy_helper.from_array(
        np.random.RandomState(0).randn(4, 8).astype(np.float32), "b"
    )
    c = numpy_helper.from_array(
        np.random.RandomState(1).randn(8).astype(np.float32), "c"
    )
    model = _model(
        """
        g (float[4,10] a) => (float[10,8] y)
        { y = Gemm<transA=1>(a, b, c) }
        """,
        initializer=[b, c],
    )
    a = np.random.RandomState(2).randn(4, 10).astype(np.float32)
    sim_model, ok = onnxsim.simplify(
        model,
        check_n=1,
        input_data={"a": a},
        extra_optimizers=["gemm_transA_to_transpose"],
    )
    assert ok

    gemm = _find(sim_model, "Gemm")
    assert _attr_int(gemm, "transA") == 0
    transpose = _find(sim_model, "Transpose")
    perm = next(a for a in transpose.attribute if a.name == "perm")
    assert list(perm.ints) == [1, 0]
    assert transpose.output[0] == gemm.input[0]


def test_gemm_without_transA_is_left_alone():
    b = numpy_helper.from_array(np.zeros((8, 4), np.float32), "b")
    model = _model(
        """
        g (float[4,8] a) => (float[4,4] y)
        { y = Gemm(a, b) }
        """,
        initializer=[b],
    )
    a = np.zeros((4, 8), np.float32)
    sim_model, ok = onnxsim.simplify(
        model,
        check_n=1,
        input_data={"a": a},
        extra_optimizers=["gemm_transA_to_transpose"],
    )
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert "Transpose" not in op_types


def test_disabled_by_default():
    """`gemm_transA_to_transpose` is `PassType::Other`, so a plain
    `simplify()` call (no `extra_optimizers`) must leave `transA` alone."""
    b = numpy_helper.from_array(np.zeros((4, 8), np.float32), "b")
    model = _model(
        """
        g (float[4,10] a) => (float[10,8] y)
        { y = Gemm<transA=1>(a, b) }
        """,
        initializer=[b],
    )
    a = np.zeros((4, 10), np.float32)
    sim_model, ok = onnxsim.simplify(model, check_n=1, input_data={"a": a})
    assert ok
    gemm = _find(sim_model, "Gemm")
    assert _attr_int(gemm, "transA") == 1
