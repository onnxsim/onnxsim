"""Tests for the ``maxpool_rowmajor_when_indices_unused`` C++ pass
(onnxsim/passes/maxpool_rowmajor_when_indices_unused.h).

A `MaxPool` with `storage_order = 1` (column-major) gets it cleared to 0
(row-major) whenever the optional `Indices` output isn't actually consumed
-- the attribute only orders that output, so with no consumer the two
settings compute the identical `Y`. The onnxsim-core counterpart of
``scripts/axelera/legalize.py``'s ``maxpool_rowmajor_when_indices_unused``
rule (that file's docstring records ``storage_order == 0`` as a real vendor
compiler requirement). This version is a little more thorough than the
Python script: it declines based on whether `Indices` actually has a
consumer (tracked by the in-memory graph IR), not merely whether a second
output name is declared, so a *declared-but-dead* `Indices` output is still
safely rewritten.

Models are built with ``onnx.parser`` per CLAUDE.md's convention. Numeric
equivalence comes from ``onnxsim.simplify``'s own ``check_n``.
"""

import numpy as np
import onnx
from onnx import parser

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


def _simplify(model, x, check_n=1):
    return onnxsim.simplify(
        model,
        check_n=check_n,
        input_data={"x": x},
        extra_optimizers=["maxpool_rowmajor_when_indices_unused"],
    )


def test_storage_order_cleared_when_no_indices_output_is_declared():
    model = _model(
        """
        g (float[1,4,8,8] x) => (float[1,4,7,7] y)
        { y = MaxPool<kernel_shape=[2, 2], storage_order=1>(x) }
        """
    )
    x = np.random.RandomState(0).randn(1, 4, 8, 8).astype(np.float32)
    sim_model, ok = _simplify(model, x)
    assert ok
    node = _find(sim_model, "MaxPool")
    assert _attr_int(node, "storage_order") == 0


def test_storage_order_cleared_when_indices_output_is_declared_but_dead():
    model = _model(
        """
        g (float[1,4,8,8] x) => (float[1,4,7,7] y)
        { y, idx = MaxPool<kernel_shape=[2, 2], storage_order=1>(x) }
        """
    )
    x = np.random.RandomState(1).randn(1, 4, 8, 8).astype(np.float32)
    sim_model, ok = _simplify(model, x)
    assert ok
    node = _find(sim_model, "MaxPool")
    assert _attr_int(node, "storage_order") == 0


def test_storage_order_is_left_alone_when_indices_output_is_used():
    model = _model(
        """
        g (float[1,4,8,8] x) => (float[1,4,7,7] y, int64[1,4,7,7] idx)
        { y, idx = MaxPool<kernel_shape=[2, 2], storage_order=1>(x) }
        """
    )
    x = np.random.RandomState(2).randn(1, 4, 8, 8).astype(np.float32)
    sim_model, ok = _simplify(model, x)
    assert ok
    node = _find(sim_model, "MaxPool")
    assert _attr_int(node, "storage_order") == 1


def test_default_storage_order_is_left_alone():
    model = _model(
        """
        g (float[1,4,8,8] x) => (float[1,4,7,7] y)
        { y = MaxPool<kernel_shape=[2, 2]>(x) }
        """
    )
    x = np.random.RandomState(3).randn(1, 4, 8, 8).astype(np.float32)
    sim_model, ok = _simplify(model, x)
    assert ok
    node = _find(sim_model, "MaxPool")
    assert _attr_int(node, "storage_order") == 0


def test_disabled_by_default():
    """`maxpool_rowmajor_when_indices_unused` is `PassType::Other`, so a
    plain `simplify()` call (no `extra_optimizers`) must leave
    `storage_order` alone."""
    model = _model(
        """
        g (float[1,4,8,8] x) => (float[1,4,7,7] y)
        { y = MaxPool<kernel_shape=[2, 2], storage_order=1>(x) }
        """
    )
    x = np.random.RandomState(4).randn(1, 4, 8, 8).astype(np.float32)
    sim_model, ok = onnxsim.simplify(model, check_n=1, input_data={"x": x})
    assert ok
    node = _find(sim_model, "MaxPool")
    assert _attr_int(node, "storage_order") == 1
