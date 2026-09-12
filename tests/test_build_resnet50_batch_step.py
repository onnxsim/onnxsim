"""Tests for the two pure ONNX-graph fixups
``scripts/axera/build_resnet50_batch_step.py`` needs before a `dynamo=True`
export of a `timm` model can go through the rest of this project's training-
step pipeline -- see that module's own docstring for why each exists (real
per-layer names require `dynamo=True`, which lands on opset 18; and
`timm`'s global-pool flatten bakes a batch-1-literal `Reshape` target).

Neither needs Docker, a device, or even `torch`/`timm` -- both operate on
plain ONNX graphs built with `onnx.parser`.
"""

import os
import sys

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

ort = pytest.importorskip("onnxruntime")

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import build_resnet50_batch_step as m  # noqa: E402


def _run(model, feeds, output_names=None):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(output_names, feeds)


def _opset18_reducemean_model():
    """`ReduceMean` with axes as an *input* (opset 18's form) over a 4-D
    tensor's spatial axes, plus a plain elementwise op untouched by the
    rewrite -- mirrors `timm`'s global-average-pool decomposition under
    `dynamo=True`."""
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 18]
        >
        g (float[1,4,3,3] x) => (float[1,4] y)
        {
          r = Relu(x)
          axes = Constant<value = int64[2] {2, 3}>()
          pooled = ReduceMean<keepdims=1, noop_with_empty_axes=0>(r, axes)
          shape = Constant<value = int64[2] {1, 4}>()
          y = Reshape(pooled, shape)
        }
        """
    )
    return model


def test_downgrade_reduce_axes_to_attr_converts_input_to_attribute():
    model = _opset18_reducemean_model()
    # fold the Constant nodes into initializers first -- ReduceMean's own
    # axes input must be a real initializer for the rewrite to read its
    # value, matching what this graph looks like after
    # build_resident_train_step._fold_constants runs.
    from onnxsim import simplify as _simplify

    model, ok = _simplify(model)
    assert ok

    before_op = [n.op_type for n in model.graph.node]
    assert "Constant" not in before_op

    reduce_before = next(n for n in model.graph.node if n.op_type == "ReduceMean")
    assert len(reduce_before.input) == 2, "fixture must start in axes-as-input form"

    out = m._downgrade_reduce_axes_to_attr(model)
    onnx.checker.check_model(out)

    reduce_after = next(n for n in out.graph.node if n.op_type == "ReduceMean")
    assert len(reduce_after.input) == 1
    axes_attr = next(a for a in reduce_after.attribute if a.name == "axes")
    assert list(axes_attr.ints) == [2, 3]
    assert not any(a.name == "noop_with_empty_axes" for a in reduce_after.attribute)
    assert all(
        (o.version if not o.domain else None) in (17, None) for o in out.opset_import
    )
    assert next(o.version for o in out.opset_import if not o.domain) == 17

    x = np.random.default_rng(0).standard_normal((1, 4, 3, 3)).astype(np.float32)
    (before_y,) = _run(model, {"x": x}, ["y"])
    (after_y,) = _run(out, {"x": x}, ["y"])
    assert np.allclose(before_y, after_y, atol=1e-6)


def test_downgrade_reduce_axes_to_attr_is_a_noop_without_axes_inputs():
    """A plain, already-attribute-form ReduceMean (opset 17 and below) must
    pass through unchanged -- the rewrite should only ever fire on the
    axes-as-input form."""
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[1,4,3,3] x) => (float[1,4] y)
        {
          y = ReduceMean<axes=[2,3], keepdims=0>(x)
        }
        """
    )
    out = m._downgrade_reduce_axes_to_attr(model)
    onnx.checker.check_model(out)
    node = out.graph.node[0]
    assert list(node.attribute[0].ints) == [2, 3]


def _flatten_reshape_model(target_dims):
    w = numpy_helper.from_array(
        np.random.default_rng(1).standard_normal((8, 4)).astype(np.float32), "w"
    )
    shape = numpy_helper.from_array(
        np.array(target_dims, dtype=np.int64), "shape_const"
    )
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[batch,8,1,1] pooled) => (float[batch,4] y)
        {
          flat = Reshape(pooled, shape_const)
          y = MatMul(flat, w)
        }
        """
    )
    model.graph.initializer.extend([w, shape])
    return model


def test_fix_flatten_reshape_rewrites_batch1_literal_to_dynamic():
    model = _flatten_reshape_model([1, 8])
    out = m._fix_flatten_reshape(model)
    onnx.checker.check_model(out)

    reshape = next(n for n in out.graph.node if n.op_type == "Reshape")
    shape_init = next(t for t in out.graph.initializer if t.name == reshape.input[1])
    assert numpy_helper.to_array(shape_init).tolist() == [-1, 8]

    # correctness: still reshapes a batch-1 input the same way as before.
    x = np.random.default_rng(2).standard_normal((1, 8, 1, 1)).astype(np.float32)
    (before_y,) = _run(model, {"pooled": x}, ["y"])
    (after_y,) = _run(out, {"pooled": x}, ["y"])
    assert np.allclose(before_y, after_y, atol=1e-6)

    # and now actually works at batch>1, which the un-rewritten [1, 8]
    # target would refuse with an element-count mismatch.
    x4 = np.random.default_rng(3).standard_normal((4, 8, 1, 1)).astype(np.float32)
    (y4,) = _run(out, {"pooled": x4}, ["y"])
    assert y4.shape == (4, 4)


def test_fix_flatten_reshape_requires_exactly_one_match():
    """A graph with no batch-1-literal Reshape (already rank-2 with a
    non-1 first dim, e.g. an already-fixed or differently-shaped graph)
    should fail loudly rather than silently doing nothing -- see the
    module's own docstring for why a silent no-op here is the wrong
    failure mode."""
    model = _flatten_reshape_model([2, 8])
    with pytest.raises(RuntimeError, match="expected exactly one"):
        m._fix_flatten_reshape(model)
