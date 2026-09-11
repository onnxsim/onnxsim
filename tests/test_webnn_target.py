"""Tests for ``onnxsim.check_webnn_support`` (the ``onnxsim/webnn_target.py``
advisory checker) and the ``gemm_fusion_backend="webnn"`` alias on
:func:`onnxsim.simplify`.
"""

import numpy as np
import onnx
import onnx.numpy_helper
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
    return model


def test_check_webnn_support_clean_model():
    model = _model(
        """
        g (float[2,3] x) => (float[6] y)
        {
          shape = Constant<value = int64[1] {6}>()
          y = Reshape(x, shape)
        }
        """
    )
    assert onnxsim.check_webnn_support(model) == []


def test_check_webnn_support_flags_int64_input():
    model = _model(
        """
        g (int64[2,3] x) => (int64[2,3] y)
        {
          y = Identity(x)
        }
        """
    )
    messages = onnxsim.check_webnn_support(model)
    assert len(messages) == 2  # both the input and the output are INT64
    assert any("input 'x'" in m for m in messages)
    assert any("output 'y'" in m for m in messages)


def test_check_webnn_support_flags_int64_output_only():
    model = _model(
        """
        g (float[2,3] x) => (int64[2,3] y)
        {
          y = Cast<to = 7>(x)
        }
        """
    )
    messages = onnxsim.check_webnn_support(model)
    assert len(messages) == 1
    assert "output 'y'" in messages[0]


def test_check_webnn_support_flags_non_constant_reshape_shape():
    # `shape` is computed at runtime (Shape() of another input), not a graph
    # boundary input/output, so only the Reshape check should fire here --
    # isolating it from the separate INT64-boundary check.
    model = _model(
        """
        g (float[2,3] x, float[6] like) => (float[?] y)
        {
          shape = Shape(like)
          y = Reshape(x, shape)
        }
        """
    )
    messages = onnxsim.check_webnn_support(model)
    assert len(messages) == 1
    assert "Reshape" in messages[0] and "shape" in messages[0]


def test_check_webnn_support_flags_non_constant_expand_shape():
    model = _model(
        """
        g (float[2,3] x, float[6] like) => (float[?] y)
        {
          shape = Shape(like)
          y = Expand(x, shape)
        }
        """
    )
    messages = onnxsim.check_webnn_support(model)
    assert len(messages) == 1
    assert "Expand" in messages[0]


def test_check_webnn_support_accepts_constant_initializer_shape():
    # A Reshape whose shape is a top-level initializer (not a Constant node)
    # must also count as constant.
    shape = onnx.numpy_helper.from_array(np.array([6], dtype=np.int64), "shape")
    model = _model(
        """
        g (float[2,3] x) => (float[6] y)
        {
          y = Reshape(x, shape)
        }
        """,
        [shape],
    )
    assert onnxsim.check_webnn_support(model) == []


def test_check_webnn_support_accepts_file_path(tmp_path):
    model = _model(
        """
        g (int64[2,3] x) => (int64[2,3] y)
        {
          y = Identity(x)
        }
        """
    )
    path = str(tmp_path / "model.onnx")
    onnx.save(model, path)
    messages = onnxsim.check_webnn_support(path)
    assert len(messages) == 2


def test_gemm_fusion_backend_webnn_matches_unrestricted():
    # Same FP16 MatMul+Add -> Gemm probe as the webgpu test: "webnn" should
    # behave exactly like "unrestricted", unlike the "ort_cpu" default.
    B, K, N = 2, 8, 8
    rng = np.random.default_rng(0)
    w = onnx.numpy_helper.from_array(
        (rng.standard_normal((K, N)) * 0.1).astype(np.float16), "w"
    )
    b = onnx.numpy_helper.from_array(
        (rng.standard_normal(N) * 0.1).astype(np.float16), "b"
    )
    model = _model(
        f"""
        g (float16[{B},{K}] x) => (float16[{B},{N}] y)
        {{
          mm = MatMul(x, w)
          y = Add(mm, b)
        }}
        """,
        [w, b],
        opset=17,
    )

    model_unrestricted, _ = onnxsim.simplify(
        model, gemm_fusion_backend="unrestricted"
    )
    model_webnn, _ = onnxsim.simplify(model, gemm_fusion_backend="webnn")

    ops_unrestricted = {n.op_type for n in model_unrestricted.graph.node}
    ops_webnn = {n.op_type for n in model_webnn.graph.node}

    assert "Gemm" in ops_unrestricted
    assert ops_webnn == ops_unrestricted
