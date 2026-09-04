"""Tests for RotateKV (see ``onnxsim/rotatekv.py``): outlier-aware,
per-stream rotation preprocessing for a decoder's KV-cache Key stream,
meant to run before ``onnxsim.quantize_kv_cache``.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=18, ir_version=9):
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


def _run(model, feeds, output_names=None):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = output_names or [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _attn_kv_cache_model(seq_past=3, seq_q=2, head_dim=8):
    # new_key is deliberately routed through an Identity node rather than
    # being a bare graph input -- same rationale as
    # tests/test_kv_cache_quantization.py's own _kv_cache_model: a real
    # exported decoder's "new" Key is always a freshly computed
    # activation, never a raw model input. present_key both feeds the
    # attention math directly (via one Transpose, to get head_dim into the
    # second-to-last axis QK^T needs) and is a graph output, per
    # onnxsim.kv_cache_quantization's own module docstring ("present_key
    # -- graph output, and consumed by the attention math").
    return _model(
        f"""
        g (float[{seq_past},{head_dim}] past_key,
           float[1,{head_dim}] new_key_raw,
           float[{seq_q},{head_dim}] Q,
           float[{seq_past + 1},{head_dim}] V)
          => (float[{seq_past + 1},{head_dim}] present_key,
              float[{seq_q},{head_dim}] Out)
        {{
          new_key = Identity(new_key_raw)
          present_key = Concat<axis = 0>(past_key, new_key)
          Kt = Transpose(present_key)
          scores = MatMul(Q, Kt)
          probs = Softmax<axis = -1>(scores)
          Out = MatMul(probs, V)
        }}
        """
    )


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _attn_kv_cache_model_dotted_name(seq_past=3, seq_q=2, head_dim=8):
    # Mirrors _attn_kv_cache_model, but named to match this repo's own
    # present.{i}.decoder.value-style convention (containing the literal
    # substring ".value") -- onnx.parser's text format has no syntax for
    # identifiers containing "." (see tests/test_kv_cache_quantization.py's
    # own _vi docstring), so this one stays on
    # onnx.helper.make_graph/make_model construction.
    present_name = "present.0.value"
    seq_present = seq_past + 1
    nodes = [
        onnx.helper.make_node("Identity", ["new_key_raw"], ["new_key"]),
        onnx.helper.make_node(
            "Concat", ["past_key", "new_key"], [present_name], axis=0
        ),
        onnx.helper.make_node("Transpose", [present_name], ["Kt"]),
        onnx.helper.make_node("MatMul", ["Q", "Kt"], ["scores"]),
        onnx.helper.make_node("Softmax", ["scores"], ["probs"], axis=-1),
        onnx.helper.make_node("MatMul", ["probs", "V"], ["Out"]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [
            _vi("past_key", [seq_past, head_dim]),
            _vi("new_key_raw", [1, head_dim]),
            _vi("Q", [seq_q, head_dim]),
            _vi("V", [seq_present, head_dim]),
        ],
        [_vi(present_name, [seq_present, head_dim]), _vi("Out", [seq_q, head_dim])],
        [],
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=9
    )


def _calibration(seq_past=3, head_dim=8, num_samples=16, seed=0):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(num_samples):
        batches.append(
            {
                "past_key": rng.standard_normal((seq_past, head_dim)).astype(
                    np.float32
                ),
                "new_key_raw": rng.standard_normal((1, head_dim)).astype(np.float32),
                "Q": rng.standard_normal((2, head_dim)).astype(np.float32),
                "V": rng.standard_normal((seq_past + 1, head_dim)).astype(np.float32),
            }
        )
    return batches


def test_rotatekv_inserts_rotation_matmuls_with_orthogonal_r():
    head_dim = 8
    model = _attn_kv_cache_model(head_dim=head_dim)
    calibration_data = _calibration(head_dim=head_dim)

    rk_model = onnxsim.apply_rotatekv(model, calibration_data=calibration_data)
    onnx.checker.check_model(rk_model)

    op_types = [n.op_type for n in rk_model.graph.node]
    # two new MatMul nodes (Key-side and Query-side rotation) on top of the
    # two original MatMul nodes (QK^T and probs@V).
    assert op_types.count("MatMul") == 4

    concat_node = next(n for n in rk_model.graph.node if n.op_type == "Concat")
    new_key_input = concat_node.input[1]
    new_key_matmul = next(
        n
        for n in rk_model.graph.node
        if n.op_type == "MatMul" and n.output[0] == new_key_input
    )
    r = onnx.numpy_helper.to_array(
        next(t for t in rk_model.graph.initializer if t.name == new_key_matmul.input[1])
    ).astype(np.float64)
    assert r.shape == (head_dim, head_dim)
    assert np.allclose(r.T @ r, np.eye(head_dim), atol=1e-4)

    qk_matmul = next(
        n
        for n in rk_model.graph.node
        if n.op_type == "MatMul" and n.output[0] == "scores"
    )
    assert qk_matmul.input[0] != "Q"
    q_matmul = next(
        n
        for n in rk_model.graph.node
        if n.op_type == "MatMul" and n.output[0] == qk_matmul.input[0]
    )
    assert q_matmul.input[0] == "Q"
    # Same rotation reused for both Key and Query -- see module docstring's
    # exactness argument.
    assert q_matmul.input[1] == new_key_matmul.input[1]


def test_rotatekv_exact_dot_product_identity():
    # The algebraic claim this module's docstring makes directly:
    # (Q @ R) @ (X @ R)^T == Q @ X^T for the fitted orthogonal R -- checked
    # in plain numpy against the model's own written R, independent of
    # onnxruntime, per this project's platform-numerics convention. Unlike
    # a diagonal-scale migration (where the scale cancels algebraically
    # regardless of its own rounding), this identity relies on R @ R.T
    # actually equalling the identity matrix -- which only holds up to
    # *R's own storage precision* (float32 here), not to float64 machine
    # epsilon, so the tolerance is set accordingly rather than at 1e-9.
    head_dim = 8
    model = _attn_kv_cache_model(seq_past=0, head_dim=head_dim)
    calibration_data = _calibration(seq_past=0, head_dim=head_dim, seed=1)

    rk_model = onnxsim.apply_rotatekv(model, calibration_data=calibration_data)

    r_init = next(t for t in rk_model.graph.initializer if "_rotatekv_r" in t.name)
    r = onnx.numpy_helper.to_array(r_init).astype(np.float64)

    feed = calibration_data[0]
    q = feed["Q"].astype(np.float64)
    new_key = feed["new_key_raw"].astype(np.float64)  # seq_past=0 -> present == new

    original = q @ new_key.T
    migrated = (q @ r) @ (new_key @ r).T
    assert np.allclose(original, migrated, rtol=1e-6, atol=1e-6)


def test_rotatekv_output_matches_float_via_onnxruntime():
    # Loose, onnxruntime-based end-to-end sanity check -- kept separate
    # from (and much looser than) the numpy-exact check above, per this
    # project's platform-numerics convention (onnxruntime's own execution
    # is not bit-exact across CPU architectures).
    head_dim = 8
    model = _attn_kv_cache_model(seq_past=0, head_dim=head_dim)
    calibration_data = _calibration(seq_past=0, head_dim=head_dim, seed=2)

    rk_model = onnxsim.apply_rotatekv(model, calibration_data=calibration_data)

    feed = calibration_data[0]
    float_out = _run(model, feed, output_names=["Out"])
    rk_out = _run(rk_model, feed, output_names=["Out"])
    assert np.all(np.isfinite(rk_out["Out"]))
    assert _rel_l2(float_out["Out"], rk_out["Out"]) < 1e-2


def test_rotatekv_then_quantize_kv_cache_reuses_existing_quantizer():
    # This module's own contribution is only the rotation migration --
    # verifies the intended pipeline usage, feeding the rotated model into
    # onnxsim.quantize_kv_cache rather than reimplementing quantization.
    head_dim = 8
    model = _attn_kv_cache_model(head_dim=head_dim)
    calibration_data = _calibration(head_dim=head_dim, seed=3)

    rk_model = onnxsim.apply_rotatekv(model, calibration_data=calibration_data)
    quantized = onnxsim.quantize_kv_cache(rk_model, calibration_data=calibration_data)
    onnx.checker.check_model(quantized)

    past_key_input = next(i for i in quantized.graph.input if i.name == "past_key")
    assert past_key_input.type.tensor_type.elem_type == onnx.TensorProto.INT8
    present_key_output = next(
        o for o in quantized.graph.output if o.name == "present_key"
    )
    assert present_key_output.type.tensor_type.elem_type == onnx.TensorProto.INT8
    # The Concat's own "new" operand should be the rotated Key, not the
    # original -- i.e. quantize_kv_cache quantized RotateKV's own output.
    concat_node = next(n for n in quantized.graph.node if n.op_type == "Concat")
    assert "_rotatekv" in concat_node.input[1] or "_kv_q" in concat_node.input[1]


def test_rotatekv_declines_stream_with_no_attention_consumer():
    # A Key-style KV-cache stream whose present output is never consumed
    # by any QK^T MatMul -- rotating it alone (with no way to compensate
    # Query) would silently change attention scores, so this module
    # declines rather than guess.
    model = _model(
        """
        g (float[3,8] past_key, float[1,8] new_key_raw)
          => (float[4,8] present_key, float summary)
        {
          new_key = Identity(new_key_raw)
          present_key = Concat<axis = 0>(past_key, new_key)
          summary = ReduceSum<keepdims = 0>(present_key)
        }
        """
    )
    result = onnxsim.apply_rotatekv(
        model,
        calibration_data=[
            {
                "past_key": np.zeros((3, 8), dtype=np.float32),
                "new_key_raw": np.zeros((1, 8), dtype=np.float32),
            }
        ],
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_rotatekv_skips_value_style_stream():
    model = _attn_kv_cache_model_dotted_name(head_dim=8)
    calibration_data = _calibration(head_dim=8, seed=4)
    result = onnxsim.apply_rotatekv(model, calibration_data=calibration_data)
    assert result.SerializeToString() == model.SerializeToString()


def test_rotatekv_noop_when_no_kv_cache_pattern():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_rotatekv(
        model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
