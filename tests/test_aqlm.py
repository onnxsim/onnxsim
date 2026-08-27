"""Tests for ``onnxsim.quantize_weight_only_aqlm`` (AQLM, see
``onnxsim/aqlm.py``) -- greedy residual k-means fits ``M`` codebooks
shared across every group in a layer, each group represented as the sum
of one lookup per codebook.
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _model(nodes, inputs, outputs, initializer, opset=13):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=8
    )


def _matmul_model(K=64, N=16, weight=None, seed=0):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    return _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _dequantize_aqlm_by_hand(model, w_name, num_codebooks):
    """Independent reference decode: reads the codebook/codes initializers
    directly and reconstructs via numpy, without using any of this
    module's own internal functions or the ops it inserts. Returns the
    ``[num_groups, group_dim]`` sum of all additive stages -- reshaping
    back to the weight's own ``[N, K]``/original layout is the caller's
    job, since only the caller knows which.
    """
    stages = []
    for m in range(num_codebooks):
        codebook = onnx.numpy_helper.to_array(
            next(
                t
                for t in model.graph.initializer
                if t.name == f"{w_name}_aqlm_codebook{m}"
            )
        ).astype(np.float64)
        codes = onnx.numpy_helper.to_array(
            next(
                t
                for t in model.graph.initializer
                if t.name == f"{w_name}_aqlm_codes{m}"
            )
        )
        stages.append(codebook[codes])
    return sum(stages)


def test_aqlm_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=0)
    q = onnxsim.quantize_weight_only_aqlm(model, group_dim=8, seed=0)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert "Gather" in op_types
    assert "Add" in op_types

    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_aqlm_dequantized_values_match_hand_decoded_reference():
    K, N = 32, 8
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _matmul_model(K=K, N=N, weight=weight)
    q = onnxsim.quantize_weight_only_aqlm(model, group_dim=8, num_codebooks=2, seed=1)

    combined = _dequantize_aqlm_by_hand(q, "W", num_codebooks=2)
    w_hand_nk = combined.reshape(N, K)
    w_hand = w_hand_nk.T  # back to original [K, N] storage

    matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
    weight_tensor_name = matmul_node.input[1]
    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(q)
    probe_model.graph.output.append(onnx.ValueInfoProto(name=weight_tensor_name))
    rng2 = np.random.default_rng(3)
    x = rng2.standard_normal((4, K)).astype(np.float32)
    (w_graph,) = _run(probe_model, {"X": x})[len(q.graph.output) :]

    assert np.allclose(w_hand, w_graph.astype(np.float64), rtol=0, atol=1e-5)


def test_aqlm_more_codebooks_never_increases_error():
    # Each additive stage fits its own codebook to exactly the residual
    # the previous stages left over, so adding more codebooks can only
    # reduce (never increase) the layer's own reconstruction error.
    K, N = 32, 8
    rng = np.random.default_rng(4)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(K=K, N=N, weight=weight)
    w_float = weight.astype(np.float64)

    errors = []
    for m in (1, 2, 3):
        q = onnxsim.quantize_weight_only_aqlm(
            model, group_dim=8, num_codebooks=m, codebook_size=8, seed=5
        )
        matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
        weight_tensor_name = matmul_node.input[1]
        probe_model = onnx.ModelProto()
        probe_model.CopyFrom(q)
        probe_model.graph.output.append(onnx.ValueInfoProto(name=weight_tensor_name))
        rng2 = np.random.default_rng(6)
        x = rng2.standard_normal((4, K)).astype(np.float32)
        (w_graph,) = _run(probe_model, {"X": x})[len(q.graph.output) :]
        errors.append(np.sum((w_graph.astype(np.float64) - w_float) ** 2))

    assert all(errors[i] >= errors[i + 1] - 1e-6 for i in range(len(errors) - 1))


def test_aqlm_gemm_transb():
    rng = np.random.default_rng(7)
    K, N = 64, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"], transB=1)]
    model = _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )
    q = onnxsim.quantize_weight_only_aqlm(model, group_dim=8, seed=2)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_aqlm_codes_stay_in_codebook_range():
    model = _matmul_model(K=32, N=8, seed=8)
    q = onnxsim.quantize_weight_only_aqlm(
        model, group_dim=8, num_codebooks=2, codebook_size=16, seed=3
    )
    for m in range(2):
        codes = onnx.numpy_helper.to_array(
            next(t for t in q.graph.initializer if t.name == f"W_aqlm_codes{m}")
        )
        assert np.all(codes >= 0) and np.all(codes < 16)


def test_aqlm_skips_non_group_divisible_k():
    model = _matmul_model(K=20, N=4, seed=9)  # 20 is not a multiple of 8
    q = onnxsim.quantize_weight_only_aqlm(model, group_dim=8)
    assert q.SerializeToString() == model.SerializeToString()


def test_aqlm_skips_non_constant_weight():
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes, [_vi("X", [4, 32]), _vi("W", [32, 4])], [_vi("Y", [4, 4])], []
    )
    q = onnxsim.quantize_weight_only_aqlm(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_aqlm_noop_when_no_matmul_present():
    nodes = [onnx.helper.make_node("Relu", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 4])], [_vi("Y", [4, 4])], [])
    result = onnxsim.quantize_weight_only_aqlm(model)
    assert result.SerializeToString() == model.SerializeToString()
