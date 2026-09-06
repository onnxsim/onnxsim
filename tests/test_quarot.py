"""Tests for ``onnxsim.apply_quarot``/``onnxsim.apply_quarot_gptq`` -- see
``onnxsim/quarot.py`` for the technique (random-rotation preprocessing,
reused from ``onnxsim.quip_sharp``, plus INT4 quantization of *both* the
weight and the activation). ``apply_quarot`` needs no calibration data
(plain round-to-nearest for the weight); ``apply_quarot_gptq`` is the same
scheme but with the weight quantized via ``onnxsim.gptq``'s Hessian-based
column algorithm, using real calibration activations captured in the
*rotated* space.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21, ir_version=10):
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


def _matmul_model(K=32, N=8, weight=None, seed=0, opset=21):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
        opset=opset,
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


def _full_rank_calibration(K, num_samples=96, seed=1):
    # A plain full-rank Gaussian, not a low-rank-plus-noise signal: keeps
    # GPTQ's own Hessian (H = X^T X) well-conditioned, matching
    # tests/test_gptaq.py's own ``_full_rank_calibration`` (see that
    # module's docstring for why: a low-rank calibration signal makes the
    # shared Cholesky factorization sensitive to which BLAS/LAPACK backend
    # a given platform uses, risking a cross-platform tie in a
    # margin-based comparison).
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_samples, K)).astype(np.float32)


def test_quarot_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=0)
    q = onnxsim.apply_quarot(model, block_size=8, seed=1)
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("MatMul") == 2  # X rotation, core
    assert "DequantizeLinear" in op_types
    assert "ReduceMax" in op_types  # data-free per-token activation scale

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    # A generous bound: both operands are INT4 here (unlike every other
    # onnxsim weight-only scheme, which leaves the activation in float),
    # so this is a strictly harder target than e.g. SpinQuant's own 0.3.
    assert _rel_l2(float_y, q_y) < 0.5


def test_quarot_needs_no_calibration_data():
    # The whole point of a random (vs. fit) rotation: no calibration_data
    # kwarg exists at all, unlike apply_spinquant/apply_smoothquant/apply_awq.
    import inspect

    sig = inspect.signature(onnxsim.apply_quarot)
    assert "calibration_data" not in sig.parameters


def test_quarot_rotation_matrix_is_orthogonal():
    model = _matmul_model(K=16, N=4, seed=3)
    q = onnxsim.apply_quarot(model, block_size=4, seed=4)
    u_init = next(t for t in q.graph.initializer if t.name.endswith("_quarot_u"))
    u = onnx.numpy_helper.to_array(u_init).astype(np.float64)
    assert np.allclose(u @ u.T, np.eye(u.shape[0]), atol=1e-4)


def test_quarot_gemm_transb_with_bias():
    rng = np.random.default_rng(5)
    K, N = 32, 8
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB=1>(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.apply_quarot(model, block_size=8, seed=6)
    onnx.checker.check_model(q)
    assert "Add" in [n.op_type for n in q.graph.node]

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.5


def test_quarot_declines_when_k_not_divisible_by_block_size():
    model = _matmul_model(K=20, N=4, seed=7)  # 20 is not a multiple of 8
    q = onnxsim.apply_quarot(model, block_size=8)
    assert q.SerializeToString() == model.SerializeToString()


def test_quarot_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.apply_quarot(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_quarot_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_quarot(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_quarot_declines_below_opset21():
    model = _matmul_model(K=32, N=8, opset=13)
    result = onnxsim.apply_quarot(model)
    assert result.SerializeToString() == model.SerializeToString()


# ---------------------------------------------------------------------------
# apply_quarot_gptq
# ---------------------------------------------------------------------------


def test_quarot_gptq_rotation_matches_plain_quarot_for_same_seed():
    # For a given seed, apply_quarot_gptq must derive the exact same
    # per-layer rotation U as apply_quarot (it reuses that function's own
    # rotation-derivation loop unchanged) -- checked byte-for-byte here.
    model = _matmul_model(K=16, N=4, seed=3)
    x = _full_rank_calibration(K=16, num_samples=32, seed=9)
    calibration_data = [{"X": x}]

    q_rtn = onnxsim.apply_quarot(model, block_size=4, seed=4)
    q_gptq = onnxsim.apply_quarot_gptq(
        model, calibration_data=calibration_data, block_size=4, seed=4
    )

    u_rtn = next(t for t in q_rtn.graph.initializer if t.name.endswith("_quarot_u"))
    u_gptq = next(
        t for t in q_gptq.graph.initializer if t.name.endswith("_quarot_gptq_u")
    )
    assert u_rtn.raw_data == u_gptq.raw_data or np.array_equal(
        onnx.numpy_helper.to_array(u_rtn), onnx.numpy_helper.to_array(u_gptq)
    )


def test_quarot_gptq_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=0)
    x = _full_rank_calibration(K=64, num_samples=96, seed=100)
    calibration_data = [{"X": x}]

    q = onnxsim.apply_quarot_gptq(
        model, calibration_data=calibration_data, block_size=16, seed=1
    )
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("MatMul") == 2  # X rotation, core
    assert "DequantizeLinear" in op_types
    assert "ReduceMax" in op_types  # data-free per-token activation scale

    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_quarot_gptq_beats_plain_round_to_nearest_on_aggregate():
    # GPTQ's whole point is a tighter reconstruction error than
    # independent round-to-nearest on the same rotated weight/activation
    # pair. Empirically (see this session's verification harness) the
    # per-seed margin is real but modest (~5-10% relative L2, GPTQ always
    # at least as good, never worse, across 20 independent seeds tried
    # while writing this test) -- thin enough that trusting a single fixed
    # seed risks a cross-platform tie (the same concern
    # tests/test_gptaq.py's own comparison test flags and works around).
    # Aggregate the squared reconstruction error across several
    # independent weight/rotation/calibration seeds instead of trusting
    # one comparison.
    total_sq_rtn = 0.0
    total_sq_gptq = 0.0
    for seed in range(5):
        model = _matmul_model(K=64, N=16, seed=seed)
        x = _full_rank_calibration(K=64, num_samples=96, seed=seed + 100)
        calibration_data = [{"X": x}]

        q_rtn = onnxsim.apply_quarot(model, block_size=16, seed=seed + 1)
        q_gptq = onnxsim.apply_quarot_gptq(
            model, calibration_data=calibration_data, block_size=16, seed=seed + 1
        )
        onnx.checker.check_model(q_rtn)
        onnx.checker.check_model(q_gptq)

        (float_y,) = _run(model, {"X": x})
        (rtn_y,) = _run(q_rtn, {"X": x})
        (gptq_y,) = _run(q_gptq, {"X": x})
        assert np.all(np.isfinite(gptq_y))

        float_y64 = float_y.astype(np.float64)
        total_sq_rtn += float(np.sum((float_y64 - rtn_y) ** 2))
        total_sq_gptq += float(np.sum((float_y64 - gptq_y) ** 2))

    # A comfortable, empirically-verified margin (aggregate GPTQ error
    # comes in around 90-95% of aggregate RTN error across these seeds).
    assert total_sq_gptq < 0.98 * total_sq_rtn


def test_quarot_gptq_skips_layer_with_no_calibration_data():
    model = _matmul_model(K=32, N=8, seed=0)
    result = onnxsim.apply_quarot_gptq(model, calibration_data=[], block_size=8)
    # No calibration batch ever reached the candidate layer's own probe,
    # so it must be left completely untouched -- not silently quantized
    # via plain round-to-nearest under GPTQ's own name.
    assert result.SerializeToString() == model.SerializeToString()


def test_quarot_gptq_skips_layer_with_mismatched_activation_shape(monkeypatch):
    # A captured activation whose feature dimension doesn't match the
    # layer's own K must be skipped (mirrors onnxsim.gptq.apply_gptq's own
    # shape-mismatch skip) rather than used anyway.
    model = _matmul_model(K=32, N=8, seed=0)

    def fake_run_model(probe_model, batch, providers=None):
        # Feature dim 16 != K=32.
        return {"X": np.zeros((4, 16), dtype=np.float32)}

    monkeypatch.setattr(onnxsim.quarot.backend, "run_model", fake_run_model)
    result = onnxsim.apply_quarot_gptq(
        model,
        calibration_data=[{"X": np.zeros((1, 32), dtype=np.float32)}],
        block_size=8,
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_quarot_gptq_skips_layer_with_non_2d_activation(monkeypatch):
    model = _matmul_model(K=32, N=8, seed=0)

    def fake_run_model(probe_model, batch, providers=None):
        return {"X": np.zeros((4, 32, 2), dtype=np.float32)}  # 3-D, not 2-D

    monkeypatch.setattr(onnxsim.quarot.backend, "run_model", fake_run_model)
    result = onnxsim.apply_quarot_gptq(
        model,
        calibration_data=[{"X": np.zeros((1, 32), dtype=np.float32)}],
        block_size=8,
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_quarot_gptq_declines_when_k_not_divisible_by_block_size():
    model = _matmul_model(K=20, N=4, seed=7)  # 20 is not a multiple of 8
    q = onnxsim.apply_quarot_gptq(model, block_size=8)
    assert q.SerializeToString() == model.SerializeToString()


def test_quarot_gptq_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.apply_quarot_gptq(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_quarot_gptq_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_quarot_gptq(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_quarot_gptq_declines_below_opset21():
    model = _matmul_model(K=32, N=8, opset=13)
    result = onnxsim.apply_quarot_gptq(model)
    assert result.SerializeToString() == model.SerializeToString()
