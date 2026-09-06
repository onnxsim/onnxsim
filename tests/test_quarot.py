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


# --- apply_quarot_fused ------------------------------------------------
# The same rotation, folded into the producing layer's weight so it costs
# nothing at inference. See onnxsim/quarot.py's own docstring.


def _u_initializers(model):
    return [t.name for t in model.graph.initializer if t.name.endswith("_quarot_u")]


def _two_layer_model(K=32, M=32, N=8, seed=0):
    rng = np.random.default_rng(seed)
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          T = MatMul(X, W1)
          Y = MatMul(T, W2)
        }}
        """,
        initializer=[
            _f32(rng.standard_normal((K, M)) * 0.5, "W1"),
            _f32(rng.standard_normal((M, N)) * 0.5, "W2"),
        ],
    )


def _three_layer_model(K=32, M=32, P=32, N=8, seed=0):
    rng = np.random.default_rng(seed)
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          A = MatMul(X, W1)
          B = MatMul(A, W2)
          Y = MatMul(B, W3)
        }}
        """,
        initializer=[
            _f32(rng.standard_normal((K, M)) * 0.5, "W1"),
            _f32(rng.standard_normal((M, P)) * 0.5, "W2"),
            _f32(rng.standard_normal((P, N)) * 0.5, "W3"),
        ],
    )


def test_quarot_fused_emits_no_rotation_node_for_the_fused_layer():
    model = _two_layer_model()
    fused = onnxsim.apply_quarot_fused(model, block_size=8, seed=1)
    onnx.checker.check_model(fused)

    # Both layers are quantized, but only the first -- whose activation is
    # the graph input X -- still needs a runtime MatMul(X, U). The second
    # layer's rotation is folded into W1, so it has no U at all.
    assert _u_initializers(fused) == ["W1_quarot_u"]
    names = {t.name for t in fused.graph.initializer}
    assert "W2_quarot_codes" in names  # ...but it is still quantized


def test_quarot_fused_has_strictly_fewer_nodes_than_apply_quarot():
    model = _two_layer_model()
    plain = onnxsim.apply_quarot(model, block_size=8, seed=1)
    fused = onnxsim.apply_quarot_fused(model, block_size=8, seed=1)
    # Exactly the one fused layer's MatMul(X, U) is gone.
    assert len(fused.graph.node) == len(plain.graph.node) - 1
    plain_ops = [n.op_type for n in plain.graph.node]
    fused_ops = [n.op_type for n in fused.graph.node]
    assert fused_ops.count("MatMul") == plain_ops.count("MatMul") - 1
    assert len(_u_initializers(fused)) == len(_u_initializers(plain)) - 1


def test_quarot_fused_is_about_as_accurate_as_apply_quarot():
    # The fold is exact algebra, so the only lossy steps are the same two
    # INT4 roundings apply_quarot already does -- but the rotation now
    # happens inside a different float matmul, so the two are not
    # bit-identical. Aggregate across seeds rather than betting on one.
    plain_errors = []
    fused_errors = []
    for s in range(6):
        model = _two_layer_model(seed=s)
        plain = onnxsim.apply_quarot(model, block_size=8, seed=s + 1)
        fused = onnxsim.apply_quarot_fused(model, block_size=8, seed=s + 1)
        x = np.random.default_rng(500 + s).standard_normal((16, 32)).astype(np.float32)
        (float_y,) = _run(model, {"X": x})
        (plain_y,) = _run(plain, {"X": x})
        (fused_y,) = _run(fused, {"X": x})
        assert np.all(np.isfinite(fused_y))
        plain_errors.append(_rel_l2(float_y, plain_y))
        fused_errors.append(_rel_l2(float_y, fused_y))

    # Verified from real runs: both land around 0.17 mean, 0.23 max here.
    assert np.mean(fused_errors) < 0.4
    # Generous slack -- a layout/axis bug in the fold shows up as a
    # several-times-worse error, not as a few percent.
    assert np.mean(fused_errors) < 1.5 * np.mean(plain_errors) + 0.05


def test_quarot_fused_chained_three_layers_round_trips():
    # A -> B -> C: the middle layer is both a fused producer (its output
    # dim carries C's rotation) and a quantized consumer (its reduction
    # dim carries its own). Both rotations must be applied while the
    # weight is still float; getting either axis wrong wrecks the numbers,
    # not the shapes, so this has to be a real numeric check.
    plain_errors = []
    fused_errors = []
    for s in range(6):
        model = _three_layer_model(seed=s)
        plain = onnxsim.apply_quarot(model, block_size=8, seed=s + 1)
        fused = onnxsim.apply_quarot_fused(model, block_size=8, seed=s + 1)
        onnx.checker.check_model(fused)
        # Only the first layer, fed by the graph input, keeps a runtime U.
        assert _u_initializers(fused) == ["W1_quarot_u"]
        assert len(fused.graph.node) == len(plain.graph.node) - 2

        x = np.random.default_rng(700 + s).standard_normal((16, 32)).astype(np.float32)
        (float_y,) = _run(model, {"X": x})
        (plain_y,) = _run(plain, {"X": x})
        (fused_y,) = _run(fused, {"X": x})
        assert np.all(np.isfinite(fused_y))
        plain_errors.append(_rel_l2(float_y, plain_y))
        fused_errors.append(_rel_l2(float_y, fused_y))

    # Verified from real runs: both land around 0.23 mean, 0.28 max here.
    assert np.mean(fused_errors) < 0.5
    assert np.mean(fused_errors) < 1.5 * np.mean(plain_errors) + 0.05


def test_quarot_fused_gemm_producer_folds_the_bias_too():
    # b_P' = b_P @ U, on the bias's output-channel axis. Left unrotated,
    # the layer would emit T @ U + b_P instead of (T + b_P) @ U, which is
    # visibly wrong rather than slightly noisy.
    rng = np.random.default_rng(21)
    K, M, N = 32, 32, 8
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          T = Gemm<transB=1>(X, W1, B1)
          Y = Gemm(T, W2, B2)
        }}
        """,
        initializer=[
            _f32(rng.standard_normal((M, K)) * 0.5, "W1"),
            _f32(rng.standard_normal((M,)) * 0.5, "B1"),
            _f32(rng.standard_normal((M, N)) * 0.5, "W2"),
            _f32(rng.standard_normal((N,)) * 0.1, "B2"),
        ],
    )
    fused = onnxsim.apply_quarot_fused(model, block_size=8, seed=2)
    onnx.checker.check_model(fused)
    assert _u_initializers(fused) == ["W1_quarot_u"]
    assert "B1_quarot_folded" in {t.name for t in fused.graph.initializer}

    x = rng.standard_normal((16, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (fused_y,) = _run(fused, {"X": x})
    assert _rel_l2(float_y, fused_y) < 0.5


def test_quarot_fused_folds_into_a_producer_it_does_not_quantize():
    # The producer's own K=20 is not divisible by block_size 8, so it is
    # left in float -- but its weight is still constant, so the consumer's
    # rotation folds into it exactly and its rotated float weight is
    # simply written back under a fresh name.
    rng = np.random.default_rng(22)
    model = _model(
        """
        g (float[batch,20] X) => (float[batch,8] Y)
        {
          T = MatMul(X, W1)
          Y = MatMul(T, W2)
        }
        """,
        initializer=[
            _f32(rng.standard_normal((20, 32)) * 0.5, "W1"),
            _f32(rng.standard_normal((32, 8)) * 0.5, "W2"),
        ],
    )
    fused = onnxsim.apply_quarot_fused(model, block_size=8, seed=2)
    onnx.checker.check_model(fused)
    assert _u_initializers(fused) == []
    assert "W1_quarot_folded" in {t.name for t in fused.graph.initializer}

    x = rng.standard_normal((16, 20)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (fused_y,) = _run(fused, {"X": x})
    assert _rel_l2(float_y, fused_y) < 0.5


def test_quarot_fused_declines_when_activation_has_two_consumers():
    # Rotating T for one consumer would corrupt the other, so both layers
    # fall back to apply_quarot's explicit runtime rotation -- which is
    # exactly what apply_quarot itself produces here, byte for byte.
    rng = np.random.default_rng(23)
    K, M, N = 32, 32, 8
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          T = MatMul(X, W1)
          Y1 = MatMul(T, W2)
          Y2 = MatMul(T, W3)
          Y = Add(Y1, Y2)
        }}
        """,
        initializer=[
            _f32(rng.standard_normal((K, M)) * 0.5, "W1"),
            _f32(rng.standard_normal((M, N)) * 0.5, "W2"),
            _f32(rng.standard_normal((M, N)) * 0.5, "W3"),
        ],
    )
    plain = onnxsim.apply_quarot(model, block_size=8, seed=3)
    fused = onnxsim.apply_quarot_fused(model, block_size=8, seed=3)
    assert len(_u_initializers(fused)) == 3
    assert fused.SerializeToString() == plain.SerializeToString()


def test_quarot_fused_declines_when_producer_output_is_a_graph_output():
    # T must keep its unrotated value for whoever reads it.
    rng = np.random.default_rng(24)
    model = _model(
        """
        g (float[batch,32] X) => (float[batch,32] T, float[batch,8] Y)
        {
          T = MatMul(X, W1)
          Y = MatMul(T, W2)
        }
        """,
        initializer=[
            _f32(rng.standard_normal((32, 32)) * 0.5, "W1"),
            _f32(rng.standard_normal((32, 8)) * 0.5, "W2"),
        ],
    )
    plain = onnxsim.apply_quarot(model, block_size=8, seed=3)
    fused = onnxsim.apply_quarot_fused(model, block_size=8, seed=3)
    assert len(_u_initializers(fused)) == 2
    assert fused.SerializeToString() == plain.SerializeToString()


def test_quarot_fused_declines_non_constant_producer_weight():
    # Nothing to fold into: W1 is a graph input, not an initializer.
    rng = np.random.default_rng(25)
    model = _model(
        """
        g (float[batch,32] X, float[32,32] W1) => (float[batch,8] Y)
        {
          T = MatMul(X, W1)
          Y = MatMul(T, W2)
        }
        """,
        initializer=[_f32(rng.standard_normal((32, 8)) * 0.5, "W2")],
    )
    plain = onnxsim.apply_quarot(model, block_size=8, seed=3)
    fused = onnxsim.apply_quarot_fused(model, block_size=8, seed=3)
    assert _u_initializers(fused) == ["W2_quarot_u"]
    assert fused.SerializeToString() == plain.SerializeToString()


def test_quarot_fused_declines_broadcast_gemm_bias():
    # A scalar Gemm bias has no output-channel axis for b @ U to act on.
    rng = np.random.default_rng(26)
    model = _model(
        """
        g (float[batch,32] X) => (float[batch,8] Y)
        {
          T = Gemm(X, W1, B1)
          Y = MatMul(T, W2)
        }
        """,
        initializer=[
            _f32(rng.standard_normal((32, 32)) * 0.5, "W1"),
            _f32(np.array(0.25, dtype=np.float32), "B1"),
            _f32(rng.standard_normal((32, 8)) * 0.5, "W2"),
        ],
    )
    plain = onnxsim.apply_quarot(model, block_size=8, seed=3)
    fused = onnxsim.apply_quarot_fused(model, block_size=8, seed=3)
    assert len(_u_initializers(fused)) == 2
    assert fused.SerializeToString() == plain.SerializeToString()


def test_quarot_fused_leaves_a_shared_producer_weight_intact():
    # W1 feeds two nodes; only the one whose output is fused gets the
    # rotated copy, under a fresh name. The other must stay exact.
    rng = np.random.default_rng(27)
    model = _model(
        """
        g (float[batch,20] X) => (float[batch,8] Y, float[batch,32] Z)
        {
          T = MatMul(X, W1)
          Z = MatMul(X, W1)
          Y = MatMul(T, W2)
        }
        """,
        initializer=[
            _f32(rng.standard_normal((20, 32)) * 0.5, "W1"),
            _f32(rng.standard_normal((32, 8)) * 0.5, "W2"),
        ],
    )
    fused = onnxsim.apply_quarot_fused(model, block_size=8, seed=3)
    onnx.checker.check_model(fused)
    x = rng.standard_normal((16, 20)).astype(np.float32)
    float_y, float_z = _run(model, {"X": x})
    fused_y, fused_z = _run(fused, {"X": x})
    assert np.allclose(float_z, fused_z, atol=1e-5)
    assert _rel_l2(float_y, fused_y) < 0.5


def test_quarot_fused_matches_apply_quarot_when_nothing_is_fusable():
    model = _matmul_model(K=32, N=8, seed=8)
    plain = onnxsim.apply_quarot(model, block_size=8, seed=9)
    fused = onnxsim.apply_quarot_fused(model, block_size=8, seed=9)
    assert fused.SerializeToString() == plain.SerializeToString()


def test_quarot_fused_declines_when_k_not_divisible_by_block_size():
    model = _matmul_model(K=20, N=4, seed=10)
    fused = onnxsim.apply_quarot_fused(model, block_size=8)
    assert fused.SerializeToString() == model.SerializeToString()


def test_quarot_fused_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    fused = onnxsim.apply_quarot_fused(model)
    assert fused.SerializeToString() == model.SerializeToString()


def test_quarot_fused_declines_below_opset21():
    model = _matmul_model(K=32, N=8, opset=13)
    fused = onnxsim.apply_quarot_fused(model)
    assert fused.SerializeToString() == model.SerializeToString()
