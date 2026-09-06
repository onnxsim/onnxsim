"""Tests for ``onnxsim.compute_activation_importance``/
``onnxsim.apply_imatrix_quantization`` (see ``onnxsim/imatrix_quant.py``) --
llama.cpp's activation-based "importance matrix" (``imatrix``) applied to
this repository's own plain block-wise INT4 weight quantizer.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.imatrix_quant import (
    quantize_dequantize_int4_imatrix,
    quantize_dequantize_int4_plain,
)

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


def _matmul_model(w, K, N, batch="batch", opset=21):
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
        opset=opset,
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _current_weight(model, weight_input_index=1):
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


# ---------------------------------------------------------------------------
# quantize_dequantize_int4_imatrix -- pure-numpy mechanism tests
# ---------------------------------------------------------------------------


def test_uniform_importance_is_never_worse_than_plain_baseline():
    # The plain baseline's own scale (factor 1.0) is always one of the grid
    # search's own candidates, so with every channel equally important
    # (weighted SSE == plain unweighted SSE), the search can only match or
    # improve on the plain baseline's own SSE -- never do worse. It need
    # NOT reproduce the plain baseline exactly: plain min/max scaling is not
    # itself unweighted-SSE-optimal in general (clipping a mild outlier
    # slightly can reduce total squared error), so the search legitimately
    # finds a strictly better scale on some blocks of generic data, as it
    # does here.
    rng = np.random.default_rng(0)
    w_nk = rng.standard_normal((6, 64))
    uniform_importance = np.ones(64)

    weighted = quantize_dequantize_int4_imatrix(w_nk, uniform_importance, block_size=32)
    plain = quantize_dequantize_int4_plain(w_nk, block_size=32)
    sse_weighted = float(np.sum((w_nk - weighted) ** 2))
    sse_plain = float(np.sum((w_nk - plain) ** 2))
    assert sse_weighted <= sse_plain + 1e-9


def test_single_candidate_grid_reproduces_plain_baseline_exactly():
    # A degenerate one-candidate grid pinned at factor 1.0 has no freedom to
    # deviate from the plain per-block scale at all -- this is the actual
    # "reduces to exactly the plain baseline" case, isolating the grid
    # search itself (rather than importance weighting) as the source of any
    # difference from plain quantization.
    rng = np.random.default_rng(0)
    w_nk = rng.standard_normal((6, 64))
    uniform_importance = np.ones(64)

    weighted = quantize_dequantize_int4_imatrix(
        w_nk,
        uniform_importance,
        block_size=32,
        num_scale_candidates=1,
        scale_search_range=(1.0, 1.0),
    )
    plain = quantize_dequantize_int4_plain(w_nk, block_size=32)
    np.testing.assert_allclose(weighted, plain)


def test_imatrix_weighting_reduces_weighted_reconstruction_error():
    # The core empirical claim: on a block where one channel has a huge
    # weight magnitude (so it dominates the plain min/max scale, forcing a
    # coarse grid) but is rarely activated (low importance), while the rest
    # of the block's channels are activated often (high importance) but
    # have smaller weight magnitude, importance-weighted quantization should
    # trade accuracy on the unimportant outlier for accuracy on the
    # important channels -- lowering the *importance-weighted* squared
    # error, exactly llama.cpp's imatrix quantizers' own objective.
    rng = np.random.default_rng(1)
    block_size = 32
    n, k = 4, block_size
    w_nk = rng.standard_normal((n, k)) * 0.5
    outlier = 5
    w_nk[:, outlier] = rng.choice([-1.0, 1.0], n) * 12.0

    importance = np.full(k, 5.0)
    importance[outlier] = 0.01

    plain = quantize_dequantize_int4_plain(w_nk, block_size=block_size)
    weighted = quantize_dequantize_int4_imatrix(w_nk, importance, block_size=block_size)

    plain_weighted_sse = float(np.sum(importance[np.newaxis, :] * (w_nk - plain) ** 2))
    imatrix_weighted_sse = float(
        np.sum(importance[np.newaxis, :] * (w_nk - weighted) ** 2)
    )
    assert imatrix_weighted_sse < plain_weighted_sse

    # The trade-off is real, not free: the plain (unweighted) SSE should get
    # *worse* under the weighted scheme, since it's now spending resolution
    # on the important channels instead of the (unweighted-SSE-dominating)
    # outlier.
    plain_unweighted_sse = float(np.sum((w_nk - plain) ** 2))
    imatrix_unweighted_sse = float(np.sum((w_nk - weighted) ** 2))
    assert imatrix_unweighted_sse > plain_unweighted_sse


def test_quantize_dequantize_int4_imatrix_rejects_bad_shapes():
    w_nk = np.zeros((2, 40))
    with pytest.raises(ValueError):
        quantize_dequantize_int4_imatrix(w_nk, np.zeros(40), block_size=32)
    with pytest.raises(ValueError):
        quantize_dequantize_int4_imatrix(np.zeros((2, 32)), np.zeros(31), block_size=32)


# ---------------------------------------------------------------------------
# compute_activation_importance
# ---------------------------------------------------------------------------


def test_compute_activation_importance_matches_manual_mean_square():
    rng = np.random.default_rng(2)
    K, N = 16, 8
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N, batch="batch")

    batches = [
        {"X": rng.standard_normal((5, K)).astype(np.float32)},
        {"X": rng.standard_normal((3, K)).astype(np.float32)},
    ]
    importance = onnxsim.compute_activation_importance(model, calibration_data=batches)

    all_x = np.concatenate([b["X"] for b in batches], axis=0).astype(np.float64)
    expected = np.mean(all_x**2, axis=0)
    np.testing.assert_allclose(importance["X"], expected, rtol=1e-6)


def test_compute_activation_importance_flags_low_variance_channel():
    rng = np.random.default_rng(3)
    K, N = 32, 4
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)

    low_var_channel = 7
    batches = []
    for _ in range(6):
        x = rng.standard_normal((10, K)).astype(np.float32)
        x[:, low_var_channel] = rng.standard_normal(10).astype(np.float32) * 0.001
        batches.append({"X": x})

    importance = onnxsim.compute_activation_importance(model, calibration_data=batches)
    assert importance["X"][low_var_channel] < 1e-4
    other = np.delete(importance["X"], low_var_channel)
    assert np.median(other) > 0.5


# ---------------------------------------------------------------------------
# apply_imatrix_quantization -- end-to-end graph transform
# ---------------------------------------------------------------------------


def test_apply_imatrix_quantization_replaces_weight():
    rng = np.random.default_rng(4)
    K, N = 64, 8
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_imatrix_quantization(model, num_samples=4, seed=0)
    onnx.checker.check_model(q)

    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert not np.array_equal(new_w, w)


def test_apply_imatrix_quantization_output_stays_close_to_float_via_onnxruntime():
    rng = np.random.default_rng(5)
    K, N = 64, 16
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_imatrix_quantization(model, num_samples=8, seed=0)
    onnx.checker.check_model(q)

    x = rng.standard_normal((6, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    rel_l2 = np.linalg.norm(float_y - q_y) / max(np.linalg.norm(float_y), 1e-9)
    assert rel_l2 < 0.5


def test_apply_imatrix_quantization_gemm_with_bias():
    rng = np.random.default_rng(6)
    K, N = 64, 8
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.4
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.apply_imatrix_quantization(model, num_samples=4, seed=0)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))


def test_apply_imatrix_quantization_respects_skip_names():
    rng = np.random.default_rng(7)
    K, N = 64, 8
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_imatrix_quantization(model, num_samples=4, skip_names={"W"})
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_imatrix_quantization_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_imatrix_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_imatrix_quantization_skips_reduction_dim_not_divisible_by_block_size():
    rng = np.random.default_rng(8)
    K, N = 50, 8  # not a multiple of block_size=32
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_imatrix_quantization(model, num_samples=4)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_imatrix_quantization_beats_plain_quantization_on_output_fidelity():
    # End-to-end version of the core empirical claim: on a layer whose
    # calibration activations make one weight-outlier input channel
    # unimportant (near-zero variance) while the rest of the channels carry
    # real signal, quantizing with the importance matrix should reproduce
    # the float model's real output more faithfully than the plain
    # (unweighted) baseline, on fresh (non-calibration) inputs drawn from
    # the same distribution.
    rng = np.random.default_rng(9)
    K, N, block_size = 64, 8, 32
    outlier_channel = 5

    w = rng.standard_normal((K, N)).astype(np.float64) * 0.3
    w[outlier_channel, :] += rng.choice([-1.0, 1.0], N) * 12.0
    w = w.astype(np.float32)
    model = _matmul_model(w, K, N)

    def _make_batch(num_rows, seed_offset):
        r = np.random.default_rng(100 + seed_offset)
        x = r.standard_normal((num_rows, K)).astype(np.float32)
        x[:, outlier_channel] = (r.standard_normal(num_rows) * 0.02).astype(np.float32)
        return {"X": x}

    calibration_data = [_make_batch(16, i) for i in range(8)]

    q_imatrix = onnxsim.apply_imatrix_quantization(
        model, calibration_data=calibration_data, block_size=block_size
    )
    w_plain_dequant = quantize_dequantize_int4_plain(w.astype(np.float64).T, block_size)
    plain_model = _matmul_model(w_plain_dequant.T.astype(np.float32), K, N)

    onnx.checker.check_model(q_imatrix)
    onnx.checker.check_model(plain_model)

    x_test = _make_batch(2000, 999)["X"]
    (float_y,) = _run(model, {"X": x_test})
    (imatrix_y,) = _run(q_imatrix, {"X": x_test})
    (plain_y,) = _run(plain_model, {"X": x_test})

    imatrix_mse = float(np.mean((float_y - imatrix_y) ** 2))
    plain_mse = float(np.mean((float_y - plain_y) ** 2))
    assert imatrix_mse < plain_mse
