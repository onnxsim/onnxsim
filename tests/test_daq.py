"""Tests for ``onnxsim.apply_daq`` (DAQ, see ``onnxsim/daq.py``) -- picks a
per-layer scalar FP8 scale that preserves a fine-tune's own weight *update*
``delta_w = w_post - w_base`` (by cosine similarity or sign-preservation
rate against the reconstructed update), instead of minimizing the raw
reconstruction error of the post-trained weight alone.

The pass is data-free -- it never runs either model -- so these tests need
no onnxruntime and no calibration data at all; every assertion is made
directly on the rewritten weight initializer.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.daq import _cosine_similarity, _sign_preservation_rate
from onnxsim.deepseek_fp8 import _fp8_round_trip


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


def _matmul_model(w, K, N):
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
    )


def _current_weight(model):
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_init = next(t for t in model.graph.initializer if t.name == node.input[1])
    return onnx.numpy_helper.to_array(w_init).astype(np.float64)


def _base_and_post_weights(K=64, N=48, seed=0, delta_scale=0.02):
    """DAQ's own motivating scenario: a fine-tune whose update is a *small*,
    structured (here rank-1) perturbation riding on top of a base weight
    that dominates the tensor's own magnitude -- so a plain absmax FP8 scale,
    fit to ``w_post`` alone, has no reason to "notice" the update at all.
    """
    rng = np.random.default_rng(seed)
    w_base = rng.standard_normal((K, N)).astype(np.float32)
    u = np.cos(np.arange(K) * 0.37).astype(np.float32)
    v = np.sin(np.arange(N) * 0.61 + 0.2).astype(np.float32)
    delta_w = (np.outer(u, v) * delta_scale).astype(np.float32)
    return w_base, (w_base + delta_w).astype(np.float32)


def _naive_round_trip(w_post):
    """What plain deepseek-style per-tensor FP8 quantization of ``w_post``
    alone would produce -- DAQ's own coarse grid is centered on exactly this
    scale, so this is the baseline candidate its search has to beat.
    """
    w_post = np.asarray(w_post, dtype=np.float64)
    scale0 = max(float(np.max(np.abs(w_post))), 1e-12) / 448.0
    return _fp8_round_trip(w_post / scale0) * scale0


def test_daq_preserves_delta_better_than_naive_absmax_scale():
    # The core claim: DAQ's chosen scale reconstructs the fine-tuning update
    # with strictly higher cosine similarity than the naive max(|w|)/448
    # scale does, even though the update is tiny next to w_post itself.
    w_base, w_post = _base_and_post_weights(seed=0)
    K, N = w_post.shape
    quantized = onnxsim.apply_daq(
        _matmul_model(w_base, K, N), _matmul_model(w_post, K, N)
    )
    onnx.checker.check_model(quantized)

    w_base64 = w_base.astype(np.float64)
    delta_w = w_post.astype(np.float64) - w_base64
    daq_similarity = _cosine_similarity(delta_w, _current_weight(quantized) - w_base64)
    naive_similarity = _cosine_similarity(delta_w, _naive_round_trip(w_post) - w_base64)

    assert daq_similarity > naive_similarity
    # ...and the weight really was quantized (not just left alone).
    assert not np.array_equal(_current_weight(quantized), w_post.astype(np.float64))


def test_daq_sign_preservation_metric_beats_naive_scale():
    w_base, w_post = _base_and_post_weights(seed=1)
    K, N = w_post.shape
    quantized = onnxsim.apply_daq(
        _matmul_model(w_base, K, N),
        _matmul_model(w_post, K, N),
        metric="sign_preservation",
    )

    w_base64 = w_base.astype(np.float64)
    delta_w = w_post.astype(np.float64) - w_base64
    daq_rate = _sign_preservation_rate(delta_w, _current_weight(quantized) - w_base64)
    naive_rate = _sign_preservation_rate(delta_w, _naive_round_trip(w_post) - w_base64)

    assert 0.0 <= daq_rate <= 1.0
    assert daq_rate >= naive_rate


def test_daq_leaves_zero_delta_layer_completely_untouched():
    # A weight the fine-tune never moved has no delta signal to preserve, so
    # DAQ declines to quantize it at all -- a deliberate scope decision.
    w_base, _ = _base_and_post_weights(seed=2)
    K, N = w_base.shape
    post = _matmul_model(w_base, K, N)

    result = onnxsim.apply_daq(_matmul_model(w_base, K, N), post)
    assert result.SerializeToString() == post.SerializeToString()


def test_daq_skips_shape_mismatched_counterpart():
    w_base, w_post = _base_and_post_weights(seed=3)
    K, N = w_post.shape
    mismatched = _base_and_post_weights(K=K // 2, N=N, seed=4)[0]
    post = _matmul_model(w_post, K, N)

    result = onnxsim.apply_daq(_matmul_model(mismatched, K // 2, N), post)
    assert result.SerializeToString() == post.SerializeToString()


def test_daq_skips_layer_with_no_counterpart_in_base_model():
    _, w_post = _base_and_post_weights(seed=5)
    K, N = w_post.shape
    post = _matmul_model(w_post, K, N)
    base = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )

    result = onnxsim.apply_daq(base, post)
    assert result.SerializeToString() == post.SerializeToString()


def test_daq_skips_non_constant_weight():
    _, w_post = _base_and_post_weights(seed=6)
    K, N = w_post.shape
    body = f"""
        g (float[batch,{K}] X, float[{K},{N}] W) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """
    post = _model(body)

    result = onnxsim.apply_daq(_model(body), post)
    assert result.SerializeToString() == post.SerializeToString()


def test_daq_rejects_unknown_metric():
    w_base, w_post = _base_and_post_weights(seed=7)
    K, N = w_post.shape
    with pytest.raises(ValueError, match="l2"):
        onnxsim.apply_daq(
            _matmul_model(w_base, K, N), _matmul_model(w_post, K, N), metric="l2"
        )


def test_daq_respects_skip_names():
    w_base, w_post = _base_and_post_weights(seed=8)
    K, N = w_post.shape
    post = _matmul_model(w_post, K, N)

    result = onnxsim.apply_daq(_matmul_model(w_base, K, N), post, skip_names={"W"})
    assert result.SerializeToString() == post.SerializeToString()


def test_daq_gemm_transb():
    # Transposed weight layout: DAQ's scale is a single scalar for the whole
    # layer, so the layout is irrelevant to the math -- but the node still
    # has to be matched and rewritten correctly.
    rng = np.random.default_rng(9)
    K, N = 64, 24
    w_base = rng.standard_normal((N, K)).astype(np.float32)
    u = np.cos(np.arange(N) * 0.37).astype(np.float32)
    v = np.sin(np.arange(K) * 0.61 + 0.2).astype(np.float32)
    w_post = (w_base + np.outer(u, v) * 0.02).astype(np.float32)

    def gemm_model(w):
        return _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{N}] Y)
            {{
              Y = Gemm<transB = 1>(X, W)
            }}
            """,
            [_f32(w, "W")],
        )

    quantized = onnxsim.apply_daq(gemm_model(w_base), gemm_model(w_post))
    onnx.checker.check_model(quantized)

    new_w = _current_weight(quantized)
    assert new_w.shape == w_post.shape
    assert not np.array_equal(new_w, w_post.astype(np.float64))

    w_base64 = w_base.astype(np.float64)
    delta_w = w_post.astype(np.float64) - w_base64
    assert _cosine_similarity(delta_w, new_w - w_base64) > _cosine_similarity(
        delta_w, _naive_round_trip(w_post) - w_base64
    )


def test_cosine_similarity_guards_degenerate_zero_vector():
    zeros = np.zeros(8)
    ones = np.ones(8)
    assert _cosine_similarity(zeros, ones) == -1.0
    assert _cosine_similarity(ones, zeros) == -1.0
    assert _cosine_similarity(ones, ones) == pytest.approx(1.0)


def test_sign_preservation_rate_is_a_fraction():
    a = np.asarray([1.0, -1.0, 2.0, -2.0])
    b = np.asarray([1.0, 1.0, 3.0, -5.0])
    assert _sign_preservation_rate(a, b) == pytest.approx(0.75)
