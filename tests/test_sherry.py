"""Tests for ``onnxsim.apply_sherry_quantization`` (see ``onnxsim/sherry.py``)
-- Sherry's published 3:4 sparse ternary "Sparse-AbsMean" rule (prune each
4-block's smallest ``|w|``, ``sign(w)`` for the other three, one
``alpha = mean(|w|)`` over an output channel's kept weights), weight-only,
represented as a float32 quantize-dequantize round trip.

The arithmetic is checked directly against numpy re-implementations and,
first of all, against a fully hand-computed example -- the module folds the
quantized weight straight into a new initializer and adds no graph nodes,
so nothing here depends on an onnxruntime round trip except the one test
that deliberately runs both graphs.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.sherry import _BLOCK_SIZE, _NONZERO_PER_BLOCK, _sherry_rows

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=8):
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


def _matmul_model(w, batch="batch"):
    k, n = w.shape
    return _model(
        f"""
        g (float[{batch},{k}] X) => (float[{batch},{n}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
    )


def _gemm_model(w_nk, batch="batch"):
    n, k = w_nk.shape
    return _model(
        f"""
        g (float[{batch},{k}] X) => (float[{batch},{n}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        [_f32(w_nk, "W")],
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


def _weight_of(model, node_op_type):
    node = next(n for n in model.graph.node if n.op_type == node_op_type)
    return onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == node.input[1])
    )


# The two hand-computed channels this file's authoritative correctness test
# uses. Every number below is worked out by hand in the test's own comments.
_CHANNEL_0 = np.array([0.5, -2.0, 3.0, 1.0, -4.0, 0.25, 2.0, -1.5])
_CHANNEL_1 = np.array([1.0, 1.0, 1.0, -8.0, 0.5, -0.5, 0.5, 0.5])
_ALPHA_0 = 13.5 / 6.0  # = 2.25
_ALPHA_1 = 11.5 / 6.0
_EXPECTED_0 = np.array([0.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, -1.0]) * _ALPHA_0
_EXPECTED_1 = np.array([0.0, 1.0, 1.0, -1.0, 0.0, -1.0, 1.0, 1.0]) * _ALPHA_1


def test_hand_computed_channel_matches_the_papers_own_sparse_absmean_rule():
    # Channel 0 = [0.5, -2.0, 3.0, 1.0 | -4.0, 0.25, 2.0, -1.5], two blocks
    # of 4, worked out by hand:
    #
    #   block A |w| = [0.5, 2.0, 3.0, 1.0] -> smallest is 0.5 (index 0), so
    #     index 0 is the pruned zero; the other three become sign(w) =
    #     [-1, +1, +1].
    #   block B |w| = [4.0, 0.25, 2.0, 1.5] -> smallest is 0.25 (index 1),
    #     pruned; the other three become sign(w) = [-1, +1, -1].
    #   alpha = mean(|w|) over the SIX kept weights only (not over all
    #     eight, and not over the pruned ones):
    #     (2.0 + 3.0 + 1.0 + 4.0 + 2.0 + 1.5) / 6 = 13.5 / 6 = 2.25.
    #
    # so the round trip is
    #   [0, -2.25, +2.25, +2.25, -2.25, 0, +2.25, -2.25].
    #
    # (Note the mean is over the kept weights: averaging all eight would
    # give 14.25 / 8 = 1.78125, and averaging the six kept over the block
    # size would give 13.5 / 8 = 1.6875 -- both wrong, and both the most
    # likely way to get this rule subtly wrong.)
    assert _ALPHA_0 == 2.25
    np.testing.assert_allclose(
        onnxsim.quantize_dequantize_sherry(_CHANNEL_0), _EXPECTED_0, rtol=0, atol=0
    )

    # Channel 1 = [1.0, 1.0, 1.0, -8.0 | 0.5, -0.5, 0.5, 0.5] exercises
    # magnitude ties, which resolve to the leftmost minimum:
    #
    #   block A |w| = [1.0, 1.0, 1.0, 8.0] -> a three-way tie for smallest;
    #     index 0 is pruned, leaving sign(w) = [+1, +1, -1].
    #   block B |w| = [0.5, 0.5, 0.5, 0.5] -> a four-way tie; index 0 is
    #     pruned, leaving sign(w) = [-1, +1, +1].
    #   alpha = (1.0 + 1.0 + 8.0 + 0.5 + 0.5 + 0.5) / 6 = 11.5 / 6.
    np.testing.assert_allclose(
        onnxsim.quantize_dequantize_sherry(_CHANNEL_1), _EXPECTED_1, rtol=0, atol=0
    )


def test_hand_computed_matmul_weight_is_quantized_per_output_channel():
    # The same two hand-computed channels, now as the two output channels
    # (columns) of a [K=8, N=2] MatMul weight. Each column must come back
    # with its own alpha -- 2.25 and 11.5/6 respectively -- which is what
    # "per-channel" means for a [K, N] weight.
    w = np.stack([_CHANNEL_0, _CHANNEL_1], axis=1)
    assert w.shape == (8, 2)

    q = onnxsim.apply_sherry_quantization(_matmul_model(w))
    onnx.checker.check_model(q)
    new_w = _weight_of(q, "MatMul")

    expected = np.stack([_EXPECTED_0, _EXPECTED_1], axis=1).astype(np.float32)
    np.testing.assert_allclose(new_w, expected, rtol=1e-6, atol=1e-6)


def test_exactly_one_zero_per_block_and_every_nonzero_is_plus_minus_alpha():
    rng = np.random.default_rng(0)
    # Continuous Gaussian draws: no element is ever exactly 0.0, so every
    # block's single zero is the pruned element and nothing else.
    rows = rng.standard_normal((6, _BLOCK_SIZE * 9)) * rng.uniform(
        0.1, 3.0, size=(6, 1)
    )
    dequant = _sherry_rows(rows)
    assert dequant.shape == rows.shape

    for row, quant_row in zip(rows, dequant):
        for block in quant_row.reshape(-1, _BLOCK_SIZE):
            assert np.count_nonzero(block == 0.0) == 1
            assert np.count_nonzero(block) == _NONZERO_PER_BLOCK

        nonzero = quant_row[quant_row != 0.0]
        magnitudes = np.unique(np.abs(nonzero))
        assert magnitudes.size == 1  # one alpha for the whole channel

        # ...and that alpha is exactly the mean |w| of the kept weights.
        kept = quant_row != 0.0
        np.testing.assert_allclose(
            magnitudes[0], np.mean(np.abs(row[kept])), rtol=1e-12
        )
        # Signs are preserved everywhere a weight survived.
        np.testing.assert_array_equal(np.sign(quant_row[kept]), np.sign(row[kept]))


def test_alpha_is_the_kept_mean_not_the_whole_channel_mean():
    # A channel with one huge and three tiny weights per block: the pruned
    # tiny element must not drag alpha down.
    values = np.array([10.0, 0.001, 1.0, 2.0])
    dequant = onnxsim.quantize_dequantize_sherry(values)
    expected_alpha = (10.0 + 1.0 + 2.0) / 3.0
    np.testing.assert_allclose(
        dequant, np.array([1.0, 0.0, 1.0, 1.0]) * expected_alpha, rtol=1e-12
    )
    # The whole-channel mean, 13.001 / 4, is what a naive implementation
    # would produce -- make sure that is *not* what came back.
    assert not np.isclose(np.abs(dequant[0]), np.mean(np.abs(values)))


def test_gemm_transb_matches_the_equivalent_untransposed_matmul():
    rng = np.random.default_rng(1)
    k, n = 4 * 5, 6
    w_kn = rng.standard_normal((k, n)) * rng.uniform(0.2, 2.0, size=(1, n))

    q_matmul = onnxsim.apply_sherry_quantization(_matmul_model(w_kn.astype(np.float32)))
    q_gemm = onnxsim.apply_sherry_quantization(
        _gemm_model(w_kn.T.copy().astype(np.float32))
    )

    matmul_w = _weight_of(q_matmul, "MatMul")  # [K, N]
    gemm_w = _weight_of(q_gemm, "Gemm")  # [N, K]
    assert matmul_w.shape == (k, n)
    assert gemm_w.shape == (n, k)
    # transB=1 means the very same layer, so the very same quantization --
    # the two initializers must be exact transposes, not merely similar.
    np.testing.assert_array_equal(gemm_w, matmul_w.T)


def test_conv_weight_is_quantized_per_output_channel_when_enabled():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((4, 2, 4, 4)).astype(np.float32)
    model = _model(
        """
        g (float[1,2,8,8] X) => (float[1,4,5,5] Y)
        {
          Y = Conv(X, W)
        }
        """,
        [_f32(w, "W")],
    )

    q_with_conv = onnxsim.apply_sherry_quantization(model, include_conv=True)
    conv_node = next(n for n in q_with_conv.graph.node if n.op_type == "Conv")
    assert conv_node.input[1] != "W"
    new_w = _weight_of(q_with_conv, "Conv")
    assert new_w.shape == w.shape
    # Conv's own axis 0 is the output channel: one alpha per filter, over
    # that filter's whole [C_in, kH, kW] flattened row.
    for filter_w in new_w.reshape(w.shape[0], -1):
        nonzero = np.abs(filter_w[filter_w != 0.0])
        assert np.unique(nonzero).size == 1
        assert np.count_nonzero(filter_w == 0.0) == filter_w.size // _BLOCK_SIZE

    q_without_conv = onnxsim.apply_sherry_quantization(model, include_conv=False)
    assert q_without_conv.SerializeToString() == model.SerializeToString()


def test_ragged_tail_keeps_its_real_weights_and_ignores_the_padding():
    # 6 elements = one full block plus a 2-element ragged tail. The packed
    # format pads the tail out to 4, and the pad is what takes the block's
    # zero slot -- so both real tail weights survive, and neither padded
    # zero enters the alpha mean.
    values = np.array([4.0, 1.0, 2.0, 3.0, 5.0, -7.0])
    dequant = onnxsim.quantize_dequantize_sherry(values)
    # Full block |w| = [4, 1, 2, 3] -> index 1 pruned.
    # Tail (padded to [5, -7, 0, 0]) -> a pad is pruned, both reals kept.
    # alpha = (4 + 2 + 3 + 5 + 7) / 5 = 21 / 5 = 4.2 -- over five kept real
    # weights, not over 6, 7 or 8 slots.
    alpha = 21.0 / 5.0
    np.testing.assert_allclose(
        dequant, np.array([1.0, 0.0, 1.0, 1.0, 1.0, -1.0]) * alpha, rtol=1e-12
    )


def _reference_sherry(values):
    """An independent, deliberately slow loop-written Sparse-AbsMean over
    one channel's *real* elements only -- no padding anywhere, so it pins
    down what the module's padded implementation is supposed to agree with.
    """
    values = np.asarray(values, dtype=np.float64)
    keep = np.zeros(values.size, dtype=bool)
    for start in range(0, values.size, _BLOCK_SIZE):
        block = values[start : start + _BLOCK_SIZE]
        # Only a whole block of 4 has an element to give up; a shorter
        # trailing block keeps everything (its zero slot is a pad).
        pruned = int(np.argmin(np.abs(block))) if block.size == _BLOCK_SIZE else -1
        for offset in range(block.size):
            keep[start + offset] = offset != pruned
    out = np.zeros_like(values)
    if keep.any():
        alpha = float(np.mean(np.abs(values[keep])))
        out[keep] = np.sign(values[keep]) * alpha
    return out


@pytest.mark.parametrize("length", range(1, 18))
def test_padding_is_inert_against_an_independent_reference(length):
    rng = np.random.default_rng(100 + length)
    values = rng.standard_normal(length) * 2.0
    np.testing.assert_allclose(
        onnxsim.quantize_dequantize_sherry(values),
        _reference_sherry(values),
        rtol=1e-12,
    )


def test_an_exactly_zero_weight_stays_zero_rather_than_becoming_plus_alpha():
    # |0| is always its block's smallest magnitude, so a lone real zero is
    # simply the element that gets pruned. With *two* real zeros in a
    # block, only one can be the pruned element -- the other is "kept", and
    # sign(0) = 0 leaves it at 0.0 rather than forcing it to +alpha (zero
    # is its own best ternary code). That block then holds two zeros, and
    # the kept zero contributes |0| to the channel's alpha mean, unlike a
    # pad. See the module docstring.
    values = np.array([0.0, 0.0, 1.0, 2.0])
    dequant = onnxsim.quantize_dequantize_sherry(values)
    assert dequant[0] == 0.0
    assert dequant[1] == 0.0
    assert np.count_nonzero(dequant) == 2
    expected_alpha = (0.0 + 1.0 + 2.0) / 3.0  # three kept, one of them |0|
    np.testing.assert_allclose(np.abs(dequant[2]), expected_alpha, rtol=1e-12)


def test_beats_a_naive_global_absmean_ternarization_across_many_seeds():
    # The naive baseline: ternarize every weight of the whole tensor
    # against a single global absmean scale (BitNet b1.58's own rule with
    # no per-channel scale and no 3:4 structure), which is what Sherry's
    # per-output-channel alpha is an improvement over. Measured over 20
    # seeds rather than asserted on one: the per-seed spread is small, so
    # the bounds below are generous multiples of what was actually seen
    # (mean ratio ~0.72, worst seed ~0.80 over these 20 seeds).
    ratios = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n, k = 32, 256
        # Channel-wise heterogeneous scales, as real layers have.
        w = rng.standard_normal((n, k)) * rng.uniform(0.05, 4.0, size=(n, 1))

        sherry = _sherry_rows(w)
        d = np.mean(np.abs(w))
        baseline = np.clip(np.round(w / d), -1.0, 1.0) * d

        ratios.append(float(np.mean((w - sherry) ** 2) / np.mean((w - baseline) ** 2)))

    assert max(ratios) < 0.95
    assert float(np.mean(ratios)) < 0.9


def test_the_3_to_4_structure_costs_only_a_little_against_unconstrained_ternary():
    # The honest other side of the previous test: against an *unconstrained*
    # per-channel absmean ternary quantizer (same per-channel scale, but
    # free to place its zeros anywhere), the fixed one-zero-per-4 structure
    # is slightly worse -- that is the price paid for a
    # hardware-friendly/packable layout, not a modelling win. Measured over
    # 20 seeds: mean ratio ~1.08, worst ~1.10.
    ratios = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n, k = 32, 256
        w = rng.standard_normal((n, k)) * rng.uniform(0.05, 4.0, size=(n, 1))

        sherry = _sherry_rows(w)
        d = np.mean(np.abs(w), axis=1, keepdims=True)
        unconstrained = np.clip(np.round(w / d), -1.0, 1.0) * d

        ratios.append(
            float(np.mean((w - sherry) ** 2) / np.mean((w - unconstrained) ** 2))
        )

    assert 1.0 < float(np.mean(ratios)) < 1.25
    assert max(ratios) < 1.3


def test_apply_sherry_quantization_respects_skip_names():
    rng = np.random.default_rng(4)
    w = (rng.standard_normal((8, 4)) * 0.5).astype(np.float32)
    model = _matmul_model(w)

    result = onnxsim.apply_sherry_quantization(model, skip_names={"W"})
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_sherry_quantization_skips_non_constant_weight():
    # W here is a graph input, not an initializer -- nothing to quantize.
    model = _model(
        """
        g (float[batch,8] X, float[8,4] W) => (float[batch,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    result = onnxsim.apply_sherry_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_sherry_quantization_skips_non_float32_weight():
    w = np.zeros((8, 4), dtype=np.int64)
    model = _model(
        """
        g (float[batch,8] X) => (float[batch,4] Y)
        {
          Wf = Cast<to=1>(W)
          Y = MatMul(X, Wf)
        }
        """,
        [onnx.numpy_helper.from_array(w, name="W")],
    )
    result = onnxsim.apply_sherry_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_sherry_quantization_skips_a_non_2d_matmul_weight():
    # A [B, K, N] batched MatMul weight has no unambiguous output channel
    # axis, so it is left alone (Conv's rank > 2 weights are handled, since
    # their channel axis is unambiguous).
    rng = np.random.default_rng(5)
    w = rng.standard_normal((2, 8, 4)).astype(np.float32)
    model = _model(
        """
        g (float[2,3,8] X) => (float[2,3,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    result = onnxsim.apply_sherry_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_sherry_quantization_noop_when_no_matmul_gemm_conv_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_sherry_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_quantized_model_runs_through_onnxruntime_with_a_sane_relative_error():
    rng = np.random.default_rng(6)
    k, n = 64, 8
    w = (rng.standard_normal((k, n)) * 0.5).astype(np.float32)
    model = _matmul_model(w)
    quantized = onnxsim.apply_sherry_quantization(model)
    onnx.checker.check_model(quantized)
    assert len(quantized.graph.node) == len(model.graph.node)

    x = rng.standard_normal((5, k)).astype(np.float32)
    (float_out,) = _run(model, {"X": x})
    (quant_out,) = _run(quantized, {"X": x})

    assert quant_out.shape == float_out.shape
    assert np.all(np.isfinite(quant_out))
    # ~1.25 bits/weight: a large but bounded error, not a broken graph.
    assert _rel_l2(float_out, quant_out) < 0.9


def test_apply_sherry_quantization_accepts_a_gemm_with_bias():
    rng = np.random.default_rng(7)
    k, n = 16, 4
    w_nk = (rng.standard_normal((n, k))).astype(np.float32)
    bias = rng.standard_normal(n).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{k}] X) => (float[batch,{n}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        [_f32(w_nk, "W"), _f32(bias, "B")],
    )

    q = onnxsim.apply_sherry_quantization(model)
    onnx.checker.check_model(q)
    new_w = _weight_of(q, "Gemm")
    np.testing.assert_allclose(new_w, _sherry_rows(w_nk.astype(np.float64)), rtol=1e-6)
    # The bias is untouched.
    assert next(t for t in q.graph.initializer if t.name == "B") is not None


def test_sherry_keeps_an_exact_zero_weight_at_zero():
    # sign(0) == 0, so a 4-block holding two exact zeros comes out with two
    # zeros rather than the format's exactly-one non-pruned-zero. The real
    # 5-bit layout has no "kept but zero" encoding (every kept slot carries a
    # sign bit and dequantizes to +-alpha), so this is a documented
    # divergence from the format, not a rounding artifact -- see this
    # module's own docstring. Pinned here so it stays a deliberate choice.
    #
    # An ordinary trained weight has no exact zeros; an already-sparsified
    # one (e.g. via onnxsim.apply_magnitude_pruning) does, which is why this
    # case is worth pinning rather than dismissing.
    rows = np.array([[0.0, 0.0, 3.0, 1.0]])
    out = _sherry_rows(rows)
    assert np.all(np.isfinite(out))
    assert int(np.count_nonzero(out == 0.0)) == 2
    # alpha averages |w| over the kept elements only: |0|, |3|, |1| -> 4/3.
    assert out[0, 2] == pytest.approx(4.0 / 3.0)
    assert out[0, 3] == pytest.approx(4.0 / 3.0)


def test_sherry_all_zero_weight_stays_zero_without_nan():
    # kept_count is floored at 1 so an all-zero channel cannot divide by
    # zero; alpha is 0 and the round trip is the zero tensor.
    out = onnxsim.quantize_dequantize_sherry(np.zeros(8, dtype=np.float32))
    assert np.all(np.isfinite(out))
    assert not np.any(out)
