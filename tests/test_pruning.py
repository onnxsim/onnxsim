"""Tests for ``onnxsim.pruning`` -- magnitude pruning (data-free baseline),
Wanda pruning (calibrated on activation norms), and structured (channel)
pruning, see ``onnxsim/pruning.py``.
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21):
    # Pinning ir_version: 10 matches the older onnxruntime bundled with some
    # CI wheels (which cap at IR version 11); `_run` and onnxsim's own
    # checks below run these models through onnxruntime.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _matmul_model(K=64, N=16, seed=0):
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
    )


def _weight(model):
    return onnx.numpy_helper.to_array(model.graph.initializer[0])


def test_magnitude_pruning_reaches_target_sparsity():
    model = _matmul_model(K=64, N=16)
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)
    # Shape is untouched -- this is a value-only rewrite.
    assert _weight(pruned).shape == _weight(model).shape


def test_magnitude_pruning_keeps_the_largest_entries_per_row():
    model = _matmul_model(K=64, N=16)
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.75)
    w = _weight(model).astype(np.float64)  # [K, N]
    w_pruned = _weight(pruned).astype(np.float64)
    # Per output column (row of W^T), the surviving entries must be exactly
    # the top-(1 - sparsity) fraction by magnitude.
    for col in range(w.shape[1]):
        kept = np.flatnonzero(w_pruned[:, col] != 0)
        assert len(kept) == 16  # round(64 * 0.25)
        threshold = np.abs(w[:, col])[kept].min()
        dropped_max = np.abs(w[:, col])[np.flatnonzero(w_pruned[:, col] == 0)].max()
        assert dropped_max <= threshold


def test_magnitude_pruning_zero_sparsity_is_a_no_op():
    model = _matmul_model(K=32, N=8)
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.0)
    np.testing.assert_array_equal(_weight(pruned), _weight(model))


def test_magnitude_pruning_nm_pattern():
    model = _matmul_model(K=64, N=16)
    pruned = onnxsim.apply_magnitude_pruning(model, n=2, m=4)
    w_pruned = _weight(pruned).T  # [N, K], row-major per output channel
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)
    for row in w_pruned:
        for start in range(0, len(row), 4):
            group = row[start : start + 4]
            assert np.count_nonzero(group) <= 2


def test_magnitude_pruning_requires_n_and_m_together():
    model = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_magnitude_pruning(model, n=2)
    with pytest.raises(ValueError):
        onnxsim.apply_magnitude_pruning(model, m=4)


def test_wanda_pruning_protects_high_activation_channels():
    # A handful of input channels carry much larger activation magnitude
    # than the rest but a merely-average weight magnitude -- Wanda's own
    # motivating scenario: plain |W| magnitude pruning is blind to this and
    # may prune those channels' weights anyway, while Wanda's
    # |W| * ||X||_2 metric should protect them.
    K, N = 64, 16
    salient = (3, 7, 40)
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )

    x = rng.standard_normal((32, K)).astype(np.float32)
    for c in salient:
        x[:, c] *= 20.0
    calibration_data = [{"X": x}]

    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    wanda_pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(wanda_pruned)
    assert onnxsim.weight_sparsity(wanda_pruned) == pytest.approx(0.5, abs=1e-9)

    w_magnitude = _weight(magnitude_pruned)
    w_wanda = _weight(wanda_pruned)
    # Wanda must keep strictly more of the salient rows' entries than plain
    # magnitude pruning -- otherwise this is just re-testing magnitude
    # pruning under a different name.
    salient_kept_magnitude = np.count_nonzero(w_magnitude[list(salient), :])
    salient_kept_wanda = np.count_nonzero(w_wanda[list(salient), :])
    assert salient_kept_wanda > salient_kept_magnitude

    (float_y,) = _run(model, {"X": x})
    (magnitude_y,) = _run(magnitude_pruned, {"X": x})
    (wanda_y,) = _run(wanda_pruned, {"X": x})
    magnitude_err = np.linalg.norm(float_y.astype(np.float64) - magnitude_y)
    wanda_err = np.linalg.norm(float_y.astype(np.float64) - wanda_y)
    assert wanda_err < magnitude_err


def test_wanda_pruning_falls_back_to_magnitude_without_matching_activation():
    # X isn't 2-D at the probe point (it's 3-D), so Wanda never observes a
    # usable activation norm and must fall back to plain |W| pruning rather
    # than leaving the layer untouched or crashing.
    K, N = 32, 8
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    x = rng.standard_normal((2, 4, K)).astype(np.float32)

    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    wanda_pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5
    )
    onnx.checker.check_model(wanda_pruned)
    np.testing.assert_array_equal(_weight(wanda_pruned), _weight(magnitude_pruned))


def test_weight_sparsity_of_unpruned_model_is_zero():
    model = _matmul_model(K=16, N=4)
    assert onnxsim.weight_sparsity(model) == 0.0


def test_weight_sparsity_ignores_non_matching_layers():
    model = _model(
        """
        g (float[4] X) => (float[4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    assert onnxsim.weight_sparsity(model) == 0.0


# --- magnitude/Wanda pruning: Conv2D -------------------------------------


def _single_conv_model(w, spatial=10, extra_attrs="", out_spatial=None):
    Cout, Cin, kh, kw = w.shape
    if out_spatial is None:
        out_spatial = spatial - kh + 1  # no padding, unit stride
    attrs = f"kernel_shape=[{kh},{kw}]"
    if extra_attrs:
        attrs += ", " + extra_attrs
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_spatial},{out_spatial}] Y)
        {{
          Y = Conv<{attrs}>(X, W1)
        }}
        """,
        initializer=[_f32(w, "W1")],
    )


def _conv_weight(model):
    return onnx.numpy_helper.to_array(model.graph.initializer[0])


def test_magnitude_pruning_conv_reaches_target_sparsity():
    Cin, Cout = 4, 8  # K = Cin*3*3 = 36, evenly halved by sparsity=0.5
    rng = np.random.default_rng(60)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    model = _single_conv_model(w)

    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)
    assert _conv_weight(pruned).shape == w.shape


def test_magnitude_pruning_conv_keeps_the_largest_entries_per_filter():
    Cin, Cout = 4, 8
    rng = np.random.default_rng(61)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32) * 0.5
    model = _single_conv_model(w)

    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.75)
    K = Cin * 3 * 3
    w_flat = w.astype(np.float64).reshape(Cout, K)
    w_pruned_flat = _conv_weight(pruned).astype(np.float64).reshape(Cout, K)
    keep_count = round(K * 0.25)
    for row in range(Cout):
        kept = np.flatnonzero(w_pruned_flat[row] != 0)
        assert len(kept) == keep_count
        threshold = np.abs(w_flat[row])[kept].min()
        dropped_max = np.abs(w_flat[row])[np.flatnonzero(w_pruned_flat[row] == 0)].max()
        assert dropped_max <= threshold


def test_magnitude_pruning_conv_nm_pattern():
    Cin, Cout = 4, 8
    rng = np.random.default_rng(62)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    model = _single_conv_model(w)

    pruned = onnxsim.apply_magnitude_pruning(model, n=2, m=4)
    K = Cin * 3 * 3
    w_flat = _conv_weight(pruned).reshape(Cout, K)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)
    for row in w_flat:
        for start in range(0, len(row), 4):
            group = row[start : start + 4]
            assert np.count_nonzero(group) <= 2


def test_magnitude_pruning_skips_grouped_conv():
    # A depthwise Conv (group == in_channels == out_channels) is left
    # completely untouched -- the same restriction the Conv2D structured
    # pruning section's own producer/consumer matching draws (see
    # :func:`_match_conv_weight_only`'s own docstring).
    C = 8
    rng = np.random.default_rng(63)
    w = rng.standard_normal((C, 1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{C},10,10] X) => (float[N,{C},8,8] Y)
        {{
          Y = Conv<kernel_shape=[3,3], group={C}>(X, W1)
        }}
        """,
        initializer=[_f32(w, "W1")],
    )
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    np.testing.assert_array_equal(_conv_weight(pruned), w)


def _naive_conv_patch_sq_sum(x, attrs):
    # Slow, obviously-correct nested-loop reference for the per-``(in_
    # channel, kh, kw)`` offset activation statistic Wanda needs for Conv
    # -- the oracle :func:`onnxsim.pruning._conv_patch_sq_sum`'s own
    # ``sliding_window_view``-based implementation is checked against
    # below, before it's ever trusted inside the actual pruning path.
    n, cin, h, w = x.shape
    xp = np.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (attrs.pad_top, attrs.pad_bottom),
            (attrs.pad_left, attrs.pad_right),
        ),
    )
    hp, wp = xp.shape[2], xp.shape[3]
    h_out = (hp - attrs.kh) // attrs.stride_h + 1
    w_out = (wp - attrs.kw) // attrs.stride_w + 1
    sq = np.zeros((cin, attrs.kh, attrs.kw), dtype=np.float64)
    count = 0
    for ni in range(n):
        for oh in range(h_out):
            for ow in range(w_out):
                count += 1
                for c in range(cin):
                    for i in range(attrs.kh):
                        for j in range(attrs.kw):
                            val = xp[
                                ni, c, oh * attrs.stride_h + i, ow * attrs.stride_w + j
                            ]
                            sq[c, i, j] += val * val
    return sq, count


def test_conv_patch_sq_sum_matches_naive_nested_loop_oracle():
    from onnxsim.pruning import _conv_patch_sq_sum, _ConvSpatialAttrs

    rng = np.random.default_rng(70)
    x = rng.standard_normal((2, 3, 4, 4))
    cases = [
        _ConvSpatialAttrs(
            kh=3,
            kw=3,
            pad_top=1,
            pad_left=1,
            pad_bottom=1,
            pad_right=1,
            stride_h=2,
            stride_w=2,
        ),
        _ConvSpatialAttrs(
            kh=2,
            kw=2,
            pad_top=0,
            pad_left=0,
            pad_bottom=0,
            pad_right=0,
            stride_h=1,
            stride_w=1,
        ),
        _ConvSpatialAttrs(
            kh=3,
            kw=3,
            pad_top=0,
            pad_left=2,
            pad_bottom=1,
            pad_right=0,
            stride_h=1,
            stride_w=2,
        ),
    ]
    for attrs in cases:
        sq_vec, count_vec = _conv_patch_sq_sum(x, attrs)
        sq_naive, count_naive = _naive_conv_patch_sq_sum(x, attrs)
        assert count_vec == count_naive
        np.testing.assert_allclose(sq_vec, sq_naive)


def test_wanda_pruning_conv_matches_manual_im2col_importance_oracle_exactly():
    # Same correctness bar as the MatMul/Gemm Wanda tests, but the oracle's
    # activation norm is computed by manually unfolding X into overlapping
    # kh*kw patches (a second, independent im2col implementation from the
    # one under test, see :func:`onnxsim.pruning._conv_patch_sq_sum`) and
    # reducing over batch and every output position.
    Cin, Cout, kh, kw, spatial = 3, 6, 3, 3, 8
    rng = np.random.default_rng(72)
    w = rng.standard_normal((Cout, Cin, kh, kw)).astype(np.float32)
    model = _single_conv_model(w, spatial=spatial)

    rng_x = np.random.default_rng(73)
    x = rng_x.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    out_spatial = spatial - kh + 1
    K = Cin * kh * kw
    patches = np.zeros((x.shape[0] * out_spatial * out_spatial, K), dtype=np.float64)
    idx = 0
    for ni in range(x.shape[0]):
        for oh in range(out_spatial):
            for ow in range(out_spatial):
                patch = x[ni, :, oh : oh + kh, ow : ow + kw].astype(np.float64)
                patches[idx] = patch.reshape(-1)
                idx += 1
    act_norm = np.sqrt(np.mean(np.square(patches), axis=0))

    w_flat = w.astype(np.float64).reshape(Cout, K)
    importance = np.abs(w_flat) * act_norm[np.newaxis, :]
    keep = round(K * 0.5)
    order = np.argsort(importance, axis=1)
    drop = order[:, : K - keep]
    mask = np.ones((Cout, K), dtype=bool)
    np.put_along_axis(mask, drop, False, axis=1)
    expected = np.where(mask, w_flat, 0.0).reshape(Cout, Cin, kh, kw)

    np.testing.assert_allclose(_conv_weight(pruned).astype(np.float64), expected)


def test_wanda_pruning_conv_protects_high_activation_channel():
    # Same motivating scenario as the MatMul Wanda test: one input channel
    # carries much larger activation magnitude than the rest, but a
    # merely-average weight magnitude -- Wanda's per-offset activation
    # norm (rolled up here to a whole input channel, since every (kh, kw)
    # offset of that channel gets boosted identically) should protect it
    # more than plain |W| magnitude pruning does.
    Cin, Cout, spatial = 3, 8, 10
    salient_channel = 1
    rng = np.random.default_rng(71)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32) * 0.5
    model = _single_conv_model(w, spatial=spatial)

    x = rng.standard_normal((4, Cin, spatial, spatial)).astype(np.float32)
    x[:, salient_channel, :, :] *= 20.0
    calibration_data = [{"X": x}]

    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    wanda_pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(wanda_pruned)

    w_magnitude = _conv_weight(magnitude_pruned)
    w_wanda = _conv_weight(wanda_pruned)
    salient_kept_magnitude = np.count_nonzero(w_magnitude[:, salient_channel, :, :])
    salient_kept_wanda = np.count_nonzero(w_wanda[:, salient_channel, :, :])
    assert salient_kept_wanda > salient_kept_magnitude

    (float_y,) = _run(model, {"X": x})
    (magnitude_y,) = _run(magnitude_pruned, {"X": x})
    (wanda_y,) = _run(wanda_pruned, {"X": x})
    magnitude_err = np.linalg.norm(float_y.astype(np.float64) - magnitude_y)
    wanda_err = np.linalg.norm(float_y.astype(np.float64) - wanda_y)
    assert wanda_err < magnitude_err


def test_wanda_pruning_conv_falls_back_to_magnitude_for_auto_pad():
    # auto_pad SAME_UPPER's padding depends on the input's own spatial
    # size, not something fixed per node -- :func:`_conv_spatial_attrs`
    # declines it, so Wanda must fall back to plain magnitude for this
    # layer rather than guessing at the padding.
    Cin, Cout, spatial = 3, 6, 10
    rng = np.random.default_rng(74)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    model = _single_conv_model(
        w, spatial=spatial, extra_attrs='auto_pad="SAME_UPPER"', out_spatial=spatial
    )
    x = rng.standard_normal((2, Cin, spatial, spatial)).astype(np.float32)

    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    wanda_pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5
    )
    onnx.checker.check_model(wanda_pruned)
    np.testing.assert_array_equal(
        _conv_weight(wanda_pruned), _conv_weight(magnitude_pruned)
    )


def test_wanda_pruning_conv_falls_back_to_magnitude_for_dilated_conv():
    # A dilated receptive field's (kh, kw) offsets aren't evenly spaced in
    # the padded input the way sliding_window_view assumes --
    # :func:`_conv_spatial_attrs` declines non-all-ones dilations, so
    # Wanda must fall back to plain magnitude for this layer too.
    Cin, Cout, spatial = 3, 6, 10
    rng = np.random.default_rng(75)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    out_spatial = spatial - 2 * (3 - 1)  # dilation=2, kernel=3, no padding
    model = _single_conv_model(
        w, spatial=spatial, extra_attrs="dilations=[2,2]", out_spatial=out_spatial
    )
    x = rng.standard_normal((2, Cin, spatial, spatial)).astype(np.float32)

    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    wanda_pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5
    )
    onnx.checker.check_model(wanda_pruned)
    np.testing.assert_array_equal(
        _conv_weight(wanda_pruned), _conv_weight(magnitude_pruned)
    )


def test_sparsegpt_pruning_leaves_conv_layers_completely_untouched():
    # apply_sparsegpt_pruning deliberately does not match Conv (see its own
    # docstring for why): a correct-but-unverified from-scratch im2col
    # Hessian was judged worse to ship than none at all.
    Cin, Cout, spatial = 4, 8, 10
    rng = np.random.default_rng(76)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    model = _single_conv_model(w, spatial=spatial)
    x_cal = rng.standard_normal((2, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    np.testing.assert_array_equal(_conv_weight(pruned), w)


# --- apply_sparsegpt_pruning --------------------------------------------


def _reference_sparsegpt(w_nk, h, sparsity, n, m, percdamp, blocksize):
    # An independent transliteration of the reference implementation's own
    # ``SparseGPT.fasterprune`` (https://github.com/IST-DASLab/sparsegpt/
    # blob/master/sparsegpt.py), written fresh from that source rather than
    # copied from onnxsim/pruning.py, to give this an oracle that isn't
    # just "the same code twice". Uses the reference's own prunen/prunem
    # naming (prunen = number pruned per group of prunem), the mirror image
    # of onnxsim's own n/m ("n kept per group of m") convention.
    w = w_nk.copy().astype(np.float64)
    rows, cols = w.shape
    h = h.copy().astype(np.float64)
    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0

    damp = percdamp * np.mean(np.diag(h))
    diag = np.arange(cols)
    h[diag, diag] += damp
    hinv = np.linalg.cholesky(np.linalg.inv(h)).T

    prunen = 0 if n is None else m - n
    prunem = 0 if m is None else m

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        w1 = w[:, i1:i2].copy()
        q1 = np.zeros_like(w1)
        err1 = np.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]

        if prunen == 0:
            tmp = w1**2 / (np.diag(hinv1).reshape(1, -1)) ** 2
            thresh = np.sort(tmp.flatten())[int(tmp.size * sparsity)]
            mask1 = tmp <= thresh
        else:
            mask1 = np.zeros_like(w1, dtype=bool)

        for i in range(count):
            w_col = w1[:, i]
            d = hinv1[i, i]
            if prunen != 0 and i % prunem == 0:
                tmp = (
                    w1[:, i : i + prunem] ** 2
                    / (np.diag(hinv1)[i : i + prunem].reshape(1, -1)) ** 2
                )
                idx = np.argsort(tmp, axis=1)[:, :prunen]
                mask1[:, i : i + prunem] = False
                np.put_along_axis(mask1[:, i : i + prunem], idx, True, axis=1)
            q_col = w_col.copy()
            q_col[mask1[:, i]] = 0.0
            q1[:, i] = q_col
            err_col = (w_col - q_col) / d
            w1[:, i + 1 :] -= np.outer(err_col, hinv1[i, i + 1 :])
            err1[:, i] = err_col

        w[:, i1:i2] = q1
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]

    return w


def test_sparsegpt_pruning_matches_reference_transliteration_exactly():
    K, N = 32, 8
    rng = np.random.default_rng(50)
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(K=K, N=N, seed=50)
    x_cal = rng.standard_normal((48, K)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5, proc_block_size=12
    )
    onnx.checker.check_model(pruned)

    w_nk = w.T.astype(np.float64)
    h = x_cal.astype(np.float64).T @ x_cal.astype(np.float64)
    expected_nk = _reference_sparsegpt(
        w_nk, h, sparsity=0.5, n=None, m=None, percdamp=0.01, blocksize=12
    )
    np.testing.assert_allclose(_weight(pruned).T, expected_nk, rtol=1e-6, atol=1e-6)


def test_sparsegpt_pruning_nm_pattern_matches_reference_transliteration():
    K, N = 32, 8
    rng = np.random.default_rng(51)
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(K=K, N=N, seed=51)
    x_cal = rng.standard_normal((48, K)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], n=2, m=4, proc_block_size=12
    )
    onnx.checker.check_model(pruned)

    w_nk = w.T.astype(np.float64)
    h = x_cal.astype(np.float64).T @ x_cal.astype(np.float64)
    expected_nk = _reference_sparsegpt(
        w_nk, h, sparsity=0.0, n=2, m=4, percdamp=0.01, blocksize=12
    )
    np.testing.assert_allclose(_weight(pruned).T, expected_nk, rtol=1e-6, atol=1e-6)

    w_pruned = _weight(pruned).T  # [N, K]
    for row in w_pruned:
        for start in range(0, len(row), 4):
            group = row[start : start + 4]
            if len(group) == 4:
                assert np.count_nonzero(group) == 2


def test_sparsegpt_pruning_zero_sparsity_is_a_no_op():
    K, N = 16, 4
    model = _matmul_model(K=K, N=N, seed=52)
    rng = np.random.default_rng(53)
    x_cal = rng.standard_normal((32, K)).astype(np.float32)
    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.0
    )
    np.testing.assert_array_equal(_weight(pruned), _weight(model))


def test_sparsegpt_pruning_no_calibration_batches_leaves_layer_untouched():
    model = _matmul_model(K=16, N=4, seed=54)
    pruned = onnxsim.apply_sparsegpt_pruning(model, calibration_data=[], sparsity=0.5)
    np.testing.assert_array_equal(_weight(pruned), _weight(model))


def test_sparsegpt_pruning_reconstructs_better_than_a_same_mask_style_baseline():
    # The actual point of the technique: given comparable calibration
    # signal, SparseGPT's Hessian-compensated result should reconstruct
    # the layer's output at least as well, on that same calibration data,
    # as simply zeroing the same-shaped lowest-magnitude entries with no
    # compensation at all -- isolating what the error-propagation step
    # buys over naive masking.
    K, N = 48, 12
    rng = np.random.default_rng(55)
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(K=K, N=N, seed=55)
    x_cal = rng.standard_normal((512, K)).astype(np.float32)  # well-conditioned H

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    w_sparsegpt = _weight(pruned).astype(np.float64)

    w64 = w.astype(np.float64)
    score = np.abs(w64)
    thresh = np.sort(score.flatten())[int(score.size * 0.5)]
    w_naive = np.where(score <= thresh, 0.0, w64)

    x64 = x_cal.astype(np.float64)
    y_orig = x64 @ w64
    err_sparsegpt = np.sum((y_orig - x64 @ w_sparsegpt) ** 2)
    err_naive = np.sum((y_orig - x64 @ w_naive) ** 2)
    assert err_sparsegpt <= err_naive


def test_sparsegpt_pruning_requires_n_and_m_together():
    model = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_sparsegpt_pruning(model, n=2)
    with pytest.raises(ValueError):
        onnxsim.apply_sparsegpt_pruning(model, m=4)


# --- apply_structured_pruning ------------------------------------------------


def _mlp_model(K=8, H=32, Out=4, bias=True, activation="Relu", seed=0):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    if bias:
        b1 = rng.standard_normal((H,)).astype(np.float32)
        gemm1 = "h = Gemm(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        gemm1 = "h = MatMul(X, W1)"
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          {gemm1}
          a = {activation}(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=initializer,
    )


def _oracle_keep_indices(w1, keep_count):
    importance = np.linalg.norm(w1.T, axis=1)
    return np.sort(np.argsort(-importance)[:keep_count])


def test_structured_pruning_shrinks_matched_layers():
    model = _mlp_model(K=8, H=32, Out=4)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 16]
    assert list(inits["B1"].dims) == [16]
    assert list(inits["W2"].dims) == [16, 4]


def test_structured_pruning_matches_manual_channel_deletion_exactly():
    # The real correctness bar isn't "close to the float model" (removing
    # half the hidden units on random weights changes the output a lot,
    # by design) -- it's exact equivalence to deleting the same channels
    # by hand in numpy.
    K, H, Out = 8, 32, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=True)
    orig = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
    w1, b1, w2 = orig["W1"], orig["B1"], orig["W2"]

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    keep = _oracle_keep_indices(w1, H // 2)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    h = x @ w1[:, keep] + b1[keep]
    a = np.maximum(h, 0)
    y_oracle = a @ w2[keep, :]

    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_matmul_only_chain_matches_oracle():
    # No Gemm bias at all -- a plain MatMul -> activation -> MatMul chain.
    K, H, Out = 8, 24, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=False, activation="Sigmoid")
    w1 = onnx.numpy_helper.to_array(model.graph.initializer[0])
    w2 = onnx.numpy_helper.to_array(model.graph.initializer[1])

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.25)
    keep = _oracle_keep_indices(w1, H - round(H * 0.25))

    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    h = x @ w1[:, keep]
    a = 1.0 / (1.0 + np.exp(-h))
    y_oracle = a @ w2[keep, :]

    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_bias_add_between_matmuls_matches_oracle():
    # Bias as a separate Add node (not Gemm's own 3rd input) must be caught
    # by the elementwise chain-walk, not just Gemm's native bias slot.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(3)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    bias = rng.standard_normal((H,)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          hb = Add(h, Bias)
          a = Relu(hb)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(bias, "Bias"), _f32(w2, "W2")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Bias"].dims) == [H // 2]

    keep = _oracle_keep_indices(w1, H // 2)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    h = x @ w1[:, keep] + bias[keep]
    a = np.maximum(h, 0)
    y_oracle = a @ w2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_skips_branching_output():
    # h feeds both the Relu->MatMul chain *and* is itself a graph output --
    # pruning it would silently change what the caller observes, so this
    # must be left completely untouched.
    K, H, Out = 8, 16, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=False)
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("h", onnx.TensorProto.FLOAT, ["batch", H])
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H]
    assert list(inits["W2"].dims) == [H, Out]


def test_structured_pruning_skips_multi_consumer_branch():
    # h feeds two separate downstream MatMuls -- not the single-consumer
    # chain this pass proves safe to cut, so it must be left untouched.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(4)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    w3 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y1, float[batch,{Out}] Y2)
        {{
          h = MatMul(X, W1)
          Y1 = MatMul(h, W2)
          Y2 = MatMul(h, W3)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(w3, "W3")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H]


def test_structured_pruning_zero_sparsity_is_a_no_op():
    model = _mlp_model(K=8, H=16, Out=4)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.0)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 16]


def test_structured_pruning_invalid_sparsity_raises():
    model = _mlp_model(K=8, H=16, Out=4)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_pruning(model, sparsity=1.0)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_pruning(model, sparsity=-0.1)


def test_structured_pruning_chains_through_a_third_layer():
    # W2 is a producer for one chain (its own output channels feeding W3)
    # and a consumer for another (W1's output channels feeding into it) --
    # independent axes of the same tensor, both must be pruned correctly.
    K, H1, H2, Out = 8, 16, 20, 4
    rng = np.random.default_rng(5)
    w1 = rng.standard_normal((K, H1)).astype(np.float32)
    w2 = rng.standard_normal((H1, H2)).astype(np.float32)
    w3 = rng.standard_normal((H2, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h1 = MatMul(X, W1)
          a1 = Relu(h1)
          h2 = MatMul(a1, W2)
          a2 = Relu(h2)
          Y = MatMul(a2, W3)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(w3, "W3")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H1 // 2]
    assert list(inits["W2"].dims) == [H1 // 2, H2 // 2]
    assert list(inits["W3"].dims) == [H2 // 2, Out]

    keep1 = _oracle_keep_indices(w1, H1 // 2)
    keep2 = _oracle_keep_indices(w2[keep1, :], H2 // 2)

    rng2 = np.random.default_rng(6)
    x = rng2.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    a1 = np.maximum(x @ w1[:, keep1], 0)
    a2 = np.maximum(a1 @ w2[np.ix_(keep1, keep2)], 0)
    y_oracle = a2 @ w3[keep2, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- apply_structured_pruning: gated FFN (SwiGLU/GeGLU) ----------------------


def _combined_keep_indices(w_gate, w_up, keep_count):
    importance = np.sqrt(
        np.square(np.linalg.norm(w_gate.T, axis=1))
        + np.square(np.linalg.norm(w_up.T, axis=1))
    )
    return np.sort(np.argsort(-importance)[:keep_count])


def _swiglu_mlp_model(K=8, H=16, Out=4, gate_activation="Sigmoid", seed=0):
    rng = np.random.default_rng(seed)
    wg = rng.standard_normal((K, H)).astype(np.float32)
    wu = rng.standard_normal((K, H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, Wg)
          gate_act = {gate_activation}(gate)
          up = MatMul(X, Wu)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(wg, "Wg"), _f32(wu, "Wu"), _f32(wd, "Wd")],
    )
    return model, wg, wu, wd


def test_structured_pruning_gated_ffn_matches_oracle():
    K, H, Out = 8, 16, 4
    model, wg, wu, wd = _swiglu_mlp_model(K=K, H=H, Out=Out)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wg"].dims) == [K, H // 2]
    assert list(inits["Wu"].dims) == [K, H // 2]
    assert list(inits["Wd"].dims) == [H // 2, Out]

    keep = _combined_keep_indices(wg, wu, H // 2)
    rng = np.random.default_rng(10)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    gate = 1.0 / (1.0 + np.exp(-(x @ wg[:, keep])))
    up = x @ wu[:, keep]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_gated_ffn_prunes_both_branches_to_same_channels():
    # The real bug this pattern risks: gate and up disagreeing on which
    # channels survive, which would silently break the elementwise
    # product's alignment. Assert they select the identical index set,
    # not just that both shrank to the same *count*.
    K, H, Out = 8, 20, 4
    model, wg, wu, _ = _swiglu_mlp_model(K=K, H=H, Out=Out, seed=1)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.3)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    keep = _combined_keep_indices(wg, wu, H - round(H * 0.3))

    np.testing.assert_array_equal(inits["Wg"], wg[:, keep])
    np.testing.assert_array_equal(inits["Wu"], wu[:, keep])


def test_structured_pruning_gelu_gated_ffn_matches_oracle():
    # GeGLU: same gated topology, a different (still-unary) gate activation.
    # Uses Gelu's tanh approximation so the oracle needs no scipy/erf.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(11)
    wg = rng.standard_normal((K, H)).astype(np.float32)
    wu = rng.standard_normal((K, H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, Wg)
          gate_act = Gelu<approximate = "tanh">(gate)
          up = MatMul(X, Wu)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(wg, "Wg"), _f32(wu, "Wu"), _f32(wd, "Wd")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    keep = _combined_keep_indices(wg, wu, H // 2)

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    g = x @ wg[:, keep]
    gate = 0.5 * g * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (g + 0.044715 * g**3)))
    up = x @ wu[:, keep]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_structured_pruning_ungated_mul_of_two_producers_still_matches_oracle():
    # No activation at all on either branch -- a plain (unactivated) GLU,
    # both Mul operands are raw producer outputs directly.
    K, H, Out = 8, 12, 4
    rng = np.random.default_rng(2)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((K, H)).astype(np.float32)
    w3 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          h = Mul(a, b)
          Y = MatMul(h, W3)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(w3, "W3")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    keep = _combined_keep_indices(w1, w2, H // 2)

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    y_oracle = ((x @ w1[:, keep]) * (x @ w2[:, keep])) @ w3[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_gated_mul_against_constant_scale_is_not_a_gate():
    # Mul(a, constant) is the existing per-channel-scale chain continuation
    # (already covered elsewhere), not a two-producer gated pair -- the
    # constant operand must never be mistaken for a second producer.
    K, H, Out = 8, 12, 4
    rng = np.random.default_rng(3)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    scale = rng.standard_normal((H,)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          h = Mul(a, Scale)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(scale, "Scale"), _f32(w2, "W2")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H // 2]
    assert list(inits["Scale"].dims) == [H // 2]
    assert list(inits["W2"].dims) == [H // 2, Out]


def test_structured_pruning_gated_ffn_skips_when_a_branch_also_feeds_elsewhere():
    # "up" also feeding a second consumer directly means pruning its
    # channels would silently change what that other consumer sees --
    # must be left completely untouched, same bar as the plain-chain case.
    K, H, Out = 8, 12, 4
    rng = np.random.default_rng(4)
    wg = rng.standard_normal((K, H)).astype(np.float32)
    wu = rng.standard_normal((K, H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    wother = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y1, float[batch,{Out}] Y2)
        {{
          gate = MatMul(X, Wg)
          gate_act = Sigmoid(gate)
          up = MatMul(X, Wu)
          h = Mul(gate_act, up)
          Y1 = MatMul(h, Wd)
          Y2 = MatMul(up, Wother)
        }}
        """,
        initializer=[
            _f32(wg, "Wg"),
            _f32(wu, "Wu"),
            _f32(wd, "Wd"),
            _f32(wother, "Wother"),
        ],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wg"].dims) == [K, H]
    assert list(inits["Wu"].dims) == [K, H]
    assert list(inits["Wd"].dims) == [H, Out]


def test_structured_pruning_native_swiglu_node_prunes_both_producers_together():
    # ONNX's native fused SwiGLU(a, b) = swish(a) * b (opset 28+): the
    # activation lives entirely inside the op, so a/b must be raw producer
    # outputs with no separate activation node in between. Not yet
    # supported by the installed onnx checker/onnxruntime in this
    # environment (opset 28 is still under development upstream), so this
    # verifies the graph surgery directly via tensor values rather than
    # onnx.checker/onnxruntime execution.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(5)
    wg = rng.standard_normal((K, H)).astype(np.float32)
    wu = rng.standard_normal((K, H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, Wg)
          up = MatMul(X, Wu)
          h = SwiGLU(gate, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(wg, "Wg"), _f32(wu, "Wu"), _f32(wd, "Wd")],
        opset=28,
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    keep = _combined_keep_indices(wg, wu, H // 2)

    np.testing.assert_array_equal(inits["Wg"], wg[:, keep])
    np.testing.assert_array_equal(inits["Wu"], wu[:, keep])
    np.testing.assert_array_equal(inits["Wd"], wd[keep, :])


# --- Conv2D structured pruning ------------------------------------------


def _conv_pair_model(w1, w2, b1=None, spatial=10, activation="Relu"):
    Cin, C2 = w1.shape[1], w2.shape[0]
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    if b1 is not None:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1)"
    out_spatial = spatial - 4  # two valid (no-pad) 3x3 convs
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{out_spatial},{out_spatial}] Y)
        {{
          {conv1}
          a = {activation}(h)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=initializer,
    )


def _conv_model(Cin=3, C1=16, C2=8, bias=True, activation="Relu", seed=0, spatial=10):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32) if bias else None
    return _conv_pair_model(w1, w2, b1=b1, spatial=spatial, activation=activation)


def _oracle_keep_indices_conv(w, keep_count):
    importance = np.linalg.norm(w.reshape(w.shape[0], -1).astype(np.float64), axis=1)
    return np.sort(np.argsort(-importance)[:keep_count])


def _depthwise_pair_model(w1, dw_hops, w2, b1=None, spatial=10, activation="Relu"):
    """The Conv-chain oracle builder, extended with zero or more depthwise
    pass-through hops between producer and consumer: `dw_hops` is a list of
    ``(weight[C1, 1, kH, kW], bias_or_None)`` depthwise Convs (``group`` is
    always `weight.shape[0]`, so slicing `w1`/`dw_hops`/`w2` down together
    -- as every test below does for its own "oracle" call -- keeps every
    depthwise hop's `group` attribute consistent with its sliced weight for
    free). Each hop, like the producer, is followed by `activation`.
    """
    Cin, C2 = w1.shape[1], w2.shape[0]
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    if b1 is not None:
        lines = ["h0 = Conv<kernel_shape=[3,3]>(X, W1, B1)"]
        initializer.append(_f32(b1, "B1"))
    else:
        lines = ["h0 = Conv<kernel_shape=[3,3]>(X, W1)"]
    lines.append(f"a0 = {activation}(h0)")
    cur = "a0"
    n_convs = 1
    for i, (wd, bd) in enumerate(dw_hops):
        group = wd.shape[0]
        w_name, b_name = f"WD{i}", f"BD{i}"
        initializer.append(_f32(wd, w_name))
        if bd is not None:
            initializer.append(_f32(bd, b_name))
            lines.append(
                f"hd{i} = Conv<kernel_shape=[3,3], group={group}>"
                f"({cur}, {w_name}, {b_name})"
            )
        else:
            lines.append(
                f"hd{i} = Conv<kernel_shape=[3,3], group={group}>({cur}, {w_name})"
            )
        lines.append(f"ad{i} = {activation}(hd{i})")
        cur = f"ad{i}"
        n_convs += 1
    lines.append(f"Y = Conv<kernel_shape=[3,3]>({cur}, W2)")
    n_convs += 1
    out_spatial = spatial - 2 * n_convs  # each 3x3 valid conv shrinks by 2
    body = "\n          ".join(lines)
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{out_spatial},{out_spatial}] Y)
        {{
          {body}
        }}
        """,
        initializer=initializer,
    )


def test_structured_pruning_depthwise_pass_through_matches_manual_channel_deletion_exactly():
    # A MobileNet/EfficientNet-style inverted-residual block:
    # Conv(group=1) -> Relu -> DepthwiseConv(group=C1) -> Relu ->
    # Conv(group=1). The depthwise layer mixes no channels at all -- output
    # channel i depends only on input channel i -- so the chain walk must
    # cross it transparently: the same channel-index set the real
    # producer/consumer pair is pruned to also slices the depthwise layer's
    # own weight and bias, and shrinks its `group` attribute to match.
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(50)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    wd = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    bd = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _depthwise_pair_model(w1, [(wd, bd)], w2, b1=b1)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["WD0"].shape == (C1 // 2, 1, 3, 3)
    assert inits["BD0"].shape == (C1 // 2,)
    dw_node = next(n for n in pruned.graph.node if "WD0" in n.input)
    group_attr = next(a for a in dw_node.attribute if a.name == "group")
    assert group_attr.i == C1 // 2

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _depthwise_pair_model(
        w1[keep], [(wd[keep], bd[keep])], w2[:, keep], b1=b1[keep]
    )

    rng_x = np.random.default_rng(51)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_multiple_consecutive_depthwise_pass_through_hops_matches_oracle():
    # Two depthwise Convs back to back (e.g. a wider spatial receptive
    # field built from stacked depthwise layers) -- both must be crossed
    # transparently by the same channel-index set, each sliced and
    # re-grouped independently. The second hop also has no bias, folding in
    # that case too.
    Cin, C1, C2 = 3, 12, 6
    rng = np.random.default_rng(52)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    wd1 = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    bd1 = rng.standard_normal((C1,)).astype(np.float32)
    wd2 = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _depthwise_pair_model(w1, [(wd1, bd1), (wd2, None)], w2, spatial=14)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.25)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv(w1, C1 - round(C1 * 0.25))
    oracle = _depthwise_pair_model(
        w1[keep], [(wd1[keep], bd1[keep]), (wd2[keep], None)], w2[:, keep], spatial=14
    )

    rng_x = np.random.default_rng(53)
    x = rng_x.standard_normal((2, Cin, 14, 14)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_depthwise_pass_through_no_bias_matches_oracle():
    # A depthwise hop with no bias at all -- its own [C1, 1, kH, kW] weight
    # is the only thing that needs slicing for it.
    Cin, C1, C2 = 4, 10, 5
    rng = np.random.default_rng(54)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    wd = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _depthwise_pair_model(w1, [(wd, None)], w2, activation="Sigmoid")

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.3)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv(w1, C1 - round(C1 * 0.3))
    oracle = _depthwise_pair_model(
        w1[keep], [(wd[keep], None)], w2[:, keep], activation="Sigmoid"
    )

    rng_x = np.random.default_rng(55)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_depthwise_pass_through_branch_is_left_untouched():
    # A depthwise Conv whose output feeds more than one consumer (a
    # branch) can't be crossed transparently either -- doing so would mean
    # picking one branch to carry the chain forward while silently leaving
    # the other reading a now-stale channel count. Left untouched, same as
    # any other branching point this pass declines to guess at (the same
    # single-consumer requirement every other hop in this pass already
    # holds every intermediate tensor to).
    Cin, C1 = 3, 8
    rng = np.random.default_rng(56)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    wd = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C1, C1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C1},4,4] Y1, float[N,{C1},6,6] Y2)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = Relu(h)
          d = Conv<kernel_shape=[3,3], group={C1}>(a, WD)
          Y1 = Conv<kernel_shape=[3,3]>(d, W2)
          Y2 = Relu(d)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(wd, "WD"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["WD"], wd)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_structured_pruning_conv_chain_shrinks_matched_layers():
    Cin, C1, C2 = 3, 16, 8
    model = _conv_model(Cin=Cin, C1=C1, C2=C2, bias=True)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [C1 // 2, Cin, 3, 3]
    assert list(inits["B1"].dims) == [C1 // 2]
    assert list(inits["W2"].dims) == [C2, C1 // 2, 3, 3]


def test_structured_pruning_conv_chain_matches_manual_channel_deletion_exactly():
    # Same correctness bar as the MatMul/Gemm chain tests: exact
    # equivalence to deleting the same output filters by hand, not just
    # "close to the float model". Conv has no simple numpy one-liner
    # standing in for the op itself, so the oracle is a second, smaller
    # ONNX graph built directly from the same sliced weights and run
    # through onnxruntime, rather than hand-rolled conv math.
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(30)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2, b1=b1)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _conv_pair_model(w1[keep], w2[:, keep], b1=b1[keep])

    rng_x = np.random.default_rng(31)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_conv_only_chain_matches_oracle_no_bias():
    # No Conv bias at all, and a non-Relu activation -- a plain
    # Conv -> Sigmoid -> Conv chain.
    Cin, C1, C2 = 4, 12, 6
    rng = np.random.default_rng(32)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2, activation="Sigmoid")

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.25)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv(w1, C1 - round(C1 * 0.25))
    oracle = _conv_pair_model(w1[keep], w2[:, keep], activation="Sigmoid")

    rng_x = np.random.default_rng(33)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_skips_grouped_producer_conv():
    # A depthwise Conv (group == in_channels == out_channels) is never
    # itself matched as a producer -- it's only ever a transparent
    # pass-through hop the chain walk may cross between two real
    # producer/consumer boundaries (see the "depthwise_pass_through" tests
    # above). With nothing upstream of it here, there's no real producer to
    # anchor a chain at all, so both layers stay completely untouched, even
    # though the topology otherwise looks identical to a matched pair.
    C = 8
    rng = np.random.default_rng(34)
    w1 = rng.standard_normal((C, 1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C, C, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{C},10,10] X) => (float[N,{C},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3], group={C}>(X, W1)
          a = Relu(h)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_structured_pruning_skips_grouped_consumer_conv():
    # A depthwise Conv is likewise never matched as a *consumer* -- when
    # its own output feeds a graph output (as here) rather than a further
    # real Conv, crossing it as a pass-through hop simply runs out of chain
    # to walk (see "depthwise Conv ... last node before a graph output" in
    # this module's own docstring), so the walk finds no real consumer and
    # the whole chain -- producer included -- is left untouched.
    Cin, C1 = 3, 8
    rng = np.random.default_rng(35)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)  # depthwise consumer
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C1},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = Relu(h)
          Y = Conv<kernel_shape=[3,3], group={C1}>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_structured_pruning_skips_general_grouped_producer_conv():
    # A *general* grouped Conv (group=2, neither 1 nor equal to its own
    # channel count) is not the depthwise special case above -- it stays
    # completely out of scope, exactly as before this pass learned to cross
    # depthwise Convs transparently.
    C = 8
    rng = np.random.default_rng(57)
    w1 = rng.standard_normal((C, C // 2, 3, 3)).astype(np.float32)  # group=2
    w2 = rng.standard_normal((C, C, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{C},10,10] X) => (float[N,{C},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3], group=2>(X, W1)
          a = Relu(h)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_structured_pruning_skips_general_grouped_consumer_conv():
    # A general group=2 Conv sitting *between* a real producer and a real
    # further consumer must also stay a hard stop, not a transparent hop --
    # only the depthwise case (group == channel count, weight [C, 1, kH,
    # kW]) is safe to cross, since only there is every output channel tied
    # 1:1 to a single input channel with no cross-channel mixing at all.
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(58)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C1, C1 // 2, 3, 3)).astype(np.float32)  # group=2
    w3 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},14,14] X) => (float[N,{C2},8,8] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = Relu(h)
          m = Conv<kernel_shape=[3,3], group=2>(a, W2)
          b = Relu(m)
          Y = Conv<kernel_shape=[3,3]>(b, W3)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(w3, "W3")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["W3"], w3)


def test_structured_pruning_conv_into_non_pass_through_op_is_left_untouched():
    # An ordinary CNN classifier tail: Conv -> GlobalAveragePool -> Flatten
    # -> MatMul head. Neither pooling nor flattening is a shape-preserving
    # elementwise op the chain walk recognizes, so the Conv producer is
    # left completely untouched rather than matched to the MatMul by
    # coincidence of a downstream reduction dimension.
    Cin, C1, Out = 3, 8, 4
    rng = np.random.default_rng(36)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C1, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Out}] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          p = GlobalAveragePool(h)
          f = Flatten<axis=1>(p)
          Y = MatMul(f, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_structured_pruning_conv_chain_scale_between_convs_is_left_untouched():
    # A per-channel Mul (e.g. an un-fused BatchNormalization's scale, or a
    # standalone SE-style gate) between two Convs isn't recognized -- unlike
    # the MatMul/Gemm chain walk, which does allow Add/Mul against a
    # per-channel constant. See this module's own docstring for why Conv
    # chains restrict to unary activations only (a real Conv already
    # carries its own bias, and onnxsim's own default optimization fuses
    # BatchNormalization into the preceding Conv before this pass would
    # ever see it).
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(37)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    scale = rng.standard_normal((1, C1, 1, 1)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          s = Mul(h, Scale)
          a = Relu(s)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(scale, "Scale"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_structured_wanda_pruning_conv_chain_matches_oracle_exactly():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(40)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, None)
    )

    rng_cal = np.random.default_rng(41)
    x_cal = rng_cal.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    _, a_cal = _run(probe_model, {"X": x_cal})
    # Reduce over every axis but the channel one (axis 1 of NCHW) -- the
    # Conv analogue of the MatMul/Gemm oracle's last-axis reduction above.
    act_norm = np.sqrt(np.mean(np.square(a_cal.astype(np.float64)), axis=(0, 2, 3)))
    importance = np.linalg.norm(
        w1.reshape(C1, -1).astype(np.float64), axis=1
    ) * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[: C1 // 2])

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    oracle = _conv_pair_model(w1[keep], w2[:, keep])
    rng_x = np.random.default_rng(42)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_depthwise_pass_through_matches_oracle_exactly():
    # Same oracle bar with a depthwise hop in the middle: the calibrated
    # activation norm is captured right where the chain feeds its *real*
    # consumer -- i.e. downstream of the (transparent) depthwise hop, not
    # at the real producer's own raw output -- since a depthwise Conv
    # contributes no importance of its own to the ranking.
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(70)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    wd = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    bd = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _depthwise_pair_model(w1, [(wd, bd)], w2)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("ad0", onnx.TensorProto.FLOAT, None)
    )

    rng_cal = np.random.default_rng(71)
    x_cal = rng_cal.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    _, ad0_cal = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(ad0_cal.astype(np.float64)), axis=(0, 2, 3)))
    importance = np.linalg.norm(
        w1.reshape(C1, -1).astype(np.float64), axis=1
    ) * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[: C1 // 2])

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    oracle = _depthwise_pair_model(w1[keep], [(wd[keep], bd[keep])], w2[:, keep])
    rng_x = np.random.default_rng(72)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- apply_structured_wanda_pruning ------------------------------------------


def _kept_columns(pruned_model, weight_name, original_w):
    w_pruned = onnx.numpy_helper.to_array(
        next(t for t in pruned_model.graph.initializer if t.name == weight_name)
    )
    kept = []
    for j in range(original_w.shape[1]):
        col = original_w[:, j]
        if any(np.array_equal(col, w_pruned[:, jj]) for jj in range(w_pruned.shape[1])):
            kept.append(j)
    return kept


def test_structured_wanda_pruning_protects_channels_with_small_weight_but_large_activation():
    # A structured analogue of Wanda's own motivating scenario: a hidden
    # unit whose own weight column is deliberately *smaller* than typical
    # (so plain L2-norm structured pruning ranks it lowest and cuts it),
    # but which is wired to an input feature that calibration data makes
    # consistently huge -- its actual contribution to the network is large
    # even though its weight norm alone doesn't show it.
    K, H, Out = 8, 32, 4
    salient = (3, 7, 20)
    k0 = 0
    rng = np.random.default_rng(20)
    w1 = rng.standard_normal((K, H)).astype(np.float32) * 0.5
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    non_salient = [j for j in range(H) if j not in salient]
    w1[k0, non_salient] = 0.0  # only salient channels respond to k0 at all
    small_scale = 0.4
    for j in salient:
        w1[:, j] = 0.0
        w1[k0, j] = small_scale  # weight norm well below the ~1.4 typical column

    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          a = Relu(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )

    x = rng.standard_normal((64, K)).astype(np.float32)
    x[:, k0] *= 40.0
    calibration_data = [{"X": x}]

    plain = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    wanda = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(plain)
    onnx.checker.check_model(wanda)

    plain_kept = _kept_columns(plain, "W1", w1)
    wanda_kept = _kept_columns(wanda, "W1", w1)
    assert all(j not in plain_kept for j in salient)
    assert all(j in wanda_kept for j in salient)


def test_structured_wanda_pruning_matches_oracle_exactly():
    K, H, Out = 8, 24, 4
    rng = np.random.default_rng(21)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{Out}] Y)
        {{
          h = MatMul(X, W1)
          a = Relu(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )

    rng_cal = np.random.default_rng(22)
    x_cal = rng_cal.standard_normal((2, 16, K)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    a_cal = np.maximum(x_cal.reshape(-1, K) @ w1, 0)
    act_norm = np.sqrt(np.mean(np.square(a_cal), axis=0))
    importance = np.linalg.norm(w1.T, axis=1) * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[: H // 2])

    x = rng_cal.standard_normal((3, 5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    h = np.maximum(x @ w1[:, keep], 0)
    y_oracle = h @ w2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_falls_back_to_plain_with_no_calibration_batches():
    K, H, Out = 8, 16, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=False)

    plain = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    wanda = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    inits_plain = {
        t.name: onnx.numpy_helper.to_array(t) for t in plain.graph.initializer
    }
    inits_wanda = {
        t.name: onnx.numpy_helper.to_array(t) for t in wanda.graph.initializer
    }
    np.testing.assert_array_equal(inits_plain["W1"], inits_wanda["W1"])
    np.testing.assert_array_equal(inits_plain["W2"], inits_wanda["W2"])


def test_structured_wanda_pruning_gated_ffn_matches_oracle():
    K, H, Out = 8, 16, 4
    model, wg, wu, wd = _swiglu_mlp_model(K=K, H=H, Out=Out, seed=23)

    rng = np.random.default_rng(24)
    x_cal = rng.standard_normal((32, K)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["Wg"].shape == (K, H // 2)
    assert inits["Wu"].shape == (K, H // 2)
    assert inits["Wd"].shape == (H // 2, Out)

    # Both branches must still select the identical channel-index set.
    kept_g = _kept_columns(pruned, "Wg", wg)
    kept_u = _kept_columns(pruned, "Wu", wu)
    assert kept_g == kept_u

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    keep = kept_g
    gate = 1.0 / (1.0 + np.exp(-(x @ wg[:, keep])))
    up = x @ wu[:, keep]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_zero_sparsity_is_a_no_op():
    model = _mlp_model(K=8, H=16, Out=4)
    pruned = onnxsim.apply_structured_wanda_pruning(model, sparsity=0.0)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 16]


def test_structured_wanda_pruning_invalid_sparsity_raises():
    model = _mlp_model(K=8, H=16, Out=4)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_wanda_pruning(model, sparsity=1.0)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_wanda_pruning(model, sparsity=-0.1)


# --- apply_attention_head_pruning ---------------------------------------


def _attention_model(
    K=8,
    H=4,
    D=4,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    bias=True,
    with_reshape=False,
    wqkv=None,
    bqkv=None,
    wout=None,
    num_heads=None,
):
    rng = np.random.default_rng(seed)
    Nq = Nk = Nv = H * D
    if wqkv is None:
        wqkv = rng.standard_normal((K, Nq + Nk + Nv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nv, Out)).astype(np.float32)
    if bias and bqkv is None:
        bqkv = rng.standard_normal((Nq + Nk + Nv,)).astype(np.float32)
    heads = H if num_heads is None else num_heads

    initializer = [_f32(wqkv, "Wqkv"), _f32(wout, "Wout")]
    qkv_inputs = "X, Wqkv"
    if bias:
        initializer.append(_f32(bqkv, "Bqkv"))
        qkv_inputs = "X, Wqkv, Bqkv"

    if with_reshape:
        shape = np.array([batch, seq, Nv], dtype=np.int64)
        initializer.append(onnx.numpy_helper.from_array(shape, "Shape"))
        tail = "ctx2 = Reshape(ctx, Shape)\n          Y = MatMul(ctx2, Wout)"
    else:
        tail = "Y = MatMul(ctx, Wout)"

    body = f"""
        g ({K} X) => ({Out} Y)
        {{
          ctx = com.microsoft.Attention <num_heads={heads}, qkv_hidden_sizes=[{Nq},{Nk},{Nv}]> ({qkv_inputs})
          {tail}
        }}
        """
    # Substitute the actual rank-3 shapes by hand -- `_model`'s own f-string
    # convention assumes a 2-D-only [batch, dim] input/output signature.
    body = body.replace(f"({K} X)", f"(float[batch,seq,{K}] X)")
    body = body.replace(f"({Out} Y)", f"(float[batch,seq,{Out}] Y)")

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K, H=H, D=D, Out=Out, Nq=Nq, Nk=Nk, Nv=Nv, wqkv=wqkv, bqkv=bqkv, wout=wout
    )


def _attention_node(model):
    return next(n for n in model.graph.node if n.op_type == "Attention")


def _attention_attrs(node):
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    qkv = next(list(a.ints) for a in node.attribute if a.name == "qkv_hidden_sizes")
    return num_heads, qkv


def _oracle_keep_heads(wqkv, nq, nk, nv, num_heads, keep_count):
    dq, dk, dv = nq // num_heads, nk // num_heads, nv // num_heads
    wq, wk, wv = wqkv[:, :nq], wqkv[:, nq : nq + nk], wqkv[:, nq + nk :]
    importance = np.zeros(num_heads)
    for h in range(num_heads):
        block = np.concatenate(
            [
                wq[:, h * dq : (h + 1) * dq],
                wk[:, h * dk : (h + 1) * dk],
                wv[:, h * dv : (h + 1) * dv],
            ],
            axis=1,
        )
        importance[h] = np.linalg.norm(block)
    return np.sort(np.argsort(-importance)[:keep_count])


def _head_idx(keep_heads, d):
    return np.concatenate([np.arange(h * d, (h + 1) * d) for h in keep_heads])


def test_attention_head_pruning_shrinks_matched_block():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _attention_node(pruned)
    num_heads, qkv = _attention_attrs(node)
    assert num_heads == 2
    assert qkv == [8, 8, 8]

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wqkv"].dims) == [8, 24]
    assert list(inits["Bqkv"].dims) == [24]
    assert list(inits["Wout"].dims) == [8, 6]


def test_attention_head_pruning_matches_manual_head_deletion_exactly():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=1)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_heads(cfg["wqkv"], cfg["Nq"], cfg["Nk"], cfg["Nv"], cfg["H"], 2)
    d = cfg["D"]
    qi, ki, vi = (
        _head_idx(keep, d),
        _head_idx(keep, d) + cfg["Nq"],
        _head_idx(keep, d) + cfg["Nq"] + cfg["Nk"],
    )
    all_idx = np.concatenate([qi, ki, vi])
    oracle_wqkv = cfg["wqkv"][:, all_idx]
    oracle_bqkv = cfg["bqkv"][all_idx]
    oracle_wout = cfg["wout"][_head_idx(keep, d), :]
    oracle, _ = _attention_model(
        K=cfg["K"],
        H=2,
        D=d,
        Out=cfg["Out"],
        seed=1,
        wqkv=oracle_wqkv,
        bqkv=oracle_bqkv,
        wout=oracle_wout,
        num_heads=2,
    )

    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 5, cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_attention_head_pruning_reshape_hop_is_recognized_and_shape_updated():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=3, with_reshape=True)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.25)
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node] == ["Attention", "Reshape", "MatMul"]

    node = _attention_node(pruned)
    num_heads, _ = _attention_attrs(node)
    assert num_heads == 3  # round(4 - 4*0.25) == 3

    shape = onnx.numpy_helper.to_array(
        next(t for t in pruned.graph.initializer if t.name == "Shape")
    )
    assert shape[-1] == 3 * cfg["D"]  # updated to the new (post-prune) Nv

    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 5, cfg["K"])).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    assert y.shape == (2, 5, cfg["Out"])


def test_attention_head_pruning_group_query_attention_missing_required_inputs_is_left_untouched():
    # GroupQueryAttention is supported (see the "-- GroupQueryAttention"
    # section below), but its schema requires seqlens_k/
    # total_sequence_length even for a plain forward pass -- a node missing
    # them (as here, only q/k/v given) isn't a complete/safe-to-act-on GQA
    # node and must not be mistaken for one, nor for a plain `Attention`
    # node (whose merged-QKV-weight shape this one doesn't have either).
    K, H, D, Out = 8, 4, 4, 6
    Nqkv = H * D
    rng = np.random.default_rng(5)
    wq = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wk = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wv = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wout = rng.standard_normal((Nqkv, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[2,5,{K}] X) => (float[2,5,{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx, present_k, present_v = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={H}> (q, k, v)
          Y = MatMul(ctx, Wout)
        }}
        """,
        initializer=[
            _f32(wq, "Wq"),
            _f32(wk, "Wk"),
            _f32(wv, "Wv"),
            _f32(wout, "Wout"),
        ],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_attention_head_pruning_mismatched_consumer_reduction_dim_is_left_untouched():
    K, H, D, Out = 8, 4, 4, 6
    Nqkv = H * D
    rng = np.random.default_rng(6)
    wqkv = rng.standard_normal((K, 3 * Nqkv)).astype(np.float32)
    wout_wrong = rng.standard_normal((Nqkv + 1, Out)).astype(np.float32)  # off-by-one
    model = _model(
        f"""
        g (float[2,5,{K}] X) => (float[2,5,{Out}] Y)
        {{
          ctx = com.microsoft.Attention <num_heads={H}, qkv_hidden_sizes=[{Nqkv},{Nqkv},{Nqkv}]> (X, Wqkv)
          padded = Pad <pads = [0,0,0,0,0,1]> (ctx)
          Y = MatMul(padded, Wout)
        }}
        """,
        initializer=[_f32(wqkv, "Wqkv"), _f32(wout_wrong, "Wout")],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_attention_head_pruning_zero_sparsity_is_a_no_op():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=7)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.0)
    node = _attention_node(pruned)
    num_heads, qkv = _attention_attrs(node)
    assert num_heads == cfg["H"]
    assert qkv == [cfg["Nq"], cfg["Nk"], cfg["Nv"]]
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wqkv"], cfg["wqkv"])


def test_attention_head_pruning_invalid_sparsity_raises():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6)
    with pytest.raises(ValueError):
        onnxsim.apply_attention_head_pruning(model, sparsity=1.0)
    with pytest.raises(ValueError):
        onnxsim.apply_attention_head_pruning(model, sparsity=-0.1)


# --- apply_attention_head_wanda_pruning ---------------------------------


def test_attention_head_wanda_pruning_matches_oracle_exactly():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=8)

    rng = np.random.default_rng(9)
    x_cal = rng.standard_normal((3, 6, cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    # Reproduce the calibrated importance from scratch: probe `ctx` (the
    # Attention node's own raw output, exactly what the consumer MatMul
    # reads here since there is no Reshape hop), reduce over every axis but
    # the channel one, combine per-head via root-sum-square, and multiply
    # into the plain Frobenius-norm weight importance.
    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(onnx.ValueInfoProto(name="ctx"))
    (_, ctx_cal) = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(ctx_cal.astype(np.float64)), axis=(0, 1)))

    d = cfg["D"]
    wq, wk, wv = (
        cfg["wqkv"][:, : cfg["Nq"]],
        cfg["wqkv"][:, cfg["Nq"] : cfg["Nq"] + cfg["Nk"]],
        cfg["wqkv"][:, cfg["Nq"] + cfg["Nk"] :],
    )
    importance = np.zeros(cfg["H"])
    for h in range(cfg["H"]):
        block = np.concatenate(
            [
                wq[:, h * d : (h + 1) * d],
                wk[:, h * d : (h + 1) * d],
                wv[:, h * d : (h + 1) * d],
            ],
            axis=1,
        )
        act_head = np.linalg.norm(act_norm[h * d : (h + 1) * d])
        importance[h] = np.linalg.norm(block) * max(act_head, 1e-8)
    keep = np.sort(np.argsort(-importance)[:2])

    qi, ki, vi = (
        _head_idx(keep, d),
        _head_idx(keep, d) + cfg["Nq"],
        _head_idx(keep, d) + cfg["Nq"] + cfg["Nk"],
    )
    all_idx = np.concatenate([qi, ki, vi])
    oracle, _ = _attention_model(
        K=cfg["K"],
        H=2,
        D=d,
        Out=cfg["Out"],
        seed=8,
        wqkv=cfg["wqkv"][:, all_idx],
        bqkv=cfg["bqkv"][all_idx],
        wout=cfg["wout"][_head_idx(keep, d), :],
        num_heads=2,
    )

    x = rng.standard_normal((2, 5, cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_attention_head_wanda_pruning_falls_back_to_plain_with_no_calibration_batches():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=10)
    plain = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    wanda = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    inits_plain = {
        t.name: onnx.numpy_helper.to_array(t) for t in plain.graph.initializer
    }
    inits_wanda = {
        t.name: onnx.numpy_helper.to_array(t) for t in wanda.graph.initializer
    }
    for name in inits_plain:
        np.testing.assert_array_equal(inits_plain[name], inits_wanda[name])


# --- apply_attention_head_pruning / _wanda_pruning -- GroupQueryAttention --


def _gqa_model(
    K=8,
    H=4,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    bias=False,
    with_reshape=False,
    wq=None,
    wk=None,
    wv=None,
    bq=None,
    bk=None,
    bv=None,
    wout=None,
    past_kv=None,  # None (empty) | "nonempty" (constant) | "dynamic" (graph input)
):
    # Real ONNX Runtime CPU kernels for GroupQueryAttention require
    # head_size to be a multiple of 8 (verified empirically -- a smaller
    # head_size segfaults/errors at run time the same way a 2-input
    # com.microsoft::Attention does elsewhere in this file), so D defaults
    # to 8 rather than mirroring _attention_model's smaller default.
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nkv)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    q_op, k_op, v_op = "MatMul(X, Wq)", "MatMul(X, Wk)", "MatMul(X, Wv)"
    if bias:
        if bq is None:
            bq = rng.standard_normal((Nq,)).astype(np.float32)
        if bk is None:
            bk = rng.standard_normal((Nkv,)).astype(np.float32)
        if bv is None:
            bv = rng.standard_normal((Nkv,)).astype(np.float32)
        initializer += [_f32(bq, "Bq"), _f32(bk, "Bk"), _f32(bv, "Bv")]
        q_op, k_op, v_op = "Gemm(X, Wq, Bq)", "Gemm(X, Wk, Bk)", "Gemm(X, Wv, Bv)"

    # seqlens_k/total_sequence_length: mandatory KV-cache bookkeeping inputs
    # GroupQueryAttention's schema requires even for a plain, no-cache
    # forward pass (see fuse_gqa.h's own top comment) -- `S-1` per batch row
    # and `S`, exactly what fuse_gqa.h itself synthesizes.
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )

    operands = ["q", "k", "v"]
    extra_graph_inputs = ""
    if past_kv == "nonempty":
        past_key = rng.standard_normal((batch, KVH, 1, D)).astype(np.float32)
        past_value = rng.standard_normal((batch, KVH, 1, D)).astype(np.float32)
        initializer += [_f32(past_key, "PastKey"), _f32(past_value, "PastValue")]
        operands += ["PastKey", "PastValue"]
    elif past_kv == "dynamic":
        operands += ["PastKeyIn", "PastValueIn"]
        extra_graph_inputs = (
            f", float[{batch},{KVH},1,{D}] PastKeyIn"
            f", float[{batch},{KVH},1,{D}] PastValueIn"
        )
    else:
        operands += ["", ""]
    operands += ["SeqLensK", "TotalSeq"]

    if with_reshape:
        shape = np.array([batch, seq, Nq], dtype=np.int64)
        initializer.append(onnx.numpy_helper.from_array(shape, "Shape"))
        tail = "ctx2 = Reshape(ctx, Shape)\n          Y = MatMul(ctx2, Wout)"
    else:
        tail = "Y = MatMul(ctx, Wout)"

    body = f"""
        g (float[{batch},{seq},{K}] X{extra_graph_inputs}) => (float[{batch},{seq},{Out}] Y)
        {{
          q = {q_op}
          k = {k_op}
          v = {v_op}
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> ({", ".join(operands)})
          {tail}
        }}
        """

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        Nq=Nq,
        Nkv=Nkv,
        wq=wq,
        wk=wk,
        wv=wv,
        bq=bq,
        bk=bk,
        bv=bv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def _gqa_node(model):
    return next(n for n in model.graph.node if n.op_type == "GroupQueryAttention")


def _gqa_attrs(node):
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return num_heads, kv_num_heads


def _oracle_keep_groups(wq, wk, wv, num_heads, kv_num_heads, head_size, keep_count):
    group_size = num_heads // kv_num_heads
    importance = np.zeros(kv_num_heads)
    for kv in range(kv_num_heads):
        q_block = np.concatenate(
            [
                wq[:, h * head_size : (h + 1) * head_size]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = wk[:, kv * head_size : (kv + 1) * head_size]
        v_block = wv[:, kv * head_size : (kv + 1) * head_size]
        importance[kv] = np.linalg.norm(
            np.concatenate([q_block, k_block, v_block], axis=1)
        )
    return np.sort(np.argsort(-importance)[:keep_count])


def _group_q_heads(keep_groups, group_size):
    return np.concatenate(
        [np.arange(g * group_size, (g + 1) * group_size) for g in keep_groups]
    )


def test_gqa_pruning_shrinks_matched_block():
    model, cfg = _gqa_model(K=8, H=4, KVH=4, D=8, Out=6)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert num_heads == 2  # round(4 - 4*0.5) query heads ...
    assert kv_num_heads == 2  # ... and KV heads alike, since group_size == 1 here

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [8, 16]
    assert list(inits["Wk"].dims) == [8, 16]
    assert list(inits["Wv"].dims) == [8, 16]
    assert list(inits["Wout"].dims) == [16, 6]


def test_gqa_pruning_zero_sparsity_is_a_no_op():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=7)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.0)
    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert num_heads == cfg["H"]
    assert kv_num_heads == cfg["KVH"]
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"])


def test_gqa_pruning_unequal_heads_drops_whole_groups_and_preserves_ratio():
    # 8 query heads sharing 4 KV heads (2 query heads per KV head); at
    # sparsity=0.5 two of the four *groups* must be dropped, never an
    # individual query head in isolation -- confirmed here by checking the
    # surviving Wq columns are exactly the two kept groups' own contiguous
    # 2-head blocks, matching the kept Wk/Wv columns' own group indices.
    model, cfg = _gqa_model(K=8, H=8, KVH=4, D=8, Out=6, seed=11)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 2
    assert num_heads == 4
    assert num_heads // kv_num_heads == cfg["H"] // cfg["KVH"]  # ratio preserved

    group_size = cfg["H"] // cfg["KVH"]
    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], 2
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"][:, _head_idx(keep_q_heads, d)])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"][:, _head_idx(keep_groups, d)])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"][:, _head_idx(keep_groups, d)])


def test_gqa_pruning_matches_oracle_exactly():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=1)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1  # max(1, 2 - round(2*0.5))
    assert num_heads == kv_num_heads * group_size

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    oracle, _ = _gqa_model(
        K=cfg["K"],
        H=num_heads,
        KVH=kv_num_heads,
        D=d,
        Out=cfg["Out"],
        seed=1,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    rng = np.random.default_rng(2)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_gqa_pruning_slices_bias_when_producer_has_one():
    # Gemm's own ONNX spec requires a rank-2 input, so a bias-carrying Gemm
    # producer can't sit directly ahead of GroupQueryAttention's rank-3
    # query/key/value inputs in a graph meant to actually run through
    # onnxruntime -- this exercises the bias-slicing path itself (shared
    # with every other producer match in this module via `_match_producer`)
    # directly against the initializers instead.
    model, cfg = _gqa_model(K=8, H=4, KVH=2, D=8, Out=6, seed=14, bias=True)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1
    assert num_heads == group_size

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"][:, kv_idx])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"][:, kv_idx])
    np.testing.assert_array_equal(inits["Bq"], cfg["bq"][q_idx])
    np.testing.assert_array_equal(inits["Bk"], cfg["bk"][kv_idx])
    np.testing.assert_array_equal(inits["Bv"], cfg["bv"][kv_idx])


def test_gqa_pruning_reshape_hop_is_recognized_and_shape_updated():
    model, cfg = _gqa_model(K=8, H=4, KVH=2, D=8, Out=6, seed=3, with_reshape=True)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node] == [
        "MatMul",
        "MatMul",
        "MatMul",
        "GroupQueryAttention",
        "Reshape",
        "MatMul",
    ]

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1  # max(1, 2 - round(2*0.5)) == 1
    assert num_heads == 2  # group_size(2) * kv_num_heads(1)

    shape = onnx.numpy_helper.to_array(
        next(t for t in pruned.graph.initializer if t.name == "Shape")
    )
    assert shape[-1] == num_heads * cfg["D"]  # updated to the new (post-prune) Nq

    rng = np.random.default_rng(4)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    assert y.shape == (cfg["batch"], cfg["seq"], cfg["Out"])


def test_gqa_pruning_nonempty_past_kv_constant_is_left_untouched():
    # A non-empty constant past_key/past_value holds real per-KV-head cache
    # data along the kv_num_heads axis that this module would need to slice
    # but doesn't attempt to -- declined outright rather than corrupted.
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=12, past_kv="nonempty")
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_gqa_pruning_dynamic_past_kv_input_is_still_pruned():
    # A *dynamic* (non-constant) past_key/past_value -- an ordinary graph
    # input here, standing in for real runtime KV-cache data -- is not a
    # weight this module could corrupt by leaving it untouched, so it must
    # not block the match the way a non-empty constant does.
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=13, past_kv="dynamic")
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1
    assert num_heads == 4
    assert list(node.input[3:5]) == ["PastKeyIn", "PastValueIn"]

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [8, 4 * cfg["D"]]
    assert list(inits["Wk"].dims) == [8, 1 * cfg["D"]]
    assert list(inits["Wv"].dims) == [8, 1 * cfg["D"]]


def test_gqa_wanda_pruning_matches_oracle_exactly():
    # Calibration and eval data must share the model's own fixed batch/seq
    # here (unlike the plain-Attention wanda test, which uses symbolic
    # batch/seq dims): seqlens_k/total_sequence_length are baked-in
    # constants tied to a specific batch/seq (see _gqa_model), a real
    # constraint of GroupQueryAttention's own KV-cache-bookkeeping inputs,
    # not a limitation of this pass.
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=8)

    rng = np.random.default_rng(9)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(onnx.ValueInfoProto(name="ctx"))
    (_, ctx_cal) = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(ctx_cal.astype(np.float64)), axis=(0, 1)))

    d = cfg["D"]
    group_size = cfg["H"] // cfg["KVH"]
    importance = np.zeros(cfg["KVH"])
    for kv in range(cfg["KVH"]):
        q_block = np.concatenate(
            [
                cfg["wq"][:, h * d : (h + 1) * d]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = cfg["wk"][:, kv * d : (kv + 1) * d]
        v_block = cfg["wv"][:, kv * d : (kv + 1) * d]
        base = np.linalg.norm(np.concatenate([q_block, k_block, v_block], axis=1))
        act_group = np.linalg.norm(
            act_norm[kv * group_size * d : (kv + 1) * group_size * d]
        )
        importance[kv] = base * max(act_group, 1e-8)
    keep_groups = np.sort(np.argsort(-importance)[:1])  # max(1, 2 - round(2*0.5)) == 1

    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    oracle, _ = _gqa_model(
        K=cfg["K"],
        H=len(keep_q_heads),
        KVH=len(keep_groups),
        D=d,
        Out=cfg["Out"],
        seed=8,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_gqa_wanda_pruning_falls_back_to_plain_with_no_calibration_batches():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=10)
    plain = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    wanda = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    inits_plain = {
        t.name: onnx.numpy_helper.to_array(t) for t in plain.graph.initializer
    }
    inits_wanda = {
        t.name: onnx.numpy_helper.to_array(t) for t in wanda.graph.initializer
    }
    for name in inits_plain:
        np.testing.assert_array_equal(inits_plain[name], inits_wanda[name])


def test_attention_head_pruning_handles_attention_and_gqa_in_one_model():
    # Regression check for _apply_attention_chains's per-chain-type
    # dispatch: a plain `Attention` block and a `GroupQueryAttention` block
    # in the same graph, sharing no tensors, must each be pruned correctly
    # and independently -- one chain family must not disturb the other.
    K, H, D, Out1 = 8, 4, 4, 6
    Nqkv = H * D
    rng = np.random.default_rng(30)
    wqkv = rng.standard_normal((K, 3 * Nqkv)).astype(np.float32)
    # A real onnxruntime CPU build's `Attention` kernel can segfault given
    # only 2 inputs (no bias) -- see this file's own `_attention_model`
    # default and the other plain-Attention tests above, all of which
    # always give it one.
    bqkv = rng.standard_normal((3 * Nqkv,)).astype(np.float32)
    wout1 = rng.standard_normal((Nqkv, Out1)).astype(np.float32)

    GH, GKVH, GD, Out2 = 8, 2, 8, 5
    Nq2, Nkv2 = GH * GD, GKVH * GD
    wq = rng.standard_normal((K, Nq2)).astype(np.float32)
    wk = rng.standard_normal((K, Nkv2)).astype(np.float32)
    wv = rng.standard_normal((K, Nkv2)).astype(np.float32)
    wout2 = rng.standard_normal((Nq2, Out2)).astype(np.float32)

    batch, seq = 2, 5
    seqlens_k = np.full((batch,), seq - 1, dtype=np.int32)
    total_seq = np.array(seq, dtype=np.int32)

    model = _model(
        f"""
        g (float[{batch},{seq},{K}] X1, float[{batch},{seq},{K}] X2) => (float[{batch},{seq},{Out1}] Y1, float[{batch},{seq},{Out2}] Y2)
        {{
          ctx1 = com.microsoft.Attention <num_heads={H}, qkv_hidden_sizes=[{Nqkv},{Nqkv},{Nqkv}]> (X1, Wqkv, Bqkv)
          Y1 = MatMul(ctx1, Wout1)
          q = MatMul(X2, Wq)
          k = MatMul(X2, Wk)
          v = MatMul(X2, Wv)
          ctx2, pk, pv = com.microsoft.GroupQueryAttention <num_heads={GH}, kv_num_heads={GKVH}> (q, k, v, , , SeqLensK, TotalSeq)
          Y2 = MatMul(ctx2, Wout2)
        }}
        """,
        initializer=[
            _f32(wqkv, "Wqkv"),
            _f32(bqkv, "Bqkv"),
            _f32(wout1, "Wout1"),
            _f32(wq, "Wq"),
            _f32(wk, "Wk"),
            _f32(wv, "Wv"),
            _f32(wout2, "Wout2"),
            onnx.numpy_helper.from_array(seqlens_k, "SeqLensK"),
            onnx.numpy_helper.from_array(total_seq, "TotalSeq"),
        ],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    attn_node = next(n for n in pruned.graph.node if n.op_type == "Attention")
    gqa_node = next(n for n in pruned.graph.node if n.op_type == "GroupQueryAttention")
    attn_heads, _ = _attention_attrs(attn_node)
    gqa_heads, gqa_kv_heads = _gqa_attrs(gqa_node)
    assert attn_heads == 2
    assert gqa_kv_heads == 1
    assert gqa_heads == 4

    rng2 = np.random.default_rng(31)
    x1 = rng2.standard_normal((batch, seq, K)).astype(np.float32)
    x2 = rng2.standard_normal((batch, seq, K)).astype(np.float32)
    y1, y2 = _run(pruned, {"X1": x1, "X2": x2})
    assert y1.shape == (batch, seq, Out1)
    assert y2.shape == (batch, seq, Out2)
