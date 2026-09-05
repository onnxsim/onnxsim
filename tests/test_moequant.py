"""Tests for ``onnxsim.apply_moequant`` (MoEQuant, see
``onnxsim/moequant.py``) -- Affinity-Guided Quantization (AGQ) and
Expert-Balanced Self-Sampling (EBSS) calibration for a ``com.microsoft::MoE``
node's per-expert weights, reusing ``onnxsim.gptq``'s own Hessian-compensated
column-update algorithm and ``onnxsim.pruning``'s own MoE chain matcher.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.moequant import _ebss_select

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=18, ir_version=10):
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
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _f16(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float16), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _moe_inits(model):
    return {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}


def _moe_router_model(
    fc1_w,
    fc2_w,
    router_w,
    router_b=None,
    fc1_b=None,
    fc3_w=None,
    activation="relu",
    k=1,
    tokens=16,
    dtype="float",
):
    num_experts, inter, hidden = fc1_w.shape
    fc1_b_arg = "FC1B" if fc1_b is not None else ""
    fc3_w_arg = "FC3W" if fc3_w is not None else ""
    router_call = "Gemm(X, RW, RB)" if router_b is not None else "Gemm(X, RW)"
    model = _model(
        f"""
        g ({dtype}[{tokens},{hidden}] X) => ({dtype}[{tokens},{hidden}] Y)
        {{
          R = {router_call}
          Y = com.microsoft.MoE <k={k}, activation_type="{activation}"> (X, R, FC1W, {fc1_b_arg}, FC2W, "", {fc3_w_arg})
        }}
        """
    )
    _cast = _f32 if dtype == "float" else _f16
    inits = [_cast(fc1_w, "FC1W"), _cast(fc2_w, "FC2W"), _cast(router_w, "RW")]
    if router_b is not None:
        inits.append(_cast(router_b, "RB"))
    if fc1_b is not None:
        inits.append(_cast(fc1_b, "FC1B"))
    if fc3_w is not None:
        inits.append(_cast(fc3_w, "FC3W"))
    model.graph.initializer.extend(inits)
    return model


def _int4_block_scale(w_nk, block_size):
    n, k = w_nk.shape
    bs = block_size if k % block_size == 0 else k
    blocks = w_nk.reshape(n, k // bs, bs)
    amax = np.max(np.abs(blocks), axis=2)
    return np.where(amax == 0.0, 1.0, amax / 7.0), bs


def _assert_on_int4_grid(original_nk, reconstructed_nk, block_size=32, atol=1e-3):
    # Confirms, directly against the ONNX initializer's own numpy value
    # (never round-tripped through onnxruntime -- see this repo's own
    # CLAUDE.md platform-numerics note), that every element of
    # ``reconstructed_nk`` is exactly ``code * scale`` for an integer
    # ``code`` in [-7, 7] and the *original* weight's own per-(row, block)
    # RTN scale -- true block-wise INT4 precision, not merely "close to" the
    # original float value. The scale must come from the pre-quantization
    # weight (matching onnxsim.moequant._quantize_expert_weight's own RTN
    # scale, computed once up front and never itself changed by GPTQ's
    # column updates), not re-derived from the already-quantized result --
    # GPTQ's error-compensated codes are not simply the original weight
    # rounded, so the reconstructed tensor's own max magnitude need not sit
    # exactly on a multiple of that scale unless the scale itself is right.
    scale, bs = _int4_block_scale(original_nk, block_size)
    n, k = original_nk.shape
    blocks = reconstructed_nk.reshape(n, k // bs, bs)
    ratio = blocks / scale[..., None]
    rounded = np.round(ratio)
    np.testing.assert_allclose(ratio, rounded, atol=atol)
    assert np.all(np.abs(rounded) <= 7 + 1e-6)


def _fixed_calibration(hidden, tokens, num_batches=4, seed=5):
    rng = np.random.default_rng(seed)
    return [
        {"X": rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(num_batches)
    ]


def test_moequant_quantizes_every_routed_expert_to_int4_grid():
    E, hidden, inter, tokens, k = 3, 8, 6, 16, 2
    rng = np.random.default_rng(1)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=k, tokens=tokens)
    onnx.checker.check_model(model)

    calibration_data = _fixed_calibration(hidden, tokens, num_batches=6, seed=11)
    quantized = onnxsim.apply_moequant(model, calibration_data=calibration_data)
    onnx.checker.check_model(quantized)
    inits = _moe_inits(quantized)

    # k == num_experts - 1 with 3 experts and 96 total calibration tokens
    # gives every expert plenty of top-k coverage -- every expert's fc1/fc2
    # should have moved onto the INT4 grid.
    for e in range(E):
        _assert_on_int4_grid(fc1_w[e], inits["FC1W"][e])
        _assert_on_int4_grid(fc2_w[e], inits["FC2W"][e])
    assert not np.allclose(inits["FC1W"], fc1_w)
    assert not np.allclose(inits["FC2W"], fc2_w)


def test_moequant_leaves_never_routed_expert_untouched():
    # Expert E-1's router bias is overwhelmingly negative -- with k=1 it can
    # never win top-1 against unit-scale logits for any calibration token,
    # so it should be left at its exact original float value while the
    # other (always routed) experts move onto the INT4 grid.
    E, hidden, inter, tokens = 3, 6, 5, 12
    rng = np.random.default_rng(3)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.1).astype(np.float32)
    router_b = np.zeros(E, dtype=np.float32)
    router_b[E - 1] = -1e6
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, router_b=router_b, k=1, tokens=tokens
    )
    onnx.checker.check_model(model)

    calibration_data = _fixed_calibration(hidden, tokens, num_batches=4, seed=13)
    quantized = onnxsim.apply_moequant(model, calibration_data=calibration_data)
    inits = _moe_inits(quantized)

    np.testing.assert_array_equal(inits["FC1W"][E - 1], fc1_w[E - 1])
    np.testing.assert_array_equal(inits["FC2W"][E - 1], fc2_w[E - 1])
    for e in range(E - 1):
        _assert_on_int4_grid(fc1_w[e], inits["FC1W"][e])
        _assert_on_int4_grid(fc2_w[e], inits["FC2W"][e])


def test_ebss_select_caps_oversubscribed_expert_by_weighted_subsampling():
    # Unit test of onnxsim.moequant._ebss_select directly -- EBSS's own
    # per-expert budget cap. An expert whose routed-token pool exceeds the
    # shared target is capped down to exactly that many tokens, drawn
    # without replacement (a real subset of the original pool, not a
    # reweighting of it); high-affinity tokens are favored, matching AGQ's
    # own "more confident tokens matter more" logic carried into which
    # tokens EBSS keeps at all.
    rng = np.random.default_rng(0)
    routed_idx = np.arange(1000)
    affinity = np.concatenate([np.full(10, 0.99), np.full(990, 0.01)])
    selected_idx, selected_aff = _ebss_select(routed_idx, affinity, 20, rng)

    assert selected_idx.size == 20
    assert selected_aff.size == 20
    assert set(selected_idx.tolist()) <= set(routed_idx.tolist())
    assert len(set(selected_idx.tolist())) == 20  # drawn without replacement
    # The 10 high-affinity tokens (indices 0-9) are drawn overwhelmingly more
    # often, across many independent trials, than any of the 990 low-affinity
    # ones -- a weighted, not uniform, subsample.
    counts = np.zeros(1000)
    trial_rng = np.random.default_rng(7)
    for _ in range(200):
        idx, _ = _ebss_select(routed_idx, affinity, 20, trial_rng)
        counts[idx] += 1
    assert counts[:10].mean() > counts[10:].mean() * 10


def test_ebss_select_is_a_no_op_at_or_below_the_budget():
    rng = np.random.default_rng(0)
    routed_idx = np.array([3, 7, 9])
    affinity = np.array([0.9, 0.5, 0.2])
    selected_idx, selected_aff = _ebss_select(routed_idx, affinity, 5, rng)
    np.testing.assert_array_equal(selected_idx, routed_idx)
    np.testing.assert_array_equal(selected_aff, affinity)


def test_moequant_ebss_and_no_ebss_both_quantize_a_dominant_expert():
    # End-to-end smoke check that ebss=True/False both run to completion and
    # land the heavily-oversubscribed expert on the INT4 grid -- the
    # per-expert token-selection behavior itself is covered directly by
    # test_ebss_select_* above (GPTQ's own column update can legitimately
    # produce bit-identical codes from two different, unequal-count
    # calibration Hessians when neither perturbs any rounding decision
    # across a quantization step boundary, so asserting the two *quantized
    # weights* differ here would be asserting an accident of this test's own
    # random data, not anything EBSS itself guarantees).
    E, hidden, inter, tokens = 3, 6, 5, 24
    rng = np.random.default_rng(21)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.05).astype(np.float32)
    router_b = np.array([8.0, 0.0, -8.0], dtype=np.float32)
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, router_b=router_b, k=1, tokens=tokens
    )
    onnx.checker.check_model(model)

    calibration_data = _fixed_calibration(hidden, tokens, num_batches=8, seed=23)

    for ebss in (True, False):
        quantized = onnxsim.apply_moequant(
            model, calibration_data=calibration_data, ebss=ebss, seed=1
        )
        inits = _moe_inits(quantized)
        _assert_on_int4_grid(fc1_w[0], inits["FC1W"][0])


def test_moequant_declines_fc3():
    # Matches onnxsim.pruning._match_moe_producer's own decline (no CPU
    # execution oracle for fc3 -- see onnxsim/pruning.py's own section
    # comment): _find_moe_chains never matches this node, so
    # apply_moequant has nothing to quantize and returns the model
    # untouched.
    E, hidden, inter, tokens = 2, 4, 3, 6
    rng = np.random.default_rng(31)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    fc3_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, fc3_w=fc3_w, tokens=tokens)

    quantized = onnxsim.apply_moequant(
        model, calibration_data=_fixed_calibration(hidden, tokens)
    )
    inits = _moe_inits(quantized)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moequant_declines_float16_experts():
    # onnxsim.pruning._match_moe_producer itself accepts FLOAT16, but
    # apply_moequant restricts itself further to FLOAT32 (see this
    # module's own docstring) -- the chain still matches (a consistent
    # all-float16 graph, since com.microsoft::MoE requires every input to
    # share one type), but is skipped entirely.
    E, hidden, inter, tokens = 2, 4, 3, 6
    rng = np.random.default_rng(33)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, tokens=tokens, dtype="float16")

    calib_rng = np.random.default_rng(34)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float16)}
        for _ in range(4)
    ]
    quantized = onnxsim.apply_moequant(model, calibration_data=calibration_data)
    inits = _moe_inits(quantized)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w.astype(np.float16))
    np.testing.assert_array_equal(inits["FC2W"], fc2_w.astype(np.float16))


def test_moequant_empty_calibration_data_is_a_no_op():
    E, hidden, inter, tokens = 2, 4, 3, 6
    rng = np.random.default_rng(37)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, tokens=tokens)

    quantized = onnxsim.apply_moequant(model, calibration_data=[])
    inits = _moe_inits(quantized)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moequant_quantized_model_still_executes_on_onnxruntime():
    # Loose, execution-level sanity check (platform-numerics note: onnxruntime
    # is not bit-exact across CPU architectures, so this is deliberately a
    # coarse relative-error bound, separate from the exact grid check above).
    E, hidden, inter, tokens, k = 4, 8, 6, 20, 2
    rng = np.random.default_rng(41)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=k, tokens=tokens)
    onnx.checker.check_model(model)

    calibration_data = _fixed_calibration(hidden, tokens, num_batches=6, seed=43)
    quantized = onnxsim.apply_moequant(model, calibration_data=calibration_data)
    onnx.checker.check_model(quantized)

    feed_rng = np.random.default_rng(47)
    feeds = {"X": feed_rng.standard_normal((tokens, hidden)).astype(np.float32)}
    (out_float,) = _run(model, feeds)
    (out_quant,) = _run(quantized, feeds)
    assert out_quant.shape == out_float.shape
    assert np.all(np.isfinite(out_quant))
    rel_err = np.linalg.norm(out_quant - out_float) / max(
        np.linalg.norm(out_float), 1e-6
    )
    assert rel_err < 0.5
