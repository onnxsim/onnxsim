"""Tests for ``onnxsim.apply_moe_whole_expert_pruning_cpp``/
``onnxsim.apply_qmoe_whole_expert_pruning_cpp`` -- the C++-backed ports of
``onnxsim.apply_moe_whole_expert_pruning``/``onnxsim.apply_qmoe_whole_expert_
pruning`` (see ``onnxsim/structured_pruning_entry.cpp``'s "MoE whole-expert
pruning"/"QMoE whole-expert pruning" section comments, and
``MoeRouterGateCalibrationStats``). Like
``tests/test_structured_wanda_pruning_cpp.py``, these run real calibration
data through a real ``onnxruntime``-backed
:class:`onnxsim.onnx_simplifier.PyModelExecutor` (via
``onnxsim.onnx_simplifier._get_model_executor``) -- never a fake/mock
executor -- to capture per-expert mean router gate weight.

Unlike the already-ported ``apply_moe_expert_channel_pruning_cpp`` (which
narrows every expert's own ``inter_size`` identically, never touching
``num_experts``), this drops WHOLE experts: the ``num_experts`` leading axis
itself shrinks, together with the upstream router projection's own matching
output column -- see ``onnxsim/pruning.py``'s own "MoE whole-expert pruning"
section comment for the full masking-equivalence safety argument this port
carries over unchanged.
"""

import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=18):
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": {opset}, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _u8(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.uint8), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _inits(model):
    return {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}


# --- Plain (float) MoE model builders ---------------------------------------


def _moe_router_model(
    fc1_w,
    fc2_w,
    router_w,
    router_b=None,
    fc1_b=None,
    k=2,
    tokens=6,
    use_sparse_mixer=0,
):
    num_experts, inter, hidden = fc1_w.shape
    fc1_b_arg = "FC1B" if fc1_b is not None else ""
    router_call = "Gemm(X, RW, RB)" if router_b is not None else "Gemm(X, RW)"
    model = _model(
        f"""
        g (float[{tokens},{hidden}] X) => (float[{tokens},{hidden}] Y)
        {{
          R = {router_call}
          Y = com.microsoft.MoE <k={k}, activation_type="relu", use_sparse_mixer={use_sparse_mixer}> (X, R, FC1W, {fc1_b_arg}, FC2W)
        }}
        """
    )
    inits = [_f32(fc1_w, "FC1W"), _f32(fc2_w, "FC2W"), _f32(router_w, "RW")]
    if router_b is not None:
        inits.append(_f32(router_b, "RB"))
    if fc1_b is not None:
        inits.append(_f32(fc1_b, "FC1B"))
    model.graph.initializer.extend(inits)
    return model


def _moe_router_masking_oracle(
    fc1_w, fc2_w, router_w, router_b, dropped, k, fc1_b=None, tokens=6
):
    # Same-shape model with every `dropped` expert's routing logit forced to
    # -1e9 (Softmax assigns it exactly 0 probability) and its own fc1/fc2
    # (+fc1_b) rows zeroed -- see onnxsim/pruning.py's own section comment
    # for why this is confirmed *exactly* equivalent to actually removing
    # the expert.
    fc1_w_m = fc1_w.copy()
    fc2_w_m = fc2_w.copy()
    fc1_b_m = fc1_b.copy() if fc1_b is not None else None
    router_b_m = (
        router_b.copy()
        if router_b is not None
        else np.zeros(fc1_w.shape[0], np.float32)
    )
    for e in dropped:
        fc1_w_m[e] = 0
        fc2_w_m[e] = 0
        if fc1_b_m is not None:
            fc1_b_m[e] = 0
        router_b_m[e] = -1e9
    return _moe_router_model(
        fc1_w_m,
        fc2_w_m,
        router_w,
        router_b=router_b_m,
        fc1_b=fc1_b_m,
        k=k,
        tokens=tokens,
    )


def _dropped_experts(router_w, kept_router_w):
    e = router_w.shape[1]
    kc = kept_router_w.shape[1]
    return [
        e_idx
        for e_idx in range(e)
        if not any(
            np.allclose(router_w[:, e_idx], kept_router_w[:, i]) for i in range(kc)
        )
    ]


# --- QMoE model builders (onnx.helper -- packed uint8 weights can't be
# expressed as onnx.parser text literals, see CLAUDE.md's own escape hatch
# for this case) ---------------------------------------------------------


def _qmoe_quantize_channel(w, bits):
    n, k = w.shape
    pack = 8 // bits
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    zp = 1 << (bits - 1)
    scale = np.abs(w).max(axis=1) / float(-qmin)
    scale = np.maximum(scale, np.finfo(np.float32).eps).astype(np.float32)
    q = np.clip(np.round(w / scale[:, None]), qmin, qmax).astype(np.int64) + zp
    q = np.clip(q, 0, (1 << bits) - 1).astype(np.uint8)
    parts = [(q[:, i::pack] & ((1 << bits) - 1)) for i in range(pack)]
    packed = np.zeros_like(parts[0])
    for i, p in enumerate(parts):
        packed = packed | (p << (bits * i))
    return packed.astype(np.uint8), scale


def _qmoe_quantize(w, bits):
    # Batched (per-expert) `_qmoe_quantize_channel`: w is [E, N, K].
    e, n, k = w.shape
    pack = 8 // bits
    packed = np.zeros((e, n, k // pack), dtype=np.uint8)
    scale = np.zeros((e, n), dtype=np.float32)
    for ei in range(e):
        packed[ei], scale[ei] = _qmoe_quantize_channel(w[ei], bits)
    return packed, scale


def _qmoe_router_model(
    fc1_q,
    fc1_scale,
    fc2_q,
    fc2_scale,
    bits,
    router_w,
    router_b=None,
    fc1_bias=None,
    k=2,
    tokens=6,
    use_sparse_mixer=0,
):
    num_experts, inter, hidden_packed = fc1_q.shape
    hidden = hidden_packed * (8 // bits)
    inputs = [
        onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [tokens, hidden]
        )
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(
            "Y", onnx.TensorProto.FLOAT, [tokens, hidden]
        )
    ]
    inits = [
        _u8(fc1_q, "FC1Q"),
        _f32(fc1_scale, "FC1S"),
        _u8(fc2_q, "FC2Q"),
        _f32(fc2_scale, "FC2S"),
        _f32(router_w, "RW"),
    ]
    if router_b is not None:
        router_node = onnx.helper.make_node(
            "Gemm", ["X", "RW", "RB"], ["R"], name="router"
        )
        inits.append(_f32(router_b, "RB"))
    else:
        router_node = onnx.helper.make_node("MatMul", ["X", "RW"], ["R"], name="router")

    node_inputs = ["X", "R", "FC1Q", "FC1S", "", "FC2Q", "FC2S", ""]
    if fc1_bias is not None:
        node_inputs[4] = "FC1B"
        inits.append(_f32(fc1_bias, "FC1B"))

    qmoe_node = onnx.helper.make_node(
        "QMoE",
        node_inputs,
        ["Y"],
        domain="com.microsoft",
        name="qmoe",
        k=k,
        activation_type="relu",
        expert_weight_bits=bits,
        quant_type="int",
        use_sparse_mixer=use_sparse_mixer,
    )
    graph = onnx.helper.make_graph(
        [router_node, qmoe_node], "g", inputs, outputs, initializer=inits
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    return model


def _qmoe_router_masking_oracle(
    fc1_q,
    fc1_scale,
    fc2_q,
    fc2_scale,
    bits,
    router_w,
    router_b,
    dropped,
    k,
    fc1_bias=None,
    tokens=6,
):
    fc1_q_m = fc1_q.copy()
    fc2_q_m = fc2_q.copy()
    fc1_bias_m = fc1_bias.copy() if fc1_bias is not None else None
    router_b_m = (
        router_b.copy()
        if router_b is not None
        else np.zeros(fc1_q.shape[0], np.float32)
    )
    for e in dropped:
        fc1_q_m[e] = 0
        fc2_q_m[e] = 0
        if fc1_bias_m is not None:
            fc1_bias_m[e] = 0
        router_b_m[e] = -1e9
    return _qmoe_router_model(
        fc1_q_m,
        fc1_scale,
        fc2_q_m,
        fc2_scale,
        bits,
        router_w,
        router_b=router_b_m,
        fc1_bias=fc1_bias_m,
        k=k,
        tokens=tokens,
    )


# =============================================================================
# Plain (float) MoE
# =============================================================================


def test_moe_whole_expert_pruning_cpp_matches_ort_masking_oracle():
    E, hidden, inter, tokens = 5, 8, 6, 10
    rng = np.random.default_rng(101)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.4).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.4).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    router_b = rng.standard_normal(E).astype(np.float32)
    k = 2
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, router_b=router_b, fc1_b=fc1_b, k=k, tokens=tokens
    )
    onnx.checker.check_model(model)

    calib_rng = np.random.default_rng(103)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(4)
    ]
    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model,
        calibration_data=calibration_data,
        sparsity=0.4,  # keep 3 of 5
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    assert inits["FC1W"].shape == (3, inter, hidden)
    assert inits["FC2W"].shape == (3, hidden, inter)
    assert inits["RW"].shape == (hidden, 3)
    assert inits["FC1B"].shape == (3, inter)

    dropped = _dropped_experts(router_w, inits["RW"])
    assert len(dropped) == 2
    masked = _moe_router_masking_oracle(
        fc1_w, fc2_w, router_w, router_b, dropped, k, fc1_b=fc1_b, tokens=tokens
    )

    feed_rng = np.random.default_rng(107)
    feeds = {"X": feed_rng.standard_normal((tokens, hidden)).astype(np.float32)}
    (out_pruned,) = _run(pruned, feeds)
    (out_masked,) = _run(masked, feeds)
    np.testing.assert_allclose(out_pruned, out_masked, rtol=1e-4, atol=1e-4)


def test_moe_whole_expert_pruning_cpp_adversarial_low_usage_expert_dropped():
    # Expert 0's router bias is large and positive (dominant), the last
    # expert's is large and negative (rarely used) -- at sparsity=1/E (drop
    # exactly one), the low-usage expert must be the one dropped. Catches a
    # ranking bug that inverted the comparison or dropped the wrong expert.
    E, hidden, inter, tokens = 4, 6, 5, 8
    rng = np.random.default_rng(109)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.05).astype(np.float32)
    router_b = np.zeros(E, dtype=np.float32)
    router_b[0] = 8.0
    router_b[E - 1] = -8.0
    k = 1
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, router_b=router_b, k=k, tokens=tokens
    )

    calib_rng = np.random.default_rng(113)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(4)
    ]
    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=1.0 / E
    )
    inits = _inits(pruned)
    assert inits["FC1W"].shape == (E - 1, inter, hidden)
    dropped = _dropped_experts(router_w, inits["RW"])
    assert dropped == [E - 1], f"expected the rarely-used expert dropped, got {dropped}"


def test_moe_whole_expert_pruning_cpp_k_is_floored_not_exceeded():
    # k=2 must never be pruned below -- requesting sparsity that would
    # remove more than num_experts - k experts is silently floored instead.
    E, hidden, inter, tokens = 5, 6, 4, 6
    rng = np.random.default_rng(127)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    k = 2
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=k, tokens=tokens)
    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.9
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    assert inits["FC1W"].shape == (k, inter, hidden)
    assert inits["RW"].shape == (hidden, k)
    # Still a valid, executable model at the k floor.
    feed_rng = np.random.default_rng(131)
    feeds = {"X": feed_rng.standard_normal((tokens, hidden)).astype(np.float32)}
    _run(pruned, feeds)


def test_moe_whole_expert_pruning_cpp_zero_sparsity_is_a_no_op():
    E, hidden, inter, tokens = 4, 6, 4, 6
    rng = np.random.default_rng(140)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=1, tokens=tokens)
    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.0
    )
    assert pruned.SerializeToString() == model.SerializeToString()


def test_moe_whole_expert_pruning_cpp_invalid_sparsity_raises():
    E, hidden, inter = 3, 6, 4
    rng = np.random.default_rng(141)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=1)
    with pytest.raises(Exception):
        onnxsim.apply_moe_whole_expert_pruning_cpp(
            model, calibration_data=[], sparsity=1.0
        )


# --- Uncalibrated (empty calibration_data) weight-norm fallback ------------


def test_moe_whole_expert_pruning_cpp_empty_calibration_falls_back_to_weight_norm():
    E, hidden, inter, tokens = 5, 8, 6, 6
    rng = np.random.default_rng(150)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, fc1_b=fc1_b, k=1, tokens=tokens)

    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.4
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    kept = [
        e
        for e in range(E)
        if any(
            np.allclose(fc1_w[e], inits["FC1W"][i])
            for i in range(inits["FC1W"].shape[0])
        )
    ]

    expected_importance = np.sqrt(
        np.sum(np.square(fc1_w.astype(np.float64)), axis=(1, 2))
        + np.sum(np.square(fc2_w.astype(np.float64)), axis=(1, 2))
        + np.sum(np.square(fc1_b.astype(np.float64)), axis=1)
    )
    keep_count = inits["FC1W"].shape[0]
    expected_keep = sorted(np.argsort(-expected_importance)[:keep_count].tolist())
    assert sorted(kept) == expected_keep


# --- Cross-check against the pure-Python reference --------------------------


def test_moe_whole_expert_pruning_cpp_matches_python_reference():
    E, hidden, inter, tokens = 6, 10, 8, 12
    rng = np.random.default_rng(160)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    router_b = rng.standard_normal(E).astype(np.float32)
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, router_b=router_b, fc1_b=fc1_b, k=2, tokens=tokens
    )
    calib_rng = np.random.default_rng(161)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(3)
    ]

    pruned_cpp = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.4
    )
    pruned_py = onnxsim.apply_moe_whole_expert_pruning(
        model, calibration_data=calibration_data, sparsity=0.4
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_moe_whole_expert_pruning_cpp_matches_python_reference_empty_calibration():
    # sparsity=0.4 (not 0.5) deliberately avoids the exact n*sparsity == x.5
    # boundary, where Python's builtin `round()` (round-half-to-even) and
    # C++'s `std::llround` (round-half-away-from-zero) -- the SAME rounding
    # discrepancy every other already-ported chain family's own keep_count
    # computation in structured_pruning_entry.cpp carries (llround is this
    # codebase's established, pre-existing precedent, not something this
    # port introduces) -- can legitimately disagree by one kept expert.
    E, hidden, inter, tokens = 5, 8, 6, 6
    rng = np.random.default_rng(170)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=1, tokens=tokens)

    pruned_cpp = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.4
    )
    pruned_py = onnxsim.apply_moe_whole_expert_pruning(
        model, calibration_data=[], sparsity=0.4
    )
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


# --- Default (auto-generated) calibration data ------------------------------


def test_moe_whole_expert_pruning_cpp_default_calibration_data_runs():
    E, hidden, inter, tokens = 4, 6, 4, 6
    rng = np.random.default_rng(180)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=1, tokens=tokens)
    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, num_samples=3, seed=5, sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    assert inits["FC1W"].shape[0] == 2


# =============================================================================
# QMoE (quantized-weight MoE)
# =============================================================================


def test_qmoe_whole_expert_pruning_cpp_matches_ort_masking_oracle():
    E, hidden, inter, bits, tokens = 5, 8, 6, 4, 10
    rng = np.random.default_rng(201)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    router_b = rng.standard_normal(E).astype(np.float32)
    k = 2
    fc1_q, fc1_s = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_router_model(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        bits,
        router_w,
        router_b=router_b,
        fc1_bias=fc1_b,
        k=k,
        tokens=tokens,
    )
    onnx.checker.check_model(model)

    calib_rng = np.random.default_rng(203)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(4)
    ]
    pruned = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.4
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    assert inits["FC1Q"].shape == (3, inter, hidden // 2)
    assert inits["RW"].shape == (hidden, 3)
    assert inits["FC1B"].shape == (3, inter)

    dropped = _dropped_experts(router_w, inits["RW"])
    assert len(dropped) == 2
    masked = _qmoe_router_masking_oracle(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        bits,
        router_w,
        router_b,
        dropped,
        k,
        fc1_bias=fc1_b,
        tokens=tokens,
    )
    onnx.checker.check_model(masked)

    feed_rng = np.random.default_rng(207)
    feeds = {"X": feed_rng.standard_normal((tokens, hidden)).astype(np.float32) * 0.3}
    (out_pruned,) = _run(pruned, feeds)
    (out_masked,) = _run(masked, feeds)
    np.testing.assert_allclose(out_pruned, out_masked, rtol=1e-4, atol=1e-4)


def test_qmoe_whole_expert_pruning_cpp_k_is_floored_not_exceeded():
    E, hidden, inter, bits, tokens = 5, 8, 6, 4, 6
    rng = np.random.default_rng(211)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    k = 2
    fc1_q, fc1_s = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_router_model(
        fc1_q, fc1_s, fc2_q, fc2_s, bits, router_w, k=k, tokens=tokens
    )
    pruned = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.9
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    assert inits["FC1Q"].shape == (k, inter, hidden // 2)
    assert inits["RW"].shape == (hidden, k)
    feed_rng = np.random.default_rng(213)
    feeds = {"X": feed_rng.standard_normal((tokens, hidden)).astype(np.float32) * 0.3}
    _run(pruned, feeds)


# --- Uncalibrated (empty calibration_data) weight-norm fallback ------------


def test_qmoe_whole_expert_pruning_cpp_empty_calibration_falls_back_to_weight_norm():
    E, hidden, inter, bits, tokens = 5, 8, 6, 4, 6
    rng = np.random.default_rng(221)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    fc1_q, fc1_s = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_router_model(
        fc1_q, fc1_s, fc2_q, fc2_s, bits, router_w, fc1_bias=fc1_b, k=1, tokens=tokens
    )

    pruned = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.4
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    kept = [
        e
        for e in range(E)
        if any(
            np.array_equal(fc1_q[e], inits["FC1Q"][i])
            for i in range(inits["FC1Q"].shape[0])
        )
    ]

    # Dequantized weight-norm importance, mirroring
    # QMoEExpertWeightImportance/_qmoe_expert_weight_importance.
    def _dequant(q, s):
        e, n, kp = q.shape
        pack = 8 // bits
        parts = [(q >> (bits * i)) & ((1 << bits) - 1) for i in range(pack)]
        unpacked = np.stack(parts, axis=-1).reshape(e, n, kp * pack)
        return (unpacked.astype(np.float64) - (1 << (bits - 1))) * s[..., None].astype(
            np.float64
        )

    fc1_dq = _dequant(fc1_q, fc1_s)
    fc2_dq = _dequant(fc2_q, fc2_s)
    expected_importance = np.sqrt(
        np.sum(np.square(fc1_dq), axis=(1, 2))
        + np.sum(np.square(fc2_dq), axis=(1, 2))
        + np.sum(np.square(fc1_b.astype(np.float64)), axis=1)
    )
    keep_count = inits["FC1Q"].shape[0]
    expected_keep = sorted(np.argsort(-expected_importance)[:keep_count].tolist())
    assert sorted(kept) == expected_keep


# --- Cross-check against the pure-Python reference --------------------------


def test_qmoe_whole_expert_pruning_cpp_matches_python_reference():
    E, hidden, inter, bits, tokens = 6, 8, 6, 4, 10
    rng = np.random.default_rng(231)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    router_b = rng.standard_normal(E).astype(np.float32)
    fc1_q, fc1_s = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_router_model(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        bits,
        router_w,
        router_b=router_b,
        fc1_bias=fc1_b,
        k=2,
        tokens=tokens,
    )
    calib_rng = np.random.default_rng(233)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(3)
    ]

    pruned_cpp = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.4
    )
    pruned_py = onnxsim.apply_qmoe_whole_expert_pruning(
        model, calibration_data=calibration_data, sparsity=0.4
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_qmoe_whole_expert_pruning_cpp_matches_python_reference_empty_calibration():
    # sparsity=0.4 (not 0.5) -- see the plain-MoE analogue of this test's own
    # comment above for why the x.5 rounding boundary is deliberately
    # avoided here.
    E, hidden, inter, bits, tokens = 5, 8, 6, 4, 6
    rng = np.random.default_rng(241)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    fc1_q, fc1_s = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_router_model(
        fc1_q, fc1_s, fc2_q, fc2_s, bits, router_w, k=1, tokens=tokens
    )

    pruned_cpp = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.4
    )
    pruned_py = onnxsim.apply_qmoe_whole_expert_pruning(
        model, calibration_data=[], sparsity=0.4
    )
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


# --- Blockwise (block_size set) ---------------------------------------------


def test_qmoe_whole_expert_pruning_cpp_blockwise_matches_python_reference():
    # Confirms this pass needs no block_size-specific handling at all --
    # every per-expert tensor keeps num_experts as its own leading axis
    # regardless of block_size.
    E, hidden, inter, bits, block_size, tokens = 4, 32, 16, 4, 16, 8
    rng = np.random.default_rng(251)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)

    def _quantize_blockwise(w, bits, block_size):
        e, n, k = w.shape
        pack = 8 // bits
        kb = k // block_size
        qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        zp = 1 << (bits - 1)
        w_blocks = w.reshape(e, n, kb, block_size)
        scale = np.abs(w_blocks).max(axis=-1) / float(-qmin)
        scale = np.maximum(scale, np.finfo(np.float32).eps).astype(np.float32)
        q = (
            np.clip(np.round(w_blocks / scale[..., None]), qmin, qmax).astype(np.int64)
            + zp
        )
        q = np.clip(q, 0, (1 << bits) - 1).astype(np.uint8).reshape(e, n, k)
        parts = [(q[:, :, i::pack] & ((1 << bits) - 1)) for i in range(pack)]
        packed = np.zeros_like(parts[0])
        for i, p in enumerate(parts):
            packed = packed | (p << (bits * i))
        return packed.astype(np.uint8), scale

    fc1_q, fc1_s = _quantize_blockwise(fc1_w, bits, block_size)
    fc2_q, fc2_s = _quantize_blockwise(fc2_w, bits, block_size)
    model = _qmoe_router_model(
        fc1_q, fc1_s, fc2_q, fc2_s, bits, router_w, k=1, tokens=tokens
    )
    # block_size isn't a _qmoe_router_model parameter -- patch the node
    # attribute directly (mirrors _qmoe_router_model's own block_size
    # kwarg in tests/test_pruning.py, kept out of this leaner helper).
    for node in model.graph.node:
        if node.op_type == "QMoE":
            node.attribute.append(onnx.helper.make_attribute("block_size", block_size))

    calib_rng = np.random.default_rng(253)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(3)
    ]
    pruned_cpp = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.4
    )
    pruned_py = onnxsim.apply_qmoe_whole_expert_pruning(
        model, calibration_data=calibration_data, sparsity=0.4
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    inits = _inits(pruned_cpp)
    assert inits["FC1Q"].shape[0] == inits["FC1S"].shape[0]


# --- Error handling ----------------------------------------------------------


def test_moe_whole_expert_pruning_cpp_missing_calibration_input_raises():
    E, hidden, inter = 3, 6, 4
    rng = np.random.default_rng(260)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=1)
    bad_batch = {"NotX": np.zeros((2, hidden), dtype=np.float32)}
    with pytest.raises(Exception):
        onnxsim.apply_moe_whole_expert_pruning_cpp(
            model, calibration_data=[bad_batch], sparsity=0.5
        )
