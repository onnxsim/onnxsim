"""Tests for ``onnxsim.apply_structured_pruning_cpp`` -- the C++-backed port
of ``onnxsim.apply_structured_pruning`` (see
``onnxsim/structured_pruning_entry.cpp``). Scope note: this port covers the
"plain chain" topologies (a MatMul/vanilla-Gemm or Conv producer feeding,
through shape-preserving elementwise ops, exactly one consumer of the same
family), the gated-FFN (SwiGLU/GeGLU) topology (two producers combined by
``Mul`` or the native ``SwiGLU`` op, pruned to a shared combined-importance
channel set), and Conv/MatMul residual (skip-connection) chains (a
channel-preserving merge point -- a bare ``Add(a, b)`` for either family, or,
MatMul/Gemm only, a fused
``com.microsoft::SkipLayerNormalization``/``SkipSimplifiedLayerNormalization``
node -- resolved via backward walk plus union-find grouping). Tests here are
adapted from ``test_pruning.py``'s own ``apply_structured_pruning`` coverage,
plus a couple of tests confirming unmatched topologies are left untouched
(never guessed at).
"""

import os
import tempfile

import ml_dtypes
import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.reference
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21):
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


def _run27(model, feeds):
    # `CausalConvWithState` is plain ai.onnx opset 27, which this
    # environment's onnxruntime treats as "under development"
    # (`ValidateOpsetForDomain` otherwise refuses to even load the graph --
    # see onnxsim/pruning.py's own "Conv/pooling/Resize/Pad pass-through"
    # section comment and tests/test_pruning.py's own identical `_run27`
    # helper for the empirical finding). Setting this env var only relaxes
    # that load-time opset-vintage check; it does not stub out or change the
    # op's own real CPU kernel. Scoped to just this helper (not a
    # module-level mutation) since only CausalConvWithState's own tests ever
    # need it -- every other test in this file keeps using plain `_run`
    # against a released opset unaffected either way.
    os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _oracle_keep_indices(w1, keep_count):
    importance = np.linalg.norm(w1.T, axis=1)
    return np.sort(np.argsort(-importance)[:keep_count])


def _oracle_keep_indices_conv(w, keep_count):
    importance = np.linalg.norm(w.reshape(w.shape[0], -1).astype(np.float64), axis=1)
    return np.sort(np.argsort(-importance)[:keep_count])


def _oracle_keep_indices_conv_grouped(w, group, sparsity):
    out_channels = w.shape[0]
    block = out_channels // group
    per_group_keep = max(1, round(block * (1.0 - sparsity)))
    parts = []
    for gi in range(group):
        lo, hi = gi * block, (gi + 1) * block
        parts.append(_oracle_keep_indices_conv(w[lo:hi], per_group_keep) + lo)
    return np.concatenate(parts)


def _oracle_slice_grouped_consumer_conv(w2, keep, group, n_channels):
    out_channels = w2.shape[0]
    out_per_group = out_channels // group
    block = n_channels // group
    parts = []
    for gi in range(group):
        lo, hi = gi * block, (gi + 1) * block
        local_keep = keep[(keep >= lo) & (keep < hi)] - lo
        parts.append(w2[gi * out_per_group : (gi + 1) * out_per_group][:, local_keep])
    return np.concatenate(parts, axis=0)


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


def _conv_pair_model(w1, w2, b1=None, spatial=10, activation="Relu"):
    Cin, C2 = w1.shape[1], w2.shape[0]
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    if b1 is not None:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1)"
    out_spatial = spatial - 4
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


def _grouped_conv_pair_model(
    w1, w2, group1=1, group2=1, b1=None, spatial=10, activation="Relu"
):
    Cin, C2 = w1.shape[1] * group1, w2.shape[0]
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    g1 = f", group={group1}" if group1 != 1 else ""
    g2 = f", group={group2}" if group2 != 1 else ""
    if b1 is not None:
        conv1 = f"h = Conv<kernel_shape=[3,3]{g1}>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = f"h = Conv<kernel_shape=[3,3]{g1}>(X, W1)"
    out_spatial = spatial - 4
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{out_spatial},{out_spatial}] Y)
        {{
          {conv1}
          a = {activation}(h)
          Y = Conv<kernel_shape=[3,3]{g2}>(a, W2)
        }}
        """,
        initializer=initializer,
    )


def _depthwise_pair_model(w1, dw_hops, w2, b1=None, spatial=10, activation="Relu"):
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
    out_spatial = spatial - 2 * n_convs
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


# --- MatMul/Gemm plain chains -----------------------------------------------


def test_cpp_structured_pruning_shrinks_matched_layers():
    model = _mlp_model(K=8, H=32, Out=4)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 16]
    assert list(inits["B1"].dims) == [16]
    assert list(inits["W2"].dims) == [16, 4]


def test_cpp_structured_pruning_matches_manual_channel_deletion_exactly():
    K, H, Out = 8, 32, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=True)
    orig = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
    w1, b1, w2 = orig["W1"], orig["B1"], orig["W2"]

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    keep = _oracle_keep_indices(w1, H // 2)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    h = x @ w1[:, keep] + b1[keep]
    a = np.maximum(h, 0)
    y_oracle = a @ w2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_only_chain_matches_oracle():
    K, H, Out = 8, 24, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=False, activation="Sigmoid")
    w1 = onnx.numpy_helper.to_array(model.graph.initializer[0])
    w2 = onnx.numpy_helper.to_array(model.graph.initializer[1])

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.25)
    keep = _oracle_keep_indices(w1, H - round(H * 0.25))

    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    h = x @ w1[:, keep]
    a = 1.0 / (1.0 + np.exp(-h))
    y_oracle = a @ w2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_bias_add_between_matmuls_matches_oracle():
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

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
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


def test_cpp_structured_pruning_skips_branching_output():
    K, H, Out = 8, 16, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=False)
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("h", onnx.TensorProto.FLOAT, ["batch", H])
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H]
    assert list(inits["W2"].dims) == [H, Out]


def test_cpp_structured_pruning_skips_multi_consumer_branch():
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

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H]


def test_cpp_structured_pruning_zero_sparsity_is_a_no_op():
    model = _mlp_model(K=8, H=16, Out=4)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.0)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 16]


def test_cpp_structured_pruning_invalid_sparsity_raises():
    model = _mlp_model(K=8, H=16, Out=4)
    with pytest.raises(Exception):
        onnxsim.apply_structured_pruning_cpp(model, sparsity=1.0)
    with pytest.raises(Exception):
        onnxsim.apply_structured_pruning_cpp(model, sparsity=-0.1)


def test_cpp_structured_pruning_chains_through_a_third_layer():
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

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
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


# --- Gated FFN (SwiGLU/GeGLU) -----------------------------------------------


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


def test_cpp_structured_pruning_gated_ffn_matches_oracle():
    K, H, Out = 8, 16, 4
    model, wg, wu, wd = _swiglu_mlp_model(K=K, H=H, Out=Out)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
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


def test_cpp_structured_pruning_gated_ffn_prunes_both_branches_to_same_channels():
    # The real bug this pattern risks: gate and up disagreeing on which
    # channels survive, which would silently break the elementwise
    # product's alignment. Assert they select the identical index set,
    # not just that both shrank to the same *count*.
    K, H, Out = 8, 20, 4
    model, wg, wu, _ = _swiglu_mlp_model(K=K, H=H, Out=Out, seed=1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.3)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    keep = _combined_keep_indices(wg, wu, H - round(H * 0.3))

    np.testing.assert_array_equal(inits["Wg"], wg[:, keep])
    np.testing.assert_array_equal(inits["Wu"], wu[:, keep])


def test_cpp_structured_pruning_gelu_gated_ffn_matches_oracle():
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

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    keep = _combined_keep_indices(wg, wu, H // 2)

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    g = x @ wg[:, keep]
    gate = 0.5 * g * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (g + 0.044715 * g**3)))
    up = x @ wu[:, keep]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_structured_pruning_ungated_mul_of_two_producers_still_matches_oracle():
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

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    keep = _combined_keep_indices(w1, w2, H // 2)

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    y_oracle = ((x @ w1[:, keep]) * (x @ w2[:, keep])) @ w3[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_gated_mul_against_constant_scale_is_not_a_gate():
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

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H // 2]
    assert list(inits["Scale"].dims) == [H // 2]
    assert list(inits["W2"].dims) == [H // 2, Out]


def test_cpp_structured_pruning_gated_ffn_skips_when_a_branch_also_feeds_elsewhere():
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

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wg"].dims) == [K, H]
    assert list(inits["Wu"].dims) == [K, H]
    assert list(inits["Wd"].dims) == [H, Out]


def test_cpp_structured_pruning_native_swiglu_node_prunes_both_producers_together():
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

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    keep = _combined_keep_indices(wg, wu, H // 2)

    np.testing.assert_array_equal(inits["Wg"], wg[:, keep])
    np.testing.assert_array_equal(inits["Wu"], wu[:, keep])
    np.testing.assert_array_equal(inits["Wd"], wd[keep, :])


# --- Split-merged (fused gate_up_proj) gated FFN chains ----------------------
#
# Real Phi-3/Phi-3.5 (onnxruntime-genai) exports use ONE gate_up_proj MatMul/
# Gemm whose 2*H-wide output is halved by a Split into a gate half and an up
# half, rather than two separate gate_proj/up_proj producers -- see
# onnxsim/structured_pruning_entry.cpp's own "Split-merged (fused
# gate_up_proj) gated FFN chains" section comment (and onnxsim/pruning.py's
# identically-named section, which this is ported from) for the full shape
# and co-selection semantics these tests exercise: "neuron" i of the
# intermediate dimension is represented by BOTH column i (gate) and column
# H + i (up) of the ONE combined weight tensor, and must always be kept or
# dropped together.


def _split_gate_up_keep_indices(w, H, keep_count):
    # The correct paired-importance ranking: combined (root-sum-square) norm
    # of the gate half (columns [0, H)) and the up half (columns [H, 2H)) of
    # the ONE combined weight `w` -- mirrors _combined_keep_indices's own
    # formula for the two-separate-producer case above.
    gate_half, up_half = w[:, :H], w[:, H:]
    importance = np.sqrt(
        np.square(np.linalg.norm(gate_half.T, axis=1))
        + np.square(np.linalg.norm(up_half.T, axis=1))
    )
    return np.sort(np.argsort(-importance)[:keep_count])


def _split_gate_up_mlp_model(
    K=8,
    H=16,
    Out=4,
    gate_activation="Sigmoid",
    seed=0,
    opset=21,
    split_attrs="axis=-1, num_outputs=2",
):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((K, 2 * H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    split_attr_text = f"<{split_attrs}>" if split_attrs else ""
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          combined = MatMul(X, W)
          gate, up = Split {split_attr_text} (combined)
          gate_act = {gate_activation}(gate)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(wd, "Wd")],
        opset=opset,
    )
    return model, w, wd


def test_cpp_structured_pruning_split_gated_ffn_matches_oracle():
    K, H, Out = 8, 16, 4
    model, w, wd = _split_gate_up_mlp_model(K=K, H=H, Out=Out)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W"].dims) == [K, H]  # 2 * (H // 2)
    assert list(inits["Wd"].dims) == [H // 2, Out]

    keep = _split_gate_up_keep_indices(w, H, H // 2)
    rng = np.random.default_rng(20)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    gate = 1.0 / (1.0 + np.exp(-(x @ w[:, keep])))
    up = x @ w[:, H + keep]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_split_gated_ffn_prunes_both_halves_of_one_tensor():
    # The real bug this pattern risks: only one of the two halves getting
    # sliced (or the two halves disagreeing on which columns survive) --
    # assert the SAME index set is dropped from both, out of the single
    # physical weight tensor.
    K, H, Out = 8, 20, 4
    model, w, wd = _split_gate_up_mlp_model(K=K, H=H, Out=Out, seed=1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.3)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    keep = _split_gate_up_keep_indices(w, H, H - round(H * 0.3))

    np.testing.assert_array_equal(inits["W"][:, : len(keep)], w[:, keep])
    np.testing.assert_array_equal(inits["W"][:, len(keep) :], w[:, H + keep])
    np.testing.assert_array_equal(inits["Wd"], wd[keep, :])


def test_cpp_structured_pruning_split_gated_ffn_uses_combined_paired_importance():
    # Adversarial case: gate-half and up-half columns for the SAME neuron
    # index deliberately have very different magnitudes, constructed so that
    # ranking by EITHER half alone (a "sliced/ranked only one half" bug)
    # picks a DIFFERENT keep-set than the correct combined (root-sum-square)
    # ranking of the pair. K=2 with only row 0 non-zero makes each column's
    # own L2 norm exactly its row-0 value, so the desired per-half
    # magnitudes can be set directly and exactly.
    K, H, Out = 2, 5, 3
    gate_vals = np.array([9.0, 1.0, 6.0, 7.0, 0.5], dtype=np.float32)
    up_vals = np.array([1.0, 8.9, 6.0, 0.5, 6.9], dtype=np.float32)
    # combined (root-sum-square) importance per column:
    #   col0: sqrt(9.0^2+1.0^2) = 9.0554  (rank 1)
    #   col1: sqrt(1.0^2+8.9^2) = 8.9556  (rank 2)
    #   col2: sqrt(6.0^2+6.0^2) = 8.4853  (rank 3)
    #   col3: sqrt(7.0^2+0.5^2) = 7.0178  (rank 4)
    #   col4: sqrt(0.5^2+6.9^2) = 6.9181  (rank 5)
    # correct keep (top 3, combined) = [0, 1, 2]; gate-only top 3 would be
    # [0, 2, 3] and up-only top 3 would be [1, 2, 4] -- both wrong.
    w = np.zeros((K, 2 * H), dtype=np.float32)
    w[0, :H] = gate_vals
    w[0, H:] = up_vals
    rng = np.random.default_rng(2)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          combined = MatMul(X, W)
          gate, up = Split <axis=-1, num_outputs=2> (combined)
          gate_act = Sigmoid(gate)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(wd, "Wd")],
    )

    # H=5, sparsity=0.4 -> keep_count = 5 - round(5*0.4) = 3.
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.4)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    correct_keep = np.array([0, 1, 2])
    gate_only_keep = np.array([0, 2, 3])  # what a gate-only-ranked bug would pick
    up_only_keep = np.array([1, 2, 4])  # what an up-only-ranked bug would pick

    np.testing.assert_array_equal(inits["W"][:, :3], w[:, correct_keep])
    np.testing.assert_array_equal(inits["W"][:, 3:], w[:, H + correct_keep])
    assert not np.array_equal(inits["W"][0, :3], w[0, gate_only_keep])
    assert not np.array_equal(inits["W"][0, :3], w[0, up_only_keep])


def test_cpp_structured_pruning_split_gated_ffn_gelu_activation_matches_oracle():
    # GeGLU: same fused-gate_up_proj topology, a different (still-unary)
    # gate activation -- mirrors this file's own
    # test_cpp_structured_pruning_gelu_gated_ffn_matches_oracle above, but
    # for the single fused-producer shape. Uses Gelu's tanh approximation so
    # the oracle needs no scipy/erf. (Native ai.onnx Swish/HardSwish are not
    # exercised here: UnaryPassThroughOps() -- shared by every gated-chain
    # family in this port, not just this one -- does not yet recognize them,
    # a pre-existing gap outside this feature's own scope.)
    K, H, Out = 8, 16, 4
    model, w, wd = _split_gate_up_mlp_model(
        K=K, H=H, Out=Out, gate_activation='Gelu<approximate = "tanh">', seed=3
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _split_gate_up_keep_indices(w, H, H // 2)
    rng = np.random.default_rng(30)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    g = x @ w[:, keep]
    gate = 0.5 * g * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (g + 0.044715 * g**3)))
    up = x @ w[:, H + keep]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_structured_pruning_split_gated_ffn_explicit_equal_split_input_matches_oracle():
    # opset 13+'s explicit `split` *input* (rather than the fully-automatic
    # even split) spelled out as literally [H, H] -- the same semantic
    # split, a different spelling; the pruned model's own Split input must
    # be rewritten to the new, still-even [h', h'].
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(4)
    w = rng.standard_normal((K, 2 * H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    sizes = onnx.numpy_helper.from_array(np.array([H, H], dtype=np.int64), name="Sizes")
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          combined = MatMul(X, W)
          gate, up = Split <axis=-1> (combined, Sizes)
          gate_act = Sigmoid(gate)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(wd, "Wd"), sizes],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    split_node = next(n for n in pruned.graph.node if n.op_type == "Split")
    sizes_init = next(
        t for t in pruned.graph.initializer if t.name == split_node.input[1]
    )
    assert list(onnx.numpy_helper.to_array(sizes_init)) == [H // 2, H // 2]

    keep = _split_gate_up_keep_indices(w, H, H // 2)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    gate = 1.0 / (1.0 + np.exp(-(x @ w[:, keep])))
    up = x @ w[:, H + keep]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_split_gated_ffn_native_swiglu_matches_oracle():
    # ONNX's native fused SwiGLU(a, b) = swish(a) * b (opset 28+), fed
    # directly by the Split's own two raw outputs -- mirrors
    # test_cpp_structured_pruning_native_swiglu_node_prunes_both_producers_together
    # above, but for the single fused-producer gate_up_proj shape. Not yet
    # supported by the installed onnx checker/onnxruntime in this
    # environment (opset 28 is still under development upstream), so this
    # verifies the graph surgery directly via tensor values rather than
    # onnx.checker/onnxruntime execution.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(12)
    w = rng.standard_normal((K, 2 * H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          combined = MatMul(X, W)
          gate, up = Split <axis=-1, num_outputs=2> (combined)
          h = SwiGLU(gate, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(wd, "Wd")],
        opset=28,
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    keep = _split_gate_up_keep_indices(w, H, H // 2)

    np.testing.assert_array_equal(inits["W"][:, : H // 2], w[:, keep])
    np.testing.assert_array_equal(inits["W"][:, H // 2 :], w[:, H + keep])
    np.testing.assert_array_equal(inits["Wd"], wd[keep, :])


def test_cpp_structured_pruning_split_gated_ffn_gemm_producer_with_bias_matches_oracle():
    # The producer may also be a vanilla Gemm with a fused constant bias
    # (_match_producer/MatchProducer's own bias support) -- unlike a
    # *separate* MatMul -> Add(bias) hop before Split (declined, see
    # test_cpp_structured_pruning_split_gated_ffn_declines_bias_add_before_split
    # below), Gemm's own bias operand is a per-channel constant riding along
    # with the rest of the combined [K, 2H] weight, so it must be sliced at
    # the same two fixed offsets.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(13)
    w = rng.standard_normal((K, 2 * H)).astype(np.float32)
    b = rng.standard_normal((2 * H,)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          combined = Gemm(X, W, B)
          gate, up = Split <axis=-1, num_outputs=2> (combined)
          gate_act = Sigmoid(gate)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(b, "B"), _f32(wd, "Wd")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W"].dims) == [K, H]
    assert list(inits["B"].dims) == [H]

    keep = _split_gate_up_keep_indices(w, H, H // 2)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    combined = (
        x @ w[:, np.concatenate([keep, H + keep])] + b[np.concatenate([keep, H + keep])]
    )
    gate = 1.0 / (1.0 + np.exp(-combined[:, : H // 2]))
    up = combined[:, H // 2 :]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_split_gated_ffn_declines_unequal_explicit_split():
    K, H = 8, 16
    rng = np.random.default_rng(5)
    w = rng.standard_normal((K, 2 * H)).astype(np.float32)
    sizes = onnx.numpy_helper.from_array(
        np.array([H + 2, H - 2], dtype=np.int64), name="Sizes"
    )
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{H + 2}] Gate, float[batch,{H - 2}] Up)
        {{
          combined = MatMul(X, W)
          Gate, Up = Split <axis=-1> (combined, Sizes)
        }}
        """,
        initializer=[_f32(w, "W"), sizes],
    )
    before = model.SerializeToString()
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == before


def test_cpp_structured_pruning_split_gated_ffn_declines_when_axis_defaults_to_zero():
    # Split's own schema default axis is 0, unlike Concat's *required*
    # attribute -- an un-annotated Split here would target the batch axis,
    # not the channel axis, and must be declined, not assumed.
    K, H, Out = 8, 16, 4
    model, w, wd = _split_gate_up_mlp_model(
        K=K, H=H, Out=Out, seed=10, split_attrs="num_outputs=2"
    )
    before = model.SerializeToString()
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == before


def test_cpp_structured_pruning_split_gated_ffn_declines_bias_add_before_split():
    # A separate MatMul -> Add(bias) -> Split, rather than the producer's
    # raw output feeding Split directly -- out of scope for this first pass
    # (see this section's own comment in structured_pruning_entry.cpp).
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(8)
    w = rng.standard_normal((K, 2 * H)).astype(np.float32)
    bias = rng.standard_normal((2 * H,)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          combined = MatMul(X, W)
          combined_b = Add(combined, Bias)
          gate, up = Split <axis=-1, num_outputs=2> (combined_b)
          gate_act = Sigmoid(gate)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(bias, "Bias"), _f32(wd, "Wd")],
    )
    before = model.SerializeToString()
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == before


# --- Conv plain chains -------------------------------------------------------


def test_cpp_structured_pruning_conv_chain_shrinks_matched_layers():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(30)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2, b1=b1)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [C1 // 2, Cin, 3, 3]
    assert list(inits["B1"].dims) == [C1 // 2]
    assert list(inits["W2"].dims) == [C2, C1 // 2, 3, 3]


def test_cpp_structured_pruning_conv_chain_matches_manual_channel_deletion_exactly():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(30)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2, b1=b1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _conv_pair_model(w1[keep], w2[:, keep], b1=b1[keep])

    rng_x = np.random.default_rng(31)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_conv_only_chain_matches_oracle_no_bias():
    Cin, C1, C2 = 4, 12, 6
    rng = np.random.default_rng(32)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2, activation="Sigmoid")

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.25)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv(w1, C1 - round(C1 * 0.25))
    oracle = _conv_pair_model(w1[keep], w2[:, keep], activation="Sigmoid")

    rng_x = np.random.default_rng(33)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_skips_grouped_producer_conv():
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
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_skips_grouped_consumer_conv():
    Cin, C1 = 3, 8
    rng = np.random.default_rng(35)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
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
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_conv_into_non_pass_through_op_is_left_untouched():
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
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_conv_chain_scale_between_convs_is_left_untouched():
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
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


# --- Depthwise Conv pass-through hops ----------------------------------------


def test_cpp_structured_pruning_depthwise_pass_through_matches_manual_channel_deletion_exactly():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(50)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    wd = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    bd = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _depthwise_pair_model(w1, [(wd, bd)], w2, b1=b1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
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


def test_cpp_structured_pruning_multiple_consecutive_depthwise_pass_through_hops_matches_oracle():
    Cin, C1, C2 = 3, 12, 6
    rng = np.random.default_rng(52)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    wd1 = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    bd1 = rng.standard_normal((C1,)).astype(np.float32)
    wd2 = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _depthwise_pair_model(w1, [(wd1, bd1), (wd2, None)], w2, spatial=14)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.25)
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


def test_cpp_structured_pruning_depthwise_pass_through_no_bias_matches_oracle():
    Cin, C1, C2 = 4, 10, 5
    rng = np.random.default_rng(54)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    wd = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _depthwise_pair_model(w1, [(wd, None)], w2, activation="Sigmoid")

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.3)
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


def test_cpp_structured_pruning_depthwise_pass_through_branch_is_left_untouched():
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
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["WD"], wd)
    np.testing.assert_array_equal(inits["W2"], w2)


# --- General grouped Conv -----------------------------------------------------


def test_cpp_structured_pruning_general_grouped_producer_conv_prunes_per_group_independently():
    Cin, C1, C2, group = 4, 8, 4, 2
    rng = np.random.default_rng(80)
    w1 = rng.standard_normal((C1, Cin // group, 3, 3)).astype(np.float32)
    w1[:4] *= 10.0
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _grouped_conv_pair_model(w1, w2, group1=group)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep_grouped = _oracle_keep_indices_conv_grouped(w1, group, 0.5)
    assert sum(i < 4 for i in keep_grouped) == 2
    assert sum(i >= 4 for i in keep_grouped) == 2

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1[keep_grouped])
    dw_node = next(n for n in pruned.graph.node if "W1" in n.input)
    group_attr = next(a.i for a in dw_node.attribute if a.name == "group")
    assert group_attr == group

    oracle = _grouped_conv_pair_model(
        w1[keep_grouped], w2[:, keep_grouped], group1=group
    )
    rng_x = np.random.default_rng(81)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_general_grouped_consumer_conv_matches_manual_channel_deletion_exactly():
    Cin, C1, C2, group = 3, 8, 6, 2
    rng = np.random.default_rng(82)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w1[:4] *= 8.0
    w2 = rng.standard_normal((C2, C1 // group, 3, 3)).astype(np.float32)
    model = _grouped_conv_pair_model(w1, w2, group2=group)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv_grouped(w1, group, 0.5)
    w2_sliced = _oracle_slice_grouped_consumer_conv(w2, keep, group, C1)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1[keep])
    np.testing.assert_array_equal(inits["W2"], w2_sliced)

    oracle = _grouped_conv_pair_model(w1[keep], w2_sliced, group2=group)
    rng_x = np.random.default_rng(83)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_both_sides_grouped_matching_group_count_matches_oracle():
    Cin, C1, C2, group = 4, 8, 6, 2
    rng = np.random.default_rng(84)
    w1 = rng.standard_normal((C1, Cin // group, 3, 3)).astype(np.float32)
    w1[:4] *= 6.0
    w2 = rng.standard_normal((C2, C1 // group, 3, 3)).astype(np.float32)
    model = _grouped_conv_pair_model(w1, w2, group1=group, group2=group)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv_grouped(w1, group, 0.5)
    w2_sliced = _oracle_slice_grouped_consumer_conv(w2, keep, group, C1)
    oracle = _grouped_conv_pair_model(w1[keep], w2_sliced, group1=group, group2=group)

    rng_x = np.random.default_rng(85)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_skips_mismatched_grouped_producer_and_consumer():
    Cin, C1, C2, gp, gc = 4, 8, 8, 2, 4
    rng = np.random.default_rng(86)
    w1 = rng.standard_normal((C1, Cin // gp, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1 // gc, 3, 3)).astype(np.float32)
    model = _grouped_conv_pair_model(w1, w2, group1=gp, group2=gc)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


# --- Conv residual (Add-merged) chains --------------------------------------


def _residual_diamond_model(w_f, w_s, w_out, spatial=10):
    # y = Conv_out(Relu(Add(Conv_f(X), Conv_s(X)))) -- a "projection
    # shortcut" residual block: two entirely independent Conv producers
    # merge via Add and must therefore share one surviving channel-index
    # set, feeding one real consumer.
    Cin = w_f.shape[1]
    Cout = w_out.shape[0]
    out_spatial = spatial - 4  # two chained 3x3 valid convs
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_spatial},{out_spatial}] Y)
        {{
          f = Conv<kernel_shape=[3,3]>(X, WF)
          s = Conv<kernel_shape=[3,3]>(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          Y = Conv<kernel_shape=[3,3]>(r, WOUT)
        }}
        """,
        initializer=[_f32(w_f, "WF"), _f32(w_s, "WS"), _f32(w_out, "WOUT")],
    )


def _residual_transitive_model(w_f1, w_s1, w_f2, w_out, spatial=10):
    # Two Add merges chained transitively, sharing one spine channel count,
    # with no branch anywhere along the chain: add1's own output feeds only
    # into add2, never reused elsewhere -- the union-find grouping extends
    # across both Adds into one group of three producers.
    Cin = w_f1.shape[1]
    Cz = w_f2.shape[1]
    Cout = w_out.shape[0]
    add1_spatial = spatial - 2  # one 3x3 valid conv each, from X
    out_spatial = add1_spatial - 2  # WOUT's own 3x3 valid conv
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X, float[N,{Cz},{add1_spatial},{add1_spatial}] Z)
            => (float[N,{Cout},{out_spatial},{out_spatial}] Y)
        {{
          f1 = Conv<kernel_shape=[3,3]>(X, WF1)
          s1 = Conv<kernel_shape=[3,3]>(X, WS1)
          add1 = Add(f1, s1)
          f2 = Conv<kernel_shape=[1,1]>(Z, WF2)
          add2 = Add(f2, add1)
          r = Relu(add2)
          Y = Conv<kernel_shape=[3,3]>(r, WOUT)
        }}
        """,
        initializer=[
            _f32(w_f1, "WF1"),
            _f32(w_s1, "WS1"),
            _f32(w_f2, "WF2"),
            _f32(w_out, "WOUT"),
        ],
    )


def test_cpp_structured_pruning_conv_residual_add_matches_oracle():
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(80)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)
    model = _residual_diamond_model(w_f, w_s, w_out)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["WF"].dims) == [C // 2, Cin, 3, 3]
    assert list(inits["WS"].dims) == [C // 2, Cin, 3, 3]
    assert list(inits["WOUT"].dims) == [Cout, C // 2, 3, 3]

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _residual_diamond_model(w_f[keep], w_s[keep], w_out[:, keep])

    rng_x = np.random.default_rng(81)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_conv_residual_add_transitive_chain_matches_oracle():
    Cin, C, Cz, Cout = 3, 16, 5, 8
    rng = np.random.default_rng(82)
    w_f1 = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s1 = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_f2 = rng.standard_normal((C, Cz, 1, 1)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)
    model = _residual_transitive_model(w_f1, w_s1, w_f2, w_out)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f1.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s1.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_f2.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _residual_transitive_model(
        w_f1[keep], w_s1[keep], w_f2[keep], w_out[:, keep]
    )

    rng_x = np.random.default_rng(83)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    z = rng_x.standard_normal((2, Cz, 8, 8)).astype(np.float32)
    (y,) = _run(pruned, {"X": x, "Z": z})
    (y_oracle,) = _run(oracle, {"X": x, "Z": z})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_conv_residual_add_matches_oracle_on_fan_out_branch():
    # A realistic multi-block residual stage's interior boundary: `r`
    # (add1's own post-block tensor) is read twice -- once by the next
    # block's own first Conv, once unchanged as that next block's own Add
    # shortcut operand. The backward walkers no longer reject this
    # mid-walk; instead the "extra" reader (`nxt`) is resolved as its own
    # independent forward branch once the group's shared keep set is
    # established (see ResolveConvFanoutBranches), so `nxt` ends up
    # pruned on *both* axes of WNEXT: its own output channels (it's also a
    # leaf producer of add2's own merge) and, via this fan-out branch, its
    # input channels (it independently reads the group's own shared spine).
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(84)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_next = rng.standard_normal((C, C, 1, 1)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          f = Conv<kernel_shape=[3,3]>(X, WF)
          s = Conv<kernel_shape=[3,3]>(X, WS)
          add1 = Add(f, s)
          r = Relu(add1)
          nxt = Conv<kernel_shape=[1,1]>(r, WNEXT)
          add2 = Add(nxt, r)
          Y = Conv<kernel_shape=[1,1]>(add2, WOUT)
        }}
        """,
        initializer=[
            _f32(w_f, "WF"),
            _f32(w_s, "WS"),
            _f32(w_next, "WNEXT"),
            _f32(w_out, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_next.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          f = Conv<kernel_shape=[3,3]>(X, WF)
          s = Conv<kernel_shape=[3,3]>(X, WS)
          add1 = Add(f, s)
          r = Relu(add1)
          nxt = Conv<kernel_shape=[1,1]>(r, WNEXT)
          add2 = Add(nxt, r)
          Y = Conv<kernel_shape=[1,1]>(add2, WOUT)
        }}
        """,
        initializer=[
            _f32(w_f[keep], "WF"),
            _f32(w_s[keep], "WS"),
            _f32(w_next[keep][:, keep], "WNEXT"),
            _f32(w_out[:, keep], "WOUT"),
        ],
    )

    rng_x = np.random.default_rng(85)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_conv_residual_add_declines_on_identity_shortcut():
    # y = Conv2(Relu(Add(Conv1(X), X))): a classic identity-shortcut
    # residual block with no Conv on the shortcut path at all. X has no
    # producer this pass owns (it's a graph input) and is itself read
    # twice (by Conv1 and directly by Add) -- either alone is enough to
    # decline.
    C, Cout = 8, 4
    rng = np.random.default_rng(85)
    w1 = rng.standard_normal((C, C, 1, 1)).astype(np.float32)
    w2 = rng.standard_normal((Cout, C, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{C},10,10] X) => (float[N,{Cout},10,10] Y)
        {{
          f = Conv<kernel_shape=[1,1]>(X, W1)
          add1 = Add(f, X)
          r = Relu(add1)
          Y = Conv<kernel_shape=[1,1]>(r, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_conv_residual_add_matches_oracle_with_grouped_conv_consumer():
    # Two independent Conv branches merge via Add, and the downstream
    # consumer is a general grouped Conv (group=2) -- now matched (see
    # this module's own docstring for why per-`group`-block top-k is a
    # provably-safe generalization once every producer/branch agrees on
    # the same `group` count), one independent top-k per `group`-sized
    # block of the combined-importance vector.
    Cin, C, Cout, group = 3, 16, 8, 2
    rng = np.random.default_rng(89)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C // group, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          f = Conv<kernel_shape=[3,3]>(X, WF)
          s = Conv<kernel_shape=[3,3]>(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          Y = Conv<kernel_shape=[1,1],group={group}>(r, WOUT)
        }}
        """,
        initializer=[_f32(w_f, "WF"), _f32(w_s, "WS"), _f32(w_out, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    block = C // group
    per_group_keep = block // 2
    keep = np.concatenate(
        [
            np.sort(
                np.argsort(-importance[gi * block : (gi + 1) * block])[:per_group_keep]
            )
            + gi * block
            for gi in range(group)
        ]
    )

    out_per_group = Cout // group
    out_parts = []
    for gi in range(group):
        local_keep = keep[(keep >= gi * block) & (keep < (gi + 1) * block)] - gi * block
        out_parts.append(
            w_out[gi * out_per_group : (gi + 1) * out_per_group, local_keep]
        )
    w_out_oracle = np.concatenate(out_parts, axis=0)

    oracle = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          f = Conv<kernel_shape=[3,3]>(X, WF)
          s = Conv<kernel_shape=[3,3]>(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          Y = Conv<kernel_shape=[1,1],group={group}>(r, WOUT)
        }}
        """,
        initializer=[
            _f32(w_f[keep], "WF"),
            _f32(w_s[keep], "WS"),
            _f32(w_out_oracle, "WOUT"),
        ],
    )

    rng_x = np.random.default_rng(90)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- MatMul/Gemm residual (Add-merged) chains -------------------------------


def _matmul_residual_diamond_model(wf, ws, wout):
    # y = MatMul_out(Relu(Add(MatMul_f(X), MatMul_s(X)))) -- the MatMul/Gemm
    # analogue of _residual_diamond_model.
    K, C = wf.shape
    Out = wout.shape[1]
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          Y = MatMul(r, WOUT)
        }}
        """,
        initializer=[_f32(wf, "WF"), _f32(ws, "WS"), _f32(wout, "WOUT")],
    )


def _matmul_residual_transitive_model(wf1, ws1, wf2, wout):
    # Two Add merges chained transitively, sharing one spine channel count
    # -- the MatMul/Gemm analogue of _residual_transitive_model.
    K, C = wf1.shape
    Kz = wf2.shape[0]
    Out = wout.shape[1]
    return _model(
        f"""
        g (float[batch,{K}] X, float[batch,{Kz}] Z) => (float[batch,{Out}] Y)
        {{
          f1 = MatMul(X, WF1)
          s1 = MatMul(X, WS1)
          add1 = Add(f1, s1)
          f2 = MatMul(Z, WF2)
          add2 = Add(f2, add1)
          r = Relu(add2)
          Y = MatMul(r, WOUT)
        }}
        """,
        initializer=[
            _f32(wf1, "WF1"),
            _f32(ws1, "WS1"),
            _f32(wf2, "WF2"),
            _f32(wout, "WOUT"),
        ],
    )


def test_cpp_structured_pruning_matmul_residual_add_matches_oracle():
    # Weights deliberately built so the two branches disagree about which
    # channels matter most, so the correct combined-importance keep set is
    # neither branch's own individual top-k.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(90)
    scale_f = np.where(np.arange(C) < C // 2, 3.0, 0.3).astype(np.float32)
    scale_s = np.where(np.arange(C) < C // 2, 0.3, 3.0).astype(np.float32)
    wf = rng.standard_normal((K, C)).astype(np.float32) * scale_f
    ws = rng.standard_normal((K, C)).astype(np.float32) * scale_s
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _matmul_residual_diamond_model(wf, ws, wout)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(wf.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    assert np.any(keep < C // 2) and np.any(keep >= C // 2)
    oracle = _matmul_residual_diamond_model(wf[:, keep], ws[:, keep], wout[keep, :])

    rng_x = np.random.default_rng(91)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_residual_add_transitive_chain_matches_oracle():
    K, C, Kz, Out = 8, 16, 5, 4
    rng = np.random.default_rng(92)
    wf1 = rng.standard_normal((K, C)).astype(np.float32)
    ws1 = rng.standard_normal((K, C)).astype(np.float32)
    wf2 = rng.standard_normal((Kz, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _matmul_residual_transitive_model(wf1, ws1, wf2, wout)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(wf1.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws1.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wf2.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _matmul_residual_transitive_model(
        wf1[:, keep], ws1[:, keep], wf2[:, keep], wout[keep, :]
    )

    rng_x = np.random.default_rng(93)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    z = rng_x.standard_normal((5, Kz)).astype(np.float32)
    (y,) = _run(pruned, {"X": x, "Z": z})
    (y_oracle,) = _run(oracle, {"X": x, "Z": z})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_residual_add_matches_oracle_on_fan_out_branch():
    # The MatMul/Gemm analogue of the Conv fan-out test above: `r` is read
    # both by `nxt` and, unchanged, by `add2` -- `nxt` ends up pruned on
    # both axes of WNEXT (its own output columns, as a leaf producer of
    # add2's own merge, and its own reduction rows, as an independent
    # fan-out branch reading the group's shared spine).
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(94)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wnext = rng.standard_normal((C, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          add1 = Add(f, s)
          r = Relu(add1)
          nxt = MatMul(r, WNEXT)
          add2 = Add(nxt, r)
          Y = MatMul(add2, WOUT)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(wnext, "WNEXT"),
            _f32(wout, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(wf.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wnext.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          add1 = Add(f, s)
          r = Relu(add1)
          nxt = MatMul(r, WNEXT)
          add2 = Add(nxt, r)
          Y = MatMul(add2, WOUT)
        }}
        """,
        initializer=[
            _f32(wf[:, keep], "WF"),
            _f32(ws[:, keep], "WS"),
            _f32(wnext[keep][:, keep], "WNEXT"),
            _f32(wout[keep, :], "WOUT"),
        ],
    )

    rng_x = np.random.default_rng(95)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_residual_add_declines_on_identity_shortcut():
    # y = MatMul2(Relu(Add(MatMul1(X), X))): the exact x = x + f(x)
    # transformer-residual identity-shortcut shape, no MatMul on the
    # shortcut path at all.
    C, Out = 8, 4
    rng = np.random.default_rng(95)
    w1 = rng.standard_normal((C, C)).astype(np.float32)
    w2 = rng.standard_normal((C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{C}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, W1)
          add1 = Add(f, X)
          r = Relu(add1)
          Y = MatMul(r, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_matmul_residual_add_with_bias_hop_matches_oracle():
    # One branch has a per-channel bias Add (a separate node, not Gemm's own
    # bias input) between its producer and the residual merge -- exercises
    # the wider MatMul/Gemm-only hop set and the self-consistent-then-
    # revalidate check that tells this bias Add apart from an eligible
    # residual-merge Add.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(96)
    w1 = rng.standard_normal((K, C)).astype(np.float32)
    bias = rng.standard_normal((C,)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          hb = Add(h, Bias)
          f = Relu(hb)
          s = MatMul(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          Y = MatMul(r, WOUT)
        }}
        """,
        initializer=[
            _f32(w1, "W1"),
            _f32(bias, "Bias"),
            _f32(ws, "WS"),
            _f32(wout, "WOUT"),
        ],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Bias"].dims) == [C // 2]

    importance = np.sqrt(
        np.square(np.linalg.norm(w1.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])

    rng_x = np.random.default_rng(97)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    f = np.maximum(x @ w1[:, keep] + bias[keep], 0)
    s = x @ ws[:, keep]
    y_oracle = np.maximum(f + s, 0) @ wout[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_residual_add_transposed_gemm_producer_matches_oracle():
    # One branch is a Gemm with transB=1 (weight stored [N, K]) rather than
    # a plain MatMul's [K, N] -- a regression test for weight_transposed
    # being carried correctly through the backward walk.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(98)
    w1t = rng.standard_normal((C, K)).astype(np.float32)  # [N, K] -- transB=1 layout
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = Gemm<transB = 1>(X, W1T)
          s = MatMul(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          Y = MatMul(r, WOUT)
        }}
        """,
        initializer=[_f32(w1t, "W1T"), _f32(ws, "WS"), _f32(wout, "WOUT")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1T"].dims) == [C // 2, K]

    importance = np.sqrt(
        np.square(np.linalg.norm(w1t.astype(np.float64), axis=1))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])

    rng_x = np.random.default_rng(99)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    f = x @ w1t[keep, :].T
    s = x @ ws[:, keep]
    y_oracle = np.maximum(f + s, 0) @ wout[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_residual_add_matches_oracle_on_gated_branch_with_no_projection():
    # A gated (SwiGLU-style) combine feeding directly into a residual Add,
    # with no output-projection MatMul between the Mul and the Add -- now
    # resolved the same way a gated pair outside a residual chain already
    # is: both `gate`'s and `up`'s own producers join the group's shared
    # leaf-producer set (see WalkMatmulProducerBackward's own "gated"
    # outcome), ranked and pruned together with `p`.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(100)
    wg = rng.standard_normal((K, C)).astype(np.float32)
    wu = rng.standard_normal((K, C)).astype(np.float32)
    wp = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, WG)
          gate_act = Sigmoid(gate)
          up = MatMul(X, WU)
          h = Mul(gate_act, up)
          p = MatMul(X, WP)
          addr = Add(p, h)
          r = Relu(addr)
          Y = MatMul(r, WOUT)
        }}
        """,
        initializer=[
            _f32(wg, "WG"),
            _f32(wu, "WU"),
            _f32(wp, "WP"),
            _f32(wout, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(wg.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wu.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wp.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, WG)
          gate_act = Sigmoid(gate)
          up = MatMul(X, WU)
          h = Mul(gate_act, up)
          p = MatMul(X, WP)
          addr = Add(p, h)
          r = Relu(addr)
          Y = MatMul(r, WOUT)
        }}
        """,
        initializer=[
            _f32(wg[:, keep], "WG"),
            _f32(wu[:, keep], "WU"),
            _f32(wp[:, keep], "WP"),
            _f32(wout[keep, :], "WOUT"),
        ],
    )

    rng_x = np.random.default_rng(101)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_residual_add_declines_on_bare_gqa_shortcut():
    # A residual branch whose backward walk would need to cross a fused
    # self-attention op boundary to reach a real producer -- ctx (a
    # GroupQueryAttention node's own raw output) feeds directly into the
    # residual Add, with no output-projection MatMul in between. Neither
    # GroupQueryAttention nor its Q/K/V MatMul producers can be reached
    # through it.
    K, H, D, Out = 8, 4, 4, 6
    Nqkv = H * D
    rng = np.random.default_rng(101)
    wq = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wk = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wv = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wp = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wout = rng.standard_normal((Nqkv, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[2,5,{K}] X) => (float[2,5,{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx, present_k, present_v = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={H}> (q, k, v)
          p = MatMul(X, Wp)
          addr = Add(p, ctx)
          r = Relu(addr)
          Y = MatMul(r, Wout)
        }}
        """,
        initializer=[
            _f32(wq, "Wq"),
            _f32(wk, "Wk"),
            _f32(wv, "Wv"),
            _f32(wp, "Wp"),
            _f32(wout, "Wout"),
        ],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], wq)
    np.testing.assert_array_equal(inits["Wk"], wk)
    np.testing.assert_array_equal(inits["Wv"], wv)
    np.testing.assert_array_equal(inits["Wp"], wp)
    np.testing.assert_array_equal(inits["Wout"], wout)


# --- Fused SkipLayerNormalization residual merge ----------------------------


def _skip_layer_norm_residual_diamond_model(
    wf, ws, wout, gamma, beta=None, bias=None, simplified=False, epsilon=1e-5
):
    # y = SkipLayerNormalization(MatMul_f(X), MatMul_s(X), gamma, beta?,
    # bias?) -- the SkipLayerNormalization/SkipSimplifiedLayerNormalization
    # analogue of _matmul_residual_diamond_model: two entirely independent
    # MatMul producers merge via the fused node instead of a bare Add, and
    # must therefore still share one surviving channel-index set, feeding
    # one real consumer.
    K, C = wf.shape
    Out = wout.shape[1]
    op = "SkipSimplifiedLayerNormalization" if simplified else "SkipLayerNormalization"
    initializer = [
        _f32(wf, "WF"),
        _f32(ws, "WS"),
        _f32(wout, "WOUT"),
        _f32(gamma, "Gamma"),
    ]
    inputs = ["f", "s", "Gamma"]
    if not simplified:
        inputs.append("Beta" if beta is not None else "")
        if beta is not None:
            initializer.append(_f32(beta, "Beta"))
    if bias is not None:
        inputs.append("Bias")
        initializer.append(_f32(bias, "Bias"))
    while inputs and inputs[-1] == "":
        inputs.pop()
    ins = ", ".join(inputs)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          y = com.microsoft.{op} <epsilon={epsilon}> ({ins})
          Y = MatMul(y, WOUT)
        }}
        """,
        initializer=initializer,
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    return model


def _skip_layer_norm_keep(wf, ws, C):
    importance = np.sqrt(
        np.square(np.linalg.norm(wf.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    assert np.any(keep < C // 2) and np.any(keep >= C // 2)
    return keep


def _conflicting_wf_ws(seed, K, C):
    rng = np.random.default_rng(seed)
    scale_f = np.where(np.arange(C) < C // 2, 3.0, 0.3).astype(np.float32)
    scale_s = np.where(np.arange(C) < C // 2, 0.3, 3.0).astype(np.float32)
    wf = rng.standard_normal((K, C)).astype(np.float32) * scale_f
    ws = rng.standard_normal((K, C)).astype(np.float32) * scale_s
    return rng, wf, ws


def test_cpp_structured_pruning_skip_layer_norm_residual_matches_oracle():
    K, C, Out = 8, 16, 4
    rng, wf, ws = _conflicting_wf_ws(110, K, C)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    model = _skip_layer_norm_residual_diamond_model(wf, ws, wout, gamma, beta=beta)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["WF"].dims) == [K, C // 2]
    assert list(inits["WS"].dims) == [K, C // 2]
    assert list(inits["WOUT"].dims) == [C // 2, Out]
    assert list(inits["Gamma"].dims) == [C // 2]
    assert list(inits["Beta"].dims) == [C // 2]

    keep = _skip_layer_norm_keep(wf, ws, C)
    oracle = _skip_layer_norm_residual_diamond_model(
        wf[:, keep], ws[:, keep], wout[keep, :], gamma[keep], beta=beta[keep]
    )

    rng_x = np.random.default_rng(111)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_skip_simplified_layer_norm_residual_matches_oracle():
    # SkipSimplifiedLayerNormalization -- the RMSNorm variant LLaMA-style
    # models use -- drops beta/mean-centering entirely.
    K, C, Out = 8, 16, 4
    rng, wf, ws = _conflicting_wf_ws(112, K, C)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    model = _skip_layer_norm_residual_diamond_model(
        wf, ws, wout, gamma, simplified=True
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert "Beta" not in inits

    keep = _skip_layer_norm_keep(wf, ws, C)
    oracle = _skip_layer_norm_residual_diamond_model(
        wf[:, keep], ws[:, keep], wout[keep, :], gamma[keep], simplified=True
    )

    rng_x = np.random.default_rng(113)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_skip_layer_norm_residual_with_bias_matches_oracle():
    # bias present (and, deliberately, beta absent -- SkipLayerNorm's own
    # optional inputs are independent of each other): exercises the
    # bias-idx-shift in SkipLayerNormConstNames (bias lives at input index 4
    # when beta is declared) and confirms Bias is sliced alongside Gamma.
    K, C, Out = 8, 16, 4
    rng, wf, ws = _conflicting_wf_ws(114, K, C)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    bias = rng.standard_normal((C,)).astype(np.float32)
    model = _skip_layer_norm_residual_diamond_model(wf, ws, wout, gamma, bias=bias)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Bias"].dims) == [C // 2]

    keep = _skip_layer_norm_keep(wf, ws, C)
    oracle = _skip_layer_norm_residual_diamond_model(
        wf[:, keep], ws[:, keep], wout[keep, :], gamma[keep], bias=bias[keep]
    )

    rng_x = np.random.default_rng(115)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_skip_layer_norm_residual_declines_on_nonconstant_beta():
    # Beta is a graph input, not a constant initializer -- gamma (also
    # required) is fine, but a present non-constant beta still means this
    # pass can't slice it, so the whole chain is declined and the model is
    # left byte-identical.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(116)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X, float[{C}] Beta) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          y = com.microsoft.SkipLayerNormalization <epsilon=1e-5> (f, s, Gamma, Beta)
          Y = MatMul(y, WOUT)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(wout, "WOUT"),
            _f32(gamma, "Gamma"),
        ],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_skip_layer_norm_residual_declines_on_consumed_mean_output():
    # The training-only mean output (index 1) is actually consumed here
    # (wired straight to a second graph output) -- onnxruntime's own CPU
    # kernel never actually populates it, and this pass has no basis for
    # whether pruning keeps it meaningful for whatever reads it, so the
    # whole chain is declined outright, leaving the model byte-identical.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(117)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch] MeanOut)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          y, MeanOut = com.microsoft.SkipLayerNormalization <epsilon=1e-5> (f, s, Gamma, Beta)
          Y = MatMul(y, WOUT)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(wout, "WOUT"),
            _f32(gamma, "Gamma"),
            _f32(beta, "Beta"),
        ],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_skip_layer_norm_residual_declines_on_consumed_sum_output():
    # The fourth output, input_skip_bias_sum (the raw, pre-normalization
    # f + s), is consumed directly here by a second graph output. Its shape
    # shrinks along with f/s, and this pass has no way to confirm the
    # outside consumer still expects the new, narrower width. Declined
    # outright, model left byte-identical.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(119)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch,{C}] SumOut)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          y, mean, inv_std, SumOut = com.microsoft.SkipSimplifiedLayerNormalization <epsilon=1e-5> (f, s, Gamma)
          Y = MatMul(y, WOUT)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(wout, "WOUT"),
            _f32(gamma, "Gamma"),
        ],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


# --- PRelu/Clip channel pass-through hops ------------------------------------
#
# Two "channel pass-through hop" features ported from pruning.py's own
# reference: a PRelu whose `slope` is either a scalar/single shared
# parameter (left untouched) or a genuine per-channel constant (sliced by
# the chain's own `keep` set, like a depthwise Conv hop's own weight), and a
# Clip (the `torch.nn.ReLU6` shape MobileNet/EfficientNet-Lite exports)
# crossed transparently whenever its `min`/`max` are each either omitted or
# a constant scalar. See _match_prelu_pass_through(_self,_matmul,
# _matmul_self) and _match_clip_channel_pass_through in pruning.py.


def test_cpp_structured_pruning_prelu_per_channel_pass_through_conv_matches_oracle():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(200)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    slope = rng.uniform(0.05, 0.3, size=(C1, 1, 1)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)

    def _mk(w1, b1, slope, w2):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
            {{
              h = Conv<kernel_shape=[3,3]>(X, W1, B1)
              a = PRelu(h, Slope)
              Y = Conv<kernel_shape=[3,3]>(a, W2)
            }}
            """,
            initializer=[
                _f32(w1, "W1"),
                _f32(b1, "B1"),
                _f32(slope, "Slope"),
                _f32(w2, "W2"),
            ],
        )

    model = _mk(w1, b1, slope, w2)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["Slope"].shape == (C1 // 2, 1, 1)
    # The per-channel-slope hop reuses ConvPassThrough (same as a depthwise
    # Conv hop), but PRelu has no `group` attribute of its own -- confirm the
    # port doesn't erroneously bolt one on.
    prelu_node = next(n for n in pruned.graph.node if n.op_type == "PRelu")
    assert len(prelu_node.attribute) == 0

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _mk(w1[keep], b1[keep], slope[keep], w2[:, keep])

    rng_x = np.random.default_rng(201)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_prelu_scalar_slope_left_untouched_on_conv_chain():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(202)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    slope = np.array([0.2], dtype=np.float32)  # single shared parameter.
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)

    def _mk(w1, slope, w2):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
            {{
              h = Conv<kernel_shape=[3,3]>(X, W1)
              a = PRelu(h, Slope)
              Y = Conv<kernel_shape=[3,3]>(a, W2)
            }}
            """,
            initializer=[_f32(w1, "W1"), _f32(slope, "Slope"), _f32(w2, "W2")],
        )

    model = _mk(w1, slope, w2)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    # Scalar slope: same value multiplies every channel, so it's left
    # completely untouched -- no "nothing of its own to slice" hop needed.
    np.testing.assert_array_equal(inits["Slope"], slope)

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _mk(w1[keep], slope, w2[:, keep])

    rng_x = np.random.default_rng(203)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_clip_relu6_pass_through_conv_matches_oracle():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(204)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    min_c = np.array(0.0, dtype=np.float32)
    max_c = np.array(6.0, dtype=np.float32)

    def _mk(w1, w2):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
            {{
              h = Conv<kernel_shape=[3,3]>(X, W1)
              a = Clip(h, Min, Max)
              Y = Conv<kernel_shape=[3,3]>(a, W2)
            }}
            """,
            initializer=[
                _f32(w1, "W1"),
                _f32(min_c, "Min"),
                _f32(max_c, "Max"),
                _f32(w2, "W2"),
            ],
        )

    model = _mk(w1, w2)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [C1 // 2, Cin, 3, 3]
    assert list(inits["W2"].dims) == [C2, C1 // 2, 3, 3]

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _mk(w1[keep], w2[:, keep])

    rng_x = np.random.default_rng(205)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_prelu_bare_rank1_slope_declines_on_conv_chain():
    # A bare [C] slope is deliberately *not* treated as per-channel on a Conv
    # chain (unlike a MatMul/Gemm chain's own last-axis convention): ONNX's
    # unidirectional broadcasting would align it against the *trailing* (W)
    # axis, not axis 1 -- declined, never guessed at.
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(206)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    slope = rng.uniform(0.05, 0.3, size=(C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = PRelu(h, Slope)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(slope, "Slope"), _f32(w2, "W2")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["Slope"], slope)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_prelu_nonconstant_slope_declines():
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(207)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X, float[{C1},1,1] Slope) => (float[N,{C2},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = PRelu(h, Slope)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_clip_nonconstant_bound_declines():
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(208)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    min_c = np.array(0.0, dtype=np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X, float Max) => (float[N,{C2},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = Clip(h, Min, Max)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(min_c, "Min"), _f32(w2, "W2")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_prelu_per_channel_pass_through_matmul_matches_oracle():
    K, H, Out = 8, 32, 4
    rng = np.random.default_rng(209)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    b1 = rng.standard_normal((H,)).astype(np.float32)
    slope = rng.uniform(0.05, 0.3, size=(H,)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)

    def _mk(w1, b1, slope, w2):
        return _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{Out}] Y)
            {{
              h = Gemm(X, W1, B1)
              a = PRelu(h, Slope)
              Y = MatMul(a, W2)
            }}
            """,
            initializer=[
                _f32(w1, "W1"),
                _f32(b1, "B1"),
                _f32(slope, "Slope"),
                _f32(w2, "W2"),
            ],
        )

    model = _mk(w1, b1, slope, w2)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["Slope"].shape == (H // 2,)

    keep = _oracle_keep_indices(w1, H // 2)
    oracle = _mk(w1[:, keep], b1[keep], slope[keep], w2[keep, :])

    rng_x = np.random.default_rng(210)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_prelu_scalar_slope_left_untouched_on_matmul_chain():
    K, H, Out = 8, 32, 4
    rng = np.random.default_rng(211)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    # Single shared parameter, shape [1] -- mirrors _match_prelu_pass_through*'s
    # own `if not dims: return None` bar, which (like the Conv-chain matcher)
    # declines a true rank-0 slope; [1]/[1,1,1] is the shape real exporters
    # (and this matcher) actually treat as "scalar".
    slope = np.array([0.25], dtype=np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)

    def _mk(w1, slope, w2):
        return _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{Out}] Y)
            {{
              h = MatMul(X, W1)
              a = PRelu(h, Slope)
              Y = MatMul(a, W2)
            }}
            """,
            initializer=[_f32(w1, "W1"), _f32(slope, "Slope"), _f32(w2, "W2")],
        )

    model = _mk(w1, slope, w2)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Slope"], slope)

    keep = _oracle_keep_indices(w1, H // 2)
    oracle = _mk(w1[:, keep], slope, w2[keep, :])

    rng_x = np.random.default_rng(212)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_clip_relu6_pass_through_matmul_matches_oracle():
    K, H, Out = 8, 32, 4
    rng = np.random.default_rng(213)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    min_c = np.array(0.0, dtype=np.float32)
    max_c = np.array([6.0], dtype=np.float32)  # single-element shape [1].

    def _mk(w1, w2):
        return _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{Out}] Y)
            {{
              h = MatMul(X, W1)
              a = Clip(h, Min, Max)
              Y = MatMul(a, W2)
            }}
            """,
            initializer=[
                _f32(w1, "W1"),
                _f32(min_c, "Min"),
                _f32(max_c, "Max"),
                _f32(w2, "W2"),
            ],
        )

    model = _mk(w1, w2)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H // 2]
    assert list(inits["W2"].dims) == [H // 2, Out]

    keep = _oracle_keep_indices(w1, H // 2)
    oracle = _mk(w1[:, keep], w2[keep, :])

    rng_x = np.random.default_rng(214)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_conv_residual_prelu_pass_through_hop_matches_oracle():
    # A PRelu per-channel hop crossed by the *backward* walk
    # (WalkConvProducerBackward/MatchPreluPassThroughSelf), not just the
    # forward one -- exercises the residual-chain insertion point and the
    # ApplyChains "group" attribute guard (PRelu must not get one).
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(215)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    slope = rng.uniform(0.05, 0.3, size=(C, 1, 1)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)

    def _mk(w_f, slope, w_s, w_out):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{Cout},6,6] Y)
            {{
              f0 = Conv<kernel_shape=[3,3]>(X, WF)
              f = PRelu(f0, Slope)
              s = Conv<kernel_shape=[3,3]>(X, WS)
              addr = Add(f, s)
              r = Relu(addr)
              Y = Conv<kernel_shape=[3,3]>(r, WOUT)
            }}
            """,
            initializer=[
                _f32(w_f, "WF"),
                _f32(slope, "Slope"),
                _f32(w_s, "WS"),
                _f32(w_out, "WOUT"),
            ],
        )

    model = _mk(w_f, slope, w_s, w_out)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["Slope"].shape == (C // 2, 1, 1)
    prelu_node = next(n for n in pruned.graph.node if n.op_type == "PRelu")
    assert len(prelu_node.attribute) == 0

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _mk(w_f[keep], slope[keep], w_s[keep], w_out[:, keep])

    rng_x = np.random.default_rng(216)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_residual_clip_pass_through_hop_matches_oracle():
    # A Clip crossed by the *backward* MatMul/Gemm walk
    # (WalkMatmulProducerBackward) -- exercises that insertion point too.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(217)
    w_f = rng.standard_normal((K, C)).astype(np.float32)
    w_s = rng.standard_normal((K, C)).astype(np.float32)
    w_out = rng.standard_normal((C, Out)).astype(np.float32)
    min_c = np.array(0.0, dtype=np.float32)
    max_c = np.array(6.0, dtype=np.float32)

    def _mk(w_f, w_s, w_out):
        return _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{Out}] Y)
            {{
              f0 = MatMul(X, WF)
              f = Clip(f0, Min, Max)
              s = MatMul(X, WS)
              addr = Add(f, s)
              r = Relu(addr)
              Y = MatMul(r, WOUT)
            }}
            """,
            initializer=[
                _f32(w_f, "WF"),
                _f32(min_c, "Min"),
                _f32(max_c, "Max"),
                _f32(w_s, "WS"),
                _f32(w_out, "WOUT"),
            ],
        )

    model = _mk(w_f, w_s, w_out)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.T.astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.T.astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _mk(w_f[:, keep], w_s[:, keep], w_out[keep, :])

    rng_x = np.random.default_rng(218)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- Conv chain: GroupNormalization pass-through hop -------------------------
#
# `Conv -> GroupNormalization -> Conv`: mirrors test_pruning.py's own
# `_group_norm_conv_pair_model` and its group-norm-pass-through test coverage.
# Unlike PRelu/Clip, a mid-chain GroupNorm hop constrains `ChainGroup()`'s own
# per-block `keep` selection to its own `num_groups` (see GroupNormPassThrough
# and ChainGroup in structured_pruning_entry.cpp), so the oracle here uses
# `_oracle_keep_indices_conv_grouped`, not the plain `_oracle_keep_indices_conv`
# every other Conv-chain hop test uses.


def _group_norm_conv_pair_model(
    w1, w2, gn_scale, gn_bias, num_groups, group1=1, b1=None
):
    Cin, C2 = w1.shape[1] * group1, w2.shape[0]
    initializer = [
        _f32(w1, "W1"),
        _f32(w2, "W2"),
        _f32(gn_scale, "GNScale"),
        _f32(gn_bias, "GNBias"),
    ]
    g1 = f", group={group1}" if group1 != 1 else ""
    if b1 is not None:
        conv1 = f"h = Conv<kernel_shape=[3,3]{g1}>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = f"h = Conv<kernel_shape=[3,3]{g1}>(X, W1)"
    return _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
        {{
          {conv1}
          gn = GroupNormalization<num_groups={num_groups}, epsilon=1e-05>(h, GNScale, GNBias)
          Y = Conv<kernel_shape=[3,3]>(gn, W2)
        }}
        """,
        initializer=initializer,
    )


def test_cpp_structured_pruning_group_norm_pass_through_matches_oracle():
    Cin, C1, C2, num_groups = 3, 16, 8, 4
    rng = np.random.default_rng(220)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    gn_scale = rng.standard_normal((C1,)).astype(np.float32)
    gn_bias = rng.standard_normal((C1,)).astype(np.float32)
    model = _group_norm_conv_pair_model(w1, w2, gn_scale, gn_bias, num_groups, b1=b1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    # `num_groups` (a node attribute, not a tensor) is unchanged; the
    # surviving channel count must still divide it evenly.
    assert inits["W1"].shape[0] % num_groups == 0
    assert inits["W1"].shape[0] < C1  # actually pruned, not a no-op
    assert inits["GNScale"].shape == inits["GNBias"].shape == (C1 // 2,)
    gn_node = next(n for n in pruned.graph.node if n.op_type == "GroupNormalization")
    assert next(a.i for a in gn_node.attribute if a.name == "num_groups") == num_groups

    keep = _oracle_keep_indices_conv_grouped(w1, num_groups, 0.5)
    oracle = _group_norm_conv_pair_model(
        w1[keep], w2[:, keep], gn_scale[keep], gn_bias[keep], num_groups, b1=b1[keep]
    )

    rng_x = np.random.default_rng(221)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_group_norm_num_groups_mismatch_with_grouped_conv_declines():
    # A mid-chain GroupNorm hop's own `num_groups` disagreeing with a
    # same-chain grouped Conv producer's own `group` -- the two partitions'
    # block boundaries wouldn't generally align, so the whole chain is
    # declined outright, never guessed at, the same bar a plain
    # producer_group != consumer_group mismatch already gets.
    Cin, C1, C2, group1, num_groups = 4, 8, 8, 2, 4
    rng = np.random.default_rng(222)
    w1 = rng.standard_normal((C1, Cin // group1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    gn_scale = rng.standard_normal((C1,)).astype(np.float32)
    gn_bias = rng.standard_normal((C1,)).astype(np.float32)
    model = _group_norm_conv_pair_model(
        w1, w2, gn_scale, gn_bias, num_groups, group1=group1
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["GNScale"], gn_scale)
    np.testing.assert_array_equal(inits["GNBias"], gn_bias)


def test_cpp_structured_pruning_group_norm_tied_scale_bias_declines():
    # `scale`/`bias` naming the *same* tensor -- double-slicing it in
    # ApplyChains's own per-hop loop would corrupt it, so this is declined
    # outright, mirroring pruning.py's own tied-name bar
    # (_match_group_norm_pass_through).
    Cin, C1, C2, num_groups = 3, 8, 4, 2
    rng = np.random.default_rng(223)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    tied = rng.standard_normal((C1,)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          gn = GroupNormalization<num_groups={num_groups}, epsilon=1e-05>(h, Tied, Tied)
          Y = Conv<kernel_shape=[3,3]>(gn, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(tied, "Tied")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["Tied"], tied)


# --- Conv chain: Resize channel-safe pass-through hop -------------------------
#
# `Conv -> Resize(scales, spatial-only) -> Conv`: the U-Net/diffusion-model-
# decoder-style upsampling shape. Mirrors test_pruning.py's own
# `_resize_conv_pair_model` and its Resize-pass-through test coverage.


def _resize_conv_pair_model(w1, w2, scales, b1=None, spatial=8, out_spatial_hw=None):
    Cin, C2 = w1.shape[1], w2.shape[0]
    initializer = [
        _f32(w1, "W1"),
        _f32(w2, "W2"),
        onnx.numpy_helper.from_array(np.asarray(scales, dtype=np.float32), "Scales"),
    ]
    if b1 is not None:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1)"
    after_conv1 = spatial - 2
    if out_spatial_hw is None:
        mid_h = round(after_conv1 * scales[2])
        out_spatial_hw = mid_h - 2
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{out_spatial_hw},{out_spatial_hw}] Y)
        {{
          {conv1}
          p = Resize<mode="nearest">(h, , Scales)
          Y = Conv<kernel_shape=[3,3]>(p, W2)
        }}
        """,
        initializer=initializer,
    )


def test_cpp_structured_pruning_resize_channel_safe_pass_through_matches_oracle():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(224)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    scales = [1.0, 1.0, 2.0, 2.0]
    model = _resize_conv_pair_model(w1, w2, scales, b1=b1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [C1 // 2, Cin, 3, 3]
    assert list(inits["W2"].dims) == [C2, C1 // 2, 3, 3]

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _resize_conv_pair_model(w1[keep], w2[:, keep], scales, b1=b1[keep])

    rng_x = np.random.default_rng(225)
    x = rng_x.standard_normal((2, Cin, 8, 8)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_resize_channel_affecting_declines():
    # scales[1] (the channel axis) == 2.0 -- genuinely resizes the channel
    # axis itself, so it must be declined outright, never guessed at.
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(226)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    scales = [1.0, 2.0, 1.0, 1.0]
    w2 = rng.standard_normal((C2, C1 * 2, 3, 3)).astype(np.float32)
    model = _resize_conv_pair_model(w1, w2, scales, out_spatial_hw=4)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_resize_dynamic_scales_declines():
    # `scales` computed at runtime (Shape -> Cast) rather than a constant
    # initializer -- this pass cannot know which axis is affected without
    # evaluating the graph, so it must decline outright, never guessed at.
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(227)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},8,8] X) => (float[N,{C2},4,4] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          shp = Shape(h)
          scales_dyn = Cast<to=1>(shp)
          p = Resize<mode="nearest">(h, , scales_dyn)
          Y = Conv<kernel_shape=[3,3]>(p, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_conv_residual_resize_pass_through_hop_matches_oracle():
    # A channel-safe Resize crossed by the *backward* walk
    # (WalkConvProducerBackward) -- exercises the residual-chain insertion
    # point, not just the forward one.
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(228)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    scales = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)

    def _mk(w_f, w_s, w_out):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{Cout},6,6] Y)
            {{
              f0 = Conv<kernel_shape=[3,3]>(X, WF)
              f = Resize<mode="nearest">(f0, , Scales)
              s = Conv<kernel_shape=[3,3]>(X, WS)
              addr = Add(f, s)
              r = Relu(addr)
              Y = Conv<kernel_shape=[3,3]>(r, WOUT)
            }}
            """,
            initializer=[
                _f32(w_f, "WF"),
                onnx.numpy_helper.from_array(scales, "Scales"),
                _f32(w_s, "WS"),
                _f32(w_out, "WOUT"),
            ],
        )

    model = _mk(w_f, w_s, w_out)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _mk(w_f[keep], w_s[keep], w_out[:, keep])

    rng_x = np.random.default_rng(229)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- Conv chain: Pad channel-safe pass-through hop ----------------------------
#
# `Conv -> Pad -> Conv`: mirrors test_pruning.py's own `_pad_conv_pair_model`
# and its Pad-pass-through test coverage.


def _pad_conv_pair_model(w1, w2, pads, b1=None, spatial=8):
    """`Conv -> Pad -> Conv`, `pads` the raw 8-element (`2 * rank` for a
    rank-4 NCHW tensor) ONNX `pads` layout: `[x1_begin, ..., xk_begin,
    x1_end, ..., xk_end]`."""
    Cin, C2 = w1.shape[1], w2.shape[0]
    pads = np.asarray(pads, dtype=np.int64)
    rank = len(pads) // 2
    initializer = [
        _f32(w1, "W1"),
        _f32(w2, "W2"),
        onnx.numpy_helper.from_array(pads, "Pads"),
    ]
    if b1 is not None:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1)"
    after_conv1 = spatial - 2
    mid_h = after_conv1 + pads[2] + pads[rank + 2]  # axis 2 (H) begin+end pad
    out_spatial = mid_h - 2
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{out_spatial},{out_spatial}] Y)
        {{
          {conv1}
          p = Pad<mode="constant">(h, Pads)
          Y = Conv<kernel_shape=[3,3]>(p, W2)
        }}
        """,
        initializer=initializer,
    )


def test_cpp_structured_pruning_pad_channel_safe_pass_through_matches_oracle():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(230)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    pads = [0, 0, 1, 1, 0, 0, 1, 1]
    model = _pad_conv_pair_model(w1, w2, pads, b1=b1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [C1 // 2, Cin, 3, 3]
    assert list(inits["W2"].dims) == [C2, C1 // 2, 3, 3]

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _pad_conv_pair_model(w1[keep], w2[:, keep], pads, b1=b1[keep])

    rng_x = np.random.default_rng(231)
    x = rng_x.standard_normal((2, Cin, 8, 8)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_pad_channel_affecting_declines():
    # Nonzero padding on axis 1 (channel) -- changes the output channel
    # count outright, so this must be declined, never guessed at.
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(232)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    pads = [0, 1, 0, 0, 0, 1, 0, 0]
    w2 = rng.standard_normal((C2, C1 + 2, 3, 3)).astype(np.float32)
    model = _pad_conv_pair_model(w1, w2, pads)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_pad_dynamic_pads_declines():
    # `pads` computed at runtime (a non-constant node output) rather than a
    # constant initializer -- declined outright for the same reason as a
    # dynamic Resize `scales` above.
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(233)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},8,8] X, int64[8] PadsIn) => (float[N,{C2},4,4] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          pads_dyn = Identity(PadsIn)
          p = Pad<mode="constant">(h, pads_dyn)
          Y = Conv<kernel_shape=[3,3]>(p, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_conv_residual_pad_pass_through_hop_matches_oracle():
    # A channel-safe Pad crossed by the *backward* walk
    # (WalkConvProducerBackward) -- exercises the residual-chain insertion
    # point, not just the forward one.
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(234)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    # All-zero pads (a no-op Pad node) -- keeps both Add operands' shapes
    # equal (the merge point's own requirement) while still exercising the
    # channel-safety matcher (pads[1] == pads[rank+1] == 0 is trivially true).
    pads = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)

    def _mk(w_f, w_s, w_out):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{Cout},6,6] Y)
            {{
              f0 = Conv<kernel_shape=[3,3]>(X, WF)
              f = Pad<mode="constant">(f0, Pads)
              s = Conv<kernel_shape=[3,3]>(X, WS)
              addr = Add(f, s)
              r = Relu(addr)
              Y = Conv<kernel_shape=[3,3]>(r, WOUT)
            }}
            """,
            initializer=[
                _f32(w_f, "WF"),
                onnx.numpy_helper.from_array(pads, "Pads"),
                _f32(w_s, "WS"),
                _f32(w_out, "WOUT"),
            ],
        )

    model = _mk(w_f, w_s, w_out)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _mk(w_f[keep], w_s[keep], w_out[:, keep])

    rng_x = np.random.default_rng(235)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- Conv chain: InstanceNormalization pass-through hop -----------------------
#
# `Conv -> InstanceNormalization -> Conv`: mirrors test_pruning.py's own
# `_instance_norm_conv_pair_model` and its instance-norm-pass-through test
# coverage. Unlike GroupNormalization above, InstanceNormalization carries no
# `num_groups`/uniform-per-block constraint at all -- its own mean/variance
# are computed per instance *per channel*, never pooled across a group, so an
# arbitrary channel subset may be kept/dropped with no group-boundary-drift
# risk to guard against, and `scale`/`B` are held to a *strictly* rank-1 bar
# (not GroupNorm's own looser `FlatChannelConst`) since they are carried on a
# plain ConvPassThrough, whose own slicing always slices axis 0 -- see
# MatchInstanceNormPassThrough's own comment in structured_pruning_entry.cpp.


def _instance_norm_conv_pair_model(w1, w2, in_scale, in_bias, b1=None, spatial=10):
    Cin, C2 = w1.shape[1], w2.shape[0]
    initializer = [
        _f32(w1, "W1"),
        _f32(w2, "W2"),
        _f32(in_scale, "INScale"),
        _f32(in_bias, "INBias"),
    ]
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
          n = InstanceNormalization<epsilon=1e-05>(h, INScale, INBias)
          Y = Conv<kernel_shape=[3,3]>(n, W2)
        }}
        """,
        initializer=initializer,
    )


def test_cpp_structured_pruning_instance_norm_pass_through_matches_oracle():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(240)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    in_scale = rng.standard_normal((C1,)).astype(np.float32)
    in_bias = rng.standard_normal((C1,)).astype(np.float32)
    model = _instance_norm_conv_pair_model(w1, w2, in_scale, in_bias, b1=b1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["W1"].shape[0] == C1 // 2
    assert inits["INScale"].shape == inits["INBias"].shape == (C1 // 2,)
    # No `group` attribute is ever added to a non-Conv hop node.
    in_node = next(n for n in pruned.graph.node if n.op_type == "InstanceNormalization")
    assert [a.name for a in in_node.attribute] == ["epsilon"]

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _instance_norm_conv_pair_model(
        w1[keep], w2[:, keep], in_scale[keep], in_bias[keep], b1=b1[keep]
    )

    rng_x = np.random.default_rng(241)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_instance_norm_tied_scale_bias_declines():
    # `scale`/`B` naming the *same* tensor -- double-slicing it would corrupt
    # it, so the whole chain is declined outright, mirroring
    # MatchInstanceNormPassThrough's own tied-name bar.
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(242)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    tied = rng.standard_normal((C1,)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          n = InstanceNormalization<epsilon=1e-05>(h, Tied, Tied)
          Y = Conv<kernel_shape=[3,3]>(n, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(tied, "Tied")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["Tied"], tied)


def test_cpp_structured_pruning_instance_norm_non_1d_scale_declines():
    # A `scale` shaped `[1, C1]` -- FlatChannelConst's own looser
    # "prod(dims) == dims[-1]" bar would admit this (a real GroupNorm hop
    # reuses that check), but MatchInstanceNormPassThrough deliberately holds
    # InstanceNorm's own `scale`/`B` to a *strictly* rank-1 bar instead (see
    # that matcher's own comment for why: this hop's `scale`/`B` are carried
    # on a plain ConvPassThrough, whose own slicing always slices axis 0,
    # which is only ever the right axis when rank == 1). Declined outright
    # rather than mis-sliced.
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(243)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    in_scale_2d = rng.standard_normal((1, C1)).astype(np.float32)
    in_bias = rng.standard_normal((C1,)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          n = InstanceNormalization<epsilon=1e-05>(h, INScale, INBias)
          Y = Conv<kernel_shape=[3,3]>(n, W2)
        }}
        """,
        initializer=[
            _f32(w1, "W1"),
            _f32(w2, "W2"),
            _f32(in_scale_2d, "INScale"),
            _f32(in_bias, "INBias"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["INScale"], in_scale_2d)
    np.testing.assert_array_equal(inits["INBias"], in_bias)


def test_cpp_structured_pruning_conv_residual_instance_norm_pass_through_matches_oracle():
    # An InstanceNormalization hop crossed by the *backward* walk
    # (WalkConvProducerBackward/MatchInstanceNormPassThroughSelf), not just
    # the forward one every test above already covers -- exercises the
    # residual-chain insertion point, mirroring
    # test_cpp_structured_pruning_conv_residual_prelu_pass_through_hop_matches_oracle's
    # own shape.
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(244)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    in_scale = rng.standard_normal((C,)).astype(np.float32)
    in_bias = rng.standard_normal((C,)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)

    def _mk(w_f, in_scale, in_bias, w_s, w_out):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{Cout},6,6] Y)
            {{
              f0 = Conv<kernel_shape=[3,3]>(X, WF)
              f = InstanceNormalization<epsilon=1e-05>(f0, INScale, INBias)
              s = Conv<kernel_shape=[3,3]>(X, WS)
              addr = Add(f, s)
              r = Relu(addr)
              Y = Conv<kernel_shape=[3,3]>(r, WOUT)
            }}
            """,
            initializer=[
                _f32(w_f, "WF"),
                _f32(in_scale, "INScale"),
                _f32(in_bias, "INBias"),
                _f32(w_s, "WS"),
                _f32(w_out, "WOUT"),
            ],
        )

    model = _mk(w_f, in_scale, in_bias, w_s, w_out)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["WF"].shape[0] == C // 2
    assert inits["INScale"].shape == inits["INBias"].shape == (C // 2,)
    assert inits["WS"].shape[0] == C // 2

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _mk(w_f[keep], in_scale[keep], in_bias[keep], w_s[keep], w_out[:, keep])

    rng_x = np.random.default_rng(245)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- CausalConvWithState pass-through (plain ai.onnx, opset 27) --------------
#
# `Conv -> CausalConvWithState(depthwise) -> Conv`: mirrors test_pruning.py's
# own `_causal_conv_model` and its causal-conv-with-state-pass-through test
# coverage. This op has a real CPU kernel in this environment's onnxruntime,
# so every test below runs the real fused op end to end (via `_run27`, since
# opset 27 is still "under development" per onnxruntime's own load-time
# guard) rather than a decomposed-proxy fallback. Unlike a depthwise Conv,
# this op also carries an optional `past_state` (input 3) -- sliceable only
# when it's itself a constant of the documented rank-3 `(*, n_channels, *)`
# shape (axis 1 == n_channels), see MatchCausalConvWithStatePassThrough's own
# comment in structured_pruning_entry.cpp -- and a second output
# (`present_state`) that is a runtime output, never a tensor this pass
# slices.


def _causal_conv_model(C=8, K=6, L=5, kernel=3, seed=0, past_state=None, bias=True):
    # Conv(K -> C, kernel=3) -> CausalConvWithState(C, depthwise, kernel) ->
    # Conv(C -> Out, kernel=1). `past_state`, if given, is a constant
    # ``(1, C, kernel - 1)`` array wired as the op's own 4th input; ``None``
    # leaves it unconnected (needs no slicing at all).
    rng = np.random.default_rng(seed)
    w0 = rng.standard_normal((C, K, 3)).astype(np.float32) * 0.5
    b0 = rng.standard_normal((C,)).astype(np.float32) * 0.1
    w1 = rng.standard_normal((C, 1, kernel)).astype(np.float32) * 0.5
    b1 = rng.standard_normal((C,)).astype(np.float32) * 0.1
    out = 5
    w2 = rng.standard_normal((out, C, 1)).astype(np.float32) * 0.5
    b2 = rng.standard_normal((out,)).astype(np.float32) * 0.1

    initializer = [
        _f32(w0, "W0"),
        _f32(b0, "B0"),
        _f32(w1, "W1"),
        _f32(w2, "W2"),
        _f32(b2, "B2"),
    ]
    cc_inputs = "h0, W1"
    if bias:
        initializer.append(_f32(b1, "B1"))
        cc_inputs += ", B1"
    else:
        cc_inputs += ", "
    if past_state is not None:
        initializer.append(_f32(np.asarray(past_state), "PastState"))
        cc_inputs += ", PastState"

    body = f"""
        g (float[1,{K},{L}] X) => (float[1,{out},{L}] Y)
        {{
          h0 = Conv<kernel_shape=[3], pads=[1,1]>(X, W0, B0)
          h1, ps = CausalConvWithState<activation="none">({cc_inputs})
          Y = Conv<kernel_shape=[1]>(h1, W2, B2)
        }}
        """
    model = _model(body, opset=27)
    model.graph.initializer.extend(initializer)
    return model, dict(
        C=C,
        K=K,
        L=L,
        kernel=kernel,
        out=out,
        w0=w0,
        b0=b0,
        w1=w1,
        b1=b1 if bias else None,
        w2=w2,
        b2=b2,
    )


def _causal_conv_node(model):
    return next(n for n in model.graph.node if n.op_type == "CausalConvWithState")


def test_cpp_structured_pruning_causal_conv_with_state_pass_through_matches_oracle():
    model, cfg = _causal_conv_model(C=8, K=6, L=5, kernel=3, seed=250)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    # No `group` attribute is ever added to a non-Conv hop node.
    node = _causal_conv_node(pruned)
    assert [a.name for a in node.attribute] == ["activation"]

    dims = {t.name: t.dims[0] for t in pruned.graph.initializer}
    assert dims["W0"] == 4  # C halved
    assert dims["W1"] == 4  # depthwise weight follows the same keep set
    w2_in_channels = next(t for t in pruned.graph.initializer if t.name == "W2").dims[1]
    assert w2_in_channels == 4  # consumer's in_channels axis

    imp = np.linalg.norm(cfg["w0"].reshape(cfg["C"], -1), axis=1)
    keep = np.sort(np.argsort(-imp)[: cfg["C"] // 2])

    rng = np.random.default_rng(251)
    x = rng.standard_normal((1, cfg["K"], cfg["L"])).astype(np.float32)
    y_pruned = _run27(pruned, {"X": x})[0]

    oracle, _ = _causal_conv_model(
        C=len(keep), K=cfg["K"], L=cfg["L"], kernel=cfg["kernel"]
    )
    # Every tensor built here overrides the oracle model's own freshly
    # (differently) randomized placeholder of the same name -- only the
    # *shape* of that placeholder construction is actually used.
    oracle_inits = {
        "W0": cfg["w0"][keep],
        "B0": cfg["b0"][keep],
        "W1": cfg["w1"][keep],
        "B1": cfg["b1"][keep],
        "W2": cfg["w2"][:, keep, :],
        "B2": cfg["b2"],  # Consumer's own out_channels axis is untouched.
    }
    for init in oracle.graph.initializer:
        init.CopyFrom(onnx.numpy_helper.from_array(oracle_inits[init.name], init.name))
    y_oracle = _run27(oracle, {"X": x})[0]
    np.testing.assert_array_equal(y_pruned, y_oracle)


def test_cpp_structured_pruning_causal_conv_with_state_constant_past_state_sliced():
    C, kernel = 8, 3
    rng = np.random.default_rng(252)
    past = rng.standard_normal((1, C, kernel - 1)).astype(np.float32)
    model, cfg = _causal_conv_model(
        C=C, K=6, L=5, kernel=kernel, seed=252, past_state=past
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    imp = np.linalg.norm(cfg["w0"].reshape(C, -1), axis=1)
    keep = np.sort(np.argsort(-imp)[: C // 2])

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["PastState"], past[:, keep, :])


def test_cpp_structured_pruning_causal_conv_with_state_wrong_shape_past_state_declines():
    # A constant `past_state` of any shape *other* than the documented
    # rank-3 `(*, n_channels, *)` -- here, axis 1 deliberately doesn't equal
    # `n_channels` -- declines the whole hop outright, never guessed at,
    # mirroring MatchCausalConvWithStatePassThrough's own bar.
    C, kernel = 8, 3
    rng = np.random.default_rng(253)
    wrong_past = rng.standard_normal((1, C + 1, kernel - 1)).astype(np.float32)
    model, cfg = _causal_conv_model(
        C=C, K=6, L=5, kernel=kernel, seed=253, past_state=wrong_past
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W0"], cfg["w0"])
    np.testing.assert_array_equal(inits["W1"], cfg["w1"])
    np.testing.assert_array_equal(inits["W2"], cfg["w2"])
    np.testing.assert_array_equal(inits["PastState"], wrong_past)


# --- Concat-merged (skip-connection) chains ----------------------------------


def test_cpp_structured_pruning_matmul_concat_matches_oracle():
    K, C1, C2, Out = 8, 6, 10, 4
    rng = np.random.default_rng(110)
    w1 = rng.standard_normal((K, C1)).astype(np.float32)
    w2 = rng.standard_normal((K, C2)).astype(np.float32)
    wout = rng.standard_normal((C1 + C2, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(a, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep1 = np.sort(
        np.argsort(-np.linalg.norm(w1.astype(np.float64), axis=0))[: C1 // 2]
    )
    keep2 = np.sort(
        np.argsort(-np.linalg.norm(w2.astype(np.float64), axis=0))[: C2 // 2]
    )
    global_keep = np.concatenate([keep1, keep2 + C1])
    oracle = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(a, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[
            _f32(w1[:, keep1], "W1"),
            _f32(w2[:, keep2], "W2"),
            _f32(wout[global_keep, :], "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(111)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_conv_concat_matches_oracle():
    Cin, C1, C2, Cout = 3, 8, 12, 6
    rng = np.random.default_rng(112)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, Cin, 3, 3)).astype(np.float32)
    wout = rng.standard_normal((Cout, C1 + C2, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          a = Conv<kernel_shape=[3,3]>(X, W1)
          b = Conv<kernel_shape=[3,3]>(X, W2)
          m = Concat<axis = 1>(a, b)
          Y = Conv<kernel_shape=[1,1]>(m, WOUT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep1 = np.sort(
        np.argsort(-np.linalg.norm(w1.reshape(C1, -1).astype(np.float64), axis=1))[
            : C1 // 2
        ]
    )
    keep2 = np.sort(
        np.argsort(-np.linalg.norm(w2.reshape(C2, -1).astype(np.float64), axis=1))[
            : C2 // 2
        ]
    )
    global_keep = np.concatenate([keep1, keep2 + C1])
    oracle = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          a = Conv<kernel_shape=[3,3]>(X, W1)
          b = Conv<kernel_shape=[3,3]>(X, W2)
          m = Concat<axis = 1>(a, b)
          Y = Conv<kernel_shape=[1,1]>(m, WOUT)
        }}
        """,
        initializer=[
            _f32(w1[keep1], "W1"),
            _f32(w2[keep2], "W2"),
            _f32(wout[:, global_keep], "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(113)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_concat_composed_residual_branch_matches_oracle():
    # One Concat operand ("r") resolves through a whole eligible-Add
    # residual group instead of a bare producer -- both WF/WS join that
    # branch's own combined-importance leaf-producer set, sharing one keep
    # index set, entirely independent of the other ("b") branch's own.
    K, C, C2, Out = 8, 16, 6, 4
    rng = np.random.default_rng(114)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    w2 = rng.standard_normal((K, C2)).astype(np.float32)
    wout = rng.standard_normal((C + C2, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(r, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(w2, "W2"),
            _f32(wout, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance1 = np.sqrt(
        np.square(np.linalg.norm(wf.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    keep1 = np.sort(np.argsort(-importance1)[: C // 2])
    keep2 = np.sort(
        np.argsort(-np.linalg.norm(w2.astype(np.float64), axis=0))[: C2 // 2]
    )
    global_keep = np.concatenate([keep1, keep2 + C])
    oracle = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(r, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[
            _f32(wf[:, keep1], "WF"),
            _f32(ws[:, keep1], "WS"),
            _f32(w2[:, keep2], "W2"),
            _f32(wout[global_keep, :], "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(115)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_concat_gated_branch_matches_oracle():
    # A gated (SwiGLU-style) combine feeds a Concat operand directly, with
    # no real producer's raw output in between -- both `gate`'s and `up`'s
    # own producers become this one branch's own `producers` tuple, ranked
    # together by combined importance, entirely independent of `b`'s own.
    K, C, C2, Out = 8, 16, 6, 4
    rng = np.random.default_rng(121)
    wg = rng.standard_normal((K, C)).astype(np.float32)
    wu = rng.standard_normal((K, C)).astype(np.float32)
    w2 = rng.standard_normal((K, C2)).astype(np.float32)
    wout = rng.standard_normal((C + C2, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, WG)
          gate_act = Sigmoid(gate)
          up = MatMul(X, WU)
          h = Mul(gate_act, up)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(h, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[
            _f32(wg, "WG"),
            _f32(wu, "WU"),
            _f32(w2, "W2"),
            _f32(wout, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance1 = np.sqrt(
        np.square(np.linalg.norm(wg.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wu.astype(np.float64), axis=0))
    )
    keep1 = np.sort(np.argsort(-importance1)[: C // 2])
    keep2 = np.sort(
        np.argsort(-np.linalg.norm(w2.astype(np.float64), axis=0))[: C2 // 2]
    )
    global_keep = np.concatenate([keep1, keep2 + C])
    oracle = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, WG)
          gate_act = Sigmoid(gate)
          up = MatMul(X, WU)
          h = Mul(gate_act, up)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(h, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[
            _f32(wg[:, keep1], "WG"),
            _f32(wu[:, keep1], "WU"),
            _f32(w2[:, keep2], "W2"),
            _f32(wout[global_keep, :], "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(122)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_concat_declines_on_fan_out_branch():
    # Branch `a` also feeds `Z` directly -- a real extra consumer a Concat
    # branch has no fan-out resolution for (unlike a residual/merge group)
    # -- the whole Concat node is declined, left completely untouched.
    K, C1, C2, Out = 8, 6, 10, 4
    rng = np.random.default_rng(116)
    w1 = rng.standard_normal((K, C1)).astype(np.float32)
    w2 = rng.standard_normal((K, C2)).astype(np.float32)
    wout = rng.standard_normal((C1 + C2, Out)).astype(np.float32)
    wextra = rng.standard_normal((C1, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch,{Out}] Z)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(a, b)
          Y = MatMul(m, WOUT)
          Z = MatMul(a, WEXTRA)
        }}
        """,
        initializer=[
            _f32(w1, "W1"),
            _f32(w2, "W2"),
            _f32(wout, "WOUT"),
            _f32(wextra, "WEXTRA"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_cpp_structured_pruning_conv_concat_admits_block_aligned_grouped_conv_consumer():
    # Each branch here is exactly one `group`-sized block wide (block =
    # (C1 + C2) // group = 8, matching both C1 and C2), so every one of the
    # grouped consumer's own blocks falls entirely within one branch --
    # ConcatBranchesAlignToConsumerGroup admits this rather than declining
    # it outright (this test used to assert the opposite -- a narrower-
    # than-Python C++-port gap, now closed; the pure-Python
    # apply_structured_pruning has always pruned this exact shape -- see
    # ConcatBranchesAlignToConsumerGroup's own comment for the full
    # block-alignment safety argument). See
    # test_cpp_structured_pruning_conv_concat_declines_on_non_block_aligned_grouped_consumer
    # below for the genuinely-unsafe (straddling-block) case, which still
    # declines.
    Cin, C1, C2, Cout, group = 3, 8, 8, 8, 2
    rng = np.random.default_rng(120)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, Cin, 3, 3)).astype(np.float32)
    wout = rng.standard_normal((Cout, (C1 + C2) // group, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          a = Conv<kernel_shape=[3,3]>(X, W1)
          b = Conv<kernel_shape=[3,3]>(X, W2)
          m = Concat<axis = 1>(a, b)
          Y = Conv<kernel_shape=[1,1],group={group}>(m, WOUT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(wout, "WOUT")],
    )
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    keep_a = _oracle_keep_indices_conv(w1, C1 - round(C1 * 0.5))
    keep_b = _oracle_keep_indices_conv(w2, C2 - round(C2 * 0.5))
    global_keep = np.concatenate([keep_a, keep_b + C1])
    wout_sliced = _oracle_slice_grouped_consumer_conv(wout, global_keep, group, C1 + C2)
    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    np.testing.assert_array_equal(inits["W1"], w1[keep_a])
    np.testing.assert_array_equal(inits["W2"], w2[keep_b])
    np.testing.assert_array_equal(inits["WOUT"], wout_sliced)


def test_cpp_structured_pruning_conv_concat_declines_on_non_block_aligned_grouped_consumer():
    # C1 (5) doesn't divide evenly into the consumer's own block size
    # (block = (C1 + C2) // group = 8), so branch B's own fixed offset (5)
    # falls in the interior of the first block rather than at its edge --
    # ConcatBranchesAlignToConsumerGroup declines this outright rather than
    # guessing how the block's own uniform-survivor-count budget should
    # split between the two branches (see that function's own comment for
    # the full straddling-block counter-example) -- the chain is left
    # completely untouched, matching the pure-Python apply_structured_pruning
    # exactly.
    Cin, C1, C2, Cout, group = 3, 5, 11, 8, 2
    rng = np.random.default_rng(121)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, Cin, 3, 3)).astype(np.float32)
    wout = rng.standard_normal((Cout, (C1 + C2) // group, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          a = Conv<kernel_shape=[3,3]>(X, W1)
          b = Conv<kernel_shape=[3,3]>(X, W2)
          m = Concat<axis = 1>(a, b)
          Y = Conv<kernel_shape=[1,1],group={group}>(m, WOUT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(wout, "WOUT")],
    )
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_cpp_structured_pruning_matmul_concat_accepts_positive_last_axis_when_rank_known():
    K, C1, C2, Out = 8, 6, 10, 4
    rng = np.random.default_rng(117)
    w1 = rng.standard_normal((K, C1)).astype(np.float32)
    w2 = rng.standard_normal((K, C2)).astype(np.float32)
    wout = rng.standard_normal((C1 + C2, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          m = Concat<axis = 1>(a, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(wout, "WOUT")],
    )
    # A positive axis is only recognized as "last" once at least one
    # operand's rank is confirmed via value_info -- add it directly rather
    # than running full-graph shape inference (whose validity elsewhere in
    # the graph this test doesn't care about).
    model.graph.value_info.append(
        onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [None, C1])
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["W1"].shape == (K, C1 // 2)
    assert inits["W2"].shape == (K, C2 // 2)


def test_cpp_structured_pruning_matmul_concat_declines_on_positive_non_last_axis():
    K, C1, C2, Out = 8, 6, 10, 4
    rng = np.random.default_rng(118)
    w1 = rng.standard_normal((K, C1)).astype(np.float32)
    w2 = rng.standard_normal((K, C2)).astype(np.float32)
    wout = rng.standard_normal((C1 + C2, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          m = Concat<axis = 0>(a, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(wout, "WOUT")],
    )
    model.graph.value_info.append(
        onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [None, C1])
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_cpp_structured_pruning_matmul_concat_declines_on_positive_axis_unknown_rank():
    K, C1, C2, Out = 8, 6, 10, 4
    rng = np.random.default_rng(119)
    w1 = rng.standard_normal((K, C1)).astype(np.float32)
    w2 = rng.standard_normal((K, C2)).astype(np.float32)
    wout = rng.standard_normal((C1 + C2, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          m = Concat<axis = 1>(a, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_cpp_structured_pruning_matches_python_reference_output_with_concat_chain():
    K, C1, C2, Out = 8, 6, 10, 4
    rng = np.random.default_rng(123)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(a, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[
            _f32(rng.standard_normal((K, C1)), "W1"),
            _f32(rng.standard_normal((K, C2)), "W2"),
            _f32(rng.standard_normal((C1 + C2, Out)), "WOUT"),
        ],
    )
    pruned_py = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    rng_x = np.random.default_rng(124)
    x = rng_x.standard_normal((6, K)).astype(np.float32)
    (y_py,) = _run(pruned_py, {"X": x})
    (y_cpp,) = _run(pruned_cpp, {"X": x})
    np.testing.assert_allclose(y_py, y_cpp, rtol=1e-5, atol=1e-5)


# --- Cross-check against the pure-Python reference --------------------------


def test_cpp_structured_pruning_matches_python_reference_output():
    K, H, Out = 8, 32, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=True, seed=9)
    pruned_py = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    rng = np.random.default_rng(10)
    x = rng.standard_normal((6, K)).astype(np.float32)
    (y_py,) = _run(pruned_py, {"X": x})
    (y_cpp,) = _run(pruned_cpp, {"X": x})
    np.testing.assert_allclose(y_py, y_cpp, rtol=1e-5, atol=1e-5)


# --- Subgraph recursion (If/Loop) --------------------------------------------
#
# Covers `structured_pruning_entry.cpp`'s own `IterSubgraphs` and the
# `ApplyStructuredPruning` loop built on it -- a straight C++ port of
# `onnxsim/pruning.py`'s own `_iter_subgraphs`/`apply_structured_pruning`
# subgraph-recursion round (see that module's "Subgraph recursion" section
# comment, and `structured_pruning_entry.cpp`'s own copy of it directly
# above `IterSubgraphs`'s definition, for the full design rationale). Model
# shapes below mirror `tests/test_pruning.py`'s own
# `_if_wrapped_mlp_model`/`test_structured_pruning_prunes_top_level_and_
# both_if_branches` fixture exactly, so a diff against those tests is the
# fastest way to see this is deliberately the same scenario, just driven
# through `apply_structured_pruning_cpp` instead of the pure-Python
# reference.
#
# `onnx.parser.parse_model`'s text format has no way to spell a graph-typed
# node attribute (an `If`'s `then_branch`/`else_branch`, a `Loop`'s
# `body`), so every model below uses `onnx.helper.make_node`/`make_graph`
# directly instead, per this repo's own CLAUDE.md guidance for exactly this
# case.


def _mlp_branch_nodes(K, H, Out, prefix, seed):
    # A minimal MatMul(Gemm)->Relu->MatMul chain, exactly `_mlp_model`'s own
    # shape, but returning bare nodes/initializers (not a whole model) so it
    # can be dropped into a subgraph's own `node`/`initializer` lists.
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    b1 = rng.standard_normal((H,)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    nodes = [
        onnx.helper.make_node(
            "Gemm", ["Xb", f"{prefix}W1", f"{prefix}B1"], [f"{prefix}h"]
        ),
        onnx.helper.make_node("Relu", [f"{prefix}h"], [f"{prefix}a"]),
        onnx.helper.make_node("MatMul", [f"{prefix}a", f"{prefix}W2"], ["Yb"]),
    ]
    inits = [
        _f32(w1, f"{prefix}W1"),
        _f32(b1, f"{prefix}B1"),
        _f32(w2, f"{prefix}W2"),
    ]
    return nodes, inits, dict(w1=w1, b1=b1, w2=w2)


def _if_wrapped_mlp_model(K0=8, H0=16, Out0=4, K1=6, H1=12, OutB=3):
    """A top-level MatMul(Gemm)->Relu->MatMul chain (`W1t`/`W2t`) PLUS an
    `If` node whose `then_branch`/`else_branch` each carry their OWN
    independent, identically-shaped MLP chain (`then_*`/`else_*`), with
    their own weights living only in that branch's own `initializer` list.
    `Xb` (the branch chains' shared activation input) is an ordinary
    top-level graph input, read by both branches purely via implicit
    capture (an `If` branch subgraph takes no formal inputs of its own).
    `cond` selects which branch actually executes at run time.
    """
    rng = np.random.default_rng(0)
    w1t = rng.standard_normal((K0, H0)).astype(np.float32)
    b1t = rng.standard_normal((H0,)).astype(np.float32)
    w2t = rng.standard_normal((H0, Out0)).astype(np.float32)
    top_nodes = [
        onnx.helper.make_node("Gemm", ["X0", "W1t", "B1t"], ["ht"]),
        onnx.helper.make_node("Relu", ["ht"], ["at"]),
        onnx.helper.make_node("MatMul", ["at", "W2t"], ["Y0"]),
    ]
    top_inits = [_f32(w1t, "W1t"), _f32(b1t, "B1t"), _f32(w2t, "W2t")]

    then_nodes, then_inits, then_cfg = _mlp_branch_nodes(K1, H1, OutB, "then_", seed=1)
    else_nodes, else_inits, else_cfg = _mlp_branch_nodes(K1, H1, OutB, "else_", seed=2)

    out_vi = onnx.helper.make_tensor_value_info(
        "Yb", onnx.TensorProto.FLOAT, ["batch", OutB]
    )
    then_graph = onnx.helper.make_graph(
        then_nodes, "then_graph", [], [out_vi], initializer=then_inits
    )
    else_graph = onnx.helper.make_graph(
        else_nodes, "else_graph", [], [out_vi], initializer=else_inits
    )
    if_node = onnx.helper.make_node(
        "If", ["cond"], ["Y1"], then_branch=then_graph, else_branch=else_graph
    )

    x0 = onnx.helper.make_tensor_value_info("X0", onnx.TensorProto.FLOAT, ["batch", K0])
    xb = onnx.helper.make_tensor_value_info("Xb", onnx.TensorProto.FLOAT, ["batch", K1])
    cond = onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
    y0 = onnx.helper.make_tensor_value_info(
        "Y0", onnx.TensorProto.FLOAT, ["batch", Out0]
    )
    y1 = onnx.helper.make_tensor_value_info(
        "Y1", onnx.TensorProto.FLOAT, ["batch", OutB]
    )

    graph = onnx.helper.make_graph(
        [*top_nodes, if_node],
        "g",
        [x0, xb, cond],
        [y0, y1],
        initializer=top_inits,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model, dict(w1t=w1t, b1t=b1t, w2t=w2t, then=then_cfg, else_=else_cfg)


def _then_else_graphs(pruned_model):
    if_node = next(n for n in pruned_model.graph.node if n.op_type == "If")
    then_g = else_g = None
    for attr in if_node.attribute:
        if attr.name == "then_branch":
            then_g = attr.g
        elif attr.name == "else_branch":
            else_g = attr.g
    return then_g, else_g


def _loop_wrapped_mlp_model(K0=8, H0=16, Out0=4, K1=6, H1=12, OutB=3, M=3):
    """The `Loop`-body counterpart of `_if_wrapped_mlp_model` above --
    covers the other half of "If/Loop/Scan" this file's own "Subgraph
    recursion" comment (and `structured_pruning_entry.cpp`'s own copy of
    it) names. A top-level MatMul(Gemm)->Relu->MatMul chain (`W1t`/`W2t`)
    PLUS a `Loop` node whose `body` carries its OWN independent MLP chain
    (`loop_*`), with its own weights living only in the body's own
    `initializer` list. `Xb` is read every iteration purely via implicit
    capture from the top-level graph's own input (`Loop`'s body takes only
    `iter_num`/`cond_in` as formal inputs -- no loop-carried dependency);
    `Yb` is emitted as a `scan_output`, stacked across all `M` iterations
    into `Ys`.
    """
    rng = np.random.default_rng(0)
    w1t = rng.standard_normal((K0, H0)).astype(np.float32)
    b1t = rng.standard_normal((H0,)).astype(np.float32)
    w2t = rng.standard_normal((H0, Out0)).astype(np.float32)
    top_nodes = [
        onnx.helper.make_node("Gemm", ["X0", "W1t", "B1t"], ["ht"]),
        onnx.helper.make_node("Relu", ["ht"], ["at"]),
        onnx.helper.make_node("MatMul", ["at", "W2t"], ["Y0"]),
    ]
    top_inits = [_f32(w1t, "W1t"), _f32(b1t, "B1t"), _f32(w2t, "W2t")]

    body_nodes, body_inits, body_cfg = _mlp_branch_nodes(K1, H1, OutB, "loop_", seed=1)
    cond_pass_through = onnx.helper.make_node("Identity", ["cond_in"], ["cond_out"])
    iter_num_vi = onnx.helper.make_tensor_value_info(
        "iter_num", onnx.TensorProto.INT64, []
    )
    cond_in_vi = onnx.helper.make_tensor_value_info(
        "cond_in", onnx.TensorProto.BOOL, []
    )
    cond_out_vi = onnx.helper.make_tensor_value_info(
        "cond_out", onnx.TensorProto.BOOL, []
    )
    yb_vi = onnx.helper.make_tensor_value_info(
        "Yb", onnx.TensorProto.FLOAT, ["batch", OutB]
    )
    body_graph = onnx.helper.make_graph(
        [*body_nodes, cond_pass_through],
        "loop_body",
        [iter_num_vi, cond_in_vi],
        [cond_out_vi, yb_vi],
        initializer=body_inits,
    )
    loop_node = onnx.helper.make_node("Loop", ["M", "cond"], ["Ys"], body=body_graph)

    x0 = onnx.helper.make_tensor_value_info("X0", onnx.TensorProto.FLOAT, ["batch", K0])
    xb = onnx.helper.make_tensor_value_info("Xb", onnx.TensorProto.FLOAT, ["batch", K1])
    m = onnx.helper.make_tensor_value_info("M", onnx.TensorProto.INT64, [])
    cond = onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
    y0 = onnx.helper.make_tensor_value_info(
        "Y0", onnx.TensorProto.FLOAT, ["batch", Out0]
    )
    ys = onnx.helper.make_tensor_value_info(
        "Ys", onnx.TensorProto.FLOAT, [M, "batch", OutB]
    )

    graph = onnx.helper.make_graph(
        [*top_nodes, loop_node],
        "g",
        [x0, xb, m, cond],
        [y0, ys],
        initializer=top_inits,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model, dict(w1t=w1t, b1t=b1t, w2t=w2t, loop=body_cfg)


def test_cpp_structured_pruning_prunes_top_level_and_both_if_branches():
    # The core repro: apply_structured_pruning_cpp must match and prune the
    # chain inside BOTH `then_branch` and `else_branch` (each with its own
    # independent weights) -- not just the top-level chain -- verified both
    # by initializer shape and by driving real execution (through
    # InferenceSession) into EACH branch via `cond`, comparing against an
    # independently reconstructed "already pruned" numpy oracle for that
    # branch's own weights. Also proves independence both ways: the
    # top-level chain's own pruning is unaffected by what's inside the `If`
    # (it lands on exactly the same oracle
    # `test_cpp_structured_pruning_matches_python_reference_output` already
    # checks for a subgraph-free model), and each branch's own pruning is
    # unaffected by the top-level chain or its sibling branch.
    K0, H0, Out0 = 8, 16, 4
    K1, H1, OutB = 6, 12, 3
    model, cfg = _if_wrapped_mlp_model(K0=K0, H0=H0, Out0=Out0, K1=K1, H1=H1, OutB=OutB)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    top_inits = {t.name: t for t in pruned.graph.initializer}
    assert list(top_inits["W1t"].dims) == [K0, H0 // 2]
    assert list(top_inits["W2t"].dims) == [H0 // 2, Out0]

    then_g, else_g = _then_else_graphs(pruned)
    then_inits = {t.name: t for t in then_g.initializer}
    else_inits = {t.name: t for t in else_g.initializer}
    assert list(then_inits["then_W1"].dims) == [K1, H1 // 2]
    assert list(then_inits["then_W2"].dims) == [H1 // 2, OutB]
    assert list(else_inits["else_W1"].dims) == [K1, H1 // 2]
    assert list(else_inits["else_W2"].dims) == [H1 // 2, OutB]

    rng = np.random.default_rng(5)
    x0 = rng.standard_normal((3, K0)).astype(np.float32)
    xb = rng.standard_normal((3, K1)).astype(np.float32)

    y0_true, y1_then = _run(pruned, {"X0": x0, "Xb": xb, "cond": np.array(True)})
    y0_false, y1_else = _run(pruned, {"X0": x0, "Xb": xb, "cond": np.array(False)})

    def _oracle(branch_cfg, keep_count):
        w1, b1, w2 = branch_cfg["w1"], branch_cfg["b1"], branch_cfg["w2"]
        keep = _oracle_keep_indices(w1, keep_count)
        h = xb @ w1[:, keep] + b1[keep]
        a = np.maximum(h, 0)
        return a @ w2[keep, :]

    np.testing.assert_allclose(
        y1_then, _oracle(cfg["then"], H1 // 2), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        y1_else, _oracle(cfg["else_"], H1 // 2), rtol=1e-5, atol=1e-5
    )

    keep_top = _oracle_keep_indices(cfg["w1t"], H0 // 2)
    h0 = x0 @ cfg["w1t"][:, keep_top] + cfg["b1t"][keep_top]
    a0 = np.maximum(h0, 0)
    y0_oracle = a0 @ cfg["w2t"][keep_top, :]
    np.testing.assert_allclose(y0_true, y0_oracle, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y0_false, y0_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_prunes_top_level_and_loop_body():
    # Same repro as the `If`-branch test above, but for a `Loop` body --
    # both the top-level chain and the chain living entirely inside the
    # `body` subgraph must be pruned to the same H1 // 2 width, each using
    # its own independent (in this case, identical-modulo-independent-
    # oracle) importance ranking, with neither affecting the other.
    K0, H0, Out0 = 8, 16, 4
    K1, H1, OutB = 6, 12, 3
    M = 3
    model, cfg = _loop_wrapped_mlp_model(
        K0=K0, H0=H0, Out0=Out0, K1=K1, H1=H1, OutB=OutB, M=M
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    top_inits = {t.name: t for t in pruned.graph.initializer}
    assert list(top_inits["W1t"].dims) == [K0, H0 // 2]
    assert list(top_inits["W2t"].dims) == [H0 // 2, Out0]

    loop_node = next(n for n in pruned.graph.node if n.op_type == "Loop")
    body = next(a.g for a in loop_node.attribute if a.name == "body")
    body_inits = {t.name: t for t in body.initializer}
    assert list(body_inits["loop_W1"].dims) == [K1, H1 // 2]
    assert list(body_inits["loop_W2"].dims) == [H1 // 2, OutB]

    rng = np.random.default_rng(6)
    x0 = rng.standard_normal((3, K0)).astype(np.float32)
    xb = rng.standard_normal((3, K1)).astype(np.float32)
    y0, ys = _run(
        pruned,
        {
            "X0": x0,
            "Xb": xb,
            "M": np.array(M, dtype=np.int64),
            "cond": np.array(True),
        },
    )

    def _oracle(branch_cfg, keep_count):
        w1, b1, w2 = branch_cfg["w1"], branch_cfg["b1"], branch_cfg["w2"]
        keep = _oracle_keep_indices(w1, keep_count)
        h = xb @ w1[:, keep] + b1[keep]
        a = np.maximum(h, 0)
        return a @ w2[keep, :]

    yb_oracle = _oracle(cfg["loop"], H1 // 2)
    # Every iteration recomputes the exact same thing (no loop-carried
    # state, `Xb` fixed across iterations), so every one of the M stacked
    # scan-output slices must equal the same oracle.
    np.testing.assert_allclose(
        ys, np.broadcast_to(yb_oracle, ys.shape), rtol=1e-5, atol=1e-5
    )

    keep_top = _oracle_keep_indices(cfg["w1t"], H0 // 2)
    h0 = x0 @ cfg["w1t"][:, keep_top] + cfg["b1t"][keep_top]
    a0 = np.maximum(h0, 0)
    y0_oracle = a0 @ cfg["w2t"][keep_top, :]
    np.testing.assert_allclose(y0, y0_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matches_python_reference_output_with_if_subgraph():
    # Cross-check against onnxsim.apply_structured_pruning (the pure-Python
    # reference this C++ port mirrors) on a model where the only prunable
    # weight worth talking about lives inside the `If`'s own branches --
    # both `cond` values are driven through InferenceSession so both
    # branches' own subgraph-recursion behavior is exercised, not just
    # whichever one a single run happens to select.
    K0, H0, Out0 = 8, 16, 4
    K1, H1, OutB = 6, 12, 3
    model, _cfg = _if_wrapped_mlp_model(
        K0=K0, H0=H0, Out0=Out0, K1=K1, H1=H1, OutB=OutB
    )

    pruned_py = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    rng = np.random.default_rng(7)
    x0 = rng.standard_normal((3, K0)).astype(np.float32)
    xb = rng.standard_normal((3, K1)).astype(np.float32)
    for cond in (True, False):
        feeds = {"X0": x0, "Xb": xb, "cond": np.array(cond)}
        y0_py, y1_py = _run(pruned_py, feeds)
        y0_cpp, y1_cpp = _run(pruned_cpp, feeds)
        np.testing.assert_allclose(y0_py, y0_cpp, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(y1_py, y1_cpp, rtol=1e-5, atol=1e-5)


# --- Decomposed (not-yet-fused) LayerNorm pass-through, MatMul/Gemm chains --
#
# `LayerNormalization`'s own schema is only opset 17+, so an export at
# opset <= 16 has no fused op to lower `nn.LayerNorm` to at all and emits its
# canonical 9-node decomposition instead -- see `structured_pruning_entry.cpp`'s
# own "Decomposed (not-yet-fused) LayerNorm pass-through" section comment
# (mirroring `onnxsim/pruning.py`'s own `_match_decomposed_layer_norm_pass_
# through`) for the full shape. Mirrors `tests/test_pruning.py`'s own
# "apply_structured_pruning: *decomposed* (not-yet-fused) LayerNorm" section.


def _decomposed_layer_norm_model(w1, gamma, beta, w2, axis=-1, opset=16, pow_exp=2.0):
    K, C = w1.shape
    Out = w2.shape[1]
    initializer = [
        _f32(w1, "W1"),
        onnx.numpy_helper.from_array(np.array(pow_exp, dtype=np.float32), "Two"),
        onnx.numpy_helper.from_array(np.array(1e-5, dtype=np.float32), "Eps"),
        _f32(gamma, "Gamma"),
        _f32(beta, "Beta"),
        _f32(w2, "W2"),
    ]
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          up = MatMul(X, W1)
          mean = ReduceMean<axes=[{axis}]>(up)
          centered = Sub(up, mean)
          sq = Pow(centered, Two)
          var = ReduceMean<axes=[{axis}]>(sq)
          var_eps = Add(var, Eps)
          std = Sqrt(var_eps)
          normed = Div(centered, std)
          scaled = Mul(normed, Gamma)
          h = Add(scaled, Beta)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=initializer,
        opset=opset,
    )


def test_cpp_structured_pruning_decomposed_layer_norm_pass_through_shrinks_matched_layers():
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(6001)
    w1 = rng.standard_normal((K, C)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    w2 = rng.standard_normal((C, Out)).astype(np.float32)
    model = _decomposed_layer_norm_model(w1, gamma, beta, w2)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, C // 2]
    assert list(inits["Gamma"].dims) == [C // 2]
    assert list(inits["Beta"].dims) == [C // 2]
    assert list(inits["W2"].dims) == [C // 2, Out]
    # Every decomposition node itself is untouched (still the same 9 nodes,
    # same op types/wiring) -- only the two per-channel constants shrink.
    assert [n.op_type for n in pruned.graph.node] == [
        "MatMul",
        "ReduceMean",
        "Sub",
        "Pow",
        "ReduceMean",
        "Add",
        "Sqrt",
        "Div",
        "Mul",
        "Add",
        "MatMul",
    ]


def test_cpp_structured_pruning_decomposed_layer_norm_pass_through_matches_oracle_adversarially():
    # W1 engineered so the surviving `keep` set is deliberately not the
    # first C//2 channels, Gamma/Beta spanning three orders of magnitude,
    # strictly increasing by channel index -- a positional (rather than
    # index-set) slice of Gamma/Beta would misapply a wildly-wrong-magnitude
    # affine term to a kept channel, detectably wrong by orders of
    # magnitude. Mirrors test_pruning.py's own identically-named test.
    K, C, Out = 4, 8, 2
    rng = np.random.default_rng(6002)
    col_scale = np.linspace(0.1, 2.0, C).astype(np.float32)
    w1 = (rng.standard_normal((K, C)) * col_scale).astype(np.float32)
    gamma = (np.arange(1, C + 1, dtype=np.float64) * 0.001).astype(np.float32)
    beta = (np.arange(1, C + 1, dtype=np.float64) * 1000.0).astype(np.float32)
    w2 = rng.standard_normal((C, Out)).astype(np.float32)
    model = _decomposed_layer_norm_model(w1, gamma, beta, w2)
    onnx.checker.check_model(model)

    keep_count = C // 2
    keep = _oracle_keep_indices(w1, keep_count)
    assert not np.array_equal(keep, np.arange(keep_count))  # confirm adversarial

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    def _layer_norm(v, g, b, eps=1e-5):
        mean = v.mean(axis=-1, keepdims=True)
        var = v.var(axis=-1, keepdims=True)
        return (v - mean) / np.sqrt(var + eps) * g + b

    h_correct = _layer_norm(x @ w1[:, keep], gamma[keep], beta[keep])
    y_correct = h_correct @ w2[keep, :]
    np.testing.assert_allclose(y, y_correct, rtol=1e-4, atol=1e-4)

    wrong_gamma = gamma[:keep_count]
    wrong_beta = beta[:keep_count]
    h_wrong = _layer_norm(x @ w1[:, keep], wrong_gamma, wrong_beta)
    y_wrong = h_wrong @ w2[keep, :]
    assert np.max(np.abs(y - y_wrong)) > 1e-2 * max(1.0, np.max(np.abs(y_correct)))


def test_cpp_structured_pruning_decomposed_layer_norm_pow_exponent_not_two_is_declined():
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(6003)
    w1 = rng.standard_normal((K, C)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    w2 = rng.standard_normal((C, Out)).astype(np.float32)
    model = _decomposed_layer_norm_model(w1, gamma, beta, w2, pow_exp=3.0)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_decomposed_layer_norm_axis_not_last_is_declined():
    # axis=-2 on a rank-2 [batch, C] tensor normalizes over the batch axis,
    # not the trailing channel axis being pruned.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(6004)
    w1 = rng.standard_normal((K, C)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    w2 = rng.standard_normal((C, Out)).astype(np.float32)
    model = _decomposed_layer_norm_model(w1, gamma, beta, w2, axis=-2)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_decomposed_layer_norm_extra_root_consumer_is_declined():
    # `up` (the decomposition's own root tensor) must be read by *exactly*
    # the ReduceMean/Sub pair this shape expects -- a third reader (here, an
    # extra Identity tapping the same tensor for a second graph output)
    # means the whole chain is declined outright, never partially matched.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(6005)
    w1 = rng.standard_normal((K, C)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    w2 = rng.standard_normal((C, Out)).astype(np.float32)
    initializer = [
        _f32(w1, "W1"),
        onnx.numpy_helper.from_array(np.array(2.0, dtype=np.float32), "Two"),
        onnx.numpy_helper.from_array(np.array(1e-5, dtype=np.float32), "Eps"),
        _f32(gamma, "Gamma"),
        _f32(beta, "Beta"),
        _f32(w2, "W2"),
    ]
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch,{C}] Extra)
        {{
          up = MatMul(X, W1)
          Extra = Identity(up)
          mean = ReduceMean<axes=[-1]>(up)
          centered = Sub(up, mean)
          sq = Pow(centered, Two)
          var = ReduceMean<axes=[-1]>(sq)
          var_eps = Add(var, Eps)
          std = Sqrt(var_eps)
          normed = Div(centered, std)
          scaled = Mul(normed, Gamma)
          h = Add(scaled, Beta)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=initializer,
        opset=16,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_decomposed_layer_norm_extra_centered_consumer_is_declined():
    # `centered` (Sub's own output) must likewise be read by *exactly* the
    # Pow/Div pair this shape expects.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(6006)
    w1 = rng.standard_normal((K, C)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    w2 = rng.standard_normal((C, Out)).astype(np.float32)
    initializer = [
        _f32(w1, "W1"),
        onnx.numpy_helper.from_array(np.array(2.0, dtype=np.float32), "Two"),
        onnx.numpy_helper.from_array(np.array(1e-5, dtype=np.float32), "Eps"),
        _f32(gamma, "Gamma"),
        _f32(beta, "Beta"),
        _f32(w2, "W2"),
    ]
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch,{C}] Extra)
        {{
          up = MatMul(X, W1)
          mean = ReduceMean<axes=[-1]>(up)
          centered = Sub(up, mean)
          Extra = Identity(centered)
          sq = Pow(centered, Two)
          var = ReduceMean<axes=[-1]>(sq)
          var_eps = Add(var, Eps)
          std = Sqrt(var_eps)
          normed = Div(centered, std)
          scaled = Mul(normed, Gamma)
          h = Add(scaled, Beta)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=initializer,
        opset=16,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_decomposed_layer_norm_after_bias_add_matches_oracle():
    # The decomposition's own root tensor (`up`, read twice) sitting right
    # after an ordinary preceding hop (a per-channel bias `Add`, itself
    # producing a tensor read twice) -- confirms the fix composes with an
    # already-established chain_ops hop (WalkToConsumer's own `out2`
    # fallback), not just a bare producer output feeding the decomposition
    # directly.
    K, C, Out = 6, 12, 3
    rng = np.random.default_rng(6007)
    w1 = rng.standard_normal((K, C)).astype(np.float32)
    b1 = rng.standard_normal((C,)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    w2 = rng.standard_normal((C, Out)).astype(np.float32)
    initializer = [
        _f32(w1, "W1"),
        _f32(b1, "B1"),
        onnx.numpy_helper.from_array(np.array(2.0, dtype=np.float32), "Two"),
        onnx.numpy_helper.from_array(np.array(1e-5, dtype=np.float32), "Eps"),
        _f32(gamma, "Gamma"),
        _f32(beta, "Beta"),
        _f32(w2, "W2"),
    ]
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          mm = MatMul(X, W1)
          up = Add(mm, B1)
          mean = ReduceMean<axes=[-1]>(up)
          centered = Sub(up, mean)
          sq = Pow(centered, Two)
          var = ReduceMean<axes=[-1]>(sq)
          var_eps = Add(var, Eps)
          std = Sqrt(var_eps)
          normed = Div(centered, std)
          scaled = Mul(normed, Gamma)
          h = Add(scaled, Beta)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=initializer,
        opset=16,
    )
    onnx.checker.check_model(model)

    keep_count = C // 2
    keep = _oracle_keep_indices(w1, keep_count)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Gamma"].dims) == [keep_count]

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    def _layer_norm(v, g, b, eps=1e-5):
        mean = v.mean(axis=-1, keepdims=True)
        var = v.var(axis=-1, keepdims=True)
        return (v - mean) / np.sqrt(var + eps) * g + b

    up_ref = x @ w1[:, keep] + b1[keep]
    h_correct = _layer_norm(up_ref, gamma[keep], beta[keep])
    y_correct = h_correct @ w2[keep, :]
    np.testing.assert_allclose(y, y_correct, rtol=1e-4, atol=1e-4)


def test_cpp_structured_pruning_decomposed_layer_norm_matches_python_reference_output():
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(6008)
    w1 = rng.standard_normal((K, C)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    w2 = rng.standard_normal((C, Out)).astype(np.float32)
    model = _decomposed_layer_norm_model(w1, gamma, beta, w2)

    pruned_py = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y_py,) = _run(pruned_py, {"X": x})
    (y_cpp,) = _run(pruned_cpp, {"X": x})
    np.testing.assert_allclose(y_py, y_cpp, rtol=1e-5, atol=1e-5)


# --- Self-gated activation decomposition (SiLU/erf-GELU, unfused) ----------
#
# A raw export of `nn.SiLU()`/`nn.GELU()` -- below the fused op's own opset
# floor -- emits the literal decomposition instead of the fused node: the
# producer's own output tensor feeds *two* consumers at once (a gate branch,
# and the `Mul` that combines it with that branch's own output). Mirrors
# `structured_pruning_entry.cpp`'s own "Self-gated activation decomposition"
# section comment (and `onnxsim/pruning.py`'s own identically-named one) --
# see there for the exact two node shapes and the design rationale. Mirrors
# `tests/test_pruning.py`'s own "Self-gated activation decomposition" test
# sections (Conv-chain and MatMul/Gemm-chain).


def _conv_silu_decomposed_pair_model(w1, w2, b1=None, spatial=10):
    Cin, C2 = w1.shape[1], w2.shape[0]
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    if b1 is not None:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1)"
    out_spatial = spatial - 4
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{out_spatial},{out_spatial}] Y)
        {{
          {conv1}
          s = Sigmoid(h)
          a = Mul(h, s)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=initializer,
    )


def _conv_erf_gelu_decomposed_pair_model(w1, w2, b1=None, spatial=10):
    Cin, C2 = w1.shape[1], w2.shape[0]
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    if b1 is not None:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1)"
    out_spatial = spatial - 4
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{out_spatial},{out_spatial}] Y)
        <float Sqrt2 = {{1.4142135}}, float One = {{1.0}}, float Half = {{0.5}}>
        {{
          {conv1}
          d = Div(h, Sqrt2)
          e = Erf(d)
          ao = Add(e, One)
          m = Mul(h, ao)
          a = Mul(m, Half)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=initializer,
    )


def test_cpp_structured_pruning_conv_chain_silu_decomposed_matches_oracle_exactly():
    # Before this hop existed, this exact chain came out completely
    # unpruned end to end -- `h`'s own two consumers broke the forward
    # walk's single-consumer-per-tensor assumption outright.
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(6101)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_silu_decomposed_pair_model(w1, w2, b1=b1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [C1 // 2, Cin, 3, 3]
    assert list(inits["W2"].dims) == [C2, C1 // 2, 3, 3]
    pruned_node_types = sorted(n.op_type for n in pruned.graph.node)
    assert pruned_node_types == sorted(n.op_type for n in model.graph.node)

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _conv_silu_decomposed_pair_model(w1[keep], w2[:, keep], b1=b1[keep])

    rng_x = np.random.default_rng(6102)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    assert np.isfinite(y).all()
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_conv_chain_erf_gelu_decomposed_matches_oracle_exactly():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(6103)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_erf_gelu_decomposed_pair_model(w1, w2, b1=b1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [C1 // 2, Cin, 3, 3]
    assert list(inits["W2"].dims) == [C2, C1 // 2, 3, 3]
    pruned_node_types = sorted(n.op_type for n in pruned.graph.node)
    assert pruned_node_types == sorted(n.op_type for n in model.graph.node)

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _conv_erf_gelu_decomposed_pair_model(w1[keep], w2[:, keep], b1=b1[keep])

    rng_x = np.random.default_rng(6104)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    assert np.isfinite(y).all()
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_conv_chain_fused_gelu_still_matches_oracle_exactly():
    # Control case: the *fused* `Gelu` node (already unary/
    # UnaryPassThroughOps()) must keep pruning correctly, unaffected by the
    # new decomposition hop.
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(6105)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2, b1=b1, activation="Gelu")

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [C1 // 2, Cin, 3, 3]
    assert list(inits["W2"].dims) == [C2, C1 // 2, 3, 3]

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _conv_pair_model(w1[keep], w2[:, keep], b1=b1[keep], activation="Gelu")

    rng_x = np.random.default_rng(6106)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    assert np.isfinite(y).all()
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_conv_chain_erf_gelu_decomposed_declines_on_branch_fanout():
    # `d` (the gate branch's own `Div` output) is *also* read by a second,
    # spurious consumer (`Extra`, tapped as a second graph output) -- the
    # gate branch is no longer a strict single-consumer chain, so
    # WalkGateBranch must decline the whole diamond outright, leaving both
    # weights completely untouched, never guessed at or partially cut.
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(6107)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y, float[N,{C1},10,10] Extra)
        <float Sqrt2 = {{1.4142135}}, float One = {{1.0}}, float Half = {{0.5}}>
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          d = Div(h, Sqrt2)
          Extra = Identity(d)
          e = Erf(d)
          ao = Add(e, One)
          m = Mul(h, ao)
          a = Mul(m, Half)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_conv_residual_silu_decomposed_pass_through_matches_oracle():
    # A decomposed SiLU diamond crossed by the *backward* walk
    # (WalkConvProducerBackward/MatchSelfGatedActivationBackward), not just
    # the forward one -- exercises the residual-chain insertion point and
    # the two-in-group-consumers `edges`/`accounted` bookkeeping the diamond's
    # own origin tensor needs.
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(6108)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)

    def _mk(w_f, w_s, w_out):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{Cout},6,6] Y)
            {{
              f0 = Conv<kernel_shape=[3,3]>(X, WF)
              fs = Sigmoid(f0)
              f = Mul(f0, fs)
              s = Conv<kernel_shape=[3,3]>(X, WS)
              addr = Add(f, s)
              r = Relu(addr)
              Y = Conv<kernel_shape=[3,3]>(r, WOUT)
            }}
            """,
            initializer=[
                _f32(w_f, "WF"),
                _f32(w_s, "WS"),
                _f32(w_out, "WOUT"),
            ],
        )

    model = _mk(w_f, w_s, w_out)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _mk(w_f[keep], w_s[keep], w_out[:, keep])

    rng_x = np.random.default_rng(6109)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- Self-gated activation decomposition, MatMul/Gemm chains ---------------


def _mlp_silu_decomposed_model(K, H, Out, bias=True, seed=0):
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
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          {gemm1}
          s = Sigmoid(h)
          a = Mul(h, s)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=initializer,
    )
    return model, w1, w2, (b1 if bias else None)


def _mlp_erf_gelu_decomposed_model(K, H, Out, bias=True, seed=0):
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
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        <float Sqrt2 = {{1.4142135}}, float One = {{1.0}}, float Half = {{0.5}}>
        {{
          {gemm1}
          d = Div(h, Sqrt2)
          e = Erf(d)
          ao = Add(e, One)
          m = Mul(h, ao)
          a = Mul(m, Half)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=initializer,
    )
    return model, w1, w2, (b1 if bias else None)


def test_cpp_structured_pruning_matmul_silu_decomposed_matches_oracle_exactly():
    K, H, Out = 8, 32, 4
    model, w1, w2, b1 = _mlp_silu_decomposed_model(K, H, Out)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    keep_count = H // 2
    assert list(inits["W1"].dims)[1] == keep_count
    assert list(inits["W2"].dims)[0] == keep_count
    pruned_node_types = sorted(n.op_type for n in pruned.graph.node)
    assert pruned_node_types == sorted(n.op_type for n in model.graph.node)

    keep = _oracle_keep_indices(w1, keep_count)
    oracle = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = Gemm(X, W1, B1)
          s = Sigmoid(h)
          a = Mul(h, s)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[
            _f32(w1[:, keep], "W1"),
            _f32(w2[keep], "W2"),
            _f32(b1[keep], "B1"),
        ],
    )

    rng_x = np.random.default_rng(6111)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_erf_gelu_decomposed_matches_oracle_exactly():
    K, H, Out = 8, 32, 4
    model, w1, w2, b1 = _mlp_erf_gelu_decomposed_model(K, H, Out)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    keep_count = H // 2
    assert list(inits["W1"].dims)[1] == keep_count
    assert list(inits["W2"].dims)[0] == keep_count
    pruned_node_types = sorted(n.op_type for n in pruned.graph.node)
    assert pruned_node_types == sorted(n.op_type for n in model.graph.node)

    keep = _oracle_keep_indices(w1, keep_count)
    oracle = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        <float Sqrt2 = {{1.4142135}}, float One = {{1.0}}, float Half = {{0.5}}>
        {{
          h = Gemm(X, W1, B1)
          d = Div(h, Sqrt2)
          e = Erf(d)
          ao = Add(e, One)
          m = Mul(h, ao)
          a = Mul(m, Half)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[
            _f32(w1[:, keep], "W1"),
            _f32(w2[keep], "W2"),
            _f32(b1[keep], "B1"),
        ],
    )

    rng_x = np.random.default_rng(6112)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_fused_gelu_still_matches_oracle_exactly():
    # Control case: the *fused* `Gelu` node (already UnaryPassThroughOps())
    # must keep pruning correctly, unaffected by the new decomposition hop.
    K, H, Out = 8, 32, 4
    rng = np.random.default_rng(6113)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    b1 = rng.standard_normal((H,)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = Gemm(X, W1, B1)
          a = Gelu(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(b1, "B1"), _f32(w2, "W2")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    keep_count = H // 2
    assert list(inits["W1"].dims)[1] == keep_count
    assert list(inits["W2"].dims)[0] == keep_count

    keep = _oracle_keep_indices(w1, keep_count)
    oracle = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = Gemm(X, W1, B1)
          a = Gelu(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[
            _f32(w1[:, keep], "W1"),
            _f32(b1[keep], "B1"),
            _f32(w2[keep], "W2"),
        ],
    )

    rng_x = np.random.default_rng(6114)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_erf_gelu_decomposed_declines_on_nonconstant_scalar():
    # `Sqrt2` (the gate branch's own `Div` divisor) fed by a non-constant
    # node (an `Identity` of a graph input) rather than a genuine constant --
    # this pass cannot confirm it's a channel-agnostic scalar without
    # evaluating the graph, so the whole diamond must decline outright.
    K, H, Out = 8, 32, 4
    rng = np.random.default_rng(6115)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X, float Sqrt2In) => (float[batch,{Out}] Y)
        <float One = {{1.0}}, float Half = {{0.5}}>
        {{
          h = MatMul(X, W1)
          Sqrt2 = Identity(Sqrt2In)
          d = Div(h, Sqrt2)
          e = Erf(d)
          ao = Add(e, One)
          m = Mul(h, ao)
          a = Mul(m, Half)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_matmul_residual_silu_decomposed_pass_through_matches_oracle():
    # The MatMul/Gemm-chain analogue of the Conv residual test above --
    # exercises WalkMatmulProducerBackward's own new Mul-ahead-of-
    # _BINARY_CHANNEL_OPS dispatch.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(6116)
    w_f = rng.standard_normal((K, C)).astype(np.float32)
    w_s = rng.standard_normal((K, C)).astype(np.float32)
    w_out = rng.standard_normal((C, Out)).astype(np.float32)

    def _mk(w_f, w_s, w_out):
        return _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{Out}] Y)
            {{
              f0 = MatMul(X, WF)
              fs = Sigmoid(f0)
              f = Mul(f0, fs)
              s = MatMul(X, WS)
              addr = Add(f, s)
              r = Relu(addr)
              Y = MatMul(r, WOUT)
            }}
            """,
            initializer=[
                _f32(w_f, "WF"),
                _f32(w_s, "WS"),
                _f32(w_out, "WOUT"),
            ],
        )

    model = _mk(w_f, w_s, w_out)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.T.astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.T.astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _mk(w_f[:, keep], w_s[:, keep], w_out[keep, :])

    rng_x = np.random.default_rng(6117)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_silu_decomposed_matches_python_reference_output():
    K, H, Out = 8, 32, 4
    model, _w1, _w2, _b1 = _mlp_silu_decomposed_model(K, H, Out)

    pruned_py = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    rng = np.random.default_rng(6118)
    x = rng.standard_normal((6, K)).astype(np.float32)
    (y_py,) = _run(pruned_py, {"X": x})
    (y_cpp,) = _run(pruned_cpp, {"X": x})
    np.testing.assert_allclose(y_py, y_cpp, rtol=1e-5, atol=1e-5)


# --- MatMulNBits (com.microsoft, block-quantized weight) structured -------
# --- pruning ------------------------------------------------------------
#
# Tests for the plain and gated MatMulNBits chain families
# (onnxsim/structured_pruning_entry.cpp's own "MatMulNBits" section),
# mirroring tests/test_pruning.py's own MatMulNBits coverage (see that
# file's own `_nbits_*` helpers, independently re-implemented here rather
# than imported -- this file's own established "no cross-import between
# test files" precedent). Models are built via onnx.parser (per CLAUDE.md's
# convention); the packed uint8 `B`/`scales`/`zero_points` initializers are
# attached programmatically via onnx.numpy_helper.from_array afterward
# (CLAUDE.md's documented exception: these must be byte-exact, not spelled
# out as text literals -- the parser encodes tensor literals as
# `float_data`/`int32_data`, not `raw_data`).


def _nbits_model(body, initializer=(), opset=21):
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


def _nbits_quantize_block(w, block_size, bits=4):
    """Independent reference block quantizer for a ``[N, K]`` float weight
    matrix -- standard asymmetric affine quantization per ``(n, k_block)``,
    zero-inclusive range. Returns ``(qcodes uint8 [N, K], scales float32
    [N, k_blocks], zero_points uint8 [N, k_blocks] UNPACKED, k_blocks)``.
    """
    n, k = w.shape
    assert k % block_size == 0
    k_blocks = k // block_size
    qmax = (1 << bits) - 1
    scales = np.zeros((n, k_blocks), dtype=np.float32)
    zero_points = np.zeros((n, k_blocks), dtype=np.uint8)
    qcodes = np.zeros((n, k), dtype=np.uint8)
    for row in range(n):
        for kb in range(k_blocks):
            k0, k1 = kb * block_size, (kb + 1) * block_size
            block = w[row, k0:k1]
            vmin = min(float(block.min()), 0.0)
            vmax = max(float(block.max()), 0.0)
            scale = (vmax - vmin) / qmax if vmax > vmin else 1.0
            zp = int(np.clip(round(-vmin / scale), 0, qmax))
            scales[row, kb] = scale
            zero_points[row, kb] = zp
            qcodes[row, k0:k1] = (
                np.round(block / scale + zp).clip(0, qmax).astype(np.uint8)
            )
    return qcodes, scales, zero_points, k_blocks


def _nbits_pack_nibbles(vals):
    """Independent reference nibble packer: last axis (uint8 in [0, 15]),
    2-per-byte, LOW nibble first -- the schema's own documented layout.
    """
    count = vals.shape[-1]
    nbytes = (count + 1) // 2
    out = np.zeros(vals.shape[:-1] + (nbytes,), dtype=np.uint8)
    for j in range(nbytes):
        lo = vals[..., 2 * j]
        hi = vals[..., 2 * j + 1] if 2 * j + 1 < count else np.zeros_like(lo)
        out[..., j] = (lo & 0xF) | ((hi & 0xF) << 4)
    return out


def _nbits_pack_codes(vals, bits):
    """``bits=8`` is a plain dtype cast (one full byte per code, no packing
    at all); ``bits=4`` delegates to :func:`_nbits_pack_nibbles`.
    """
    if bits == 8:
        return vals.astype(np.uint8)
    assert bits == 4, bits
    return _nbits_pack_nibbles(vals)


def _nbits_pack_b(qcodes, n, k_blocks, block_size, bits=4):
    blob_size = block_size * bits // 8
    b = np.zeros((n, k_blocks, blob_size), dtype=np.uint8)
    for kb in range(k_blocks):
        k0 = kb * block_size
        b[:, kb, :] = _nbits_pack_codes(qcodes[:, k0 : k0 + block_size], bits)
    return b


def _nbits_dequant(qcodes, scales, zero_points, block_size, bits=4):
    """Independent reference dequantizer -- ``zero_points=None`` uses the
    schema's own documented default, ``2 ** (bits - 1)``.
    """
    n, k = qcodes.shape
    k_blocks = k // block_size
    out = np.zeros((n, k), dtype=np.float64)
    default_zp = float(1 << (bits - 1))
    for row in range(n):
        for kb in range(k_blocks):
            k0, k1 = kb * block_size, (kb + 1) * block_size
            zp = float(zero_points[row, kb]) if zero_points is not None else default_zp
            out[row, k0:k1] = (qcodes[row, k0:k1].astype(np.float64) - zp) * scales[
                row, kb
            ]
    return out


def _nbits_weight_initializers(w, block_size, prefix, bits=4, zp_mode="packed"):
    """Quantizes ``w`` (``[N, K]``) and returns ``(initializer_list,
    info_dict)`` -- `info_dict` carries the independently-computed
    quantization artifacts (`qcodes`, `scales`, `zp` UNPACKED, `k_blocks`)
    needed to hand-build an oracle, plus the actual initializer names used
    (`b_name`/`scales_name`/`zp_name` -- the last ``None`` for
    ``zp_mode="absent"``). ``zp_mode``: ``"packed"`` (nibble/byte-packed
    uint8, the schema's own PRIMARY documented `zero_points` encoding),
    ``"unpacked"`` (float32, same dtype as `scales`, the schema's OTHER
    documented encoding), or ``"absent"`` (no `zero_points` input at all --
    schema default ``2 ** (bits - 1)`` applies).
    """
    qcodes, scales, zp, k_blocks = _nbits_quantize_block(w, block_size, bits)
    b = _nbits_pack_b(qcodes, w.shape[0], k_blocks, block_size, bits)
    inits = [
        onnx.numpy_helper.from_array(b, name=f"{prefix}_B"),
        onnx.numpy_helper.from_array(scales, name=f"{prefix}_scales"),
    ]
    zp_name = None
    if zp_mode == "packed":
        inits.append(
            onnx.numpy_helper.from_array(
                _nbits_pack_codes(zp, bits), name=f"{prefix}_zp"
            )
        )
        zp_name = f"{prefix}_zp"
    elif zp_mode == "unpacked":
        inits.append(
            onnx.numpy_helper.from_array(zp.astype(np.float32), name=f"{prefix}_zp")
        )
        zp_name = f"{prefix}_zp"
    else:
        assert zp_mode == "absent", zp_mode
    return inits, dict(
        qcodes=qcodes,
        scales=scales,
        zp=zp,
        k_blocks=k_blocks,
        b_name=f"{prefix}_B",
        scales_name=f"{prefix}_scales",
        zp_name=zp_name,
    )


def test_cpp_structured_pruning_matmul_nbits_plain_chain_matches_oracle():
    # bits=4, zero_points PACKED (nibble), both nodes biased. N1=32,
    # block_size=16 -> the consumer's own K2=N1=32 axis is exactly 2 blocks.
    # W1's rows are engineered so the pass's own L2-norm importance ranking
    # keeps EXACTLY rows 0-15 (block 0) -- a real, non-trivial keep-set that
    # also happens to land on a whole-block boundary.
    N1, K1, N2, block_size = 32, 64, 8, 16
    rng = np.random.default_rng(9001)
    w1 = rng.standard_normal((N1, K1)).astype(np.float32) * 0.2
    w1[:16] *= 6.0  # rows 0-15 large (kept), 16-31 small (dropped)
    w2 = rng.standard_normal((N2, N1)).astype(np.float32) * 0.2
    bias1 = (rng.standard_normal(N1) * 0.05).astype(np.float32)
    bias2 = (rng.standard_normal(N2) * 0.05).astype(np.float32)

    inits1, info1 = _nbits_weight_initializers(w1, block_size, "w1")
    inits2, info2 = _nbits_weight_initializers(w2, block_size, "w2")

    model = _nbits_model(
        f"""
        g (float[batch,{K1}] A) => (float[batch,{N2}] Y)
        {{
          h1 = com.microsoft.MatMulNBits<K={K1},N={N1},bits=4,block_size={block_size}>(A, {info1["b_name"]}, {info1["scales_name"]}, {info1["zp_name"]}, , bias1)
          h1a = Relu(h1)
          Y = com.microsoft.MatMulNBits<K={N1},N={N2},bits=4,block_size={block_size}>(h1a, {info2["b_name"]}, {info2["scales_name"]}, {info2["zp_name"]}, , bias2)
        }}
        """,
        initializer=[*inits1, *inits2, _f32(bias1, "bias1"), _f32(bias2, "bias2")],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = np.arange(16)
    keep_blocks = np.array([0])
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits[info1["b_name"]].dims) == [
        16,
        info1["k_blocks"],
        block_size * 4 // 8,
    ]
    assert list(inits[info1["scales_name"]].dims) == [16, info1["k_blocks"]]
    assert list(inits["bias1"].dims) == [16]
    assert list(inits[info2["b_name"]].dims) == [N2, 1, block_size * 4 // 8]
    assert list(inits[info2["scales_name"]].dims) == [N2, 1]

    # Exact (byte-level, not merely close) "slice, don't recompute" checks --
    # the pruned graph's own tensors must equal a HAND-SLICE of the ORIGINAL
    # quantized codes, never a re-quantization of the sliced float weight.
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits[info1["b_name"]]),
        _nbits_pack_b(info1["qcodes"][keep], 16, info1["k_blocks"], block_size),
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits[info1["zp_name"]]),
        _nbits_pack_codes(info1["zp"][keep], 4),
    )
    np.testing.assert_allclose(onnx.numpy_helper.to_array(inits["bias1"]), bias1[keep])
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits[info2["b_name"]]),
        _nbits_pack_b(info2["qcodes"][:, keep], N2, 1, block_size),
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits[info2["zp_name"]]),
        _nbits_pack_codes(info2["zp"][:, keep_blocks], 4),
    )

    rng2 = np.random.default_rng(9002)
    x = rng2.standard_normal((3, K1)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"A": x})

    w1_dequant = _nbits_dequant(
        info1["qcodes"], info1["scales"], info1["zp"], block_size
    )
    w2_dequant = _nbits_dequant(
        info2["qcodes"], info2["scales"], info2["zp"], block_size
    )
    h1 = np.maximum(x.astype(np.float64) @ w1_dequant[keep].T + bias1[keep], 0.0)
    y_oracle = h1 @ w2_dequant[:, keep].T + bias2
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-3, atol=1e-3)


def test_cpp_structured_pruning_matmul_nbits_bits8_chain_matches_oracle():
    # bits=8: one full byte per code, no nibble packing at all -- a
    # different, independently-verified code path than bits=4. First
    # node's own `zero_points` is ABSENT (schema default 2**(bits-1)
    # applies); second node's is PACKED (a plain byte-per-code identity
    # "pack" for bits=8, still routed through the same code path as bits=4).
    N1, K1, N2, block_size, bits = 32, 32, 4, 16, 8
    rng = np.random.default_rng(9010)
    w1 = rng.standard_normal((N1, K1)).astype(np.float32) * 0.2
    w1[:16] *= 6.0
    w2 = rng.standard_normal((N2, N1)).astype(np.float32) * 0.2
    inits1, info1 = _nbits_weight_initializers(
        w1, block_size, "w1b8", bits=bits, zp_mode="absent"
    )
    inits2, info2 = _nbits_weight_initializers(
        w2, block_size, "w2b8", bits=bits, zp_mode="packed"
    )

    model = _nbits_model(
        f"""
        g (float[batch,{K1}] A) => (float[batch,{N2}] Y)
        {{
          h1 = com.microsoft.MatMulNBits<K={K1},N={N1},bits={bits},block_size={block_size}>(A, {info1["b_name"]}, {info1["scales_name"]})
          h1a = Relu(h1)
          Y = com.microsoft.MatMulNBits<K={N1},N={N2},bits={bits},block_size={block_size}>(h1a, {info2["b_name"]}, {info2["scales_name"]}, {info2["zp_name"]})
        }}
        """,
        initializer=[*inits1, *inits2],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = np.arange(16)
    keep_blocks = np.array([0])
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits[info1["b_name"]].dims) == [
        16,
        info1["k_blocks"],
        block_size * bits // 8,
    ]
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits[info1["b_name"]]),
        _nbits_pack_b(info1["qcodes"][keep], 16, info1["k_blocks"], block_size, bits),
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits[info2["b_name"]]),
        _nbits_pack_b(info2["qcodes"][:, keep], N2, 1, block_size, bits),
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits[info2["zp_name"]]),
        _nbits_pack_codes(info2["zp"][:, keep_blocks], bits),
    )

    rng2 = np.random.default_rng(9011)
    x = rng2.standard_normal((3, K1)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"A": x})
    w1_dequant = _nbits_dequant(
        info1["qcodes"], info1["scales"], None, block_size, bits
    )
    w2_dequant = _nbits_dequant(
        info2["qcodes"], info2["scales"], info2["zp"], block_size, bits
    )
    h1 = np.maximum(x.astype(np.float64) @ w1_dequant[keep].T, 0.0)
    y_oracle = h1 @ w2_dequant[:, keep].T
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-3, atol=1e-3)


def test_cpp_structured_pruning_matmul_nbits_unpacked_float_zero_points_matches_oracle():
    # zero_points as an UNPACKED float32 tensor (same dtype as scales, shape
    # [N, k_blocks]) -- the live schema's OTHER documented encoding.
    N1, K1, N2, block_size = 32, 64, 8, 16
    rng = np.random.default_rng(9020)
    w1 = rng.standard_normal((N1, K1)).astype(np.float32) * 0.2
    w1[:16] *= 6.0
    w2 = rng.standard_normal((N2, N1)).astype(np.float32) * 0.2
    inits1, info1 = _nbits_weight_initializers(
        w1, block_size, "w1u", zp_mode="unpacked"
    )
    inits2, info2 = _nbits_weight_initializers(
        w2, block_size, "w2u", zp_mode="unpacked"
    )

    model = _nbits_model(
        f"""
        g (float[batch,{K1}] A) => (float[batch,{N2}] Y)
        {{
          h1 = com.microsoft.MatMulNBits<K={K1},N={N1},bits=4,block_size={block_size}>(A, {info1["b_name"]}, {info1["scales_name"]}, {info1["zp_name"]})
          h1a = Relu(h1)
          Y = com.microsoft.MatMulNBits<K={N1},N={N2},bits=4,block_size={block_size}>(h1a, {info2["b_name"]}, {info2["scales_name"]}, {info2["zp_name"]})
        }}
        """,
        initializer=[*inits1, *inits2],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = np.arange(16)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert inits[info1["zp_name"]].data_type == onnx.TensorProto.FLOAT
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(inits[info1["zp_name"]]),
        info1["zp"][keep].astype(np.float32),
    )

    rng2 = np.random.default_rng(9021)
    x = rng2.standard_normal((3, K1)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"A": x})
    w1_dequant = _nbits_dequant(
        info1["qcodes"], info1["scales"], info1["zp"], block_size
    )
    w2_dequant = _nbits_dequant(
        info2["qcodes"], info2["scales"], info2["zp"], block_size
    )
    h1 = np.maximum(x.astype(np.float64) @ w1_dequant[keep].T, 0.0)
    y_oracle = h1 @ w2_dequant[:, keep].T
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-3, atol=1e-3)


def test_cpp_structured_pruning_matmul_nbits_gated_ffn_matches_oracle():
    # gate_proj/up_proj (both MatMulNBits, bits=4) combined by Mul, feeding
    # down_proj (also MatMulNBits) -- H=32 output channels, block_size=16 ->
    # down_proj's own K2=H=32 axis is exactly 2 blocks. Both W_gate/W_up
    # engineered so the combined L2-norm importance keeps EXACTLY rows 0-15
    # (block 0), landing on a whole-block boundary.
    K, H, Out, block_size = 16, 32, 4, 16
    rng = np.random.default_rng(9030)
    w_gate = (rng.standard_normal((H, K)) * 0.2).astype(np.float32)
    w_gate[:16] *= 6.0
    w_up = (rng.standard_normal((H, K)) * 0.2).astype(np.float32)
    w_up[:16] *= 6.0
    w_down = (rng.standard_normal((Out, H)) * 0.2).astype(np.float32)

    inits_g, info_g = _nbits_weight_initializers(w_gate, block_size, "wg")
    inits_u, info_u = _nbits_weight_initializers(w_up, block_size, "wu")
    inits_d, info_d = _nbits_weight_initializers(w_down, block_size, "wd")

    model = _nbits_model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = com.microsoft.MatMulNBits<K={K},N={H},bits=4,block_size={block_size}>(X, {info_g["b_name"]}, {info_g["scales_name"]}, {info_g["zp_name"]})
          gate_act = Sigmoid(gate)
          up = com.microsoft.MatMulNBits<K={K},N={H},bits=4,block_size={block_size}>(X, {info_u["b_name"]}, {info_u["scales_name"]}, {info_u["zp_name"]})
          h = Mul(gate_act, up)
          Y = com.microsoft.MatMulNBits<K={H},N={Out},bits=4,block_size={block_size}>(h, {info_d["b_name"]}, {info_d["scales_name"]}, {info_d["zp_name"]})
        }}
        """,
        initializer=[*inits_g, *inits_u, *inits_d],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = np.arange(16)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits[info_g["b_name"]].dims) == [
        16,
        info_g["k_blocks"],
        block_size * 4 // 8,
    ]
    assert list(inits[info_d["b_name"]].dims) == [Out, 1, block_size * 4 // 8]

    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits[info_g["b_name"]]),
        _nbits_pack_b(info_g["qcodes"][keep], 16, info_g["k_blocks"], block_size),
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits[info_u["b_name"]]),
        _nbits_pack_b(info_u["qcodes"][keep], 16, info_u["k_blocks"], block_size),
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits[info_d["b_name"]]),
        _nbits_pack_b(info_d["qcodes"][:, keep], Out, 1, block_size),
    )

    wg_dequant = _nbits_dequant(
        info_g["qcodes"], info_g["scales"], info_g["zp"], block_size
    )
    wu_dequant = _nbits_dequant(
        info_u["qcodes"], info_u["scales"], info_u["zp"], block_size
    )
    wd_dequant = _nbits_dequant(
        info_d["qcodes"], info_d["scales"], info_d["zp"], block_size
    )

    rng2 = np.random.default_rng(9031)
    x = rng2.standard_normal((3, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    gate_lin = x.astype(np.float64) @ wg_dequant[keep].T
    gate = 1.0 / (1.0 + np.exp(-gate_lin))
    up_lin = x.astype(np.float64) @ wu_dequant[keep].T
    h = gate * up_lin
    y_oracle = h @ wd_dequant[:, keep].T
    np.testing.assert_allclose(y, y_oracle, rtol=1e-3, atol=1e-3)


def test_cpp_structured_pruning_matmul_nbits_declines_non_block_aligned():
    # Interleaved importance (even rows large, odd rows small) makes the
    # top-keep_count-by-norm set straddle every block boundary -- the pass
    # must decline this chain entirely (both nodes' B/scales/zero_points
    # left byte-for-byte unchanged), never force a partial-block
    # re-quantization or a mismatched producer/consumer keep-set.
    N1, K1, N2, block_size = 32, 64, 8, 16
    rng = np.random.default_rng(9040)
    w1 = rng.standard_normal((N1, K1)).astype(np.float32) * 0.2
    w1[0::2] *= 8.0  # even rows large ("important"), odd rows small
    w2 = rng.standard_normal((N2, N1)).astype(np.float32) * 0.2
    inits1, info1 = _nbits_weight_initializers(w1, block_size, "w1d")
    inits2, info2 = _nbits_weight_initializers(w2, block_size, "w2d")

    model = _nbits_model(
        f"""
        g (float[batch,{K1}] A) => (float[batch,{N2}] Y)
        {{
          h1 = com.microsoft.MatMulNBits<K={K1},N={N1},bits=4,block_size={block_size}>(A, {info1["b_name"]}, {info1["scales_name"]}, {info1["zp_name"]})
          h1a = Relu(h1)
          Y = com.microsoft.MatMulNBits<K={N1},N={N2},bits=4,block_size={block_size}>(h1a, {info2["b_name"]}, {info2["scales_name"]}, {info2["zp_name"]})
        }}
        """,
        initializer=[*inits1, *inits2],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    before = {t.name: t.SerializeToString() for t in model.graph.initializer}
    after = {t.name: t.SerializeToString() for t in pruned.graph.initializer}
    assert before == after


def test_cpp_structured_pruning_matmul_nbits_mixed_producer_plain_consumer_matches_oracle():
    # MatMulNBits producer -> Relu -> plain-float MatMul consumer: the
    # "quantized transformer-block layer feeding an unquantized lm_head"
    # export shape. A plain-float CONSUMER has no block structure at all,
    # so the producer's own top-keep_count-by-norm keep-set (rows 0-15
    # here) applies directly -- no block-alignment check needed.
    N1, K1, Out, block_size = 32, 64, 8, 16
    rng = np.random.default_rng(9050)
    w1 = rng.standard_normal((N1, K1)).astype(np.float32) * 0.2
    w1[:16] *= 6.0
    bias1 = (rng.standard_normal(N1) * 0.05).astype(np.float32)
    w2 = rng.standard_normal((N1, Out)).astype(np.float32) * 0.3  # [K, N] storage

    inits1, info1 = _nbits_weight_initializers(w1, block_size, "w1m")

    model = _nbits_model(
        f"""
        g (float[batch,{K1}] A) => (float[batch,{Out}] Y)
        {{
          h1 = com.microsoft.MatMulNBits<K={K1},N={N1},bits=4,block_size={block_size}>(A, {info1["b_name"]}, {info1["scales_name"]}, {info1["zp_name"]}, , bias1)
          h1a = Relu(h1)
          Y = MatMul(h1a, W2)
        }}
        """,
        initializer=[*inits1, _f32(bias1, "bias1"), _f32(w2, "W2")],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = np.arange(16)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits[info1["b_name"]].dims) == [
        16,
        info1["k_blocks"],
        block_size * 4 // 8,
    ]
    assert list(inits["W2"].dims) == [16, Out]
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(inits["W2"]), w2[keep])

    rng2 = np.random.default_rng(9051)
    x = rng2.standard_normal((3, K1)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"A": x})
    w1_dequant = _nbits_dequant(
        info1["qcodes"], info1["scales"], info1["zp"], block_size
    )
    h1 = np.maximum(x.astype(np.float64) @ w1_dequant[keep].T + bias1[keep], 0.0)
    y_oracle = h1 @ w2[keep].astype(np.float64)
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-3, atol=1e-3)


def test_cpp_structured_pruning_matmul_nbits_mixed_plain_producer_declines_non_block_aligned():
    # The mixed-chain analogue of the decline test above: a plain-float
    # PRODUCER (no block structure of its own) feeding a MatMulNBits
    # CONSUMER whose own block-alignment requirement still applies -- the
    # pass must decline the whole chain, never force a partial-block
    # re-quantization or a mismatched producer/consumer keep-set.
    N1, K1, N2, block_size = 32, 64, 8, 16
    rng = np.random.default_rng(9080)
    w1 = rng.standard_normal((K1, N1)).astype(np.float32) * 0.2
    w1[:, 0::2] *= 8.0  # even output channels large, odd small

    w2 = rng.standard_normal((N2, N1)).astype(np.float32) * 0.2
    inits2, info2 = _nbits_weight_initializers(w2, block_size, "w2mp")

    model = _nbits_model(
        f"""
        g (float[batch,{K1}] A) => (float[batch,{N2}] Y)
        {{
          h1 = MatMul(A, W1)
          h1a = Relu(h1)
          Y = com.microsoft.MatMulNBits<K={N1},N={N2},bits=4,block_size={block_size}>(h1a, {info2["b_name"]}, {info2["scales_name"]}, {info2["zp_name"]})
        }}
        """,
        initializer=[*inits2, _f32(w1, "W1")],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    before_w1 = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "W1")
    )
    after_w1 = onnx.numpy_helper.to_array(
        next(t for t in pruned.graph.initializer if t.name == "W1")
    )
    np.testing.assert_array_equal(before_w1, after_w1)
    before = {t.name: t.SerializeToString() for t in model.graph.initializer}
    after = {t.name: t.SerializeToString() for t in pruned.graph.initializer}
    assert before == after


def test_cpp_structured_pruning_matmul_nbits_declines_shared_weight():
    # The same B/scales tensors read by two different MatMulNBits nodes --
    # slicing them for one chain would silently corrupt the other reader.
    N1, K1, N2, block_size = 32, 64, 8, 16
    rng = np.random.default_rng(9070)
    w1 = rng.standard_normal((N1, K1)).astype(np.float32) * 0.2
    w1[:16] *= 6.0
    w2 = rng.standard_normal((N2, N1)).astype(np.float32) * 0.2
    inits1, info1 = _nbits_weight_initializers(w1, block_size, "w1s")
    inits2, info2 = _nbits_weight_initializers(w2, block_size, "w2s")

    model = _nbits_model(
        f"""
        g (float[batch,{K1}] A) => (float[batch,{N2}] Y, float[batch,{N1}] Y2)
        {{
          h1 = com.microsoft.MatMulNBits<K={K1},N={N1},bits=4,block_size={block_size}>(A, {info1["b_name"]}, {info1["scales_name"]}, {info1["zp_name"]})
          h1a = Relu(h1)
          Y = com.microsoft.MatMulNBits<K={N1},N={N2},bits=4,block_size={block_size}>(h1a, {info2["b_name"]}, {info2["scales_name"]}, {info2["zp_name"]})
          Y2 = com.microsoft.MatMulNBits<K={K1},N={N1},bits=4,block_size={block_size}>(A, {info1["b_name"]}, {info1["scales_name"]})
        }}
        """,
        initializer=[*inits1, *inits2],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    before = {t.name: t.SerializeToString() for t in model.graph.initializer}
    after = {t.name: t.SerializeToString() for t in pruned.graph.initializer}
    assert before == after


def test_cpp_structured_pruning_matmul_nbits_matches_python_reference_output():
    N1, K1, N2, block_size = 32, 64, 8, 16
    rng = np.random.default_rng(9060)
    w1 = rng.standard_normal((N1, K1)).astype(np.float32) * 0.2
    w1[:16] *= 6.0
    w2 = rng.standard_normal((N2, N1)).astype(np.float32) * 0.2
    bias1 = (rng.standard_normal(N1) * 0.05).astype(np.float32)

    inits1, info1 = _nbits_weight_initializers(w1, block_size, "w1p")
    inits2, info2 = _nbits_weight_initializers(w2, block_size, "w2p")

    model = _nbits_model(
        f"""
        g (float[batch,{K1}] A) => (float[batch,{N2}] Y)
        {{
          h1 = com.microsoft.MatMulNBits<K={K1},N={N1},bits=4,block_size={block_size}>(A, {info1["b_name"]}, {info1["scales_name"]}, {info1["zp_name"]}, , bias1)
          h1a = Relu(h1)
          Y = com.microsoft.MatMulNBits<K={N1},N={N2},bits=4,block_size={block_size}>(h1a, {info2["b_name"]}, {info2["scales_name"]}, {info2["zp_name"]})
        }}
        """,
        initializer=[*inits1, *inits2, _f32(bias1, "bias1")],
    )
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_structured_pruning_matmul_nbits(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    rng2 = np.random.default_rng(9061)
    x = rng2.standard_normal((4, K1)).astype(np.float32)
    (y_py,) = _run(pruned_py, {"A": x})
    (y_cpp,) = _run(pruned_cpp, {"A": x})
    np.testing.assert_allclose(y_py, y_cpp, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_nbits_separate_bias_add_hop_matches_oracle():
    # Regression test for the per-channel bias/scale `Add`/`Mul`
    # pass-through hop (`_BINARY_CHANNEL_OPS` in pruning.py,
    # WalkToConsumer's own identical hop in structured_pruning_entry.cpp)
    # newly ported into WalkToMatMulNBitsConsumer: a SEPARATE trailing bias
    # `Add` node (not MatMulNBits's own fused `bias` input) between the
    # producer and the plain Relu/consumer -- the real shape a
    # ``MatMulNBitsQuantizer`` round trip commonly emits before any later
    # ORT graph-optimizer bias-folding pass. Before this hop was ported, the
    # walk would decline at the `Add` node entirely (an unrecognized
    # mid-chain op), leaving both `MatMulNBits` nodes byte-unchanged; with
    # it, the chain is found, the bias is co-sliced, and the producer/
    # consumer N/K axes are still pruned exactly as
    # test_cpp_structured_pruning_matmul_nbits_plain_chain_matches_oracle
    # confirms for the fused-bias case.
    N1, K1, N2, block_size = 32, 64, 8, 16
    rng = np.random.default_rng(9101)
    w1 = rng.standard_normal((N1, K1)).astype(np.float32) * 0.2
    w1[:16] *= 6.0  # rows 0-15 large (kept), 16-31 small (dropped)
    w2 = rng.standard_normal((N2, N1)).astype(np.float32) * 0.2
    bias1 = (rng.standard_normal(N1) * 0.05).astype(np.float32)

    inits1, info1 = _nbits_weight_initializers(w1, block_size, "sb_w1")
    inits2, info2 = _nbits_weight_initializers(w2, block_size, "sb_w2")

    model = _nbits_model(
        f"""
        g (float[batch,{K1}] A) => (float[batch,{N2}] Y)
        {{
          h1 = com.microsoft.MatMulNBits<K={K1},N={N1},bits=4,block_size={block_size}>(A, {info1["b_name"]}, {info1["scales_name"]}, {info1["zp_name"]})
          hb = Add(h1, Bias1)
          h1a = Relu(hb)
          Y = com.microsoft.MatMulNBits<K={N1},N={N2},bits=4,block_size={block_size}>(h1a, {info2["b_name"]}, {info2["scales_name"]}, {info2["zp_name"]})
        }}
        """,
        initializer=[*inits1, *inits2, _f32(bias1, "Bias1")],
    )
    onnx.checker.check_model(model)

    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_structured_pruning_matmul_nbits(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)

    keep = np.arange(16)
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    # The chain must actually be found and pruned (not silently declined):
    # both N-axes and the separate Bias1 all shrink to the same keep_count.
    assert list(inits["Bias1"].dims) == [16]
    assert list(inits[info1["b_name"]].dims) == [
        16,
        info1["k_blocks"],
        block_size * 4 // 8,
    ]
    assert list(inits[info2["b_name"]].dims) == [N2, 1, block_size * 4 // 8]
    np.testing.assert_allclose(onnx.numpy_helper.to_array(inits["Bias1"]), bias1[keep])

    # Byte-for-byte parity with the (already-correct) Python reference.
    py_bytes = {t.name: t.SerializeToString() for t in pruned_py.graph.initializer}
    cpp_bytes = {t.name: t.SerializeToString() for t in pruned_cpp.graph.initializer}
    assert py_bytes == cpp_bytes

    rng2 = np.random.default_rng(9102)
    x = rng2.standard_normal((3, K1)).astype(np.float32)
    (y_pruned,) = _run(pruned_cpp, {"A": x})
    w1_dequant = _nbits_dequant(
        info1["qcodes"], info1["scales"], info1["zp"], block_size
    )
    w2_dequant = _nbits_dequant(
        info2["qcodes"], info2["scales"], info2["zp"], block_size
    )
    h1 = np.maximum(x.astype(np.float64) @ w1_dequant[keep].T + bias1[keep], 0.0)
    y_oracle = h1 @ w2_dequant[:, keep].T
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-3, atol=1e-3)


def test_cpp_structured_pruning_matmul_nbits_separate_bias_add_tied_declines():
    # Bias1 (the mid-chain Add's own per-channel constant) is ALSO read by a
    # second, unrelated Add elsewhere in the graph -- the tied/shared-tensor
    # bar this hop's own C++ comment documents (mirrors
    # `len(consumers_of.get(other, [])) == 1` in pruning.py): slicing Bias1
    # in place would silently corrupt that second reader, so the walk must
    # decline the Add hop (and hence the whole chain) entirely, leaving the
    # model byte-unchanged. `h1` itself keeps exactly one consumer (`hb`),
    # so this exercises the Add-hop's own tied-constant check specifically,
    # not the ordinary "more than one consumer" top-of-loop dispatch.
    N1, K1, N2, block_size = 32, 64, 8, 16
    rng = np.random.default_rng(9111)
    w1 = rng.standard_normal((N1, K1)).astype(np.float32) * 0.2
    w2 = rng.standard_normal((N2, N1)).astype(np.float32) * 0.2
    bias1 = (rng.standard_normal(N1) * 0.05).astype(np.float32)

    inits1, info1 = _nbits_weight_initializers(w1, block_size, "td_w1")
    inits2, info2 = _nbits_weight_initializers(w2, block_size, "td_w2")

    model = _nbits_model(
        f"""
        g (float[batch,{K1}] A, float[batch,{N1}] B) => (float[batch,{N2}] Y, float[batch,{N1}] Y2)
        {{
          h1 = com.microsoft.MatMulNBits<K={K1},N={N1},bits=4,block_size={block_size}>(A, {info1["b_name"]}, {info1["scales_name"]}, {info1["zp_name"]})
          hb = Add(h1, Bias1)
          h1a = Relu(hb)
          Y = com.microsoft.MatMulNBits<K={N1},N={N2},bits=4,block_size={block_size}>(h1a, {info2["b_name"]}, {info2["scales_name"]}, {info2["zp_name"]})
          Y2 = Add(B, Bias1)
        }}
        """,
        initializer=[*inits1, *inits2, _f32(bias1, "Bias1")],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


# --- MatMulNBitsMlp (com.microsoft, fused gated-MLP block-quantized weight)
# --- structured pruning ---------------------------------------------------
#
# Tests for the fused ``MatMulNBitsMlp`` chain family
# (``onnxsim/structured_pruning_entry.cpp``'s own "MatMulNBitsMlp/
# MatMulNBitsQkv" subsection, wired into `ApplyStructuredPruning` -- see that
# subsection's own top comment for exactly why), mirroring
# ``tests/test_pruning.py``'s own ``test_matmul_nbits_mlp_pruning_matches_
# decomposed_oracle``. Neither this op nor ``MatMulNBitsQkv`` has a
# ``zero_points`` input at all (confirmed via live schema introspection --
# see that subsection's own top comment), so every weight slot here uses the
# schema's own DEFAULT zero point (``2 ** (bits - 1)``) --
# ``_nbits_quantize_default_zp`` below, independent from
# ``_nbits_quantize_block`` above (which explicitly quantizes an OWN
# zero_points array plain ``MatMulNBits`` can carry). ``MatMulNBitsMlp``
# itself cannot be executed via a real ``InferenceSession`` on this
# environment's CPU EP at all (confirmed empirically: ``NOT_IMPLEMENTED :
# Could not find an implementation for MatMulNBitsMlp`` -- the same
# WebGPU-EP-only finding ``tests/test_pruning.py``'s own section comment
# documents at length), so the end-to-end oracle test below DECOMPOSES the
# PRUNED fused node's own tensors back into two standalone ``MatMulNBits``
# nodes (a real CPU kernel) and runs THOSE through ``_run``, mirroring
# ``test_pruning.py``'s own identical proxy-topology technique.


def _nbits_quantize_default_zp(w, block_size, bits=4):
    """Independent reference block quantizer using the schema's own DEFAULT
    zero point (``2 ** (bits - 1)``) rather than an explicit ``zero_points``
    tensor -- the only encoding ``MatMulNBitsMlp``/``MatMulNBitsQkv`` support.
    Returns ``(qcodes uint8 [N, K], scales float32 [N, k_blocks], k_blocks)``.
    """
    n, k = w.shape
    assert k % block_size == 0
    k_blocks = k // block_size
    qmax = (1 << bits) - 1
    zp = float(1 << (bits - 1))
    scales = np.zeros((n, k_blocks), dtype=np.float32)
    qcodes = np.zeros((n, k), dtype=np.uint8)
    for row in range(n):
        for kb in range(k_blocks):
            k0, k1 = kb * block_size, (kb + 1) * block_size
            block = w[row, k0:k1]
            maxabs = max(float(np.max(np.abs(block))), 1e-8)
            scale = maxabs / max(zp, qmax - zp)
            scales[row, kb] = scale
            codes = np.round(block / scale + zp).clip(0, qmax)
            qcodes[row, k0:k1] = codes.astype(np.uint8)
    return qcodes, scales, k_blocks


def _nbits_no_zp_initializers(w, block_size, prefix, bits=4):
    """``_nbits_weight_initializers``'s zero_points-free analogue, for one
    branch of a matched ``MatMulNBitsMlp``/``MatMulNBitsQkv`` node -- see
    this section's own top comment for why no ``zero_points`` tensor is ever
    attached here at all.
    """
    qcodes, scales, k_blocks = _nbits_quantize_default_zp(w, block_size, bits)
    b = _nbits_pack_b(qcodes, w.shape[0], k_blocks, block_size, bits)
    inits = [
        onnx.numpy_helper.from_array(b, name=f"{prefix}_B"),
        onnx.numpy_helper.from_array(scales, name=f"{prefix}_scales"),
    ]
    return inits, dict(
        qcodes=qcodes,
        scales=scales,
        k_blocks=k_blocks,
        b_name=f"{prefix}_B",
        scales_name=f"{prefix}_scales",
    )


def test_cpp_structured_pruning_matmul_nbits_mlp_matches_decomposed_oracle():
    # Channels 0-3 engineered LARGE (kept), 4-7 SMALL (dropped) on BOTH
    # gate/up branches, so the combined (root-sum-square) L2 importance
    # ranking is unambiguous: sparsity=0.5 must keep exactly {0, 1, 2, 3}.
    N, K, N2, block_size = 8, 32, 5, 32
    rng = np.random.default_rng(9100)
    w_gate = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    w_up = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    w_gate[:4] *= 8.0
    w_up[:4] *= 8.0
    w_gate[4:] *= 0.05
    w_up[4:] *= 0.05
    bias_gate = (rng.standard_normal(N) * 0.05).astype(np.float32)
    bias_up = (rng.standard_normal(N) * 0.05).astype(np.float32)
    down_w = (rng.standard_normal((N, N2)) * 0.3).astype(np.float32)

    inits_g, info_g = _nbits_no_zp_initializers(w_gate, block_size, "mlpgate")
    inits_u, info_u = _nbits_no_zp_initializers(w_up, block_size, "mlpup")

    model = _nbits_model(
        f"""
        g (float[batch,{K}] A) => (float[batch,{N2}] Z)
        {{
          Y = com.microsoft.MatMulNBitsMlp<activation="Relu",block_size={block_size},bits=4,N={N},K={K}>(A, , , {info_g["b_name"]}, {info_g["scales_name"]}, gate_bias, {info_u["b_name"]}, {info_u["scales_name"]}, up_bias)
          Z = MatMul(Y, down_w)
        }}
        """,
        initializer=[
            *inits_g,
            *inits_u,
            _f32(bias_gate, "gate_bias"),
            _f32(bias_up, "up_bias"),
            _f32(down_w, "down_w"),
        ],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = np.array([0, 1, 2, 3])
    mlp_node = next(n for n in pruned.graph.node if n.op_type == "MatMulNBitsMlp")
    assert next(a.i for a in mlp_node.attribute if a.name == "N") == 4
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits[info_g["b_name"]].dims) == [
        4,
        info_g["k_blocks"],
        block_size * 4 // 8,
    ]
    assert list(inits[info_u["b_name"]].dims) == [
        4,
        info_u["k_blocks"],
        block_size * 4 // 8,
    ]
    assert list(inits["down_w"].dims) == [4, N2]

    # "slice, don't recompute": the pruned packed tensors must be BYTE-
    # IDENTICAL to the original ones restricted to `keep`, not merely
    # numerically close.
    gate_B_expected = _nbits_pack_b(
        info_g["qcodes"][keep], 4, info_g["k_blocks"], block_size
    )
    up_B_expected = _nbits_pack_b(
        info_u["qcodes"][keep], 4, info_u["k_blocks"], block_size
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits[info_g["b_name"]]), gate_B_expected
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits[info_u["b_name"]]), up_B_expected
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["down_w"]), down_w[keep]
    )

    # Decompose the PRUNED fused node's own tensors into two standalone
    # MatMulNBits nodes and run them through a real CPU-kernel
    # InferenceSession -- see this section's own top comment for why the
    # literal fused node itself cannot be executed here.
    decomposed = _nbits_model(
        f"""
        g (float[batch,{K}] A) => (float[batch,4] gate_out, float[batch,4] up_out)
        {{
          gate_out = com.microsoft.MatMulNBits<K={K},N=4,bits=4,block_size={block_size}>(A, {info_g["b_name"]}, {info_g["scales_name"]}, , , gate_bias)
          up_out = com.microsoft.MatMulNBits<K={K},N=4,bits=4,block_size={block_size}>(A, {info_u["b_name"]}, {info_u["scales_name"]}, , , up_bias)
        }}
        """,
        initializer=[
            inits[info_g["b_name"]],
            inits[info_g["scales_name"]],
            inits["gate_bias"],
            inits[info_u["b_name"]],
            inits[info_u["scales_name"]],
            inits["up_bias"],
        ],
    )
    onnx.checker.check_model(decomposed)

    x = np.random.default_rng(9101).standard_normal((3, K)).astype(np.float32)
    gate_out, up_out = _run(decomposed, {"A": x})
    y_actual = np.maximum(gate_out, 0) * up_out

    w_gate_dequant = _nbits_dequant(
        info_g["qcodes"], info_g["scales"], None, block_size
    )
    w_up_dequant = _nbits_dequant(info_u["qcodes"], info_u["scales"], None, block_size)
    gate_ref = x.astype(np.float64) @ w_gate_dequant[keep].T + bias_gate[keep]
    up_ref = x.astype(np.float64) @ w_up_dequant[keep].T + bias_up[keep]
    y_ref = np.maximum(gate_ref, 0) * up_ref
    np.testing.assert_allclose(y_actual, y_ref, rtol=1e-3, atol=1e-3)


def test_cpp_structured_pruning_matmul_nbits_mlp_declines_non_block_aligned_consumer():
    # Interleaved importance (even channels large, odd small) on both gate/up
    # branches makes the top-keep_count-by-norm set {0, 2, 4, 6} straddle
    # every block boundary of the downstream MatMulNBits consumer's own
    # block_size=4 axis (blocks [0,4) and [4,8), each half kept/half
    # dropped) -- the pass must decline the whole chain (fused node AND
    # consumer left byte-for-byte unchanged), mirroring the plain
    # MatMulNBits section's own identical decline precedent.
    N, K, N2, block_size = 8, 32, 4, 32
    consumer_block_size = 4
    rng = np.random.default_rng(9110)
    w_gate = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    w_up = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    w_gate[0::2] *= 8.0  # even channels large ("important"), odd small
    w_up[0::2] *= 8.0
    bias_gate = (rng.standard_normal(N) * 0.05).astype(np.float32)
    bias_up = (rng.standard_normal(N) * 0.05).astype(np.float32)
    w_down = (rng.standard_normal((N2, N)) * 0.3).astype(np.float32)

    inits_g, info_g = _nbits_no_zp_initializers(w_gate, block_size, "mlpgated")
    inits_u, info_u = _nbits_no_zp_initializers(w_up, block_size, "mlpupd")
    inits_d, info_d = _nbits_weight_initializers(
        w_down, consumer_block_size, "mlpdownd"
    )

    model = _nbits_model(
        f"""
        g (float[batch,{K}] A) => (float[batch,{N2}] Z)
        {{
          Y = com.microsoft.MatMulNBitsMlp<activation="Relu",block_size={block_size},bits=4,N={N},K={K}>(A, , , {info_g["b_name"]}, {info_g["scales_name"]}, gate_bias, {info_u["b_name"]}, {info_u["scales_name"]}, up_bias)
          Z = com.microsoft.MatMulNBits<K={N},N={N2},bits=4,block_size={consumer_block_size}>(Y, {info_d["b_name"]}, {info_d["scales_name"]}, {info_d["zp_name"]})
        }}
        """,
        initializer=[
            *inits_g,
            *inits_u,
            *inits_d,
            _f32(bias_gate, "gate_bias"),
            _f32(bias_up, "up_bias"),
        ],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    before = {t.name: t.SerializeToString() for t in model.graph.initializer}
    after = {t.name: t.SerializeToString() for t in pruned.graph.initializer}
    assert before == after
    attrs_before = [
        (n.name, [(a.name, a.i) for a in n.attribute]) for n in model.graph.node
    ]
    attrs_after = [
        (n.name, [(a.name, a.i) for a in n.attribute]) for n in pruned.graph.node
    ]
    assert attrs_before == attrs_after


# --- QDQ (quantized-weight) structured pruning -----------------------------
#
# Mirrors ``tests/test_pruning.py``'s own "apply_structured_pruning_qdq"
# coverage for ``onnxsim.apply_structured_pruning_cpp`` -- the C++ port also
# runs the QDQ-quantized chain families (see
# ``onnxsim/structured_pruning_entry.cpp``'s own "QDQ (quantized-weight)
# structured pruning" section comment), wired into the SAME entry point as
# the plain-float chains above (unlike ``onnxsim.pruning``, which keeps
# ``apply_structured_pruning_qdq`` a separate top-level function). Numeric-
# oracle tests below run the pruned QDQ graph through onnxruntime with graph
# optimizations disabled (``_run_unfused``), not the default-optimization
# ``_run`` every other test in this file uses -- confirmed empirically (see
# ``test_pruning.py``'s own identical comment): with default optimizations,
# onnxruntime's own QDQ-aware graph optimizer fuses the
# DequantizeLinear->MatMul/Conv pattern into an internal integer kernel whose
# accumulation order differs from naive float dequantize-then-matmul,
# changing the result by more than float32 rounding -- an onnxruntime
# *optimization* artifact orthogonal to this pass's own correctness.


def _i8(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.int8), name)


def _int4(array, name):
    return onnx.numpy_helper.from_array(
        array.astype(np.int8).astype(ml_dtypes.int4), name
    )


def _uint4(array, name):
    return onnx.numpy_helper.from_array(
        array.astype(np.uint8).astype(ml_dtypes.uint4), name
    )


def _run_unfused(model, feeds):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _qdq_dequant(q, scale, zero_point, axis):
    # Independent reference dequantization -- mirrors the ONNX
    # DequantizeLinear formula `y = (x - x_zero_point) * x_scale` directly,
    # not anything from onnxsim itself.
    q = q.astype(np.float64)
    scale = scale.astype(np.float64)
    if scale.ndim == 1:
        shape = [1] * q.ndim
        shape[axis] = -1
        scale = scale.reshape(shape)
    if zero_point is not None:
        zp = zero_point.astype(np.float64)
        if zp.ndim == 1:
            shape = [1] * q.ndim
            shape[axis] = -1
            zp = zp.reshape(shape)
        q = q - zp
    return q * scale


def _qdq_block_dequant(q, scale, zero_point, axis, block_size):
    # Independent reference for BLOCKWISE dequantization -- mirrors opset
    # 21's own `DequantizeLinear` formula with a `block_size` attribute.
    q = q.astype(np.float64)
    scale = np.repeat(scale.astype(np.float64), block_size, axis=axis)
    slicer = [slice(None)] * q.ndim
    slicer[axis] = slice(0, q.shape[axis])
    scale = scale[tuple(slicer)]
    if zero_point is not None:
        zp = np.repeat(zero_point.astype(np.float64), block_size, axis=axis)
        zp = zp[tuple(slicer)]
        q = q - zp
    return q * scale


def _requantize_symmetric_int8_per_channel(w_float):
    # An independent "dequantize-then-prune-then-REQUANTIZE-from-scratch"
    # oracle -- NOT what this pass does (it slices the ORIGINAL codes/scale
    # unchanged, see this file's own section comment above), used only to
    # positively confirm "slice, don't recompute": a fresh symmetric
    # per-output-channel INT8 quantization fit to the pruned float
    # submatrix's own max-abs generally derives a numerically DIFFERENT
    # scale than the original tensor's own (unrelated) scale values, so if
    # the pass's own pruned scale matched THIS oracle instead of
    # `Wscale[keep]`, that would be conclusive evidence of a
    # dequant-requant bug rather than a direct slice.
    amax = np.max(np.abs(w_float), axis=0)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    codes = np.clip(np.round(w_float / scale), -127, 127).astype(np.int8)
    return codes, scale


def _matmul_qdq_producer_model(K=8, H=16, Out=4, seed=0, zero_point=False):
    rng = np.random.default_rng(seed)
    Wq = rng.integers(-100, 100, size=(K, H)).astype(np.int8)
    Wscale = np.abs(rng.standard_normal(H)).astype(np.float32) * 0.02 + 0.001
    W2 = rng.standard_normal((H, Out)).astype(np.float32)
    initializer = [_i8(Wq, "Wq"), _f32(Wscale, "Wscale"), _f32(W2, "W2")]
    zp_input = ""
    Wzp = None
    if zero_point:
        Wzp = rng.integers(-20, 20, size=H).astype(np.int8)
        initializer.append(_i8(Wzp, "Wzp"))
        zp_input = ", Wzp"
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          Wdq = DequantizeLinear<axis=1>(Wq, Wscale{zp_input})
          h = MatMul(X, Wdq)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=initializer,
    )
    return model, Wq, Wscale, Wzp, W2


def test_cpp_qdq_structured_pruning_matmul_producer_per_channel_matches_oracle():
    # Per-channel INT8 QDQ MatMul producer feeding a plain float MatMul
    # consumer -- co-sliced Wq/Wscale, verified end to end AND confirmed to
    # be a genuine slice rather than a dequant-prune-requantize round trip
    # (per this task's own verification bar).
    K, H, Out = 8, 16, 4
    model, Wq, Wscale, _Wzp, W2 = _matmul_qdq_producer_model(K, H, Out, seed=0)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [K, H // 2]
    assert list(inits["Wscale"].dims) == [H // 2]
    assert list(inits["W2"].dims) == [H // 2, Out]

    w_dequant = _qdq_dequant(Wq, Wscale, None, axis=1)
    keep = _oracle_keep_indices(w_dequant, H // 2)

    rng = np.random.default_rng(9)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run_unfused(pruned, {"X": x})
    h = x @ w_dequant[:, keep]
    y_oracle = h @ W2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)

    # "Slice, don't recompute": the pruned Wq/Wscale are BIT-IDENTICAL to
    # the original tensors at the surviving indices, never anything derived
    # from a fresh quantization of the pruned float weight.
    wq_pruned = onnx.numpy_helper.to_array(inits["Wq"])
    wscale_pruned = onnx.numpy_helper.to_array(inits["Wscale"])
    np.testing.assert_array_equal(wq_pruned, Wq[:, keep])
    np.testing.assert_allclose(wscale_pruned, Wscale[keep])

    # Oracle cross-check: an independent dequantize-prune-REQUANTIZE pass
    # over the SAME pruned float submatrix derives a scale that generally
    # differs from the pass's own (original, sliced) scale -- confirming
    # the equality above is a genuine "slice, don't recompute" result, not
    # a coincidence of a requantization oracle landing on the same values.
    _requant_codes, requant_scale = _requantize_symmetric_int8_per_channel(
        w_dequant[:, keep]
    )
    assert not np.allclose(wscale_pruned, requant_scale)


def test_cpp_qdq_structured_pruning_matmul_producer_asymmetric_zero_point_matches_oracle():
    K, H, Out = 8, 16, 4
    model, Wq, Wscale, Wzp, W2 = _matmul_qdq_producer_model(
        K, H, Out, seed=1, zero_point=True
    )
    onnx.checker.check_model(model)
    assert Wzp is not None
    assert np.any(Wzp != 0)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wzp"].dims) == [H // 2]

    w_dequant = _qdq_dequant(Wq, Wscale, Wzp, axis=1)
    keep = _oracle_keep_indices(w_dequant, H // 2)

    rng = np.random.default_rng(10)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run_unfused(pruned, {"X": x})
    h = x @ w_dequant[:, keep]
    y_oracle = h @ W2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)

    wzp_pruned = onnx.numpy_helper.to_array(inits["Wzp"])
    np.testing.assert_array_equal(wzp_pruned, Wzp[keep])


def test_cpp_qdq_structured_pruning_matmul_consumer_matches_oracle_and_leaves_scale_untouched():
    # Plain float producer (W1) feeding a QDQ (per-channel) MatMul consumer
    # -- only Wq's own input/reduction axis is sliced; Wscale (indexed by
    # the consumer's own OUTPUT channel) must come out byte-identical.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(4)
    W1 = rng.standard_normal((K, H)).astype(np.float32)
    Wq = rng.integers(-100, 100, size=(H, Out)).astype(np.int8)
    Wscale = np.abs(rng.standard_normal(Out)).astype(np.float32) * 0.03 + 0.001
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          Wdq = DequantizeLinear<axis=1>(Wq, Wscale)
          Y = MatMul(h, Wdq)
        }}
        """,
        initializer=[_f32(W1, "W1"), _i8(Wq, "Wq"), _f32(Wscale, "Wscale")],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [H // 2, Out]
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(inits["Wscale"]), Wscale)

    keep = _oracle_keep_indices(W1, H // 2)
    w_dequant = _qdq_dequant(Wq, Wscale, None, axis=1)

    rng2 = np.random.default_rng(5)
    x = rng2.standard_normal((5, K)).astype(np.float32)
    (y,) = _run_unfused(pruned, {"X": x})
    h = x @ W1[:, keep]
    y_oracle = h @ w_dequant[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_qdq_structured_pruning_per_tensor_scale_left_untouched():
    # Per-tensor (scalar) QDQ weight on the producer side: only Wq is
    # sliced -- the scalar Wscale is byte-identical regardless of channel
    # count.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(6)
    Wq = rng.integers(-100, 100, size=(K, H)).astype(np.int8)
    Wscale = np.array(0.05, dtype=np.float32)
    W2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          Wdq = DequantizeLinear(Wq, Wscale)
          h = MatMul(X, Wdq)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=[_i8(Wq, "Wq"), _f32(Wscale, "Wscale"), _f32(W2, "W2")],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [K, H // 2]
    assert onnx.numpy_helper.to_array(inits["Wscale"]) == pytest.approx(0.05)

    w_dequant = Wq.astype(np.float64) * 0.05
    keep = _oracle_keep_indices(w_dequant, H // 2)
    rng2 = np.random.default_rng(7)
    x = rng2.standard_normal((5, K)).astype(np.float32)
    (y,) = _run_unfused(pruned, {"X": x})
    y_oracle = (x @ w_dequant[:, keep]) @ W2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_qdq_structured_pruning_conv_both_sides_qdq_matches_oracle():
    Cin, Cmid, Cout = 4, 8, 4
    rng = np.random.default_rng(8)
    Wq1 = rng.integers(-100, 100, size=(Cmid, Cin, 1, 1)).astype(np.int8)
    Wscale1 = np.abs(rng.standard_normal(Cmid)).astype(np.float32) * 0.02 + 0.001
    Wq2 = rng.integers(-100, 100, size=(Cout, Cmid, 1, 1)).astype(np.int8)
    Wscale2 = np.abs(rng.standard_normal(Cout)).astype(np.float32) * 0.02 + 0.001
    model = _model(
        f"""
        g (float[1,{Cin},4,4] X) => (float[1,{Cout},4,4] Y)
        {{
          W1dq = DequantizeLinear<axis=0>(Wq1, Wscale1)
          h = Conv(X, W1dq)
          W2dq = DequantizeLinear<axis=0>(Wq2, Wscale2)
          Y = Conv(h, W2dq)
        }}
        """,
        initializer=[
            _i8(Wq1, "Wq1"),
            _f32(Wscale1, "Wscale1"),
            _i8(Wq2, "Wq2"),
            _f32(Wscale2, "Wscale2"),
        ],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq1"].dims) == [Cmid // 2, Cin, 1, 1]
    assert list(inits["Wscale1"].dims) == [Cmid // 2]
    assert list(inits["Wq2"].dims) == [Cout, Cmid // 2, 1, 1]
    assert list(inits["Wscale2"].dims) == [Cout]  # consumer's own scale untouched

    w1_dequant = _qdq_dequant(Wq1, Wscale1, None, axis=0)
    w2_dequant = _qdq_dequant(Wq2, Wscale2, None, axis=0)
    w1_nk = w1_dequant.reshape(Cmid, -1)
    keep = _oracle_keep_indices(w1_nk.T, Cmid // 2)

    rng2 = np.random.default_rng(11)
    x = rng2.standard_normal((1, Cin, 4, 4)).astype(np.float32)
    (y,) = _run_unfused(pruned, {"X": x})

    ref_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Conv", ["X", "W1"], ["h"]),
                onnx.helper.make_node("Conv", ["h", "W2"], ["Y"]),
            ],
            "oracle",
            [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, None)],
            [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)],
            initializer=[
                onnx.numpy_helper.from_array(
                    w1_dequant[keep].astype(np.float32), name="W1"
                ),
                onnx.numpy_helper.from_array(
                    w2_dequant[:, keep].astype(np.float32), name="W2"
                ),
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 21)],
        ir_version=10,
    )
    (y_oracle,) = onnx.reference.ReferenceEvaluator(ref_model).run(None, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_qdq_structured_pruning_declines_grouped_conv():
    # A general grouped/depthwise QDQ Conv is out of scope -- see this
    # file's own section comment above; must be left completely untouched.
    Cin, Cout, group = 8, 8, 4
    rng = np.random.default_rng(13)
    Wq = rng.integers(-100, 100, size=(Cout, Cin // group, 1, 1)).astype(np.int8)
    Wscale = np.abs(rng.standard_normal(Cout)).astype(np.float32) * 0.02 + 0.001
    model = _model(
        f"""
        g (float[1,{Cin},2,2] X) => (float[1,{Cout},2,2] Y)
        {{
          Wdq = DequantizeLinear<axis=0>(Wq, Wscale)
          Y = Conv<group={group}>(X, Wdq)
        }}
        """,
        initializer=[_i8(Wq, "Wq"), _f32(Wscale, "Wscale")],
    )
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == list(Wq.shape)


def test_cpp_qdq_structured_pruning_declines_shared_quantized_weight():
    # The same int8 initializer feeding two independent DequantizeLinear
    # consumers -- slicing it for one chain would silently corrupt the
    # other's own use of the identical tensor, so it must be declined.
    K, H, Out = 8, 16, 4
    model, Wq, _Wscale, _Wzp, _W2 = _matmul_qdq_producer_model(K, H, Out, seed=14)
    extra_dq = onnx.helper.make_node(
        "DequantizeLinear", ["Wq", "Wscale"], ["Wdq2"], axis=1
    )
    extra_identity = onnx.helper.make_node("Identity", ["Wdq2"], ["Wdq2_out"])
    model.graph.node.extend([extra_dq, extra_identity])
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("Wdq2_out", onnx.TensorProto.FLOAT, [K, H])
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [K, H]  # untouched -- Wq is shared


def test_cpp_qdq_structured_pruning_both_sides_float_is_a_no_op():
    # A plain float/float chain never touches the QDQ pass's own matchers --
    # already exercised by ApplyChains above, confirmed here for the same
    # single combined entry point.
    model = _mlp_model(K=8, H=16, Out=4)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 8]  # halved by the plain-float pass


# --- QDQ: opset 21+ blockwise INT4/UINT4 ------------------------------------


def _matmul_qdq_blockwise_producer_model(
    K=8, H=16, Out=4, block_size=4, seed=0, zero_point=False, uint4=False
):
    rng = np.random.default_rng(seed)
    k_blocks = K // block_size
    lo, hi = (0, 16) if uint4 else (-8, 8)
    pack = _uint4 if uint4 else _int4
    Wq = rng.integers(lo, hi, size=(K, H)).astype(np.int8)
    Wscale = (
        np.abs(rng.standard_normal((k_blocks, H))).astype(np.float32) * 0.02 + 0.001
    )
    W2 = rng.standard_normal((H, Out)).astype(np.float32)
    initializer = [pack(Wq, "Wq"), _f32(Wscale, "Wscale"), _f32(W2, "W2")]
    zp_input = ""
    Wzp = None
    if zero_point:
        Wzp = rng.integers(lo, hi, size=(k_blocks, H)).astype(np.int8)
        initializer.append(pack(Wzp, "Wzp"))
        zp_input = ", Wzp"
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          Wdq = DequantizeLinear<axis=0, block_size={block_size}>(Wq, Wscale{zp_input})
          h = MatMul(X, Wdq)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=initializer,
    )
    return model, Wq, Wscale, Wzp, W2


def test_cpp_qdq_blockwise_int4_producer_matches_oracle():
    K, H, Out, block_size = 8, 16, 4, 4
    model, Wq, Wscale, _Wzp, W2 = _matmul_qdq_blockwise_producer_model(
        K, H, Out, block_size, seed=100
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [K, H // 2]
    assert list(inits["Wscale"].dims) == [K // block_size, H // 2]
    assert list(inits["W2"].dims) == [H // 2, Out]

    w_dequant = _qdq_block_dequant(Wq, Wscale, None, axis=0, block_size=block_size)
    keep = _oracle_keep_indices(w_dequant, H // 2)

    rng = np.random.default_rng(101)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run_unfused(pruned, {"X": x})
    h = x @ w_dequant[:, keep]
    y_oracle = h @ W2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)

    wq_pruned = onnx.numpy_helper.to_array(inits["Wq"]).astype(np.int8)
    wscale_pruned = onnx.numpy_helper.to_array(inits["Wscale"])
    np.testing.assert_array_equal(wq_pruned, Wq[:, keep])
    np.testing.assert_allclose(wscale_pruned, Wscale[:, keep])


def test_cpp_qdq_blockwise_uint4_producer_with_zero_point_matches_oracle():
    K, H, Out, block_size = 8, 16, 4, 4
    model, Wq, Wscale, Wzp, W2 = _matmul_qdq_blockwise_producer_model(
        K, H, Out, block_size, seed=102, zero_point=True, uint4=True
    )
    onnx.checker.check_model(model)
    assert Wzp is not None
    assert np.any(Wzp != 0)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wzp"].dims) == [K // block_size, H // 2]

    w_dequant = _qdq_block_dequant(Wq, Wscale, Wzp, axis=0, block_size=block_size)
    keep = _oracle_keep_indices(w_dequant, H // 2)

    rng = np.random.default_rng(103)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run_unfused(pruned, {"X": x})
    h = x @ w_dequant[:, keep]
    y_oracle = h @ W2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)

    wzp_pruned = onnx.numpy_helper.to_array(inits["Wzp"]).astype(np.uint8)
    np.testing.assert_array_equal(wzp_pruned, Wzp[:, keep])


def _block_grouped_column_weight(
    rng, rows, n_blocks, block_size, high_block_idx, lo=1.0, hi=6.0
):
    # A float weight whose column magnitude is constant WITHIN each
    # `block_size`-sized group of columns (large for every block index in
    # `high_block_idx`, small otherwise) -- so the top-L2-norm importance
    # ranking always drops/keeps whole blocks together, making a "consumer
    # block-aligned" test deterministic rather than relying on chance
    # alignment.
    cols = n_blocks * block_size
    base = rng.standard_normal((rows, cols)).astype(np.float64) * 0.1
    for b in range(n_blocks):
        mag = hi if b in high_block_idx else lo
        base[:, b * block_size : (b + 1) * block_size] += mag * np.sign(
            rng.standard_normal((rows, block_size))
        )
    return base.astype(np.float32)


def _matmul_qdq_blockwise_consumer_model(
    K, H, Out, block_size, seed=0, zero_point=False, uint4=False, W1=None
):
    rng = np.random.default_rng(seed)
    h_blocks = H // block_size
    lo, hi = (0, 16) if uint4 else (-8, 8)
    pack = _uint4 if uint4 else _int4
    if W1 is None:
        W1 = rng.standard_normal((K, H)).astype(np.float32)
    Wq = rng.integers(lo, hi, size=(H, Out)).astype(np.int8)
    Wscale = (
        np.abs(rng.standard_normal((h_blocks, Out))).astype(np.float32) * 0.02 + 0.001
    )
    initializer = [_f32(W1, "W1"), pack(Wq, "Wq"), _f32(Wscale, "Wscale")]
    zp_input = ""
    Wzp = None
    if zero_point:
        Wzp = rng.integers(lo, hi, size=(h_blocks, Out)).astype(np.int8)
        initializer.append(pack(Wzp, "Wzp"))
        zp_input = ", Wzp"
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          Wdq = DequantizeLinear<axis=0, block_size={block_size}>(Wq, Wscale{zp_input})
          Y = MatMul(h, Wdq)
        }}
        """,
        initializer=initializer,
    )
    return model, W1, Wq, Wscale, Wzp


def test_cpp_qdq_blockwise_consumer_block_aligned_matches_oracle():
    K, H, Out, block_size = 4, 16, 4, 4
    n_blocks = H // block_size
    rng = np.random.default_rng(104)
    W1 = _block_grouped_column_weight(
        rng, K, n_blocks, block_size, high_block_idx={0, 2}
    )
    model, W1, Wq, Wscale, _Wzp = _matmul_qdq_blockwise_consumer_model(
        K, H, Out, block_size, seed=105, W1=W1
    )
    onnx.checker.check_model(model)

    keep = _oracle_keep_indices(W1, H // 2)
    keep_blocks = sorted({int(k) // block_size for k in keep})
    assert keep_blocks == [0, 2]  # engineered, genuinely block-aligned

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [H // 2, Out]
    assert list(inits["Wscale"].dims) == [n_blocks // 2, Out]

    w_dequant = _qdq_block_dequant(Wq, Wscale, None, axis=0, block_size=block_size)
    rng2 = np.random.default_rng(106)
    x = rng2.standard_normal((5, K)).astype(np.float32)
    (y,) = _run_unfused(pruned, {"X": x})
    h = x @ W1[:, keep]
    y_oracle = h @ w_dequant[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)

    # Scale sliced by BLOCK index, not element -- confirms co-slicing
    # against the correct (coarser) axis.
    wscale_pruned = onnx.numpy_helper.to_array(inits["Wscale"])
    np.testing.assert_array_equal(wscale_pruned, Wscale[keep_blocks, :])


def test_cpp_qdq_blockwise_consumer_non_block_aligned_declines():
    # keep_count (6) is not a multiple of block_size (4) -- no whole-block
    # partition of a 16-element axis can ever produce exactly 6 kept
    # elements, so this chain's consumer is unconditionally non-block-
    # aligned; the real call must decline the whole chain.
    K, H, Out, block_size = 4, 16, 4, 4
    model, W1, Wq, _Wscale, _Wzp = _matmul_qdq_blockwise_consumer_model(
        K, H, Out, block_size, seed=107
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.625)  # keep 6 of 16
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H]  # left completely untouched
    assert list(inits["Wq"].dims) == [H, Out]
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(inits["W1"]), W1)
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["Wq"]).astype(np.int8), Wq
    )


def test_cpp_qdq_blockwise_conv_producer_matches_oracle():
    # Blockwise INT4 quantization on a Conv weight: axis=1 blocks the
    # in_channels axis, exercised as the producer's own reduction axis
    # while its OUTPUT channels (axis=0) are what gets pruned.
    Cin, Cmid, Cout, block_size = 8, 8, 4, 4
    cin_blocks = Cin // block_size
    rng = np.random.default_rng(112)
    Wq1 = rng.integers(-8, 8, size=(Cmid, Cin, 1, 1)).astype(np.int8)
    Wscale1 = (
        np.abs(rng.standard_normal((Cmid, cin_blocks, 1, 1))).astype(np.float32) * 0.02
        + 0.001
    )
    Wq2 = rng.integers(-100, 100, size=(Cout, Cmid, 1, 1)).astype(np.int8)
    Wscale2 = np.abs(rng.standard_normal(Cout)).astype(np.float32) * 0.02 + 0.001
    model = _model(
        f"""
        g (float[1,{Cin},4,4] X) => (float[1,{Cout},4,4] Y)
        {{
          W1dq = DequantizeLinear<axis=1, block_size={block_size}>(W1q, W1s)
          h = Conv(X, W1dq)
          W2dq = DequantizeLinear<axis=0>(W2q, W2s)
          Y = Conv(h, W2dq)
        }}
        """,
        initializer=[
            _int4(Wq1, "W1q"),
            _f32(Wscale1, "W1s"),
            _i8(Wq2, "W2q"),
            _f32(Wscale2, "W2s"),
        ],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1q"].dims) == [Cmid // 2, Cin, 1, 1]
    assert list(inits["W1s"].dims) == [Cmid // 2, cin_blocks, 1, 1]
    assert list(inits["W2q"].dims) == [Cout, Cmid // 2, 1, 1]

    w1_dequant = _qdq_block_dequant(Wq1, Wscale1, None, axis=1, block_size=block_size)
    w2_dequant = _qdq_dequant(Wq2, Wscale2, None, axis=0)
    w1_nk = w1_dequant.reshape(Cmid, -1)
    keep = _oracle_keep_indices(w1_nk.T, Cmid // 2)

    rng2 = np.random.default_rng(113)
    x = rng2.standard_normal((1, Cin, 4, 4)).astype(np.float32)
    (y,) = _run_unfused(pruned, {"X": x})

    ref_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Conv", ["X", "W1"], ["h"]),
                onnx.helper.make_node("Conv", ["h", "W2"], ["Y"]),
            ],
            "oracle",
            [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, None)],
            [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)],
            initializer=[
                onnx.numpy_helper.from_array(
                    w1_dequant[keep].astype(np.float32), name="W1"
                ),
                onnx.numpy_helper.from_array(
                    w2_dequant[:, keep].astype(np.float32), name="W2"
                ),
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 21)],
        ir_version=10,
    )
    (y_oracle,) = onnx.reference.ReferenceEvaluator(ref_model).run(None, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_qdq_blockwise_declines_non_exact_multiple_block_dim():
    # A reduction-axis dim that isn't an exact multiple of block_size (a
    # padded/partial final block) is out of scope -- declined by the
    # matcher itself, not just by the block-alignment check.
    K, H, Out, block_size = 6, 8, 4, 4  # K=6 not a multiple of block_size=4
    rng = np.random.default_rng(114)
    Wq = rng.integers(-8, 8, size=(K, H)).astype(np.int8)
    Wscale = np.abs(rng.standard_normal((2, H))).astype(np.float32) * 0.02 + 0.001
    # 2 blocks is wrong for K=6/block_size=4 (ceil is 2, but 6 % 4 != 0) --
    # deliberately mismatched/declined regardless of the scale shape chosen.
    W2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          Wdq = DequantizeLinear<axis=0, block_size={block_size}>(Wq, Wscale)
          h = MatMul(X, Wdq)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=[_int4(Wq, "Wq"), _f32(Wscale, "Wscale"), _f32(W2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [K, H]  # left completely untouched


# --- QDQ: gated (SwiGLU/GeGLU) pair ------------------------------------------


def _qdq_gated_mlp_model(K=8, H=16, Out=4, gate_activation="Sigmoid", seed=0):
    rng = np.random.default_rng(seed)
    Wgq = rng.integers(-100, 100, size=(K, H)).astype(np.int8)
    Wgscale = np.abs(rng.standard_normal(H)).astype(np.float32) * 0.02 + 0.001
    Wuq = rng.integers(-100, 100, size=(K, H)).astype(np.int8)
    Wuscale = np.abs(rng.standard_normal(H)).astype(np.float32) * 0.02 + 0.001
    Wd = rng.standard_normal((H, Out)).astype(np.float32)
    initializer = [
        _i8(Wgq, "Wgq"),
        _f32(Wgscale, "Wgscale"),
        _i8(Wuq, "Wuq"),
        _f32(Wuscale, "Wuscale"),
        _f32(Wd, "Wd"),
    ]
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          Wgdq = DequantizeLinear<axis=1>(Wgq, Wgscale)
          gate = MatMul(X, Wgdq)
          gate_act = {gate_activation}(gate)
          Wudq = DequantizeLinear<axis=1>(Wuq, Wuscale)
          up = MatMul(X, Wudq)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=initializer,
    )
    return model, Wgq, Wgscale, Wuq, Wuscale, Wd


def test_cpp_qdq_structured_pruning_gated_ffn_matches_oracle():
    K, H, Out = 8, 16, 4
    model, Wgq, Wgscale, Wuq, Wuscale, Wd = _qdq_gated_mlp_model(K, H, Out, seed=21)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wgq"].dims) == [K, H // 2]
    assert list(inits["Wgscale"].dims) == [H // 2]
    assert list(inits["Wuq"].dims) == [K, H // 2]
    assert list(inits["Wuscale"].dims) == [H // 2]
    assert list(inits["Wd"].dims) == [H // 2, Out]

    wg_dequant = _qdq_dequant(Wgq, Wgscale, None, axis=1)
    wu_dequant = _qdq_dequant(Wuq, Wuscale, None, axis=1)
    keep = _combined_keep_indices(wg_dequant, wu_dequant, H // 2)

    # Exact co-slice: both producers' own int8 codes AND scales sliced by
    # the identical `keep` set, never independently re-derived.
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["Wgq"]), Wgq[:, keep]
    )
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(inits["Wgscale"]), Wgscale[keep]
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["Wuq"]), Wuq[:, keep]
    )
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(inits["Wuscale"]), Wuscale[keep]
    )

    rng = np.random.default_rng(22)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run_unfused(pruned, {"X": x})

    gate = 1.0 / (1.0 + np.exp(-(x @ wg_dequant[:, keep])))
    up = x @ wu_dequant[:, keep]
    y_oracle = (gate * up) @ Wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_qdq_structured_pruning_gated_ffn_prunes_both_branches_to_same_channels():
    K, H, Out = 8, 20, 4
    model, Wgq, Wgscale, Wuq, Wuscale, _Wd = _qdq_gated_mlp_model(K, H, Out, seed=23)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.3)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}

    wg_dequant = _qdq_dequant(Wgq, Wgscale, None, axis=1)
    wu_dequant = _qdq_dequant(Wuq, Wuscale, None, axis=1)
    keep = _combined_keep_indices(wg_dequant, wu_dequant, H - round(H * 0.3))

    np.testing.assert_array_equal(inits["Wgq"], Wgq[:, keep])
    np.testing.assert_array_equal(inits["Wuq"], Wuq[:, keep])


def test_cpp_qdq_structured_pruning_gelu_gated_ffn_matches_oracle():
    # GeGLU: the gate activation is Gelu (tanh approximation) rather than
    # Sigmoid -- exercises FindQdqGatedChains's own reuse of the plain-float
    # gated matcher's unary-activation set.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(24)
    Wgq = rng.integers(-100, 100, size=(K, H)).astype(np.int8)
    Wgscale = np.abs(rng.standard_normal(H)).astype(np.float32) * 0.02 + 0.001
    Wuq = rng.integers(-100, 100, size=(K, H)).astype(np.int8)
    Wuscale = np.abs(rng.standard_normal(H)).astype(np.float32) * 0.02 + 0.001
    Wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          Wgdq = DequantizeLinear<axis=1>(Wgq, Wgscale)
          gate = MatMul(X, Wgdq)
          gate_act = Gelu<approximate = "tanh">(gate)
          Wudq = DequantizeLinear<axis=1>(Wuq, Wuscale)
          up = MatMul(X, Wudq)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[
            _i8(Wgq, "Wgq"),
            _f32(Wgscale, "Wgscale"),
            _i8(Wuq, "Wuq"),
            _f32(Wuscale, "Wuscale"),
            _f32(Wd, "Wd"),
        ],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    wg_dequant = _qdq_dequant(Wgq, Wgscale, None, axis=1)
    wu_dequant = _qdq_dequant(Wuq, Wuscale, None, axis=1)
    keep = _combined_keep_indices(wg_dequant, wu_dequant, H // 2)

    rng2 = np.random.default_rng(25)
    x = rng2.standard_normal((5, K)).astype(np.float32)
    (y,) = _run_unfused(pruned, {"X": x})

    def gelu_tanh(v):
        return 0.5 * v * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (v + 0.044715 * v**3)))

    gate = gelu_tanh(x @ wg_dequant[:, keep])
    up = x @ wu_dequant[:, keep]
    y_oracle = (gate * up) @ Wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-3, atol=1e-3)


# --- MatMulBnb4 (bitsandbytes FP4/NF4 block-quantized weight) structured ---
# --- pruning ---------------------------------------------------------------
#
# Tests for the MatMulBnb4 producer-only chain family
# (onnxsim/structured_pruning_entry.cpp's own "MatMulBnb4" section), mirroring
# tests/test_pruning.py's own MatMulBnb4 coverage (see that file's own
# `_bnb4_*` helpers, independently re-implemented here rather than imported --
# this file's own established "no cross-import between test files" precedent,
# same as the MatMulNBits section above). Fixture weights are quantized via
# the REAL ``onnxruntime.quantization.matmul_bnb4_quantizer.MatMulBnb4
# Quantizer``'s own ``bnb4_block_quant`` -- never a hand-rolled
# reimplementation of the packing scheme -- so every packed ``B``/``absmax``
# pair a test builds is genuine quantizer output, per CLAUDE.md's own
# guidance for a numpy-computed "large/packed" tensor. Models are built via
# onnx.parser (per CLAUDE.md's convention); the packed ``B``/``absmax``
# initializers are attached programmatically via onnx.numpy_helper.from_array
# afterward (CLAUDE.md's documented exception -- these must be byte-exact,
# not spelled out as text literals).


def _bnb4_quantize(w, quant_type, block_size):
    """Quantizes a real ``[K, N]`` float weight (ordinary ONNX MatMul weight
    convention: ``Y = A @ W``) via the actual onnxruntime quantizer code
    path -- returns ``(packed uint8 [(N*K+1)//2], absmax float32
    [(N*K+block_size-1)//block_size])``, byte-for-byte what a real
    ``MatMulBnb4Quantizer.process()`` call would emit for this weight.
    """
    from onnxruntime.quantization.matmul_bnb4_quantizer import MatMulBnb4Quantizer

    q = MatMulBnb4Quantizer.__new__(MatMulBnb4Quantizer)
    q.quant_type = quant_type
    q.block_size = block_size
    packed, absmax = q.bnb4_block_quant(w)
    return packed, absmax.astype(np.float32)


def _bnb4_model(K, N1, N2, block_size, quant_type, B1, absmax1, W2, activation="Relu"):
    """Builds ``A -> MatMulBnb4(mm1) -> activation -> MatMul(mm2) -> Y``, a
    real, bit-packed FP4/NF4-quantized ``com.microsoft::MatMulBnb4`` node
    feeding a plain-float ``MatMul`` consumer.
    """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        g (float[3,{K}] A) => (float[3,{N2}] Y)
        {{
          h1 = com.microsoft.MatMulBnb4 <K={K}, N={N1}, block_size={block_size}, quant_type={quant_type}> (A, B1, absmax1)
          h1_act = {activation}(h1)
          Y = MatMul(h1_act, W2)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(B1, name="B1"),
            onnx.numpy_helper.from_array(absmax1, name="absmax1"),
            _f32(W2, "W2"),
        ]
    )
    return model


def _bnb4_chain_model(K, N1, N2, block_size, quant_type, W1, W2, activation="Relu"):
    """Like :func:`_bnb4_model`, but takes W1 as a plain ``[K, N1]`` float
    weight and quantizes it itself (:func:`_bnb4_quantize`).
    """
    B1, absmax1 = _bnb4_quantize(W1, quant_type, block_size)
    return _bnb4_model(K, N1, N2, block_size, quant_type, B1, absmax1, W2, activation)


@pytest.mark.parametrize("quant_type", [0, 1])  # 0 = FP4, 1 = NF4
def test_cpp_structured_pruning_matmul_bnb4_plain_chain_matches_oracle(quant_type):
    # The core "slice, don't recompute" correctness invariant: the pruned
    # graph's own packed B/absmax tensors must equal a HAND-SLICE of the
    # ORIGINAL quantizer's own output -- byte-identical, not merely close --
    # confirmed both directly on the tensors and via a real InferenceSession
    # run against an independently-requantized-subset reference model
    # (mirrors tests/test_pruning.py's own
    # test_matmul_bnb4_pruning_producer_matches_independently_requantized_subset
    # and test_matmul_bnb4_pruning_producer_bytes_are_byte_identical_to_
    # requantized_subset).
    K, N1, N2, block_size = 64, 32, 8, 16
    rng = np.random.default_rng(31024)
    W1 = rng.uniform(-1, 1, size=(K, N1)).astype(np.float32)
    # Column-scale W1 so the top-16-by-L2-norm keep-set is unambiguous
    # (well separated from the dropped half) regardless of the random draw.
    W1 = W1 * np.linspace(0.1, 3.0, N1, dtype=np.float32)[None, :]
    W2 = rng.uniform(-1, 1, size=(N1, N2)).astype(np.float32)

    model = _bnb4_chain_model(K, N1, N2, block_size, quant_type, W1, W2)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    mm1 = next(n for n in pruned.graph.node if n.op_type == "MatMulBnb4")
    assert {a.name: a.i for a in mm1.attribute}["N"] == 16

    col_norms = np.linalg.norm(W1, axis=0)
    keep = np.sort(np.argsort(-col_norms, kind="stable")[:16])

    pruned_inits = {t.name: t for t in pruned.graph.initializer}
    B1_pruned = onnx.numpy_helper.to_array(pruned_inits[mm1.input[1]])
    absmax1_pruned = onnx.numpy_helper.to_array(pruned_inits[mm1.input[2]])
    B1_ref, absmax1_ref = _bnb4_quantize(W1[:, keep], quant_type, block_size)
    np.testing.assert_array_equal(B1_pruned, B1_ref)
    np.testing.assert_array_equal(absmax1_pruned, absmax1_ref)

    ref_model = _bnb4_chain_model(
        K, len(keep), N2, block_size, quant_type, W1[:, keep], W2[keep, :]
    )
    A = rng.uniform(-1, 1, size=(3, K)).astype(np.float32)
    (pruned_out,) = _run(pruned, {"A": A})
    (ref_out,) = _run(ref_model, {"A": A})
    # A bnb4 producer-row-slice is a plain byte-range copy of the original
    # quantizer's own output (every row's own packed bytes/absmax blocks
    # depend only on that row's own K values, never on which other rows
    # exist or how many there are) -- so this is an exact equality check,
    # not a float-tolerance one.
    np.testing.assert_array_equal(pruned_out, ref_out)


def test_cpp_structured_pruning_matmul_bnb4_declines_k_not_multiple_of_block_size():
    # K=24, block_size=16: 24 % 16 != 0, so block 1 ([16, 32) in the
    # flattened [N, K] array) straddles the row-0/row-1 boundary -- the whole
    # chain (producer and consumer) must be left byte-unchanged. Mirrors
    # tests/test_pruning.py's own
    # test_matmul_bnb4_pruning_declines_k_not_multiple_of_block_size.
    K, N1, N2, block_size = 24, 16, 8, 16
    rng = np.random.default_rng(31009)
    W1 = rng.uniform(-1, 1, size=(K, N1)).astype(np.float32)
    W2 = rng.uniform(-1, 1, size=(N1, N2)).astype(np.float32)
    model = _bnb4_chain_model(K, N1, N2, block_size, 1, W1, W2)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_matmul_bnb4_declines_transb_false():
    # transB=0 ("backward pass" orientation) is unverified semantics --
    # always declined, model left untouched.
    K, N1, N2, block_size = 64, 32, 8, 16
    rng = np.random.default_rng(31003)
    W1 = rng.uniform(-1, 1, size=(K, N1)).astype(np.float32)
    B1, absmax1 = _bnb4_quantize(W1, 1, block_size)
    W2 = rng.uniform(-1, 1, size=(N1, N2)).astype(np.float32)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        g (float[3,{K}] A) => (float[3,{N2}] Y)
        {{
          h1 = com.microsoft.MatMulBnb4 <K={K}, N={N1}, block_size={block_size}, quant_type=1, transB=0> (A, B1, absmax1)
          Y = MatMul(h1, W2)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(B1, name="B1"),
            onnx.numpy_helper.from_array(absmax1, name="absmax1"),
            _f32(W2, "W2"),
        ]
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_matmul_bnb4_declines_shared_weight():
    # The same B tensor read by two different MatMulBnb4 nodes -- slicing it
    # for one chain would silently corrupt the other reader.
    K, N1, N2, block_size = 64, 32, 8, 16
    rng = np.random.default_rng(31014)
    W1 = rng.uniform(-1, 1, size=(K, N1)).astype(np.float32)
    W2 = rng.uniform(-1, 1, size=(N1, N2)).astype(np.float32)
    model = _bnb4_chain_model(K, N1, N2, block_size, 1, W1, W2)

    extra = onnx.helper.make_node(
        "MatMulBnb4",
        inputs=["A", "B1", "absmax1"],
        outputs=["Y2"],
        domain="com.microsoft",
        name="mm_extra",
        K=K,
        N=N1,
        block_size=block_size,
        quant_type=1,
    )
    model.graph.node.append(extra)
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("Y2", onnx.TensorProto.FLOAT, [3, N1])
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_matmul_bnb4_zero_sparsity_is_a_no_op():
    K, N1, N2, block_size = 64, 32, 8, 16
    rng = np.random.default_rng(31005)
    W1 = rng.uniform(-1, 1, size=(K, N1)).astype(np.float32)
    W2 = rng.uniform(-1, 1, size=(N1, N2)).astype(np.float32)
    model = _bnb4_chain_model(K, N1, N2, block_size, 1, W1, W2)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.0)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_matmul_bnb4_matches_python_reference_output():
    K, N1, N2, block_size = 64, 32, 8, 16
    rng = np.random.default_rng(31060)
    W1 = rng.uniform(-1, 1, size=(K, N1)).astype(np.float32)
    W1 = W1 * np.linspace(0.1, 3.0, N1, dtype=np.float32)[None, :]
    W2 = rng.uniform(-1, 1, size=(N1, N2)).astype(np.float32)
    model = _bnb4_chain_model(K, N1, N2, block_size, 1, W1, W2)
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_structured_pruning_matmul_bnb4(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    rng2 = np.random.default_rng(31061)
    A = rng2.uniform(-1, 1, size=(3, K)).astype(np.float32)
    (y_py,) = _run(pruned_py, {"A": A})
    (y_cpp,) = _run(pruned_cpp, {"A": A})
    np.testing.assert_array_equal(y_py, y_cpp)


def test_cpp_structured_pruning_matmul_bnb4_separate_bias_add_hop_matches_oracle():
    # Regression test for the per-channel bias/scale `Add`/`Mul`
    # pass-through hop newly ported into WalkToMatMulBnb4Consumer: a
    # SEPARATE trailing bias `Add` node between the `MatMulBnb4` producer
    # and the plain-float `MatMul` consumer -- the ONLY way a biased
    # `MatMulBnb4` layer is representable at all, since the op's own live
    # 3-input schema (`A, B, absmax`) has no bias slot of its own. Before
    # this hop was ported, the walk would decline at the `Add` node
    # entirely, leaving the producer byte-unchanged; with it, the chain is
    # found, the bias is co-sliced, and `N` is still pruned exactly as
    # test_cpp_structured_pruning_matmul_bnb4_plain_chain_matches_oracle
    # confirms for the no-bias case.
    K, N1, N2, block_size, quant_type = 64, 32, 8, 16, 1
    rng = np.random.default_rng(31200)
    W1 = rng.uniform(-1, 1, size=(K, N1)).astype(np.float32)
    # Column-scale W1 so the top-16-by-L2-norm keep-set is unambiguous.
    W1 = W1 * np.linspace(0.1, 3.0, N1, dtype=np.float32)[None, :]
    W2 = rng.uniform(-1, 1, size=(N1, N2)).astype(np.float32)
    bias1 = (rng.standard_normal(N1) * 0.05).astype(np.float32)
    B1, absmax1 = _bnb4_quantize(W1, quant_type, block_size)

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        g (float[3,{K}] A) => (float[3,{N2}] Y)
        {{
          h1 = com.microsoft.MatMulBnb4 <K={K}, N={N1}, block_size={block_size}, quant_type={quant_type}> (A, B1, absmax1)
          hb = Add(h1, Bias1)
          h1a = Relu(hb)
          Y = MatMul(h1a, W2)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(B1, name="B1"),
            onnx.numpy_helper.from_array(absmax1, name="absmax1"),
            _f32(W2, "W2"),
            _f32(bias1, "Bias1"),
        ]
    )
    onnx.checker.check_model(model)

    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_structured_pruning_matmul_bnb4(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)

    mm1 = next(n for n in pruned_cpp.graph.node if n.op_type == "MatMulBnb4")
    assert {a.name: a.i for a in mm1.attribute}["N"] == 16

    col_norms = np.linalg.norm(W1, axis=0)
    keep = np.sort(np.argsort(-col_norms, kind="stable")[:16])

    pruned_inits = {t.name: t for t in pruned_cpp.graph.initializer}
    B1_pruned = onnx.numpy_helper.to_array(pruned_inits[mm1.input[1]])
    absmax1_pruned = onnx.numpy_helper.to_array(pruned_inits[mm1.input[2]])
    B1_ref, absmax1_ref = _bnb4_quantize(W1[:, keep], quant_type, block_size)
    np.testing.assert_array_equal(B1_pruned, B1_ref)
    np.testing.assert_array_equal(absmax1_pruned, absmax1_ref)
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(pruned_inits["Bias1"]), bias1[keep]
    )

    # Byte-for-byte parity with the (already-correct) Python reference.
    py_bytes = {t.name: t.SerializeToString() for t in pruned_py.graph.initializer}
    cpp_bytes = {t.name: t.SerializeToString() for t in pruned_cpp.graph.initializer}
    assert py_bytes == cpp_bytes

    ref_model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        g (float[3,{K}] A) => (float[3,{N2}] Y)
        {{
          h1 = com.microsoft.MatMulBnb4 <K={K}, N={len(keep)}, block_size={block_size}, quant_type={quant_type}> (A, B1r, absmax1r)
          hb = Add(h1, Bias1r)
          h1a = Relu(hb)
          Y = MatMul(h1a, W2r)
        }}
        """
    )
    ref_model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(B1_ref, name="B1r"),
            onnx.numpy_helper.from_array(absmax1_ref, name="absmax1r"),
            _f32(W2[keep, :], "W2r"),
            _f32(bias1[keep], "Bias1r"),
        ]
    )
    A = rng.uniform(-1, 1, size=(3, K)).astype(np.float32)
    (pruned_out,) = _run(pruned_cpp, {"A": A})
    (ref_out,) = _run(ref_model, {"A": A})
    np.testing.assert_array_equal(pruned_out, ref_out)


def test_cpp_structured_pruning_matmul_bnb4_separate_bias_add_tied_declines():
    # Bias1 is ALSO read by a second, unrelated Add -- the tied/shared-
    # tensor bar this hop's own C++ comment documents: slicing Bias1 in
    # place would silently corrupt that second reader, so the walk must
    # decline the Add hop (and hence the whole chain) entirely.
    K, N1, N2, block_size, quant_type = 64, 32, 8, 16, 1
    rng = np.random.default_rng(31210)
    W1 = rng.uniform(-1, 1, size=(K, N1)).astype(np.float32)
    W2 = rng.uniform(-1, 1, size=(N1, N2)).astype(np.float32)
    bias1 = (rng.standard_normal(N1) * 0.05).astype(np.float32)
    B1, absmax1 = _bnb4_quantize(W1, quant_type, block_size)

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        g (float[3,{K}] A, float[3,{N1}] B) => (float[3,{N2}] Y, float[3,{N1}] Y2)
        {{
          h1 = com.microsoft.MatMulBnb4 <K={K}, N={N1}, block_size={block_size}, quant_type={quant_type}> (A, B1, absmax1)
          hb = Add(h1, Bias1)
          h1a = Relu(hb)
          Y = MatMul(h1a, W2)
          Y2 = Add(B, Bias1)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(B1, name="B1"),
            onnx.numpy_helper.from_array(absmax1, name="absmax1"),
            _f32(W2, "W2"),
            _f32(bias1, "Bias1"),
        ]
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


# --- MatMulBlockQuantizedFp4Weight/MatMulBlockQuantizedFp8Weight -----------
# --- (NVFP4/FP8 block-quantized weight) structured pruning ----------------
#
# Tests for the Fp8Weight/Fp4Weight plain chain families
# (onnxsim/structured_pruning_entry.cpp's own "MatMulBlockQuantizedFp4Weight/
# MatMulBlockQuantizedFp8Weight" section), mirroring tests/test_pruning.py's
# own coverage for both ops (see that file's own `_fp8bw_*`/`_fp4bw_*`
# helpers, independently re-implemented here rather than imported -- this
# file's own established "no cross-import between test files" precedent).
# Models are built via onnx.parser (per CLAUDE.md's convention); the packed
# FLOAT8E4M3FN/uint8 `B`/scale-table initializers are attached
# programmatically via onnx.numpy_helper.from_array afterward (CLAUDE.md's
# documented exception -- these must be byte-exact, not spelled out as text
# literals). Neither op has a real CPU kernel in this environment (see
# onnxsim/pruning.py's own section comment for the empirical confirmation --
# the same fact holds for this C++ port, which links the identical
# onnxruntime), so every "oracle" test below uses the same decomposed-proxy-
# topology methodology tests/test_pruning.py's own equivalent tests
# establish: the pruned node's own packed tensors are dequantized BY HAND
# (independently of onnxsim's own code) into a plain float32 weight, wired
# into an equivalent plain Gemm/MatMul chain, and run through a REAL CPU
# onnxruntime.InferenceSession against an independently-computed numpy
# oracle.


def _fp8bq_quantize(w, block_size):
    """Independent (onnxsim-code-free) per-block E4M3 quantizer for one
    ``MatMulBlockQuantizedFp8Weight``-style ``[N, K]`` float weight -- scale
    chosen as ``amax(block) / 6.0``, a plausible representation (there is no
    real kernel/reference quantizer to match, see this section's own top
    comment). Returns ``(b_f8 [N, K] ml_dtypes.float8_e4m3fn, b_scale
    [N, K // block_size] float32, dequant [N, K] float64)``.
    """
    n, k = w.shape
    kb = k // block_size
    w_blocks = w.reshape(n, kb, block_size).astype(np.float64)
    amax = np.maximum(np.abs(w_blocks).max(axis=-1), 1e-8)
    scale = (amax / 6.0).astype(np.float32)
    scale_expanded = np.repeat(scale.astype(np.float64), block_size, axis=-1)
    b_f8 = (w_blocks / scale_expanded.reshape(n, kb, block_size)).astype(
        ml_dtypes.float8_e4m3fn
    )
    dequant = b_f8.astype(np.float64) * scale_expanded.reshape(n, kb, block_size)
    return b_f8.reshape(n, k), scale, dequant.reshape(n, k)


def _fp8bq_dequantize(b_f8, scale, block_size):
    """Independent dequantizer -- inverse of :func:`_fp8bq_quantize`'s own
    formula, used to re-dequantize the PRUNED node's own surviving tensors
    (never the pass's own dequantize code) for the decomposed-proxy oracle.
    """
    n, k = b_f8.shape
    scale_expanded = np.repeat(scale.astype(np.float64), block_size, axis=-1)
    return b_f8.astype(np.float64) * scale_expanded


_FP4BQ_E2M1_LUT = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float64,
)


def _fp4bq_e2m1_encode_nearest(values):
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(-1)
    codes = np.argmin(np.abs(_FP4BQ_E2M1_LUT[None, :] - flat[:, None]), axis=1)
    return codes.reshape(values.shape).astype(np.uint8)


def _fp4bq_e2m1_decode(codes):
    return _FP4BQ_E2M1_LUT[np.asarray(codes).astype(np.int64)]


def _fp4bq_unpack_nibbles(packed, count):
    """Independent inverse of :func:`_nbits_pack_nibbles` (defined above, in
    this same file's own "MatMulNBits" section -- reused here unchanged,
    same low-nibble-first convention, not a cross-file import): 2 codes per
    byte, LOW nibble first. Drops the last, padding, half-byte when `count`
    is odd.
    """
    packed = np.asarray(packed)
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    interleaved = np.stack([lo, hi], axis=-1).reshape(*packed.shape[:-1], -1)
    return interleaved[..., :count].astype(np.uint8)


def _fp4bq_quantize(w, global_scale, block_size):
    """Independent (onnxsim-code-free) NVFP4 quantizer for one
    ``MatMulBlockQuantizedFp4Weight``-style ``[N, K]`` float weight --
    per-`block_size` E2M1 magnitude/sign code plus a ``float8e4m3fn`` block
    scale, combined with a caller-supplied scalar `global_scale`. Returns
    ``(packed uint8 [N, K/2] -- 2 codes/byte, low nibble first, flat across
    the whole K axis, weight_scale uint8 [N, k_blocks] -- raw E4M3 bytes,
    dequant [N, K] float64)``. Reuses :func:`_nbits_pack_nibbles` (defined
    above in this same file) for the packing -- identical convention to
    `Fp4Weight`'s own live schema, per onnxsim/pruning.py's own section
    comment.
    """
    n, k = w.shape
    kb = k // block_size
    w_blocks = w.reshape(n, kb, block_size).astype(np.float64)
    amax = np.maximum(np.abs(w_blocks).max(axis=-1), 1e-12)
    block_scale_f64 = amax / (6.0 * global_scale)  # 6.0 == E2M1's own max magnitude
    block_scale_f8 = block_scale_f64.astype(ml_dtypes.float8_e4m3fn)
    block_scale_rt = block_scale_f8.astype(np.float64)
    codes = _fp4bq_e2m1_encode_nearest(
        w_blocks / (block_scale_rt[..., None] * global_scale)
    )
    packed = _nbits_pack_nibbles(codes.reshape(n, k))
    dequant = (
        _fp4bq_e2m1_decode(codes) * block_scale_rt[..., None] * global_scale
    ).reshape(n, k)
    return packed, block_scale_f8.view(np.uint8), dequant


def _fp8bq_chain_model(N1, K1, N2, block_size, W1, W2):
    """Builds ``A -> MatMulBlockQuantizedFp8Weight(mm1) -> Relu ->
    MatMulBlockQuantizedFp8Weight(mm2) -> Y``, both nodes real. Returns
    ``(model, info)`` where `info` carries the independently-quantized
    artifacts needed to hand-build an oracle.
    """
    b1, s1, dq1 = _fp8bq_quantize(W1, block_size)
    b2, s2, dq2 = _fp8bq_quantize(W2, block_size)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        g (float16[3,{K1}] A) => (float16[3,{N2}] Y)
        {{
          h1 = com.microsoft.MatMulBlockQuantizedFp8Weight <block_size={block_size}> (A, B1, S1)
          h1a = Relu(h1)
          Y = com.microsoft.MatMulBlockQuantizedFp8Weight <block_size={block_size}> (h1a, B2, S2)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(b1, name="B1"),
            _f32(s1, "S1"),
            onnx.numpy_helper.from_array(b2, name="B2"),
            _f32(s2, "S2"),
        ]
    )
    return model, dict(b1=b1, s1=s1, dequant1=dq1, b2=b2, s2=s2, dequant2=dq2)


def test_cpp_structured_pruning_matmul_block_quantized_fp8_chain_matches_oracle():
    # The core "slice, don't recompute" correctness invariant: the pruned
    # graph's own B/b_scale tensors must equal a byte-exact hand-slice of
    # the ORIGINAL quantized codes, and running an equivalent decomposed
    # plain-float Gemm chain (built from the PRUNED tensors' own dequantized
    # values) through a real CPU onnxruntime session matches an
    # independently-computed numpy oracle.
    N1, K1, N2, block_size = 16, 32, 8, 8
    rng = np.random.default_rng(41001)
    W1 = (rng.standard_normal((N1, K1)) * 0.2).astype(np.float32)
    W1[:8] *= 6.0  # first half of rows clearly more important
    W2 = (rng.standard_normal((N2, N1)) * 0.2).astype(np.float32)
    model, info = _fp8bq_chain_model(N1, K1, N2, block_size, W1, W2)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = np.sort(np.argsort(-np.abs(info["dequant1"]).sum(axis=1))[:8])
    assert list(keep) == list(range(8))  # engineered: rows 0-7 are important

    inits = {t.name: t for t in pruned.graph.initializer}
    b1p = onnx.numpy_helper.to_array(inits["B1"])
    s1p = onnx.numpy_helper.to_array(inits["S1"])
    b2p = onnx.numpy_helper.to_array(inits["B2"])
    assert b1p.shape == (8, K1)
    assert s1p.shape == (8, K1 // block_size)
    assert b2p.shape == (N2, 8)

    np.testing.assert_array_equal(b1p.view(np.uint8), info["b1"].view(np.uint8)[keep])
    np.testing.assert_array_equal(s1p, info["s1"][keep])
    np.testing.assert_array_equal(
        b2p.view(np.uint8), info["b2"].view(np.uint8)[:, keep]
    )

    proxy_graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Gemm", ["A", "W1f"], ["T"], transB=1),
            onnx.helper.make_node("Relu", ["T"], ["Ta"]),
            onnx.helper.make_node("Gemm", ["Ta", "W2f"], ["Y"], transB=1),
        ],
        "g",
        [onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [3, K1])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3, N2])],
        initializer=[
            _f32(info["dequant1"][keep].astype(np.float32), "W1f"),
            _f32(info["dequant2"][:, keep].astype(np.float32), "W2f"),
        ],
    )
    proxy_model = onnx.helper.make_model(
        proxy_graph, opset_imports=[onnx.helper.make_opsetid("", 21)]
    )
    proxy_model.ir_version = 10
    onnx.checker.check_model(proxy_model)

    a = rng.standard_normal((3, K1)).astype(np.float32)
    sess = ort.InferenceSession(
        proxy_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y,) = sess.run(None, {"A": a})
    oracle = (
        np.maximum(a.astype(np.float64) @ info["dequant1"][keep].T, 0.0)
        @ info["dequant2"][:, keep].T
    )
    np.testing.assert_allclose(y.astype(np.float64), oracle, rtol=1e-3, atol=1e-3)


def test_cpp_structured_pruning_matmul_block_quantized_fp8_declines_non_block_aligned():
    # block_size=8, sparsity=0.25 -> keep_count=12, not a multiple of 8 --
    # the consumer's own scale table can't represent a partial block, so the
    # whole chain must be left byte-for-byte unchanged.
    N1, K1, N2, block_size = 16, 32, 8, 8
    rng = np.random.default_rng(41010)
    W1 = (rng.standard_normal((N1, K1)) * 0.3).astype(np.float32)
    W2 = (rng.standard_normal((N2, N1)) * 0.3).astype(np.float32)
    model, _info = _fp8bq_chain_model(N1, K1, N2, block_size, W1, W2)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.25)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_matmul_block_quantized_fp8_declines_shared_weight():
    # B1/S1 read by both the plain chain's own producer node AND an extra
    # node -- slicing them for the chain would silently corrupt that extra
    # reader, so the whole model must be left untouched.
    N1, K1, N2, block_size = 16, 32, 8, 8
    rng = np.random.default_rng(41020)
    W1 = (rng.standard_normal((N1, K1)) * 0.3).astype(np.float32)
    W2 = (rng.standard_normal((N2, N1)) * 0.3).astype(np.float32)
    model, _info = _fp8bq_chain_model(N1, K1, N2, block_size, W1, W2)

    extra = onnx.helper.make_node(
        "MatMulBlockQuantizedFp8Weight",
        inputs=["A", "B1", "S1"],
        outputs=["Y2"],
        domain="com.microsoft",
        name="mm_extra",
        block_size=block_size,
    )
    model.graph.node.append(extra)
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("Y2", onnx.TensorProto.FLOAT16, [3, N1])
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_matmul_block_quantized_fp8_matches_python_reference():
    N1, K1, N2, block_size = 16, 32, 8, 8
    rng = np.random.default_rng(41030)
    W1 = (rng.standard_normal((N1, K1)) * 0.2).astype(np.float32)
    W1[:8] *= 6.0
    W2 = (rng.standard_normal((N2, N1)) * 0.2).astype(np.float32)
    model, _info = _fp8bq_chain_model(N1, K1, N2, block_size, W1, W2)
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_structured_pruning_matmul_block_quantized_fp8(
        model, sparsity=0.5
    )
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    py_bytes = {t.name: t.SerializeToString() for t in pruned_py.graph.initializer}
    cpp_bytes = {t.name: t.SerializeToString() for t in pruned_cpp.graph.initializer}
    assert py_bytes == cpp_bytes


def test_cpp_structured_pruning_matmul_block_quantized_fp8_separate_bias_add_hop_matches_python_reference():
    # Regression test for the per-channel bias/scale `Add`/`Mul`
    # pass-through hop newly ported into WalkToFp8Consumer: a separate bias
    # `Add` node between the first `MatMulBlockQuantizedFp8Weight` producer
    # and the second consumer. Before this hop was ported, the walk would
    # decline at the `Add` node entirely, leaving both nodes byte-unchanged;
    # with it, the chain is found and both N-axes are pruned. As
    # pruning.py's own comment on this exact hop notes, it was "mechanically
    # mirrored from the MatMulNBits/Bnb4 fix, not independently verified
    # against a live FP8 quantizer" (neither `MatMulBlockQuantizedFp8Weight`
    # nor `Add(float16, float32)` has a real CPU kernel/strict type-checker
    # opinion in this environment either way -- see this section's own top
    # comment), so byte-for-byte parity with that Python reference
    # implementation -- not a real onnxruntime kernel run -- is the
    # correctness bar this test applies, mirroring the plain-chain
    # matches_python_reference test above.
    N1, K1, N2, block_size = 16, 32, 8, 8
    rng = np.random.default_rng(41200)
    W1 = (rng.standard_normal((N1, K1)) * 0.2).astype(np.float32)
    W1[:8] *= 6.0
    W2 = (rng.standard_normal((N2, N1)) * 0.2).astype(np.float32)
    bias1 = (rng.standard_normal(N1) * 0.05).astype(np.float32)
    b1, s1, _dq1 = _fp8bq_quantize(W1, block_size)
    b2, s2, _dq2 = _fp8bq_quantize(W2, block_size)

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        g (float16[3,{K1}] A) => (float16[3,{N2}] Y)
        {{
          h1 = com.microsoft.MatMulBlockQuantizedFp8Weight <block_size={block_size}> (A, B1, S1)
          hb = Add(h1, Bias1)
          h1a = Relu(hb)
          Y = com.microsoft.MatMulBlockQuantizedFp8Weight <block_size={block_size}> (h1a, B2, S2)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(b1, name="B1"),
            _f32(s1, "S1"),
            _f32(bias1, "Bias1"),
            onnx.numpy_helper.from_array(b2, name="B2"),
            _f32(s2, "S2"),
        ]
    )
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_structured_pruning_matmul_block_quantized_fp8(
        model, sparsity=0.5
    )
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    # The chain must actually be found and pruned (not silently declined).
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    assert onnx.numpy_helper.to_array(inits["B1"]).shape == (8, K1)
    assert onnx.numpy_helper.to_array(inits["Bias1"]).shape == (8,)
    np.testing.assert_allclose(onnx.numpy_helper.to_array(inits["Bias1"]), bias1[:8])

    py_bytes = {t.name: t.SerializeToString() for t in pruned_py.graph.initializer}
    cpp_bytes = {t.name: t.SerializeToString() for t in pruned_cpp.graph.initializer}
    assert py_bytes == cpp_bytes


def test_cpp_structured_pruning_matmul_block_quantized_fp8_separate_bias_add_tied_declines():
    # Bias1 is ALSO read by a second, unrelated Add -- the tied/shared-
    # tensor bar this hop's own C++ comment documents: slicing Bias1 in
    # place would silently corrupt that second reader, so the walk must
    # decline the Add hop (and hence the whole chain) entirely.
    N1, K1, N2, block_size = 16, 32, 8, 8
    rng = np.random.default_rng(41210)
    W1 = (rng.standard_normal((N1, K1)) * 0.2).astype(np.float32)
    W2 = (rng.standard_normal((N2, N1)) * 0.2).astype(np.float32)
    bias1 = (rng.standard_normal(N1) * 0.05).astype(np.float32)
    b1, s1, _dq1 = _fp8bq_quantize(W1, block_size)
    b2, s2, _dq2 = _fp8bq_quantize(W2, block_size)

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        g (float16[3,{K1}] A, float16[3,{N1}] B) => (float16[3,{N2}] Y, float16[3,{N1}] Y2)
        {{
          h1 = com.microsoft.MatMulBlockQuantizedFp8Weight <block_size={block_size}> (A, B1, S1)
          hb = Add(h1, Bias1)
          h1a = Relu(hb)
          Y = com.microsoft.MatMulBlockQuantizedFp8Weight <block_size={block_size}> (h1a, B2, S2)
          Y2 = Add(B, Bias1)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(b1, name="B1"),
            _f32(s1, "S1"),
            _f32(bias1, "Bias1"),
            onnx.numpy_helper.from_array(b2, name="B2"),
            _f32(s2, "S2"),
        ]
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def _fp4bq_chain_model(N1, K1, N2, block_size, W1, gs1, W2, gs2):
    """``Fp4Weight`` analogue of :func:`_fp8bq_chain_model`."""
    p1, ws1, dq1 = _fp4bq_quantize(W1, gs1, block_size)
    p2, ws2, dq2 = _fp4bq_quantize(W2, gs2, block_size)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        g (float16[3,{K1}] A) => (float16[3,{N2}] Y)
        {{
          h1 = com.microsoft.MatMulBlockQuantizedFp4Weight <block_size={block_size}> (A, B1, WS1, WS21)
          h1a = Relu(h1)
          Y = com.microsoft.MatMulBlockQuantizedFp4Weight <block_size={block_size}> (h1a, B2, WS2, WS22)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(p1, name="B1"),
            onnx.numpy_helper.from_array(ws1, name="WS1"),
            _f32(np.array([gs1], dtype=np.float32), "WS21"),
            onnx.numpy_helper.from_array(p2, name="B2"),
            onnx.numpy_helper.from_array(ws2, name="WS2"),
            _f32(np.array([gs2], dtype=np.float32), "WS22"),
        ]
    )
    return model, dict(p1=p1, ws1=ws1, dequant1=dq1, p2=p2, ws2=ws2, dequant2=dq2)


def test_cpp_structured_pruning_matmul_block_quantized_fp4_chain_matches_oracle():
    # Same "slice, don't recompute" invariant as the Fp8Weight test above --
    # here the byte-identity check on B needs an unpack/repack (Fp4Weight's
    # own flat, whole-row nibble packing, not block-relative like
    # MatMulNBits's), which this test checks explicitly.
    N1, K1, N2, block_size = 16, 32, 8, 8
    rng = np.random.default_rng(41101)
    W1 = (rng.standard_normal((N1, K1)) * 1.0).astype(np.float32)
    W1[:8] *= 6.0
    W2 = (rng.standard_normal((N2, N1)) * 1.0).astype(np.float32)
    model, info = _fp4bq_chain_model(N1, K1, N2, block_size, W1, 1.0, W2, 1.0)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = np.sort(np.argsort(-np.abs(info["dequant1"]).sum(axis=1))[:8])
    assert list(keep) == list(range(8))

    inits = {t.name: t for t in pruned.graph.initializer}
    b1p = onnx.numpy_helper.to_array(inits["B1"])
    ws1p = onnx.numpy_helper.to_array(inits["WS1"])
    b2p = onnx.numpy_helper.to_array(inits["B2"])
    assert b1p.shape == (8, K1 // 2)
    assert ws1p.shape == (8, K1 // block_size)
    assert b2p.shape == (N2, 8 // 2)  # K axis co-pruned: 8 kept K-values -> 4 bytes

    # B1 (producer role) is a plain whole-row byte slice -- no unpack needed.
    np.testing.assert_array_equal(b1p, info["p1"][keep])
    np.testing.assert_array_equal(ws1p, info["ws1"][keep])

    # B2's own kept COLUMNS need an unpack/repack (flat-across-K packing).
    orig_codes2 = _fp4bq_unpack_nibbles(info["p2"], N1)
    expect_b2 = _nbits_pack_nibbles(orig_codes2[:, keep])
    np.testing.assert_array_equal(b2p, expect_b2)

    proxy_graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Gemm", ["A", "W1f"], ["T"], transB=1),
            onnx.helper.make_node("Relu", ["T"], ["Ta"]),
            onnx.helper.make_node("Gemm", ["Ta", "W2f"], ["Y"], transB=1),
        ],
        "g",
        [onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [3, K1])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3, N2])],
        initializer=[
            _f32(info["dequant1"][keep].astype(np.float32), "W1f"),
            _f32(info["dequant2"][:, keep].astype(np.float32), "W2f"),
        ],
    )
    proxy_model = onnx.helper.make_model(
        proxy_graph, opset_imports=[onnx.helper.make_opsetid("", 21)]
    )
    proxy_model.ir_version = 10
    onnx.checker.check_model(proxy_model)

    a = rng.standard_normal((3, K1)).astype(np.float32)
    sess = ort.InferenceSession(
        proxy_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y,) = sess.run(None, {"A": a})
    oracle = (
        np.maximum(a.astype(np.float64) @ info["dequant1"][keep].T, 0.0)
        @ info["dequant2"][:, keep].T
    )
    np.testing.assert_allclose(y.astype(np.float64), oracle, rtol=1e-3, atol=1e-3)


def test_cpp_structured_pruning_matmul_block_quantized_fp4_declines_non_block_aligned():
    N1, K1, N2, block_size = 16, 32, 8, 8
    rng = np.random.default_rng(41110)
    W1 = (rng.standard_normal((N1, K1)) * 1.0).astype(np.float32)
    W2 = (rng.standard_normal((N2, N1)) * 1.0).astype(np.float32)
    model, _info = _fp4bq_chain_model(N1, K1, N2, block_size, W1, 1.0, W2, 1.0)
    onnx.checker.check_model(model)

    # sparsity=0.25 -> keep_count=12, not a multiple of block_size=8.
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.25)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_matmul_block_quantized_fp4_declines_odd_resulting_k():
    # block_size=3 (odd), N1=12 -> k_blocks=4. Engineered so the top-9-by-
    # importance keep-set is EXACTLY channels {0..8} -- blocks {0, 1, 2}
    # fully kept, block 3 fully dropped: perfectly block-aligned by the
    # SHARED (MatMulNBits-style) check, but the resulting K (9) is odd,
    # which Fp4Weight's own flat (non-block-relative) nibble packing cannot
    # honestly represent (the live schema derives K as exactly
    # 2 * B.shape[1]) -- declined on top of the shared check, per this
    # section's own top comment.
    N1, K1, N2, block_size = 12, 15, 4, 3  # K1=15 divisible by block_size=3 too
    rng = np.random.default_rng(41120)
    W1 = (rng.standard_normal((N1, K1)) * 0.2).astype(np.float32)
    W1[:9] *= 6.0  # channels 0-8 important, 9-11 unimportant
    W2 = (rng.standard_normal((N2, N1)) * 0.2).astype(np.float32)
    model, info = _fp4bq_chain_model(N1, K1, N2, block_size, W1, 1.0, W2, 1.0)
    onnx.checker.check_model(model)

    keep = np.sort(np.argsort(-np.abs(info["dequant1"]).sum(axis=1))[:9])
    assert list(keep) == list(range(9))  # exactly blocks {0, 1, 2} of block_size=3

    # sparsity=0.25 -> keep_count = 12 - round(12*0.25) = 9.
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.25)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_matmul_block_quantized_fp4_declines_shared_weight():
    N1, K1, N2, block_size = 16, 32, 8, 8
    rng = np.random.default_rng(41130)
    W1 = (rng.standard_normal((N1, K1)) * 1.0).astype(np.float32)
    W2 = (rng.standard_normal((N2, N1)) * 1.0).astype(np.float32)
    model, _info = _fp4bq_chain_model(N1, K1, N2, block_size, W1, 1.0, W2, 1.0)

    extra = onnx.helper.make_node(
        "MatMulBlockQuantizedFp4Weight",
        inputs=["A", "B1", "WS1", "WS21"],
        outputs=["Y2"],
        domain="com.microsoft",
        name="mm_extra",
        block_size=block_size,
    )
    model.graph.node.append(extra)
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("Y2", onnx.TensorProto.FLOAT16, [3, N1])
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_matmul_block_quantized_fp4_mixed_plain_float_producer_matches_oracle():
    # A plain-float (directly-constant weight) MatMul producer feeding a
    # quantized Fp4Weight consumer -- the "unquantized embedding feeding the
    # first quantized block" export shape this section's own top comment
    # motivates (MatchPlainMatMulNBitsPeer, reused unchanged from the
    # MatMulNBits section). The plain-float peer side of THIS C++ port only
    # ever recognizes a FLOAT32 weight (see the MatMulNBits section's own
    # top comment for why), so A/W1 are declared FLOAT32 here -- the
    # quantized consumer's own `A` input dtype is never actually inspected
    # by the matcher (see this section's own top comment), so this is not a
    # scope violation of the live schema, just this test's own choice of a
    # graph that both the plain MatMul op (real ai.onnx schema, dtype-
    # checked by onnx.checker) and the quantized consumer accept.
    N1, K1, N2, block_size = 16, 32, 8, 8
    rng = np.random.default_rng(41140)
    W1 = (rng.standard_normal((K1, N1)) * 0.3).astype(
        np.float32
    )  # [K, N], untransposed
    W1[:, :8] *= 6.0  # first half of N1 (producer's own output axis) important
    W2 = (rng.standard_normal((N2, N1)) * 1.0).astype(np.float32)
    p2, ws2, dq2 = _fp4bq_quantize(W2, 1.0, block_size)

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        g (float[3,{K1}] A) => (float16[3,{N2}] Y)
        {{
          T = MatMul(A, W1)
          Y = com.microsoft.MatMulBlockQuantizedFp4Weight <block_size={block_size}> (T, B2, WS2, WS22)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            _f32(W1, "W1"),
            onnx.numpy_helper.from_array(p2, name="B2"),
            onnx.numpy_helper.from_array(ws2, name="WS2"),
            _f32(np.array([1.0], dtype=np.float32), "WS22"),
        ]
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    w1p = onnx.numpy_helper.to_array(inits["W1"])
    b2p = onnx.numpy_helper.to_array(inits["B2"])
    assert w1p.shape == (K1, 8)  # plain-float producer sliced on axis 1 (col)
    assert b2p.shape == (N2, 4)  # consumer's own K axis co-pruned (8/2 nibbles)

    keep = np.sort(np.argsort(-np.linalg.norm(W1.astype(np.float64), axis=0))[:8])
    assert list(keep) == list(range(8))
    np.testing.assert_array_equal(
        w1p.astype(np.float64), W1.astype(np.float64)[:, keep]
    )

    orig_codes2 = _fp4bq_unpack_nibbles(p2, N1)
    expect_b2 = _nbits_pack_nibbles(orig_codes2[:, keep])
    np.testing.assert_array_equal(b2p, expect_b2)


def test_cpp_structured_pruning_matmul_block_quantized_fp4_matches_python_reference():
    N1, K1, N2, block_size = 16, 32, 8, 8
    rng = np.random.default_rng(41150)
    W1 = (rng.standard_normal((N1, K1)) * 1.0).astype(np.float32)
    W1[:8] *= 6.0
    W2 = (rng.standard_normal((N2, N1)) * 1.0).astype(np.float32)
    model, _info = _fp4bq_chain_model(N1, K1, N2, block_size, W1, 1.0, W2, 1.0)
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_structured_pruning_matmul_block_quantized_fp4(
        model, sparsity=0.5
    )
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    py_bytes = {t.name: t.SerializeToString() for t in pruned_py.graph.initializer}
    cpp_bytes = {t.name: t.SerializeToString() for t in pruned_cpp.graph.initializer}
    assert py_bytes == cpp_bytes


def test_cpp_structured_pruning_matmul_block_quantized_fp4_separate_bias_add_hop_matches_python_reference():
    # `Fp4Weight` analogue of
    # test_cpp_structured_pruning_matmul_block_quantized_fp8_separate_bias_
    # add_hop_matches_python_reference -- see that test's own docstring for
    # why byte-for-byte parity with the Python reference implementation
    # (rather than a real onnxruntime kernel run) is the correctness bar
    # here too.
    N1, K1, N2, block_size = 16, 32, 8, 8
    rng = np.random.default_rng(41250)
    W1 = (rng.standard_normal((N1, K1)) * 1.0).astype(np.float32)
    W1[:8] *= 6.0
    W2 = (rng.standard_normal((N2, N1)) * 1.0).astype(np.float32)
    bias1 = (rng.standard_normal(N1) * 0.05).astype(np.float32)
    p1, ws1, _dq1 = _fp4bq_quantize(W1, 1.0, block_size)
    p2, ws2, _dq2 = _fp4bq_quantize(W2, 1.0, block_size)

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        g (float16[3,{K1}] A) => (float16[3,{N2}] Y)
        {{
          h1 = com.microsoft.MatMulBlockQuantizedFp4Weight <block_size={block_size}> (A, B1, WS1, WS21)
          hb = Add(h1, Bias1)
          h1a = Relu(hb)
          Y = com.microsoft.MatMulBlockQuantizedFp4Weight <block_size={block_size}> (h1a, B2, WS2, WS22)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(p1, name="B1"),
            onnx.numpy_helper.from_array(ws1, name="WS1"),
            _f32(np.array([1.0], dtype=np.float32), "WS21"),
            _f32(bias1, "Bias1"),
            onnx.numpy_helper.from_array(p2, name="B2"),
            onnx.numpy_helper.from_array(ws2, name="WS2"),
            _f32(np.array([1.0], dtype=np.float32), "WS22"),
        ]
    )
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_structured_pruning_matmul_block_quantized_fp4(
        model, sparsity=0.5
    )
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    # The chain must actually be found and pruned (not silently declined).
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    assert onnx.numpy_helper.to_array(inits["B1"]).shape == (8, K1 // 2)
    assert onnx.numpy_helper.to_array(inits["Bias1"]).shape == (8,)
    np.testing.assert_allclose(onnx.numpy_helper.to_array(inits["Bias1"]), bias1[:8])

    py_bytes = {t.name: t.SerializeToString() for t in pruned_py.graph.initializer}
    cpp_bytes = {t.name: t.SerializeToString() for t in pruned_cpp.graph.initializer}
    assert py_bytes == cpp_bytes


def test_cpp_structured_pruning_matmul_block_quantized_fp4_separate_bias_add_tied_declines():
    # Bias1 is ALSO read by a second, unrelated Add -- the tied/shared-
    # tensor bar this hop's own C++ comment documents: slicing Bias1 in
    # place would silently corrupt that second reader, so the walk must
    # decline the Add hop (and hence the whole chain) entirely.
    N1, K1, N2, block_size = 16, 32, 8, 8
    rng = np.random.default_rng(41260)
    W1 = (rng.standard_normal((N1, K1)) * 1.0).astype(np.float32)
    W2 = (rng.standard_normal((N2, N1)) * 1.0).astype(np.float32)
    bias1 = (rng.standard_normal(N1) * 0.05).astype(np.float32)
    p1, ws1, _dq1 = _fp4bq_quantize(W1, 1.0, block_size)
    p2, ws2, _dq2 = _fp4bq_quantize(W2, 1.0, block_size)

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        g (float16[3,{K1}] A, float16[3,{N1}] B) => (float16[3,{N2}] Y, float16[3,{N1}] Y2)
        {{
          h1 = com.microsoft.MatMulBlockQuantizedFp4Weight <block_size={block_size}> (A, B1, WS1, WS21)
          hb = Add(h1, Bias1)
          h1a = Relu(hb)
          Y = com.microsoft.MatMulBlockQuantizedFp4Weight <block_size={block_size}> (h1a, B2, WS2, WS22)
          Y2 = Add(B, Bias1)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(p1, name="B1"),
            onnx.numpy_helper.from_array(ws1, name="WS1"),
            _f32(np.array([1.0], dtype=np.float32), "WS21"),
            _f32(bias1, "Bias1"),
            onnx.numpy_helper.from_array(p2, name="B2"),
            onnx.numpy_helper.from_array(ws2, name="WS2"),
            _f32(np.array([1.0], dtype=np.float32), "WS22"),
        ]
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


# --- QOperator (QLinearConv/QLinearMatMul/QGemm static-quantization) -------
# --- structured pruning -----------------------------------------------------
#
# Mirrors ``tests/test_pruning.py``'s own "QOperator (QLinearConv/
# QLinearMatMul/QGemm) structured pruning tests" coverage for
# ``onnxsim.apply_structured_pruning_cpp`` -- the C++ port also runs these
# QOperator chains (see ``onnxsim/structured_pruning_entry.cpp``'s own
# "QOperator (QLinearConv/QLinearMatMul/QGemm static-quantization) structured
# pruning" section comment), wired into the SAME entry point as the
# plain-float/QDQ/etc. chains above (unlike ``onnxsim.pruning``, which keeps
# ``apply_structured_pruning_qoperator`` a separate top-level function).
# QLinearConv/QLinearMatMul/QGemm carry their own quantized weight/scale/
# zero-point/bias directly as node inputs -- there is no DequantizeLinear
# node to build through parser text the way the QDQ tests above do, and the
# scale/zero-point/int32-bias tensors need real int8/uint8/int32 array data a
# parser text literal can't spell out -- so, mirroring this file's own QDQ/
# MatMulNBits/Fp4/Fp8 test helpers above (and ``test_pruning.py``'s own
# identical choice for this same section), these models are built directly
# via ``onnx.helper``, with real numpy-quantized tensors attached as
# initializers.


def _u8(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.uint8), name)


def _i32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.int32), name)


def _qop_quantize_per_channel_i8(w, axis):
    """Independent (onnxsim-code-free) symmetric per-channel INT8
    quantizer -- `axis` is the output-channel axis. Returns
    ``(q, scale, zero_point)`` with `zero_point` all-zero (symmetric),
    matching this repo's own ``quantize_qoperator``/``quantize_qoperator_gemm``
    convention (see ``onnxsim/pruning.py``'s own "QOperator" section
    comment).
    """
    absmax = np.max(np.abs(w), axis=tuple(i for i in range(w.ndim) if i != axis))
    absmax = np.maximum(absmax, 1e-6)
    scale = (absmax / 127.0).astype(np.float32)
    shp = [1] * w.ndim
    shp[axis] = -1
    q = np.clip(np.round(w / scale.reshape(shp)), -127, 127).astype(np.int8)
    zp = np.zeros_like(scale, dtype=np.int8)
    return q, scale, zp


def _qop_dequant(q, scale, zero_point, axis):
    q = q.astype(np.float64)
    scale = scale.astype(np.float64)
    zp = zero_point.astype(np.float64)
    if scale.ndim >= 1 and scale.size > 1:
        shape = [1] * q.ndim
        shape[axis] = -1
        scale = scale.reshape(shape)
        zp = zp.reshape(shape)
    return (q - zp) * scale


def _qlinearconv_node(
    name,
    x,
    y,
    w,
    w_scale,
    w_zp,
    y_scale,
    y_zp,
    bias,
    x_scale="x_scale",
    x_zp="x_zp",
    **attrs,
):
    inputs = [x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp]
    if bias is not None:
        inputs.append(bias)
    return onnx.helper.make_node("QLinearConv", inputs, [y], name=name, **attrs)


def _qlinearconv_chain_model(
    N1=6, C1=3, N2=4, kh=3, kw=3, bias=True, per_channel=True, seed=0
):
    """Builds ``QLinearConv(n1) -> QLinearConv(n2)`` -- a same-family
    QOperator chain -- with real, independently-quantized weights. Returns
    ``(model, info)`` where `info` carries every quantized array needed to
    hand-build an "already pruned" oracle.
    """
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((N1, C1, kh, kw)).astype(np.float32) * 0.3
    W2 = rng.standard_normal((N2, N1, 1, 1)).astype(np.float32) * 0.3
    B1f = (rng.standard_normal(N1).astype(np.float32) * 0.05) if bias else None
    B2f = (rng.standard_normal(N2).astype(np.float32) * 0.05) if bias else None

    x_scale = np.float32(0.02)
    x_zp = np.uint8(120)

    if per_channel:
        W1q, W1s, W1zp = _qop_quantize_per_channel_i8(W1, axis=0)
        W2q, W2s, W2zp = _qop_quantize_per_channel_i8(W2, axis=0)
    else:
        w1q_flat, w1s_flat, w1zp_flat = _qop_quantize_per_channel_i8(
            W1.reshape(1, -1), axis=0
        )
        W1q, W1s, W1zp = w1q_flat.reshape(W1.shape), w1s_flat[0], w1zp_flat[0]
        w2q_flat, w2s_flat, w2zp_flat = _qop_quantize_per_channel_i8(
            W2.reshape(1, -1), axis=0
        )
        W2q, W2s, W2zp = w2q_flat.reshape(W2.shape), w2s_flat[0], w2zp_flat[0]

    y1_scale = np.float32(0.03)
    y1_zp = np.uint8(120)
    y2_scale = np.float32(0.04)
    y2_zp = np.uint8(120)

    B1q = None
    if B1f is not None:
        b1_dequant_scale = x_scale.astype(np.float64) * W1s.astype(np.float64)
        B1q = np.round(B1f.astype(np.float64) / b1_dequant_scale).astype(np.int32)
    B2q = None
    if B2f is not None:
        b2_dequant_scale = y1_scale.astype(np.float64) * W2s.astype(np.float64)
        B2q = np.round(B2f.astype(np.float64) / b2_dequant_scale).astype(np.int32)

    inits = [
        _f32(np.array(x_scale), "x_scale"),
        _u8(np.array(x_zp), "x_zp"),
        onnx.numpy_helper.from_array(W1q, "W1"),
        _f32(np.atleast_1d(W1s) if per_channel else np.array(W1s), "W1_scale"),
        _i8(np.atleast_1d(W1zp) if per_channel else np.array(W1zp), "W1_zp"),
        _f32(np.array(y1_scale), "y1_scale"),
        _u8(np.array(y1_zp), "y1_zp"),
        onnx.numpy_helper.from_array(W2q, "W2"),
        _f32(np.atleast_1d(W2s) if per_channel else np.array(W2s), "W2_scale"),
        _i8(np.atleast_1d(W2zp) if per_channel else np.array(W2zp), "W2_zp"),
        _f32(np.array(y2_scale), "y2_scale"),
        _u8(np.array(y2_zp), "y2_zp"),
    ]
    if B1q is not None:
        inits.append(_i32(B1q, "B1"))
    if B2q is not None:
        inits.append(_i32(B2q, "B2"))

    n1 = _qlinearconv_node(
        "n1",
        "x_q",
        "y1_q",
        "W1",
        "W1_scale",
        "W1_zp",
        "y1_scale",
        "y1_zp",
        "B1" if B1q is not None else None,
        kernel_shape=[kh, kw],
        pads=[0, 0, 0, 0],
    )
    n2 = _qlinearconv_node(
        "n2",
        "y1_q",
        "y2_q",
        "W2",
        "W2_scale",
        "W2_zp",
        "y2_scale",
        "y2_zp",
        "B2" if B2q is not None else None,
        x_scale="y1_scale",
        x_zp="y1_zp",
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
    )
    spatial = 8
    out_spatial = spatial - kh + 1
    graph = onnx.helper.make_graph(
        [n1, n2],
        "g",
        [
            onnx.helper.make_tensor_value_info(
                "x_q", onnx.TensorProto.UINT8, [1, C1, spatial, spatial]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                "y2_q", onnx.TensorProto.UINT8, [1, N2, out_spatial, out_spatial]
            )
        ],
        initializer=inits,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 10
    return model, {
        "W1": W1q,
        "W1_scale": W1s,
        "W1_zp": W1zp,
        "B1": B1q,
        "W2": W2q,
        "W2_scale": W2s,
        "W2_zp": W2zp,
        "B2": B2q,
        "x_scale": x_scale,
        "spatial": spatial,
        "C1": C1,
    }


def test_cpp_qlinearconv_chain_matches_oracle_via_real_inference_session():
    # Byte-identity ("slice, don't recompute") AND a real onnxruntime
    # end-to-end run against an independently reconstructed "already pruned"
    # reference model, per this task's own correctness bar.
    N1, C1, N2 = 6, 3, 4
    model, info = _qlinearconv_chain_model(N1=N1, C1=C1, N2=N2, seed=2)
    onnx.checker.check_model(model)

    w1_dequant = _qop_dequant(info["W1"], info["W1_scale"], info["W1_zp"], axis=0)
    keep = _oracle_keep_indices_conv(w1_dequant, N1 // 2)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [N1 // 2, C1, 3, 3]
    assert list(inits["W1_scale"].dims) == [N1 // 2]
    assert list(inits["W1_zp"].dims) == [N1 // 2]
    assert list(inits["B1"].dims) == [N1 // 2]
    assert list(inits["W2"].dims) == [N2, N1 // 2, 1, 1]

    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["W1"]), info["W1"][keep]
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["W1_scale"]), info["W1_scale"][keep]
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["W1_zp"]), info["W1_zp"][keep]
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["B1"]), info["B1"][keep]
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["W2"]), info["W2"][:, keep]
    )

    ref_model, _ = _qlinearconv_chain_model(N1=len(keep), C1=C1, N2=N2, seed=1234567)
    ref_inits = {t.name: t for t in ref_model.graph.initializer}
    ref_inits["W1"].CopyFrom(onnx.numpy_helper.from_array(info["W1"][keep], "W1"))
    ref_inits["W1_scale"].CopyFrom(
        onnx.numpy_helper.from_array(info["W1_scale"][keep], "W1_scale")
    )
    ref_inits["W1_zp"].CopyFrom(
        onnx.numpy_helper.from_array(info["W1_zp"][keep], "W1_zp")
    )
    ref_inits["B1"].CopyFrom(onnx.numpy_helper.from_array(info["B1"][keep], "B1"))
    ref_inits["W2"].CopyFrom(onnx.numpy_helper.from_array(info["W2"][:, keep], "W2"))
    ref_inits["W2_scale"].CopyFrom(
        onnx.numpy_helper.from_array(info["W2_scale"], "W2_scale")
    )
    ref_inits["W2_zp"].CopyFrom(onnx.numpy_helper.from_array(info["W2_zp"], "W2_zp"))
    ref_inits["B2"].CopyFrom(onnx.numpy_helper.from_array(info["B2"], "B2"))

    rng = np.random.default_rng(11)
    spatial = info["spatial"]
    xq = rng.integers(0, 255, size=(1, C1, spatial, spatial)).astype(np.uint8)
    (y_pruned,) = _run(pruned, {"x_q": xq})
    (y_ref,) = _run(ref_model, {"x_q": xq})
    np.testing.assert_array_equal(y_pruned, y_ref)


def test_cpp_qlinearconv_per_tensor_scale_left_untouched():
    model, _info = _qlinearconv_chain_model(N1=6, C1=3, N2=4, per_channel=False, seed=3)
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    # per-tensor scale/zero-point are scalars -- never sliced, regardless of
    # channel count.
    assert list(inits["W1_scale"].dims) == []
    assert list(inits["W1_zp"].dims) == []
    assert list(inits["W1"].dims) == [3, 3, 3, 3]
    assert list(inits["B1"].dims) == [3]  # bias always co-sliced regardless


def test_cpp_qlinearconv_declines_grouped_conv():
    model, _info = _qlinearconv_chain_model(N1=6, C1=3, N2=4, seed=4)
    for node in model.graph.node:
        if node.name == "n1":
            node.attribute.append(onnx.helper.make_attribute("group", 3))
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    # nothing changes -- no matched chain to prune
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [6, 3, 3, 3]


def test_cpp_qlinearconv_declines_shared_weight():
    model, _info = _qlinearconv_chain_model(N1=6, C1=3, N2=4, seed=6)
    # a second, unrelated node also reads W1 -- shared/tied, can't be sliced.
    extra = onnx.helper.make_node("Identity", ["W1"], ["W1_alias"], name="alias")
    model.graph.node.append(extra)
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(
            "W1_alias", onnx.TensorProto.INT8, [6, 3, 3, 3]
        )
    )
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [6, 3, 3, 3]


# --- QLinearMatMul --------------------------------------------------------


def _qlinearmatmul_node(
    name, a, y, b, b_scale, b_zp, y_scale, y_zp, a_scale="a_scale", a_zp="a_zp"
):
    return onnx.helper.make_node(
        "QLinearMatMul",
        [a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp],
        [y],
        name=name,
    )


def _qlinearmatmul_chain_model(K=8, H=6, Out=4, seed=0):
    """Builds ``QLinearMatMul(n1) -> QLinearMatMul(n2)`` -- both real,
    per-channel INT8 weights, matching this repo's own
    ``quantize_qoperator`` convention (`b` stored ``[K, N]``, `b_scale`/
    `b_zero_point` per-column).
    """
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((K, H)).astype(np.float32) * 0.3
    W2 = rng.standard_normal((H, Out)).astype(np.float32) * 0.3
    W1q, W1s, W1zp = _qop_quantize_per_channel_i8(W1, axis=1)
    W2q, W2s, W2zp = _qop_quantize_per_channel_i8(W2, axis=1)

    a_scale = np.float32(0.02)
    a_zp = np.uint8(120)
    y1_scale = np.float32(0.03)
    y1_zp = np.uint8(120)
    y2_scale = np.float32(0.04)
    y2_zp = np.uint8(120)

    inits = [
        _f32(np.array(a_scale), "a_scale"),
        _u8(np.array(a_zp), "a_zp"),
        onnx.numpy_helper.from_array(W1q, "W1"),
        _f32(W1s, "W1_scale"),
        _i8(W1zp, "W1_zp"),
        _f32(np.array(y1_scale), "y1_scale"),
        _u8(np.array(y1_zp), "y1_zp"),
        onnx.numpy_helper.from_array(W2q, "W2"),
        _f32(W2s, "W2_scale"),
        _i8(W2zp, "W2_zp"),
        _f32(np.array(y2_scale), "y2_scale"),
        _u8(np.array(y2_zp), "y2_zp"),
    ]
    n1 = _qlinearmatmul_node(
        "n1", "x_q", "y1_q", "W1", "W1_scale", "W1_zp", "y1_scale", "y1_zp"
    )
    n2 = _qlinearmatmul_node(
        "n2",
        "y1_q",
        "y2_q",
        "W2",
        "W2_scale",
        "W2_zp",
        "y2_scale",
        "y2_zp",
        a_scale="y1_scale",
        a_zp="y1_zp",
    )
    graph = onnx.helper.make_graph(
        [n1, n2],
        "g",
        [onnx.helper.make_tensor_value_info("x_q", onnx.TensorProto.UINT8, [2, K])],
        [onnx.helper.make_tensor_value_info("y2_q", onnx.TensorProto.UINT8, [2, Out])],
        initializer=inits,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 21)]
    )
    model.ir_version = 10
    return model, {
        "W1": W1q,
        "W1_scale": W1s,
        "W1_zp": W1zp,
        "W2": W2q,
        "W2_scale": W2s,
        "W2_zp": W2zp,
    }


def test_cpp_qlinearmatmul_chain_matches_oracle():
    K, H, Out = 8, 6, 4
    model, info = _qlinearmatmul_chain_model(K, H, Out, seed=10)
    onnx.checker.check_model(model)

    w1_dequant = _qop_dequant(info["W1"], info["W1_scale"], info["W1_zp"], axis=1)
    keep = _oracle_keep_indices(w1_dequant, H // 2)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["W1"]), info["W1"][:, keep]
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["W1_scale"]), info["W1_scale"][keep]
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["W2"]), info["W2"][keep, :]
    )

    ref_model, _ = _qlinearmatmul_chain_model(K, len(keep), Out, seed=999)
    ref_inits = {t.name: t for t in ref_model.graph.initializer}
    ref_inits["W1"].CopyFrom(onnx.numpy_helper.from_array(info["W1"][:, keep], "W1"))
    ref_inits["W1_scale"].CopyFrom(
        onnx.numpy_helper.from_array(info["W1_scale"][keep], "W1_scale")
    )
    ref_inits["W1_zp"].CopyFrom(
        onnx.numpy_helper.from_array(info["W1_zp"][keep], "W1_zp")
    )
    ref_inits["W2"].CopyFrom(onnx.numpy_helper.from_array(info["W2"][keep, :], "W2"))
    ref_inits["W2_scale"].CopyFrom(
        onnx.numpy_helper.from_array(info["W2_scale"], "W2_scale")
    )
    ref_inits["W2_zp"].CopyFrom(onnx.numpy_helper.from_array(info["W2_zp"], "W2_zp"))

    rng = np.random.default_rng(12)
    xq = rng.integers(0, 255, size=(2, K)).astype(np.uint8)
    (y_pruned,) = _run(pruned, {"x_q": xq})
    (y_ref,) = _run(ref_model, {"x_q": xq})
    np.testing.assert_array_equal(y_pruned, y_ref)


def test_cpp_qlinearmatmul_declines_shared_weight():
    model, _info = _qlinearmatmul_chain_model(K=8, H=6, Out=4, seed=13)
    extra = onnx.helper.make_node("Identity", ["W1"], ["W1_alias"], name="alias")
    model.graph.node.append(extra)
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("W1_alias", onnx.TensorProto.INT8, [8, 6])
    )
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 6]


def test_cpp_qlinearmatmul_declines_flatten_crossing_conv_producer():
    # The real onnxruntime.quantization.quantize_static(quant_format=
    # QOperator) round trip this section's own top comment (see
    # onnxsim/pruning.py) reproduces emits exactly QLinearConv -> Flatten ->
    # QLinearMatMul -- confirmed NOT matched here (same-family-only walk,
    # mirroring the QDQ section's own identical, pre-existing gap for
    # Flatten-crossing chains).
    conv_model, conv_info = _qlinearconv_chain_model(N1=6, C1=3, N2=4, seed=20)
    n1 = [n for n in conv_model.graph.node if n.name == "n1"][0]
    keep_inits = [
        t
        for t in conv_model.graph.initializer
        if t.name
        in ("x_scale", "x_zp", "W1", "W1_scale", "W1_zp", "y1_scale", "y1_zp", "B1")
    ]
    spatial = conv_info["spatial"]
    out_spatial = spatial - 3 + 1
    flat_dim = 6 * out_spatial * out_spatial
    Kmm, Out = flat_dim, 4
    rng = np.random.default_rng(21)
    Wmm = rng.standard_normal((Kmm, Out)).astype(np.float32) * 0.3
    Wmmq, Wmms, Wmmzp = _qop_quantize_per_channel_i8(Wmm, axis=1)
    flatten = onnx.helper.make_node("Flatten", ["y1_q"], ["flat_q"], axis=1)
    mm = _qlinearmatmul_node(
        "mm",
        "flat_q",
        "y_q",
        "Wmm",
        "Wmm_scale",
        "Wmm_zp",
        "y2_scale",
        "y2_zp",
        a_scale="y1_scale",
        a_zp="y1_zp",
    )
    inits = list(keep_inits) + [
        onnx.numpy_helper.from_array(Wmmq, "Wmm"),
        _f32(Wmms, "Wmm_scale"),
        _i8(Wmmzp, "Wmm_zp"),
        _f32(np.array(0.05, dtype=np.float32), "y2_scale"),
        _u8(np.array(120, dtype=np.uint8), "y2_zp"),
    ]
    graph = onnx.helper.make_graph(
        [n1, flatten, mm],
        "g",
        [
            onnx.helper.make_tensor_value_info(
                "x_q", onnx.TensorProto.UINT8, [1, 3, spatial, spatial]
            )
        ],
        [onnx.helper.make_tensor_value_info("y_q", onnx.TensorProto.UINT8, [1, Out])],
        initializer=inits,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 10
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    # nothing changes -- no matched chain to prune (no Flatten-crossing hop).
    inits_after = {t.name: t for t in pruned.graph.initializer}
    assert list(inits_after["W1"].dims) == [6, 3, 3, 3]


# --- QGemm ------------------------------------------------------------


def _qgemm_node(name, a, y, b, b_scale, b_zp, c, y_scale, y_zp, **attrs):
    inputs = [a, "a_scale", "a_zp", b, b_scale, b_zp]
    inputs.append(c or "")
    inputs.append(y_scale or "")
    inputs.append(y_zp or "")
    while inputs and not inputs[-1]:
        inputs.pop()
    return onnx.helper.make_node(
        "QGemm", inputs, [y], domain="com.microsoft", name=name, **attrs
    )


def _qgemm_chain_model(K=8, H=6, Out=4, transB1=0, transB2=0, bias=True, seed=0):
    rng = np.random.default_rng(seed)
    W1f = rng.standard_normal((H, K) if transB1 else (K, H)).astype(np.float32) * 0.3
    W2f = (
        rng.standard_normal((Out, H) if transB2 else (H, Out)).astype(np.float32) * 0.3
    )
    axis1 = 0 if transB1 else 1
    axis2 = 0 if transB2 else 1
    W1q, W1s, W1zp = _qop_quantize_per_channel_i8(W1f, axis=axis1)
    W2q, W2s, W2zp = _qop_quantize_per_channel_i8(W2f, axis=axis2)

    a_scale = np.float32(0.02)
    a_zp = np.uint8(120)
    y1_scale = np.float32(0.03)
    y1_zp = np.uint8(120)
    y2_scale = np.float32(0.04)
    y2_zp = np.uint8(120)

    C1f = (rng.standard_normal(H).astype(np.float32) * 0.05) if bias else None
    C2f = (rng.standard_normal(Out).astype(np.float32) * 0.05) if bias else None
    C1q = None
    if C1f is not None:
        c1_scale = a_scale.astype(np.float64) * W1s.astype(np.float64)
        C1q = np.round(C1f.astype(np.float64) / c1_scale).astype(np.int32)
    C2q = None
    if C2f is not None:
        c2_scale = y1_scale.astype(np.float64) * W2s.astype(np.float64)
        C2q = np.round(C2f.astype(np.float64) / c2_scale).astype(np.int32)

    inits = [
        _f32(np.array(a_scale), "a_scale"),
        _u8(np.array(a_zp), "a_zp"),
        onnx.numpy_helper.from_array(W1q, "W1"),
        _f32(W1s, "W1_scale"),
        _i8(W1zp, "W1_zp"),
        _f32(np.array(y1_scale), "y1_scale"),
        _u8(np.array(y1_zp), "y1_zp"),
        onnx.numpy_helper.from_array(W2q, "W2"),
        _f32(W2s, "W2_scale"),
        _i8(W2zp, "W2_zp"),
        _f32(np.array(y2_scale), "y2_scale"),
        _u8(np.array(y2_zp), "y2_zp"),
    ]
    if C1q is not None:
        inits.append(_i32(C1q, "C1"))
    if C2q is not None:
        inits.append(_i32(C2q, "C2"))

    n1 = _qgemm_node(
        "n1",
        "x_q",
        "y1_q",
        "W1",
        "W1_scale",
        "W1_zp",
        "C1" if C1q is not None else None,
        "y1_scale",
        "y1_zp",
        transB=transB1,
    )
    n2 = _qgemm_node(
        "n2",
        "y1_q",
        "y2_q",
        "W2",
        "W2_scale",
        "W2_zp",
        "C2" if C2q is not None else None,
        "y2_scale",
        "y2_zp",
        transB=transB2,
    )
    n2.input[1] = "y1_scale"
    n2.input[2] = "y1_zp"
    graph = onnx.helper.make_graph(
        [n1, n2],
        "g",
        [onnx.helper.make_tensor_value_info("x_q", onnx.TensorProto.UINT8, [2, K])],
        [onnx.helper.make_tensor_value_info("y2_q", onnx.TensorProto.UINT8, [2, Out])],
        initializer=inits,
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    return model, {
        "W1": W1q,
        "W1_scale": W1s,
        "W1_zp": W1zp,
        "C1": C1q,
        "W2": W2q,
        "W2_scale": W2s,
        "W2_zp": W2zp,
        "C2": C2q,
    }


@pytest.mark.parametrize("transB1,transB2", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_cpp_qgemm_chain_transb_matches_oracle(transB1, transB2):
    K, H, Out = 8, 6, 4
    model, info = _qgemm_chain_model(
        K, H, Out, transB1=transB1, transB2=transB2, seed=30
    )
    onnx.checker.check_model(model)

    axis1 = 0 if transB1 else 1
    w1_dequant = _qop_dequant(info["W1"], info["W1_scale"], info["W1_zp"], axis=axis1)
    w1_nk = w1_dequant if transB1 else w1_dequant.T
    keep = np.sort(np.argsort(-np.linalg.norm(w1_nk, axis=1))[: H // 2])

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}

    w1_expected = info["W1"][keep, :] if transB1 else info["W1"][:, keep]
    w2_expected = info["W2"][:, keep] if transB2 else info["W2"][keep, :]
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(inits["W1"]), w1_expected)
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["W1_scale"]), info["W1_scale"][keep]
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["C1"]), info["C1"][keep]
    )
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(inits["W2"]), w2_expected)

    ref_model, _ = _qgemm_chain_model(
        K, len(keep), Out, transB1=transB1, transB2=transB2, seed=555
    )
    ref_inits = {t.name: t for t in ref_model.graph.initializer}
    ref_inits["W1"].CopyFrom(onnx.numpy_helper.from_array(w1_expected, "W1"))
    ref_inits["W1_scale"].CopyFrom(
        onnx.numpy_helper.from_array(info["W1_scale"][keep], "W1_scale")
    )
    ref_inits["W1_zp"].CopyFrom(
        onnx.numpy_helper.from_array(info["W1_zp"][keep], "W1_zp")
    )
    ref_inits["C1"].CopyFrom(onnx.numpy_helper.from_array(info["C1"][keep], "C1"))
    ref_inits["W2"].CopyFrom(onnx.numpy_helper.from_array(w2_expected, "W2"))
    ref_inits["W2_scale"].CopyFrom(
        onnx.numpy_helper.from_array(info["W2_scale"], "W2_scale")
    )
    ref_inits["W2_zp"].CopyFrom(onnx.numpy_helper.from_array(info["W2_zp"], "W2_zp"))
    ref_inits["C2"].CopyFrom(onnx.numpy_helper.from_array(info["C2"], "C2"))

    rng = np.random.default_rng(31)
    xq = rng.integers(0, 255, size=(2, K)).astype(np.uint8)
    (y_pruned,) = _run(pruned, {"x_q": xq})
    (y_ref,) = _run(ref_model, {"x_q": xq})
    np.testing.assert_array_equal(y_pruned, y_ref)


def test_cpp_qgemm_declines_shared_weight():
    model, _info = _qgemm_chain_model(seed=44)
    extra = onnx.helper.make_node("Identity", ["W1"], ["W1_alias"], name="alias")
    model.graph.node.append(extra)
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("W1_alias", onnx.TensorProto.INT8, [8, 6])
    )
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 6]


def test_cpp_qgemm_declines_trans_a_nonzero():
    model, _info = _qgemm_chain_model(seed=40)
    for node in model.graph.node:
        if node.name == "n1":
            node.attribute.append(onnx.helper.make_attribute("transA", 1))
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 6]


def test_cpp_qgemm_declines_alpha_nonone():
    model, _info = _qgemm_chain_model(seed=41)
    for node in model.graph.node:
        if node.name == "n1":
            node.attribute.append(onnx.helper.make_attribute("alpha", 2.0))
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 6]


def test_cpp_qgemm_scalar_bias_left_untouched():
    K, H, Out = 8, 6, 4
    model, _info = _qgemm_chain_model(K, H, Out, bias=True, seed=42)
    inits = {t.name: t for t in model.graph.initializer}
    inits["C1"].CopyFrom(_i32(np.array([7]), "C1"))
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    pruned_inits = {t.name: t for t in pruned.graph.initializer}
    assert list(pruned_inits["C1"].dims) == [1]
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(pruned_inits["C1"]), [7])
    assert list(pruned_inits["W1"].dims) == [K, H // 2]


def test_cpp_qgemm_declines_ambiguous_bias_shape():
    K, H, Out = 8, 6, 4
    model, _info = _qgemm_chain_model(K, H, Out, bias=True, seed=43)
    inits = {t.name: t for t in model.graph.initializer}
    inits["C1"].CopyFrom(_i32(np.zeros((1, H)), "C1"))
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits_after = {t.name: t for t in pruned.graph.initializer}
    assert list(inits_after["W1"].dims) == [K, H]


def test_cpp_qoperator_matches_python_reference():
    # Byte-for-byte parity between the Python `apply_structured_pruning_
    # qoperator` reference and the C++-backed `apply_structured_pruning_cpp`
    # entry point -- mirrors this file's own identical parity checks for
    # every other quantized-weight chain family above.
    K, H, Out = 8, 6, 4
    model, _info = _qlinearmatmul_chain_model(K, H, Out, seed=61)
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_structured_pruning_qoperator(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    py_bytes = {t.name: t.SerializeToString() for t in pruned_py.graph.initializer}
    cpp_bytes = {t.name: t.SerializeToString() for t in pruned_cpp.graph.initializer}
    assert py_bytes == cpp_bytes


# --- DynamicQuantizeMatMul / MatMulIntegerToFloat (ORT dynamic-quantization
# --- fusion) structured pruning ---------------------------------------------
#
# Mirrors ``tests/test_pruning.py``'s own "DynamicQuantizeMatMul /
# MatMulIntegerToFloat" coverage for ``onnxsim.apply_structured_pruning_cpp``
# -- the C++ port also runs this chain family (see
# ``onnxsim/structured_pruning_entry.cpp``'s own "DynamicQuantizeMatMul /
# MatMulIntegerToFloat" section comment), wired into the SAME entry point as
# every other chain family above (unlike ``onnxsim.pruning``, which keeps
# ``apply_structured_pruning_dynamic_quantize_matmul`` a separate top-level
# function). Fixture weights are quantized via the REAL
# ``onnxruntime.quantization.quantize_dynamic`` tool -- never a hand-rolled
# re-implementation of the quantization scheme -- by wrapping each weight in
# a minimal ``Y = MatMul(X, W)`` model, running the real tool on it, and
# reading back its own emitted ``W_quantized``/``W_scale``/``W_zero_point``
# initializers, mirroring ``test_pruning.py``'s own identical
# ``_quantize_dynamic_weight`` helper. Unlike the QDQ section above, these
# oracle tests run the pruned graph through onnxruntime with DEFAULT graph
# optimizations (plain ``_run``, not ``_run_unfused``): `DynamicQuantizeMatMul`
# is already the fully-fused kernel itself -- there is no
# `DequantizeLinear -> MatMul` pattern left for onnxruntime's own optimizer to
# additionally re-fuse, so the "unfused vs. optimized accumulation order"
# concern the QDQ section's own top comment describes does not apply here.
# `DynamicQuantizeMatMul`'s own `A`-side quantization is computed FRESH from
# the actual runtime input on every run (no calibration data, no rewriting
# needed) -- since pruning only removes some of the PRODUCER's own output
# columns (leaving `A`, and every surviving column's own int8 codes/scale/
# zero-point, byte-for-byte unchanged -- "slice, don't recompute"), a kept
# output column's value is mathematically identical whether or not its
# sibling columns were dropped, so the oracle checks below hold to plain
# float32 rounding tolerance.


def _quantize_dynamic_weight(W, per_channel=True):
    """Runs the REAL ``onnxruntime.quantization.quantize_dynamic`` tool on a
    minimal ``Y = MatMul(X, W)`` wrapper model and returns the genuine
    quantized ``(W_int8[K,N], W_scale, W_zero_point)`` triple it emits for
    `W` -- never a hand-rolled re-implementation of the quantization scheme.
    Mirrors ``test_pruning.py``'s own identical helper.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    K, N = W.shape
    src = _model(
        f"""
        g (float[1,{K}] X) => (float[1,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(W, "W")],
        opset=21,
    )
    with tempfile.TemporaryDirectory() as d:
        src_path = os.path.join(d, "src.onnx")
        dst_path = os.path.join(d, "dst.onnx")
        onnx.save(src, src_path)
        quantize_dynamic(
            src_path, dst_path, per_channel=per_channel, weight_type=QuantType.QInt8
        )
        q = onnx.load(dst_path)
    inits = {t.name: t for t in q.graph.initializer}
    Wq = onnx.numpy_helper.to_array(inits["W_quantized"]).copy()
    Wscale = onnx.numpy_helper.to_array(inits["W_scale"]).copy()
    Wzp = onnx.numpy_helper.to_array(inits["W_zero_point"]).copy()
    return Wq, Wscale, Wzp


def _dqmm_chain_model(
    K, N1, N2, per_channel=True, activation="Relu", bias1=False, seed=0
):
    """Builds ``DynamicQuantizeMatMul(mm1) -> activation ->
    DynamicQuantizeMatMul(mm2)`` -- a same-family chain -- from random float
    weights, quantized via the real tool (:func:`_quantize_dynamic_weight`).
    Returns ``(model, info)`` where `info` carries every float/quantized
    array needed to hand-build an "already pruned" oracle.
    """
    rng = np.random.default_rng(seed)
    W1f = (rng.standard_normal((K, N1)) * 0.3).astype(np.float32)
    W2f = (rng.standard_normal((N1, N2)) * 0.3).astype(np.float32)
    B1f = (rng.standard_normal(N1) * 0.05).astype(np.float32) if bias1 else None
    W1q, W1s, W1zp = _quantize_dynamic_weight(W1f, per_channel)
    W2q, W2s, W2zp = _quantize_dynamic_weight(W2f, per_channel)

    bias_arg = ", B1" if B1f is not None else ""
    body = f"""
    g (float[1,{K}] X) => (float[1,{N2}] Y)
    {{
      h1 = com.microsoft.DynamicQuantizeMatMul(X, W1, W1_scale, W1_zero_point{bias_arg})
      h1a = {activation}(h1)
      Y = com.microsoft.DynamicQuantizeMatMul(h1a, W2, W2_scale, W2_zero_point)
    }}
    """
    inits = [
        onnx.numpy_helper.from_array(W1q, "W1"),
        onnx.numpy_helper.from_array(W1s, "W1_scale"),
        onnx.numpy_helper.from_array(W1zp, "W1_zero_point"),
        onnx.numpy_helper.from_array(W2q, "W2"),
        onnx.numpy_helper.from_array(W2s, "W2_scale"),
        onnx.numpy_helper.from_array(W2zp, "W2_zero_point"),
    ]
    if B1f is not None:
        inits.append(_f32(B1f, "B1"))
    model = _nbits_model(body, initializer=inits, opset=21)
    return model, {
        "W1f": W1f,
        "W2f": W2f,
        "B1f": B1f,
        "W1q": W1q,
        "W1s": W1s,
        "W1zp": W1zp,
        "W2q": W2q,
        "W2s": W2s,
        "W2zp": W2zp,
    }


def _dqmm_importance_keep(W_q, W_s, W_zp, keep_count):
    w_dequant = (
        W_q.astype(np.float64) - W_zp.astype(np.float64)[None, :]
    ) * W_s.astype(np.float64)[None, :]
    importance = np.linalg.norm(w_dequant, axis=0)  # per output channel
    return np.sort(np.argsort(-importance)[:keep_count])


def test_cpp_dynamic_quantize_matmul_chain_matches_independently_requantized_oracle():
    # The core correctness invariant this project's earlier rounds
    # established, adapted from ``test_pruning.py``'s own
    # ``test_dynamic_quantize_matmul_producer_matches_independently_
    # requantized_subset``: a pruned model's output must match a reference
    # built by quantizing the already-column-subset weight directly via the
    # REAL quantizer -- NOT the full unpruned model's own output. Also
    # confirms the `DynamicQuantizeMatMul` nodes themselves need no
    # rewriting at all: both nodes survive pruning with their op_type/domain/
    # input count completely unchanged, and the SECOND node (whose own `A`
    # input is the first node's smaller, pruned output) still computes a
    # correct result -- proof that its own internal dynamic quantization of
    # the activation recomputes correctly against the smaller downstream
    # weight with no help from this pass at all.
    K, N1, N2 = 16, 24, 12
    model, info = _dqmm_chain_model(K, N1, N2, per_channel=True, seed=2)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}

    dq_nodes = [n for n in pruned.graph.node if n.op_type == "DynamicQuantizeMatMul"]
    assert len(dq_nodes) == 2  # neither node was removed, replaced, or split
    assert {n.domain for n in dq_nodes} == {"com.microsoft"}

    keep = _dqmm_importance_keep(info["W1q"], info["W1s"], info["W1zp"], N1 // 2)
    assert list(inits["W1"].dims) == [K, N1 // 2]
    assert list(inits["W1_scale"].dims) == [N1 // 2]
    assert list(inits["W1_zero_point"].dims) == [N1 // 2]
    assert list(inits["W2"].dims) == [N1 // 2, N2]

    # "Slice, don't recompute": the pruned graph's own quantized codes/
    # scale/zero_point are EXACTLY a hand-slice of the original ones.
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["W1"]), info["W1q"][:, keep]
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["W1_scale"]), info["W1s"][keep]
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["W1_zero_point"]), info["W1zp"][keep]
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["W2"]), info["W2q"][keep, :]
    )

    # Independently reconstructed "already pruned" reference model: the
    # PRODUCER's kept-column weight is quantized FRESH via the real
    # quantizer (per-channel quantization is column-independent, so this is
    # byte-identical to the slice above). The CONSUMER's own weight is NEVER
    # requantized -- its quantized CODES are sliced directly and its scale/
    # zero-point copied UNCHANGED from the original, exactly what the
    # consumer role itself does.
    ref_model, _ = _dqmm_chain_model(K, len(keep), N2, per_channel=True, seed=987654)
    ref_inits = {t.name: t for t in ref_model.graph.initializer}
    # Overwrite the reference's own randomly-seeded producer weight with the
    # REAL subset requantization of the ORIGINAL W1f -- not unrelated random
    # data.
    W1q_sub, W1s_sub, W1zp_sub = _quantize_dynamic_weight(
        info["W1f"][:, keep], per_channel=True
    )
    ref_inits["W1"].CopyFrom(onnx.numpy_helper.from_array(W1q_sub, "W1"))
    ref_inits["W1_scale"].CopyFrom(onnx.numpy_helper.from_array(W1s_sub, "W1_scale"))
    ref_inits["W1_zero_point"].CopyFrom(
        onnx.numpy_helper.from_array(W1zp_sub, "W1_zero_point")
    )
    ref_inits["W2"].CopyFrom(onnx.numpy_helper.from_array(info["W2q"][keep, :], "W2"))
    ref_inits["W2_scale"].CopyFrom(
        onnx.numpy_helper.from_array(info["W2s"], "W2_scale")
    )
    ref_inits["W2_zero_point"].CopyFrom(
        onnx.numpy_helper.from_array(info["W2zp"], "W2_zero_point")
    )

    rng = np.random.default_rng(11)
    x = rng.standard_normal((1, K)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_ref,) = _run(ref_model, {"X": x})
    np.testing.assert_array_equal(y_pruned, y_ref)


def test_cpp_dynamic_quantize_matmul_per_tensor_scale_left_untouched():
    K, N1, N2 = 16, 24, 12
    model, _info = _dqmm_chain_model(K, N1, N2, per_channel=False, seed=3)
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    # Per-tensor (scalar) scale/zero-point are never sliced, regardless of
    # channel count -- byte-identical to the original.
    assert list(inits["W1_scale"].dims) == []
    assert list(inits["W1_zero_point"].dims) == []
    assert list(inits["W1"].dims) == [K, N1 // 2]

    rng = np.random.default_rng(12)
    x = rng.standard_normal((1, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    assert y.shape == (1, N2)
    assert np.all(np.isfinite(y))


def test_cpp_dynamic_quantize_matmul_bias_is_sliced_as_plain_float():
    K, N1, N2 = 16, 24, 12
    model, info = _dqmm_chain_model(K, N1, N2, per_channel=True, bias1=True, seed=8)
    onnx.checker.check_model(model)
    keep = _dqmm_importance_keep(info["W1q"], info["W1s"], info["W1zp"], N1 // 2)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert inits["B1"].data_type == onnx.TensorProto.FLOAT
    assert list(inits["B1"].dims) == [N1 // 2]
    # Bias values are UNCHANGED (plain float, never quantized/rescaled) --
    # unlike QGemm/QLinearConv's own quantized int32 bias.
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["B1"]), info["B1f"][keep]
    )


def test_cpp_dynamic_quantize_matmul_tied_weight_is_declined():
    # A second, unrelated node also reads W1 -- shared/tied, can't be sliced
    # -- mirrors ``test_pruning.py``'s own
    # ``test_dynamic_quantize_matmul_tied_weight_is_declined``.
    K, N1, N2 = 16, 24, 12
    model, _info = _dqmm_chain_model(K, N1, N2, per_channel=True, seed=6)
    extra = onnx.helper.make_node("Identity", ["W1"], ["W1_alias"], name="alias")
    model.graph.node.append(extra)
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("W1_alias", onnx.TensorProto.INT8, [K, N1])
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    # Left completely untouched -- W1 is shared, so this chain is declined.
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_dynamic_quantize_matmul_missing_zero_point_is_declined():
    # `b_zero_point` is schema-optional, but never confirmed empirically to
    # be omitted by any real fixture -- declined outright (see this file's
    # own section comment above ApplyDynQuantChains in
    # ``structured_pruning_entry.cpp``), mirroring ``test_pruning.py``'s own
    # ``test_dynamic_quantize_matmul_missing_zero_point_is_declined``.
    K, N1, N2 = 16, 24, 12
    model, _info = _dqmm_chain_model(K, N1, N2, per_channel=True, seed=5)
    for node in model.graph.node:
        if node.output and node.output[0] == "h1":
            node.input[3] = ""  # blank b_zero_point
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_dynamic_quantize_matmul_zero_sparsity_is_a_no_op():
    K, N1, N2 = 16, 24, 12
    model, _info = _dqmm_chain_model(K, N1, N2, per_channel=True, seed=13)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.0)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_dynamic_quantize_matmul_matches_python_reference():
    K, N1, N2 = 16, 24, 12
    model, _info = _dqmm_chain_model(K, N1, N2, per_channel=True, seed=42)
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_structured_pruning_dynamic_quantize_matmul(
        model, sparsity=0.5
    )
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    py_bytes = {t.name: t.SerializeToString() for t in pruned_py.graph.initializer}
    cpp_bytes = {t.name: t.SerializeToString() for t in pruned_cpp.graph.initializer}
    assert py_bytes == cpp_bytes


# --- DynamicQuantizeConv (ConvInteger, ORT dynamic-quantization Conv
# --- pattern) structured pruning ---------------------------------------------
#
# Mirrors ``tests/test_pruning.py``'s own "DynamicQuantizeConv (ConvInteger)"
# coverage for ``onnxsim.apply_structured_pruning_cpp`` -- the C++ port now
# also runs this chain family (see ``onnxsim/structured_pruning_entry.cpp``'s
# own "DynamicQuantizeConv (ConvInteger, ...)" section comment), wired into
# the SAME entry point as every other chain family above. Fixture weights are
# quantized via the REAL ``onnxruntime.quantization.quantize_dynamic`` tool
# (never a hand-rolled re-implementation), mirroring ``test_pruning.py``'s own
# identical ``_quantize_dynamic_conv_weight``/``_dqconv_model_from_weights``
# helpers (trimmed here of that file's own ``Reshape([0,-1])`` classifier-head
# variant, out of scope for this C++ port -- see below).
#
# KNOWN, DELIBERATE GAP (documented in the C++ section comment above
# ``ApplyConvIntegerChains``): the classifier-head
# ``GlobalAveragePool -> Flatten -> {MatMul, Gemm}`` hop pruning.py's own
# reference additionally matches is NOT recognized by this C++ port --
# ``test_cpp_dynamic_quantize_conv_gap_flatten_gemm_classifier_head_is_not_matched``
# below confirms that shape is left completely untouched by
# ``apply_structured_pruning_cpp`` (while the Python reference DOES prune
# it), so this is documented directly by a test rather than merely implied.
# Because of this gap, ``onnxsim.pruning.apply_structured_pruning_dynamic_
# quantize_conv`` itself is NOT aliased to the C++ port (see that function's
# own docstring/pruning.py's own section comment) -- only the ordinary
# same-family producer/consumer chain (with the Clip/depthwise-mid-chain
# hops) is covered here.


def _quantize_dynamic_conv_weight(W, spatial=8, per_channel=False):
    """Runs the REAL ``onnxruntime.quantization.quantize_dynamic`` tool on a
    minimal ``Y = Conv(X, W)`` wrapper model and returns the genuine
    quantized ``(W_int8[M,C,kH,kW], W_scale, W_zero_point)`` triple it emits
    for `W` -- mirrors ``test_pruning.py``'s own identical helper.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    m, c, kh, kw = W.shape
    pad_h, pad_w = (kh - 1) // 2, (kw - 1) // 2
    src = _model(
        f"""
        g (float[1,{c},{spatial},{spatial}] X) => (float[1,{m},{spatial},{spatial}] Y)
        {{
          Y = Conv<kernel_shape=[{kh},{kw}], pads=[{pad_h},{pad_w},{pad_h},{pad_w}]>(X, W)
        }}
        """,
        initializer=[_f32(W, "W")],
        opset=21,
    )
    with tempfile.TemporaryDirectory() as d:
        src_path = os.path.join(d, "src.onnx")
        dst_path = os.path.join(d, "dst.onnx")
        onnx.save(src, src_path)
        quantize_dynamic(
            src_path,
            dst_path,
            per_channel=per_channel,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv"],
        )
        q = onnx.load(dst_path)
    inits = {t.name: t for t in q.graph.initializer}
    Wq = onnx.numpy_helper.to_array(inits["W_quantized"]).copy()
    Wscale = onnx.numpy_helper.to_array(inits["W_scale"]).copy()
    Wzp = onnx.numpy_helper.to_array(inits["W_zero_point"]).copy()
    return Wq, Wscale, Wzp


def _dqconv_model_from_weights(
    c,
    m1,
    m2,
    w1f,
    w2f,
    b1f=None,
    spatial=8,
    activation="Relu",
    activation_node=None,
    extra_initializer=(),
):
    """Builds ``DynamicQuantizeLinear -> ConvInteger -> Cast -> Mul
    [-> Reshape -> Add] -> activation -> DynamicQuantizeLinear ->
    ConvInteger -> Cast -> Mul`` -- a same-family chain -- mirrors
    ``test_pruning.py``'s own identical ``_dqconv_model_from_weights``.
    `activation_node`, when given, is a complete ONNX-text node line
    overriding the plain ``h1 = {activation}(...)`` node below (e.g. a
    ``Clip`` with explicit Min/Max operands), with `extra_initializer`
    carrying whatever constant tensors that text line references.
    """
    w1q, w1s, w1zp = _quantize_dynamic_conv_weight(w1f, spatial=spatial)
    w2q, w2s, w2zp = _quantize_dynamic_conv_weight(w2f, spatial=spatial)

    kh1, kw1 = w1f.shape[2], w1f.shape[3]
    kh2, kw2 = w2f.shape[2], w2f.shape[3]
    ph1, pw1 = (kh1 - 1) // 2, (kw1 - 1) // 2
    ph2, pw2 = (kh2 - 1) // 2, (kw2 - 1) // 2

    if b1f is not None:
        bias_ops = """
          b1r = Reshape(b1, b1_shape)
          c1 = Add(c1s, b1r)
        """
        c1_out = "c1"
    else:
        bias_ops = ""
        c1_out = "c1s"

    activation_ops = (
        activation_node
        if activation_node is not None
        else f"h1 = {activation}({c1_out})"
    )

    body = f"""
    g (float[1,{c},{spatial},{spatial}] x) => (float[1,{m2},{spatial},{spatial}] y)
    {{
      xq, x_scale, x_zero_point = DynamicQuantizeLinear(x)
      combined_scale1 = Mul(x_scale, w1_scale)
      c1i = ConvInteger<kernel_shape=[{kh1},{kw1}], pads=[{ph1},{pw1},{ph1},{pw1}]>(xq, w1_quantized, x_zero_point, w1_zero_point)
      c1f = Cast<to=1>(c1i)
      c1s = Mul(c1f, combined_scale1)
      {bias_ops}
      {activation_ops}
      rq, r_scale, r_zero_point = DynamicQuantizeLinear(h1)
      combined_scale2 = Mul(r_scale, w2_scale)
      c2i = ConvInteger<kernel_shape=[{kh2},{kw2}], pads=[{ph2},{pw2},{ph2},{pw2}]>(rq, w2_quantized, r_zero_point, w2_zero_point)
      c2f = Cast<to=1>(c2i)
      y = Mul(c2f, combined_scale2)
    }}
    """
    inits = [
        onnx.numpy_helper.from_array(w1q, "w1_quantized"),
        onnx.numpy_helper.from_array(w1s, "w1_scale"),
        onnx.numpy_helper.from_array(w1zp, "w1_zero_point"),
        onnx.numpy_helper.from_array(w2q, "w2_quantized"),
        onnx.numpy_helper.from_array(w2s, "w2_scale"),
        onnx.numpy_helper.from_array(w2zp, "w2_zero_point"),
        *extra_initializer,
    ]
    if b1f is not None:
        inits.append(_f32(b1f, "b1"))
        inits.append(
            onnx.numpy_helper.from_array(
                np.array([1, -1, 1, 1], dtype=np.int64), "b1_shape"
            )
        )
    model = _model(body, initializer=inits, opset=21)
    return model, {
        "W1f": w1f,
        "W2f": w2f,
        "B1f": b1f,
        "W1q": w1q,
        "W1s": w1s,
        "W1zp": w1zp,
        "W2q": w2q,
        "W2s": w2s,
        "W2zp": w2zp,
        "spatial": spatial,
    }


def _dqconv_chain_model(c, m1, m2, kh=3, kw=3, bias1=False, spatial=8, seed=0):
    rng = np.random.default_rng(seed)
    w1f = (rng.standard_normal((m1, c, kh, kw)) * 0.3).astype(np.float32)
    w2f = (rng.standard_normal((m2, m1, kh, kw)) * 0.3).astype(np.float32)
    b1f = (rng.standard_normal(m1) * 0.05).astype(np.float32) if bias1 else None
    return _dqconv_model_from_weights(c, m1, m2, w1f, w2f, b1f=b1f, spatial=spatial)


def _dqconv_clip_model_from_weights(
    c, m1, m2, w1f, w2f, b1f=None, spatial=8, min_val=0.0, max_val=6.0
):
    """The `_dqconv_model_from_weights` analogue exercising
    `WalkToConvIntegerConsumer`'s own ``Clip`` pass-through hop (the ReLU6
    shape MobileNetV2/V3 and EfficientNet-Lite Conv chains use) in place of a
    plain unary activation between the two ``ConvInteger`` chains.
    """
    extra_initializer = []
    if min_val is not None:
        extra_initializer.append(
            onnx.numpy_helper.from_array(np.array(min_val, dtype=np.float32), "ClipMin")
        )
    if max_val is not None:
        extra_initializer.append(
            onnx.numpy_helper.from_array(np.array(max_val, dtype=np.float32), "ClipMax")
        )
    c1_out = "c1" if b1f is not None else "c1s"
    if min_val is not None and max_val is not None:
        clip_node = f"h1 = Clip({c1_out}, ClipMin, ClipMax)"
    elif min_val is not None:
        clip_node = f"h1 = Clip({c1_out}, ClipMin)"
    elif max_val is not None:
        clip_node = f"h1 = Clip({c1_out}, , ClipMax)"
    else:
        clip_node = f"h1 = Clip({c1_out})"
    return _dqconv_model_from_weights(
        c,
        m1,
        m2,
        w1f,
        w2f,
        b1f=b1f,
        spatial=spatial,
        activation_node=clip_node,
        extra_initializer=extra_initializer,
    )


def _dqconv_clip_chain_model(
    c, m1, m2, kh=3, kw=3, bias1=False, spatial=8, seed=0, min_val=0.0, max_val=6.0
):
    rng = np.random.default_rng(seed)
    w1f = (rng.standard_normal((m1, c, kh, kw)) * 0.3).astype(np.float32)
    w2f = (rng.standard_normal((m2, m1, kh, kw)) * 0.3).astype(np.float32)
    b1f = (rng.standard_normal(m1) * 0.05).astype(np.float32) if bias1 else None
    return _dqconv_clip_model_from_weights(
        c, m1, m2, w1f, w2f, b1f=b1f, spatial=spatial, min_val=min_val, max_val=max_val
    )


def _dqconv_depthwise_chain_model_from_weights(
    c, m1, m2, w1f, wdf, w2f, b1f=None, bdf=None, spatial=8, activation="Relu"
):
    """The `_dqconv_model_from_weights` analogue for a full ``pointwise
    ConvInteger -> depthwise ConvInteger -> pointwise ConvInteger`` chain --
    the canonical MobileNetV1/V2/V3 depthwise-separable-conv unit -- after
    real dynamic quantization. `w1f`/`w2f` are ordinary (`group == 1`)
    pointwise (``1x1``) Conv weights; `wdf` is the depthwise stage's own
    ``[m1, 1, khd, kwd]`` weight (`group == m1`). Mirrors ``test_pruning.py``'s
    own identical helper.
    """
    w1q, w1s, w1zp = _quantize_dynamic_conv_weight(w1f, spatial=spatial)
    wdq, wds, wdzp = _quantize_dynamic_conv_weight(wdf, spatial=spatial)
    w2q, w2s, w2zp = _quantize_dynamic_conv_weight(w2f, spatial=spatial)

    khd, kwd = wdf.shape[2], wdf.shape[3]
    phd, pwd = (khd - 1) // 2, (kwd - 1) // 2

    if b1f is not None:
        bias1_ops = """
          b1r = Reshape(b1, b1_shape)
          c1 = Add(c1s, b1r)
        """
        c1_out = "c1"
    else:
        bias1_ops = ""
        c1_out = "c1s"

    if bdf is not None:
        biasd_ops = """
          bdr = Reshape(bd, bd_shape)
          cd = Add(cds, bdr)
        """
        cd_out = "cd"
    else:
        biasd_ops = ""
        cd_out = "cds"

    body = f"""
    g (float[1,{c},{spatial},{spatial}] x) => (float[1,{m2},{spatial},{spatial}] y)
    {{
      xq, x_scale, x_zero_point = DynamicQuantizeLinear(x)
      combined_scale1 = Mul(x_scale, w1_scale)
      c1i = ConvInteger<kernel_shape=[1,1]>(xq, w1_quantized, x_zero_point, w1_zero_point)
      c1f = Cast<to=1>(c1i)
      c1s = Mul(c1f, combined_scale1)
      {bias1_ops}
      h1 = {activation}({c1_out})
      rq, r_scale, r_zero_point = DynamicQuantizeLinear(h1)
      combined_scaled = Mul(r_scale, wd_scale)
      cdi = ConvInteger<kernel_shape=[{khd},{kwd}], pads=[{phd},{pwd},{phd},{pwd}], group={m1}>(rq, wd_quantized, r_zero_point, wd_zero_point)
      cdf = Cast<to=1>(cdi)
      cds = Mul(cdf, combined_scaled)
      {biasd_ops}
      hd = {activation}({cd_out})
      rq2, r_scale2, r_zero_point2 = DynamicQuantizeLinear(hd)
      combined_scale2 = Mul(r_scale2, w2_scale)
      c2i = ConvInteger<kernel_shape=[1,1]>(rq2, w2_quantized, r_zero_point2, w2_zero_point)
      c2f = Cast<to=1>(c2i)
      y = Mul(c2f, combined_scale2)
    }}
    """
    inits = [
        onnx.numpy_helper.from_array(w1q, "w1_quantized"),
        onnx.numpy_helper.from_array(w1s, "w1_scale"),
        onnx.numpy_helper.from_array(w1zp, "w1_zero_point"),
        onnx.numpy_helper.from_array(wdq, "wd_quantized"),
        onnx.numpy_helper.from_array(wds, "wd_scale"),
        onnx.numpy_helper.from_array(wdzp, "wd_zero_point"),
        onnx.numpy_helper.from_array(w2q, "w2_quantized"),
        onnx.numpy_helper.from_array(w2s, "w2_scale"),
        onnx.numpy_helper.from_array(w2zp, "w2_zero_point"),
    ]
    if b1f is not None:
        inits.append(_f32(b1f, "b1"))
        inits.append(
            onnx.numpy_helper.from_array(
                np.array([1, -1, 1, 1], dtype=np.int64), "b1_shape"
            )
        )
    if bdf is not None:
        inits.append(_f32(bdf, "bd"))
        inits.append(
            onnx.numpy_helper.from_array(
                np.array([1, -1, 1, 1], dtype=np.int64), "bd_shape"
            )
        )
    model = _model(body, initializer=inits, opset=21)
    return model, {
        "W1f": w1f,
        "Wdf": wdf,
        "W2f": w2f,
        "B1f": b1f,
        "Bdf": bdf,
        "W1q": w1q,
        "W1s": w1s,
        "W1zp": w1zp,
        "Wdq": wdq,
        "Wds": wds,
        "Wdzp": wdzp,
        "W2q": w2q,
        "W2s": w2s,
        "W2zp": w2zp,
        "spatial": spatial,
    }


def _dqconv_depthwise_chain_model(
    c, m1, m2, khd=3, kwd=3, bias1=False, biasd=False, spatial=8, seed=0
):
    rng = np.random.default_rng(seed)
    w1f = (rng.standard_normal((m1, c, 1, 1)) * 0.3).astype(np.float32)
    wdf = (rng.standard_normal((m1, 1, khd, kwd)) * 0.3).astype(np.float32)
    w2f = (rng.standard_normal((m2, m1, 1, 1)) * 0.3).astype(np.float32)
    b1f = (rng.standard_normal(m1) * 0.05).astype(np.float32) if bias1 else None
    bdf = (rng.standard_normal(m1) * 0.05).astype(np.float32) if biasd else None
    return _dqconv_depthwise_chain_model_from_weights(
        c, m1, m2, w1f, wdf, w2f, b1f=b1f, bdf=bdf, spatial=spatial
    )


def _dqconv_gap_flatten_gemm_model_from_weights(c, m1, out, w1f, w3f, spatial=8):
    """The classifier-head shape pruning.py's own reference matches --
    a single ``ConvInteger`` producer's own logical output feeding
    ``GlobalAveragePool -> Flatten(axis=1) -> Gemm`` DIRECTLY (no downstream
    ``ConvInteger`` consumer at all, isolating this hop from the ordinary
    producer/consumer chain shape already covered by the tests above) --
    but this C++ port deliberately does NOT match (see this section's own
    top comment) -- used only to confirm that gap directly: with no
    recognized consumer of any kind, `WalkToConvIntegerConsumer` declines at
    the very first hop (`GlobalAveragePool` isn't a unary/`Clip`/
    `DynamicQuantizeLinear`-leading-to-`ConvInteger` hop), so
    `FindConvIntegerChains` finds nothing at all here and the whole graph is
    left untouched.
    """
    w1q, w1s, w1zp = _quantize_dynamic_conv_weight(w1f, spatial=spatial)
    kh1, kw1 = w1f.shape[2], w1f.shape[3]
    ph1, pw1 = (kh1 - 1) // 2, (kw1 - 1) // 2

    body = f"""
    g (float[1,{c},{spatial},{spatial}] x) => (float[1,{out}] y)
    {{
      xq, x_scale, x_zero_point = DynamicQuantizeLinear(x)
      combined_scale1 = Mul(x_scale, w1_scale)
      c1i = ConvInteger<kernel_shape=[{kh1},{kw1}], pads=[{ph1},{pw1},{ph1},{pw1}]>(xq, w1_quantized, x_zero_point, w1_zero_point)
      c1f = Cast<to=1>(c1i)
      h1 = Mul(c1f, combined_scale1)
      p = GlobalAveragePool(h1)
      f = Flatten<axis=1>(p)
      y = Gemm<transB=1>(f, w3)
    }}
    """
    inits = [
        onnx.numpy_helper.from_array(w1q, "w1_quantized"),
        onnx.numpy_helper.from_array(w1s, "w1_scale"),
        onnx.numpy_helper.from_array(w1zp, "w1_zero_point"),
        _f32(w3f, "w3"),
    ]
    model = _model(body, initializer=inits, opset=21)
    return model


def test_cpp_dynamic_quantize_conv_matches_python_reference():
    # Byte-for-byte parity between the Python `apply_structured_pruning_
    # dynamic_quantize_conv` reference and the C++-backed
    # `apply_structured_pruning_cpp` entry point, for the core (bias-bearing)
    # producer -> consumer chain -- mirrors this file's own identical parity
    # checks for every other quantized-weight chain family above.
    c, m1, m2 = 4, 8, 6
    model, _info = _dqconv_chain_model(c, m1, m2, bias1=True, seed=21)
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_structured_pruning_dynamic_quantize_conv(
        model, sparsity=0.5
    )
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    py_bytes = {t.name: t.SerializeToString() for t in pruned_py.graph.initializer}
    cpp_bytes = {t.name: t.SerializeToString() for t in pruned_cpp.graph.initializer}
    assert py_bytes == cpp_bytes
    # A real channel was actually removed (not a vacuous no-op comparison).
    assert list(cpp_bytes.keys())
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    assert list(inits["w1_quantized"].dims)[0] == m1 // 2
    assert list(inits["w2_quantized"].dims)[1] == m1 // 2


def test_cpp_dynamic_quantize_conv_clip_pass_through_matches_python_reference():
    # The ReLU6 (`Clip(0, 6)`) pass-through hop between the two `ConvInteger`
    # layers -- the MobileNetV2/V3/EfficientNet-Lite shape.
    c, m1, m2 = 4, 8, 6
    model, _info = _dqconv_clip_chain_model(c, m1, m2, bias1=True, seed=22)
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_structured_pruning_dynamic_quantize_conv(
        model, sparsity=0.5
    )
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    py_bytes = {t.name: t.SerializeToString() for t in pruned_py.graph.initializer}
    cpp_bytes = {t.name: t.SerializeToString() for t in pruned_cpp.graph.initializer}
    assert py_bytes == cpp_bytes


def test_cpp_dynamic_quantize_conv_depthwise_mid_chain_matches_python_reference():
    # The canonical MobileNetV1/V2/V3 `pointwise -> depthwise -> pointwise`
    # unit: the depthwise `ConvInteger` sitting mid-chain must be recognized
    # as a transparent pass-through hop (its own weight/bias sliced by the
    # SAME `keep` set, `group` shrunk to match) rather than declined as an
    # endpoint.
    c, m1, m2 = 4, 8, 6
    model, _info = _dqconv_depthwise_chain_model(
        c, m1, m2, bias1=True, biasd=True, seed=23
    )
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_structured_pruning_dynamic_quantize_conv(
        model, sparsity=0.5
    )
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    py_bytes = {t.name: t.SerializeToString() for t in pruned_py.graph.initializer}
    cpp_bytes = {t.name: t.SerializeToString() for t in pruned_cpp.graph.initializer}
    assert py_bytes == cpp_bytes
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    # The depthwise hop's own weight/bias were resized/re-grouped too, not
    # just the two ordinary endpoints.
    assert list(inits["wd_quantized"].dims)[0] == m1 // 2
    conv_nodes = {
        n.output[0]: n for n in pruned_cpp.graph.node if n.op_type == "ConvInteger"
    }
    depthwise_node = conv_nodes["cdi"]
    group_attr = next(a.i for a in depthwise_node.attribute if a.name == "group")
    assert group_attr == m1 // 2


def test_cpp_dynamic_quantize_conv_tied_weight_is_declined():
    # A second, unrelated node also reads w1_quantized -- shared/tied, can't
    # be sliced -- mirrors this file's own identical decline tests for the
    # other quantized-weight chain families above.
    c, m1, m2 = 4, 8, 6
    model, _info = _dqconv_chain_model(c, m1, m2, seed=24)
    extra = onnx.helper.make_node(
        "Identity", ["w1_quantized"], ["w1_alias"], name="alias"
    )
    model.graph.node.append(extra)
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(
            "w1_alias", onnx.TensorProto.INT8, [m1, c, 3, 3]
        )
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    # Left completely untouched -- w1_quantized is shared, so this chain is
    # declined.
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_dynamic_quantize_conv_zero_sparsity_is_a_no_op():
    c, m1, m2 = 4, 8, 6
    model, _info = _dqconv_chain_model(c, m1, m2, seed=25)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.0)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_dynamic_quantize_conv_gap_flatten_gemm_classifier_head_is_not_matched():
    # KNOWN, DELIBERATE GAP (see this section's own top comment and
    # ``structured_pruning_entry.cpp``'s own "DynamicQuantizeConv
    # (ConvInteger, ...)" section comment): the C++ port has no
    # `GlobalAveragePool`/`Flatten`/`Gemm`-walking machinery at all, so this
    # classifier-head shape -- which the Python reference DOES match and
    # prune -- is left completely untouched here, exactly like any other
    # unrecognized topology (never mis-sliced).
    c, m1, out = 4, 8, 5
    rng = np.random.default_rng(26)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = (rng.standard_normal((out, m1)) * 0.3).astype(np.float32)
    model = _dqconv_gap_flatten_gemm_model_from_weights(c, m1, out, w1f, w3f)
    onnx.checker.check_model(model)

    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()

    # Confirm this is a genuine "C++ doesn't recognize it" gap, not simply an
    # unmatchable fixture: the Python reference DOES prune it (the whole
    # point of this test).
    pruned_py = onnxsim.apply_structured_pruning_dynamic_quantize_conv(
        model, sparsity=0.5
    )
    assert pruned_py.SerializeToString() != model.SerializeToString()
    py_inits = {t.name: t for t in pruned_py.graph.initializer}
    assert list(py_inits["w1_quantized"].dims)[0] == m1 // 2


# --- importance_norm ("l1" vs "l2") and global_sparsity ---------------------
#
# Adapted from test_pruning.py's own `test_structured_pruning_l1_norm_favors_
# total_magnitude_single_producer` and `test_structured_pruning_global_
# sparsity_redistributes_across_chains_and_matches_oracle`: adversarial
# weight layouts/scales engineered so the new parameter provably changes
# which channels survive, cross-checked byte-for-byte against the verified
# pure-Python reference.


def test_cpp_structured_pruning_importance_norm_l1_matches_python_reference_and_differs_from_l2():
    # Column "concentrated": one entry of magnitude 8, rest zero -- L2 == L1
    # == 8. Column "spread": all 16 entries == 1 -- L2 = 4, L1 = 16. L2 ranks
    # "concentrated" above "spread"; L1 ranks "spread" above "concentrated" --
    # a genuine disagreement a correct L1 port must reproduce, not merely
    # score differently. "filler_high"/"filler_low" pin the other surviving
    # slot under either norm.
    K, H, Out = 16, 4, 3
    w1 = np.zeros((K, H), dtype=np.float32)
    w1[0, 0] = 8.0  # "concentrated"
    w1[:, 1] = 1.0  # "spread"
    w1[2, 2] = 1000.0  # "filler_high"
    w1[3, 3] = 0.001  # "filler_low"
    rng = np.random.default_rng(90)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
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
    onnx.checker.check_model(model)

    for norm in ("l2", "l1"):
        pruned_cpp = onnxsim.apply_structured_pruning_cpp(
            model, sparsity=0.5, importance_norm=norm
        )
        pruned_py = onnxsim.apply_structured_pruning(
            model, sparsity=0.5, importance_norm=norm
        )
        onnx.checker.check_model(pruned_cpp)
        assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    kept_l2 = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    kept_l1 = onnxsim.apply_structured_pruning_cpp(
        model, sparsity=0.5, importance_norm="l1"
    )
    # "l2" keeps {concentrated, filler_high}, "l1" keeps {spread,
    # filler_high} -- a real flip in which column survives, not just score.
    assert kept_l2.SerializeToString() != kept_l1.SerializeToString()


def _two_scale_mlp_model(K=8, H=16, Out=4, big_scale=50.0, small_scale=1.0, seed=0):
    # Two independent, ordinary (single-producer, group=1) MLP chains
    # sharing one input, at very different weight-magnitude scales -- the
    # adversarial case `global_sparsity` exists for: the default per-chain
    # mode cuts both to the same channel *count* regardless of scale, while
    # `global_sparsity` redistributes toward the uniformly-smaller chain.
    rng = np.random.default_rng(seed)
    w1_big = (rng.standard_normal((K, H)) * big_scale).astype(np.float32)
    w2_big = rng.standard_normal((H, Out)).astype(np.float32)
    w1_small = (rng.standard_normal((K, H)) * small_scale).astype(np.float32)
    w2_small = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Ybig, float[batch,{Out}] Ysmall)
        {{
          hbig = MatMul(X, W1big)
          abig = Relu(hbig)
          Ybig = MatMul(abig, W2big)
          hsmall = MatMul(X, W1small)
          asmall = Relu(hsmall)
          Ysmall = MatMul(asmall, W2small)
        }}
        """,
        initializer=[
            _f32(w1_big, "W1big"),
            _f32(w2_big, "W2big"),
            _f32(w1_small, "W1small"),
            _f32(w2_small, "W2small"),
        ],
    )
    return model


def test_cpp_structured_pruning_global_sparsity_matches_python_reference_and_redistributes():
    K, H, Out = 8, 16, 4
    sparsity = 0.5
    model = _two_scale_mlp_model(
        K=K, H=H, Out=Out, big_scale=50.0, small_scale=0.5, seed=7
    )
    onnx.checker.check_model(model)

    local_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=sparsity)
    global_cpp = onnxsim.apply_structured_pruning_cpp(
        model, sparsity=sparsity, global_sparsity=True
    )
    global_py = onnxsim.apply_structured_pruning(
        model, sparsity=sparsity, global_sparsity=True
    )
    onnx.checker.check_model(global_cpp)
    assert global_cpp.SerializeToString() == global_py.SerializeToString()

    inits_local = {t.name: t for t in local_cpp.graph.initializer}
    inits_global = {t.name: t for t in global_cpp.graph.initializer}

    # Per-chain-uniform (default) mode: both chains cut to the same count.
    assert inits_local["W1big"].dims[1] == H // 2
    assert inits_local["W1small"].dims[1] == H // 2
    # global_sparsity mode: the uniformly-larger chain keeps strictly more
    # channels than the uniformly-smaller one -- provably different from the
    # local, non-global default above.
    big_kept = inits_global["W1big"].dims[1]
    small_kept = inits_global["W1small"].dims[1]
    assert big_kept > H // 2 > small_kept


def test_cpp_structured_pruning_global_sparsity_and_importance_norm_l1_together_matches_python_reference():
    # Both new parameters at once -- global_sparsity's pooled ranking must
    # itself be computed with the requested importance_norm (see
    # ApplyChainsGlobal's own `importance_norm` threading), not silently
    # fall back to L2.
    K, H, Out = 8, 16, 4
    sparsity = 0.5
    model = _two_scale_mlp_model(
        K=K, H=H, Out=Out, big_scale=50.0, small_scale=0.5, seed=11
    )
    onnx.checker.check_model(model)

    for norm in ("l2", "l1"):
        pruned_cpp = onnxsim.apply_structured_pruning_cpp(
            model, sparsity=sparsity, importance_norm=norm, global_sparsity=True
        )
        pruned_py = onnxsim.apply_structured_pruning(
            model, sparsity=sparsity, importance_norm=norm, global_sparsity=True
        )
        onnx.checker.check_model(pruned_cpp)
        assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_structured_pruning_global_sparsity_leaves_gated_ffn_chain_untouched():
    # A gated (SwiGLU) pair's two producers must agree on one shared `keep`
    # set already -- ineligible for global_sparsity's own pooled ranking
    # (see ChainIsGlobalSparsityEligible), so the whole chain is left
    # completely untouched in this mode, exactly like any other topology
    # this pass can't prove safe to pool.
    h, inter, out_dim = 8, 6, 4
    rng = np.random.default_rng(12)
    w_gate = rng.standard_normal((h, inter)).astype(np.float32)
    w_up = rng.standard_normal((h, inter)).astype(np.float32)
    w_down = rng.standard_normal((inter, out_dim)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{h}] X) => (float[batch,{out_dim}] Y)
        {{
          gate = MatMul(X, Wgate)
          act = Sigmoid(gate)
          up = MatMul(X, Wup)
          prod = Mul(act, up)
          Y = MatMul(prod, Wdown)
        }}
        """,
        initializer=[
            _f32(w_gate, "Wgate"),
            _f32(w_up, "Wup"),
            _f32(w_down, "Wdown"),
        ],
    )
    onnx.checker.check_model(model)

    pruned_cpp = onnxsim.apply_structured_pruning_cpp(
        model, sparsity=0.5, global_sparsity=True
    )
    pruned_py = onnxsim.apply_structured_pruning(
        model, sparsity=0.5, global_sparsity=True
    )
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()
