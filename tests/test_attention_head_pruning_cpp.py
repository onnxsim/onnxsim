"""Tests for ``onnxsim.apply_attention_head_pruning_cpp`` -- the C++-backed
port of ``onnxsim.apply_attention_head_pruning`` (see
``onnxsim/structured_pruning_entry.cpp``'s "Attention-head pruning" section).
Data-free (magnitude/Frobenius-norm) only -- the calibration-driven Wanda
variant (``onnxsim.apply_attention_head_wanda_pruning``) is not fully ported
(see ``test_attention_head_wanda_pruning_cpp.py``'s own docstring for exactly
which families it does cover). Tests here are adapted from
``test_pruning.py``'s own ``apply_attention_head_pruning`` coverage for nine
of pruning.py's own ten matched op families: plain ``com.microsoft::Attention``
(merged QKV weight), ``com.microsoft::GroupQueryAttention``, the plain
``ai.onnx::Attention`` op (opset 24+), ``com.microsoft::MultiHeadAttention``,
``com.microsoft::PackedMultiHeadAttention``,
``com.microsoft::DecoderMaskedMultiHeadAttention``,
``com.microsoft::PagedAttention``, the plain ``ai.onnx::LinearAttention`` op
(opset 27+, "linear" update_rule only), and ``com.microsoft::SparseAttention``.
The tenth family, the fully decomposed (un-fused, "eager SDPA export") shape
pruning.py's own ``_find_decomposed_gqa_chains`` matches, now HAS a C++ port
(``FindDecomposedGqaChains``/``ApplyOneDecomposedGqaChain`` in
``structured_pruning_entry.cpp`` -- see that section's own comment) --
covered by its own dedicated "Decomposed (un-fused) GQA/MQA/plain-MHA
attention head pruning" test section below. An additive mask (`attn_mask`/
causal bias) before ``Softmax``, constant or genuinely dynamic, IS now
recognized and pruned there too (``ResolveDecomposedQkRoot``'s own mask
detection, ``HeadBiasInputIsSafe``/``SliceOrGatherHeadBias`` -- this port's
own analogue of pruning.py's ``_head_bias_input_is_safe``/
``_slice_or_gather_head_bias`` for this one chain family); an ``Einsum``-based
QK^T/AV product IS now recognized too, in place of a plain ``MatMul``, for
both products (see the dedicated ``Einsum`` tests in that same section);
decomposed RoPE and decomposed Q/K-norm pass-through ARE now recognized as
well, in the narrower sub-scope structured_pruning_entry.cpp's own section
comment documents (Q's own branch and K's separate-``perm=[0,1,3,2]`` swap
branch only -- never K's combined-perm branch, never combined with the
``Einsum`` case, never on V's own branch); and a packed-QKV-then-``Split``
producer (``MatchDecomposedPackedQkvProducer``) and a true-MQA fast path
(individual query-head-granularity ranking/pruning when ``kv_num_heads == 1``)
are ALSO now recognized -- see the dedicated tests for each below. Every
remaining shape not covered by one of those still declines to match in this
C++ port (never mis-sliced), so ``apply_attention_head_pruning``/
``apply_attention_head_wanda_pruning`` remain pure-Python (NOT yet aliased to
this port) -- see that section's own tests for the exact, explicit divergence
each remaining narrowing produces.

``GroupQueryAttention``'s and the plain ``ai.onnx::Attention`` op's own
optional per-head inputs -- ``attention_bias``/``attn_mask``,
``past_key``/``past_value`` (plus, GQA-only, ``k_scale``/``v_scale`` and
``head_sink``) -- as well as the plain ``com.microsoft::Attention``/
``DecoderMaskedSelfAttention``/``PackedAttention`` family's own optional
``attention_bias``, and ``com.microsoft::MultiHeadAttention``'s/
``com.microsoft::PackedMultiHeadAttention``'s/
``com.microsoft::DecoderMaskedMultiHeadAttention``'s own optional
``attention_bias`` (and, ``MultiHeadAttention``'s/
``DecoderMaskedMultiHeadAttention``'s own ``past_key``/``past_value``) now
reuse this exact same ``HeadBiasInputIsSafe``/``SliceOrGatherHeadBias``
machinery (``MatchAttentionProducer``/``MatchGqaProducer``/
``MatchOnnxAttentionProducer``/``MatchMultiHeadAttentionProducer``/
``MatchPackedMultiHeadAttentionProducer``/
``MatchDecoderMaskedMultiHeadAttentionProducer``/
``ApplyOnePlainAttentionChain``/``ApplyOneGqaChain``, plus this file's own
``PastKvConstantsAreSliceable``/``SliceKvCacheAxis1`` for the KV-cache pair):
a constant that resolves to a genuine per-head/per-KV-group tensor is sliced
in place, a constant that resolves to a broadcast is left untouched (without
declining the match), a dynamic one gets a ``Gather`` node spliced in ahead
of it when genuinely per-head, and only a shape that doesn't statically
resolve either way declines the whole match -- see the dedicated tests in
each family's own section below (the ``_attention_model`` helper's own
``attention_bias`` parameter, the ``_gqa_model`` helper's own
``attention_bias``/``head_sink``/``past_kv`` parameters,
``_onnx_attention_model``'s own ``attn_mask``/``past_kv`` parameters, and
``MultiHeadAttention``'s/``PackedMultiHeadAttention``'s/
``DecoderMaskedMultiHeadAttention``'s own model builders' own
``attention_bias``/``past_kv`` parameters). Each of ``MultiHeadAttention``'s
(indices 6/7) and ``DecoderMaskedMultiHeadAttention``'s (indices 5/6) own
optional ``past_key``/``past_value`` are validated with
``PastKvConstantsAreSliceable`` called with no ``scale_indices`` (neither op
has ``k_scale``/``v_scale``) and sliced along their own BNSH
``kv_num_heads`` axis when a constant of that exact shape;
``PackedMultiHeadAttention`` has no ``past_key``/``past_value`` inputs on its
own schema at all (packing mode is encoder-only).

``com.microsoft::PagedAttention`` -- the one remaining separate-Q/K/V family
with a per-head optional input beyond a combined bias -- has no
``attention_bias``/``attn_mask``-equivalent input on its own schema at all,
but its own ``head_sink`` (index 11, a genuine ``(num_heads,)`` constant),
``q_norm_weight``/``k_norm_weight`` (indices 12/13, paired-presence only --
neither ever sliced, since a ``(head_size,)`` shape never changes under
whole-head/KV-group pruning), and ``k_scale``/``v_scale`` (indices 14/15,
``PagedKvScaleIsSliceable`` -- a *different*, rank-3/axis-0
``(kv_num_heads, 1, head_size)`` PER_CHANNEL layout from every other family's
own rank-4/axis-1 one, since this op's own schema documents it that way) are
now all validated/sliced too, via ``FindSeparateQkvChains``'s own new
``qk_norm_weight_indices`` parameter and ``ApplyOneGqaChain``'s own
``is_paged`` branch -- see the dedicated PagedAttention tests below and
``MatchPagedAttentionProducer``'s/``PagedKvScaleIsSliceable``'s own comments
in ``structured_pruning_entry.cpp``.

``com.microsoft::SparseAttention``'s own required ``past_key``/``past_value``
(indices 3/4, the *same* positions ``GroupQueryAttention``'s own occupy) now
get this exact same ``PastKvConstantsAreSliceable``/``SliceKvCacheAxis1``
validate-and-slice treatment too (no ``scale_indices`` -- this op's own `T`
type constraint has no quantized cache dtype at all, unlike
``GroupQueryAttention``'s own ``T_CACHE``) -- see the dedicated
``SparseAttention`` tests below. ``LinearAttention``, the one remaining
fused-op family, has no optional per-head bias/mask/past-KV input on its own
schema at all, so none of this machinery applies to it.
"""

import ml_dtypes
import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _f16(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float16), name)


def _bf16(array, name):
    return onnx.numpy_helper.from_array(array.astype(ml_dtypes.bfloat16), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _head_idx(keep_heads, d):
    return np.concatenate([np.arange(h * d, (h + 1) * d) for h in keep_heads])


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


def _oracle_keep_groups(
    wq, wk, wv, num_heads, kv_num_heads, head_size, keep_count, v_head_size=None
):
    # `v_head_size` (V's own per-head column stride into `wv`) defaults to
    # `head_size` (Q's/K's shared one) -- the uniform case every caller but
    # the plain-ai.onnx-Attention "diff V head size" tests wants; those pass a
    # genuinely different `v_head_size` explicitly. Mirrors test_pruning.py's
    # own `_oracle_keep_groups`.
    if v_head_size is None:
        v_head_size = head_size
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
        v_block = wv[:, kv * v_head_size : (kv + 1) * v_head_size]
        importance[kv] = np.linalg.norm(
            np.concatenate([q_block, k_block, v_block], axis=1)
        )
    return np.sort(np.argsort(-importance)[:keep_count])


def _oracle_keep_groups_cross(
    wq, wk, wv, num_heads, kv_num_heads, head_size, keep_count
):
    # Like `_oracle_keep_groups` above, but combines each KV group's Q/K/V
    # block importance via sqrt(sum of squared per-block Frobenius norms)
    # rather than norm(concatenate(...)) -- required once wq's own row count
    # (Q's source tensor's own feature dimension) differs from wk's/wv's own
    # (K/V's source tensor's own feature dimension), exactly the cross-
    # attention shape this helper is for. Mirrors test_pruning.py's own
    # `_oracle_keep_groups_cross`.
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
        importance[kv] = np.sqrt(
            np.linalg.norm(q_block) ** 2
            + np.linalg.norm(k_block) ** 2
            + np.linalg.norm(v_block) ** 2
        )
    return np.sort(np.argsort(-importance)[:keep_count])


def _group_q_heads(keep_groups, group_size):
    return np.concatenate(
        [np.arange(g * group_size, (g + 1) * group_size) for g in keep_groups]
    )


# --- Plain com.microsoft::Attention (merged QKV weight) ---------------------


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
    attention_bias=None,  # constant/declared-shape attention_bias array, or None
    attention_bias_dynamic=False,  # declare AttentionBias as a graph INPUT instead
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
    extra_inputs = ""
    operands = ["X", "Wqkv"]
    if bias:
        initializer.append(_f32(bqkv, "Bqkv"))
        operands.append("Bqkv")
    else:
        operands.append("")

    # `attention_bias` (index 5) sits behind `mask_index` (3) and `past` (4),
    # both always left unconnected here -- threaded through as empty
    # positional placeholders to reach index 5.
    if attention_bias is not None:
        operands += ["", ""]
        if attention_bias_dynamic:
            shape_str = ",".join(str(d) for d in np.asarray(attention_bias).shape)
            extra_inputs += f", float[{shape_str}] AttentionBias"
        else:
            initializer.append(_f32(np.asarray(attention_bias), "AttentionBias"))
        operands.append("AttentionBias")

    # Trailing optional inputs may simply be omitted rather than spelled out
    # as empty placeholders.
    while operands and operands[-1] == "":
        operands.pop()
    qkv_inputs = ", ".join(operands)

    if with_reshape:
        shape = np.array([batch, seq, Nv], dtype=np.int64)
        initializer.append(onnx.numpy_helper.from_array(shape, "Shape"))
        tail = "ctx2 = Reshape(ctx, Shape)\n          Y = MatMul(ctx2, Wout)"
    else:
        tail = "Y = MatMul(ctx, Wout)"

    body = f"""
        g (float[batch,seq,{K}] X{extra_inputs}) => (float[batch,seq,{Out}] Y)
        {{
          ctx = com.microsoft.Attention <num_heads={heads}, qkv_hidden_sizes=[{Nq},{Nk},{Nv}]> ({qkv_inputs})
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
        K=K, H=H, D=D, Out=Out, Nq=Nq, Nk=Nk, Nv=Nv, wqkv=wqkv, bqkv=bqkv, wout=wout
    )


def _attention_node(model):
    return next(n for n in model.graph.node if n.op_type == "Attention")


def _attention_attrs(node):
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    qkv = next(list(a.ints) for a in node.attribute if a.name == "qkv_hidden_sizes")
    return num_heads, qkv


def test_cpp_attention_head_pruning_shrinks_matched_block():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _attention_node(pruned)
    num_heads, qkv = _attention_attrs(node)
    assert num_heads == 2
    assert qkv == [8, 8, 8]

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wqkv"].dims) == [8, 24]
    assert list(inits["Bqkv"].dims) == [24]
    assert list(inits["Wout"].dims) == [8, 6]


def test_cpp_attention_head_pruning_matches_manual_head_deletion_exactly():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=1)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
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


def test_cpp_attention_head_pruning_reshape_hop_is_recognized_and_shape_updated():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=3, with_reshape=True)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.25)
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


def test_cpp_attention_head_pruning_mismatched_consumer_reduction_dim_is_left_untouched():
    K, H, D, Out = 8, 4, 4, 6
    Nqkv = H * D
    rng = np.random.default_rng(6)
    wqkv = rng.standard_normal((K, 3 * Nqkv)).astype(np.float32)
    wout_wrong = rng.standard_normal((Nqkv + 1, Out)).astype(np.float32)  # off-by-one
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        g (float[2,5,{K}] X) => (float[2,5,{Out}] Y)
        {{
          ctx = com.microsoft.Attention <num_heads={H}, qkv_hidden_sizes=[{Nqkv},{Nqkv},{Nqkv}]> (X, Wqkv)
          padded = Pad <pads = [0,0,0,0,0,1]> (ctx)
          Y = MatMul(padded, Wout)
        }}
        """
    )
    model.graph.initializer.extend([_f32(wqkv, "Wqkv"), _f32(wout_wrong, "Wout")])

    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_cpp_attention_head_pruning_zero_sparsity_is_a_no_op():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=7)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.0)
    node = _attention_node(pruned)
    num_heads, qkv = _attention_attrs(node)
    assert num_heads == cfg["H"]
    assert qkv == [cfg["Nq"], cfg["Nk"], cfg["Nv"]]
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wqkv"], cfg["wqkv"])


def test_cpp_attention_head_pruning_invalid_sparsity_raises():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6)
    with pytest.raises(Exception):
        onnxsim.apply_attention_head_pruning_cpp(model, sparsity=1.0)
    with pytest.raises(Exception):
        onnxsim.apply_attention_head_pruning_cpp(model, sparsity=-0.1)


def test_cpp_attention_head_pruning_broadcast_attention_bias_is_left_untouched():
    # A genuinely BROADCAST attention_bias (dims[1] == 1, not num_heads) --
    # HeadBiasAxis classifies this as "already correct for any head count",
    # so the match succeeds (unlike the pre-fix behavior, which never even
    # inspected `attention_bias` at all) and SliceOrGatherHeadBias leaves the
    # tensor itself completely untouched while every other matched weight is
    # still pruned normally.
    H, D, seq = 4, 4, 5
    bias = (
        np.random.default_rng(40).standard_normal((1, 1, seq, seq)).astype(np.float32)
    )
    model, _ = _attention_model(K=8, H=H, D=D, Out=6, seed=40, attention_bias=bias)
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert pruned_cpp.SerializeToString() != model.SerializeToString()

    node = _attention_node(pruned_cpp)
    num_heads, _ = _attention_attrs(node)
    assert num_heads == 2
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    np.testing.assert_array_equal(
        inits_before["AttentionBias"], inits_after["AttentionBias"]
    )


def test_cpp_attention_head_pruning_per_head_attention_bias_is_sliced_matches_python():
    # A genuinely PER-HEAD constant attention_bias (dims[1] == num_heads) --
    # HeadBiasInputIsSafe confirms it's safe, so SliceOrGatherHeadBias slices
    # it in place along its own head axis by the kept heads, exactly like
    # every other per-head tensor in the chain. An earlier version of
    # MatchAttentionProducer never inspected `attention_bias` at all, so
    # pruning would have silently left a now-wrong-head-count bias connected
    # to a pruned-head-count node.
    H, D, seq = 4, 4, 5
    bias = (
        np.random.default_rng(41).standard_normal((1, H, seq, seq)).astype(np.float32)
    )
    model, cfg = _attention_model(K=8, H=H, D=D, Out=6, seed=41, attention_bias=bias)
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _attention_node(pruned_cpp)
    num_heads, _ = _attention_attrs(node)
    assert num_heads == 2
    keep = _oracle_keep_heads(
        cfg["wqkv"], cfg["Nq"], cfg["Nk"], cfg["Nv"], H, num_heads
    )
    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    assert inits["AttentionBias"].shape == (1, num_heads, seq, seq)
    np.testing.assert_array_equal(inits["AttentionBias"], bias[:, keep, :, :])

    rng = np.random.default_rng(42)
    x = rng.standard_normal((2, seq, cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned_cpp, {"X": x})
    (y_py,) = _run(pruned_py, {"X": x})
    np.testing.assert_allclose(y_pruned, y_py, rtol=1e-4, atol=1e-4)


def test_cpp_attention_head_pruning_dynamic_per_head_attention_bias_gathers_matches_python():
    # A genuinely PER-HEAD, but DYNAMIC (graph-input, not constant),
    # attention_bias -- the one case that needs SliceOrGatherHeadBias's own
    # Gather-insertion path (InsertDynamicHeadBiasGather), since the tensor's
    # own real values aren't available at prune time.
    H, D, seq = 4, 4, 5
    bias_shape = (1, H, seq, seq)
    model, cfg = _attention_model(
        K=8,
        H=H,
        D=D,
        Out=6,
        seed=43,
        attention_bias=np.zeros(bias_shape, dtype=np.float32),
        attention_bias_dynamic=True,
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert pruned_cpp.SerializeToString() != model.SerializeToString()

    gather_nodes = [n for n in pruned_cpp.graph.node if n.op_type == "Gather"]
    assert len(gather_nodes) == 1
    gather = gather_nodes[0]
    assert gather.input[0] == "AttentionBias"
    axis = next(a.i for a in gather.attribute if a.name == "axis")
    assert axis == 1
    indices_init = next(
        t for t in pruned_cpp.graph.initializer if t.name == gather.input[1]
    )
    indices = onnx.numpy_helper.to_array(indices_init)

    node = _attention_node(pruned_cpp)
    num_heads, _ = _attention_attrs(node)
    keep = _oracle_keep_heads(
        cfg["wqkv"], cfg["Nq"], cfg["Nk"], cfg["Nv"], H, num_heads
    )
    np.testing.assert_array_equal(indices, keep)
    # The node's own attention_bias input (index 5) is rewired onto the new
    # Gather's own output, not left pointing directly at "AttentionBias".
    assert node.input[5] == gather.output[0]

    rng = np.random.default_rng(44)
    x = rng.standard_normal((2, seq, cfg["K"])).astype(np.float32)
    bias = rng.standard_normal(bias_shape).astype(np.float32)
    (y_pruned,) = _run(pruned_cpp, {"X": x, "AttentionBias": bias})
    (y_py,) = _run(pruned_py, {"X": x, "AttentionBias": bias})
    np.testing.assert_allclose(y_pruned, y_py, rtol=1e-4, atol=1e-4)


def test_cpp_attention_head_pruning_unsafe_attention_bias_shape_declines_match_matches_python():
    # rank-4 with axis-1 length neither 1 (broadcast) nor num_heads
    # (per-head) -- HeadBiasAxis can't resolve this either way, so
    # HeadBiasInputIsSafe declines the whole match outright, exactly like
    # pruning.py's own `_head_bias_input_is_safe`/`_match_attention_producer`.
    H, D, seq = 4, 4, 5
    bias = (
        np.random.default_rng(45).standard_normal((1, 3, seq, seq)).astype(np.float32)
    )
    model, _ = _attention_model(K=8, H=H, D=D, Out=6, seed=45, attention_bias=bias)
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    # Declined outright -- the whole model is left byte-for-byte unchanged.
    assert pruned_cpp.SerializeToString() == model.SerializeToString()


# --- com.microsoft::GroupQueryAttention (separate Q/K/V producers) ---------


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
    #  | "unsafe" (constant, wrong kv_num_heads-axis length -- declines the match)
    attention_bias=None,  # None | "broadcast" (rank-4, axis1 size 1) | "per_head"
    #  (rank-4, axis1 == num_heads, sliced) | "dynamic_per_head" (graph input, same
    #  shape -- Gather inserted) | "unsafe" (axis1 neither 1 nor num_heads)
    head_sink=None,  # None | "nonempty" (constant (num_heads,), sliced)
    #  | "dynamic" (graph input, left alone) | "wrong_shape" (declines the match)
    q_norm_weight=None,  # constant q_norm_weight array (shape (D,)), or None (unconnected)
    k_norm_weight=None,  # constant k_norm_weight array (shape (D,)), or None (unconnected)
):
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
    elif past_kv == "unsafe":
        # rank-4 but axis-1 length != kv_num_heads -- not a shape either port
        # can safely slice, so the whole match is declined outright by both.
        past_key = rng.standard_normal((batch, KVH + 1, 1, D)).astype(np.float32)
        past_value = rng.standard_normal((batch, KVH + 1, 1, D)).astype(np.float32)
        initializer += [_f32(past_key, "PastKey"), _f32(past_value, "PastValue")]
        operands += ["PastKey", "PastValue"]
    else:
        operands += ["", ""]
    operands += ["SeqLensK", "TotalSeq"]

    # cos_cache/sin_cache/position_ids (indices 7/8/9) -- always left empty here
    # (this test file's own GQA models never exercise rotary embedding), just
    # placeholders so attention_bias/head_sink (indices 10/11) land correctly.
    operands += ["", "", ""]

    if attention_bias == "broadcast":
        bias_t = rng.standard_normal((1, 1, seq, seq)).astype(np.float32)
        initializer.append(_f32(bias_t, "AttnBias"))
        operands.append("AttnBias")
    elif attention_bias == "per_head":
        bias_t = rng.standard_normal((1, H, seq, seq)).astype(np.float32)
        initializer.append(_f32(bias_t, "AttnBias"))
        operands.append("AttnBias")
    elif attention_bias == "dynamic_per_head":
        operands.append("AttnBiasIn")
        extra_graph_inputs += f", float[1,{H},{seq},{seq}] AttnBiasIn"
    elif attention_bias == "unsafe":
        # axis-1 length neither 1 (broadcast) nor num_heads -- unresolvable,
        # declines the whole match rather than guessing.
        bias_t = rng.standard_normal((1, H + 1, seq, seq)).astype(np.float32)
        initializer.append(_f32(bias_t, "AttnBias"))
        operands.append("AttnBias")
    else:
        operands.append("")

    if head_sink == "nonempty":
        sink_t = rng.standard_normal((H,)).astype(np.float32)
        initializer.append(_f32(sink_t, "HeadSink"))
        operands.append("HeadSink")
    elif head_sink == "dynamic":
        operands.append("HeadSinkIn")
        extra_graph_inputs += f", float[{H}] HeadSinkIn"
    elif head_sink == "wrong_shape":
        sink_t = rng.standard_normal((H + 1,)).astype(np.float32)
        initializer.append(_f32(sink_t, "HeadSink"))
        operands.append("HeadSink")
    else:
        operands.append("")

    # k_scale/v_scale (indices 12/13) sit between head_sink and
    # q_norm_weight/k_norm_weight (indices 14/15) -- always left unconnected
    # here (this test file's own dedicated k_scale/v_scale tests build their
    # own model directly), just placeholders so q_norm_weight/k_norm_weight
    # land at the right index.
    if q_norm_weight is not None or k_norm_weight is not None:
        operands += ["", ""]
        if q_norm_weight is not None:
            initializer.append(_f32(np.asarray(q_norm_weight), "QNorm"))
            operands.append("QNorm")
        else:
            operands.append("")
        if k_norm_weight is not None:
            initializer.append(_f32(np.asarray(k_norm_weight), "KNorm"))
            operands.append("KNorm")
        else:
            operands.append("")

    while operands and operands[-1] == "":
        operands.pop()

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


def test_cpp_gqa_pruning_shrinks_matched_block():
    model, cfg = _gqa_model(K=8, H=4, KVH=4, D=8, Out=6)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert num_heads == 2
    assert kv_num_heads == 2

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [8, 16]
    assert list(inits["Wk"].dims) == [8, 16]
    assert list(inits["Wv"].dims) == [8, 16]
    assert list(inits["Wout"].dims) == [16, 6]


def test_cpp_gqa_pruning_zero_sparsity_is_a_no_op():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=7)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.0)
    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert num_heads == cfg["H"]
    assert kv_num_heads == cfg["KVH"]
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"])


def test_cpp_gqa_pruning_unequal_heads_drops_whole_groups_and_preserves_ratio():
    model, cfg = _gqa_model(K=8, H=8, KVH=4, D=8, Out=6, seed=11)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 2
    assert num_heads == 4
    assert num_heads // kv_num_heads == cfg["H"] // cfg["KVH"]

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


def test_cpp_gqa_pruning_matches_oracle_exactly():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=1)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1
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


def test_cpp_gqa_pruning_slices_bias_when_producer_has_one():
    model, cfg = _gqa_model(K=8, H=4, KVH=2, D=8, Out=6, seed=14, bias=True)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)

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


def test_cpp_gqa_pruning_reshape_hop_is_recognized_and_shape_updated():
    model, cfg = _gqa_model(K=8, H=4, KVH=2, D=8, Out=6, seed=3, with_reshape=True)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
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
    assert kv_num_heads == 1
    assert num_heads == 2

    shape = onnx.numpy_helper.to_array(
        next(t for t in pruned.graph.initializer if t.name == "Shape")
    )
    assert shape[-1] == num_heads * cfg["D"]

    rng = np.random.default_rng(4)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    assert y.shape == (cfg["batch"], cfg["seq"], cfg["Out"])


def test_cpp_gqa_pruning_nonempty_sliceable_past_kv_constant_is_sliced():
    # A constant past_key/past_value in the schema's own BNSH layout, with its
    # axis-1 length matching kv_num_heads, is now validated-and-sliced along
    # that axis by `keep_groups` -- the same index set K's/V's own producer
    # weights are sliced by -- rather than declining the whole match outright
    # (this test used to be named `..._is_left_untouched`, covering the OLD,
    # narrower C++ behavior; see `..._past_kv_constant_is_unsafe_and_declined`
    # below for the shape that genuinely still declines). Byte-for-byte
    # cross-check against the live pure-Python reference.
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=12, past_kv="nonempty")
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _gqa_node(pruned_cpp)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1
    assert num_heads == group_size
    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    assert inits["PastKey"].shape == (cfg["batch"], kv_num_heads, 1, cfg["D"])
    assert inits["PastValue"].shape == (cfg["batch"], kv_num_heads, 1, cfg["D"])


def test_cpp_gqa_pruning_past_kv_constant_is_unsafe_and_declined():
    # A constant past_key/past_value whose axis-1 length does NOT match
    # kv_num_heads (rank-4, but not the schema's own BNSH layout) is not a
    # shape either port can safely slice -- both decline the whole match
    # outright, identically.
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=75, past_kv="unsafe")
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


def test_cpp_gqa_pruning_dynamic_past_kv_input_is_still_pruned():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=13, past_kv="dynamic")
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1
    assert num_heads == group_size


def test_cpp_gqa_pruning_broadcast_attention_bias_is_left_untouched_block_still_pruned():
    # A constant attention_bias whose axis-1 (num_heads) length is 1 is an
    # unconditional broadcast -- it carries no per-head values at all, so it
    # is left byte-for-byte untouched even though the rest of the block IS
    # pruned (this does NOT decline the whole match, unlike the old,
    # overly-conservative C++ behavior this fix replaces).
    model, cfg = _gqa_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=31, attention_bias="broadcast"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _gqa_node(pruned_cpp)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1
    assert num_heads == cfg["H"] // cfg["KVH"]
    bias_before = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "AttnBias")
    )
    bias_after = onnx.numpy_helper.to_array(
        next(t for t in pruned_cpp.graph.initializer if t.name == "AttnBias")
    )
    np.testing.assert_array_equal(bias_before, bias_after)


def test_cpp_gqa_pruning_per_head_attention_bias_constant_is_sliced():
    # A constant attention_bias whose axis-1 length equals num_heads is a
    # genuine per-(query-)head tensor -- sliced in place by `keep_q_heads`,
    # the same query-head granularity Q's own producer weight is sliced by.
    model, cfg = _gqa_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=32, attention_bias="per_head"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _gqa_node(pruned_cpp)
    num_heads, kv_num_heads = _gqa_attrs(node)
    bias_after = onnx.numpy_helper.to_array(
        next(t for t in pruned_cpp.graph.initializer if t.name == "AttnBias")
    )
    assert bias_after.shape == (1, num_heads, cfg["seq"], cfg["seq"])


def test_cpp_gqa_pruning_dynamic_per_head_attention_bias_gets_gather_inserted():
    # A DYNAMIC (graph-input) attention_bias whose declared shape resolves to
    # a genuine per-head axis gets a new Gather node spliced in ahead of the
    # GroupQueryAttention node, selecting the kept query heads' own slice at
    # runtime, rather than being left stale or blocking the match.
    model, cfg = _gqa_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=33, attention_bias="dynamic_per_head"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    gather_nodes = [n for n in pruned_cpp.graph.node if n.op_type == "Gather"]
    assert len(gather_nodes) == 1
    assert gather_nodes[0].input[0] == "AttnBiasIn"

    node = _gqa_node(pruned_cpp)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert node.input[10] == gather_nodes[0].output[0]

    rng = np.random.default_rng(34)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    attn_bias_in = rng.standard_normal((1, cfg["H"], cfg["seq"], cfg["seq"])).astype(
        np.float32
    )
    (y,) = _run(pruned_cpp, {"X": x, "AttnBiasIn": attn_bias_in})
    assert y.shape == (cfg["batch"], cfg["seq"], cfg["Out"])


def test_cpp_gqa_pruning_unsafe_attention_bias_constant_is_declined():
    # A constant attention_bias whose axis-1 length is neither 1 (broadcast)
    # nor num_heads (genuine per-head) doesn't statically resolve -- declined
    # rather than guessed at, by both ports identically.
    model, cfg = _gqa_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=35, attention_bias="unsafe"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


def test_cpp_gqa_pruning_head_sink_constant_is_sliced():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=36, head_sink="nonempty")
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _gqa_node(pruned_cpp)
    num_heads, kv_num_heads = _gqa_attrs(node)
    sink_after = onnx.numpy_helper.to_array(
        next(t for t in pruned_cpp.graph.initializer if t.name == "HeadSink")
    )
    assert sink_after.shape == (num_heads,)


def test_cpp_gqa_pruning_dynamic_head_sink_is_left_untouched_block_still_pruned():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=37, head_sink="dynamic")
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    node = _gqa_node(pruned_cpp)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1


def test_cpp_gqa_pruning_wrong_shape_head_sink_constant_is_declined():
    model, cfg = _gqa_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=38, head_sink="wrong_shape"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


def test_cpp_gqa_pruning_qk_norm_weight_wrong_shape_is_declined_matches_python():
    # Regression coverage for a real, previously-undetected parity gap:
    # `FindGqaChains` used to never pass `qk_norm_weight_indices=(14, 15)` to
    # `FindSeparateQkvChains` (unlike pruning.py's own `_find_gqa_chains`, see
    # that function's own comment), so GroupQueryAttention's own
    # `q_norm_weight`/`k_norm_weight` shape was never validated here -- a
    # wrong-shaped connected pair would have been silently pruned through
    # instead of declining the whole match, unlike the pure-Python reference.
    # Now fixed: the whole match is declined here too, the node left
    # completely untouched, matching `apply_attention_head_pruning` exactly.
    model, cfg = _gqa_model(
        K=8,
        H=4,
        KVH=2,
        D=8,
        Out=6,
        seed=68,
        q_norm_weight=np.ones(9, dtype=np.float32),  # D + 1 -- wrong shape
        k_norm_weight=np.ones(9, dtype=np.float32),
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


def _gqa_cross_model(
    K_dec=8,
    K_enc=6,
    H=8,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    wq=None,
    wk=None,
    wv=None,
    wout=None,
):
    # Q fed from a different producer/input (`Xdec`) than K/V (`Xenc`) -- a
    # real, valid shape (e.g. encoder-decoder cross-attention): Q's own
    # producer weight has `K_dec` rows, K's/V's own has `K_enc`, genuinely
    # different row counts. Mirrors test_pruning.py's own `_gqa_cross_model`.
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wq is None:
        wq = rng.standard_normal((K_dec, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K_enc, Nkv)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K_enc, Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )

    body = f"""
        g (float[{batch},{seq},{K_dec}] Xdec, float[{batch},{seq},{K_enc}] Xenc) => (float[{batch},{seq},{Out}] Y)
        {{
          q = MatMul(Xdec, Wq)
          k = MatMul(Xenc, Wk)
          v = MatMul(Xenc, Wv)
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> (q, k, v, , , SeqLensK, TotalSeq)
          Y = MatMul(ctx, Wout)
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
        K_dec=K_dec,
        K_enc=K_enc,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        Nq=Nq,
        Nkv=Nkv,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def test_cpp_gqa_pruning_cross_attention_matches_oracle_exactly():
    # Without the `Kq`/`Kk`/`Kv` fix (a single shared `K` row-count reused to
    # index into `wk_kn`/`wv_kn` too), this either reads out of bounds or
    # produces a wrong ranking whenever K/V's producer has a different row
    # count than Q's -- K_dec=8 != K_enc=6 here is deliberate.
    model, cfg = _gqa_cross_model(K_dec=8, K_enc=6, H=8, KVH=2, D=8, Out=6, seed=20)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _gqa_node(pruned_cpp)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1  # max(1, 2 - round(2*0.5))
    assert num_heads == kv_num_heads * group_size

    keep_groups = _oracle_keep_groups_cross(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    oracle, _ = _gqa_cross_model(
        K_dec=cfg["K_dec"],
        K_enc=cfg["K_enc"],
        H=num_heads,
        KVH=kv_num_heads,
        D=d,
        Out=cfg["Out"],
        seed=20,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    rng = np.random.default_rng(21)
    xdec = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K_dec"])).astype(
        np.float32
    )
    xenc = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K_enc"])).astype(
        np.float32
    )
    (y_pruned,) = _run(pruned_cpp, {"Xdec": xdec, "Xenc": xenc})
    (y_oracle,) = _run(oracle, {"Xdec": xdec, "Xenc": xenc})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)

    # Sanity: Q and K/V really are independently sourced, not accidentally
    # both reading the same tensor -- perturbing Xenc alone (Xdec held fixed)
    # must still change the output.
    (y_pruned2,) = _run(pruned_cpp, {"Xdec": xdec, "Xenc": xenc + 1.0})
    assert not np.allclose(y_pruned, y_pruned2)


# --- Packed-QKV-then-Split + RoPE/QK-norm walk-back (GroupQueryAttention/ ---
# --- plain ai.onnx::Attention) -- FindSeparateQkvChains's own new
# --- WalkBackThroughQkNormRope/WalkBackThroughGemmaRopePair machinery,
# --- mirroring pruning.py's own `_walk_back_through_qk_norm_rope`/
# --- `_walk_back_through_gemma_rope_pair` exactly -------------------------
#
# A single packed MatMul/Gemm projection feeding a `Split` whose three
# outputs feed the fused attention op's own three separate query/key/value
# inputs, optionally through an intervening per-head `SimplifiedLayerNorm`
# sandwich and/or `RotaryEmbedding`/`MRotaryEmbedding`/`GemmaRotaryEmbedding`
# hop on Q's/K's own branch -- see MatchPackedQkvSplit's/
# WalkBackThroughQkNormRope's/WalkBackThroughGemmaRopePair's own comments in
# structured_pruning_entry.cpp for the exact topology and the real
# onnxruntime-genai model-builder export shape this was confirmed against.
# Every test below verifies byte-for-byte parity against the (still
# pure-Python, unaliased) `apply_attention_head_pruning` reference -- the
# same bar `test_cpp_decomposed_packed_qkv_pruning_matches_python_reference_
# exactly` above already holds the decomposed-attention family's own packed
# producer to.


def _gqa_packed_model(
    K=8,
    H=4,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    bias=False,
    wqkv=None,
    bqkv=None,
    wout=None,
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wqkv is None:
        wqkv = rng.standard_normal((K, Nq + 2 * Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)

    initializer = [_f32(wqkv, "Wqkv"), _f32(wout, "Wout")]
    qkv_op = "MatMul(X, Wqkv)"
    if bias:
        if bqkv is None:
            bqkv = rng.standard_normal((Nq + 2 * Nkv,)).astype(np.float32)
        initializer.append(_f32(bqkv, "Bqkv"))
        qkv_op = "Gemm(X, Wqkv, Bqkv)"

    split_sizes = np.array([Nq, Nkv, Nkv], dtype=np.int64)
    initializer.append(onnx.numpy_helper.from_array(split_sizes, "SplitSizes"))
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )

    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          qkv = {qkv_op}
          q, k, v = Split <axis = -1> (qkv, SplitSizes)
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> (q, k, v, , , SeqLensK, TotalSeq)
          Y = MatMul(ctx, Wout)
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
        wqkv=wqkv,
        bqkv=bqkv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def test_cpp_gqa_packed_qkv_split_pruning_matches_python_reference_exactly():
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    model, cfg = _gqa_packed_model(K=K, H=H, KVH=KVH, D=D, Out=Out, seed=301)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _gqa_node(pruned_cpp)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1
    assert num_heads == kv_num_heads * (H // KVH)

    Nq, Nkv = cfg["Nq"], cfg["Nkv"]
    wqkv = cfg["wqkv"]
    wq, wk, wv = wqkv[:, :Nq], wqkv[:, Nq : Nq + Nkv], wqkv[:, Nq + Nkv :]
    keep_groups = _oracle_keep_groups(wq, wk, wv, H, KVH, D, kv_num_heads)
    keep_q_heads = _group_q_heads(keep_groups, H // KVH)
    q_idx, kv_idx = _head_idx(keep_q_heads, D), _head_idx(keep_groups, D)
    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    expected_wqkv = np.concatenate([wq[:, q_idx], wk[:, kv_idx], wv[:, kv_idx]], axis=1)
    np.testing.assert_array_equal(inits["Wqkv"], expected_wqkv)
    np.testing.assert_array_equal(
        inits["SplitSizes"],
        np.array([len(q_idx), len(kv_idx), len(kv_idx)], dtype=np.int64),
    )


def test_cpp_gqa_packed_qkv_split_pruning_slices_packed_bias():
    K, H, KVH, D, Out = 8, 4, 2, 8, 6
    model, cfg = _gqa_packed_model(K=K, H=H, KVH=KVH, D=D, Out=Out, seed=302, bias=True)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _gqa_node(pruned_cpp)
    num_heads, kv_num_heads = _gqa_attrs(node)
    Nq, Nkv = cfg["Nq"], cfg["Nkv"]
    wqkv, bqkv = cfg["wqkv"], cfg["bqkv"]
    wq, wk, wv = wqkv[:, :Nq], wqkv[:, Nq : Nq + Nkv], wqkv[:, Nq + Nkv :]
    bq, bk, bv = bqkv[:Nq], bqkv[Nq : Nq + Nkv], bqkv[Nq + Nkv :]
    keep_groups = _oracle_keep_groups(wq, wk, wv, H, KVH, D, kv_num_heads)
    keep_q_heads = _group_q_heads(keep_groups, H // KVH)
    q_idx, kv_idx = _head_idx(keep_q_heads, D), _head_idx(keep_groups, D)
    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    expected_bqkv = np.concatenate([bq[q_idx], bk[kv_idx], bv[kv_idx]])
    np.testing.assert_array_equal(inits["Bqkv"], expected_bqkv)


def _qk_norm_rope_body(
    prefix,
    raw_name,
    gamma_name,
    reshape1_shape_name,
    reshape2_shape_name,
    with_norm,
    with_rope,
    cos_name="CosCache",
    sin_name="SinCache",
    pos_name="PosIds",
    interleaved=0,
    rotary_embedding_dim=0,
    rotary_num_heads=None,
):
    lines = []
    cur = raw_name
    if with_norm:
        lines.append(f"{prefix}_r1 = Reshape({cur}, {reshape1_shape_name})")
        lines.append(
            f"{prefix}_ln = SimplifiedLayerNormalization <axis=-1, epsilon=1e-6> "
            f"({prefix}_r1, {gamma_name})"
        )
        lines.append(f"{prefix}_normed = Reshape({prefix}_ln, {reshape2_shape_name})")
        cur = f"{prefix}_normed"
    if with_rope:
        nh = rotary_num_heads if rotary_num_heads is not None else 0
        lines.append(
            f"{prefix}_rot = com.microsoft.RotaryEmbedding "
            f"<num_heads={nh}, interleaved={interleaved}, "
            f"rotary_embedding_dim={rotary_embedding_dim}> "
            f"({cur}, {pos_name}, {cos_name}, {sin_name})"
        )
        cur = f"{prefix}_rot"
    return "\n          ".join(lines), cur


def _gqa_qk_norm_rope_model(
    K=8,
    H=8,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    with_norm=True,
    with_rope=True,
    interleaved=0,
    rotary_embedding_dim=0,
    q_rotary_num_heads=None,
    k_rotary_num_heads=None,
    wqkv=None,
    wout=None,
    q_gamma=None,
    k_gamma=None,
    cos=None,
    sin=None,
    position_ids=None,
    max_pos=32,
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wqkv is None:
        wqkv = rng.standard_normal((K, Nq + 2 * Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)
    initializer = [_f32(wqkv, "Wqkv"), _f32(wout, "Wout")]
    split_sizes = np.array([Nq, Nkv, Nkv], dtype=np.int64)
    initializer.append(onnx.numpy_helper.from_array(split_sizes, "SplitSizes"))
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )

    half = D // 2
    if cos is None:
        cos = rng.standard_normal((max_pos, half)).astype(np.float32)
    if sin is None:
        sin = rng.standard_normal((max_pos, half)).astype(np.float32)
    if position_ids is None:
        position_ids = np.tile(np.arange(seq, dtype=np.int64), (batch, 1))
    initializer += [_f32(cos, "CosCache"), _f32(sin, "SinCache")]
    initializer.append(onnx.numpy_helper.from_array(position_ids, "PosIds"))
    if q_gamma is None:
        q_gamma = rng.standard_normal((D,)).astype(np.float32)
    if k_gamma is None:
        k_gamma = rng.standard_normal((D,)).astype(np.float32)
    initializer += [_f32(q_gamma, "QGamma"), _f32(k_gamma, "KGamma")]
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array([0, -1, D], dtype=np.int64), "QReshape1Shape"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array([0, -1, Nq], dtype=np.int64), "QReshape2Shape"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array([0, -1, D], dtype=np.int64), "KReshape1Shape"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array([0, -1, Nkv], dtype=np.int64), "KReshape2Shape"
        )
    )

    rope_kwargs = dict(
        interleaved=interleaved, rotary_embedding_dim=rotary_embedding_dim
    )
    q_text, q_body = _qk_norm_rope_body(
        "q",
        "q_raw",
        "QGamma",
        "QReshape1Shape",
        "QReshape2Shape",
        with_norm,
        with_rope,
        rotary_num_heads=q_rotary_num_heads,
        **rope_kwargs,
    )
    k_text, k_body = _qk_norm_rope_body(
        "k",
        "k_raw",
        "KGamma",
        "KReshape1Shape",
        "KReshape2Shape",
        with_norm,
        with_rope,
        rotary_num_heads=k_rotary_num_heads,
        **rope_kwargs,
    )
    hop_body = q_text + "\n          " + k_text

    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          qkv = MatMul(X, Wqkv)
          q_raw, k_raw, v = Split <axis = -1> (qkv, SplitSizes)
          {hop_body}
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> ({q_body}, {k_body}, v, , , SeqLensK, TotalSeq)
          Y = MatMul(ctx, Wout)
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
        wqkv=wqkv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def test_cpp_gqa_packed_qkv_qk_norm_rope_pruning_matches_python_reference_exactly():
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    model, cfg = _gqa_qk_norm_rope_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        seed=303,
        with_norm=True,
        with_rope=True,
        interleaved=1,
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    # No onnx.checker.check_model here -- `SimplifiedLayerNormalization`
    # registers under the default ("") domain but isn't in onnx's own
    # schema registry at all (pruning.py's own "Attention-head pruning"
    # section comment/`_norm_pass_through_const_names`'s own docstring),
    # so the checker itself declines this exact op/domain combination
    # regardless of this port's own correctness -- matching
    # test_pruning.py's own `test_gqa_packed_qkv_qk_norm_rope_pruning_
    # matches_oracle_exactly`, which likewise never calls the checker.
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _gqa_node(pruned_cpp)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1
    assert num_heads == kv_num_heads * (H // KVH)
    rotary_nodes = [n for n in pruned_cpp.graph.node if n.op_type == "RotaryEmbedding"]
    assert len(rotary_nodes) == 2


def test_cpp_gqa_packed_qkv_qk_norm_rope_pruning_updates_explicit_rotary_num_heads():
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    model, cfg = _gqa_qk_norm_rope_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        seed=304,
        with_norm=True,
        with_rope=True,
        q_rotary_num_heads=H,
        k_rotary_num_heads=KVH,
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    q_rot = next(
        n
        for n in pruned_cpp.graph.node
        if n.op_type == "RotaryEmbedding" and n.input[0] == "q_normed"
    )
    k_rot = next(
        n
        for n in pruned_cpp.graph.node
        if n.op_type == "RotaryEmbedding" and n.input[0] == "k_normed"
    )
    node = _gqa_node(pruned_cpp)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert next(a.i for a in q_rot.attribute if a.name == "num_heads") == num_heads
    assert next(a.i for a in k_rot.attribute if a.name == "num_heads") == kv_num_heads


def test_cpp_gqa_packed_qkv_norm_only_pruning_matches_python_reference_exactly():
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    model, cfg = _gqa_qk_norm_rope_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        seed=305,
        with_norm=True,
        with_rope=False,
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    # No onnx.checker.check_model here -- see the qk_norm_rope test above.
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_gqa_packed_qkv_rope_only_pruning_matches_python_reference_exactly():
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    model, cfg = _gqa_qk_norm_rope_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        seed=306,
        with_norm=False,
        with_rope=True,
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def _gqa_mrope_model(
    K=8,
    H=8,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    mrope_section=(2, 1, 1),
    mrope_layout=0,
    wqkv=None,
    wout=None,
    cos=None,
    sin=None,
    position_ids=None,
    max_pos=32,
    q_rotary_num_heads=None,
    k_rotary_num_heads=None,
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    half = D // 2
    if wqkv is None:
        wqkv = rng.standard_normal((K, Nq + 2 * Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)
    initializer = [_f32(wqkv, "Wqkv"), _f32(wout, "Wout")]
    split_sizes = np.array([Nq, Nkv, Nkv], dtype=np.int64)
    initializer.append(onnx.numpy_helper.from_array(split_sizes, "SplitSizes"))
    if cos is None:
        cos = rng.standard_normal((max_pos, half)).astype(np.float32)
    if sin is None:
        sin = rng.standard_normal((max_pos, half)).astype(np.float32)
    if position_ids is None:
        p = np.tile(np.arange(seq, dtype=np.int64), (batch, 1))
        position_ids = np.stack([p, p, p], axis=0)
    initializer += [_f32(cos, "Cos"), _f32(sin, "Sin")]
    initializer.append(onnx.numpy_helper.from_array(position_ids, "PosIds"))
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )
    q_nh = q_rotary_num_heads if q_rotary_num_heads is not None else H
    k_nh = k_rotary_num_heads if k_rotary_num_heads is not None else KVH
    section_str = "[" + ", ".join(str(s) for s in mrope_section) + "]"
    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          qkv = MatMul(X, Wqkv)
          q_raw, k_raw, v = Split <axis = -1> (qkv, SplitSizes)
          q_rot = com.microsoft.MRotaryEmbedding
            <num_heads={q_nh}, mrope_section={section_str}, mrope_layout={mrope_layout}>
            (q_raw, PosIds, Cos, Sin)
          k_rot = com.microsoft.MRotaryEmbedding
            <num_heads={k_nh}, mrope_section={section_str}, mrope_layout={mrope_layout}>
            (k_raw, PosIds, Cos, Sin)
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> (q_rot, k_rot, v, , , SeqLensK, TotalSeq)
          Y = MatMul(ctx, Wout)
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
        wqkv=wqkv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def test_cpp_gqa_packed_qkv_mrope_pruning_matches_python_reference_exactly():
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    model, cfg = _gqa_mrope_model(K=K, H=H, KVH=KVH, D=D, Out=Out, seed=307)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_gqa_packed_qkv_mrope_pruning_updates_explicit_rotary_num_heads():
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    model, cfg = _gqa_mrope_model(K=K, H=H, KVH=KVH, D=D, Out=Out, seed=308)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _gqa_node(pruned_cpp)
    num_heads, kv_num_heads = _gqa_attrs(node)
    q_rot = next(
        n
        for n in pruned_cpp.graph.node
        if n.op_type == "MRotaryEmbedding" and n.input[0] == "q_raw"
    )
    k_rot = next(
        n
        for n in pruned_cpp.graph.node
        if n.op_type == "MRotaryEmbedding" and n.input[0] == "k_raw"
    )
    assert next(a.i for a in q_rot.attribute if a.name == "num_heads") == num_heads
    assert next(a.i for a in k_rot.attribute if a.name == "num_heads") == kv_num_heads


def _gqa_gemma_rope_model(
    K=8,
    H=8,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    wqkv=None,
    wout=None,
    emb=None,
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wqkv is None:
        wqkv = rng.standard_normal((K, Nq + 2 * Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)
    initializer = [_f32(wqkv, "Wqkv"), _f32(wout, "Wout")]
    split_sizes = np.array([Nq, Nkv, Nkv], dtype=np.int64)
    initializer.append(onnx.numpy_helper.from_array(split_sizes, "SplitSizes"))
    if emb is None:
        emb = rng.standard_normal((batch, seq, D)).astype(np.float32)
    initializer.append(_f32(emb, "Emb"))
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array([D // 2, D // 2], dtype=np.int64), "HalfSizes"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array([0, -1, H, D], dtype=np.int64), "QReshapeA"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array([0, -1, KVH, D], dtype=np.int64), "KReshapeA"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array([0, -1, Nq], dtype=np.int64), "QReshapeB")
    )
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array([0, -1, Nkv], dtype=np.int64), "KReshapeB"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )
    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          qkv = MatMul(X, Wqkv)
          q_raw, k_raw, v = Split <axis = -1> (qkv, SplitSizes)
          q_bnsh_in = Reshape(q_raw, QReshapeA)
          q_bnsh = Transpose <perm=[0,2,1,3]> (q_bnsh_in)
          q_x1, q_x2 = Split <axis=-1> (q_bnsh, HalfSizes)
          q_x2n = Neg(q_x2)
          q_rot_in = Concat <axis=-1> (q_x2n, q_x1)
          k_bnsh_in = Reshape(k_raw, KReshapeA)
          k_bnsh = Transpose <perm=[0,2,1,3]> (k_bnsh_in)
          k_x1, k_x2 = Split <axis=-1> (k_bnsh, HalfSizes)
          k_x2n = Neg(k_x2)
          k_rot_in = Concat <axis=-1> (k_x2n, k_x1)
          q_embed, k_embed = com.microsoft.GemmaRotaryEmbedding (Emb, q_bnsh, q_rot_in, k_bnsh, k_rot_in)
          q_flat_bnsh = Transpose <perm=[0,2,1,3]> (q_embed)
          q_flat = Reshape(q_flat_bnsh, QReshapeB)
          k_flat_bnsh = Transpose <perm=[0,2,1,3]> (k_embed)
          k_flat = Reshape(k_flat_bnsh, KReshapeB)
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> (q_flat, k_flat, v, , , SeqLensK, TotalSeq)
          Y = MatMul(ctx, Wout)
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
        wqkv=wqkv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def test_cpp_gqa_packed_qkv_gemma_rope_pruning_matches_python_reference_exactly():
    # `GemmaRotaryEmbedding` has no usable CPU kernel in this environment (see
    # pruning.py's own "Attention-head pruning" section comment) -- so, unlike
    # every other test in this section, this one is verified purely by
    # byte-for-byte parity against the pure-Python reference rather than a
    # real `InferenceSession` run (`test_gqa_packed_qkv_gemma_rope_pruning_
    # matches_decomposed_oracle` in test_pruning.py already covers the
    # decomposed-execution oracle check for the shared Python logic itself).
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    model, cfg = _gqa_gemma_rope_model(K=K, H=H, KVH=KVH, D=D, Out=Out, seed=309)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _gqa_node(pruned_cpp)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1
    assert num_heads == kv_num_heads * (H // KVH)


def _onnx_attention_packed_rope_model(
    K=8,
    H=8,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    rope_domain="",
    q_rotary_num_heads=None,
    k_rotary_num_heads=None,
    wqkv=None,
    wout=None,
    cos=None,
    sin=None,
    position_ids=None,
    max_pos=32,
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wqkv is None:
        wqkv = rng.standard_normal((K, Nq + 2 * Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)
    initializer = [_f32(wqkv, "Wqkv"), _f32(wout, "Wout")]
    split_sizes = np.array([Nq, Nkv, Nkv], dtype=np.int64)
    initializer.append(onnx.numpy_helper.from_array(split_sizes, "SplitSizes"))

    half = D // 2
    if cos is None:
        cos = rng.standard_normal((max_pos, half)).astype(np.float32)
    if sin is None:
        sin = rng.standard_normal((max_pos, half)).astype(np.float32)
    if position_ids is None:
        position_ids = np.tile(np.arange(seq, dtype=np.int64), (batch, 1))
    initializer += [_f32(cos, "CosCache"), _f32(sin, "SinCache")]
    initializer.append(onnx.numpy_helper.from_array(position_ids, "PosIds"))

    qnh = q_rotary_num_heads if q_rotary_num_heads is not None else 0
    knh = k_rotary_num_heads if k_rotary_num_heads is not None else 0
    domain_prefix = f"{rope_domain}." if rope_domain else ""
    opset_imports = {"": 24}
    if rope_domain == "":
        hop_body = (
            f"q = {domain_prefix}RotaryEmbedding <num_heads={qnh}> "
            f"(q_raw, CosCache, SinCache, PosIds)\n"
            f"          k = {domain_prefix}RotaryEmbedding <num_heads={knh}> "
            f"(k_raw, CosCache, SinCache, PosIds)"
        )
    else:
        hop_body = (
            f"q = {domain_prefix}RotaryEmbedding <num_heads={qnh}> "
            f"(q_raw, PosIds, CosCache, SinCache)\n"
            f"          k = {domain_prefix}RotaryEmbedding <num_heads={knh}> "
            f"(k_raw, PosIds, CosCache, SinCache)"
        )
        opset_imports[rope_domain] = 1

    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          qkv = MatMul(X, Wqkv)
          q_raw, k_raw, v = Split <axis = -1> (qkv, SplitSizes)
          {hop_body}
          ctx = Attention <q_num_heads={H}, kv_num_heads={KVH}> (q, k, v)
          Y = MatMul(ctx, Wout)
        }}
        """
    opset_import_text = (
        "[" + ", ".join(f'"{d}": {v}' for d, v in opset_imports.items()) + "]"
    )
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: {opset_import_text}
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
        wqkv=wqkv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def _onnx_attention_node(model):
    return next(
        n for n in model.graph.node if n.domain == "" and n.op_type == "Attention"
    )


def _onnx_attention_attrs(node):
    q_num_heads = next(a.i for a in node.attribute if a.name == "q_num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return q_num_heads, kv_num_heads


def test_cpp_onnx_attention_packed_qkv_native_rotary_embedding_pruning_matches_python_reference_exactly():
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    model, cfg = _onnx_attention_packed_rope_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        seed=310,
        rope_domain="",
        q_rotary_num_heads=H,
        k_rotary_num_heads=KVH,
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    q_num_heads, kv_num_heads = _onnx_attention_attrs(_onnx_attention_node(pruned_cpp))
    assert kv_num_heads == 1
    assert q_num_heads == kv_num_heads * (H // KVH)
    rotary_nodes = [n for n in pruned_cpp.graph.node if n.op_type == "RotaryEmbedding"]
    assert len(rotary_nodes) == 2
    for n in rotary_nodes:
        assert n.domain == ""


def test_cpp_onnx_attention_packed_qkv_contrib_rotary_embedding_pruning_matches_python_reference_exactly():
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    model, cfg = _onnx_attention_packed_rope_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        seed=311,
        rope_domain="com.microsoft",
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    q_num_heads, kv_num_heads = _onnx_attention_attrs(_onnx_attention_node(pruned_cpp))
    assert kv_num_heads == 1
    assert q_num_heads == kv_num_heads * (H // KVH)
    rotary_nodes = [n for n in pruned_cpp.graph.node if n.op_type == "RotaryEmbedding"]
    assert len(rotary_nodes) == 2
    for n in rotary_nodes:
        assert n.domain == "com.microsoft"


# --- True MQA (kv_num_heads == 1) fused GroupQueryAttention fast path ------
#
# Regression coverage for the bug the Python reference (pruning.py's own
# `_apply_one_gqa_chain`) fixed and this C++ port (`ApplyOneGqaChain` in
# structured_pruning_entry.cpp) now mirrors: with `kv_num_heads == 1`, the
# ordinary KV-*group* formula (`max(1, kv_num_heads - round(kv_num_heads *
# sparsity))`) is always `1 == kv_num_heads` for every `sparsity` in
# `[0, 1)`, so `ApplyOneGqaChain` used to hit its own "nothing to prune"
# early exit and leave the whole fused block byte-for-byte untouched no
# matter how many query heads (`num_heads`) shared that one KV head -- a
# complete no-op for every real-world MQA export (Falcon-7B/StarCoder/
# PaLM-family-style `kv_num_heads=1`). `ApplyOneGqaChain` now has a
# dedicated fast path for this case: individual query heads are ranked and
# dropped directly, leaving the sole KV head -- and both its own K/V
# producer weights and `kv_num_heads` itself -- completely untouched.


def _oracle_keep_fused_mqa_query_heads(wq, num_heads, head_size, keep_count):
    importance = np.zeros(num_heads)
    for h in range(num_heads):
        importance[h] = np.linalg.norm(wq[:, h * head_size : (h + 1) * head_size])
    return np.sort(np.argsort(-importance)[:keep_count])


def test_cpp_mqa_pruning_shrinks_query_heads_kv_fully_untouched():
    model, cfg = _gqa_model(K=8, H=8, KVH=1, D=8, Out=6, seed=5)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1  # untouched -- true MQA always has exactly one
    assert num_heads == 4  # max(1, 8 - round(8*0.5)) query heads dropped directly

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["Wq"].shape == (cfg["K"], 4 * cfg["D"])
    assert inits["Wout"].shape == (4 * cfg["D"], cfg["Out"])
    # K/V producer weights completely untouched -- same shape AND same
    # values as the original, unpruned model (the confirmed-fixed bug's own
    # "byte-identical in/out" no-op signature, but now scoped to just K/V,
    # not the whole block).
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"])


def test_cpp_mqa_pruning_is_not_a_complete_no_op():
    # The confirmed bug itself: before this fix, this call was byte-identical
    # to `model` for any `sparsity` (since `kv_num_heads == 1` always forced
    # the group formula's early exit). It must not be anymore.
    model, cfg = _gqa_model(K=8, H=8, KVH=1, D=8, Out=6, seed=6)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.75)
    assert pruned.SerializeToString() != model.SerializeToString()

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1
    assert num_heads == 2  # max(1, 8 - round(8*0.75)) < original 8


def test_cpp_mqa_pruning_zero_sparsity_is_a_no_op():
    model, cfg = _gqa_model(K=8, H=8, KVH=1, D=8, Out=6, seed=7)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.0)
    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert num_heads == cfg["H"]
    assert kv_num_heads == cfg["KVH"]
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"])


def test_cpp_mqa_pruning_matches_oracle_exactly():
    model, cfg = _gqa_model(K=8, H=8, KVH=1, D=8, Out=6, seed=9)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1
    assert num_heads == 4

    d = cfg["D"]
    keep_q_heads = _oracle_keep_fused_mqa_query_heads(cfg["wq"], cfg["H"], d, num_heads)
    q_idx = _head_idx(keep_q_heads, d)

    oracle, _ = _gqa_model(
        K=cfg["K"],
        H=num_heads,
        KVH=kv_num_heads,
        D=d,
        Out=cfg["Out"],
        seed=9,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"],
        wv=cfg["wv"],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    rng = np.random.default_rng(10)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_mqa_pruning_slices_bias_when_producer_has_one():
    model, cfg = _gqa_model(K=8, H=4, KVH=1, D=8, Out=6, seed=14, bias=True)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1
    assert num_heads == 2  # max(1, 4 - round(4*0.5))

    d = cfg["D"]
    keep_q_heads = _oracle_keep_fused_mqa_query_heads(cfg["wq"], cfg["H"], d, num_heads)
    q_idx = _head_idx(keep_q_heads, d)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"])
    np.testing.assert_array_equal(inits["Bq"], cfg["bq"][q_idx])
    np.testing.assert_array_equal(inits["Bk"], cfg["bk"])
    np.testing.assert_array_equal(inits["Bv"], cfg["bv"])


def test_cpp_attention_head_pruning_group_query_attention_missing_required_inputs_is_left_untouched():
    K, H, D, Out = 8, 4, 4, 6
    Nqkv = H * D
    rng = np.random.default_rng(5)
    wq = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wk = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wv = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wout = rng.standard_normal((Nqkv, Out)).astype(np.float32)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        g (float[2,5,{K}] X) => (float[2,5,{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx, present_k, present_v = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={H}> (q, k, v)
          Y = MatMul(ctx, Wout)
        }}
        """
    )
    model.graph.initializer.extend(
        [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    )

    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


# --- Plain ai.onnx::Attention (opset 24+) -----------------------------------


def _onnx_attention_model(
    K=8,
    H=4,
    KVH=2,
    D=4,
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
    attn_mask=None,  # None (omitted) | "nonempty" (constant, rank-2 broadcast)
    #  | "dynamic" (graph input, rank-2 broadcast) | "per_head" (constant,
    #  rank-3, axis0 == q_num_heads, sliced) | "dynamic_per_head" (graph
    #  input, same shape -- Gather inserted) | "unsafe" (axis0 neither 1 nor
    #  q_num_heads -- declines the match)
    past_kv=None,  # None (omitted) | "nonempty" (constant, sliced) | "dynamic"
    #  (graph input) | "unsafe" (constant, wrong kv_num_heads-axis length --
    #  declines the match)
    Dv=None,  # V's own head_size, if it should genuinely differ from D -- this
    #  op's real schema (unlike com.microsoft::GroupQueryAttention) allows it;
    #  defaults to D (the uniform case every other caller wants). The raw
    #  output/output-projection's own reduction dim is sized off Dv, not D.
):
    if Dv is None:
        Dv = D
    rng = np.random.default_rng(seed)
    Nq, Nkv, Nv = H * D, KVH * D, KVH * Dv
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nkv)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((H * Dv, Out)).astype(np.float32)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    q_op, k_op, v_op = "MatMul(X, Wq)", "MatMul(X, Wk)", "MatMul(X, Wv)"
    if bias:
        if bq is None:
            bq = rng.standard_normal((Nq,)).astype(np.float32)
        if bk is None:
            bk = rng.standard_normal((Nkv,)).astype(np.float32)
        if bv is None:
            bv = rng.standard_normal((Nv,)).astype(np.float32)
        initializer += [_f32(bq, "Bq"), _f32(bk, "Bk"), _f32(bv, "Bv")]
        q_op, k_op, v_op = "Gemm(X, Wq, Bq)", "Gemm(X, Wk, Bk)", "Gemm(X, Wv, Bv)"

    operands = ["q", "k", "v"]
    extra_graph_inputs = ""

    if attn_mask == "nonempty":
        mask = np.zeros((seq, seq), dtype=np.float32)
        initializer.append(_f32(mask, "AttnMask"))
        operands.append("AttnMask")
    elif attn_mask == "dynamic":
        operands.append("AttnMaskIn")
        extra_graph_inputs += f", float[{seq},{seq}] AttnMaskIn"
    elif attn_mask == "per_head":
        mask = rng.standard_normal((H, seq, seq)).astype(np.float32)
        initializer.append(_f32(mask, "AttnMask"))
        operands.append("AttnMask")
    elif attn_mask == "dynamic_per_head":
        operands.append("AttnMaskIn")
        extra_graph_inputs += f", float[{H},{seq},{seq}] AttnMaskIn"
    elif attn_mask == "unsafe":
        # axis0 length neither 1 (broadcast) nor q_num_heads -- unresolvable,
        # declines the whole match rather than guessing.
        mask = rng.standard_normal((H + 1, seq, seq)).astype(np.float32)
        initializer.append(_f32(mask, "AttnMask"))
        operands.append("AttnMask")
    else:
        operands.append("")

    if past_kv == "nonempty":
        past_key = rng.standard_normal((batch, KVH, 1, D)).astype(np.float32)
        past_value = rng.standard_normal((batch, KVH, 1, Dv)).astype(np.float32)
        initializer += [_f32(past_key, "PastKey"), _f32(past_value, "PastValue")]
        operands += ["PastKey", "PastValue"]
    elif past_kv == "dynamic":
        operands += ["PastKeyIn", "PastValueIn"]
        extra_graph_inputs += (
            f", float[{batch},{KVH},1,{D}] PastKeyIn"
            f", float[{batch},{KVH},1,{Dv}] PastValueIn"
        )
    elif past_kv == "unsafe":
        past_key = rng.standard_normal((batch, KVH + 1, 1, D)).astype(np.float32)
        past_value = rng.standard_normal((batch, KVH + 1, 1, Dv)).astype(np.float32)
        initializer += [_f32(past_key, "PastKey"), _f32(past_value, "PastValue")]
        operands += ["PastKey", "PastValue"]
    else:
        operands += ["", ""]

    while operands and operands[-1] == "":
        operands.pop()

    if with_reshape:
        shape = np.array([batch, seq, H * Dv], dtype=np.int64)
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
          ctx = Attention <q_num_heads={H}, kv_num_heads={KVH}> ({", ".join(operands)})
          {tail}
        }}
        """

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 24]
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
        Dv=Dv,
        Out=Out,
        Nq=Nq,
        Nkv=Nkv,
        Nv=Nv,
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


def _onnx_attention_node(model):
    return next(
        n for n in model.graph.node if n.op_type == "Attention" and n.domain == ""
    )


def _onnx_attention_attrs(node):
    q_num_heads = next(a.i for a in node.attribute if a.name == "q_num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return q_num_heads, kv_num_heads


def test_cpp_onnx_attention_pruning_shrinks_matched_block():
    model, cfg = _onnx_attention_model(K=8, H=4, KVH=2, D=4, Out=6)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert kv_num_heads == 1
    assert q_num_heads == kv_num_heads * (cfg["H"] // cfg["KVH"])


def test_cpp_onnx_attention_pruning_matches_oracle_exactly():
    model, cfg = _onnx_attention_model(K=8, H=8, KVH=2, D=4, Out=6, seed=1)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1
    assert q_num_heads == kv_num_heads * group_size

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    oracle, _ = _onnx_attention_model(
        K=cfg["K"],
        H=len(keep_q_heads),
        KVH=len(keep_groups),
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


def test_cpp_onnx_attention_pruning_slices_bias_when_producer_has_one():
    model, cfg = _onnx_attention_model(K=8, H=4, KVH=2, D=4, Out=6, seed=14, bias=True)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1
    assert q_num_heads == group_size

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


def test_cpp_onnx_attention_pruning_nonempty_2d_attn_mask_constant_is_pruned():
    # A rank-2 (seq, seq) attn_mask never has a q_num_heads axis at all -- an
    # unconditional broadcast -- so it needs no slicing and is left
    # byte-for-byte untouched, but (unlike the OLD, overly-conservative C++
    # behavior this fix replaces -- this test used to be named
    # `..._is_left_untouched` and asserted the WHOLE model was unchanged) the
    # rest of the block IS still pruned.
    model, cfg = _onnx_attention_model(
        K=8, H=4, KVH=2, D=4, Out=6, seed=17, attn_mask="nonempty"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _onnx_attention_node(pruned_cpp)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert kv_num_heads == 1
    assert q_num_heads == kv_num_heads * (cfg["H"] // cfg["KVH"])
    mask_before = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "AttnMask")
    )
    mask_after = onnx.numpy_helper.to_array(
        next(t for t in pruned_cpp.graph.initializer if t.name == "AttnMask")
    )
    np.testing.assert_array_equal(mask_before, mask_after)


def test_cpp_onnx_attention_pruning_per_head_attn_mask_constant_is_sliced():
    model, cfg = _onnx_attention_model(
        K=8, H=4, KVH=2, D=4, Out=6, seed=41, attn_mask="per_head"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _onnx_attention_node(pruned_cpp)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    mask_after = onnx.numpy_helper.to_array(
        next(t for t in pruned_cpp.graph.initializer if t.name == "AttnMask")
    )
    assert mask_after.shape == (q_num_heads, cfg["seq"], cfg["seq"])


def test_cpp_onnx_attention_pruning_dynamic_per_head_attn_mask_gets_gather_inserted():
    model, cfg = _onnx_attention_model(
        K=8, H=4, KVH=2, D=4, Out=6, seed=42, attn_mask="dynamic_per_head"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    gather_nodes = [n for n in pruned_cpp.graph.node if n.op_type == "Gather"]
    assert len(gather_nodes) == 1
    assert gather_nodes[0].input[0] == "AttnMaskIn"

    node = _onnx_attention_node(pruned_cpp)
    assert node.input[3] == gather_nodes[0].output[0]

    rng = np.random.default_rng(43)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    attn_mask_in = rng.standard_normal((cfg["H"], cfg["seq"], cfg["seq"])).astype(
        np.float32
    )
    (y,) = _run(pruned_cpp, {"X": x, "AttnMaskIn": attn_mask_in})
    assert y.shape == (cfg["batch"], cfg["seq"], cfg["Out"])


def test_cpp_onnx_attention_pruning_unsafe_attn_mask_constant_is_declined():
    model, cfg = _onnx_attention_model(
        K=8, H=4, KVH=2, D=4, Out=6, seed=44, attn_mask="unsafe"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


def test_cpp_onnx_attention_pruning_nonempty_sliceable_past_kv_constant_is_sliced():
    model, cfg = _onnx_attention_model(
        K=8, H=8, KVH=2, D=4, Out=6, seed=45, past_kv="nonempty"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _onnx_attention_node(pruned_cpp)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    assert inits["PastKey"].shape == (cfg["batch"], kv_num_heads, 1, cfg["D"])
    assert inits["PastValue"].shape == (cfg["batch"], kv_num_heads, 1, cfg["D"])


def test_cpp_onnx_attention_pruning_past_kv_constant_is_unsafe_and_declined():
    model, cfg = _onnx_attention_model(
        K=8, H=8, KVH=2, D=4, Out=6, seed=46, past_kv="unsafe"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


def test_cpp_onnx_attention_pruning_dynamic_past_kv_input_is_still_pruned():
    model, cfg = _onnx_attention_model(
        K=8, H=8, KVH=2, D=4, Out=6, seed=47, past_kv="dynamic"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    node = _onnx_attention_node(pruned_cpp)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert kv_num_heads == 1


def test_cpp_onnx_attention_pruning_diff_v_head_size_matches_oracle_exactly():
    # This op's real schema (unlike GroupQueryAttention, which fuse_gqa.h
    # always emits with equal Q/K/V head_size) genuinely allows V its own,
    # independent head_size -- confirmed via the op's own backend-test suite
    # and real onnxruntime execution. `FindSeparateQkvChains` now takes an
    # `allow_differing_v_head_size` parameter (`true` for this op only,
    # mirroring pruning.py's own `_find_onnx_attention_chains`), `AttnChain`
    # carries Q's/K's shared `head_size` and V's own (possibly different)
    # `v_head_size` as separate fields, and `ApplyOneGqaChain` slices Q's/K's
    # own producer weight at `head_size` while V's own producer weight -- and
    # the output projection's own reduction dim, and the raw output's own
    # width -- at `v_head_size`. Each KV group's own Q+K+V block is scaled by
    # a distinct, well-separated factor so which 2 of 4 groups the importance
    # ranking keeps is unambiguous.
    K, H, KVH, D, Dv, Out = 8, 8, 4, 4, 6, 5
    group_size = H // KVH
    rng = np.random.default_rng(19)
    wq = rng.standard_normal((K, H * D)).astype(np.float32)
    wk = rng.standard_normal((K, KVH * D)).astype(np.float32)
    wv = rng.standard_normal((K, KVH * Dv)).astype(np.float32)
    wout = rng.standard_normal((H * Dv, Out)).astype(np.float32)

    scales = [3.0, 0.1, 2.0, 0.05]
    for kv, scale in enumerate(scales):
        for h in range(kv * group_size, (kv + 1) * group_size):
            wq[:, h * D : (h + 1) * D] *= scale
        wk[:, kv * D : (kv + 1) * D] *= scale
        wv[:, kv * Dv : (kv + 1) * Dv] *= scale

    model, cfg = _onnx_attention_model(
        K=K, H=H, KVH=KVH, D=D, Dv=Dv, Out=Out, seed=19, wq=wq, wk=wk, wv=wv, wout=wout
    )
    onnx.checker.check_model(model)

    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _onnx_attention_node(pruned_cpp)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert kv_num_heads == 2  # max(1, 4 - round(4*0.5))
    assert q_num_heads == kv_num_heads * group_size

    keep_groups = _oracle_keep_groups(
        wq, wk, wv, H, KVH, D, kv_num_heads, v_head_size=Dv
    )
    assert list(keep_groups) == [0, 2]

    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx = _head_idx(keep_q_heads, D)  # Q's own producer weight columns
    kv_idx = _head_idx(keep_groups, D)  # K's own producer weight columns
    v_idx = _head_idx(keep_groups, Dv)  # V's own producer weight columns
    y_idx = _head_idx(keep_q_heads, Dv)  # output/consumer-side columns

    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    np.testing.assert_array_equal(inits["Wq"], wq[:, q_idx])
    np.testing.assert_array_equal(inits["Wk"], wk[:, kv_idx])
    np.testing.assert_array_equal(inits["Wv"], wv[:, v_idx])
    np.testing.assert_array_equal(inits["Wout"], wout[y_idx, :])

    oracle, _ = _onnx_attention_model(
        K=K,
        H=q_num_heads,
        KVH=kv_num_heads,
        D=D,
        Dv=Dv,
        Out=Out,
        seed=19,
        wq=wq[:, q_idx],
        wk=wk[:, kv_idx],
        wv=wv[:, v_idx],
        wout=wout[y_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )
    onnx.checker.check_model(oracle)

    rng2 = np.random.default_rng(22)
    x = rng2.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    (y_pruned,) = _run(pruned_cpp, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def _onnx_attention_cross_model(
    K_dec=8,
    K_enc=6,
    H=8,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    wq=None,
    wk=None,
    wv=None,
    wout=None,
):
    # Q fed from a different producer/input (`Xdec`) than K/V (`Xenc`) --
    # mirrors `_gqa_cross_model` above, for the plain ai.onnx op instead.
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wq is None:
        wq = rng.standard_normal((K_dec, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K_enc, Nkv)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K_enc, Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    body = f"""
        g (float[{batch},{seq},{K_dec}] Xdec, float[{batch},{seq},{K_enc}] Xenc) => (float[{batch},{seq},{Out}] Y)
        {{
          q = MatMul(Xdec, Wq)
          k = MatMul(Xenc, Wk)
          v = MatMul(Xenc, Wv)
          ctx = Attention <q_num_heads={H}, kv_num_heads={KVH}> (q, k, v)
          Y = MatMul(ctx, Wout)
        }}
        """

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 24]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K_dec=K_dec,
        K_enc=K_enc,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        Nq=Nq,
        Nkv=Nkv,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def test_cpp_onnx_attention_pruning_cross_attention_matches_oracle_exactly():
    model, cfg = _onnx_attention_cross_model(
        K_dec=8, K_enc=6, H=8, KVH=2, D=8, Out=6, seed=24
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _onnx_attention_node(pruned_cpp)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1  # max(1, 2 - round(2*0.5))
    assert q_num_heads == kv_num_heads * group_size

    keep_groups = _oracle_keep_groups_cross(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    oracle, _ = _onnx_attention_cross_model(
        K_dec=cfg["K_dec"],
        K_enc=cfg["K_enc"],
        H=q_num_heads,
        KVH=kv_num_heads,
        D=d,
        Out=cfg["Out"],
        seed=24,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    rng = np.random.default_rng(25)
    xdec = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K_dec"])).astype(
        np.float32
    )
    xenc = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K_enc"])).astype(
        np.float32
    )
    (y_pruned,) = _run(pruned_cpp, {"Xdec": xdec, "Xenc": xenc})
    (y_oracle,) = _run(oracle, {"Xdec": xdec, "Xenc": xenc})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)

    (y_pruned2,) = _run(pruned_cpp, {"Xdec": xdec, "Xenc": xenc + 1.0})
    assert not np.allclose(y_pruned, y_pruned2)


# --- Cross-check against the pure-Python reference --------------------------


def test_cpp_attention_head_pruning_matches_python_reference_output():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6, seed=21)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    rng = np.random.default_rng(22)
    x = rng.standard_normal((2, 5, 8)).astype(np.float32)
    (y_py,) = _run(pruned_py, {"X": x})
    (y_cpp,) = _run(pruned_cpp, {"X": x})
    np.testing.assert_allclose(y_py, y_cpp, rtol=1e-4, atol=1e-4)


# --- Subgraph recursion (If) -------------------------------------------------
#
# Covers `structured_pruning_entry.cpp`'s own `IterSubgraphs` and the
# `ApplyAttentionHeadPruning` loop built on it -- a straight C++ port of
# `onnxsim/pruning.py`'s own `_iter_subgraphs`/`apply_attention_head_pruning`
# subgraph-recursion round (see that module's "Subgraph recursion" section
# comment, and `structured_pruning_entry.cpp`'s own copy of it above
# `IterSubgraphs`'s definition, for the full design rationale). Model shape
# mirrors `tests/test_pruning.py`'s own `_if_attention_model`/
# `test_attention_head_pruning_prunes_blocks_inside_if_branches` fixture,
# just driven through `apply_attention_head_pruning_cpp` instead of the
# pure-Python reference.
#
# `onnx.parser.parse_model`'s text format has no way to spell a graph-typed
# node attribute (an `If`'s `then_branch`/`else_branch`), so the model below
# uses `onnx.helper.make_node`/`make_graph` directly, per this repo's own
# CLAUDE.md guidance for exactly this case -- see `test_structured_pruning_
# cpp.py`'s own matching "Subgraph recursion" section for the `Loop`-body
# half of this coverage (the two C++ port test files split the `If`/`Loop`
# cases between them rather than duplicating both in each).


def _attention_branch_nodes(K, H, D, Out, prefix, seed):
    rng = np.random.default_rng(seed)
    Nq = Nk = Nv = H * D
    wqkv = rng.standard_normal((K, Nq + Nk + Nv)).astype(np.float32)
    bqkv = rng.standard_normal((Nq + Nk + Nv,)).astype(np.float32)
    wout = rng.standard_normal((Nv, Out)).astype(np.float32)
    nodes = [
        onnx.helper.make_node(
            "Attention",
            ["Xb", f"{prefix}Wqkv", f"{prefix}Bqkv"],
            [f"{prefix}ctx"],
            domain="com.microsoft",
            num_heads=H,
            qkv_hidden_sizes=[Nq, Nk, Nv],
        ),
        onnx.helper.make_node("MatMul", [f"{prefix}ctx", f"{prefix}Wout"], ["Yb"]),
    ]
    inits = [
        _f32(wqkv, f"{prefix}Wqkv"),
        _f32(bqkv, f"{prefix}Bqkv"),
        _f32(wout, f"{prefix}Wout"),
    ]
    return nodes, inits, dict(wqkv=wqkv, bqkv=bqkv, wout=wout)


def _if_attention_model(K=8, H=4, D=4, Out=6):
    then_nodes, then_inits, then_cfg = _attention_branch_nodes(
        K, H, D, Out, "then_", seed=1
    )
    else_nodes, else_inits, else_cfg = _attention_branch_nodes(
        K, H, D, Out, "else_", seed=2
    )
    out_vi = onnx.helper.make_tensor_value_info(
        "Yb", onnx.TensorProto.FLOAT, ["batch", "seq", Out]
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
    xb = onnx.helper.make_tensor_value_info(
        "Xb", onnx.TensorProto.FLOAT, ["batch", "seq", K]
    )
    cond = onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
    y1 = onnx.helper.make_tensor_value_info(
        "Y1", onnx.TensorProto.FLOAT, ["batch", "seq", Out]
    )
    graph = onnx.helper.make_graph([if_node], "g", [xb, cond], [y1])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model, dict(then=then_cfg, else_=else_cfg)


def _then_else_graphs(pruned_model):
    if_node = next(n for n in pruned_model.graph.node if n.op_type == "If")
    then_g = else_g = None
    for attr in if_node.attribute:
        if attr.name == "then_branch":
            then_g = attr.g
        elif attr.name == "else_branch":
            else_g = attr.g
    return then_g, else_g


def test_cpp_attention_head_pruning_prunes_blocks_inside_if_branches():
    # The core repro: apply_attention_head_pruning_cpp must match and prune
    # the merged-QKV Attention block inside BOTH `then_branch` and
    # `else_branch` -- each with its own independent weights and its own
    # independently-computed importance ranking/kept-head set -- not just a
    # top-level block. Verified by initializer shape, by the node's own
    # updated `num_heads` attribute, and by an exact oracle cross-check per
    # branch (mirroring `test_cpp_attention_head_pruning_matches_manual_
    # head_deletion_exactly`'s own oracle, just built independently once per
    # branch to prove neither branch's ranking leaked into the other's).
    K, H, D, Out = 8, 4, 4, 6
    model, cfg = _if_attention_model(K=K, H=H, D=D, Out=Out)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    then_g, else_g = _then_else_graphs(pruned)
    rng = np.random.default_rng(4)
    xb = rng.standard_normal((2, 3, K)).astype(np.float32)

    for g, prefix, branch_cfg, cond in [
        (then_g, "then_", cfg["then"], True),
        (else_g, "else_", cfg["else_"], False),
    ]:
        inits = {t.name: t for t in g.initializer}
        assert list(inits[f"{prefix}Wqkv"].dims) == [K, 3 * (H // 2) * D]
        assert list(inits[f"{prefix}Wout"].dims) == [(H // 2) * D, Out]
        node = next(n for n in g.node if n.op_type == "Attention")
        num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
        assert num_heads == H // 2

        wqkv, bqkv, wout = branch_cfg["wqkv"], branch_cfg["bqkv"], branch_cfg["wout"]
        Nq = Nk = Nv = H * D
        keep = _oracle_keep_heads(wqkv, Nq, Nk, Nv, H, H // 2)
        qi, ki, vi = (
            _head_idx(keep, D),
            _head_idx(keep, D) + Nq,
            _head_idx(keep, D) + Nq + Nk,
        )
        all_idx = np.concatenate([qi, ki, vi])
        np.testing.assert_allclose(
            onnx.numpy_helper.to_array(inits[f"{prefix}Wqkv"]),
            wqkv[:, all_idx],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            onnx.numpy_helper.to_array(inits[f"{prefix}Wout"]),
            wout[_head_idx(keep, D), :],
            rtol=1e-5,
            atol=1e-5,
        )

        oracle_bqkv = bqkv[all_idx]
        np.testing.assert_allclose(
            onnx.numpy_helper.to_array(inits[f"{prefix}Bqkv"]),
            oracle_bqkv,
            rtol=1e-5,
            atol=1e-5,
        )

        (yb,) = _run(pruned, {"Xb": xb, "cond": np.array(cond)})
        assert yb.shape == (2, 3, Out)
        assert np.all(np.isfinite(yb))


def test_cpp_attention_head_pruning_matches_python_reference_output_with_if_subgraph():
    # Cross-check against onnxsim.apply_attention_head_pruning (the
    # pure-Python reference this C++ port mirrors) on a model where the
    # only prunable attention block lives inside the `If`'s own branches --
    # both `cond` values are driven through InferenceSession so both
    # branches' own subgraph-recursion behavior is exercised.
    K, H, D, Out = 8, 4, 4, 6
    model, _cfg = _if_attention_model(K=K, H=H, D=D, Out=Out)

    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    rng = np.random.default_rng(23)
    xb = rng.standard_normal((2, 3, K)).astype(np.float32)
    for cond in (True, False):
        feeds = {"Xb": xb, "cond": np.array(cond)}
        (y_py,) = _run(pruned_py, feeds)
        (y_cpp,) = _run(pruned_cpp, feeds)
        np.testing.assert_allclose(y_py, y_cpp, rtol=1e-4, atol=1e-4)


# --- com.microsoft::MatMulNBitsQkv (fused, block-quantized Q/K/V projection
# --- feeding GroupQueryAttention) -------------------------------------------
#
# Tests for the fused ``MatMulNBitsQkv`` chain family
# (``onnxsim/structured_pruning_entry.cpp``'s own "MatMulNBitsMlp/
# MatMulNBitsQkv" subsection -- despite living in that file's own
# "MatMulNBits" section, this chain kind is wired into
# `ApplyAttentionHeadPruning`/``apply_attention_head_pruning_cpp`` rather
# than `ApplyStructuredPruning`/``apply_structured_pruning_cpp``, since
# pruning a whole KV group needs THIS function's own GQA head-count
# matching machinery -- see that subsection's own top comment for the full
# reasoning), mirroring ``tests/test_pruning.py``'s own
# ``test_matmul_nbits_qkv_pruning_matches_decomposed_oracle``/
# ``test_matmul_nbits_qkv_pruning_declines_non_block_aligned_consumer``.
# Like ``MatMulNBitsMlp``, neither this op nor its Q/K/V branches has a
# ``zero_points`` input at all -- every weight slot uses the schema's own
# DEFAULT zero point (``2 ** (bits - 1)``, i.e. 8 for ``bits=4``). Unlike
# ``MatMulNBitsMlp``, ``MatMulNBitsQkv`` itself ALSO cannot be executed via a
# real CPU-EP ``InferenceSession`` here (confirmed the same empirical way) --
# so its own oracle test below decomposes the PRUNED fused node's own
# tensors into a real ``SimplifiedLayerNormalization`` + 3x real
# ``MatMulNBits`` (both genuine CPU kernels) and checks their own outputs
# against an independent RMSNorm + dequantize-then-matmul numpy oracle,
# mirroring ``test_pruning.py``'s own identical proxy-topology technique.
# The downstream ``GroupQueryAttention``/consumer half of this pass's own
# slicing -- num_heads/kv_num_heads attribute rewrite, consumer weight
# slicing -- is exactly the same code path every plain-GQA test above
# already runs end to end through a real ``GroupQueryAttention`` CPU kernel
# (`ApplyOneGqaChain`'s own head-count/attribute handling, reused verbatim
# by `ApplyMatMulNBitsQkvChains` -- see that function's own comment), so
# this section's own tests below check it via direct assertion instead
# (attribute values, byte-exact consumer-weight slices) rather than
# re-proving already-oracle-tested machinery a second time.


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


def _nbits_pack_b(qcodes, n, k_blocks, block_size):
    blob_size = block_size * 4 // 8
    b = np.zeros((n, k_blocks, blob_size), dtype=np.uint8)
    for kb in range(k_blocks):
        k0 = kb * block_size
        b[:, kb, :] = _nbits_pack_nibbles(qcodes[:, k0 : k0 + block_size])
    return b


def _nbits_quantize_default_zp(w, block_size, bits=4):
    """Independent reference block quantizer using the schema's own DEFAULT
    zero point (``2 ** (bits - 1)``) -- the only encoding ``MatMulNBitsQkv``
    (and ``MatMulNBitsMlp``) supports, since neither has a ``zero_points``
    input at all. Returns ``(qcodes uint8 [N, K], scales float32 [N,
    k_blocks], k_blocks)``.
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


def _nbits_dequant(qcodes, scales, block_size, bits=4):
    """Independent reference dequantizer, schema DEFAULT zero point only."""
    n, k = qcodes.shape
    k_blocks = k // block_size
    out = np.zeros((n, k), dtype=np.float64)
    zp = float(1 << (bits - 1))
    for row in range(n):
        for kb in range(k_blocks):
            k0, k1 = kb * block_size, (kb + 1) * block_size
            out[row, k0:k1] = (qcodes[row, k0:k1].astype(np.float64) - zp) * scales[
                row, kb
            ]
    return out


def _nbits_no_zp_initializers(w, block_size, prefix, bits=4):
    """Quantizes ``w`` (``[N, K]``) with the schema's own default zero point
    and returns ``(initializer_list, info_dict)`` -- the zero_points-free
    analogue of a ``_nbits_weight_initializers`` helper, since neither
    ``MatMulNBitsQkv`` nor ``MatMulNBitsMlp`` ever carries one.
    """
    qcodes, scales, k_blocks = _nbits_quantize_default_zp(w, block_size, bits)
    b = _nbits_pack_b(qcodes, w.shape[0], k_blocks, block_size)
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


def _matmul_nbits_qkv_model(
    num_heads,
    kv_num_heads,
    d,
    K,
    block_size,
    N2,
    w_q,
    w_k,
    w_v,
    bias_q,
    bias_k,
    bias_v,
    norm_scale,
    batch=2,
    seq=5,
    consumer="plain",
    consumer_block_size=None,
    attention_bias=None,
):
    """Builds ``A -> MatMulNBitsQkv(qkv) -> (Q, K, V) ->
    GroupQueryAttention(attn) -> MatMul/MatMulNBits(down) -> Z``. ``Q``/``K``/
    ``V`` feed the attention node's own query/key/value inputs DIRECTLY (no
    per-head norm/RoPE hop -- a deliberate, documented scope boundary, see
    ``structured_pruning_entry.cpp``'s own section comment).
    ``consumer="plain"`` builds a plain-float output projection;
    ``consumer="nbits"`` a real ``MatMulNBits`` one (block size
    `consumer_block_size`), to exercise the block-alignment decline path.
    `attention_bias`, if given, is wired as a constant GQA `attention_bias`
    input (index 10) -- to exercise `MatMulNBitsQkvAttentionExtrasSafe`'s own
    decline path.
    """
    Nq = num_heads * d
    Nkv = kv_num_heads * d
    inits_q, info_q = _nbits_no_zp_initializers(w_q, block_size, "qkvq")
    inits_k, info_k = _nbits_no_zp_initializers(w_k, block_size, "qkvk")
    inits_v, info_v = _nbits_no_zp_initializers(w_v, block_size, "qkvv")

    seqlens_k = np.full((batch,), seq - 1, dtype=np.int32)
    total_seq = np.array(seq, dtype=np.int32)
    initializer = [
        *inits_q,
        *inits_k,
        *inits_v,
        _f32(bias_q, "qkv_bias_q"),
        _f32(bias_k, "qkv_bias_k"),
        _f32(bias_v, "qkv_bias_v"),
        _f32(norm_scale, "qkv_norm_scale"),
        onnx.numpy_helper.from_array(seqlens_k, "qkv_seqlens_k"),
        onnx.numpy_helper.from_array(total_seq, "qkv_total_seq"),
    ]

    if consumer == "plain":
        rng = np.random.default_rng(9202)
        down_w = (rng.standard_normal((Nq, N2)) * 0.3).astype(np.float32)
        initializer.append(_f32(down_w, "qkv_down_w"))
        consumer_body = "Z = MatMul(ctx, qkv_down_w)"
        consumer_info = dict(down_w=down_w)
    else:
        assert consumer_block_size is not None
        rng = np.random.default_rng(9203)
        w_c = (rng.standard_normal((N2, Nq)) * 0.3).astype(np.float32)
        inits_c, info_c = _nbits_no_zp_initializers(w_c, consumer_block_size, "qkvdown")
        initializer += inits_c
        consumer_body = (
            f"Z = com.microsoft.MatMulNBits<K={Nq},N={N2},bits=4,"
            f"block_size={consumer_block_size}>"
            f"(ctx, {info_c['b_name']}, {info_c['scales_name']})"
        )
        consumer_info = dict(
            qcodes_c=info_c["qcodes"], scales_c=info_c["scales"], kbc=info_c["k_blocks"]
        )

    gqa_extra_inputs = ""
    if attention_bias is not None:
        initializer.append(_f32(attention_bias, "qkv_attention_bias"))
        # attention_bias is input 10 -- pad cos_cache/sin_cache/position_ids
        # (7/8/9) with blanks.
        gqa_extra_inputs = ", , , , qkv_attention_bias"

    body = f"""
        g (float[{batch},{seq},{K}] A) => (float[{batch},{seq},{N2}] Z)
        {{
          Q, Kt, V = com.microsoft.MatMulNBitsQkv<block_size={block_size},bits=4,Nq={Nq},Nkv={Nkv},K={K}>(A, , qkv_norm_scale, {info_q["b_name"]}, {info_q["scales_name"]}, qkv_bias_q, {info_k["b_name"]}, {info_k["scales_name"]}, qkv_bias_k, {info_v["b_name"]}, {info_v["scales_name"]}, qkv_bias_v)
          ctx, pk, pv = com.microsoft.GroupQueryAttention<num_heads={num_heads}, kv_num_heads={kv_num_heads}>(Q, Kt, V, , , qkv_seqlens_k, qkv_total_seq{gqa_extra_inputs})
          {consumer_body}
        }}
        """

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    info = dict(
        qcodes_q=info_q["qcodes"],
        scales_q=info_q["scales"],
        qcodes_k=info_k["qcodes"],
        scales_k=info_k["scales"],
        qcodes_v=info_v["qcodes"],
        scales_v=info_v["scales"],
        kb=info_q["k_blocks"],
        Nq=Nq,
        Nkv=Nkv,
    )
    info.update(consumer_info)
    return model, info


def test_cpp_matmul_nbits_qkv_pruning_matches_decomposed_oracle():
    # 4 query heads, 2 KV heads (group_size=2), head_size=2. KV group 0
    # (query heads 0,1 + kv head 0) engineered LARGE (kept); KV group 1
    # (query heads 2,3 + kv head 1) engineered SMALL (dropped). sparsity=0.5
    # -> keep exactly 1 of 2 groups: group 0 -- num_heads 4->2, kv_num_heads
    # 2->1, preserving the 2:1 group ratio exactly.
    num_heads, kv_num_heads, d, K, block_size, N2 = 4, 2, 2, 32, 32, 5
    rng = np.random.default_rng(9300)
    w_q = (rng.standard_normal((num_heads * d, K)) * 0.3).astype(np.float32)
    w_k = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    w_v = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    w_q[:4] *= 8.0
    w_q[4:] *= 0.05
    w_k[:2] *= 8.0
    w_k[2:] *= 0.05
    w_v[:2] *= 8.0
    w_v[2:] *= 0.05
    bias_q = (rng.standard_normal(num_heads * d) * 0.05).astype(np.float32)
    bias_k = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    bias_v = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    norm_scale = (1.0 + rng.standard_normal(K) * 0.1).astype(np.float32)

    model, info = _matmul_nbits_qkv_model(
        num_heads,
        kv_num_heads,
        d,
        K,
        block_size,
        N2,
        w_q,
        w_k,
        w_v,
        bias_q,
        bias_k,
        bias_v,
        norm_scale,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    q_keep = np.array([0, 1, 2, 3])  # heads 0,1 (group 0), d=2 -> rows 0-3
    kv_keep = np.array([0, 1])  # kv head 0, d=2 -> rows 0-1

    qkv_node = next(n for n in pruned.graph.node if n.op_type == "MatMulNBitsQkv")
    assert next(a.i for a in qkv_node.attribute if a.name == "Nq") == 4
    assert next(a.i for a in qkv_node.attribute if a.name == "Nkv") == 2
    attn_node = next(n for n in pruned.graph.node if n.op_type == "GroupQueryAttention")
    assert next(a.i for a in attn_node.attribute if a.name == "num_heads") == 2
    assert next(a.i for a in attn_node.attribute if a.name == "kv_num_heads") == 1

    inits = {t.name: t for t in pruned.graph.initializer}
    q_B_expected = _nbits_pack_b(info["qcodes_q"][q_keep], 4, info["kb"], block_size)
    k_B_expected = _nbits_pack_b(info["qcodes_k"][kv_keep], 2, info["kb"], block_size)
    v_B_expected = _nbits_pack_b(info["qcodes_v"][kv_keep], 2, info["kb"], block_size)
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["qkvq_B"]), q_B_expected
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["qkvk_B"]), k_B_expected
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["qkvv_B"]), v_B_expected
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["qkv_down_w"]), info["down_w"][q_keep]
    )
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(inits["qkv_bias_q"]), bias_q[q_keep]
    )
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(inits["qkv_bias_k"]), bias_k[kv_keep]
    )
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(inits["qkv_bias_v"]), bias_v[kv_keep]
    )

    # Decompose the fused MatMulNBitsQkv node's own (PRUNED) tensors into a
    # real SimplifiedLayerNormalization + 3x real MatMulNBits and run THOSE
    # through a CPU-kernel InferenceSession against an independent RMSNorm +
    # dequantize-then-matmul numpy oracle -- see this section's own top
    # comment for why the fused node itself cannot be executed here.
    decomposed = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        g (float[batch,{K}] A) => (float[batch,4] q_out, float[batch,2] k_out, float[batch,2] v_out)
        {{
          A_norm = SimplifiedLayerNormalization<axis=-1,epsilon=1e-5>(A, qkv_norm_scale)
          q_out = com.microsoft.MatMulNBits<K={K},N=4,bits=4,block_size={block_size}>(A_norm, qkvq_B, qkvq_scales, , , qkv_bias_q)
          k_out = com.microsoft.MatMulNBits<K={K},N=2,bits=4,block_size={block_size}>(A_norm, qkvk_B, qkvk_scales, , , qkv_bias_k)
          v_out = com.microsoft.MatMulNBits<K={K},N=2,bits=4,block_size={block_size}>(A_norm, qkvv_B, qkvv_scales, , , qkv_bias_v)
        }}
        """
    )
    decomposed.graph.initializer.extend(
        [
            inits["qkv_norm_scale"],
            inits["qkvq_B"],
            inits["qkvq_scales"],
            inits["qkv_bias_q"],
            inits["qkvk_B"],
            inits["qkvk_scales"],
            inits["qkv_bias_k"],
            inits["qkvv_B"],
            inits["qkvv_scales"],
            inits["qkv_bias_v"],
        ]
    )
    # No onnx.checker.check_model here -- the plain ONNX checker doesn't
    # recognize `SimplifiedLayerNormalization` (an onnxruntime-only op
    # registered under the "" domain, confirmed via live schema
    # introspection) even though onnxruntime itself executes it fine;
    # mirrors test_pruning.py's own identical omission for this same
    # decomposed proxy graph.

    x = np.random.default_rng(9301).standard_normal((3, K)).astype(np.float32)
    q_actual, k_actual, v_actual = _run(decomposed, {"A": x})

    def _rmsnorm(a, scale, eps):
        rms = np.sqrt(np.mean(a.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)
        return (a.astype(np.float64) / rms) * scale.astype(np.float64)

    a_norm_ref = _rmsnorm(x, norm_scale, 1e-5)
    w_q_dequant = _nbits_dequant(info["qcodes_q"], info["scales_q"], block_size)
    w_k_dequant = _nbits_dequant(info["qcodes_k"], info["scales_k"], block_size)
    w_v_dequant = _nbits_dequant(info["qcodes_v"], info["scales_v"], block_size)
    q_ref = a_norm_ref @ w_q_dequant[q_keep].T + bias_q[q_keep]
    k_ref = a_norm_ref @ w_k_dequant[kv_keep].T + bias_k[kv_keep]
    v_ref = a_norm_ref @ w_v_dequant[kv_keep].T + bias_v[kv_keep]

    np.testing.assert_allclose(q_actual, q_ref, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(k_actual, k_ref, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(v_actual, v_ref, rtol=1e-3, atol=1e-3)


def test_cpp_matmul_nbits_qkv_pruning_declines_non_block_aligned_consumer():
    # head_size=3 (unlike the block-aligned test above's head_size=2):
    # keep_q_heads=[0, 1] (group 0) -> q_idx = rows [0..5] (6 elements),
    # which straddles the MatMulNBits consumer's own block boundary at row 4
    # (block_size=4: blocks [0,4), [4,8), [8,12)) -- rows 4, 5 are only PART
    # of block 1, so this keep-set is NOT block-aligned. The whole chain
    # (qkv node, attention node, AND consumer) must be left completely
    # untouched, mirroring the plain-MatMulNBits consumer's own identical
    # decline precedent.
    num_heads, kv_num_heads, d, K, block_size, N2 = 4, 2, 3, 32, 32, 4
    consumer_block_size = 4
    rng = np.random.default_rng(9310)
    w_q = (rng.standard_normal((num_heads * d, K)) * 0.3).astype(np.float32)
    w_k = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    w_v = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    w_q[:6] *= 8.0
    w_q[6:] *= 0.05
    w_k[:3] *= 8.0
    w_k[3:] *= 0.05
    w_v[:3] *= 8.0
    w_v[3:] *= 0.05
    bias_q = (rng.standard_normal(num_heads * d) * 0.05).astype(np.float32)
    bias_k = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    bias_v = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    norm_scale = (1.0 + rng.standard_normal(K) * 0.1).astype(np.float32)

    model, _info = _matmul_nbits_qkv_model(
        num_heads,
        kv_num_heads,
        d,
        K,
        block_size,
        N2,
        w_q,
        w_k,
        w_v,
        bias_q,
        bias_k,
        bias_v,
        norm_scale,
        consumer="nbits",
        consumer_block_size=consumer_block_size,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)

    inits_before = {t.name: t.SerializeToString() for t in model.graph.initializer}
    inits_after = {t.name: t.SerializeToString() for t in pruned.graph.initializer}
    assert inits_before == inits_after
    attrs_before = [
        (n.name, [(a.name, a.i) for a in n.attribute]) for n in model.graph.node
    ]
    attrs_after = [
        (n.name, [(a.name, a.i) for a in n.attribute]) for n in pruned.graph.node
    ]
    assert attrs_before == attrs_after


def test_cpp_matmul_nbits_qkv_pruning_declines_when_attention_bias_present():
    # A genuinely per-head-shaped `attention_bias` (dims[1] == num_heads,
    # not 1) on the downstream GroupQueryAttention node -- this port has no
    # dynamic-Gather-insertion machinery to correctly re-slice it (see
    # `MatMulNBitsQkvAttentionExtrasSafe`'s own comment), so the whole chain
    # must be declined outright rather than silently leave it stale.
    num_heads, kv_num_heads, d, K, block_size, N2 = 4, 2, 2, 32, 32, 5
    batch, seq = 2, 5
    rng = np.random.default_rng(9320)
    w_q = (rng.standard_normal((num_heads * d, K)) * 0.3).astype(np.float32)
    w_k = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    w_v = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    bias_q = (rng.standard_normal(num_heads * d) * 0.05).astype(np.float32)
    bias_k = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    bias_v = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    norm_scale = (1.0 + rng.standard_normal(K) * 0.1).astype(np.float32)
    attention_bias = rng.standard_normal((1, num_heads, seq, seq)).astype(np.float32)

    model, _info = _matmul_nbits_qkv_model(
        num_heads,
        kv_num_heads,
        d,
        K,
        block_size,
        N2,
        w_q,
        w_k,
        w_v,
        bias_q,
        bias_k,
        bias_v,
        norm_scale,
        batch=batch,
        seq=seq,
        attention_bias=attention_bias,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    inits_before = {t.name: t.SerializeToString() for t in model.graph.initializer}
    inits_after = {t.name: t.SerializeToString() for t in pruned.graph.initializer}
    assert inits_before == inits_after
    attrs_before = [
        (n.name, [(a.name, a.i) for a in n.attribute]) for n in model.graph.node
    ]
    attrs_after = [
        (n.name, [(a.name, a.i) for a in n.attribute]) for n in pruned.graph.node
    ]
    assert attrs_before == attrs_after


def test_cpp_matmul_nbits_qkv_pruning_zero_sparsity_is_a_no_op():
    num_heads, kv_num_heads, d, K, block_size, N2 = 4, 2, 2, 32, 32, 5
    rng = np.random.default_rng(9330)
    w_q = (rng.standard_normal((num_heads * d, K)) * 0.3).astype(np.float32)
    w_k = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    w_v = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    bias_q = (rng.standard_normal(num_heads * d) * 0.05).astype(np.float32)
    bias_k = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    bias_v = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    norm_scale = (1.0 + rng.standard_normal(K) * 0.1).astype(np.float32)

    model, _info = _matmul_nbits_qkv_model(
        num_heads,
        kv_num_heads,
        d,
        K,
        block_size,
        N2,
        w_q,
        w_k,
        w_v,
        bias_q,
        bias_k,
        bias_v,
        norm_scale,
    )
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.0)
    qkv_node = next(n for n in pruned.graph.node if n.op_type == "MatMulNBitsQkv")
    assert next(a.i for a in qkv_node.attribute if a.name == "Nq") == num_heads * d


# --- importance_norm ("l1" vs "l2") ------------------------------------------
#
# Adapted from test_pruning.py's own `test_attention_head_pruning_l1_norm_
# favors_total_magnitude`/`test_gqa_pruning_l1_norm_favors_total_magnitude`:
# adversarial per-head/per-group weight blocks engineered so L2 (Frobenius)
# and L1 (entrywise abs-sum) importance disagree on which unit survives --
# a bug that silently keeps ranking by L2 under the hood even when "l1" is
# requested would keep the WRONG head/group, not merely score it slightly
# differently.


def test_cpp_attention_head_pruning_importance_norm_l1_matches_python_reference_and_differs_from_l2():
    K, H, D, Out = 16, 4, 4, 3
    Nq = Nk = Nv = H * D
    rng_qk = np.random.default_rng(52)
    wqkv = np.zeros((K, Nq + Nk + Nv), dtype=np.float32)
    wqkv[:, :Nq] = rng_qk.standard_normal((K, Nq)).astype(np.float32) * 0.01
    wqkv[:, Nq : Nq + Nk] = rng_qk.standard_normal((K, Nk)).astype(np.float32) * 0.01
    v_offset = Nq + Nk
    wqkv[0, v_offset + 0] = 16.0  # head 0 ("concentrated")
    wqkv[:, v_offset + D : v_offset + 2 * D] = 1.0  # head 1 ("spread")
    wqkv[2, v_offset + 2 * D] = 1000.0  # head 2 ("filler_high")
    wqkv[3, v_offset + 3 * D] = 0.001  # head 3 ("filler_low")
    bqkv = np.zeros((Nq + Nk + Nv,), dtype=np.float32)

    model, _cfg = _attention_model(
        K=K, H=H, D=D, Out=Out, seed=50, bias=True, wqkv=wqkv, bqkv=bqkv
    )

    for norm in ("l2", "l1"):
        pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(
            model, sparsity=0.5, importance_norm=norm
        )
        pruned_py = onnxsim.apply_attention_head_pruning(
            model, sparsity=0.5, importance_norm=norm
        )
        onnx.checker.check_model(pruned_cpp)
        assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    kept_l2 = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    kept_l1 = onnxsim.apply_attention_head_pruning_cpp(
        model, sparsity=0.5, importance_norm="l1"
    )
    # "l2" keeps {concentrated, filler_high} (16 & 1000 dominate Frobenius),
    # "l1" keeps {spread, filler_high} (64 total magnitude beats 16) --
    # provably different surviving Wqkv shapes/values, not just a different
    # score.
    assert kept_l2.SerializeToString() != kept_l1.SerializeToString()


def test_cpp_gqa_pruning_importance_norm_l1_matches_python_reference_and_differs_from_l2():
    K, H, KVH, D, Out = 8, 4, 2, 8, 3
    Nq, Nkv = H * D, KVH * D
    wq = np.zeros((K, Nq), dtype=np.float32)
    wk = np.zeros((K, Nkv), dtype=np.float32)
    wv = np.zeros((K, Nkv), dtype=np.float32)
    wv[0, 0] = 16.0  # KV group 0's own V slice -- concentrated
    wv[:, D : 2 * D] = 1.0  # KV group 1's own V slice -- spread

    model, _cfg = _gqa_model(
        K=K, H=H, KVH=KVH, D=D, Out=Out, seed=60, wq=wq, wk=wk, wv=wv
    )

    for norm in ("l2", "l1"):
        pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(
            model, sparsity=0.5, importance_norm=norm
        )
        pruned_py = onnxsim.apply_attention_head_pruning(
            model, sparsity=0.5, importance_norm=norm
        )
        onnx.checker.check_model(pruned_cpp)
        assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    kept_l2 = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    kept_l1 = onnxsim.apply_attention_head_pruning_cpp(
        model, sparsity=0.5, importance_norm="l1"
    )
    # "l2" keeps KV group 0 (Frobenius 16 > 8), "l1" keeps KV group 1
    # (total magnitude 64 > 16) -- a real flip in which group survives.
    assert kept_l2.SerializeToString() != kept_l1.SerializeToString()


# ============================================================================
# Six more matched families, sharing the identical FindSeparateQkvChains/
# ApplyOneGqaChain machinery GroupQueryAttention/plain ai.onnx::Attention
# already exercise above -- see structured_pruning_entry.cpp's own
# "Attention-head pruning" section comment for each Find*Chains function's
# exact scope, and this file's own module docstring update. Every "matches
# python reference" test below is the strongest verification this module
# uses: a direct byte-for-byte ``SerializeToString()`` comparison against
# ``onnxsim.apply_attention_head_pruning`` (the pure-Python reference) run on
# the IDENTICAL input model -- not merely "both produce a valid model", but
# the same keep-set and the same sliced values.
#
# `MultiHeadAttention`/`PackedMultiHeadAttention` (the next two sections) now
# carry the same dynamic-attention-bias-Gather-insertion machinery
# (`HeadBiasInputIsSafe`/`SliceOrGatherHeadBias`) the decomposed-GQA family
# already has, plus (`MultiHeadAttention` only) `PastKvConstantsAreSliceable`
# for its own optional `past_key`/`past_value` -- see
# MatchMultiHeadAttentionProducer's/MatchPackedMultiHeadAttentionProducer's
# own comments in structured_pruning_entry.cpp. `DecoderMaskedMultiHeadAttention`
# (own `attention_bias`/`past_key`/`past_value`), `PagedAttention` (own
# `head_sink`/`q_norm_weight`/`k_norm_weight`/`k_scale`/`v_scale`), and
# `SparseAttention` (own required `past_key`/`past_value`) all now carry the
# same validate-and-slice treatment too -- see each one's own dedicated test
# section below and each `Match*Producer` function's own comment in
# structured_pruning_entry.cpp. `LinearAttention`, the one remaining family,
# has no optional per-head bias/mask/sink/norm/scale/past-KV input on its own
# schema at all, so none of this machinery applies to it.


# --- com.microsoft::MultiHeadAttention (separate Q/K/V producers, no --------
# --- independent kv_num_heads) ----------------------------------------------


def _mha_model(
    K=8,
    H=4,
    D=4,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    combined_bias=False,
    attention_bias=None,  # constant/declared-shape attention_bias array, or None
    attention_bias_dynamic=False,  # declare AttentionBias as a graph INPUT instead
    past_key=None,  # constant BNSH past_key array, or None (unconnected)
    past_value=None,  # constant BNSH past_value array, or None (unconnected)
    wq=None,
    wk=None,
    wv=None,
    bq=None,
    bk=None,
    bv=None,
    wout=None,
):
    rng = np.random.default_rng(seed)
    Nq = Nk = Nv = H * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nk)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nv, Out)).astype(np.float32)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    extra_inputs = ""
    operands = ["q", "k", "v"]
    if combined_bias:
        if bq is None:
            bq = rng.standard_normal((Nq,)).astype(np.float32)
        if bk is None:
            bk = rng.standard_normal((Nk,)).astype(np.float32)
        if bv is None:
            bv = rng.standard_normal((Nv,)).astype(np.float32)
        combined = np.concatenate([bq, bk, bv]).astype(np.float32)
        initializer.append(_f32(combined, "Bias"))
        operands.append("Bias")
    else:
        operands.append("")

    operands.append("")  # key_padding_mask (index 4) -- never touched by this pass

    if attention_bias is not None:
        if attention_bias_dynamic:
            shape_str = ",".join(str(d) for d in np.asarray(attention_bias).shape)
            extra_inputs += f", float[{shape_str}] AttentionBias"
        else:
            initializer.append(_f32(np.asarray(attention_bias), "AttentionBias"))
        operands.append("AttentionBias")
    else:
        operands.append("")

    if past_key is not None:
        initializer.append(_f32(np.asarray(past_key), "PastKey"))
        operands.append("PastKey")
    else:
        operands.append("")
    if past_value is not None:
        initializer.append(_f32(np.asarray(past_value), "PastValue"))
        operands.append("PastValue")
    else:
        operands.append("")

    while operands and operands[-1] == "":
        operands.pop()

    body = f"""
        g (float[{batch},{seq},{K}] X{extra_inputs}) => (float[{batch},{seq},{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx = com.microsoft.MultiHeadAttention <num_heads={H}> ({", ".join(operands)})
          Y = MatMul(ctx, Wout)
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
        D=D,
        Out=Out,
        Nq=Nq,
        Nk=Nk,
        Nv=Nv,
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


def _mha_node(model):
    return next(n for n in model.graph.node if n.op_type == "MultiHeadAttention")


def _mha_num_heads(node):
    return next(a.i for a in node.attribute if a.name == "num_heads")


def test_cpp_mha_pruning_shrinks_matched_block():
    model, cfg = _mha_model(K=8, H=4, D=8, Out=6)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _mha_node(pruned)
    assert _mha_num_heads(node) == 2
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [8, 16]
    assert list(inits["Wk"].dims) == [8, 16]
    assert list(inits["Wv"].dims) == [8, 16]
    assert list(inits["Wout"].dims) == [16, 6]


def test_cpp_mha_pruning_matches_python_reference():
    model, _ = _mha_model(K=8, H=8, D=4, Out=6, seed=21)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_mha_pruning_matches_oracle_exactly():
    model, cfg = _mha_model(K=8, H=8, D=4, Out=6, seed=22)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _mha_node(pruned)
    num_heads = _mha_num_heads(node)
    assert num_heads == 4
    keep_heads = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["H"], cfg["D"], num_heads
    )
    d = cfg["D"]
    idx = _head_idx(keep_heads, d)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"][:, idx])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"][:, idx])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"][:, idx])
    np.testing.assert_array_equal(inits["Wout"], cfg["wout"][idx, :])


def test_cpp_mha_pruning_slices_combined_bias_matches_python_reference():
    model, _ = _mha_model(K=8, H=8, D=4, Out=6, seed=23, combined_bias=True)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    # (nq+nk+nv) at new_num_heads=4, D=4 -- 3 equal-width segments since
    # kv_num_heads == num_heads always for this op.
    assert list(inits["Bias"].dims) == [4 * 4 * 3]


def test_cpp_mha_pruning_broadcast_attention_bias_is_left_untouched():
    # A genuinely BROADCAST attention_bias (dims[1] == 1, not num_heads) --
    # HeadBiasAxis classifies this as "already correct for any head count",
    # so the match succeeds (unlike the pre-fix behavior, which declined the
    # whole chain outright for ANY non-empty constant here) and
    # SliceOrGatherHeadBias leaves the tensor itself completely untouched
    # while every other matched weight is still pruned normally.
    bias = np.random.default_rng(24).standard_normal((1, 1, 5, 5)).astype(np.float32)
    model, _ = _mha_model(K=8, H=4, D=4, Out=6, seed=24, attention_bias=bias)
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert pruned_cpp.SerializeToString() != model.SerializeToString()

    node = _mha_node(pruned_cpp)
    assert _mha_num_heads(node) == 2
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    np.testing.assert_array_equal(
        inits_before["AttentionBias"], inits_after["AttentionBias"]
    )


def test_cpp_mha_pruning_per_head_attention_bias_is_sliced_matches_python():
    # A genuinely PER-HEAD constant attention_bias (dims[1] == num_heads) --
    # HeadBiasInputIsSafe confirms it's safe, so SliceOrGatherHeadBias slices
    # it in place along its own head axis by the kept query heads, exactly
    # like every other per-head weight in the chain.
    H = 4
    bias = np.random.default_rng(25).standard_normal((2, H, 5, 5)).astype(np.float32)
    model, cfg = _mha_model(K=8, H=H, D=4, Out=6, seed=25, attention_bias=bias)
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _mha_node(pruned_cpp)
    num_heads = _mha_num_heads(node)
    assert num_heads == 2
    keep_heads = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], H, H, cfg["D"], num_heads
    )
    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    assert inits["AttentionBias"].shape == (2, num_heads, 5, 5)
    np.testing.assert_array_equal(inits["AttentionBias"], bias[:, keep_heads, :, :])


def test_cpp_mha_pruning_dynamic_per_head_attention_bias_gathers_matches_python():
    # A genuinely PER-HEAD, but DYNAMIC (graph-input, not constant),
    # attention_bias -- the one case that needs SliceOrGatherHeadBias's own
    # Gather-insertion path (InsertDynamicHeadBiasGather), since the tensor's
    # own real values aren't available at prune time.
    H = 4
    bias_shape = (1, H, 5, 5)
    model, cfg = _mha_model(
        K=8,
        H=H,
        D=4,
        Out=6,
        seed=27,
        attention_bias=np.zeros(bias_shape, dtype=np.float32),
        attention_bias_dynamic=True,
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert pruned_cpp.SerializeToString() != model.SerializeToString()

    gather_nodes = [n for n in pruned_cpp.graph.node if n.op_type == "Gather"]
    assert len(gather_nodes) == 1
    gather = gather_nodes[0]
    assert gather.input[0] == "AttentionBias"
    axis = next(a.i for a in gather.attribute if a.name == "axis")
    assert axis == 1
    indices_init = next(
        t for t in pruned_cpp.graph.initializer if t.name == gather.input[1]
    )
    indices = onnx.numpy_helper.to_array(indices_init)

    node = _mha_node(pruned_cpp)
    num_heads = _mha_num_heads(node)
    keep_heads = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], H, H, cfg["D"], num_heads
    )
    np.testing.assert_array_equal(indices, keep_heads)
    # MHA's own attention_bias input (index 5) is rewired onto the new
    # Gather's own output, not left pointing directly at "AttentionBias".
    assert node.input[5] == gather.output[0]


def test_cpp_mha_pruning_unsafe_attention_bias_shape_declines_match_matches_python():
    # rank-4 with axis-1 length neither 1 (broadcast) nor num_heads
    # (per-head) -- HeadBiasAxis can't resolve this either way, so
    # HeadBiasInputIsSafe declines the whole match outright, exactly like
    # pruning.py's own `_head_bias_input_is_safe`/`_match_multi_head_attention_producer`.
    bias = np.random.default_rng(26).standard_normal((2, 3, 5, 5)).astype(np.float32)
    model, _ = _mha_model(K=8, H=4, D=4, Out=6, seed=26, attention_bias=bias)
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    # Declined outright -- the whole model is left byte-for-byte unchanged.
    assert pruned_cpp.SerializeToString() == model.SerializeToString()


def test_cpp_mha_pruning_constant_past_kv_is_sliced_matches_python():
    # A constant, BNSH-format (batch, kv_num_heads, past_seq, head_size)
    # past_key/past_value -- PastKvConstantsAreSliceable confirms the shape,
    # so both are sliced along their own kv_num_heads axis (axis 1) by the
    # same kept-head index set K's/V's own producer weight is sliced by.
    H, D, seq, batch, past_seq = 4, 4, 5, 2, 3
    rng = np.random.default_rng(28)
    past_key = rng.standard_normal((batch, H, past_seq, D)).astype(np.float32)
    past_value = rng.standard_normal((batch, H, past_seq, D)).astype(np.float32)
    model, cfg = _mha_model(
        K=8,
        H=H,
        D=D,
        Out=6,
        seed=28,
        batch=batch,
        seq=seq,
        past_key=past_key,
        past_value=past_value,
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert pruned_cpp.SerializeToString() != model.SerializeToString()

    node = _mha_node(pruned_cpp)
    num_heads = _mha_num_heads(node)
    assert num_heads == 2
    keep_heads = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], H, H, D, num_heads
    )
    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    assert inits["PastKey"].shape == (batch, num_heads, past_seq, D)
    assert inits["PastValue"].shape == (batch, num_heads, past_seq, D)
    np.testing.assert_array_equal(inits["PastKey"], past_key[:, keep_heads, :, :])
    np.testing.assert_array_equal(inits["PastValue"], past_value[:, keep_heads, :, :])


def test_cpp_mha_pruning_unsafe_past_key_shape_declines_match_matches_python():
    # axis-1 length neither `kv_num_heads` nor otherwise resolvable --
    # PastKvConstantsAreSliceable declines the whole match outright, exactly
    # like pruning.py's own `_past_kv_constants_are_sliceable`.
    H, D, seq, batch, past_seq = 4, 4, 5, 2, 3
    rng = np.random.default_rng(29)
    past_key = rng.standard_normal((batch, 3, past_seq, D)).astype(np.float32)
    model, _ = _mha_model(
        K=8, H=H, D=D, Out=6, seed=29, batch=batch, seq=seq, past_key=past_key
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert pruned_cpp.SerializeToString() == model.SerializeToString()


def _mha_cross_model(
    K_dec=8,
    K_enc=6,
    H=4,
    D=4,
    Out=6,
    seed=0,
    batch=2,
    seq_q=5,
    seq_kv=7,
    wq=None,
    wk=None,
    wv=None,
    wout=None,
):
    # Q fed from a different producer/input (`Xdec`) than K/V (`Xenc`) -- a
    # real, valid shape (e.g. encoder-decoder cross-attention). Mirrors
    # test_pruning.py's own `_mha_cross_model`.
    rng = np.random.default_rng(seed)
    Nq = H * D
    if wq is None:
        wq = rng.standard_normal((K_dec, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K_enc, Nq)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K_enc, Nq)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]

    body = f"""
        g (float[{batch},{seq_q},{K_dec}] Xdec, float[{batch},{seq_kv},{K_enc}] Xenc) => (float[{batch},{seq_q},{Out}] Y)
        {{
          q = MatMul(Xdec, Wq)
          k = MatMul(Xenc, Wk)
          v = MatMul(Xenc, Wv)
          ctx = com.microsoft.MultiHeadAttention <num_heads={H}> (q, k, v)
          Y = MatMul(ctx, Wout)
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
        K_dec=K_dec,
        K_enc=K_enc,
        H=H,
        D=D,
        Out=Out,
        Nq=Nq,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        batch=batch,
        seq_q=seq_q,
        seq_kv=seq_kv,
    )


def test_cpp_mha_pruning_cross_attention_matches_oracle_exactly():
    # Q's own producer weight (`K_dec` rows) and K/V's own (`K_enc` rows)
    # have genuinely different row counts here -- `_oracle_keep_groups_cross`
    # (sqrt(sum of squared per-block norms), not concatenate-then-norm) is
    # required for exactly the reason `ApplyOneGqaChain`'s own `Kq`/`Kk`/`Kv`
    # fix (this same shared function MultiHeadAttention also goes through)
    # exists.
    model, cfg = _mha_cross_model(K_dec=8, K_enc=6, H=8, D=4, Out=6, seed=26)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _mha_node(pruned_cpp)
    num_heads = _mha_num_heads(node)
    d = cfg["D"]
    keep_heads = _oracle_keep_groups_cross(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["H"], d, num_heads
    )
    idx = _head_idx(keep_heads, d)

    oracle, _ = _mha_cross_model(
        K_dec=cfg["K_dec"],
        K_enc=cfg["K_enc"],
        H=len(keep_heads),
        D=d,
        Out=cfg["Out"],
        seed=26,
        wq=cfg["wq"][:, idx],
        wk=cfg["wk"][:, idx],
        wv=cfg["wv"][:, idx],
        wout=cfg["wout"][idx, :],
        batch=cfg["batch"],
        seq_q=cfg["seq_q"],
        seq_kv=cfg["seq_kv"],
    )

    rng = np.random.default_rng(27)
    xdec = rng.standard_normal((cfg["batch"], cfg["seq_q"], cfg["K_dec"])).astype(
        np.float32
    )
    xenc = rng.standard_normal((cfg["batch"], cfg["seq_kv"], cfg["K_enc"])).astype(
        np.float32
    )
    (y_pruned,) = _run(pruned_cpp, {"Xdec": xdec, "Xenc": xenc})
    (y_oracle,) = _run(oracle, {"Xdec": xdec, "Xenc": xenc})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)

    (y_pruned2,) = _run(pruned_cpp, {"Xdec": xdec, "Xenc": xenc + 1.0})
    assert not np.allclose(y_pruned, y_pruned2)


# --- com.microsoft::PackedMultiHeadAttention --------------------------------


def _packed_mha_model(
    K=8,
    H=4,
    D=4,
    Out=6,
    seed=0,
    tok=5,
    combined_bias=False,
    attention_bias=None,
    attention_bias_dynamic=False,  # declare AttentionBias as a graph INPUT instead
    wq=None,
    wk=None,
    wv=None,
    bq=None,
    bk=None,
    bv=None,
    wout=None,
):
    rng = np.random.default_rng(seed)
    Nq = Nk = Nv = H * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nk)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nv, Out)).astype(np.float32)

    token_offset = np.arange(tok, dtype=np.int32).reshape(1, tok)
    cum_seq_len = np.array([0, tok], dtype=np.int32)
    initializer = [
        _f32(wq, "Wq"),
        _f32(wk, "Wk"),
        _f32(wv, "Wv"),
        _f32(wout, "Wout"),
        onnx.numpy_helper.from_array(token_offset, "TokenOffset"),
        onnx.numpy_helper.from_array(cum_seq_len, "CumSeqLen"),
    ]
    operands = ["q", "k", "v"]
    if combined_bias:
        if bq is None:
            bq = rng.standard_normal((Nq,)).astype(np.float32)
        if bk is None:
            bk = rng.standard_normal((Nk,)).astype(np.float32)
        if bv is None:
            bv = rng.standard_normal((Nv,)).astype(np.float32)
        combined = np.concatenate([bq, bk, bv]).astype(np.float32)
        initializer.append(_f32(combined, "Bias"))
        operands.append("Bias")
    else:
        operands.append("")
    operands += ["TokenOffset", "CumSeqLen"]
    extra_inputs = ""
    if attention_bias is not None:
        if attention_bias_dynamic:
            shape_str = ",".join(str(d) for d in np.asarray(attention_bias).shape)
            extra_inputs += f", float[{shape_str}] AttentionBias"
        else:
            initializer.append(_f32(np.asarray(attention_bias), "AttentionBias"))
        operands.append("AttentionBias")

    while operands and operands[-1] == "":
        operands.pop()
    qkv_inputs = ", ".join(operands)

    body = f"""
        g (float[{tok},{K}] X{extra_inputs}) => (float[{tok},{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx = com.microsoft.PackedMultiHeadAttention <num_heads={H}> ({qkv_inputs})
          Y = MatMul(ctx, Wout)
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
        D=D,
        Out=Out,
        Nq=Nq,
        Nk=Nk,
        Nv=Nv,
        wq=wq,
        wk=wk,
        wv=wv,
        bq=bq,
        bk=bk,
        bv=bv,
        wout=wout,
        tok=tok,
    )


def _packed_mha_node(model):
    return next(n for n in model.graph.node if n.op_type == "PackedMultiHeadAttention")


def test_cpp_packed_mha_pruning_shrinks_matched_block():
    model, cfg = _packed_mha_model(K=8, H=4, D=8, Out=6)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    node = _packed_mha_node(pruned)
    assert _mha_num_heads(node) == 2


def test_cpp_packed_mha_pruning_matches_python_reference():
    model, _ = _packed_mha_model(K=8, H=8, D=4, Out=6, seed=31)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_packed_mha_pruning_slices_combined_bias_matches_python_reference():
    model, _ = _packed_mha_model(K=8, H=8, D=4, Out=6, seed=32, combined_bias=True)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_packed_mha_pruning_broadcast_attention_bias_is_left_untouched():
    # A genuinely BROADCAST attention_bias (dims[1] == 1) -- see
    # test_cpp_mha_pruning_broadcast_attention_bias_is_left_untouched's own
    # comment; identical reasoning, just at this op's own attention_bias
    # index (6, not 5).
    bias = np.random.default_rng(33).standard_normal((1, 1, 5, 5)).astype(np.float32)
    model, _ = _packed_mha_model(
        K=8, H=4, D=4, Out=6, seed=33, tok=5, attention_bias=bias
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert pruned_cpp.SerializeToString() != model.SerializeToString()

    node = _packed_mha_node(pruned_cpp)
    assert _mha_num_heads(node) == 2
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    np.testing.assert_array_equal(
        inits_before["AttentionBias"], inits_after["AttentionBias"]
    )


def test_cpp_packed_mha_pruning_per_head_attention_bias_is_sliced_matches_python():
    # A genuinely PER-HEAD constant attention_bias (dims[1] == num_heads) --
    # see test_cpp_mha_pruning_per_head_attention_bias_is_sliced_matches_python's
    # own comment; identical reasoning, just at index 6.
    H = 4
    bias = np.random.default_rng(34).standard_normal((1, H, 5, 5)).astype(np.float32)
    model, cfg = _packed_mha_model(
        K=8, H=H, D=4, Out=6, seed=34, tok=5, attention_bias=bias
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = _packed_mha_node(pruned_cpp)
    num_heads = _mha_num_heads(node)
    assert num_heads == 2
    keep_heads = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], H, H, cfg["D"], num_heads
    )
    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    assert inits["AttentionBias"].shape == (1, num_heads, 5, 5)
    np.testing.assert_array_equal(inits["AttentionBias"], bias[:, keep_heads, :, :])


def test_cpp_packed_mha_pruning_dynamic_per_head_attention_bias_gathers_matches_python():
    # A genuinely PER-HEAD, but DYNAMIC, attention_bias -- see
    # test_cpp_mha_pruning_dynamic_per_head_attention_bias_gathers_matches_python's
    # own comment; identical reasoning, just at index 6.
    H = 4
    model, cfg = _packed_mha_model(
        K=8,
        H=H,
        D=4,
        Out=6,
        seed=35,
        tok=5,
        attention_bias=np.zeros((1, H, 5, 5), dtype=np.float32),
        attention_bias_dynamic=True,
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert pruned_cpp.SerializeToString() != model.SerializeToString()

    gather_nodes = [n for n in pruned_cpp.graph.node if n.op_type == "Gather"]
    assert len(gather_nodes) == 1
    gather = gather_nodes[0]
    assert gather.input[0] == "AttentionBias"
    axis = next(a.i for a in gather.attribute if a.name == "axis")
    assert axis == 1
    indices_init = next(
        t for t in pruned_cpp.graph.initializer if t.name == gather.input[1]
    )
    indices = onnx.numpy_helper.to_array(indices_init)

    node = _packed_mha_node(pruned_cpp)
    num_heads = _mha_num_heads(node)
    keep_heads = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], H, H, cfg["D"], num_heads
    )
    np.testing.assert_array_equal(indices, keep_heads)
    # PackedMultiHeadAttention's own attention_bias input (index 6) is
    # rewired onto the new Gather's own output.
    assert node.input[6] == gather.output[0]


def test_cpp_packed_mha_pruning_unsafe_attention_bias_shape_declines_match_matches_python():
    # rank-4 with axis-1 length neither 1 nor num_heads -- HeadBiasAxis can't
    # resolve this either way, so HeadBiasInputIsSafe declines the whole
    # match outright, exactly like pruning.py's own
    # `_match_packed_multi_head_attention_producer`.
    bias = np.random.default_rng(36).standard_normal((1, 3, 5, 5)).astype(np.float32)
    model, _ = _packed_mha_model(
        K=8, H=4, D=4, Out=6, seed=36, tok=5, attention_bias=bias
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert pruned_cpp.SerializeToString() == model.SerializeToString()


# --- com.microsoft::DecoderMaskedMultiHeadAttention -------------------------


def _dmmha_model(
    K=8,
    H=4,
    D=4,
    Out=6,
    seed=0,
    batch=2,
    combined_bias=False,
    wq=None,
    wk=None,
    wv=None,
    bq=None,
    bk=None,
    bv=None,
    wout=None,
    # attention_bias (index 4): None (unconnected) | "broadcast" (constant
    # [1, 1, 1, total_seq]) | "per_head" (constant [1, H, 1, total_seq]) |
    # "dynamic_per_head" (graph input [1, H, 1, total_seq]) | "bad" (constant
    # [1, H + 1, 1, total_seq] -- neither a genuine broadcast nor exactly
    # num_heads-wide) -- see HeadBiasInputIsSafe's own comment.
    attention_bias=None,
    attention_bias_array=None,
    total_seq=5,
    # past_key/past_value (indices 5/6): None (unconnected) | "nonempty"
    # (constant BNSH [batch, H, past_seq, D]) | "dynamic" (graph input) --
    # see PastKvConstantsAreSliceable's own comment.
    past_kv=None,
    past_seq=3,
):
    rng = np.random.default_rng(seed)
    Nq = Nk = Nv = H * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nk)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nv, Out)).astype(np.float32)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    # Fixed-index layout: query, key, value, mask_index, attention_bias,
    # past_key, past_value, past_sequence_length, beam_width,
    # cache_indirection, bias -- see MatchDecoderMaskedMultiHeadAttentionProducer's
    # own comment for why this differs from MultiHeadAttention's own layout.
    operands = ["q", "k", "v", "", "", "", "", "", "", "", ""]
    extra_graph_inputs = ""

    if attention_bias in ("broadcast", "per_head", "bad"):
        bias_heads = {"broadcast": 1, "per_head": H, "bad": H + 1}[attention_bias]
        if attention_bias_array is None:
            attention_bias_array = rng.standard_normal(
                (1, bias_heads, 1, total_seq)
            ).astype(np.float32)
        initializer.append(_f32(attention_bias_array, "AttnBias"))
        operands[4] = "AttnBias"
    elif attention_bias == "dynamic_per_head":
        operands[4] = "AttnBiasIn"
        extra_graph_inputs += f", float[1,{H},1,{total_seq}] AttnBiasIn"

    if past_kv == "nonempty":
        past_key = rng.standard_normal((batch, H, past_seq, D)).astype(np.float32)
        past_value = rng.standard_normal((batch, H, past_seq, D)).astype(np.float32)
        initializer += [_f32(past_key, "PastKey"), _f32(past_value, "PastValue")]
        operands[5] = "PastKey"
        operands[6] = "PastValue"
    elif past_kv == "dynamic":
        operands[5] = "PastKeyIn"
        operands[6] = "PastValueIn"
        extra_graph_inputs += (
            f", float[{batch},{H},{past_seq},{D}] PastKeyIn"
            f", float[{batch},{H},{past_seq},{D}] PastValueIn"
        )

    if combined_bias:
        if bq is None:
            bq = rng.standard_normal((Nq,)).astype(np.float32)
        if bk is None:
            bk = rng.standard_normal((Nk,)).astype(np.float32)
        if bv is None:
            bv = rng.standard_normal((Nv,)).astype(np.float32)
        combined = np.concatenate([bq, bk, bv]).astype(np.float32)
        initializer.append(_f32(combined, "Bias"))
        operands[10] = "Bias"
    while operands and operands[-1] == "":
        operands.pop()

    body = f"""
        g (float[{batch},1,{K}] X{extra_graph_inputs}) => (float[{batch},1,{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx = com.microsoft.DecoderMaskedMultiHeadAttention <num_heads={H}> ({", ".join(operands)})
          Y = MatMul(ctx, Wout)
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
        D=D,
        Out=Out,
        Nq=Nq,
        Nk=Nk,
        Nv=Nv,
        wq=wq,
        wk=wk,
        wv=wv,
        bq=bq,
        bk=bk,
        bv=bv,
        wout=wout,
        batch=batch,
        attention_bias_array=attention_bias_array,
    )


def _dmmha_node(model):
    return next(
        n for n in model.graph.node if n.op_type == "DecoderMaskedMultiHeadAttention"
    )


def test_cpp_dmmha_pruning_shrinks_matched_block():
    model, cfg = _dmmha_model(K=8, H=4, D=8, Out=6)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    node = _dmmha_node(pruned)
    assert _mha_num_heads(node) == 2


def test_cpp_dmmha_pruning_matches_python_reference():
    model, _ = _dmmha_model(K=8, H=8, D=4, Out=6, seed=41)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_dmmha_pruning_slices_combined_bias_matches_python_reference():
    model, _ = _dmmha_model(K=8, H=8, D=4, Out=6, seed=42, combined_bias=True)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_dmmha_pruning_constant_broadcast_attention_bias_is_untouched_matches_python():
    # A constant attention_bias of the schema-documented broadcastable shape
    # ([1, 1, 1, total_seq], axis 1 -- the num_heads-aligned slot -- sized 1)
    # is left completely untouched (HeadBiasAxis resolves -1: already correct
    # for any head count), matching pruning.py's own
    # `_head_bias_input_is_safe`/`_slice_or_gather_head_bias` exactly.
    model, cfg = _dmmha_model(K=8, H=8, D=4, Out=6, seed=43, attention_bias="broadcast")
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    bias_after = next(
        onnx.numpy_helper.to_array(t)
        for t in pruned_cpp.graph.initializer
        if t.name == "AttnBias"
    )
    np.testing.assert_array_equal(bias_after, cfg["attention_bias_array"])


def test_cpp_dmmha_pruning_constant_per_head_attention_bias_sliced_matches_python():
    # A constant attention_bias genuinely per-head-shaped ([1, H, 1,
    # total_seq], axis 1 sized exactly num_heads) is sliced in place along
    # axis 1 by the kept query heads -- HeadBiasAxis resolves 1,
    # SliceAxisGeneric performs the slice -- matching pruning.py's own
    # `_slice_axis` exactly.
    H, D = 8, 4
    model, cfg = _dmmha_model(K=8, H=H, D=D, Out=6, seed=44, attention_bias="per_head")
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    keep_heads = _oracle_keep_heads(
        np.concatenate([cfg["wq"], cfg["wk"], cfg["wv"]], axis=1),
        cfg["Nq"],
        cfg["Nk"],
        cfg["Nv"],
        H,
        H - round(H * 0.5),
    )
    bias_after = next(
        onnx.numpy_helper.to_array(t)
        for t in pruned_cpp.graph.initializer
        if t.name == "AttnBias"
    )
    np.testing.assert_array_equal(
        bias_after, cfg["attention_bias_array"][:, keep_heads, :, :]
    )


def test_cpp_dmmha_pruning_dynamic_per_head_attention_bias_gathers_matches_python():
    # A genuinely dynamic (graph-input) per-head attention_bias -- the one
    # case needing SliceOrGatherHeadBias's own `Gather`-insertion path
    # (InsertDynamicHeadBiasGather), since its own real values aren't
    # available to slice in place at prune time.
    H, D = 8, 4
    model, cfg = _dmmha_model(
        K=8, H=H, D=D, Out=6, seed=45, attention_bias="dynamic_per_head"
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    gather_nodes = [n for n in pruned_cpp.graph.node if n.op_type == "Gather"]
    assert len(gather_nodes) == 1
    gather = gather_nodes[0]
    assert gather.input[0] == "AttnBiasIn"
    axis = next(a.i for a in gather.attribute if a.name == "axis")
    assert axis == 1
    indices_init = next(
        t for t in pruned_cpp.graph.initializer if t.name == gather.input[1]
    )
    indices = onnx.numpy_helper.to_array(indices_init)
    keep_heads = _oracle_keep_heads(
        np.concatenate([cfg["wq"], cfg["wk"], cfg["wv"]], axis=1),
        cfg["Nq"],
        cfg["Nk"],
        cfg["Nv"],
        H,
        H - round(H * 0.5),
    )
    np.testing.assert_array_equal(indices, np.sort(keep_heads))
    node = _dmmha_node(pruned_cpp)
    assert node.input[4] == gather.output[0]


def test_cpp_dmmha_pruning_unresolvable_attention_bias_shape_declines_matches_python():
    # An attention_bias whose own axis-1 size is neither a genuine broadcast
    # (1) nor exactly num_heads (H + 1 here) -- HeadBiasAxis/
    # `_head_bias_axis` decline to classify it either way, so BOTH ports
    # must decline the WHOLE chain rather than guess.
    model, _ = _dmmha_model(K=8, H=8, D=4, Out=6, seed=46, attention_bias="bad")
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


def test_cpp_dmmha_pruning_constant_sliceable_past_kv_matches_python():
    # A constant past_key/past_value of the schema-documented BNSH layout
    # ([batch, num_heads, past_seq, head_size], axis 1 the num_heads axis) --
    # now sliced along axis 1 by the kept query heads (PastKvConstantsAreSliceable
    # confirms the shape, SliceAxisGeneric performs the slice), matching
    # pruning.py's own `_past_kv_constants_are_sliceable`/`_slice_axis1` exactly.
    H, D = 8, 4
    model, cfg = _dmmha_model(K=8, H=H, D=D, Out=6, seed=47, past_kv="nonempty")
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    keep_heads = _oracle_keep_heads(
        np.concatenate([cfg["wq"], cfg["wk"], cfg["wv"]], axis=1),
        cfg["Nq"],
        cfg["Nk"],
        cfg["Nv"],
        H,
        H - round(H * 0.5),
    )
    past_key_before = next(
        onnx.numpy_helper.to_array(t)
        for t in model.graph.initializer
        if t.name == "PastKey"
    )
    past_key_after = next(
        onnx.numpy_helper.to_array(t)
        for t in pruned_cpp.graph.initializer
        if t.name == "PastKey"
    )
    np.testing.assert_array_equal(past_key_after, past_key_before[:, keep_heads, :, :])


def test_cpp_dmmha_pruning_dynamic_past_kv_input_is_still_pruned_matches_python():
    model, _ = _dmmha_model(K=8, H=8, D=4, Out=6, seed=48, past_kv="dynamic")
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    node = _dmmha_node(pruned_cpp)
    assert _mha_num_heads(node) == 4


def test_cpp_dmmha_pruning_nonsliceable_past_kv_shape_declines_matches_python():
    # A constant past_key whose own axis-1 length doesn't match num_heads at
    # all (a shape PastKvConstantsAreSliceable can't confidently confirm is
    # the documented BNSH layout) declines the whole match outright, rather
    # than guessed at -- matching pruning.py's own identical decline.
    H, D, batch = 8, 4, 2
    rng = np.random.default_rng(49)
    model, _ = _dmmha_model(K=8, H=H, D=D, Out=6, seed=49, past_kv=None)
    bad_past_key = rng.standard_normal((batch, H + 1, 3, D)).astype(np.float32)
    bad_past_value = rng.standard_normal((batch, H + 1, 3, D)).astype(np.float32)
    model.graph.initializer.append(_f32(bad_past_key, "PastKey"))
    model.graph.initializer.append(_f32(bad_past_value, "PastValue"))
    node = _dmmha_node(model)
    del node.input[:]
    node.input.extend(["q", "k", "v", "", "", "PastKey", "PastValue"])
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


# --- com.microsoft::PagedAttention -------------------------------------------
#
# NOTE: this op's real onnxruntime schema constrains `query`/`key`/`value` to
# float16/bfloat16 only (no plain-float32 member at all -- see
# MatchPagedAttentionProducer's own comment in structured_pruning_entry.cpp)
# -- but this whole "Attention-head pruning" C++ section's shared Q/K/V-
# producer-matching machinery (MatchProducer/ReadFloatTensor/
# SetFloatTensorData) is FLOAT32-only, a pre-existing restriction shared by
# every family in this file, GQA/OnnxAttention included, not something new to
# this one. So -- exactly like every other test in this file -- these tests
# build a float32 model: a real onnxruntime-executable PagedAttention export
# always uses float16, so in practice this matcher only ever fires on a
# synthetic/float32 graph, never a genuine PagedAttention export; the
# matching/slicing logic itself is still verified correct here, and pruning.py
# itself has no dtype gate to decline the same float32 case (`_match_producer`
# also accepts plain FLOAT), so this is a byte-for-byte cross-check against a
# real code path in the pure-Python reference, not a fabricated comparison.


def _paged_attention_model(
    K=8,
    H=4,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    num_tokens=3,
    num_blocks=2,
    block_size=4,
    wq=None,
    wk=None,
    wv=None,
    wout=None,
    # head_sink (index 11): None (unconnected) | "per_head" (constant
    # (num_heads,)) | "bad" (constant (num_heads + 1,) -- not the schema's
    # own documented shape) -- see MatchPagedAttentionProducer's own comment.
    head_sink=None,
    head_sink_array=None,
    # q_norm_weight/k_norm_weight (indices 12/13): "both" connects a
    # (head_size,) constant pair (the schema's own "must be provided
    # together" rule); "only_q" connects just one, violating that rule.
    qk_norm=None,
    # k_scale/v_scale (indices 14/15): None (unconnected) | "broadcast"
    # (constant [1], PER_TENSOR) | "per_channel" (constant (kv_num_heads, 1,
    # D), PER_CHANNEL) -- see PagedKvScaleIsSliceable's own comment.
    kv_scale=None,
    k_scale_array=None,
    v_scale_array=None,
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32) * 0.5
    if wk is None:
        wk = rng.standard_normal((K, Nkv)).astype(np.float32) * 0.5
    if wv is None:
        wv = rng.standard_normal((K, Nkv)).astype(np.float32) * 0.5
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32) * 0.5

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array([0, num_tokens], dtype=np.int32), "CumSeqLen"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.zeros((1,), dtype=np.int32), "PastSeqLens")
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.zeros((1, 1), dtype=np.int32), "BlockTable")
    )

    extra_inputs = (
        f", float[{num_blocks},{block_size},{KVH},{D}] KeyCache"
        f", float[{num_blocks},{block_size},{KVH},{D}] ValueCache"
    )
    # Fixed-index layout: query, key, value, key_cache, value_cache,
    # cumulative_sequence_length, past_seqlens, block_table, cos_cache,
    # sin_cache, slot_mapping, head_sink, q_norm_weight, k_norm_weight,
    # k_scale, v_scale -- see MatchPagedAttentionProducer's own comment.
    operands = [
        "q",
        "k",
        "v",
        "KeyCache",
        "ValueCache",
        "CumSeqLen",
        "PastSeqLens",
        "BlockTable",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ]

    if head_sink in ("per_head", "bad"):
        sink_len = H if head_sink == "per_head" else H + 1
        if head_sink_array is None:
            head_sink_array = rng.standard_normal((sink_len,)).astype(np.float32)
        initializer.append(_f32(head_sink_array, "HeadSink"))
        operands[11] = "HeadSink"

    if qk_norm in ("both", "only_q"):
        q_norm = rng.standard_normal((D,)).astype(np.float32)
        initializer.append(_f32(q_norm, "QNorm"))
        operands[12] = "QNorm"
        if qk_norm == "both":
            k_norm = rng.standard_normal((D,)).astype(np.float32)
            initializer.append(_f32(k_norm, "KNorm"))
            operands[13] = "KNorm"

    if kv_scale in ("broadcast", "per_channel"):
        if kv_scale == "broadcast":
            if k_scale_array is None:
                k_scale_array = np.array([2.0], dtype=np.float32)
            if v_scale_array is None:
                v_scale_array = np.array([3.0], dtype=np.float32)
        else:
            if k_scale_array is None:
                k_scale_array = rng.standard_normal((KVH, 1, D)).astype(np.float32)
            if v_scale_array is None:
                v_scale_array = rng.standard_normal((KVH, 1, D)).astype(np.float32)
        initializer.append(_f32(k_scale_array, "KScale"))
        initializer.append(_f32(v_scale_array, "VScale"))
        operands[14] = "KScale"
        operands[15] = "VScale"

    while operands and operands[-1] == "":
        operands.pop()

    body = f"""
        g (float[{num_tokens},{K}] X{extra_inputs}) => (float[{num_tokens},{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx, key_cache_out, value_cache_out = com.microsoft.PagedAttention <num_heads={H}, kv_num_heads={KVH}> ({", ".join(operands)})
          Y = MatMul(ctx, Wout)
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
        wout=wout,
        num_tokens=num_tokens,
        head_sink_array=head_sink_array,
        k_scale_array=k_scale_array,
        v_scale_array=v_scale_array,
    )


def _paged_node(model):
    return next(n for n in model.graph.node if n.op_type == "PagedAttention")


def _paged_attrs(node):
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return num_heads, kv_num_heads


def test_cpp_paged_attention_pruning_shrinks_matched_block():
    model, cfg = _paged_attention_model(K=8, H=4, KVH=4, D=8, Out=6)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    node = _paged_node(pruned)
    num_heads, kv_num_heads = _paged_attrs(node)
    assert num_heads == 2
    assert kv_num_heads == 2


def test_cpp_paged_attention_pruning_matches_python_reference():
    model, _ = _paged_attention_model(K=8, H=8, KVH=4, D=8, Out=6, seed=51)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_paged_attention_pruning_matches_oracle_exactly():
    model, cfg = _paged_attention_model(K=8, H=8, KVH=2, D=8, Out=6, seed=52)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _paged_node(pruned)
    num_heads, kv_num_heads = _paged_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1
    assert num_heads == kv_num_heads * group_size

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
    np.testing.assert_array_equal(inits["Wout"], cfg["wout"][q_idx, :])


def _paged_keep(cfg, sparsity=0.5):
    group_size = cfg["H"] // cfg["KVH"]
    new_kv = cfg["KVH"] - round(cfg["KVH"] * sparsity)
    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], new_kv
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    return keep_groups, keep_q_heads


def test_cpp_paged_attention_pruning_constant_head_sink_sliced_matches_python():
    # A constant head_sink of the schema-documented (num_heads,) shape is
    # sliced by the kept query heads -- matching pruning.py's own
    # `_match_paged_attention_producer`/`_slice_last_axis` exactly. This also
    # exercises the underlying parity gap this port closes: previously ANY
    # non-empty head_sink declined the whole match outright.
    model, cfg = _paged_attention_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=53, head_sink="per_head"
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    _, keep_q_heads = _paged_keep(cfg)
    sink_after = next(
        onnx.numpy_helper.to_array(t)
        for t in pruned_cpp.graph.initializer
        if t.name == "HeadSink"
    )
    np.testing.assert_array_equal(sink_after, cfg["head_sink_array"][keep_q_heads])


def test_cpp_paged_attention_pruning_bad_head_sink_shape_declines_matches_python():
    # A constant head_sink whose own shape isn't exactly (num_heads,) --
    # neither port can confirm this is the documented layout, so both
    # decline the whole match rather than guess.
    model, _ = _paged_attention_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=54, head_sink="bad"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


def test_cpp_paged_attention_pruning_qk_norm_paired_presence_now_matches_python():
    # A connected q_norm_weight/k_norm_weight PAIR (the schema's own "must be
    # provided together" rule satisfied) now matches and prunes -- neither is
    # ever sliced (a (head_size,)-shaped tensor never needs touching), so
    # both stay byte-identical across the prune. Previously this port
    # declined the whole match whenever EITHER was present at all.
    model, cfg = _paged_attention_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=55, qk_norm="both"
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    q_norm_before = next(
        onnx.numpy_helper.to_array(t)
        for t in model.graph.initializer
        if t.name == "QNorm"
    )
    q_norm_after = next(
        onnx.numpy_helper.to_array(t)
        for t in pruned_cpp.graph.initializer
        if t.name == "QNorm"
    )
    np.testing.assert_array_equal(q_norm_after, q_norm_before)


def test_cpp_paged_attention_pruning_qk_norm_only_one_present_declines_matches_python():
    # Only q_norm_weight connected, k_norm_weight absent -- violates the
    # schema's own "must be provided together" rule, declined outright by
    # both ports.
    model, _ = _paged_attention_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=56, qk_norm="only_q"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


def test_cpp_paged_attention_pruning_bad_qk_norm_shape_declines_matches_python():
    # A connected q_norm_weight/k_norm_weight pair whose own shape ISN'T
    # exactly (head_size,) -- the deferred shape check FindSeparateQkvChains
    # performs once head_size is known -- declines the whole match, matching
    # pruning.py's own identical deferred check.
    model, cfg = _paged_attention_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=57, qk_norm="both"
    )
    for i, t in enumerate(model.graph.initializer):
        if t.name == "QNorm":
            bad = np.zeros((cfg["D"] + 1,), dtype=np.float32)
            model.graph.initializer[i].CopyFrom(_f32(bad, "QNorm"))
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


def test_cpp_paged_attention_pruning_constant_broadcast_kv_scale_is_untouched_matches_python():
    # A constant k_scale/v_scale of the schema's own "PER_TENSOR" broadcast
    # shape ([1]) needs no slicing at all -- left completely untouched,
    # matching pruning.py's own `_paged_kv_scale_is_sliceable` exactly.
    model, cfg = _paged_attention_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=58, kv_scale="broadcast"
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    k_scale_after = next(
        onnx.numpy_helper.to_array(t)
        for t in pruned_cpp.graph.initializer
        if t.name == "KScale"
    )
    np.testing.assert_array_equal(k_scale_after, cfg["k_scale_array"])


def test_cpp_paged_attention_pruning_constant_per_channel_kv_scale_sliced_matches_python():
    # A constant k_scale/v_scale of the schema's own "PER_CHANNEL"
    # (kv_num_heads, 1, head_size) shape -- the KV axis at position *0*, not
    # 1 the way GroupQueryAttention's own k_scale/v_scale is -- sliced along
    # axis 0 by the kept KV groups, matching pruning.py's own
    # `_paged_kv_scale_is_sliceable`/`_slice_axis` exactly.
    model, cfg = _paged_attention_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=59, kv_scale="per_channel"
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    keep_groups, _ = _paged_keep(cfg)
    k_scale_after = next(
        onnx.numpy_helper.to_array(t)
        for t in pruned_cpp.graph.initializer
        if t.name == "KScale"
    )
    v_scale_after = next(
        onnx.numpy_helper.to_array(t)
        for t in pruned_cpp.graph.initializer
        if t.name == "VScale"
    )
    np.testing.assert_array_equal(k_scale_after, cfg["k_scale_array"][keep_groups])
    np.testing.assert_array_equal(v_scale_after, cfg["v_scale_array"][keep_groups])


# --- Plain ai.onnx::LinearAttention (opset 27+) -----------------------------
#
# This port only ever matches this op's stateless "linear" update_rule shape
# (no `past_state`/`decay`/`beta` connected at all) -- see
# MatchLinearAttentionProducer's own comment for why this narrower-than-
# pruning.py scope gives up essentially nothing over the realistic export
# shape pruning.py's own docstring already narrows *its* real-world scope to.


def _linear_attention_model(
    Hq=4,
    Hkv=2,
    D=4,
    K=16,
    seed=0,
    wq=None,
    wk=None,
    wv=None,
    wo=None,
    decay=None,  # constant array, or None (unconnected) -- forces a decline
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = Hq * D, Hkv * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32) * 0.3
    if wk is None:
        wk = rng.standard_normal((K, Nkv)).astype(np.float32) * 0.3
    if wv is None:
        wv = rng.standard_normal((K, Nkv)).astype(np.float32) * 0.3
    if wo is None:
        wo = rng.standard_normal((Nq, K)).astype(np.float32) * 0.3

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wo, "Wo")]
    operands = ["q", "k", "v"]
    if decay is not None:
        operands.append("")  # past_state -- unconnected
        initializer.append(_f32(np.asarray(decay), "Decay"))
        operands.append("Decay")

    body = f"""
        g (float[1,3,{K}] X) => (float[1,3,{K}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          attn_out, ps = LinearAttention<q_num_heads={Hq}, kv_num_heads={Hkv}, update_rule="{"gated" if decay is not None else "linear"}">({", ".join(operands)})
          Y = MatMul(attn_out, Wo)
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 27]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(Hq=Hq, Hkv=Hkv, D=D, K=K, wq=wq, wk=wk, wv=wv, wo=wo)


def _linear_attention_node(model):
    return next(n for n in model.graph.node if n.op_type == "LinearAttention")


def _linear_attention_attrs(node):
    q_num_heads = next(a.i for a in node.attribute if a.name == "q_num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return q_num_heads, kv_num_heads


def test_cpp_linear_attention_pruning_shrinks_matched_block():
    model, cfg = _linear_attention_model(Hq=4, Hkv=4, D=4, K=16)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    node = _linear_attention_node(pruned)
    q_num_heads, kv_num_heads = _linear_attention_attrs(node)
    assert q_num_heads == 2
    assert kv_num_heads == 2


def test_cpp_linear_attention_pruning_matches_python_reference():
    model, _ = _linear_attention_model(Hq=8, Hkv=4, D=4, K=16, seed=61)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_linear_attention_pruning_matches_oracle_exactly():
    model, cfg = _linear_attention_model(Hq=8, Hkv=2, D=4, K=16, seed=62)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _linear_attention_node(pruned)
    q_num_heads, kv_num_heads = _linear_attention_attrs(node)
    group_size = cfg["Hq"] // cfg["Hkv"]
    assert kv_num_heads == 1
    assert q_num_heads == kv_num_heads * group_size

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["Hq"], cfg["Hkv"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"][:, kv_idx])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"][:, kv_idx])
    np.testing.assert_array_equal(inits["Wo"], cfg["wo"][q_idx, :])


def test_cpp_linear_attention_pruning_gated_mode_decay_present_is_left_untouched():
    model, cfg = _linear_attention_model(
        Hq=4, Hkv=4, D=4, K=16, seed=63, decay=np.zeros((1, 3, 4), dtype=np.float32)
    )
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


# --- com.microsoft::SparseAttention ------------------------------------------


def _sparse_attention_model(
    K=8,
    H=4,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=1,
    seq=16,
    sparse_block_size=16,
    num_layout=1,
    wq=None,
    wk=None,
    wv=None,
    wout=None,
    past_kv="dynamic",  # "dynamic" (graph input only) | "nonempty" (constant, sliced)
    row_indices=None,
    col_indices=None,
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32) * 0.1
    if wk is None:
        wk = rng.standard_normal((K, Nkv)).astype(np.float32) * 0.1
    if wv is None:
        wv = rng.standard_normal((K, Nkv)).astype(np.float32) * 0.1
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32) * 0.1

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]

    extra_inputs = ""
    past_key = past_value = None
    if past_kv == "nonempty":
        past_key = rng.standard_normal((batch, KVH, seq, D)).astype(np.float32) * 0.1
        past_value = rng.standard_normal((batch, KVH, seq, D)).astype(np.float32) * 0.1
        initializer += [
            onnx.numpy_helper.from_array(past_key, "PastKey"),
            onnx.numpy_helper.from_array(past_value, "PastValue"),
        ]
    else:
        extra_inputs = (
            f", float[{batch},{KVH},{seq},{D}] PastKey"
            f", float[{batch},{KVH},{seq},{D}] PastValue"
        )

    if row_indices is None:
        row_indices = np.tile(np.array([0, 1], dtype=np.int32), (num_layout, 1))
    if col_indices is None:
        col_indices = np.tile(np.array([0], dtype=np.int32), (num_layout, 1))
    initializer.append(onnx.numpy_helper.from_array(row_indices, "RowIdx"))
    initializer.append(onnx.numpy_helper.from_array(col_indices, "ColIdx"))
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq, dtype=np.int32), "KeyTotalSeq"
        )
    )

    body = f"""
        g (float[{batch},{seq},{K}] X{extra_inputs}) => (float[{batch},{seq},{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          attn, PresentKey, PresentValue = com.microsoft.SparseAttention <num_heads={H}, kv_num_heads={KVH}, sparse_block_size={sparse_block_size}> (q, k, v, PastKey, PastValue, RowIdx, ColIdx, TotalSeq, KeyTotalSeq)
          Y = MatMul(attn, Wout)
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 18, "com.microsoft": 1]
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
        wout=wout,
        batch=batch,
        seq=seq,
        num_layout=num_layout,
        past_key=past_key,
        past_value=past_value,
    )


def _sparse_attention_node(model):
    return next(n for n in model.graph.node if n.op_type == "SparseAttention")


def _sparse_attention_attrs(node):
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return num_heads, kv_num_heads


def test_cpp_sparse_attention_pruning_shrinks_matched_block():
    model, cfg = _sparse_attention_model(K=8, H=4, KVH=4, D=8, Out=6)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    node = _sparse_attention_node(pruned)
    num_heads, kv_num_heads = _sparse_attention_attrs(node)
    assert num_heads == 2
    assert kv_num_heads == 2


def test_cpp_sparse_attention_pruning_matches_python_reference():
    model, _ = _sparse_attention_model(K=8, H=8, KVH=4, D=8, Out=6, seed=71)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_sparse_attention_pruning_matches_oracle_exactly():
    model, cfg = _sparse_attention_model(K=8, H=8, KVH=2, D=8, Out=6, seed=72)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _sparse_attention_node(pruned)
    num_heads, kv_num_heads = _sparse_attention_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1
    assert num_heads == kv_num_heads * group_size

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
    np.testing.assert_array_equal(inits["Wout"], cfg["wout"][q_idx, :])


def test_cpp_sparse_attention_pruning_nonempty_past_kv_constant_is_sliced_matches_python():
    # A constant, schema-conforming BNSH-format (batch, kv_num_heads,
    # past_seq, head_size) `past_key`/`past_value` -- PastKvConstantsAreSliceable
    # confirms the shape, so both are sliced along their own kv_num_heads
    # axis (axis 1) by the same kept-KV-group index set K's/V's own producer
    # weights are sliced by, exactly mirroring pruning.py's own
    # `_apply_one_gqa_chain` handling of GroupQueryAttention's/the plain
    # ai.onnx op's own analogous `past_key`/`past_value`. An earlier version
    # of this matcher declined the whole match outright whenever either
    # resolved to a non-empty constant at all (`NoNonEmptyConstantAt`) -- this
    # test would have found the whole model left untouched against that
    # earlier version, not the genuine slice verified here.
    model, cfg = _sparse_attention_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=73, past_kv="nonempty"
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert pruned_cpp.SerializeToString() != model.SerializeToString()

    node = _sparse_attention_node(pruned_cpp)
    num_heads, kv_num_heads = _sparse_attention_attrs(node)
    assert kv_num_heads == 1
    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    np.testing.assert_array_equal(inits["PastKey"], cfg["past_key"][:, keep_groups])
    np.testing.assert_array_equal(inits["PastValue"], cfg["past_value"][:, keep_groups])
    assert list(inits["PastKey"].shape) == [
        cfg["batch"],
        kv_num_heads,
        cfg["seq"],
        cfg["D"],
    ]


def test_cpp_sparse_attention_pruning_num_layout_divisibility_declined_matches_python_reference():
    # H=4, KVH=4 (group_size=1, ordinary per-head groups), num_layout=2:
    # matches fine at match time (4 % 2 == 0), but at sparsity=0.75,
    # keep_count == max(1, 4 - round(4*0.75)) == 1 -- the post-pruning query
    # head count would be 1, not divisible by num_layout == 2, which would
    # silently break this op's own "num_heads is divisible by num_layout"
    # invariant even though it held before pruning. Exercises
    # ApplyOneGqaChain's own `is_sparse_attention` APPLY-TIME branch (not
    # merely MatchSparseAttentionProducer's own match-time check already
    # covered by `test_cpp_sparse_attention_pruning_matches_python_reference`
    # and friends) -- declined as a no-op for this block, mirroring
    # pruning.py's own `test_sparse_attention_pruning_num_layout_divisibility_
    # declined` exactly: both ports agree byte-for-byte.
    model, _ = _sparse_attention_model(
        K=8, H=4, KVH=4, D=8, Out=6, seed=74, num_layout=2
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.75)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.75)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    assert pruned_cpp.SerializeToString() == model.SerializeToString()


# --- Decomposed (un-fused) GQA/MQA/plain-MHA attention head pruning --------
#
# `apply_attention_head_pruning_cpp` now also matches this shape, via a
# genuinely new, dedicated matcher/rewriter (FindDecomposedGqaChains/
# ApplyOneDecomposedGqaChain in structured_pruning_entry.cpp) -- NOT the
# FindSeparateQkvChains/ApplyOneGqaChain machinery every family above shares,
# since there is no single fused node whose `num_heads`/`kv_num_heads`
# *attribute* records the head count here; instead it is baked, redundantly,
# into up to six `Reshape`/`Expand` target-shape `Constant` tensors, every
# one of which is rewritten in lock-step post-pruning -- in place, or by
# cloning a fresh tensor when some genuinely unrelated node also reads the
# same (CSE-shared) constant. See structured_pruning_entry.cpp's own
# "Decomposed (un-fused) GQA/MQA/plain-MHA attention head pruning" section
# comment for the exact scope this port matches. An additive mask
# (`attn_mask`/causal bias) before `Softmax` IS matched and pruned here now
# -- `ResolveDecomposedQkRoot`'s own optional `Add(mask) -> [Mul/Div(scale)
# ->] MatMul` prefix, and `HeadBiasInputIsSafe`/`SliceOrGatherHeadBias`, this
# port's own analogue of pruning.py's `_head_bias_input_is_safe`/
# `_slice_or_gather_head_bias` for this chain family: a constant per-head
# mask is sliced in place, a genuinely dynamic one gets a new `Gather`
# spliced in ahead of it, and an unresolvable shape declines the WHOLE chain
# -- exactly mirroring pruning.py's own behavior. An `Einsum`-based QK^T/AV
# product, in place of a plain `MatMul`, for either or both products IS also
# recognized here (via `EinsumEquationIsBatchedMatmul`, a line-for-line port
# of pruning.py's own `_einsum_equation_is_batched_matmul`) -- see
# `_decomposed_einsum_gqa_model` and its own tests below, further down this
# section. Decomposed RoPE and decomposed Q/K-norm pass-through ARE now
# recognized too (see the dedicated tests below and structured_pruning_entry.
# cpp's own section comment for that narrower-still sub-scope: only Q's own
# branch and K's separate-`perm=[0,1,3,2]` swap branch, never K's
# combined-perm shape, an `Einsum` QK^T product, or V's own branch). A
# packed-QKV-then-`Split` producer (MatchDecomposedPackedQkvProducer) AND a
# true-MQA fast path (ApplyOneDecomposedGqaChain's own `is_mqa` handling) ARE
# both matched/applied here now too, mirroring pruning.py's own
# `_match_decomposed_packed_qkv_producer`/`is_mqa` handling exactly -- see
# each's own comment in structured_pruning_entry.cpp and the dedicated tests
# below. Every remaining shape not covered by one of those still declines to
# match here (never mis-sliced) and is still fully handled by pruning.py's
# own pure-Python `apply_attention_head_pruning`, so this tenth family is NOT
# yet aliased.


def _rope_lines(prefix, src, out_name):
    # `x * cos + rotate_half(x) * sin`, HuggingFace's own fixed formula --
    # see structured_pruning_entry.cpp's own comment above
    # `DecomposedRopePassThrough` for the exact confirmed shape this mirrors,
    # node for node. Shared by `_decomposed_gqa_model` (Q's/K's own MatMul
    # branch) and `_decomposed_einsum_gqa_model` (Q's own branch only -- see
    # that function's own `q_rope` param) -- both rely on the same fixed
    # `CosU`/`SinU`/`SliceStart0`/`SliceHalf`/`SliceEnd`/`SliceAxis3`/
    # `SliceStep1` initializer names their own callers already set up.
    return [
        f"{prefix}direct = Mul({src}, CosU)",
        f"{prefix}x1 = Slice({src}, SliceStart0, SliceHalf, SliceAxis3, SliceStep1)",
        f"{prefix}x2 = Slice({src}, SliceHalf, SliceEnd, SliceAxis3, SliceStep1)",
        f"{prefix}neg = Neg({prefix}x2)",
        f"{prefix}rot = Concat<axis=-1>({prefix}neg, {prefix}x1)",
        f"{prefix}rotated = Mul({prefix}rot, SinU)",
        f"{out_name} = Add({prefix}direct, {prefix}rotated)",
    ]


def _decomposed_gqa_model(
    K=32,
    H=4,
    KVH=2,
    D=8,
    Dv=None,
    Out=16,
    batch=1,
    seq=4,
    seed=0,
    bias=True,
    masked=False,
    mask_dynamic=False,
    mask=None,
    wq=None,
    wk=None,
    wv=None,
    wout=None,
    bq=None,
    bk=None,
    bv=None,
    bout=None,
    out_reshape_wildcard=False,
    share_kv_reshape_shape=True,
    extra_foreign_q_reshape_consumer=False,
    extra_foreign_repeat_kv_consumer=False,
    rope=False,
    qk_norm=False,
    q_norm_weight=None,
    k_norm_weight=None,
    qk_norm_eps=1e-6,
):
    """Builds the decomposed-attention graph FindDecomposedGqaChains matches:
    ``Linear(Q/K/V) -> Reshape(to heads) -> Transpose(to BHSD) -> [repeat_kv:
    Unsqueeze -> Expand -> Reshape, only when KVH < H] -> MatMul(Q, K^T) ->
    scale -> [+mask] -> Softmax -> MatMul(., V) -> Transpose(back) ->
    Reshape(back to hidden) -> Linear(O)`` -- mirrors
    ``tests/test_pruning.py``'s own ``_decomposed_gqa_model``.

    K's own dot-product transpose is built in whichever of the two real
    shapes a real export uses: the single, pre-composed ``perm=[0, 2, 3,
    1]`` form when ``KVH == H`` (no ``repeat_kv``) and neither ``rope`` nor
    ``qk_norm`` is set, or the ordinary separate ``perm=[0, 2, 1, 3]``
    head-split followed later by its own ``perm=[0, 1, 3, 2]`` swap when
    ``KVH < H`` (``repeat_kv`` present) or either of ``rope``/``qk_norm`` is
    set (both hops' own nodes always break the head-split/swap-transpose
    adjacency the combined form requires -- see
    ``structured_pruning_entry.cpp``'s own comment above
    ``DecomposedRopePassThrough``).

    ``masked`` (default ``False``) adds an additive mask before ``Softmax``
    -- this port's own matcher recognizes it exactly as pruning.py's own
    `_resolve_decomposed_qk_root`/`_head_bias_input_is_safe`/
    `_slice_or_gather_head_bias` do (see this section's own comment above),
    so a per-head-shaped mask is pruned in lock-step with every other tensor
    in the chain. ``mask_dynamic`` (default ``False``, only meaningful when
    ``masked``) makes the mask a genuinely dynamic graph input (``Mask``,
    declared shape ``[1, 1, seq, seq]``) instead of a constant -- exercising
    ``SliceOrGatherHeadBias``'s own ``Gather``-insertion path
    (``InsertDynamicHeadBiasGather``) rather than its in-place constant-slice
    path. ``mask`` lets a caller (an oracle rebuild) pass the identical
    pre-pruning mask array back in -- a causal upper-triangular ``-1e4``
    mask, shape ``[1, 1, seq, seq]``, when left ``None``.

    ``extra_foreign_q_reshape_consumer``/``extra_foreign_repeat_kv_consumer``
    each add a second, wholly unrelated node reading the same shape constant
    as Q's own head-split ``Reshape``/K's own ``repeat_kv`` ``Expand``
    (simulating cross-layer CSE) -- exercising
    ``ApplyOneDecomposedGqaChain``'s own clone-rather-than-edit-in-place
    path (``RewriteDecomposedShapeDim``'s C++ mirror of pruning.py's own
    ``_rewrite_shape_dim``).

    ``rope`` (default ``False``) additionally applies the decomposed
    Llama/HF-style RoPE hop (``MatchDecomposedRopePassThrough`` in
    ``structured_pruning_entry.cpp``, mirroring pruning.py's own
    ``_match_decomposed_rope_pass_through``) to Q's and K's own branch, each
    independently, right after that branch's own head-split ``Transpose``
    and before (for K) ``repeat_kv``/the dot-product "swap" transpose -- two
    new graph inputs, ``Cos``/``Sin`` (``[batch, seq, D]``), feed both hops
    identically. ``qk_norm`` (default ``False``) additionally applies the
    decomposed per-head Q/K-norm hop (``MatchDecomposedQkNormPassThrough``,
    mirroring pruning.py's own ``_match_decomposed_qk_norm_pass_through``)
    to Q's and K's own branch, each independently, sitting between that
    branch's own head-split ``Reshape`` output and its own head-split
    ``Transpose`` -- the opposite side of that ``Transpose`` from where
    ``rope``'s own hop is rooted. ``q_norm_weight``/``k_norm_weight`` (each
    ``[D]``, random when left ``None``) let a caller (an oracle rebuild)
    pass the identical pre-pruning weight back in, since this hop's own
    weight is never sliced.
    """
    if Dv is None:
        Dv = D
    rng = np.random.default_rng(seed)
    Nq, Nk, Nv = H * D, KVH * D, KVH * Dv
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nk)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((H * Dv, Out)).astype(np.float32)
    if qk_norm and q_norm_weight is None:
        q_norm_weight = rng.standard_normal((D,)).astype(np.float32)
    if qk_norm and k_norm_weight is None:
        k_norm_weight = rng.standard_normal((D,)).astype(np.float32)

    def _i64(arr, name):
        return onnx.numpy_helper.from_array(np.array(arr, dtype=np.int64), name)

    def _qk_norm_lines(prefix, src, out_name, weight_name):
        # `weight * (x * rsqrt(mean(x**2, axis=-1) + eps))` -- see
        # structured_pruning_entry.cpp's own comment above
        # `DecomposedQKNormPassThrough` for the exact confirmed shape this
        # mirrors, node for node (a real export's own `Div(One, Sqrt(...))`
        # reciprocal, never `Reciprocal` for this project's own torch/opset
        # combination).
        return [
            f"{prefix}sq = Pow({src}, QKNormTwo)",
            f"{prefix}var = ReduceMean<axes=[-1],keepdims=1>({prefix}sq)",
            f"{prefix}var_eps = Add({prefix}var, QKNormEps)",
            f"{prefix}std = Sqrt({prefix}var_eps)",
            f"{prefix}inv_std = Div(QKNormOne, {prefix}std)",
            f"{prefix}scaled = Mul({src}, {prefix}inv_std)",
            f"{out_name} = Mul({weight_name}, {prefix}scaled)",
        ]

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    extra_inputs = ""

    if rope:
        assert D % 2 == 0
        initializer += [
            _i64([0], "SliceStart0"),
            _i64([D // 2], "SliceHalf"),
            _i64([D], "SliceEnd"),
            _i64([3], "SliceAxis3"),
            _i64([1], "SliceStep1"),
            _i64([1], "Ax1"),
        ]
        extra_inputs += f", float[{batch},{seq},{D}] Cos, float[{batch},{seq},{D}] Sin"

    if qk_norm:
        initializer += [
            _f32(np.array(2.0), "QKNormTwo"),
            _f32(np.array(qk_norm_eps), "QKNormEps"),
            _f32(np.array(1.0), "QKNormOne"),
            _f32(q_norm_weight, "QNormWeight"),
            _f32(k_norm_weight, "KNormWeight"),
        ]

    # Gemm requires a rank-2 input -- flattening `X` to `[batch*seq, K]`
    # first, exactly the shape onnxsim's own `fuse_matmul_add_bias_into_gemm`
    # optimizer pass produces for a real export-then-simplify pipeline (see
    # ``tests/test_pruning.py``'s own ``_decomposed_gqa_model`` docstring),
    # keeping this hand-built model runnable through a real
    # ``onnxruntime.InferenceSession`` regardless of ``bias``.
    initializer.append(_i64([batch * seq, K], "XFlatShape"))
    lines = ["xf = Reshape(X, XFlatShape)"]
    if rope:
        lines += ["CosU = Unsqueeze(Cos, Ax1)", "SinU = Unsqueeze(Sin, Ax1)"]
    q_op, k_op, v_op, o_op = (
        "MatMul(xf, Wq)",
        "MatMul(xf, Wk)",
        "MatMul(xf, Wv)",
        "MatMul(ctx2, Wout)",
    )
    if bias:
        if bq is None:
            bq = rng.standard_normal((Nq,)).astype(np.float32)
        if bk is None:
            bk = rng.standard_normal((Nk,)).astype(np.float32)
        if bv is None:
            bv = rng.standard_normal((Nv,)).astype(np.float32)
        if bout is None:
            bout = rng.standard_normal((Out,)).astype(np.float32)
        initializer += [
            _f32(bq, "Bq"),
            _f32(bk, "Bk"),
            _f32(bv, "Bv"),
            _f32(bout, "Bout"),
        ]
        q_op, k_op, v_op = "Gemm(xf, Wq, Bq)", "Gemm(xf, Wk, Bk)", "Gemm(xf, Wv, Bv)"
        o_op = "Gemm(ctx2, Wout, Bout)"

    initializer.append(_i64([batch, seq, H, D], "Sq"))
    q_transpose_src = "qr"
    if qk_norm:
        q_transpose_src = "qnout"
    lines += [
        "q0 = " + q_op,
        "qr = Reshape(q0, Sq)",
    ]
    if qk_norm:
        lines += _qk_norm_lines("qn", "qr", "qnout", "QNormWeight")
    lines.append(
        ("qt0" if rope else "qt") + f" = Transpose<perm=[0,2,1,3]>({q_transpose_src})"
    )
    if rope:
        lines += _rope_lines("q", "qt0", "qt")

    if extra_foreign_q_reshape_consumer:
        # A second, wholly unrelated Reshape reading the SAME shape
        # constant as Q's own head-split -- simulates cross-layer CSE
        # merging two textually-identical `[batch, seq, H, D]` shape
        # constants. Reshapes `xf` itself (always exactly
        # `batch*seq*K == batch*seq*H*D` elements since callers of this
        # scenario keep `K == H * D`).
        lines.append("foreign_out = Reshape(xf, Sq)")

    kv_shape_name = "Skv" if share_kv_reshape_shape and Dv == D else None
    if kv_shape_name:
        initializer.append(_i64([batch, seq, KVH, D], kv_shape_name))
        sk_name = sv_name = kv_shape_name
    else:
        initializer.append(_i64([batch, seq, KVH, D], "Sk"))
        initializer.append(_i64([batch, seq, KVH, Dv], "Sv"))
        sk_name, sv_name = "Sk", "Sv"

    lines += ["k0 = " + k_op, f"kr = Reshape(k0, {sk_name})"]
    lines += ["v0 = " + v_op, f"vr = Reshape(v0, {sv_name})"]

    n_rep = H // KVH
    needs_repeat_kv = KVH < H
    # RoPE/Q-K-norm each force the separate head-split-then-swap form for
    # K's own dot-product transpose, even when `KVH == H` leaves no genuine
    # `repeat_kv` to also force it -- see this function's own docstring.
    needs_separate_k_transpose = needs_repeat_kv or rope or qk_norm
    if needs_repeat_kv:
        assert H % KVH == 0
        initializer.append(_i64([2], "Ax2"))
        initializer.append(_i64([batch, KVH, n_rep, seq, D], "KExpandShape"))
        initializer.append(_i64([batch, H, seq, D], "KMergeShape"))
        initializer.append(_i64([batch, KVH, n_rep, seq, Dv], "VExpandShape"))
        initializer.append(_i64([batch, H, seq, Dv], "VMergeShape"))

    if needs_separate_k_transpose:
        k_headsplit_src = "kr"
        if qk_norm:
            lines += _qk_norm_lines("kn", "kr", "knout", "KNormWeight")
            k_headsplit_src = "knout"
        lines.append(f"kt0 = Transpose<perm=[0,2,1,3]>({k_headsplit_src})")
        k_src = "kt0"
        if rope:
            lines += _rope_lines("k", "kt0", "krope")
            k_src = "krope"
        if needs_repeat_kv:
            lines += [
                f"ku = Unsqueeze({k_src}, Ax2)",
                "ke = Expand(ku, KExpandShape)",
                "kre = Reshape(ke, KMergeShape)",
                "kt = Transpose<perm=[0,1,3,2]>(kre)",
            ]
            if extra_foreign_repeat_kv_consumer:
                # A second, wholly unrelated Expand reading the SAME
                # `KExpandShape` constant as K's own `repeat_kv` broadcast --
                # simulates cross-layer CSE merging two textually-identical
                # 5-D expand targets. A fresh size-[1] data operand (rather
                # than reusing any real tensor) trivially broadcasts to any
                # target shape, so this stays a valid, runnable (if unused)
                # node.
                initializer.append(
                    _f32(np.zeros((1,), dtype=np.float32), "ForeignExpandData")
                )
                lines.append(
                    "foreign_expand_out = Expand(ForeignExpandData, KExpandShape)"
                )
        else:
            lines.append(f"kt = Transpose<perm=[0,1,3,2]>({k_src})")
    else:
        lines.append("kt = Transpose<perm=[0,2,3,1]>(kr)")

    if needs_repeat_kv:
        lines += [
            "vt0 = Transpose<perm=[0,2,1,3]>(vr)",
            "vu = Unsqueeze(vt0, Ax2)",
            "ve = Expand(vu, VExpandShape)",
            "vt = Reshape(ve, VMergeShape)",
        ]
    else:
        lines.append("vt = Transpose<perm=[0,2,1,3]>(vr)")

    initializer.append(_f32(np.array(D**-0.5, dtype=np.float32), "Scale"))
    lines += ["qk = MatMul(qt, kt)", "scaled = Mul(qk, Scale)"]

    if masked:
        if mask is None:
            mask = np.triu(np.full((seq, seq), -1e4, dtype=np.float32), k=1)[
                None, None, :, :
            ]
        if mask_dynamic:
            extra_inputs += f", float[1,1,{seq},{seq}] Mask"
        else:
            initializer.append(_f32(np.asarray(mask), "Mask"))
        lines.append("premask = Add(scaled, Mask)")
        smax_in = "premask"
    else:
        smax_in = "scaled"
    lines.append(f"attn = Softmax<axis=-1>({smax_in})")
    lines.append("ctx0 = MatMul(attn, vt)")
    lines.append("ctx1 = Transpose<perm=[0,2,1,3]>(ctx0)")

    if out_reshape_wildcard:
        initializer.append(_i64([batch * seq, -1], "OutShape"))
    else:
        initializer.append(_i64([batch * seq, H * Dv], "OutShape"))
    initializer.append(_i64([batch, seq, Out], "YShape"))
    lines.append("ctx2 = Reshape(ctx1, OutShape)")
    lines.append("y0 = " + o_op)
    lines.append("Y = Reshape(y0, YShape)")

    body_lines = "\n          ".join(lines)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[{batch},{seq},{K}] X{extra_inputs}) => (float[{batch},{seq},{Out}] Y)
        {{
          {body_lines}
        }}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Dv=Dv,
        Out=Out,
        batch=batch,
        seq=seq,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        bq=bq,
        bk=bk,
        bv=bv,
        bout=bout,
        mask=mask,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_eps=qk_norm_eps,
    )


def _decomposed_weight_shapes(model):
    inits = {t.name: t for t in model.graph.initializer}
    return (
        onnx.numpy_helper.to_array(inits["Wq"]),
        onnx.numpy_helper.to_array(inits["Wk"]),
        onnx.numpy_helper.to_array(inits["Wv"]),
        onnx.numpy_helper.to_array(inits["Wout"]),
    )


def test_cpp_decomposed_gqa_pruning_matches_oracle_and_python_reference_exactly():
    model, cfg = _decomposed_gqa_model(K=32, H=8, KVH=2, D=8, Out=16, seed=1)
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)

    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    # This basic decomposed-GQA shape is fully within this C++ port's own
    # scope (no mask/RoPE/Q-K-norm/Einsum/packed-QKV, genuine GQA so no MQA
    # fast-path gap either) -- both ports must agree byte-for-byte, the
    # strongest possible parity check.
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    wq_new, wk_new, wv_new, wo_new = _decomposed_weight_shapes(pruned_cpp)
    group_size = cfg["H"] // cfg["KVH"]
    new_kv = cfg["KVH"] - round(cfg["KVH"] * 0.5)
    assert wk_new.shape[1] == new_kv * cfg["D"]
    assert wq_new.shape[1] == new_kv * group_size * cfg["D"]

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], new_kv
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)
    np.testing.assert_array_equal(wq_new, cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(wk_new, cfg["wk"][:, kv_idx])
    np.testing.assert_array_equal(wv_new, cfg["wv"][:, kv_idx])
    np.testing.assert_array_equal(wo_new, cfg["wout"][q_idx, :])

    oracle, _ = _decomposed_gqa_model(
        K=cfg["K"],
        H=len(keep_q_heads),
        KVH=new_kv,
        D=d,
        Out=cfg["Out"],
        seed=1,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        bq=cfg["bq"][q_idx],
        bk=cfg["bk"][kv_idx],
        bv=cfg["bv"][kv_idx],
        bout=cfg["bout"],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )
    rng = np.random.default_rng(2)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned_cpp, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_decomposed_plain_attention_pruning_matches_python_reference_exactly():
    # KVH == H -- the no-GQA, single pre-composed `perm=[0,2,3,1]`
    # dot-product-transpose form. Every "group" is exactly one query head
    # wide.
    model, cfg = _decomposed_gqa_model(K=32, H=4, KVH=4, D=8, Out=16, seed=3)
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    wq_new, wk_new, wv_new, _ = _decomposed_weight_shapes(pruned_cpp)
    assert wq_new.shape[1] == 2 * cfg["D"]
    assert wk_new.shape[1] == 2 * cfg["D"]
    assert wv_new.shape[1] == 2 * cfg["D"]


def test_cpp_decomposed_gqa_pruning_shared_kv_reshape_shape_edits_in_place_matches_python():
    # `share_kv_reshape_shape=True` (the default, and the common,
    # empirically-confirmed onnxsim-CSE outcome): K's and V's own head-split
    # `Reshape` reference the *same* shape-constant tensor -- exercises
    # RewriteDecomposedShapeDim's own "edit in place, no cloning needed"
    # fast path for a jointly-owned constant.
    model, cfg = _decomposed_gqa_model(
        K=32, H=8, KVH=2, D=8, Out=16, seed=5, share_kv_reshape_shape=True
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    # Exactly one `Skv`-named shape constant remains (no `_pruned` clone) --
    # both K's and V's own `Reshape` still share it, edited in place.
    skv_names = [n for n in inits if n.startswith("Skv")]
    assert skv_names == ["Skv"]
    new_kv = cfg["KVH"] - round(cfg["KVH"] * 0.5)
    dims = onnx.numpy_helper.to_array(inits["Skv"])
    assert dims[2] == new_kv


def test_cpp_decomposed_gqa_pruning_clones_shape_constant_shared_with_foreign_reader():
    # A wholly unrelated `Reshape` reads Q's own head-split shape constant
    # (`Sq`) -- ApplyOneDecomposedGqaChain must clone a fresh tensor for
    # Q's own edit rather than mutate `Sq` in place, or the foreign
    # `Reshape` would silently see the pruned (wrong) shape too.
    model, cfg = _decomposed_gqa_model(
        K=64,  # == H * D, so `foreign_out`'s own Reshape(xf, Sq) is valid.
        H=8,
        KVH=2,
        D=8,
        Out=16,
        seed=7,
        extra_foreign_q_reshape_consumer=True,
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    # The original `Sq` is left COMPLETELY untouched (still [batch, seq, H,
    # D] with the original, pre-pruning H) -- the foreign `Reshape` (whose
    # own output is unused but still present) still reads it.
    np.testing.assert_array_equal(
        inits["Sq"], [cfg["batch"], cfg["seq"], cfg["H"], cfg["D"]]
    )
    # A freshly minted `Sq_pruned` now holds the new post-pruning head
    # count, and only Q's own head-split `Reshape` was rewired onto it.
    assert "Sq_pruned" in inits
    new_kv = cfg["KVH"] - round(cfg["KVH"] * 0.5)
    group_size = cfg["H"] // cfg["KVH"]
    new_num_heads = new_kv * group_size
    assert inits["Sq_pruned"][2] == new_num_heads

    q_reshape = next(
        n for n in pruned_cpp.graph.node if n.output and n.output[0] == "qr"
    )
    assert q_reshape.input[1] == "Sq_pruned"
    foreign_reshape = next(
        n for n in pruned_cpp.graph.node if n.output and n.output[0] == "foreign_out"
    )
    assert foreign_reshape.input[1] == "Sq"

    # Functional correctness too: the foreign branch (dead code, feeds
    # nothing) doesn't affect `Y`, and pruning still computes the right
    # answer.
    rng = np.random.default_rng(8)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y,) = _run(pruned_cpp, {"X": x})
    assert y.shape == (cfg["batch"], cfg["seq"], cfg["Out"])


def test_cpp_decomposed_gqa_pruning_clones_repeat_kv_expand_shape_shared_with_foreign_reader():
    # A wholly unrelated `Expand` reads K's own `repeat_kv` broadcast target
    # shape (`KExpandShape`) -- must be cloned, not edited in place, exactly
    # like the head-split-Reshape case above but for the `repeat_kv` shape
    # plumbing this family's own section comment calls out as the genuinely
    # novel part of this port.
    model, cfg = _decomposed_gqa_model(
        K=32,
        H=8,
        KVH=2,
        D=8,
        Out=16,
        seed=9,
        extra_foreign_repeat_kv_consumer=True,
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    n_rep = cfg["H"] // cfg["KVH"]
    np.testing.assert_array_equal(
        inits["KExpandShape"],
        [cfg["batch"], cfg["KVH"], n_rep, cfg["seq"], cfg["D"]],
    )
    assert "KExpandShape_pruned" in inits
    new_kv = cfg["KVH"] - round(cfg["KVH"] * 0.5)
    assert inits["KExpandShape_pruned"][1] == new_kv

    real_expand = next(
        n for n in pruned_cpp.graph.node if n.output and n.output[0] == "ke"
    )
    assert real_expand.input[1] == "KExpandShape_pruned"
    foreign_expand = next(
        n
        for n in pruned_cpp.graph.node
        if n.output and n.output[0] == "foreign_expand_out"
    )
    assert foreign_expand.input[1] == "KExpandShape"


def test_cpp_decomposed_gqa_pruning_out_reshape_wildcard_still_prunes_matches_python():
    # The combine-back Reshape's own target shape's last entry is already a
    # wildcard (-1) -- `Reshape` auto-infers it, so
    # ApplyOneDecomposedGqaChain never needs to (and doesn't) touch it at
    # all; pruning still proceeds normally otherwise.
    model, cfg = _decomposed_gqa_model(
        K=32, H=8, KVH=2, D=8, Out=16, seed=11, out_reshape_wildcard=True
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    wq_new, _, _, _ = _decomposed_weight_shapes(pruned_cpp)
    new_kv = cfg["KVH"] - round(cfg["KVH"] * 0.5)
    group_size = cfg["H"] // cfg["KVH"]
    assert wq_new.shape[1] == new_kv * group_size * cfg["D"]

    rng = np.random.default_rng(12)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y,) = _run(pruned_cpp, {"X": x})
    assert y.shape == (cfg["batch"], cfg["seq"], cfg["Out"])


def test_cpp_decomposed_gqa_pruning_zero_sparsity_is_a_no_op():
    model, _ = _decomposed_gqa_model(K=32, H=8, KVH=2, D=8, Out=16, seed=13)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.0)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_decomposed_gqa_pruning_unsupported_transpose_perm_is_declined_matches_python():
    # A non-standard perm on Q's own head-split Transpose -- this pass only
    # ever recognizes the exact perms real exports use; anything else must
    # decline outright, never guessed at. A genuinely invalid topology, so
    # BOTH ports must agree it's a no-op, not merely a documented scope
    # divergence.
    model, _ = _decomposed_gqa_model(K=32, H=4, KVH=4, D=8, Out=16, seed=19)
    for n in model.graph.node:
        if n.output and n.output[0] == "qt":
            n.ClearField("attribute")
            n.attribute.append(onnx.helper.make_attribute("perm", [0, 1, 2, 3]))
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


def test_cpp_decomposed_gqa_pruning_with_constant_mask_matches_python_reference_exactly():
    # An additive mask `Add` before `Softmax`, a constant of the
    # schema-documented per-head-broadcastable shape (`[1, 1, seq, seq]`,
    # rank-4 with the num_heads-aligned axis broadcast-sized) -- now matched
    # by `ResolveDecomposedQkRoot`'s own mask detection and left correctly
    # untouched (a broadcast, already correct for any head count) in lock
    # step with pruning.py's own pure-Python `apply_attention_head_pruning`.
    # Both ports must agree byte-for-byte, the strongest possible parity
    # check.
    model, cfg = _decomposed_gqa_model(
        K=32, H=8, KVH=2, D=8, Out=16, seed=21, masked=True
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    # The mask is rank-4 `[1, 1, seq, seq]` -- axis 1 (the num_heads-aligned
    # slot) is a broadcast (size 1), so HeadBiasAxis resolves to `-1` and
    # SliceOrGatherHeadBias leaves it completely untouched, exactly as
    # pruning.py's own `_head_bias_axis`/`_slice_or_gather_head_bias` do.
    mask_after = next(
        onnx.numpy_helper.to_array(t)
        for t in pruned_cpp.graph.initializer
        if t.name == "Mask"
    )
    np.testing.assert_array_equal(mask_after, cfg["mask"])

    rng = np.random.default_rng(22)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y,) = _run(pruned_cpp, {"X": x})
    assert y.shape == (cfg["batch"], cfg["seq"], cfg["Out"])
    assert np.all(np.isfinite(y))


def test_cpp_decomposed_gqa_pruning_with_dynamic_mask_matches_python_reference_exactly():
    # The genuinely dynamic (graph-input, non-constant) analogue of the test
    # above -- `SliceOrGatherHeadBias`'s own "leave a confirmed broadcast
    # alone" path applies identically whether the mask is a constant or a
    # dynamic tensor (DynamicHeadBiasAxis resolves the SAME `-1` from the
    # graph input's own declared `[1, 1, seq, seq]` shape, via
    # `ValueInfoByName`), so no `Gather` is ever inserted here either -- both
    # ports must still agree byte-for-byte.
    model, cfg = _decomposed_gqa_model(
        K=32, H=8, KVH=2, D=8, Out=16, seed=23, masked=True, mask_dynamic=True
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    # No new `Gather` node is expected for a broadcast mask -- confirms this
    # test genuinely exercises "already correct, leave alone", not a
    # trivially-always-passing insertion.
    assert not any(n.op_type == "Gather" for n in pruned_cpp.graph.node)

    rng = np.random.default_rng(24)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y,) = _run(pruned_cpp, {"X": x, "Mask": cfg["mask"]})
    assert y.shape == (cfg["batch"], cfg["seq"], cfg["Out"])
    assert np.all(np.isfinite(y))


def test_cpp_decomposed_gqa_pruning_with_per_head_dynamic_mask_gathers_matches_python():
    # A genuinely PER-HEAD (not broadcast) dynamic mask -- rank-4
    # `[1, num_heads, seq, seq]`, axis 1 sized exactly `num_heads` -- the one
    # case that actually needs SliceOrGatherHeadBias's own `Gather`-insertion
    # path (InsertDynamicHeadBiasGather), since the mask's own real values
    # aren't available to slice in place at prune time the way a constant's
    # are. Both ports must still agree byte-for-byte, and the pruned graph's
    # own newly-spliced `Gather` must select exactly the kept query heads.
    H, KVH, D, seq = 8, 2, 8, 4
    rng = np.random.default_rng(25)
    mask = rng.standard_normal((1, H, seq, seq)).astype(np.float32)
    model, cfg = _decomposed_gqa_model(
        K=32,
        H=H,
        KVH=KVH,
        D=D,
        Out=16,
        seq=seq,
        seed=25,
        masked=True,
        mask_dynamic=True,
        mask=mask,
    )
    # `_decomposed_gqa_model` only ever declares a broadcast `[1, 1, seq,
    # seq]` dynamic mask input -- widen it here to the genuine per-head shape
    # this test needs (the model's own `Add(scaled, Mask)` node is agnostic
    # to Mask's exact declared shape; only the graph's own declared
    # ValueInfo, which HeadBiasInputIsSafe/SliceOrGatherHeadBias consult, and
    # the runtime feed need to agree).
    for vi in model.graph.input:
        if vi.name == "Mask":
            vi.type.tensor_type.shape.dim[1].dim_value = H

    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    gather_nodes = [n for n in pruned_cpp.graph.node if n.op_type == "Gather"]
    assert len(gather_nodes) == 1
    gather = gather_nodes[0]
    assert gather.input[0] == "Mask"
    axis = next(a.i for a in gather.attribute if a.name == "axis")
    assert axis == 1
    indices_init = next(
        t for t in pruned_cpp.graph.initializer if t.name == gather.input[1]
    )
    indices = onnx.numpy_helper.to_array(indices_init)

    group_size = H // KVH
    new_kv = KVH - round(KVH * 0.5)
    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], H, KVH, D, new_kv
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    np.testing.assert_array_equal(indices, np.sort(keep_q_heads))

    # The pruned graph's own "Mask" graph INPUT keeps its ORIGINAL declared
    # shape (`[1, H, seq, seq]`) -- SliceOrGatherHeadBias only ever rewires
    # the mask consumer's own edge onto the new Gather's output, it never
    # touches `graph.input` itself (the real dynamic tensor is still
    # supplied, at its original shape, by whatever produces it at
    # inference); the freshly spliced `Gather` performs the actual
    # per-head selection at runtime. So the full, ORIGINAL-shaped mask is
    # fed here -- not a pre-sliced one -- exactly as a real caller would
    # still supply it, unaware the model was pruned underneath it.
    q_idx, kv_idx = _head_idx(keep_q_heads, D), _head_idx(keep_groups, D)
    oracle, _ = _decomposed_gqa_model(
        K=cfg["K"],
        H=len(keep_q_heads),
        KVH=new_kv,
        D=D,
        Out=cfg["Out"],
        seq=seq,
        seed=25,
        masked=True,
        mask_dynamic=True,
        mask=mask[:, keep_q_heads, :, :],
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        bq=cfg["bq"][q_idx],
        bk=cfg["bk"][kv_idx],
        bv=cfg["bv"][kv_idx],
        bout=cfg["bout"],
        batch=cfg["batch"],
    )
    # Same fixup as the pruned-under-test model above: `_decomposed_gqa_model`
    # always declares a broadcast `[1, 1, seq, seq]` dynamic mask input --
    # widen it to match the actual (post-pruning) per-head array being fed.
    for vi in oracle.graph.input:
        if vi.name == "Mask":
            vi.type.tensor_type.shape.dim[1].dim_value = len(keep_q_heads)

    rng = np.random.default_rng(26)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_cpp,) = _run(pruned_cpp, {"X": x, "Mask": mask})
    (y_oracle,) = _run(oracle, {"X": x, "Mask": mask[:, keep_q_heads, :, :]})
    np.testing.assert_allclose(y_cpp, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_decomposed_gqa_pruning_with_unresolvable_mask_shape_declines_matches_python():
    # A mask whose own per-head axis is neither a genuine broadcast (size 1)
    # nor the pre-pruning head count -- HeadBiasAxis/`_head_bias_axis`
    # decline to classify it either way, so BOTH ports must decline the
    # WHOLE chain rather than guess (never a mis-slice). A genuinely invalid
    # topology, so this is a true agreement check, not a documented
    # divergence.
    H, KVH, D, seq = 8, 2, 8, 4
    bad_mask = np.zeros((1, H + 1, seq, seq), dtype=np.float32)
    model, _ = _decomposed_gqa_model(
        K=32,
        H=H,
        KVH=KVH,
        D=D,
        Out=16,
        seq=seq,
        seed=27,
        masked=True,
        mask=bad_mask,
    )
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


def _oracle_keep_query_heads(wq, d, num_heads, keep_count):
    # Mirrors pruning.py's own `_gqa_query_head_importance`: each query
    # head ranked directly by its own Q weight block alone (the shared K/V
    # block deliberately omitted -- see that function's own docstring for
    # why it cannot change which heads rank highest).
    importance = np.array(
        [np.linalg.norm(wq[:, h * d : (h + 1) * d]) for h in range(num_heads)]
    )
    return np.sort(np.argsort(-importance)[:keep_count])


def test_cpp_decomposed_mqa_pruning_matches_python_reference_exactly():
    # True decomposed MQA (kv_num_heads == 1, more than one query head):
    # ApplyOneDecomposedGqaChain's own `is_mqa` fast path -- mirrors
    # pruning.py's own `_apply_one_decomposed_gqa_chain` docstring exactly.
    # The ordinary group-granularity formula (`max(1, 1 - round(1*
    # sparsity))`) is always `1 == kv_num_heads`, so it can never drop
    # anything no matter how many query heads share the one KV head; this
    # path instead ranks and drops individual query heads directly, leaving
    # the single shared KV head -- and both its own K/V producer weights --
    # completely untouched. Both ports must now agree byte-for-byte.
    K, H, KVH, D, Out = 32, 8, 1, 8, 16
    model, cfg = _decomposed_gqa_model(K=K, H=H, KVH=KVH, D=D, Out=Out, seed=201)
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)

    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    wq_new, wk_new, wv_new, wo_new = _decomposed_weight_shapes(pruned_cpp)
    q_keep_count = max(1, H - round(H * 0.5))
    assert q_keep_count < H
    assert wq_new.shape[1] == q_keep_count * D
    # K/V (and their own biases) are completely untouched by the MQA fast
    # path -- byte-for-byte identical to the pre-pruning tensors.
    np.testing.assert_array_equal(wk_new, cfg["wk"])
    np.testing.assert_array_equal(wv_new, cfg["wv"])

    keep_q_heads = _oracle_keep_query_heads(cfg["wq"], D, H, q_keep_count)
    q_idx = _head_idx(keep_q_heads, D)
    np.testing.assert_array_equal(wq_new, cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(wo_new, cfg["wout"][q_idx, :])

    oracle, _ = _decomposed_gqa_model(
        K=K,
        H=len(keep_q_heads),
        KVH=KVH,
        D=D,
        Out=Out,
        seed=201,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"],
        wv=cfg["wv"],
        wout=cfg["wout"][q_idx, :],
        bq=cfg["bq"][q_idx],
        bk=cfg["bk"],
        bv=cfg["bv"],
        bout=cfg["bout"],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )
    rng = np.random.default_rng(2)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    (y_pruned,) = _run(pruned_cpp, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_decomposed_mqa_pruning_zero_sparsity_is_a_no_op():
    model, _ = _decomposed_gqa_model(K=32, H=8, KVH=1, D=8, Out=16, seed=203)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.0)
    assert pruned.SerializeToString() == model.SerializeToString()


# --- Packed-QKV-then-Split producer, decomposed shape --------------------
#
# Q's/K's/V's own raw projection output all trace back to ONE combined
# MatMul/vanilla-Gemm projection (optionally biased) whose wide
# `[..., nq+nk+nv]` output is split, by a plain `Split(axis=-1)` node, into
# exactly three column ranges in Q-then-K-then-V order -- mirrors
# pruning.py's own `_match_decomposed_packed_qkv_producer`/
# `_match_packed_qkv_split` (see MatchDecomposedPackedQkvProducer's own
# comment in structured_pruning_entry.cpp for the exact topology).


def _decomposed_packed_qkv_model(
    K=32,
    H=4,
    KVH=2,
    D=8,
    Dv=None,
    Out=16,
    batch=1,
    seq=4,
    seed=0,
    bias=True,
    with_flatten_reshape=False,
    masked=False,
    mask_dynamic=False,
    mask=None,
):
    """Builds the same decomposed-attention graph `_decomposed_gqa_model`
    does, EXCEPT Q's/K's/V's raw projections are produced by ONE packed
    MatMul/Gemm projection feeding a `Split(axis=-1)` node into
    Q-then-K-then-V column ranges, rather than three independent
    producers -- the shape MatchDecomposedPackedQkvProducer recognizes.

    `with_flatten_reshape=True` additionally inserts a pass-through
    `Reshape` between the packed projection and the `Split` -- the real
    `fuse_matmul_add_bias_into_gemm`-produced artifact
    MatchDecomposedPackedQkvProducer's own one-hop resolution handles (see
    that function's own comment).

    `masked`/`mask_dynamic`/`mask` (identical semantics/defaults to
    `_decomposed_gqa_model`'s own same-named params) additionally insert an
    additive mask before `Softmax` -- a genuinely NEW combined scenario none
    of the four sub-shape branches individually tested: a packed-QKV-then-
    `Split` producer together with an additive mask on the SAME chain. Both
    `ResolveDecomposedQkRoot`'s own mask detection and
    `MatchDecomposedPackedQkvProducer`'s own producer resolution are
    independent match-time steps in `FindDecomposedGqaChains` -- one
    resolving backward from `Softmax`, the other forward from Q's/K's/V's own
    raw projection outputs -- so this combination is within this port's own
    matched scope.
    """
    if Dv is None:
        Dv = D
    rng = np.random.default_rng(seed)
    Nq, Nk, Nv = H * D, KVH * D, KVH * Dv
    N = Nq + Nk + Nv
    w = rng.standard_normal((K, N)).astype(np.float32)
    wout = rng.standard_normal((H * Dv, Out)).astype(np.float32)

    def _i64(arr, name):
        return onnx.numpy_helper.from_array(np.array(arr, dtype=np.int64), name)

    initializer = [_f32(w, "W"), _f32(wout, "Wout")]
    initializer.append(_i64([batch * seq, K], "XFlatShape"))
    initializer.append(_i64([Nq, Nk, Nv], "SplitSizes"))
    extra_inputs = ""
    lines = ["xf = Reshape(X, XFlatShape)"]

    b = None
    bout = None
    if bias:
        b = rng.standard_normal((N,)).astype(np.float32)
        bout = rng.standard_normal((Out,)).astype(np.float32)
        initializer += [_f32(b, "B"), _f32(bout, "Bout")]
        proj_op, o_op = "Gemm(xf, W, B)", "Gemm(ctx2, Wout, Bout)"
    else:
        proj_op, o_op = "MatMul(xf, W)", "MatMul(ctx2, Wout)"

    lines.append("packed = " + proj_op)
    split_src = "packed"
    if with_flatten_reshape:
        initializer.append(_i64([batch, seq, N], "PackedShape"))
        lines.append("packed_r = Reshape(packed, PackedShape)")
        split_src = "packed_r"
    lines.append(f"q0, k0, v0 = Split <axis = -1> ({split_src}, SplitSizes)")

    initializer.append(_i64([batch, seq, H, D], "Sq"))
    lines.append("qr = Reshape(q0, Sq)")
    lines.append("qt = Transpose<perm=[0,2,1,3]>(qr)")

    kv_shape_name = "Skv" if Dv == D else None
    if kv_shape_name:
        initializer.append(_i64([batch, seq, KVH, D], kv_shape_name))
        sk_name = sv_name = kv_shape_name
    else:
        initializer.append(_i64([batch, seq, KVH, D], "Sk"))
        initializer.append(_i64([batch, seq, KVH, Dv], "Sv"))
        sk_name, sv_name = "Sk", "Sv"
    lines.append(f"kr = Reshape(k0, {sk_name})")
    lines.append(f"vr = Reshape(v0, {sv_name})")

    n_rep = H // KVH
    needs_repeat_kv = KVH < H
    if needs_repeat_kv:
        assert H % KVH == 0
        initializer.append(_i64([2], "Ax2"))
        initializer.append(_i64([batch, KVH, n_rep, seq, D], "KExpandShape"))
        initializer.append(_i64([batch, H, seq, D], "KMergeShape"))
        initializer.append(_i64([batch, KVH, n_rep, seq, Dv], "VExpandShape"))
        initializer.append(_i64([batch, H, seq, Dv], "VMergeShape"))
        lines.append("kt0 = Transpose<perm=[0,2,1,3]>(kr)")
        lines += [
            "ku = Unsqueeze(kt0, Ax2)",
            "ke = Expand(ku, KExpandShape)",
            "kre = Reshape(ke, KMergeShape)",
            "kt = Transpose<perm=[0,1,3,2]>(kre)",
        ]
        lines += [
            "vt0 = Transpose<perm=[0,2,1,3]>(vr)",
            "vu = Unsqueeze(vt0, Ax2)",
            "ve = Expand(vu, VExpandShape)",
            "vt = Reshape(ve, VMergeShape)",
        ]
    else:
        lines.append("kt = Transpose<perm=[0,2,3,1]>(kr)")
        lines.append("vt = Transpose<perm=[0,2,1,3]>(vr)")

    initializer.append(_f32(np.array(D**-0.5, dtype=np.float32), "Scale"))
    lines += ["qk = MatMul(qt, kt)", "scaled = Mul(qk, Scale)"]
    if masked:
        if mask is None:
            mask = np.triu(np.full((seq, seq), -1e4, dtype=np.float32), k=1)[
                None, None, :, :
            ]
        if mask_dynamic:
            extra_inputs += f", float[1,1,{seq},{seq}] Mask"
        else:
            initializer.append(_f32(np.asarray(mask), "Mask"))
        lines.append("premask = Add(scaled, Mask)")
        smax_in = "premask"
    else:
        smax_in = "scaled"
    lines.append(f"attn = Softmax<axis=-1>({smax_in})")
    lines.append("ctx0 = MatMul(attn, vt)")
    lines.append("ctx1 = Transpose<perm=[0,2,1,3]>(ctx0)")

    initializer.append(_i64([batch * seq, H * Dv], "OutShape"))
    initializer.append(_i64([batch, seq, Out], "YShape"))
    lines.append("ctx2 = Reshape(ctx1, OutShape)")
    lines.append("y0 = " + o_op)
    lines.append("Y = Reshape(y0, YShape)")

    body_lines = "\n          ".join(lines)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[{batch},{seq},{K}] X{extra_inputs}) => (float[{batch},{seq},{Out}] Y)
        {{
          {body_lines}
        }}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Dv=Dv,
        Out=Out,
        batch=batch,
        seq=seq,
        w=w,
        wout=wout,
        b=b,
        bout=bout,
        Nq=Nq,
        Nk=Nk,
        Nv=Nv,
        mask=mask,
    )


def test_cpp_decomposed_packed_qkv_pruning_matches_python_reference_exactly():
    # Q's/K's/V's "own weight" is really one shared packed tensor here,
    # sliced once by a combined index set, with the `Split`'s own
    # split-sizes constant shrunk to match -- mirrors pruning.py's own
    # `_apply_one_decomposed_gqa_chain` packed branch exactly (see
    # ApplyOneDecomposedGqaChain's own comment). Genuine GQA (KVH < H), so
    # no MQA fast path involved -- isolates the packed-weight column-index
    # arithmetic on its own.
    K, H, KVH, D, Out = 32, 8, 2, 8, 16
    model, cfg = _decomposed_packed_qkv_model(
        K=K, H=H, KVH=KVH, D=D, Out=Out, seed=11, bias=True
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)

    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    Nq, Nk = cfg["Nq"], cfg["Nk"]
    w, b = cfg["w"], cfg["b"]
    wq, wk, wv = w[:, :Nq], w[:, Nq : Nq + Nk], w[:, Nq + Nk :]
    bq, bk, bv = b[:Nq], b[Nq : Nq + Nk], b[Nq + Nk :]

    group_size = H // KVH
    new_kv = max(1, KVH - round(KVH * 0.5))
    keep_groups = _oracle_keep_groups(wq, wk, wv, H, KVH, D, new_kv)
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx, kv_idx = _head_idx(keep_q_heads, D), _head_idx(keep_groups, D)

    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    expected_w = np.concatenate([wq[:, q_idx], wk[:, kv_idx], wv[:, kv_idx]], axis=1)
    expected_b = np.concatenate([bq[q_idx], bk[kv_idx], bv[kv_idx]])
    np.testing.assert_array_equal(inits["W"], expected_w)
    np.testing.assert_array_equal(inits["B"], expected_b)
    np.testing.assert_array_equal(
        inits["SplitSizes"],
        np.array([len(q_idx), len(kv_idx), len(kv_idx)], dtype=np.int64),
    )

    oracle, _ = _decomposed_gqa_model(
        K=K,
        H=len(keep_q_heads),
        KVH=new_kv,
        D=D,
        Out=Out,
        seed=1,
        wq=wq[:, q_idx],
        wk=wk[:, kv_idx],
        wv=wv[:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        bq=bq[q_idx],
        bk=bk[kv_idx],
        bv=bv[kv_idx],
        bout=cfg["bout"],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )
    rng = np.random.default_rng(2)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    (y_pruned,) = _run(pruned_cpp, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_decomposed_packed_qkv_pruning_through_flatten_reshape_matches_python():
    # The genuinely different, ALSO real export artifact
    # MatchDecomposedPackedQkvProducer's own one-hop resolution handles: a
    # pass-through `Reshape` sitting directly between the packed
    # MatMul/Gemm projection and the `Split` node (onnxsim's own
    # `fuse_matmul_add_bias_into_gemm` optimizer pass flattening a packed
    # projection's rank-3 `MatMul`/`Add` into a rank-2 `Gemm`, then
    # reshaping back to rank-3 since `Split` isn't a `Reshape` it can
    # trivially absorb).
    K, H, KVH, D, Out = 32, 4, 2, 8, 16
    model, cfg = _decomposed_packed_qkv_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        seed=17,
        bias=False,
        with_flatten_reshape=True,
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)

    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    # The pass-through Reshape's own target shape's last entry (the packed
    # weight's own real output width) must shrink in lock-step with the
    # packed weight's own combined slice.
    reshape = next(
        n
        for n in pruned_cpp.graph.node
        if n.op_type == "Reshape" and n.output and n.output[0] == "packed_r"
    )
    shape_init = next(
        t for t in pruned_cpp.graph.initializer if t.name == reshape.input[1]
    )
    new_kv = max(1, KVH - round(KVH * 0.5))
    group_size = H // KVH
    new_num_heads = new_kv * group_size
    expected_last = new_num_heads * D + 2 * new_kv * D
    assert onnx.numpy_helper.to_array(shape_init)[-1] == expected_last


def test_cpp_decomposed_packed_qkv_pruning_zero_sparsity_is_a_no_op():
    model, _ = _decomposed_packed_qkv_model(
        K=32, H=8, KVH=2, D=8, Out=16, seed=23, bias=True
    )
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.0)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_decomposed_packed_qkv_with_dynamic_mask_pruning_matches_python_reference_exactly():
    # Combined-scenario regression test (not covered by any single one of the
    # four independent sub-shape branches this merge combines): a
    # packed-QKV-then-`Split` producer (`MatchDecomposedPackedQkvProducer`)
    # TOGETHER with a genuinely dynamic additive mask (`ResolveDecomposedQkRoot`'s
    # own mask detection, `HeadBiasInputIsSafe`/`SliceOrGatherHeadBias` ->
    # `InsertDynamicHeadBiasGather`) on the SAME chain. The two matchers work
    # in opposite directions over disjoint parts of the chain -- the mask
    # resolves backward from `Softmax`, the packed producer resolves forward
    # from Q's/K's/V's own raw projection outputs -- so nothing about either
    # one's own match-time or apply-time logic depends on the other, and this
    # combination is within this port's own matched scope. Both ports must
    # agree byte-for-byte.
    K, H, KVH, D, Out = 32, 8, 2, 8, 16
    model, cfg = _decomposed_packed_qkv_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        seed=29,
        bias=True,
        masked=True,
        mask_dynamic=True,
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    rng = np.random.default_rng(30)
    mask_val = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)

    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    # A dynamic mask ([1, 1, seq, seq]) broadcasts identically across every
    # head -- `HeadBiasInputIsSafe`/`DynamicHeadBiasAxis` resolve this as a
    # broadcast (`axis == -1`), so no `Gather` is ever spliced in for it, and
    # `Mask` itself stays a plain, untouched graph input either way; this
    # confirms the packed-weight slicing above didn't somehow also touch the
    # mask machinery (or vice versa).
    assert not any(n.op_type == "Gather" for n in pruned_cpp.graph.node)
    mask_inputs = [i for i in pruned_cpp.graph.input if i.name == "Mask"]
    assert len(mask_inputs) == 1

    Nq, Nk = cfg["Nq"], cfg["Nk"]
    w, b = cfg["w"], cfg["b"]
    wq, wk, wv = w[:, :Nq], w[:, Nq : Nq + Nk], w[:, Nq + Nk :]
    bq, bk, bv = b[:Nq], b[Nq : Nq + Nk], b[Nq + Nk :]

    group_size = H // KVH
    new_kv = max(1, KVH - round(KVH * 0.5))
    keep_groups = _oracle_keep_groups(wq, wk, wv, H, KVH, D, new_kv)
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx, kv_idx = _head_idx(keep_q_heads, D), _head_idx(keep_groups, D)

    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    expected_w = np.concatenate([wq[:, q_idx], wk[:, kv_idx], wv[:, kv_idx]], axis=1)
    expected_b = np.concatenate([bq[q_idx], bk[kv_idx], bv[kv_idx]])
    np.testing.assert_array_equal(inits["W"], expected_w)
    np.testing.assert_array_equal(inits["B"], expected_b)

    rng2 = np.random.default_rng(31)
    x = rng2.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    (y_pruned,) = _run(pruned_cpp, {"X": x, "Mask": mask_val})
    assert y_pruned.shape == (cfg["batch"], cfg["seq"], Out)


# --- Decomposed attention with an `Einsum`-based QK^T/AV product -----------
#
# A plain `torch.einsum`/`tf.einsum`-written attention block (common in
# `x-transformers`/`vit-pytorch`/`performer-pytorch`-style reimplementations,
# and `tf2onnx`-exported `tf.einsum` code) never emits the `MatMul` the model
# above builds for either the QK^T score or the AV output product -- it uses
# a literal `Einsum` node for each instead, with K's own natural
# `[..., seq_k, head_size]` layout contracted directly (no separate
# dot-product-transpose `Transpose` at all, unlike the `MatMul` shape above).
# `EinsumEquationIsBatchedMatmul` (`structured_pruning_entry.cpp`, a
# line-for-line port of pruning.py's own `_einsum_equation_is_batched_
# matmul`) recognizes any equation string PROVABLY equivalent to that
# specific batched matmul -- regardless of which letters name each axis, and
# regardless of incidental whitespace -- so both `FindDecomposedGqaChains`'s
# QK^T and AV matching accept an `Einsum` node here exactly where they'd
# otherwise require `MatMul`.


def _decomposed_einsum_gqa_model(
    K=32,
    H=4,
    KVH=2,
    D=8,
    Out=16,
    batch=1,
    seq=4,
    seed=0,
    qk_equation="bhid,bhjd->bhij",
    av_equation="bhij,bhjd->bhid",
    av_attn_first=True,
    q_rope=False,
):
    """Builds the `Einsum`-based analogue of `_decomposed_gqa_model`:
    ``Linear(Q/K/V) -> Reshape(to heads) -> Transpose(to BHSD) -> [repeat_kv:
    Unsqueeze -> Expand -> Reshape, only when KVH < H] -> Einsum(Q, K) ->
    scale -> Softmax -> Einsum(., V) -> Transpose(back) -> Reshape(back to
    hidden) -> Linear(O)``.

    Unlike `_decomposed_gqa_model`, K's own branch is built EXACTLY like V's
    own (plain head-split `Transpose(perm=[0, 2, 1, 3])`, then the identical
    optional `repeat_kv` broadcast when `KVH < H`) -- there is no combined
    `perm=[0, 2, 3, 1]` or separate `perm=[0, 1, 3, 2]` dot-product-transpose
    `Transpose` at all, since a literal `Einsum` node contracts K's own
    natural `[..., seq_k, head_size]` layout directly (that's the entire
    reason a real export uses `Einsum` here instead of `MatMul` in the first
    place) -- mirrors pruning.py's own identical `Einsum` branch in
    `_find_decomposed_gqa_chains`/`_resolve_decomposed_qk_root`.

    `qk_equation`/`av_equation` let callers exercise different (but
    equivalent) equation-string spellings -- renamed axis letters, incidental
    whitespace -- all of which `EinsumEquationIsBatchedMatmul` must accept
    identically. `av_attn_first` selects which of the AV `Einsum` node's own
    two input slots the attention-weights (`Softmax` output) operand lands
    in -- `EinsumEquationIsBatchedMatmul`'s own `first_operand_index` must
    track whichever slot it actually is, exactly as flexibly as the `MatMul`
    shape already handles via `attn_is_first`.

    `q_rope` (default `False`) additionally applies the decomposed
    Llama/HF-style RoPE hop (`_rope_lines`, shared with `_decomposed_gqa_
    model`) to Q's own branch only, right after Q's own head-split
    `Transpose` and before the `Einsum` QK^T product -- a genuinely NEW
    combined scenario none of the four sub-shape branches individually
    tested: an `Einsum`-based QK^T product together with a decomposed RoPE
    pass-through on Q's own branch. This is within this port's own matched
    scope (`WalkBackThroughDecomposedRope` is called for Q's own branch
    unconditionally in `FindDecomposedGqaChains`, regardless of whether the
    QK^T product resolves via `MatMul` or `Einsum` -- only K's own `Einsum`
    branch unconditionally skips it, since K's raw operand feeds the
    `Einsum` directly with no separate dot-product-transpose hop for RoPE to
    sit in front of, see MatchDecomposedPackedQkvProducer's neighboring
    comment for the analogous K-side scope note). K's own branch is left
    untouched (no RoPE) either way.
    """
    rng = np.random.default_rng(seed)
    Nq, Nk, Nv = H * D, KVH * D, KVH * D
    wq = rng.standard_normal((K, Nq)).astype(np.float32)
    wk = rng.standard_normal((K, Nk)).astype(np.float32)
    wv = rng.standard_normal((K, Nv)).astype(np.float32)
    wout = rng.standard_normal((H * D, Out)).astype(np.float32)
    bq = rng.standard_normal((Nq,)).astype(np.float32)
    bk = rng.standard_normal((Nk,)).astype(np.float32)
    bv = rng.standard_normal((Nv,)).astype(np.float32)
    bout = rng.standard_normal((Out,)).astype(np.float32)

    def _i64(arr, name):
        return onnx.numpy_helper.from_array(np.array(arr, dtype=np.int64), name)

    initializer = [
        _f32(wq, "Wq"),
        _f32(wk, "Wk"),
        _f32(wv, "Wv"),
        _f32(wout, "Wout"),
        _f32(bq, "Bq"),
        _f32(bk, "Bk"),
        _f32(bv, "Bv"),
        _f32(bout, "Bout"),
        _i64([batch * seq, K], "XFlatShape"),
        _i64([batch, seq, H, D], "Sq"),
        _i64([batch, seq, KVH, D], "Skv"),
        _i64([batch * seq, H * D], "OutShape"),
        _i64([batch, seq, Out], "YShape"),
    ]
    extra_inputs = ""

    lines = ["xf = Reshape(X, XFlatShape)"]
    if q_rope:
        assert D % 2 == 0
        initializer += [
            _i64([0], "SliceStart0"),
            _i64([D // 2], "SliceHalf"),
            _i64([D], "SliceEnd"),
            _i64([3], "SliceAxis3"),
            _i64([1], "SliceStep1"),
            _i64([1], "Ax1"),
        ]
        extra_inputs += f", float[{batch},{seq},{D}] Cos, float[{batch},{seq},{D}] Sin"
        lines += ["CosU = Unsqueeze(Cos, Ax1)", "SinU = Unsqueeze(Sin, Ax1)"]
    lines += [
        "q0 = Gemm(xf, Wq, Bq)",
        "qr = Reshape(q0, Sq)",
        ("qt0" if q_rope else "qt") + " = Transpose<perm=[0,2,1,3]>(qr)",
    ]
    if q_rope:
        lines += _rope_lines("q", "qt0", "qt")
    lines += [
        "k0 = Gemm(xf, Wk, Bk)",
        "kr = Reshape(k0, Skv)",
        "ktr = Transpose<perm=[0,2,1,3]>(kr)",
        "v0 = Gemm(xf, Wv, Bv)",
        "vr = Reshape(v0, Skv)",
        "vtr = Transpose<perm=[0,2,1,3]>(vr)",
    ]

    needs_repeat_kv = KVH < H
    if needs_repeat_kv:
        assert H % KVH == 0
        n_rep = H // KVH
        initializer += [
            _i64([2], "Ax2"),
            _i64([batch, KVH, n_rep, seq, D], "KExpandShape"),
            _i64([batch, H, seq, D], "KMergeShape"),
            _i64([batch, KVH, n_rep, seq, D], "VExpandShape"),
            _i64([batch, H, seq, D], "VMergeShape"),
        ]
        lines += [
            "ku = Unsqueeze(ktr, Ax2)",
            "ke = Expand(ku, KExpandShape)",
            "kt = Reshape(ke, KMergeShape)",
            "vu = Unsqueeze(vtr, Ax2)",
            "ve = Expand(vu, VExpandShape)",
            "vt = Reshape(ve, VMergeShape)",
        ]
        kt_name, vt_name = "kt", "vt"
    else:
        # No `repeat_kv` -- the `Einsum` node's own operand must be K's/V's
        # own plain head-split `Transpose` output DIRECTLY (`ktr`/`vtr`), the
        # same way `q_bhsd_name` (`qt`) already feeds `qk` directly above --
        # any extra pass-through node here would break
        # `MatchDecomposedHeadSplit`'s own fixed node-adjacency requirement.
        kt_name, vt_name = "ktr", "vtr"

    initializer.append(_f32(np.array(D**-0.5, dtype=np.float32), "Scale"))
    lines += [
        f'qk = Einsum<equation="{qk_equation}">(qt, {kt_name})',
        "scaled = Mul(qk, Scale)",
        "attn = Softmax<axis=-1>(scaled)",
    ]
    av_inputs = f"attn, {vt_name}" if av_attn_first else f"{vt_name}, attn"
    lines += [
        f'ctx0 = Einsum<equation="{av_equation}">({av_inputs})',
        "ctx1 = Transpose<perm=[0,2,1,3]>(ctx0)",
        "ctx2 = Reshape(ctx1, OutShape)",
        "y0 = Gemm(ctx2, Wout, Bout)",
        "Y = Reshape(y0, YShape)",
    ]

    body_lines = "\n          ".join(lines)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[{batch},{seq},{K}] X{extra_inputs}) => (float[{batch},{seq},{Out}] Y)
        {{
          {body_lines}
        }}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        batch=batch,
        seq=seq,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        bq=bq,
        bk=bk,
        bv=bv,
        bout=bout,
    )


def test_cpp_decomposed_einsum_gqa_pruning_matches_python_reference_exactly():
    # Canonical equation spelling straight from `torch.onnx.export`'s own
    # confirmed literal output (see `_einsum_equation_is_batched_matmul`'s
    # own docstring): `"bhid,bhjd->bhij"` for QK^T, `"bhij,bhjd->bhid"` for
    # AV, attention-weights as the AV `Einsum` node's own first input.
    model, cfg = _decomposed_einsum_gqa_model(K=32, H=8, KVH=2, D=8, Out=16, seed=1)
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)

    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    # Fully within this C++ port's own `Einsum` scope (no mask/RoPE/Q-K-norm,
    # genuine GQA so no MQA fast-path gap either) -- both ports must agree
    # byte-for-byte, the strongest possible parity check.
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    wq_new, wk_new, wv_new, wo_new = _decomposed_weight_shapes(pruned_cpp)
    group_size = cfg["H"] // cfg["KVH"]
    new_kv = cfg["KVH"] - round(cfg["KVH"] * 0.5)
    assert wk_new.shape[1] == new_kv * cfg["D"]
    assert wq_new.shape[1] == new_kv * group_size * cfg["D"]

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], new_kv
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)
    np.testing.assert_array_equal(wq_new, cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(wk_new, cfg["wk"][:, kv_idx])
    np.testing.assert_array_equal(wv_new, cfg["wv"][:, kv_idx])
    np.testing.assert_array_equal(wo_new, cfg["wout"][q_idx, :])

    # The pruned graph must still actually run and produce the same result
    # as an independently-built, already-pruned-shape oracle model -- proof
    # the `Einsum` QK^T/AV nodes were left semantically intact (never
    # mis-sliced) by the pruning rewrite.
    oracle, _ = _decomposed_einsum_gqa_model(
        K=cfg["K"],
        H=len(keep_q_heads),
        KVH=new_kv,
        D=d,
        Out=cfg["Out"],
        seed=1,
    )
    oracle_inits = {t.name: t for t in oracle.graph.initializer}
    for name, arr in (
        ("Wq", cfg["wq"][:, q_idx]),
        ("Wk", cfg["wk"][:, kv_idx]),
        ("Wv", cfg["wv"][:, kv_idx]),
        ("Wout", cfg["wout"][q_idx, :]),
        ("Bq", cfg["bq"][q_idx]),
        ("Bk", cfg["bk"][kv_idx]),
        ("Bv", cfg["bv"][kv_idx]),
        ("Bout", cfg["bout"]),
    ):
        oracle_inits[name].CopyFrom(_f32(arr, name))

    rng = np.random.default_rng(2)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned_cpp, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_decomposed_einsum_plain_attention_pruning_alternate_equation_spelling_matches_python():
    # A second, structurally-equivalent-but-differently-spelled pair of
    # equations -- renamed axis letters (`n`/`q`/`k` rather than `b`/`i`/`j`),
    # incidental whitespace around the commas/arrow, AND the AV `Einsum`
    # node's own attention-weights operand in its SECOND input slot rather
    # than its first (`av_attn_first=False`) -- exercising
    # `EinsumEquationIsBatchedMatmul`'s own `first_operand_index=1` swap.
    # `KVH == H` here (no `repeat_kv`) -- every "group" is exactly one query
    # head wide, closing out this port's own `Einsum` coverage for the
    # no-GQA shape too.
    model, cfg = _decomposed_einsum_gqa_model(
        K=32,
        H=4,
        KVH=4,
        D=8,
        Out=16,
        seed=3,
        qk_equation="nhqd, nhkd -> nhqk",
        av_equation="nhkd,nhqk->nhqd",
        av_attn_first=False,
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    wq_new, wk_new, wv_new, _ = _decomposed_weight_shapes(pruned_cpp)
    assert wq_new.shape[1] == 2 * cfg["D"]
    assert wk_new.shape[1] == 2 * cfg["D"]
    assert wv_new.shape[1] == 2 * cfg["D"]

    rng = np.random.default_rng(4)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned_cpp, {"X": x})
    (y_unpruned,) = _run(model_for_py, {"X": x})
    assert y_pruned.shape == y_unpruned.shape


def test_cpp_decomposed_einsum_gqa_with_q_rope_pruning_matches_python_reference_exactly():
    # Combined-scenario regression test (not covered by any single one of the
    # four independent sub-shape branches this merge combines): an
    # `Einsum`-based QK^T/AV product (`EinsumEquationIsBatchedMatmul`) TOGETHER
    # with a decomposed RoPE pass-through on Q's own branch
    # (`WalkBackThroughDecomposedRope`/`DecomposedRopePassThrough`) in the
    # SAME chain. `FindDecomposedGqaChains` resolves Q's own branch via
    # `WalkBackThroughDecomposedRope` unconditionally -- regardless of
    # whether the QK^T product it feeds turns out to be `MatMul` or `Einsum`
    # -- so this combination is within this port's own matched scope (see
    # `_decomposed_einsum_gqa_model`'s own `q_rope` docstring for exactly why,
    # and structured_pruning_entry.cpp's own top-of-section comment for the
    # one combination that's still NOT recognized: RoPE/Q-K-norm on K's own
    # branch together with an `Einsum` QK^T product). Both ports must agree
    # byte-for-byte.
    model, cfg = _decomposed_einsum_gqa_model(
        K=32, H=8, KVH=2, D=8, Out=16, seed=5, q_rope=True
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    rng = np.random.default_rng(9)
    cos = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["D"])).astype(np.float32)
    sin = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["D"])).astype(np.float32)

    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    wq_new, wk_new, wv_new, wo_new = _decomposed_weight_shapes(pruned_cpp)
    group_size = cfg["H"] // cfg["KVH"]
    new_kv = cfg["KVH"] - round(cfg["KVH"] * 0.5)
    assert wk_new.shape[1] == new_kv * cfg["D"]
    assert wq_new.shape[1] == new_kv * group_size * cfg["D"]

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], new_kv
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)
    np.testing.assert_array_equal(wq_new, cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(wk_new, cfg["wk"][:, kv_idx])
    np.testing.assert_array_equal(wv_new, cfg["wv"][:, kv_idx])
    np.testing.assert_array_equal(wo_new, cfg["wout"][q_idx, :])

    # Also runs a real forward pass through the pruned model itself (feeding
    # the same `Cos`/`Sin` the byte-for-byte-checked `pruned_py` above
    # declares) -- confirms the RoPE hop's own nodes (never rewritten, only
    # marked stale post-pruning) still form a runnable graph with the
    # `Einsum`-based QK^T/AV product after Q's own head-split shrinks.
    rng2 = np.random.default_rng(10)
    x = rng2.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned_cpp, {"X": x, "Cos": cos, "Sin": sin})
    assert y_pruned.shape == (cfg["batch"], cfg["seq"], cfg["Out"])


# ---------------------------------------------------------------------------
# Decomposed (un-fused) attention -- RoPE / Q-K-norm pass-through
# ---------------------------------------------------------------------------
# `FindDecomposedGqaChains`/`ApplyOneDecomposedGqaChain` now also recognize
# the two "pass through an intermediate op between the Q/K head-split and the
# QK^T matmul" hops pruning.py's own `_find_decomposed_gqa_chains` matches:
# decomposed RoPE (`MatchDecomposedRopePassThrough`, mirroring pruning.py's
# own `_match_decomposed_rope_pass_through`/`_DecomposedRopePassThrough`) and
# decomposed per-head Q/K-norm (`MatchDecomposedQkNormPassThrough`, mirroring
# pruning.py's own `_match_decomposed_qk_norm_pass_through`/
# `_DecomposedQKNormPassThrough`) -- see structured_pruning_entry.cpp's own
# "Decomposed (un-fused) GQA/MQA/plain-MHA attention head pruning" section
# comment for the exact (still narrower-than-pruning.py) scope: both hops are
# only ever tried for Q's own branch and K's separate-`perm=[0,1,3,2]` swap
# branch, never for K's combined-`perm=[0,2,3,1]` shape or an `Einsum`-based
# QK^T product (see the dedicated `Einsum` section below -- this port DOES
# recognize `Einsum`-based QK^T/AV products, just never combined with either
# hop here), and Q/K-norm is never tried on V's own branch. Neither hop's own
# matched nodes are ever rewritten -- both are recognized purely so the
# surrounding chain still matches and prunes correctly, exactly mirroring
# pruning.py's own "passed through untouched" treatment.


def test_cpp_decomposed_gqa_rope_pruning_matches_oracle_and_python_reference_exactly():
    model, cfg = _decomposed_gqa_model(K=32, H=8, KVH=2, D=8, Out=16, seed=1, rope=True)
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    rng = np.random.default_rng(7)
    cos = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["D"])).astype(np.float32)
    sin = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["D"])).astype(np.float32)

    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    # RoPE pass-through is fully within this C++ port's own scope for this
    # shape (separate-transpose K, no mask/Einsum/packed-QKV, genuine GQA so
    # no MQA fast-path gap) -- both ports must agree byte-for-byte.
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    wq_new, wk_new, wv_new, wo_new = _decomposed_weight_shapes(pruned_cpp)
    group_size = cfg["H"] // cfg["KVH"]
    new_kv = cfg["KVH"] - round(cfg["KVH"] * 0.5)
    assert wk_new.shape[1] == new_kv * cfg["D"]
    assert wq_new.shape[1] == new_kv * group_size * cfg["D"]

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], new_kv
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)
    np.testing.assert_array_equal(wq_new, cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(wk_new, cfg["wk"][:, kv_idx])
    np.testing.assert_array_equal(wv_new, cfg["wv"][:, kv_idx])
    np.testing.assert_array_equal(wo_new, cfg["wout"][q_idx, :])

    oracle, _ = _decomposed_gqa_model(
        K=cfg["K"],
        H=len(keep_q_heads),
        KVH=new_kv,
        D=d,
        Out=cfg["Out"],
        seed=1,
        rope=True,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        bq=cfg["bq"][q_idx],
        bk=cfg["bk"][kv_idx],
        bv=cfg["bv"][kv_idx],
        bout=cfg["bout"],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )
    rng2 = np.random.default_rng(2)
    x = rng2.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    feed = {"X": x, "Cos": cos, "Sin": sin}
    (y_pruned,) = _run(pruned_cpp, feed)
    (y_oracle,) = _run(oracle, feed)
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_decomposed_gqa_qk_norm_pruning_matches_oracle_and_python_reference_exactly():
    # `qk_norm=True`, `rope=False` -- exercises the Q/K-norm pass-through in
    # isolation (KVH < H already forces K's own separate-transpose form, so
    # this doesn't rely on `qk_norm` alone to force it).
    model, cfg = _decomposed_gqa_model(
        K=32, H=8, KVH=2, D=8, Out=16, seed=1, qk_norm=True
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)

    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_attention_head_pruning(model_for_py, sparsity=0.5)
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    wq_new, wk_new, wv_new, wo_new = _decomposed_weight_shapes(pruned_cpp)
    group_size = cfg["H"] // cfg["KVH"]
    new_kv = cfg["KVH"] - round(cfg["KVH"] * 0.5)
    assert wk_new.shape[1] == new_kv * cfg["D"]
    assert wq_new.shape[1] == new_kv * group_size * cfg["D"]

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], new_kv
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)
    np.testing.assert_array_equal(wq_new, cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(wk_new, cfg["wk"][:, kv_idx])
    np.testing.assert_array_equal(wv_new, cfg["wv"][:, kv_idx])
    np.testing.assert_array_equal(wo_new, cfg["wout"][q_idx, :])

    # The Q/K-norm hop's own `weight` is never sliced -- whole-head/KV-group
    # pruning never touches `head_size`, the axis it normalizes over (see
    # `DecomposedQKNormPassThrough`'s own comment). Confirm it survives
    # byte-for-byte.
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["QNormWeight"]), cfg["q_norm_weight"]
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["KNormWeight"]), cfg["k_norm_weight"]
    )

    oracle, _ = _decomposed_gqa_model(
        K=cfg["K"],
        H=len(keep_q_heads),
        KVH=new_kv,
        D=d,
        Out=cfg["Out"],
        seed=1,
        qk_norm=True,
        q_norm_weight=cfg["q_norm_weight"],
        k_norm_weight=cfg["k_norm_weight"],
        qk_norm_eps=cfg["qk_norm_eps"],
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        bq=cfg["bq"][q_idx],
        bk=cfg["bk"][kv_idx],
        bv=cfg["bv"][kv_idx],
        bout=cfg["bout"],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )
    rng = np.random.default_rng(2)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned_cpp, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


# --- FLOAT16 / BFLOAT16 weight support ---------------------------------------
#
# `_is_supported_float_dtype` (pruning.py) accepts FLOAT/FLOAT16/BFLOAT16
# uniformly for every producer/consumer weight and bias across all matched
# attention families; this port's own matchers (MatchAttentionProducer,
# MatchProducerAnyFloat, WalkToAttentionConsumer, and every downstream
# apply-time weight read/rank/slice/write) now mirror that exactly (see each
# function's own comment in structured_pruning_entry.cpp). One FLOAT16 and
# one BFLOAT16 case per major family group below (merged-QKV `Attention`,
# separate-producer `GroupQueryAttention`, separate-producer
# `MultiHeadAttention`, and the decomposed/un-fused shape), each checked
# against the live pure-Python `apply_attention_head_pruning` reference
# byte-for-byte (`SerializeToString()` equality) -- both `onnx.checker`
# and this repo's own ONNX Runtime confirm FLOAT16 has a real CPU kernel for
# every one of these ops in this environment, and BFLOAT16 at least passes
# `onnx.checker.check_model` (its own schema's `T` type constraint includes
# it) even where no CPU kernel exists, so both are exercised the same
# structural way here -- via the graph-rewrite comparison alone, never a
# real session run (mirroring this module's own decomposed-GQA/MQA tests'
# "matches_python_reference_exactly" naming and style).


@pytest.mark.parametrize(
    "dtype,dtype_name",
    [(np.float16, "float16"), (ml_dtypes.bfloat16, "bfloat16")],
)
def test_cpp_attention_head_pruning_widened_dtype_matches_python_reference_exactly(
    dtype, dtype_name
):
    K, H, D, Out = 8, 4, 4, 6
    Nq = Nk = Nv = H * D
    rng = np.random.default_rng(301)
    wqkv = (rng.standard_normal((K, Nq + Nk + Nv)) * 0.3).astype(dtype)
    bqkv = (rng.standard_normal((Nq + Nk + Nv,)) * 0.1).astype(dtype)
    wout = (rng.standard_normal((Nv, Out)) * 0.3).astype(dtype)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        g ({dtype_name}[batch,seq,{K}] X) => ({dtype_name}[batch,seq,{Out}] Y)
        {{
          ctx = com.microsoft.Attention <num_heads={H}, qkv_hidden_sizes=[{Nq},{Nk},{Nv}]> (X, Wqkv, Bqkv)
          Y = MatMul(ctx, Wout)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(wqkv, "Wqkv"),
            onnx.numpy_helper.from_array(bqkv, "Bqkv"),
            onnx.numpy_helper.from_array(wout, "Wout"),
        ]
    )
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    onnx_dtype = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    assert inits["Wqkv"].data_type == onnx_dtype
    assert list(inits["Wqkv"].dims) == [K, 6 * D]  # H=4 -> keep_count=2 heads


@pytest.mark.parametrize(
    "dtype,dtype_name",
    [(np.float16, "float16"), (ml_dtypes.bfloat16, "bfloat16")],
)
def test_cpp_gqa_pruning_widened_dtype_matches_python_reference_exactly(
    dtype, dtype_name
):
    K, H, KVH, D, Out, batch, seq = 8, 4, 2, 8, 6, 2, 5
    Nq, Nkv = H * D, KVH * D
    rng = np.random.default_rng(302)
    wq = (rng.standard_normal((K, Nq)) * 0.3).astype(dtype)
    wk = (rng.standard_normal((K, Nkv)) * 0.3).astype(dtype)
    wv = (rng.standard_normal((K, Nkv)) * 0.3).astype(dtype)
    wout = (rng.standard_normal((Nq, Out)) * 0.3).astype(dtype)
    seqlens_k = np.full((batch,), seq - 1, dtype=np.int32)
    total_seq = np.array(seq, dtype=np.int32)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        g ({dtype_name}[{batch},{seq},{K}] X) => ({dtype_name}[{batch},{seq},{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> (q, k, v, , , SeqLensK, TotalSeq)
          Y = MatMul(ctx, Wout)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(wq, "Wq"),
            onnx.numpy_helper.from_array(wk, "Wk"),
            onnx.numpy_helper.from_array(wv, "Wv"),
            onnx.numpy_helper.from_array(wout, "Wout"),
            onnx.numpy_helper.from_array(seqlens_k, "SeqLensK"),
            onnx.numpy_helper.from_array(total_seq, "TotalSeq"),
        ]
    )
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    onnx_dtype = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    assert inits["Wq"].data_type == onnx_dtype
    assert inits["Wk"].data_type == onnx_dtype
    assert list(inits["Wk"].dims) == [K, D]  # KVH=2 -> keep_count=1 KV group


@pytest.mark.parametrize(
    "dtype,dtype_name",
    [(np.float16, "float16"), (ml_dtypes.bfloat16, "bfloat16")],
)
def test_cpp_mha_pruning_widened_dtype_matches_python_reference_exactly(
    dtype, dtype_name
):
    K, H, D, Out, batch, seq = 8, 8, 4, 6, 2, 5
    Nq = Nk = Nv = H * D
    rng = np.random.default_rng(303)
    wq = (rng.standard_normal((K, Nq)) * 0.3).astype(dtype)
    wk = (rng.standard_normal((K, Nk)) * 0.3).astype(dtype)
    wv = (rng.standard_normal((K, Nv)) * 0.3).astype(dtype)
    wout = (rng.standard_normal((Nv, Out)) * 0.3).astype(dtype)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        g ({dtype_name}[{batch},{seq},{K}] X) => ({dtype_name}[{batch},{seq},{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx = com.microsoft.MultiHeadAttention <num_heads={H}> (q, k, v)
          Y = MatMul(ctx, Wout)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(wq, "Wq"),
            onnx.numpy_helper.from_array(wk, "Wk"),
            onnx.numpy_helper.from_array(wv, "Wv"),
            onnx.numpy_helper.from_array(wout, "Wout"),
        ]
    )
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    onnx_dtype = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    assert inits["Wq"].data_type == onnx_dtype
    assert list(inits["Wq"].dims) == [K, 4 * D]  # H=8 -> keep_count=4 heads


def _decomposed_gqa_model_dtype(dtype, dtype_name, seed):
    """Trimmed, dtype-parametrized rebuild of `_decomposed_gqa_model`'s own
    "plain" (bias=True, needs_repeat_kv, no mask/rope/qk_norm) shape --
    covers this port's own downstream-arithmetic FLOAT16/BFLOAT16 widening
    for `ApplyOneDecomposedGqaChain` (`ReadTensorAsF64`/`WriteF64TensorAs`/
    `SliceAxisGeneric`/`TransposeFlat<double>` throughout).
    """
    K, H, KVH, D, Out, batch, seq = 32, 4, 2, 8, 16, 1, 4
    Nq, Nk, Nv = H * D, KVH * D, KVH * D
    rng = np.random.default_rng(seed)
    wq = (rng.standard_normal((K, Nq)) * 0.3).astype(dtype)
    wk = (rng.standard_normal((K, Nk)) * 0.3).astype(dtype)
    wv = (rng.standard_normal((K, Nv)) * 0.3).astype(dtype)
    wout = (rng.standard_normal((H * D, Out)) * 0.3).astype(dtype)
    bq = (rng.standard_normal((Nq,)) * 0.1).astype(dtype)
    bk = (rng.standard_normal((Nk,)) * 0.1).astype(dtype)
    bv = (rng.standard_normal((Nv,)) * 0.1).astype(dtype)
    bout = (rng.standard_normal((Out,)) * 0.1).astype(dtype)
    scale = np.array(D**-0.5).astype(dtype)
    n_rep = H // KVH

    def _i64(arr, name):
        return onnx.numpy_helper.from_array(np.array(arr, dtype=np.int64), name)

    initializer = [
        onnx.numpy_helper.from_array(wq, "Wq"),
        onnx.numpy_helper.from_array(wk, "Wk"),
        onnx.numpy_helper.from_array(wv, "Wv"),
        onnx.numpy_helper.from_array(wout, "Wout"),
        onnx.numpy_helper.from_array(bq, "Bq"),
        onnx.numpy_helper.from_array(bk, "Bk"),
        onnx.numpy_helper.from_array(bv, "Bv"),
        onnx.numpy_helper.from_array(bout, "Bout"),
        _i64([batch * seq, K], "XFlatShape"),
        _i64([batch, seq, H, D], "Sq"),
        _i64([batch, seq, KVH, D], "Sk"),
        _i64([batch, seq, KVH, D], "Sv"),
        _i64([2], "Ax2"),
        _i64([batch, KVH, n_rep, seq, D], "KExpandShape"),
        _i64([batch, H, seq, D], "KMergeShape"),
        _i64([batch, KVH, n_rep, seq, D], "VExpandShape"),
        _i64([batch, H, seq, D], "VMergeShape"),
        onnx.numpy_helper.from_array(scale, "Scale"),
        _i64([batch * seq, H * D], "OutShape"),
        _i64([batch, seq, Out], "YShape"),
    ]
    body = f"""
        g ({dtype_name}[{batch},{seq},{K}] X) => ({dtype_name}[{batch},{seq},{Out}] Y)
        {{
          xf = Reshape(X, XFlatShape)
          q0 = Gemm(xf, Wq, Bq)
          qr = Reshape(q0, Sq)
          qt = Transpose<perm=[0,2,1,3]>(qr)
          k0 = Gemm(xf, Wk, Bk)
          kr = Reshape(k0, Sk)
          kt0 = Transpose<perm=[0,2,1,3]>(kr)
          ku = Unsqueeze(kt0, Ax2)
          ke = Expand(ku, KExpandShape)
          kre = Reshape(ke, KMergeShape)
          kt = Transpose<perm=[0,1,3,2]>(kre)
          v0 = Gemm(xf, Wv, Bv)
          vr = Reshape(v0, Sv)
          vt0 = Transpose<perm=[0,2,1,3]>(vr)
          vu = Unsqueeze(vt0, Ax2)
          ve = Expand(vu, VExpandShape)
          vt = Reshape(ve, VMergeShape)
          qk = MatMul(qt, kt)
          scaled = Mul(qk, Scale)
          attn = Softmax<axis=-1>(scaled)
          ctx0 = MatMul(attn, vt)
          ctx1 = Transpose<perm=[0,2,1,3]>(ctx0)
          ctx2 = Reshape(ctx1, OutShape)
          y0 = Gemm(ctx2, Wout, Bout)
          Y = Reshape(y0, YShape)
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(K=K, H=H, KVH=KVH, D=D, wq=wq, wk=wk, wv=wv)


@pytest.mark.parametrize(
    "dtype,dtype_name",
    [(np.float16, "float16"), (ml_dtypes.bfloat16, "bfloat16")],
)
def test_cpp_decomposed_gqa_pruning_widened_dtype_matches_python_reference_exactly(
    dtype, dtype_name
):
    model, cfg = _decomposed_gqa_model_dtype(dtype, dtype_name, seed=304)
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    onnx_dtype = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    assert inits["Wq"].data_type == onnx_dtype
    assert inits["Wk"].data_type == onnx_dtype
    # KVH=2 -> keep_count=1 KV group -> Wk/Wv shrink to a single D-wide head;
    # Wq shrinks to that group's own 2 (of 4) query heads.
    assert list(inits["Wk"].dims) == [cfg["K"], cfg["D"]]
    assert list(inits["Wq"].dims) == [cfg["K"], 2 * cfg["D"]]
    # Value-preserving slice -- the surviving KV group's own column block
    # (whichever of the two KVH=2 groups importance ranking kept) must
    # reproduce the exact original fp16/bf16 bit pattern, not a re-rounded
    # one -- checked against both candidate groups since which one survives
    # is a ranking outcome, not fixed by construction.
    d = cfg["D"]
    kept_wk = onnx.numpy_helper.to_array(inits["Wk"]).view(np.uint16)
    candidates = [
        cfg["wk"][:, :d].view(np.uint16),
        cfg["wk"][:, d : 2 * d].view(np.uint16),
    ]
    assert any(np.array_equal(kept_wk, c) for c in candidates)


# --- com.microsoft::DecoderMaskedSelfAttention / PackedAttention producer
# --- recognition (MatchAttentionProducer's own three-op-type scope) --------
#
# MatchAttentionProducer (structured_pruning_entry.cpp) now recognizes
# `com.microsoft::DecoderMaskedSelfAttention` and `com.microsoft::
# PackedAttention` as the same merged-QKV family plain `Attention` already
# was -- mirroring pruning.py's own `_match_attention_producer`, including
# `DecoderMaskedSelfAttention`'s own schema quirks (no `qkv_hidden_sizes`
# attribute -- confirmed below -- and a required `past` input, left
# untouched here since it's never a constant in this model). Both compared
# against the live pure-Python reference byte-for-byte, same as every other
# family above.


def test_cpp_decoder_masked_self_attention_pruning_matches_python_reference_exactly():
    K, H, D, Out, batch = 8, 4, 4, 6, 2
    Nq = Nk = Nv = H * D
    rng = np.random.default_rng(305)
    wqkv = (rng.standard_normal((K, Nq + Nk + Nv)) * 0.3).astype(np.float32)
    bqkv = (rng.standard_normal((Nq + Nk + Nv,)) * 0.1).astype(np.float32)
    wout = (rng.standard_normal((Nv, Out)) * 0.3).astype(np.float32)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        g (float[{batch},1,{K}] X) => (float[{batch},1,{Out}] Y)
        {{
          ctx = com.microsoft.DecoderMaskedSelfAttention <num_heads={H}> (X, Wqkv, Bqkv)
          Y = MatMul(ctx, Wout)
        }}
        """
    )
    model.graph.initializer.extend(
        [_f32(wqkv, "Wqkv"), _f32(bqkv, "Bqkv"), _f32(wout, "Wout")]
    )
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = next(
        n for n in pruned_cpp.graph.node if n.op_type == "DecoderMaskedSelfAttention"
    )
    assert next(a.i for a in node.attribute if a.name == "num_heads") == 2
    # No real `qkv_hidden_sizes` attribute on this op's own schema -- never
    # added, mirroring pruning.py's own `_apply_one_plain_attention_chain`.
    assert not any(a.name == "qkv_hidden_sizes" for a in node.attribute)
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    assert list(inits["Wqkv"].dims) == [K, 6 * D]


def test_cpp_packed_attention_pruning_matches_python_reference_exactly():
    K, H, D, Out, batch = 8, 4, 4, 6, 2
    Nq = Nk = Nv = H * D
    rng = np.random.default_rng(306)
    wqkv = (rng.standard_normal((K, Nq + Nk + Nv)) * 0.3).astype(np.float32)
    bqkv = (rng.standard_normal((Nq + Nk + Nv,)) * 0.1).astype(np.float32)
    wout = (rng.standard_normal((Nv, Out)) * 0.3).astype(np.float32)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        g (float[{batch * 3},{K}] X, int32[{batch},3] TokenOffset, int32[{batch + 1}] CumSeqLen) => (float[{batch * 3},{Out}] Y)
        {{
          ctx = com.microsoft.PackedAttention <num_heads={H}, qkv_hidden_sizes=[{Nq},{Nk},{Nv}]> (X, Wqkv, Bqkv, TokenOffset, CumSeqLen)
          Y = MatMul(ctx, Wout)
        }}
        """
    )
    model.graph.initializer.extend(
        [_f32(wqkv, "Wqkv"), _f32(bqkv, "Bqkv"), _f32(wout, "Wout")]
    )
    onnx.checker.check_model(model)

    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    node = next(n for n in pruned_cpp.graph.node if n.op_type == "PackedAttention")
    assert next(a.i for a in node.attribute if a.name == "num_heads") == 2
    qkv = next(list(a.ints) for a in node.attribute if a.name == "qkv_hidden_sizes")
    assert qkv == [2 * D, 2 * D, 2 * D]
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    assert list(inits["Wqkv"].dims) == [K, 6 * D]


# --- Gap C: `com.microsoft::MatMulNBitsQkv` scope -- `ApplyAttentionHead
# --- Pruning` vs. `ApplyAttentionHeadWandaPruning` ---------------------------
#
# Investigated as part of this same round of parity fixes: unlike a
# hypothetical accidental scope-creep, `ApplyAttentionHeadPruning`'s own
# `FindMatMulNBitsQkvChains`/`ApplyMatMulNBitsQkvChains` call is a
# DELIBERATE, documented consolidation -- see structured_pruning_entry.cpp's
# own comment directly above that call site, and this file's own "com.
# microsoft::MatMulNBitsQkv" section comment above `_matmul_nbits_qkv_model`
# (which pruning.py's own `apply_structured_pruning_matmul_nbits` explicitly
# defers to `apply_attention_head_pruning_cpp` for -- see
# `onnxsim.apply_structured_pruning_cpp`'s own docstring). Removing it would
# leave `MatMulNBitsQkv` with NO C++ pruning path anywhere (pure-Python's own
# `apply_attention_head_pruning` never touches it either -- that's
# `apply_structured_pruning_matmul_nbits`'s own job there), a real regression
# against this module's own already-extensive `test_cpp_matmul_nbits_qkv_
# pruning_*` coverage above -- so it is intentionally KEPT here, unlike
# `ApplyAttentionHeadWandaPruning`, which correctly has no such call at all
# (mirroring pruning.py's own scope: no calibration-driven counterpart of
# `apply_structured_pruning_matmul_nbits` exists there either). This test
# pins that asymmetry: the plain entry point prunes a `MatMulNBitsQkv` chain
# (already proven above), the Wanda entry point leaves the identical chain
# completely untouched.


def test_cpp_attention_head_wanda_pruning_leaves_matmul_nbits_qkv_chains_untouched():
    num_heads, kv_num_heads, d, K, block_size, N2 = 4, 2, 8, 32, 16, 24
    rng = np.random.default_rng(307)
    Nq, Nkv = num_heads * d, kv_num_heads * d
    w_q = (rng.standard_normal((Nq, K)) * 0.3).astype(np.float32)
    w_k = (rng.standard_normal((Nkv, K)) * 0.3).astype(np.float32)
    w_v = (rng.standard_normal((Nkv, K)) * 0.3).astype(np.float32)
    bias_q = (rng.standard_normal((Nq,)) * 0.1).astype(np.float32)
    bias_k = (rng.standard_normal((Nkv,)) * 0.1).astype(np.float32)
    bias_v = (rng.standard_normal((Nkv,)) * 0.1).astype(np.float32)
    norm_scale = np.ones((K,), dtype=np.float32)
    model, _info = _matmul_nbits_qkv_model(
        num_heads,
        kv_num_heads,
        d,
        K,
        block_size,
        N2,
        w_q,
        w_k,
        w_v,
        bias_q,
        bias_k,
        bias_v,
        norm_scale,
    )
    onnx.checker.check_model(model)

    calibration_data = [
        {"A": rng.standard_normal((2, 5, K)).astype(np.float32)} for _ in range(2)
    ]
    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    assert pruned.SerializeToString() == model.SerializeToString()


# --- Cross-family regression checks --------------------------------------
#
# The three per-head-input fixes above (GroupQueryAttention/plain
# ai.onnx::Attention, MultiHeadAttention/PackedMultiHeadAttention,
# DecoderMaskedMultiHeadAttention/PagedAttention) were each authored
# independently against the SAME shared dispatch machinery
# (``FindSeparateQkvChains``, ``ApplyOneGqaChain``, ``HeadBiasInputIsSafe``/
# ``SliceOrGatherHeadBias``, ``PastKvConstantsAreSliceable``) and merged
# together afterwards. The two tests below are not about any one family's
# own new behavior (already covered above) -- they specifically probe the
# merge-time risk of one family's own per-head-input index/handling
# accidentally leaking onto a DIFFERENT family's node when both are matched
# and pruned together in the same graph: a wrong hard-coded input index
# reused across op types, a `keep_heads`/`keep_q_heads` index set computed
# for one chain applied to another's own attention_bias/past_key/past_value,
# or a `used_names` collision between two independently-inserted dynamic
# Gather nodes.


def test_cpp_attention_head_pruning_gqa_and_mha_attention_bias_do_not_cross_contaminate():
    # One graph, two independent chains: a GroupQueryAttention node
    # (num_heads=4, kv_num_heads=2) and a MultiHeadAttention node
    # (num_heads=4), each with its OWN dynamic (graph-input) attention_bias.
    # head_size=8 (onnxruntime's own GroupQueryAttention kernel requires a
    # multiple of 8) with every column WITHIN a head sharing that head's own
    # single value keeps each chain's own kept-head set fully deterministic
    # and, by construction, DISJOINT from the other's -- so if the merged
    # code ever applied one chain's own `keep_q_heads`/`keep_heads` index set
    # (or wrong input index) to the other's node, the assertions below on the
    # inserted Gathers' own indices content would catch it immediately.
    seq = 3
    batch = 1
    d = 8

    def _per_head(values):
        return np.repeat(np.asarray(values, dtype=np.float32), d).reshape(1, -1)

    # GQA branch: group 1 (query heads {2, 3}) is the more important KV
    # group by construction (larger Wk/Wv magnitude), so sparsity=0.5 (1 of
    # 2 KV groups dropped) keeps exactly heads {2, 3}.
    wq_g = _per_head([1.0, 2.0, 3.0, 4.0])
    wk_g = _per_head([10.0, 20.0])
    wv_g = _per_head([100.0, 200.0])
    wout_g = np.eye(4 * d, dtype=np.float32)

    # MHA branch: heads are ranked in the OPPOSITE order (head 0 most
    # important), so the same sparsity=0.5 keeps exactly heads {0, 1} --
    # disjoint from the GQA branch's own kept {2, 3}.
    wq_m = _per_head([40.0, 30.0, 20.0, 10.0])
    wk_m = np.ones((1, 4 * d), dtype=np.float32)
    wv_m = np.ones((1, 4 * d), dtype=np.float32)
    wout_m = np.eye(4 * d, dtype=np.float32)

    seqlensk = np.full((batch,), seq - 1, dtype=np.int32)
    totalseq = np.array(seq, dtype=np.int32)

    body = f"""
        g (float[{batch},{seq},1] X,
           float[1,4,{seq},{seq}] AttnBiasGqaIn,
           float[1,4,{seq},{seq}] AttnBiasMhaIn)
          => (float[{batch},{seq},{4 * d}] Yg, float[{batch},{seq},{4 * d}] Ym)
        {{
          qg = MatMul(X, WqG)
          kg = MatMul(X, WkG)
          vg = MatMul(X, WvG)
          ctxg, pkg, pvg = com.microsoft.GroupQueryAttention <num_heads=4, kv_num_heads=2> (qg, kg, vg, , , SeqLensK, TotalSeq, , , , AttnBiasGqaIn)
          Yg = MatMul(ctxg, WoutG)
          qm = MatMul(X, WqM)
          km = MatMul(X, WkM)
          vm = MatMul(X, WvM)
          ctxm = com.microsoft.MultiHeadAttention <num_heads=4> (qm, km, vm, , , AttnBiasMhaIn)
          Ym = MatMul(ctxm, WoutM)
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
    model.graph.initializer.extend(
        [
            _f32(wq_g, "WqG"),
            _f32(wk_g, "WkG"),
            _f32(wv_g, "WvG"),
            _f32(wout_g, "WoutG"),
            _f32(wq_m, "WqM"),
            _f32(wk_m, "WkM"),
            _f32(wv_m, "WvM"),
            _f32(wout_m, "WoutM"),
            onnx.numpy_helper.from_array(seqlensk, "SeqLensK"),
            onnx.numpy_helper.from_array(totalseq, "TotalSeq"),
        ]
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    gqa_node = next(n for n in pruned.graph.node if n.op_type == "GroupQueryAttention")
    mha_node = next(n for n in pruned.graph.node if n.op_type == "MultiHeadAttention")
    assert next(a.i for a in gqa_node.attribute if a.name == "num_heads") == 2
    assert next(a.i for a in gqa_node.attribute if a.name == "kv_num_heads") == 1
    assert next(a.i for a in mha_node.attribute if a.name == "num_heads") == 2

    # Each node's own attention_bias input now names a freshly-inserted
    # Gather's own output -- distinct nodes reading from each's own original
    # dynamic input, never each other's.
    gqa_bias_name = gqa_node.input[10]
    mha_bias_name = mha_node.input[5]
    assert gqa_bias_name != mha_bias_name

    gather_by_output = {
        n.output[0]: n for n in pruned.graph.node if n.op_type == "Gather"
    }
    init_by_name = {t.name: t for t in pruned.graph.initializer}
    gqa_gather = gather_by_output[gqa_bias_name]
    mha_gather = gather_by_output[mha_bias_name]
    assert gqa_gather.name != mha_gather.name
    assert gqa_gather.input[0] == "AttnBiasGqaIn"
    assert mha_gather.input[0] == "AttnBiasMhaIn"

    gqa_indices = list(onnx.numpy_helper.to_array(init_by_name[gqa_gather.input[1]]))
    mha_indices = list(onnx.numpy_helper.to_array(init_by_name[mha_gather.input[1]]))
    assert gqa_indices == [2, 3]
    assert mha_indices == [0, 1]

    # No name collision between the two independently-inserted Gathers/
    # indices initializers -- a shared, incorrectly-threaded `used_names`
    # set across the two chains would otherwise silently produce two
    # differently-behaving nodes with the same name.
    node_names = [n.name for n in pruned.graph.node if n.name]
    assert len(node_names) == len(set(node_names))
    init_names = [t.name for t in pruned.graph.initializer]
    assert len(init_names) == len(set(init_names))

    # Functional smoke test: the rewritten graph is still valid, executable
    # ONNX -- each branch's own Gather correctly feeds its own node.
    rng = np.random.default_rng(0)
    feeds = {
        "X": rng.standard_normal((batch, seq, 1)).astype(np.float32),
        "AttnBiasGqaIn": rng.standard_normal((1, 4, seq, seq)).astype(np.float32),
        "AttnBiasMhaIn": rng.standard_normal((1, 4, seq, seq)).astype(np.float32),
    }
    outs = _run(pruned, feeds)
    assert all(np.isfinite(o).all() for o in outs)


def test_cpp_attention_head_pruning_dmmha_and_mha_constant_inputs_do_not_cross_contaminate():
    # A DecoderMaskedMultiHeadAttention node (attention_bias at index 4,
    # past_key/past_value at 5/6) and a MultiHeadAttention node
    # (attention_bias at index 5, past_key/past_value at 6/7) side by side in
    # one graph -- deliberately overlapping-but-different index conventions,
    # the exact shape of bug a copy-paste merge error between the two
    # families' own matchers/appliers would produce. Every per-head constant
    # here is filled with the head index scaled by a distinct base, so a
    # sliced tensor's own surviving values pin exactly which head indices
    # each op's own inputs were sliced by.
    K = 1
    Hd, Hm = 3, 4
    total_seq = 2
    past_seq = 2

    # DecoderMaskedMultiHeadAttention: head 0 most important, so
    # sparsity=1/3 (round(3/3)=1 KV/query head dropped) keeps heads {0, 1}.
    wq_d = np.array([[100.0, 10.0, 1.0]], dtype=np.float32)
    wk_d = np.ones((1, Hd), dtype=np.float32)
    wv_d = np.ones((1, Hd), dtype=np.float32)
    wout_d = np.eye(Hd, dtype=np.float32)
    attn_bias_d = np.array(
        [[[[h * 10.0] * total_seq] for h in range(Hd)]], dtype=np.float32
    )
    past_key_d = np.array(
        [[[[h * 1000.0] * 1] * past_seq for h in range(Hd)]], dtype=np.float32
    )
    past_value_d = np.array(
        [[[[h * 1000.0 + 1] * 1] * past_seq for h in range(Hd)]], dtype=np.float32
    )

    # MultiHeadAttention: head 3 most important, so sparsity=1/3
    # (round(4/3)=1 head dropped) keeps heads {1, 2, 3}.
    wq_m = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    wk_m = np.ones((1, Hm), dtype=np.float32)
    wv_m = np.ones((1, Hm), dtype=np.float32)
    wout_m = np.eye(Hm, dtype=np.float32)
    attn_bias_m = np.array(
        [[[[h * 100.0] * total_seq] * total_seq for h in range(Hm)]], dtype=np.float32
    )
    past_key_m = np.array(
        [[[[h * 10000.0] * 1] * past_seq for h in range(Hm)]], dtype=np.float32
    )
    past_value_m = np.array(
        [[[[h * 10000.0 + 1] * 1] * past_seq for h in range(Hm)]], dtype=np.float32
    )

    body = f"""
        g (float[1,1,{K}] X) => (float[1,1,{Hd}] Yd, float[1,1,{Hm}] Ym)
        {{
          qd = MatMul(X, WqD)
          kd = MatMul(X, WkD)
          vd = MatMul(X, WvD)
          ctxd = com.microsoft.DecoderMaskedMultiHeadAttention <num_heads={Hd}> (qd, kd, vd, , AttnBiasD, PastKeyD, PastValueD)
          Yd = MatMul(ctxd, WoutD)
          qm = MatMul(X, WqM)
          km = MatMul(X, WkM)
          vm = MatMul(X, WvM)
          ctxm = com.microsoft.MultiHeadAttention <num_heads={Hm}> (qm, km, vm, , , AttnBiasM, PastKeyM, PastValueM)
          Ym = MatMul(ctxm, WoutM)
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
    model.graph.initializer.extend(
        [
            _f32(wq_d, "WqD"),
            _f32(wk_d, "WkD"),
            _f32(wv_d, "WvD"),
            _f32(wout_d, "WoutD"),
            _f32(attn_bias_d, "AttnBiasD"),
            _f32(past_key_d, "PastKeyD"),
            _f32(past_value_d, "PastValueD"),
            _f32(wq_m, "WqM"),
            _f32(wk_m, "WkM"),
            _f32(wv_m, "WvM"),
            _f32(wout_m, "WoutM"),
            _f32(attn_bias_m, "AttnBiasM"),
            _f32(past_key_m, "PastKeyM"),
            _f32(past_value_m, "PastValueM"),
        ]
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=1 / 3)
    onnx.checker.check_model(pruned)

    dmmha_node = next(
        n for n in pruned.graph.node if n.op_type == "DecoderMaskedMultiHeadAttention"
    )
    mha_node = next(n for n in pruned.graph.node if n.op_type == "MultiHeadAttention")
    assert next(a.i for a in dmmha_node.attribute if a.name == "num_heads") == 2
    assert next(a.i for a in mha_node.attribute if a.name == "num_heads") == 3

    init_by_name = {t.name: t for t in pruned.graph.initializer}

    # DecoderMaskedMultiHeadAttention's own attention_bias/past_key/past_value
    # (indices 4/5/6) are sliced to its OWN kept heads {0, 1} -- never
    # touched by MultiHeadAttention's own {1, 2, 3}.
    pruned_bias_d = onnx.numpy_helper.to_array(init_by_name[dmmha_node.input[4]])
    assert list(pruned_bias_d[0, :, 0, 0]) == [0.0, 10.0]
    pruned_past_key_d = onnx.numpy_helper.to_array(init_by_name[dmmha_node.input[5]])
    assert list(pruned_past_key_d[0, :, 0, 0]) == [0.0, 1000.0]

    # MultiHeadAttention's own attention_bias/past_key/past_value (indices
    # 5/6/7) are sliced to ITS OWN kept heads {1, 2, 3} -- never touched by
    # DecoderMaskedMultiHeadAttention's own {0, 1}.
    pruned_bias_m = onnx.numpy_helper.to_array(init_by_name[mha_node.input[5]])
    assert list(pruned_bias_m[0, :, 0, 0]) == [100.0, 200.0, 300.0]
    pruned_past_key_m = onnx.numpy_helper.to_array(init_by_name[mha_node.input[6]])
    assert list(pruned_past_key_m[0, :, 0, 0]) == [10000.0, 20000.0, 30000.0]
