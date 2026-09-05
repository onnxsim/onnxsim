"""Tests for ``onnxsim.apply_attention_head_wanda_pruning_cpp`` -- the
C++-backed port of ``onnxsim.apply_attention_head_wanda_pruning`` (see
``onnxsim/structured_pruning_entry.cpp``'s ``ApplyAttentionHeadWandaPruning``
and its own "Wanda calibration"/"Attention-head pruning" section comments).
The calibrated upgrade of ``onnxsim.apply_attention_head_pruning_cpp``,
exactly as ``onnxsim.apply_structured_wanda_pruning_cpp`` is to
``onnxsim.apply_structured_pruning_cpp`` -- runs the model over real
calibration data through a real ``onnxruntime``-backed
:class:`onnxsim.onnx_simplifier.PyModelExecutor` (via
``onnxsim.onnx_simplifier._get_model_executor``, the same executor
:func:`onnxsim.simplify` itself uses) to capture per-channel activation
norms right where each matched block's output projection reads them --
never a fake/mock executor.

Same chain-finding scope as ``onnxsim.apply_attention_head_pruning_cpp``
(plain ``com.microsoft::Attention``, ``com.microsoft::GroupQueryAttention``,
the plain ``ai.onnx::Attention`` op, ``com.microsoft::MultiHeadAttention``,
``com.microsoft::PackedMultiHeadAttention``,
``com.microsoft::DecoderMaskedMultiHeadAttention``,
``com.microsoft::PagedAttention``, the plain ``ai.onnx::LinearAttention`` op,
``com.microsoft::SparseAttention``, and the tenth, decomposed (un-fused,
"eager SDPA export") shape -- ``FindDecomposedGqaChains``/
``ApplyOneDecomposedGqaChain``, threaded through with a real calibrated
``act_norm`` map here exactly like every other family), minus the fused
``com.microsoft::MatMulNBitsQkv`` variant that port also matches -- this
Wanda port has no quantized-weight counterpart, mirroring the pure-Python
``onnxsim.apply_attention_head_wanda_pruning`` exactly (see
``structured_pruning_entry.h``'s own ``ApplyAttentionHeadWandaPruning``
declaration comment). See ``test_attention_head_pruning_cpp.py``'s own
docstring for this port's shared narrower-than-pruning.py scope decisions
across the six newer fused-op families (no dynamic-attention-bias-Gather-
insertion machinery) and across the decomposed-GQA family (no mask/RoPE/
Q-K-norm/Einsum/packed-QKV, no true-MQA fast path) -- every one of those
narrowings means ``apply_attention_head_wanda_pruning`` is NOT yet aliased
to this port either -- EXCEPT the additive-mask branch of the decomposed-GQA
family, which this port's ``ApplyAttentionHeadWandaPruning`` now shares with
``ApplyAttentionHeadPruning`` (both dispatch through the same
``FindDecomposedGqaChains``/``ApplyOneDecomposedGqaChain``/
``HeadBiasInputIsSafe``/``SliceOrGatherHeadBias`` machinery): a constant or
genuinely dynamic additive mask before ``Softmax`` is matched and pruned
here too, exactly mirroring pruning.py's own
``_head_bias_input_is_safe``/``_slice_or_gather_head_bias`` for this one
chain family.
"""

import os

import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _head_idx(keep_heads, d):
    return np.concatenate([np.arange(h * d, (h + 1) * d) for h in keep_heads])


def _group_q_heads(keep_groups, group_size):
    return np.concatenate(
        [np.arange(g * group_size, (g + 1) * group_size) for g in keep_groups]
    )


def _probe_act_norm(model, probe_name, feeds):
    # Mirrors bias_correction.py's own `_add_probe_outputs`/pruning.py's own
    # `_wanda_attention_calibration_stats`: expose `probe_name` as an extra
    # graph output, run the (unmodified otherwise) graph, and reduce over
    # every axis but the last (channel) one -- the same reduction
    # WandaCalibrationStats performs in structured_pruning_entry.cpp.
    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(onnx.ValueInfoProto(name=probe_name))
    _, act = _run(probe_model, feeds)
    act = np.asarray(act, dtype=np.float64)
    reduce_axes = tuple(range(act.ndim - 1))
    return np.sqrt(np.mean(np.square(act), axis=reduce_axes))


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
        g (float[batch,seq,{K}] X) => (float[batch,seq,{Out}] Y)
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


def _plain_attention_importance(wqkv, nq, nk, nv, num_heads):
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
    return importance


def _wanda_attention_keep_heads(wqkv, nq, nk, nv, num_heads, act_norm, keep_count):
    dv = nv // num_heads
    base = _plain_attention_importance(wqkv, nq, nk, nv, num_heads)
    act_head = np.array(
        [np.linalg.norm(act_norm[h * dv : (h + 1) * dv]) for h in range(num_heads)]
    )
    importance = base * np.maximum(act_head, 1e-8)
    return np.sort(np.argsort(-importance)[:keep_count])


def test_cpp_attention_head_wanda_pruning_matches_oracle_and_differs_from_plain():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=8)

    rng = np.random.default_rng(9)
    x_cal = rng.standard_normal((3, 6, cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    act_norm = _probe_act_norm(model, "ctx", {"X": x_cal})
    keep = _wanda_attention_keep_heads(
        cfg["wqkv"], cfg["Nq"], cfg["Nk"], cfg["Nv"], cfg["H"], act_norm, 2
    )
    plain_importance = _plain_attention_importance(
        cfg["wqkv"], cfg["Nq"], cfg["Nk"], cfg["Nv"], cfg["H"]
    )
    plain_keep = np.sort(np.argsort(-plain_importance)[:2])
    assert not np.array_equal(keep, plain_keep)  # calibration actually matters here

    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    node = _attention_node(pruned)
    num_heads, _ = _attention_attrs(node)
    assert num_heads == 2

    d = cfg["D"]
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

    # Confirm plain (magnitude-only) pruning on the same weights would have
    # kept a genuinely different -- and, against this calibration signal,
    # worse -- head set.
    plain_pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    (y_plain,) = _run(plain_pruned, {"X": x})
    assert not np.allclose(y_plain, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_attention_head_wanda_pruning_reshape_hop_matches_oracle():
    model, cfg = _attention_model(
        K=8, H=4, D=4, Out=6, seed=12, with_reshape=True, batch=2, seq=5
    )
    rng = np.random.default_rng(13)
    x_cal = rng.standard_normal((2, 5, cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    # The probed activation is the output projection's OWN input -- here
    # that's the Reshape's output ("ctx2"), not the Attention node's raw
    # output ("ctx") -- exercises `chain.consumer_node->input(0)` resolving
    # past the Reshape hop exactly as pruning.py's own
    # `chain.consumer_node.input[0]` does.
    act_norm = _probe_act_norm(model, "ctx2", {"X": x_cal})
    keep = _wanda_attention_keep_heads(
        cfg["wqkv"], cfg["Nq"], cfg["Nk"], cfg["Nv"], cfg["H"], act_norm, 2
    )

    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node] == ["Attention", "Reshape", "MatMul"]

    d = cfg["D"]
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
        seed=12,
        wqkv=cfg["wqkv"][:, all_idx],
        bqkv=cfg["bqkv"][all_idx],
        wout=cfg["wout"][_head_idx(keep, d), :],
        num_heads=2,
        with_reshape=True,
        batch=2,
        seq=5,
    )

    x = rng.standard_normal((2, 5, cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_attention_head_wanda_pruning_empty_calibration_data_matches_plain():
    # An empty (but present) calibration_data means no activation was ever
    # observed for any probe point, so every matched block falls back to
    # apply_attention_head_pruning_cpp's own plain ||W||_F ranking -- exactly
    # byte-identical output.
    model, _ = _attention_model(K=8, H=4, D=4, Out=6, seed=10)
    wanda_empty = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    plain = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    assert wanda_empty.SerializeToString() == plain.SerializeToString()


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
    wq=None,
    wk=None,
    wv=None,
    bq=None,
    bk=None,
    bv=None,
    wout=None,
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

    operands = ["q", "k", "v", "", "", "SeqLensK", "TotalSeq"]

    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          q = {q_op}
          k = {k_op}
          v = {v_op}
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> ({", ".join(operands)})
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
        batch=batch,
        seq=seq,
    )


def _gqa_node(model):
    return next(n for n in model.graph.node if n.op_type == "GroupQueryAttention")


def _gqa_attrs(node):
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return num_heads, kv_num_heads


def _plain_gqa_importance(wq, wk, wv, num_heads, kv_num_heads, head_size):
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
    return importance


def _wanda_gqa_keep_groups(
    wq, wk, wv, num_heads, kv_num_heads, head_size, act_norm, keep_count
):
    group_size = num_heads // kv_num_heads
    base = _plain_gqa_importance(wq, wk, wv, num_heads, kv_num_heads, head_size)
    act_group = np.array(
        [
            np.linalg.norm(
                act_norm[
                    kv * group_size * head_size : (kv + 1) * group_size * head_size
                ]
            )
            for kv in range(kv_num_heads)
        ]
    )
    importance = base * np.maximum(act_group, 1e-8)
    return np.sort(np.argsort(-importance)[:keep_count])


def test_cpp_gqa_wanda_pruning_matches_oracle_exactly():
    # Calibration and eval data must share the model's own fixed batch/seq
    # (seqlens_k/total_sequence_length are baked-in constants tied to a
    # specific batch/seq -- see _gqa_model -- a real GroupQueryAttention
    # KV-cache-bookkeeping constraint, not a limitation of this pass).
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=8)

    rng = np.random.default_rng(9)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    act_norm = _probe_act_norm(model, "ctx", {"X": x_cal})
    keep_groups = _wanda_gqa_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], act_norm, 1
    )
    group_size = cfg["H"] // cfg["KVH"]
    keep_q_heads = _group_q_heads(keep_groups, group_size)

    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == len(keep_groups)
    assert num_heads == len(keep_q_heads)

    d = cfg["D"]
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


def test_cpp_gqa_wanda_pruning_empty_calibration_data_matches_plain():
    model, _ = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=10)
    wanda_empty = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    plain = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    assert wanda_empty.SerializeToString() == plain.SerializeToString()


# --- Cross-check against the pure-Python reference --------------------------


def test_cpp_attention_head_wanda_pruning_matches_python_reference_plain():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6, seed=80)
    rng_cal = np.random.default_rng(81)
    x_cal = rng_cal.standard_normal((3, 6, 8)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_attention_head_wanda_pruning_matches_python_reference_multi_batch():
    # Multiple calibration batches, accumulated sum-of-squares across all of
    # them -- exercises WandaCalibrationStats' own per-batch accumulation
    # loop against pruning.py's own identical `for batch in calibration_data`
    # loop.
    model, _ = _attention_model(K=6, H=4, D=2, Out=5, seed=90, batch=2, seq=4)
    rng_cal = np.random.default_rng(91)
    calibration_data = [
        {"X": rng_cal.standard_normal((2, 4, 6)).astype(np.float32)} for _ in range(4)
    ]

    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.6
    )
    pruned_py = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.6
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_gqa_wanda_pruning_matches_python_reference():
    model, cfg = _gqa_model(K=8, H=8, KVH=4, D=8, Out=6, seed=100)
    rng_cal = np.random.default_rng(101)
    x_cal = rng_cal.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
        np.float32
    )
    calibration_data = [{"X": x_cal}]

    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


# --- Error handling ----------------------------------------------------------


def test_cpp_attention_head_wanda_pruning_missing_calibration_input_raises():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6, seed=120)
    bad_batch = {"NotX": np.zeros((2, 5, 8), dtype=np.float32)}
    with pytest.raises(Exception):
        onnxsim.apply_attention_head_wanda_pruning_cpp(
            model, calibration_data=[bad_batch], sparsity=0.5
        )


def test_cpp_attention_head_wanda_pruning_invalid_sparsity_raises():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6, seed=121)
    with pytest.raises(Exception):
        onnxsim.apply_attention_head_wanda_pruning_cpp(
            model, calibration_data=[], sparsity=1.0
        )
    with pytest.raises(Exception):
        onnxsim.apply_attention_head_wanda_pruning_cpp(
            model, calibration_data=[], sparsity=-0.1
        )


# --- Default (auto-generated) calibration data ------------------------------


def test_cpp_attention_head_wanda_pruning_default_calibration_data_runs():
    # calibration_data=None generates random calibration batches via
    # onnxsim.generate_random_calibration_data (symbolic batch/seq dims
    # fixed to 1), matching the pure-Python
    # apply_attention_head_wanda_pruning's own default -- just confirms the
    # whole path runs end to end and produces a valid, actually-pruned
    # model, not a specific oracle (random data has no fixed oracle here).
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=130)
    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, num_samples=4, seed=5, sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    node = _attention_node(pruned)
    num_heads, _ = _attention_attrs(node)
    assert num_heads == 2


# --- importance_norm ("l1" vs "l2") ------------------------------------------
#
# Same adversarial per-head/per-group weight blocks as
# test_attention_head_pruning_cpp.py's own importance_norm tests, driven
# through the *empty-calibration-data* fallback path (mirrors
# `test_cpp_attention_head_wanda_pruning_empty_calibration_data_matches_plain`/
# `test_cpp_gqa_wanda_pruning_empty_calibration_data_matches_plain` above):
# with no observed activation, `_wanda_attention_head_importance`/
# `_wanda_gqa_group_importance` fall straight back to the plain
# `||W||`-only ranking, so this isolates the *weight*-magnitude term's own
# L1-vs-L2 switch from the (always-L2) activation-norm term -- while still
# exercising the real Wanda entry point/binding end to end, not just the
# plain one.


def test_cpp_attention_head_wanda_pruning_importance_norm_l1_matches_python_reference_and_differs_from_l2():
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
        pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
            model, calibration_data=[], sparsity=0.5, importance_norm=norm
        )
        pruned_py = onnxsim.apply_attention_head_wanda_pruning(
            model, calibration_data=[], sparsity=0.5, importance_norm=norm
        )
        onnx.checker.check_model(pruned_cpp)
        assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    kept_l2 = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    kept_l1 = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5, importance_norm="l1"
    )
    assert kept_l2.SerializeToString() != kept_l1.SerializeToString()


def test_cpp_gqa_wanda_pruning_importance_norm_l1_matches_python_reference_and_differs_from_l2():
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
        pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
            model, calibration_data=[], sparsity=0.5, importance_norm=norm
        )
        pruned_py = onnxsim.apply_attention_head_wanda_pruning(
            model, calibration_data=[], sparsity=0.5, importance_norm=norm
        )
        onnx.checker.check_model(pruned_cpp)
        assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    kept_l2 = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    kept_l1 = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5, importance_norm="l1"
    )
    assert kept_l2.SerializeToString() != kept_l1.SerializeToString()


# ============================================================================
# Six more matched families -- see test_attention_head_pruning_cpp.py's own
# identical section for the model builders' full rationale and this port's
# deliberate, narrower-than-pruning.py scope decisions (no dynamic-
# attention-bias-Gather-insertion machinery, so every new matcher declines
# outright whenever such an optional input resolves to a non-empty
# constant). Every "matches python reference" test below cross-checks
# against ``onnxsim.apply_attention_head_wanda_pruning`` (the pure-Python
# reference) run with the SAME real calibration data through the SAME
# ``onnxruntime``-backed executor, so a difference here would mean the two
# ports' calibration-crossing/importance-ranking logic genuinely disagree,
# not merely "both produce a valid model".


def _mha_model(
    K=8, H=4, D=4, Out=6, seed=0, batch=2, seq=5, wq=None, wk=None, wv=None, wout=None
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
    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
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
        wout=wout,
        batch=batch,
        seq=seq,
    )


def _mha_node(model):
    return next(n for n in model.graph.node if n.op_type == "MultiHeadAttention")


def _mha_num_heads(node):
    return next(a.i for a in node.attribute if a.name == "num_heads")


def _plain_gqa_importance(wq, wk, wv, num_heads, kv_num_heads, head_size):
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
    return importance


def _wanda_gqa_keep_groups(
    wq, wk, wv, num_heads, kv_num_heads, head_size, act_norm, keep_count
):
    group_size = num_heads // kv_num_heads
    base = _plain_gqa_importance(wq, wk, wv, num_heads, kv_num_heads, head_size)
    act_group = np.array(
        [
            np.linalg.norm(
                act_norm[
                    kv * group_size * head_size : (kv + 1) * group_size * head_size
                ]
            )
            for kv in range(kv_num_heads)
        ]
    )
    importance = base * np.maximum(act_group, 1e-8)
    return np.sort(np.argsort(-importance)[:keep_count])


def test_cpp_mha_wanda_pruning_matches_oracle_exactly():
    model, cfg = _mha_model(K=8, H=8, D=4, Out=6, seed=200)
    rng = np.random.default_rng(201)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    act_norm = _probe_act_norm(model, "ctx", {"X": x_cal})
    keep_heads = _wanda_gqa_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["H"], cfg["D"], act_norm, 4
    )

    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    node = _mha_node(pruned)
    assert _mha_num_heads(node) == len(keep_heads)

    d = cfg["D"]
    idx = _head_idx(keep_heads, d)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"][:, idx])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"][:, idx])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"][:, idx])
    np.testing.assert_array_equal(inits["Wout"], cfg["wout"][idx, :])


def test_cpp_mha_wanda_pruning_matches_python_reference():
    model, cfg = _mha_model(K=8, H=8, D=4, Out=6, seed=202)
    rng = np.random.default_rng(203)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


# --- com.microsoft::PackedMultiHeadAttention --------------------------------


def _packed_mha_model(
    K=8, H=4, D=4, Out=6, seed=0, tok=5, wq=None, wk=None, wv=None, wout=None
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
    body = f"""
        g (float[{tok},{K}] X) => (float[{tok},{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx = com.microsoft.PackedMultiHeadAttention <num_heads={H}> (q, k, v, , TokenOffset, CumSeqLen)
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
        wout=wout,
        tok=tok,
    )


def _packed_mha_node(model):
    return next(n for n in model.graph.node if n.op_type == "PackedMultiHeadAttention")


def test_cpp_packed_mha_wanda_pruning_empty_calibration_data_matches_python_reference():
    # No CPU kernel exists for `PackedMultiHeadAttention` in this environment
    # (confirmed via a real `onnxruntime.InferenceSession` load -- see
    # `_match_packed_multi_head_attention_producer`'s own docstring and this
    # file's sibling `test_attention_head_pruning_cpp.py`'s own note on the
    # same op) -- so a *real* calibration run (which must actually execute
    # the graph) can never succeed for this op on any input, Python or C++
    # port alike. `calibration_data=[]` exercises the real Wanda entry point
    # end to end without ever executing the graph (falls back to plain
    # `||W||_F` ranking, the same "no observed activation" fallback every
    # other family's own analogous empty-calibration-data test already
    # relies on), while still cross-checking this port's own combined-bias/
    # importance-ranking logic against the Python reference exactly.
    model, _ = _packed_mha_model(K=8, H=8, D=4, Out=6, seed=210)
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    pruned_py = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    plain = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == plain.SerializeToString()


# --- com.microsoft::DecoderMaskedMultiHeadAttention -------------------------


def _dmmha_model(
    K=8, H=4, D=4, Out=6, seed=0, batch=2, wq=None, wk=None, wv=None, wout=None
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
    body = f"""
        g (float[{batch},1,{K}] X) => (float[{batch},1,{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx = com.microsoft.DecoderMaskedMultiHeadAttention <num_heads={H}> (q, k, v)
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
        wout=wout,
        batch=batch,
    )


def _dmmha_node(model):
    return next(
        n for n in model.graph.node if n.op_type == "DecoderMaskedMultiHeadAttention"
    )


def test_cpp_dmmha_wanda_pruning_matches_python_reference():
    model, cfg = _dmmha_model(K=8, H=8, D=4, Out=6, seed=220)
    rng = np.random.default_rng(221)
    x_cal = rng.standard_normal((cfg["batch"], 1, cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


# --- com.microsoft::PagedAttention -------------------------------------------
#
# See test_attention_head_pruning_cpp.py's own identical section for why
# this deliberately builds a float32 (not the real schema's float16-only)
# model -- the shared Q/K/V-producer-matching machinery this whole "Attention
# -head pruning" C++ section uses is FLOAT32-only, a pre-existing restriction
# shared by every family, GQA/OnnxAttention included.


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
    operands = [
        "q",
        "k",
        "v",
        "KeyCache",
        "ValueCache",
        "CumSeqLen",
        "PastSeqLens",
        "BlockTable",
    ]
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
        num_blocks=num_blocks,
        block_size=block_size,
    )


def _paged_node(model):
    return next(n for n in model.graph.node if n.op_type == "PagedAttention")


def test_cpp_paged_attention_wanda_pruning_empty_calibration_data_matches_python_reference():
    # This op's real onnxruntime schema requires `query`/`key`/`value` be
    # float16/bfloat16 (see this file's sibling
    # `test_attention_head_pruning_cpp.py`'s own note on this op) -- so a
    # float32 model, the only kind this port's shared FLOAT32-only Q/K/V-
    # producer-matching machinery can match at all, fails outright at
    # `InferenceSession` graph-load time ("Type Error: Type 'tensor(float)'
    # ... is invalid"), confirmed empirically. A *real* calibration run
    # (which must actually execute the graph) can therefore never succeed
    # for this port's own matched shape -- `calibration_data=[]` exercises
    # the real Wanda entry point end to end without ever executing the
    # graph (falls back to plain `||W||_F` ranking), still cross-checking
    # this port's own importance-ranking/slicing logic against the Python
    # reference exactly.
    model, _ = _paged_attention_model(K=8, H=8, KVH=4, D=8, Out=6, seed=230)
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    pruned_py = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    plain = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == plain.SerializeToString()


# --- Plain ai.onnx::LinearAttention (opset 27+, "linear" update_rule only) --


def _linear_attention_model(
    Hq=4, Hkv=2, D=4, K=16, seed=0, wq=None, wk=None, wv=None, wo=None
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
    body = f"""
        g (float[1,3,{K}] X) => (float[1,3,{K}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          attn_out, ps = LinearAttention<q_num_heads={Hq}, kv_num_heads={Hkv}, update_rule="linear">(q, k, v)
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


def test_cpp_linear_attention_wanda_pruning_matches_python_reference():
    # `LinearAttention` is plain ai.onnx opset 27, which this environment's
    # onnxruntime treats as "under development" and refuses to load by
    # default (`ValidateOpsetForDomain`) -- a real calibration run must
    # actually execute the graph, so this relaxes that load-time check the
    # same way `test_pruning.py`'s own `_run27` helper does (see that
    # helper's own comment); it does not stub out or change this op's own
    # real CPU kernel.
    os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"
    model, cfg = _linear_attention_model(Hq=8, Hkv=4, D=4, K=16, seed=240)
    rng = np.random.default_rng(241)
    x_cal = rng.standard_normal((1, 3, cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


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
    row_indices = np.tile(np.array([0, 1], dtype=np.int32), (num_layout, 1))
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
    extra_inputs = (
        f", float[{batch},{KVH},{seq},{D}] PastKey"
        f", float[{batch},{KVH},{seq},{D}] PastValue"
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
    )


def _sparse_attention_node(model):
    return next(n for n in model.graph.node if n.op_type == "SparseAttention")


def _sparse_attention_attrs(node):
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return num_heads, kv_num_heads


def test_cpp_sparse_attention_wanda_pruning_empty_calibration_data_matches_python_reference():
    # The real onnxruntime CPU kernel for `SparseAttention` requires
    # `past_key`/`past_value` be bound to the EXACT SAME buffer as
    # `present_key`/`present_value` (`past_key->DataRaw() ==
    # present_key->DataRaw()`, confirmed empirically -- see this file's
    # sibling `test_pruning.py`'s own `_run_sparse_attention` helper, which
    # exists specifically to satisfy this via `onnxruntime`'s IOBinding
    # API). Plain `sess.run()` -- what this port's (and pruning.py's own)
    # Wanda calibration machinery uses to probe activations -- can never
    # satisfy that in-place-update requirement, so a *real* calibration run
    # can never succeed for this op regardless of pruning correctness.
    # `calibration_data=[]` exercises the real Wanda entry point end to end
    # without ever executing the graph (falls back to plain `||W||_F`
    # ranking), still cross-checking this port's own num_layout-divisibility/
    # importance-ranking/slicing logic against the Python reference exactly.
    model, _ = _sparse_attention_model(K=8, H=8, KVH=4, D=8, Out=6, seed=250)
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    pruned_py = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    plain = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == plain.SerializeToString()


# --- Decomposed (un-fused) GQA/MQA/plain-MHA attention head pruning --------
#
# `apply_attention_head_wanda_pruning_cpp` now also matches this shape, via
# the same genuinely new, dedicated FindDecomposedGqaChains/
# ApplyOneDecomposedGqaChain machinery `apply_attention_head_pruning_cpp`
# uses (threaded through with a real calibrated `act_norm` map, exactly like
# every other family here) -- see
# ``test_attention_head_pruning_cpp.py``'s own "Decomposed (un-fused)
# GQA/MQA/plain-MHA attention head pruning" section comment for the exact
# scope this port matches (deliberately narrower than pruning.py's own
# ``_find_decomposed_gqa_chains``: no mask/RoPE/Q-K-norm/Einsum/packed-QKV,
# no true-MQA fast path) -- so this tenth family is NOT yet aliased either.


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
    wq=None,
    wk=None,
    wv=None,
    wout=None,
    bq=None,
    bk=None,
    bv=None,
    bout=None,
    share_kv_reshape_shape=True,
    extra_foreign_q_reshape_consumer=False,
):
    """Builds the decomposed-attention graph FindDecomposedGqaChains matches
    -- mirrors ``test_attention_head_pruning_cpp.py``'s own
    ``_decomposed_gqa_model`` (trimmed further: no ``out_reshape_wildcard``/
    ``extra_foreign_repeat_kv_consumer`` params, not needed by this file's
    own coverage -- see that copy's own docstring for the full shape and
    every other parameter's meaning)."""
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

    def _i64(arr, name):
        return onnx.numpy_helper.from_array(np.array(arr, dtype=np.int64), name)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]

    initializer.append(_i64([batch * seq, K], "XFlatShape"))
    lines = ["xf = Reshape(X, XFlatShape)"]
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
    lines += [
        "q0 = " + q_op,
        "qr = Reshape(q0, Sq)",
        "qt = Transpose<perm=[0,2,1,3]>(qr)",
    ]

    if extra_foreign_q_reshape_consumer:
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
        mask = np.triu(np.full((seq, seq), -1e4, dtype=np.float32), k=1)[
            None, None, :, :
        ]
        initializer.append(_f32(mask, "Mask"))
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
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
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
    )


def _decomposed_weight_shapes(model):
    inits = {t.name: t for t in model.graph.initializer}
    return (
        onnx.numpy_helper.to_array(inits["Wq"]),
        onnx.numpy_helper.to_array(inits["Wk"]),
        onnx.numpy_helper.to_array(inits["Wv"]),
        onnx.numpy_helper.to_array(inits["Wout"]),
    )


def test_cpp_decomposed_gqa_wanda_pruning_matches_oracle_exactly():
    model, cfg = _decomposed_gqa_model(K=32, H=8, KVH=2, D=8, Out=16, seed=101)
    rng = np.random.default_rng(102)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    act_norm = _probe_act_norm(model, "ctx2", {"X": x_cal})
    new_kv = 1  # KVH=2, sparsity=0.5 -> keep_count = max(1, 2 - round(1)) == 1
    keep_groups = _wanda_gqa_keep_groups(
        cfg["wq"],
        cfg["wk"],
        cfg["wv"],
        cfg["H"],
        cfg["KVH"],
        cfg["D"],
        act_norm,
        new_kv,
    )
    group_size = cfg["H"] // cfg["KVH"]
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    pruned = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    wq_new, wk_new, wv_new, wo_new = _decomposed_weight_shapes(pruned)
    np.testing.assert_array_equal(wq_new, cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(wk_new, cfg["wk"][:, kv_idx])
    np.testing.assert_array_equal(wv_new, cfg["wv"][:, kv_idx])
    np.testing.assert_array_equal(wo_new, cfg["wout"][q_idx, :])


def test_cpp_decomposed_gqa_wanda_pruning_matches_python_reference_exactly():
    model, cfg = _decomposed_gqa_model(K=32, H=8, KVH=2, D=8, Out=16, seed=103)
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)

    rng = np.random.default_rng(104)
    calibration_data = [
        {
            "X": rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
                np.float32
            )
        }
        for _ in range(3)
    ]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_attention_head_wanda_pruning(
        model_for_py, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_decomposed_gqa_wanda_pruning_clones_shape_constant_shared_with_foreign_reader():
    model, cfg = _decomposed_gqa_model(
        K=64,  # == H * D, so `foreign_out`'s own Reshape(xf, Sq) is valid.
        H=8,
        KVH=2,
        D=8,
        Out=16,
        seed=105,
        extra_foreign_q_reshape_consumer=True,
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)

    rng = np.random.default_rng(106)
    calibration_data = [
        {
            "X": rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
                np.float32
            )
        }
    ]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_attention_head_wanda_pruning(
        model_for_py, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_cpp.graph.initializer
    }
    np.testing.assert_array_equal(
        inits["Sq"], [cfg["batch"], cfg["seq"], cfg["H"], cfg["D"]]
    )
    assert "Sq_pruned" in inits


def test_cpp_decomposed_gqa_wanda_pruning_with_constant_mask_matches_python_reference_exactly():
    # An additive mask `Add` before `Softmax`, a constant of the
    # schema-documented per-head-broadcastable shape -- `ApplyAttentionHead
    # WandaPruning` shares `FindDecomposedGqaChains`/`ApplyOneDecomposedGqaChain`
    # with the data-free entry point above, so it now matches and correctly
    # leaves this broadcast mask untouched here too (see
    # test_attention_head_pruning_cpp.py's own identical, more thorough
    # coverage of this branch for the full reasoning) -- both ports must
    # agree byte-for-byte, whether or not calibration data is supplied.
    model, cfg = _decomposed_gqa_model(
        K=32, H=8, KVH=2, D=8, Out=16, seed=107, masked=True
    )
    model_for_py = onnx.ModelProto()
    model_for_py.CopyFrom(model)
    rng = np.random.default_rng(108)
    calibration_data = [
        {
            "X": rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
                np.float32
            )
        }
    ]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_attention_head_wanda_pruning(
        model_for_py, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() != model.SerializeToString()
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()


def test_cpp_decomposed_mqa_wanda_pruning_is_a_permanent_no_op_unlike_python():
    model, cfg = _decomposed_gqa_model(K=32, H=8, KVH=1, D=8, Out=16, seed=109)
    rng = np.random.default_rng(110)
    calibration_data = [
        {
            "X": rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(
                np.float32
            )
        }
    ]
    pruned_cpp = onnxsim.apply_attention_head_wanda_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    onnx.checker.check_model(pruned_py)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() != model.SerializeToString()
