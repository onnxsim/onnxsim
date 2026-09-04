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
and the plain ``ai.onnx::Attention`` op), minus the fused
``com.microsoft::MatMulNBitsQkv`` variant that port also matches -- this
Wanda port has no quantized-weight counterpart, mirroring the pure-Python
``onnxsim.apply_attention_head_wanda_pruning`` exactly (see
``structured_pruning_entry.h``'s own ``ApplyAttentionHeadWandaPruning``
declaration comment).
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
