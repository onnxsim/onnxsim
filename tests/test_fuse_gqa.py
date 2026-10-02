"""Tests for the ``fuse_gqa`` C++ pass (``onnxsim/passes/fuse_gqa.h``) --
pattern-matches a causal grouped-query/multi-query attention block (fewer K/V
heads than Q heads, broadcast via HuggingFace's standard ``repeat_kv``, plus
an additive causal mask) into a single ONNX Runtime "com.microsoft" contrib
op, ``GroupQueryAttention``. Like ``fuse_attention``, this is a default-on
graph-shape fusion that always runs as part of plain ``onnxsim.simplify()``.

Every model here is built directly with ``onnx.helper`` (no torch dependency)
to mirror what a real traced ``repeat_kv``-based GQA export produces -- see
``fuse_gqa.h``'s own top-of-file comment for the exact node shape this
targets and, importantly, why it only fires when the additive mask is a
*provable constant* matching the causal pattern exactly (``GroupQueryAttention``
always applies causal masking internally and unconditionally -- confirmed
during development by comparing its output against manual bidirectional vs.
causal references on the same random inputs).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim import model_checking

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for; the fused output is a
# "com.microsoft" contrib op that only onnxruntime can execute.
ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _i64(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.int64), name)


def _causal_mask(seq_len):
    mask = np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)
    mask[0, 0][np.triu_indices(seq_len, k=1)] = -np.inf
    return mask


def _gqa_model(
    B=2, S=6, NH=8, NKV=2, Dh=16, mask=None, mask_is_input=False, scale=None
):
    # Builds Y = Linear(ctx) where ctx is a causal GQA/MQA self-attention
    # context: separate Q/K/V nn.Linear-style (bias-free) projections,
    # head-split, K/V's repeat_kv broadcast up to Q's head count, scaled
    # dot-product, an additive mask, softmax, weighted sum -- see
    # fuse_gqa.h's own top comment for the exact shape this mirrors.
    n_rep = NH // NKV
    H = NH * Dh
    HKV = NKV * Dh
    rng = np.random.default_rng(0)

    inits = [
        _f32(rng.standard_normal((H, H)) * 0.1, "wq"),
        _f32(rng.standard_normal((H, HKV)) * 0.1, "wk"),
        _f32(rng.standard_normal((H, HKV)) * 0.1, "wv"),
        _f32(rng.standard_normal((H, H)) * 0.1, "wo"),
        _i64([B, S, NH, Dh], "shape_q"),
        _i64([B, S, NKV, Dh], "shape_kv"),
        _i64([2], "unsq_axes"),
        _i64([B, NKV, n_rep, S, Dh], "expand_shape"),
        _i64([B, NH, S, Dh], "merge_shape"),
        _i64([B, S, H], "shape_ctx"),
    ]
    if scale is None:
        inits.append(_f32(np.array(float(Dh) ** 0.5), "sqrt_dh"))
    else:
        inits.append(_f32(np.array(scale), "score_scale"))

    def repeat_kv_body(raw_name, prefix):
        return f"""
        {prefix}_unsq = Unsqueeze({raw_name}, unsq_axes)
        {prefix}_exp = Expand({prefix}_unsq, expand_shape)
        {prefix}_rep = Reshape({prefix}_exp, merge_shape)
        """

    body = f"""
    q_mm = MatMul(x, wq)
    q_r = Reshape(q_mm, shape_q)
    q_t = Transpose<perm = [0, 2, 1, 3]>(q_r)
    k_mm = MatMul(x, wk)
    k_r = Reshape(k_mm, shape_kv)
    k_raw = Transpose<perm = [0, 2, 1, 3]>(k_r)
    {repeat_kv_body("k_raw", "k")}
    k_t = Transpose<perm = [0, 1, 3, 2]>(k_rep)
    v_mm = MatMul(x, wv)
    v_r = Reshape(v_mm, shape_kv)
    v_raw = Transpose<perm = [0, 2, 1, 3]>(v_r)
    {repeat_kv_body("v_raw", "v")}
    qk = MatMul(q_t, k_t)
    """
    body += (
        "scores = Div(qk, sqrt_dh)\n"
        if scale is None
        else "scores = Mul(qk, score_scale)\n"
    )

    inputs = [f"float[{B},{S},{H}] x"]
    if mask is None:
        softmax_input = "scores"
    elif mask_is_input:
        body += "masked = Add(scores, mask)\n"
        inputs.append(f"float[1,1,{S},{S}] mask")
        softmax_input = "masked"
    else:
        inits.append(_f32(mask, "mask"))
        body += "masked = Add(scores, mask)\n"
        softmax_input = "masked"

    body += f"""
    probs = Softmax<axis = -1>({softmax_input})
    ctx0 = MatMul(probs, v_rep)
    ctx1 = Transpose<perm = [0, 2, 1, 3]>(ctx0)
    ctx2 = Reshape(ctx1, shape_ctx)
    y = MatMul(ctx2, wo)
    """

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g ({", ".join(inputs)}) => (float[{B},{S},{H}] y)
        {{
          {body}
        }}
        """
    )
    model.graph.initializer.extend(inits)
    return model


def _op_counts(model):
    import collections

    return collections.Counter(n.op_type for n in model.graph.node)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _assert_close(float_outputs, fused_outputs):
    for f, q in zip(float_outputs, fused_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < 1e-4, f"relative L2 error too large: {rel_l2:.6f}"


def test_fuse_gqa_basic():
    B, S, NH, NKV, Dh = 2, 6, 8, 2, 16
    model = _gqa_model(B=B, S=S, NH=NH, NKV=NKV, Dh=Dh, mask=_causal_mask(S))
    simplified, ok = onnxsim.simplify(model)
    assert ok
    ops = _op_counts(simplified)
    assert ops["GroupQueryAttention"] == 1
    assert ops["Softmax"] == 0
    gqa = next(n for n in simplified.graph.node if n.op_type == "GroupQueryAttention")
    num_heads = next(a for a in gqa.attribute if a.name == "num_heads").i
    kv_num_heads = next(a for a in gqa.attribute if a.name == "kv_num_heads").i
    assert num_heads == NH
    assert kv_num_heads == NKV
    domains = {o.domain for o in simplified.opset_import}
    assert "com.microsoft" in domains
    onnx.checker.check_model(simplified)

    rng = np.random.default_rng(42)
    x = rng.standard_normal((B, S, NH * Dh)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_gqa_multi_query():
    # MQA: a single shared K/V head (NKV=1) is the extreme case of GQA.
    B, S, NH, NKV, Dh = 2, 5, 4, 1, 8
    model = _gqa_model(B=B, S=S, NH=NH, NKV=NKV, Dh=Dh, mask=_causal_mask(S))
    simplified, ok = onnxsim.simplify(model)
    assert ok
    ops = _op_counts(simplified)
    assert ops["GroupQueryAttention"] == 1
    onnx.checker.check_model(simplified)

    rng = np.random.default_rng(7)
    x = rng.standard_normal((B, S, NH * Dh)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_gqa_declines_zero_scale():
    B, S, NH, NKV, Dh = 2, 5, 4, 1, 8
    model = _gqa_model(
        B=B,
        S=S,
        NH=NH,
        NKV=NKV,
        Dh=Dh,
        mask=_causal_mask(S),
        scale=0.0,
    )
    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _op_counts(simplified)["GroupQueryAttention"] == 0
    x = np.random.default_rng(8).standard_normal((B, S, NH * Dh)).astype(np.float32)
    assert model_checking.compare(
        simplified, model, n_times=1, input_data={"x": x}, verbose=False
    )


def test_fuse_gqa_declines_without_mask():
    # Bidirectional (no additive mask at all): GroupQueryAttention always
    # applies causal masking internally with no way to disable it, so this
    # must decline rather than silently turn a bidirectional block causal.
    B, S, NH, NKV, Dh = 2, 6, 8, 2, 16
    model = _gqa_model(B=B, S=S, NH=NH, NKV=NKV, Dh=Dh, mask=None)
    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _op_counts(simplified)["GroupQueryAttention"] == 0

    rng = np.random.default_rng(1)
    x = rng.standard_normal((B, S, NH * Dh)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_gqa_declines_non_causal_mask():
    # A mask that's present but numerically not the standard causal pattern
    # (all zeros here -- i.e. no actual masking) must not be assumed causal.
    B, S, NH, NKV, Dh = 2, 6, 8, 2, 16
    model = _gqa_model(
        B=B, S=S, NH=NH, NKV=NKV, Dh=Dh, mask=np.zeros((1, 1, S, S), dtype=np.float32)
    )
    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _op_counts(simplified)["GroupQueryAttention"] == 0

    rng = np.random.default_rng(2)
    x = rng.standard_normal((B, S, NH * Dh)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_gqa_declines_finite_causal_penalty():
    B, S, NH, NKV, Dh = 2, 5, 4, 1, 8
    mask = _causal_mask(S)
    mask[np.isneginf(mask)] = -1e30
    model = _gqa_model(B=B, S=S, NH=NH, NKV=NKV, Dh=Dh, mask=mask)
    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _op_counts(simplified)["GroupQueryAttention"] == 0
    x = np.random.default_rng(9).standard_normal((B, S, NH * Dh)).astype(np.float32)
    assert model_checking.compare(
        simplified, model, n_times=1, input_data={"x": x}, verbose=False
    )


def test_fuse_gqa_declines_runtime_mask():
    # A mask that's present, causal-shaped, and would numerically pass
    # VerifyCausalMaskConstant -- but is a runtime graph *input*, not a
    # compile-time constant. Real GQA exports almost always pass their mask
    # this way; this pass deliberately declines rather than trust an
    # un-provable runtime tensor is exactly causal-shaped (see fuse_gqa.h's
    # own top comment for why).
    B, S, NH, NKV, Dh = 2, 6, 8, 2, 16
    model = _gqa_model(
        B=B, S=S, NH=NH, NKV=NKV, Dh=Dh, mask=_causal_mask(S), mask_is_input=True
    )
    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _op_counts(simplified)["GroupQueryAttention"] == 0

    rng = np.random.default_rng(3)
    x = rng.standard_normal((B, S, NH * Dh)).astype(np.float32)
    mask = _causal_mask(S)
    _assert_close(
        _run(model, {"x": x, "mask": mask}), _run(simplified, {"x": x, "mask": mask})
    )
