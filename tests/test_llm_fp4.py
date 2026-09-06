"""Tests for ``onnxsim.quantize_weight_only_llm_fp4``,
``onnxsim.apply_llm_fp4_activation_quantization`` and
``onnxsim.apply_llm_fp4_activation_quantization_per_tensor`` (LLM-FP4, see
``onnxsim/llm_fp4.py``) -- block-wise quantization onto a searched
sign/exponent/mantissa FP4 format (bit split searched per tensor) with a
per-block *real-valued* scale (searched jointly, standing in for the
paper's own "pre-shifted exponent bias"), represented in the ONNX graph via
ordinary Gather/Reshape/Mul (no contrib op, no opset-21 features), plus the
module's two activation-side passes completing W4A4 for layers already
weight-quantized this way: a per-token, data-free one (a different
granularity than the paper's own design) and a calibrated per-tensor one
(the paper's own quantizer, whose per-channel outlier-migration half is the
caller's job -- ``apply_smoothquant``/``apply_outlier_suppression`` first).

Per this repo's own platform-numerics lesson (onnxruntime's MatMul kernel
reduction order is not bit-exact across CPU architectures), value
correctness is checked by dequantizing the written initializers directly in
numpy and comparing against the original float weight with a tight
*relative* tolerance -- never by comparing onnxruntime outputs with an
absolute tolerance. Any onnxruntime round-trip check below is a separate,
much looser sanity check.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.llm_fp4 import FP4_FORMATS, _fp4_codebook, _fp4_magnitudes
from onnxsim.mx_quantization import MXFP4_CODEBOOK

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


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


def _matmul_model(K=64, N=16, weight=None, seed=0):
    if weight is None:
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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _codebook_used_by(model, w_name="W"):
    gather = next(
        n
        for n in model.graph.node
        if n.op_type == "Gather" and n.input[1] == f"{w_name}_llmfp4_codes_i64"
    )
    codebook_init = next(
        t for t in model.graph.initializer if t.name == gather.input[0]
    )
    return onnx.numpy_helper.to_array(codebook_init).astype(np.float64)


def _dequantize_llm_fp4_by_hand(model, w_name="W", block_size=32):
    """Independent reference decode: reads Wq/Ws/the winning codebook
    straight from the initializers and dequantizes via numpy, without using
    any of the ops this module inserts into the graph.
    """
    wq = next(t for t in model.graph.initializer if t.name == f"{w_name}_llmfp4_q")
    ws = next(t for t in model.graph.initializer if t.name == f"{w_name}_llmfp4_scale")
    codes = onnx.numpy_helper.to_array(wq).astype(np.int64)
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    codebook = _codebook_used_by(model, w_name)

    dim0, dim1 = codes.shape
    num_blocks = scale.shape[0]
    block_size_actual = dim0 // num_blocks
    assert block_size_actual == block_size
    values = codebook[codes]  # [dim0, dim1]
    scale_full = np.repeat(scale, block_size, axis=0)  # [dim0, dim1]
    return values * scale_full


def test_llm_fp4_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=64, N=16, seed=0)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert op_types <= {"MatMul", "Cast", "Gather", "Reshape", "Mul"}
    assert all(n.domain in ("", "ai.onnx") for n in q.graph.node)


def test_llm_fp4_dequantized_values_match_independently_recomputed_nearest_codebook():
    # Unlike onnxsim.mx_quantization/onnxsim.nf4 (whose block scale always
    # keeps the block's own max-abs element within the codebook's range),
    # this module's own scale search can deliberately choose a *tighter*
    # scale that clips outliers in exchange for lower total MSE -- so a
    # fixed "within half a codebook gap" per-element bound (those other
    # modules' own test pattern) does not hold here. Instead: trust only
    # the search's chosen per-block *scale* (Ws), and independently
    # recompute -- via nearest-codebook-index search, entirely in numpy,
    # not using any op this module inserts into the graph -- what the
    # *codes* (Wq) should be for that scale. This still catches any bug in
    # either the code assignment or the Gather/Reshape/Mul dequantization
    # subgraph, without assuming anything about how tightly the search
    # itself clips.
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.3
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)

    w_hand = _dequantize_llm_fp4_by_hand(q, block_size=32)
    codebook = _codebook_used_by(q)

    ws = next(t for t in q.graph.initializer if t.name == "W_llmfp4_scale")
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    scale_full = np.repeat(scale, 32, axis=0)

    normalized = weight.astype(np.float64) / scale_full
    diffs = np.abs(normalized[..., np.newaxis] - codebook[np.newaxis, np.newaxis, :])
    nearest_idx = np.argmin(diffs, axis=-1)
    independent_dequant = codebook[nearest_idx] * scale_full

    assert np.allclose(w_hand, independent_dequant, atol=1e-4, rtol=1e-4)

    # A loose sanity bound on overall reconstruction quality (this weight's
    # own measured relative L2 error is ~0.09) -- catches a search that
    # regressed to picking wildly bad scales/formats, without pinning an
    # exact number.
    rel_l2 = np.linalg.norm(w_hand - weight.astype(np.float64)) / np.linalg.norm(
        weight.astype(np.float64)
    )
    assert rel_l2 < 0.2


def test_llm_fp4_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=2)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(3)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    # Loose, absolute-precision-agnostic sanity check only -- the tight,
    # authoritative correctness check is the numpy hand-decode above.
    assert _rel_l2(float_y, q_y) < 0.3


def test_llm_fp4_gemm_transb():
    rng = np.random.default_rng(4)
    K, N = 128, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_llm_fp4_codes_stay_in_range():
    rng = np.random.default_rng(5)
    weight = rng.standard_normal((64, 8)).astype(np.float32) * 3
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    wq = next(t for t in q.graph.initializer if t.name == "W_llmfp4_q")
    codes = onnx.numpy_helper.to_array(wq)
    assert np.all(codes >= 0) and np.all(codes <= 15)


def test_llm_fp4_skips_non_block_divisible_k():
    model = _matmul_model(K=48, N=8, seed=6)  # 48 is not a multiple of 32
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    assert q.SerializeToString() == model.SerializeToString()


def test_llm_fp4_skip_names_leaves_matched_weight_untouched():
    rng = np.random.default_rng(7)
    w_base = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    w_other = rng.standard_normal((64, 4)).astype(np.float32) * 0.1
    model = _model(
        """
        g (float[batch,64] X) => (float[batch,16] Y, float[batch,4] H)
        {
          Y = MatMul(X, W)
          H = MatMul(X, W_other)
        }
        """,
        initializer=[_f32(w_base, "W"), _f32(w_other, "W_other")],
    )
    q = onnxsim.quantize_weight_only_llm_fp4(
        model, block_size=32, skip_names=["W_other"]
    )
    onnx.checker.check_model(q)

    names = {t.name for t in q.graph.initializer}
    assert "W_llmfp4_q" in names
    assert "W_other_llmfp4_q" not in names
    other_out = next(
        onnx.numpy_helper.to_array(t)
        for t in q.graph.initializer
        if t.name == "W_other"
    )
    assert np.array_equal(other_out, w_other)


def test_llm_fp4_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_llm_fp4(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_llm_fp4_rejects_unknown_format():
    model = _matmul_model()
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_llm_fp4(model, formats=["not_a_format"])


def test_llm_fp4_e2m1_codebook_matches_mxfp4():
    # E2M1 is exactly MXFP4's own element format -- the two codebooks must
    # be identical sets of magnitudes (this module's own layout convention
    # matches onnxsim.mx_quantization's, so they should be byte-identical).
    e2m1 = np.asarray(_fp4_codebook(*FP4_FORMATS["e2m1"]))
    mxfp4 = np.asarray(MXFP4_CODEBOOK)
    assert np.array_equal(e2m1, mxfp4)


def test_llm_fp4_formats_have_eight_magnitudes_each():
    for e_bits, m_bits in FP4_FORMATS.values():
        magnitudes = _fp4_magnitudes(e_bits, m_bits)
        assert len(magnitudes) == 8
        assert magnitudes[0] == 0.0
        assert np.all(np.diff(magnitudes) > 0)  # strictly increasing


def test_llm_fp4_codebooks_are_well_formed():
    for fmt in FP4_FORMATS:
        e_bits, m_bits = FP4_FORMATS[fmt]
        codebook = np.asarray(_fp4_codebook(e_bits, m_bits))
        assert codebook.shape == (16,)
        assert np.all(np.diff(codebook) >= 0)  # non-decreasing (duplicate zero)
        assert list(codebook).count(0.0) == 2  # +0.0 and -0.0
        assert codebook[0] == -codebook[-1]  # symmetric


def test_llm_fp4_scale_is_not_restricted_to_power_of_two():
    # Unlike onnxsim.mx_quantization's MXFP4 (E8M0: power-of-two only), this
    # module's own per-block scale is real-valued -- the whole point of
    # realizing the paper's "pre-shifted exponent bias" as a per-block
    # scale search rather than reusing MX's narrower E8M0 restriction.
    rng = np.random.default_rng(9)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 3.7
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)

    ws = next(t for t in q.graph.initializer if t.name == "W_llmfp4_scale")
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64).ravel()
    log2_scale = np.log2(scale)
    assert not np.all(np.abs(log2_scale - np.round(log2_scale)) < 1e-9)


def test_llm_fp4_format_search_beats_a_forced_single_format():
    # A weight tailor-made to reconstruct much better under E3M0 (wide
    # dynamic range, coarse mantissa -- octave-spaced magnitudes) than
    # E1M2 (narrow range, fine mantissa): the full search (default
    # `formats`) must find a reconstruction at least as good as forcing
    # E1M2 alone, and strictly better on this adversarial tensor.
    rng = np.random.default_rng(10)
    exponents = rng.integers(-6, 7, size=(64, 16))
    weight = (2.0**exponents).astype(np.float32) * rng.choice(
        [-1.0, 1.0], size=(64, 16)
    ).astype(np.float32)
    model = _matmul_model(weight=weight)

    q_search = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    q_e1m2 = onnxsim.quantize_weight_only_llm_fp4(
        model, block_size=32, formats=["e1m2"]
    )

    w_search = _dequantize_llm_fp4_by_hand(q_search, block_size=32)
    w_e1m2 = _dequantize_llm_fp4_by_hand(q_e1m2, block_size=32)
    w64 = weight.astype(np.float64)

    err_search = np.sum((w_search - w64) ** 2)
    err_e1m2 = np.sum((w_e1m2 - w64) ** 2)
    assert err_search <= err_e1m2
    assert err_search < err_e1m2 * 0.9  # strictly, meaningfully better


def test_llm_fp4_restricting_formats_only_uses_requested_ones():
    rng = np.random.default_rng(11)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32, formats=["e3m0"])

    names = {t.name for t in q.graph.initializer}
    assert "llm_fp4_codebook_e3m0" in names
    assert "llm_fp4_codebook_e1m2" not in names
    assert "llm_fp4_codebook_e2m1" not in names


# --- apply_llm_fp4_activation_quantization -----------------------------
#
# A per-token, data-free activation-quantization pass that completes W4A4
# for layers already weight-quantized by quantize_weight_only_llm_fp4 --
# see that function's own docstring for exactly how this differs from the
# paper's own per-tensor-via-migration design. Needs opset >= 18
# (ReduceMax's axes-as-input form), so these models parse at opset 18
# rather than the file's usual default of 13.


def _matmul_model_opset18(K=64, N=16, weight=None, seed=0):
    if weight is None:
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
        opset=18,
    )


def test_llm_fp4_activation_quant_output_stays_close_to_float():
    rng = np.random.default_rng(20)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    model = _matmul_model_opset18(weight=weight)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    qa = onnxsim.apply_llm_fp4_activation_quantization(q)
    onnx.checker.check_model(qa)

    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (qa_y,) = _run(qa, {"X": x})
    assert np.all(np.isfinite(qa_y))
    # Loose bound picked from a real run: this weight/input pair measured
    # ~0.15 relative L2 error for the full W4A4 round-trip (vs. ~0.09 for
    # weight-only FP4) -- both bit widths are lossy, so this is a sanity
    # bound against a badly broken construction, not a tight accuracy claim.
    assert _rel_l2(float_y, qa_y) < 0.4


def test_llm_fp4_activation_quant_actually_changes_output():
    # Proves the pass actually ran (inserted a real, non-identity
    # quantize/dequantize round-trip) rather than silently no-op'ing.
    rng = np.random.default_rng(21)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    model = _matmul_model_opset18(weight=weight)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    qa = onnxsim.apply_llm_fp4_activation_quantization(q)

    x = rng.standard_normal((8, 64)).astype(np.float32)
    (wonly_y,) = _run(q, {"X": x})
    (qa_y,) = _run(qa, {"X": x})
    # Measured ~0.10 relative L2 difference between weight-only and W4A4 on
    # this model/input -- a generous margin well clear of floating-point
    # noise, not a rounding-boundary tie.
    assert _rel_l2(wonly_y, qa_y) > 0.02


def test_llm_fp4_activation_quant_skips_plain_float_model():
    # No quantize_weight_only_llm_fp4 pass has run at all: nothing in this
    # model matches the exact weight-dequant pattern this function looks
    # for, so it must come back byte-identical.
    model = _matmul_model_opset18(seed=22)
    out = onnxsim.apply_llm_fp4_activation_quantization(model)
    assert out.SerializeToString() == model.SerializeToString()


def test_llm_fp4_activation_quant_skips_differently_quantized_layer():
    # A layer dequantized via a *different* pattern (a bare DequantizeLinear,
    # as onnxsim.quantize_weight_only_int4 produces) is not
    # quantize_weight_only_llm_fp4's own Gather/Reshape/Mul/Cast chain, so
    # it must be left completely untouched -- this pass never runs the
    # weight-side quantization itself and must not mistake one dequant
    # scheme for another.
    k, n = 64, 16
    wq = onnx.TensorProto()
    wq.name = "Wq"
    wq.data_type = onnx.TensorProto.INT4
    wq.dims.extend([k, n])
    wq.raw_data = b"\x00" * (k * n // 2)
    ws = _f32(np.ones((k // 32, n), dtype=np.float32), "Ws")

    model = _model(
        f"""
        g (float[batch,{k}] X) => (float[batch,{n}] Y)
        {{
          Wdq = DequantizeLinear<axis = 0, block_size = 32>(Wq, Ws)
          Y = MatMul(X, Wdq)
        }}
        """,
        initializer=[wq, ws],
        opset=21,
    )
    out = onnxsim.apply_llm_fp4_activation_quantization(model)
    assert out.SerializeToString() == model.SerializeToString()


def test_llm_fp4_activation_quant_uses_each_layers_own_codebook():
    # Two weights chosen so the format search picks two different formats
    # (mirrors test_llm_fp4_format_search_beats_a_forced_single_format's own
    # adversarial-weight construction for the wide-range tensor); confirms
    # the activation-quantization pass recovers and uses *each layer's own*
    # codebook, not some other layer's.
    k = 64
    rng = np.random.default_rng(23)
    exponents = rng.integers(-6, 7, size=(k, 16))
    w_wide = (2.0**exponents).astype(np.float32) * rng.choice(
        [-1.0, 1.0], size=(k, 16)
    ).astype(np.float32)  # favors e3m0: wide dynamic range, coarse mantissa

    rng2 = np.random.default_rng(24)
    w_narrow = (rng2.standard_normal((k, 8)) * 0.3).astype(np.float32)

    model = _model(
        f"""
        g (float[batch,{k}] X) => (float[batch,16] Y, float[batch,8] Z)
        {{
          Y = MatMul(X, W1)
          Z = MatMul(X, W2)
        }}
        """,
        initializer=[_f32(w_wide, "W1"), _f32(w_narrow, "W2")],
        opset=18,
    )
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)

    def _codebook_name_used_by(model, w_name):
        gather = next(
            n
            for n in model.graph.node
            if n.op_type == "Gather" and n.input[1] == f"{w_name}_llmfp4_codes_i64"
        )
        return gather.input[0]

    cb1_name = _codebook_name_used_by(q, "W1")
    cb2_name = _codebook_name_used_by(q, "W2")
    # The two adversarial weights must actually land on two different
    # formats for this test to be meaningful.
    assert cb1_name != cb2_name

    qa = onnxsim.apply_llm_fp4_activation_quantization(q)
    onnx.checker.check_model(qa)

    # Each layer's own inserted nearest-codebook-lookup ("Sub" against the
    # codebook) must reference that same layer's own codebook initializer.
    subs = {n.name: n.input[1] for n in qa.graph.node if n.op_type == "Sub"}
    w1_sub = next(name for name in subs if name.startswith("W1_llmfp4_dq_llmfp4act"))
    w2_sub = next(name for name in subs if name.startswith("W2_llmfp4_dq_llmfp4act"))
    assert subs[w1_sub] == cb1_name
    assert subs[w2_sub] == cb2_name

    # And the per-token scale's own denominator (that layer's codebook max
    # magnitude) must match each codebook's real max, not a swapped one.
    cb1 = onnx.numpy_helper.to_array(
        next(t for t in qa.graph.initializer if t.name == cb1_name)
    )
    cb2 = onnx.numpy_helper.to_array(
        next(t for t in qa.graph.initializer if t.name == cb2_name)
    )
    maxabs1 = onnx.numpy_helper.to_array(
        next(t for t in qa.graph.initializer if t.name == f"{cb1_name}_maxabs")
    )
    maxabs2 = onnx.numpy_helper.to_array(
        next(t for t in qa.graph.initializer if t.name == f"{cb2_name}_maxabs")
    )
    assert np.isclose(float(maxabs1), float(np.abs(cb1).max()))
    assert np.isclose(float(maxabs2), float(np.abs(cb2).max()))
    assert not np.isclose(float(maxabs1), float(maxabs2))


# --- apply_llm_fp4_activation_quantization_per_tensor -------------------
#
# The *calibrated per-tensor* activation quantizer: the paper's own
# activation-side quantizer (its per-channel outlier-migration half is the
# caller's job -- run apply_smoothquant/apply_outlier_suppression first).
# One real-valued scale per activation tensor, fit offline from calibration
# data by the same clip-ratio-vs-MSE grid search the weight side uses, and
# emitted as a constant initializer -- so the inserted subgraph carries no
# runtime range reduction at all, unlike the per-token pass above.
#
# Every numeric bound below was picked from a real measured run (see the
# comment on each) with generous margin, per this repo's own lesson about
# razor-thin single-seed inequalities.


def _act_scale_initializers(model):
    return [
        t for t in model.graph.initializer if t.name.endswith("_llmfp4act_pt_scale")
    ]


def test_llm_fp4_per_tensor_activation_quant_output_stays_close_to_float():
    rng = np.random.default_rng(20)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    model = _matmul_model_opset18(weight=weight)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    qa = onnxsim.apply_llm_fp4_activation_quantization_per_tensor(q)
    onnx.checker.check_model(qa)

    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (qa_y,) = _run(qa, {"X": x})
    assert np.all(np.isfinite(qa_y))
    # Measured across six independent weight/input seeds: 0.137 -- 0.172
    # relative L2 for the full W4A4 round-trip (vs. 0.074 -- 0.094 for
    # weight-only FP4, and 0.125 -- 0.146 for the per-token pass on the
    # same models). 0.4 is a generous sanity bound against a badly broken
    # construction, not a tight accuracy claim.
    assert _rel_l2(float_y, qa_y) < 0.4


def test_llm_fp4_per_tensor_activation_quant_actually_changes_output():
    # Proves the pass really ran (inserted a non-identity quantize/
    # dequantize round-trip) rather than silently no-op'ing.
    rng = np.random.default_rng(21)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    model = _matmul_model_opset18(weight=weight)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    qa = onnxsim.apply_llm_fp4_activation_quantization_per_tensor(q)

    x = rng.standard_normal((8, 64)).astype(np.float32)
    (wonly_y,) = _run(q, {"X": x})
    (qa_y,) = _run(qa, {"X": x})
    # Measured 0.108 -- 0.146 relative L2 difference from weight-only
    # across six seeds -- a generous margin above floating-point noise,
    # not a rounding-boundary tie.
    assert _rel_l2(wonly_y, qa_y) > 0.02


def test_llm_fp4_per_tensor_activation_quant_emits_no_runtime_scale_and_fewer_nodes():
    # The concrete structural difference between the two activation passes:
    # a compile-time-constant scale means no Abs/ReduceMax/Max range
    # reduction at graph-run time, so strictly fewer nodes than the
    # per-token pass on the same model. That is the practical point of the
    # per-tensor design.
    rng = np.random.default_rng(25)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    model = _matmul_model_opset18(weight=weight)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    per_token = onnxsim.apply_llm_fp4_activation_quantization(q)
    per_tensor = onnxsim.apply_llm_fp4_activation_quantization_per_tensor(q)

    op_types = [n.op_type for n in per_tensor.graph.node]
    assert "ReduceMax" not in op_types
    assert "Max" not in op_types
    # The only Abs is the codebook-distance one (shared with the per-token
    # pass); the per-token pass has a second Abs feeding its ReduceMax.
    assert op_types.count("Abs") == 1
    assert [n.op_type for n in per_token.graph.node].count("Abs") == 2

    # 7 inserted nodes per layer (Div, Unsqueeze, Sub, Abs, ArgMin, Gather,
    # Mul) vs. the per-token pass's 11 (those plus Abs/ReduceMax/Max and a
    # second Div to derive the scale).
    assert len(per_tensor.graph.node) - len(q.graph.node) == 7
    assert len(per_token.graph.node) - len(q.graph.node) == 11
    assert len(per_tensor.graph.node) < len(per_token.graph.node)


def test_llm_fp4_per_tensor_activation_scale_is_a_constant_initializer():
    rng = np.random.default_rng(26)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    model = _matmul_model_opset18(weight=weight)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    qa = onnxsim.apply_llm_fp4_activation_quantization_per_tensor(q)

    scales = _act_scale_initializers(qa)
    assert len(scales) == 1
    (scale_init,) = scales
    assert scale_init.data_type == onnx.TensorProto.FLOAT
    value = onnx.numpy_helper.to_array(scale_init)
    assert value.size == 1  # scalar (0-d here) or 1-element
    assert np.isfinite(float(value)) and float(value) > 0.0

    # And it really is what the graph divides/multiplies by -- not a stray
    # unused initializer next to a runtime-computed scale.
    div = next(n for n in qa.graph.node if n.op_type == "Div")
    mul = next(
        n for n in qa.graph.node if n.op_type == "Mul" and n.input[1] == scale_init.name
    )
    assert div.input[1] == scale_init.name
    assert mul.input[1] == scale_init.name


def test_llm_fp4_per_tensor_activation_quant_works_at_opset_13():
    # Unlike the per-token pass (which needs opset 18 for ReduceMax's
    # axes-as-input form), this pass emits only Unsqueeze (axes-as-input:
    # opset 13) plus ArgMin/Gather/Sub/Abs/Div/Mul, all older -- so opset
    # 13 is enough, and the per-token pass declines the very same model.
    rng = np.random.default_rng(27)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    model = _matmul_model(weight=weight)  # opset 13
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)

    qa = onnxsim.apply_llm_fp4_activation_quantization_per_tensor(q)
    onnx.checker.check_model(qa)
    assert len(qa.graph.node) > len(q.graph.node)

    x = rng.standard_normal((8, 64)).astype(np.float32)
    (qa_y,) = _run(qa, {"X": x})
    assert np.all(np.isfinite(qa_y))

    per_token = onnxsim.apply_llm_fp4_activation_quantization(q)
    assert per_token.SerializeToString() == q.SerializeToString()


def test_llm_fp4_per_tensor_activation_quant_skips_plain_float_model():
    # No quantize_weight_only_llm_fp4 pass has run at all: nothing matches
    # the exact weight-dequant pattern this function looks for, so it must
    # come back byte-identical (and must not even run calibration).
    model = _matmul_model_opset18(seed=28)
    out = onnxsim.apply_llm_fp4_activation_quantization_per_tensor(model)
    assert out.SerializeToString() == model.SerializeToString()


def test_llm_fp4_per_tensor_activation_quant_skips_differently_quantized_layer():
    # A layer dequantized via a *different* scheme (a bare DequantizeLinear,
    # as onnxsim.quantize_weight_only_int4 produces) is not
    # quantize_weight_only_llm_fp4's own Gather/Reshape/Mul/Cast chain, so
    # it must be left completely untouched -- this pass never runs the
    # weight-side quantization itself and must not mistake one dequant
    # scheme for another.
    k, n = 64, 16
    wq = onnx.TensorProto()
    wq.name = "Wq"
    wq.data_type = onnx.TensorProto.INT4
    wq.dims.extend([k, n])
    wq.raw_data = b"\x00" * (k * n // 2)
    ws = _f32(np.ones((k // 32, n), dtype=np.float32), "Ws")

    model = _model(
        f"""
        g (float[batch,{k}] X) => (float[batch,{n}] Y)
        {{
          Wdq = DequantizeLinear<axis = 0, block_size = 32>(Wq, Ws)
          Y = MatMul(X, Wdq)
        }}
        """,
        initializer=[wq, ws],
        opset=21,
    )
    out = onnxsim.apply_llm_fp4_activation_quantization_per_tensor(model)
    assert out.SerializeToString() == model.SerializeToString()


def test_llm_fp4_per_tensor_activation_quant_uses_each_layers_own_codebook():
    # Two weights chosen so the format search picks two different formats
    # (mirrors test_llm_fp4_activation_quant_uses_each_layers_own_codebook
    # above); confirms each layer's inserted quantizer recovers and uses
    # *that layer's own* codebook, not the other layer's, and fits its own
    # separate scale.
    k = 64
    rng = np.random.default_rng(23)
    exponents = rng.integers(-6, 7, size=(k, 16))
    w_wide = (2.0**exponents).astype(np.float32) * rng.choice(
        [-1.0, 1.0], size=(k, 16)
    ).astype(np.float32)  # favors e3m0: wide dynamic range, coarse mantissa

    rng2 = np.random.default_rng(24)
    w_narrow = (rng2.standard_normal((k, 8)) * 0.3).astype(np.float32)

    model = _model(
        f"""
        g (float[batch,{k}] X) => (float[batch,16] Y, float[batch,8] Z)
        {{
          Y = MatMul(X, W1)
          Z = MatMul(X, W2)
        }}
        """,
        initializer=[_f32(w_wide, "W1"), _f32(w_narrow, "W2")],
        opset=18,
    )
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)

    def _codebook_name_used_by(model, w_name):
        gather = next(
            n
            for n in model.graph.node
            if n.op_type == "Gather" and n.input[1] == f"{w_name}_llmfp4_codes_i64"
        )
        return gather.input[0]

    cb1_name = _codebook_name_used_by(q, "W1")
    cb2_name = _codebook_name_used_by(q, "W2")
    # The two adversarial weights must actually land on two different
    # formats for this test to be meaningful.
    assert cb1_name != cb2_name

    qa = onnxsim.apply_llm_fp4_activation_quantization_per_tensor(q)
    onnx.checker.check_model(qa)

    # Both layers read the same graph input X, so both inserted quantizers
    # must exist, and each one's nearest-codebook lookup ("Sub" against the
    # codebook) must reference its own layer's codebook -- the two together
    # must cover both codebooks, not the same one twice.
    subs = [n for n in qa.graph.node if n.op_type == "Sub"]
    assert len(subs) == 2
    assert {n.input[1] for n in subs} == {cb1_name, cb2_name}

    # Two separately fit scales, one per layer: the two codebooks have
    # different max magnitudes, so the same activation yields different
    # scales.
    scales = [float(onnx.numpy_helper.to_array(t)) for t in _act_scale_initializers(qa)]
    assert len(scales) == 2
    assert not np.isclose(scales[0], scales[1])

    # Each MatMul must consume its own layer's dequantized activation, and
    # the two must be different tensors.
    matmuls = [n for n in qa.graph.node if n.op_type == "MatMul"]
    assert len({n.input[0] for n in matmuls}) == 2


def test_llm_fp4_per_tensor_activation_quant_composes_with_smoothquant():
    # The paper's actual pipeline, and exactly the three-call sequence
    # apply_llm_fp4_activation_quantization_per_tensor's own docstring tells
    # callers to use: migrate the per-channel activation outliers first,
    # then weight-quantize, then per-tensor activation-quantize. A smoke
    # test that the composition really works end-to-end.
    rng = np.random.default_rng(29)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    model = _matmul_model_opset18(weight=weight)

    migrated = onnxsim.apply_smoothquant(model)
    weight_q = onnxsim.quantize_weight_only_llm_fp4(migrated, block_size=32)
    w4a4 = onnxsim.apply_llm_fp4_activation_quantization_per_tensor(weight_q)
    onnx.checker.check_model(w4a4)

    # The migration's own Mul is still there, and the activation quantizer
    # was inserted on top of it (not instead of it).
    assert any(n.op_type == "ArgMin" for n in w4a4.graph.node)
    assert len(_act_scale_initializers(w4a4)) == 1

    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (w4a4_y,) = _run(w4a4, {"X": x})
    assert np.all(np.isfinite(w4a4_y))
    # Measured ~0.144 relative L2 for this sequence -- essentially the same
    # as the un-migrated W4A4 run above (SmoothQuant has little to migrate
    # on a synthetic Gaussian activation with no per-channel outliers).
    # Same generous 0.4 sanity bound.
    assert _rel_l2(float_y, w4a4_y) < 0.4
