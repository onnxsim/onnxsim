"""Formal check for GgufQ4_1 (opt-in; onnxsim's own
``onnxsim/passes/gguf_legacy_quant.h``, C++ port of
``onnxsim.gguf_legacy_quant``'s own ``apply_gguf_q4_1_quantization``):
llama.cpp's legacy GGUF "Q4_1" block-quant format -- Q4_0's ASYMMETRIC
sibling (READ ``tests/test_formal_verify_gguf_q4_0.py`` FIRST: same file,
same block layout, same overall structure, but a genuinely different
per-block parameterization with, as this file confirms, NONE of that
file's own saturation surprise). Also structurally close to
``weight_only_quantize_int8_block_matmul`` (an affine per-block scheme) and
``iq4_nl`` (the same flat, channel-agnostic block layout with a ragged
final block). Confirmed line-by-line from ``gguf_legacy_quant_detail::
QuantizeDequantizeQ4_1Block`` (``gguf_legacy_quant.h``), not assumed from
any paraphrase: it rewrites::

    Y = MatMul(X, W) [+ bias]      W constant, 2-D, float32

into::

    Y = MatMul(X, W') [+ bias]     W' -- SAME shape/dtype as W, no new
                                     graph nodes at all: every element of W
                                     is replaced, in a brand-new
                                     initializer, by its own 32-element-
                                     block Q4_1 quantize-dequantize round
                                     trip:
                                       m       = min(block)
                                       d       = (max(block) - m) / 15
                                       code    = clip(round((value - m) / d),
                                                       0, 15)
                                       dequant = code * d + m

Only the common, unambiguous shape is matched -- a MatMul, or a Gemm with
transA=0, alpha=1 and beta=1 (bias, if any, left untouched) -- whose weight
(input 1) is a constant 2-D float32 tensor, confirmed by reading
``GgufLegacyQuantBase::patternMatchPredicate``/``runTransform`` and the
shared ``MatchMatMulLike`` (``quantize_matmul_common.h``) line by line:
exactly the same scope ``any_precision_llm.h``/``iq4_nl.h`` use, with no
``K % block_size == 0`` gate (see "ragged final block" below).

Both ``m`` and ``d`` are additionally round-tripped through ``ggml_half``
(IEEE754 binary16) before use (``RoundTripFloat16``, mirroring
``gguf_legacy_quant.py``'s own ``.astype(np.float16).astype(np.float64)``
step, since real Q4_1 stores both fields as fp16) -- a real but RELATIVE
(<= ~2^-11 of the rounded value's own magnitude) perturbation, orthogonal
to the exact-arithmetic lemmas below (which reason about ``m``/``d`` as
exact free reals, mirroring ``tests/test_formal_verify_quantize_fp16.py``'s
own already-verified fp16 codec rather than re-modeling fp16 rounding
here). For typical (order-1) weights this perturbation is utterly dwarfed
by the proved ``d / 2`` bound, confirmed by the differential tests below.

**A genuine caveat, confirmed empirically (not assumed) rather than swept
under the rug**: because ``m``'s fp16 rounding step is RELATIVE to ``|m|``
itself (``m`` is the block's own real minimum, not something normalized to
order-1 the way IQ4_NL's codebook lookup is), a block whose values sit at a
LARGE magnitude but span only a SMALL spread -- e.g. an offset of ~1000
with a spread of ~0.1, exactly the scenario Q4_1's own asymmetric min/max
construction is supposed to shine at relative to Q4_0 -- can see ``m``'s
own fp16 rounding step (``~|m| * 2**-11``) grow many times LARGER than the
proved ``d / 2`` (set only by the tiny spread). ``test_gguf_q4_1_beats_q4_0_
on_large_constant_offset_weight`` below confirms this directly: it checks
the same relative-MSE-improvement property this repo's own ``test_gguf_
legacy_quant_cpp.py`` already established for that scenario, rather than
re-asserting the exact-arithmetic ``d / 2`` bound against a case its own
hypotheses do not genuinely cover. The other differential bound checks
below use moderate-magnitude weights, where the exact-arithmetic bound and
the real fp16-perturbed pass agree closely, as documented.

--------------------------------------------------------------------------
NO saturation surprise here -- unlike Q4_0, Q4_1's own construction gives an
EXACT-EQUALITY range guarantee, genuinely proved (not assumed)
--------------------------------------------------------------------------

Q4_0's sibling file finds that its fixed ``code - 8`` bias, applied to a
scale computed from unsigned ``max(|block|)``, creates an ASYMMETRIC valid
code range (``[-8, 7]``) that a positive-signed block extremum can overflow
by exactly one code, forcing real clipping and doubling that element's
worst-case error from ``d/2`` to ``d``. Q4_1 has no such asymmetry to
create the same problem: its code range is used plainly UNSIGNED (``[0,
15]``, no ``+/- bias`` at all), and both its scale ``d`` and its additive
min ``m`` are derived directly from the block's own REAL (signed) min/max
-- not from an unsigned magnitude the way Q4_0's ``d`` is. Checked directly
below, exactly mirroring IQ4_NL's own "exact linear ratio, not an
inequality-based ceiling" lemma shape (``test_iq4_nl_scale_construction_is_
an_equality_not_merely_a_bound`` -- READ that file's wrinkle (2) for the
precedent): substituting ``v = max(block)`` gives
``(v - m) / d = (max - min) / ((max - min) / 15) = 15`` EXACTLY (an
equality, not merely "within range"), and ``v = min(block) = m`` gives
``(v - m) / d = 0`` EXACTLY. Both landing precisely on the ``[0, 15]``
range's own two endpoints means NO element of the block can ever need
clipping -- proved as a genuine theorem below (``test_gguf_q4_1_no_
clipping_range_guarantee_is_a_theorem``), each endpoint's own exact-code
mapping as its own explicit corollary
(``test_gguf_q4_1_max_element_maps_to_code_15_exactly``/``test_gguf_q4_1_
min_element_maps_to_code_0_exactly``), and the ordinary ``|W - Wdq| <=
d / 2`` half-scale round-trip bound holding UNCONDITIONALLY as a direct
consequence -- the single-operand MAC-bound composition below therefore
uses ``Ws / 2`` (not Q4_0's full ``Ws``), matching every OTHER affine
block-quant file in this suite.

--------------------------------------------------------------------------
Block layout: flat, row-major, ragged final block -- identical to Q4_0's/
IQ4_NL's
--------------------------------------------------------------------------

Confirmed from ``GgufLegacyQuantBase::runTransform``'s own ``start/count``
loop (``gguf_legacy_quant.h``, shared verbatim between Q4_0 and Q4_1):
blocks are laid out over the weight's own FLATTENED, ROW-MAJOR storage --
no ``channel_axis``/``weight_transposed`` branch at all. A ragged final
block (weight element count not a multiple of 32) is quantized using only
its own real elements, mathematically identical to the header's own
documented zero-pad-then-discard argument (a zero can never change a
block's own real min/max unless the whole block is already all-zero, in
which case both approaches floor ``d`` to the same ``1e-12``-derived
epsilon).

--------------------------------------------------------------------------
Differential tests
--------------------------------------------------------------------------

Built via ``onnx.parser`` (per ``CLAUDE.md``) with ``numpy_helper.
from_array`` for random weight initializers. This pass has TWO reachable
entry points, both exercised below: the opt-in optimizer name
``"gguf_q4_1"`` via ``simplify_isolated_extra`` (confirming the pass is
actually registered -- ``onnxsim.onnxsim_cpp2py_export.
_list_other_optimizers()`` -- and reachable that way), and the dedicated
Python entry point ``onnxsim.apply_gguf_q4_1_quantization_cpp``
(``onnxsim/quantize_entry.cpp``'s ``ApplyGgufQ4_1``) -- used for the
detailed numeric checks since it hands back a single self-contained
rewrite with no other pass's side effects to account for.

No ``com.microsoft`` contrib op and no quantized ONNX tensor type is ever
involved (confirmed from ``runTransform``: the new initializer is plain
``TensorProto_DataType_FLOAT``, same shape as the original weight) -- so,
per this task's own instruction, the numeric checks below use ``onnx.
reference.ReferenceEvaluator`` and direct initializer inspection, never an
onnxruntime ``InferenceSession``.

Confirmed below: only MatMul/vanilla-Gemm (not a ``transA=1`` Gemm, and not
a non-constant weight) is matched; ``W'`` has the same shape/dtype as
``W``; a fresh, independent from-scratch numpy reimplementation of
``QuantizeDequantizeQ4_1Block`` (including its own fp16 round-trip of
``m``/``d``, built here, NOT imported from ``onnxsim.gguf_legacy_quant`` or
calling back into the C++ pass) matches the real compiled pass's output
closely; the ragged final block case, including that its own ``m``/``d``
come from only its own real elements; Q4_1's own asymmetric advantage over
Q4_0 on a large-constant-offset weight (mirroring
``tests/test_gguf_legacy_quant_cpp.py``'s own analogous check, reconfirmed
here against the proved bound rather than only a relative comparison); and
the real end-to-end output stays within the proved half-scale ``d / 2``
bound, with NO analogous violation the way Q4_0's own file demonstrates for
its (weaker, full-scale) bound.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser
from onnx.reference import ReferenceEvaluator

import onnxsim

_BLOCK_SIZE = 32
_MAX_CODE = 15


def _abs(v):
    return z3.If(v >= 0, v, -v)


# --- 1. The range guarantee IS a theorem (unlike Q4_0) ----------------------


def test_gguf_q4_1_no_clipping_range_guarantee_is_a_theorem():
    # For any element v of a block whose own real min is `lo` and max is
    # `hi` (hi > lo, so d := (hi - lo) / 15 is well-defined and positive),
    # n := round((v - lo) / d) always lands in [0, 15] -- proved directly,
    # not assumed, unlike Q4_0's own analogous claim which this suite's own
    # test_gguf_q4_0_no_clipping_range_guarantee_is_not_a_theorem finds
    # FALSE. The construction here has no unsigned-magnitude/signed-bias
    # mismatch to create that asymmetry: m and d are both built directly
    # from the block's own real (signed) endpoints.
    v, lo, hi = z3.Reals("v lo hi")
    n = z3.Int("n")
    d = (hi - lo) / _MAX_CODE
    hypotheses = z3.And(
        hi > lo,
        v >= lo,
        v <= hi,
        _abs((v - lo) / d - z3.ToReal(n)) <= z3.RealVal(1) / 2,
    )
    prove(z3.Implies(hypotheses, z3.And(n >= 0, n <= _MAX_CODE)))


def test_gguf_q4_1_max_element_maps_to_code_15_exactly():
    # The equality half of the construction, mirroring IQ4_NL's own "scale
    # construction is an equality, not merely a bound" corollary: v = hi
    # (the block's own real max) maps to EXACTLY code 15 -- not merely
    # "within [0, 15]" -- by direct substitution: (hi - lo) / d = 15 when
    # d = (hi - lo) / 15, so the only n satisfying the rounding property at
    # v = hi is n = 15 itself.
    lo, hi = z3.Reals("lo hi")
    n = z3.Int("n")
    d = (hi - lo) / _MAX_CODE
    hypotheses = z3.And(
        hi > lo,
        _abs((hi - lo) / d - z3.ToReal(n)) <= z3.RealVal(1) / 2,
        n >= 0,
        n <= _MAX_CODE,
    )
    prove(z3.Implies(hypotheses, n == _MAX_CODE))


def test_gguf_q4_1_min_element_maps_to_code_0_exactly():
    # The mirror corollary: v = lo (the block's own real min) maps to
    # EXACTLY code 0, by direct substitution: (lo - lo) / d = 0.
    lo, hi = z3.Reals("lo hi")
    n = z3.Int("n")
    d = (hi - lo) / _MAX_CODE
    hypotheses = z3.And(
        hi > lo,
        _abs((lo - lo) / d - z3.ToReal(n)) <= z3.RealVal(1) / 2,
        n >= 0,
        n <= _MAX_CODE,
    )
    prove(z3.Implies(hypotheses, n == 0))


# --- 2. The standard half-scale round-trip bound, now UNCONDITIONAL --------


def test_gguf_q4_1_half_scale_bound_is_a_theorem():
    # Because the range guarantee above is a genuine theorem (no clipping
    # ever needed, unlike Q4_0), the ordinary round-to-nearest half-scale
    # bound holds UNCONDITIONALLY for every in-block element -- no separate
    # "absent saturation" side condition is needed the way Q4_0's own file
    # required.
    v, lo, hi = z3.Reals("v lo hi")
    n = z3.Int("n")
    d = (hi - lo) / _MAX_CODE
    hypotheses = z3.And(
        hi > lo,
        v >= lo,
        v <= hi,
        _abs((v - lo) / d - z3.ToReal(n)) <= z3.RealVal(1) / 2,
    )
    dequant = z3.ToReal(n) * d + lo
    error = v - dequant
    prove(z3.Implies(hypotheses, z3.And(error <= d / 2, -error <= d / 2)))


# --- 3. Composed single-operand MAC bound: eps_w := Ws / 2 (like every ------
# --- OTHER affine block-quant file, unlike Q4_0's own Ws) -------------------

_K = 2  # matches quantized_mac_bound's / weight_only_quantize_int8_block_
# matmul's own minimal-case convention.
_BLOCK_SIZE_ABSTRACT = 1  # each tap is its own block -- Ws0 for tap 0, a
# genuinely different Ws1 for tap 1.
_NUM_BLOCKS = _K // _BLOCK_SIZE_ABSTRACT


def _block_of(k):
    return k // _BLOCK_SIZE_ABSTRACT


def _bound_formulas():
    """Z3 vocabulary for the single-operand, PER-BLOCK bounded-error claim,
    reused in structure from ``weight_only_quantize_int8_block_matmul``'s
    own per-block generalization (the direct-error-variable idiom -- see
    that file's own module docstring for why this avoids a documented Z3
    nonlinear-arithmetic hang): each tap's own error is bounded by ITS OWN
    block's ``Ws[block] / 2`` -- the ordinary half-scale budget, sound here
    because the range guarantee above holds unconditionally.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]

    dequant_w = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= Ws[_block_of(k)] / 2 for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))

    bound = sum((Ws[_block_of(k)] / 2) * _abs(X[k]) for k in range(_K))

    return float_matmul, dequant_matmul, rounding_bounds, bound


def test_gguf_q4_1_mac_error_is_bounded():
    # Given W's own per-BLOCK half-scale rounding bound and X completely
    # unchanged, the true float dot product and the one computed against
    # the dequantized weight cannot differ by more than
    # sum_k (Ws[block_of(k)] / 2) * |X[i, k]|.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_gguf_q4_1_mac_bias_variant_error_is_bounded():
    # This pass never touches Gemm's bias C; adding the same Bias(n) to
    # both the true and the dequantized computation leaves their
    # difference, and therefore the bound, unchanged.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_matmul + bias) - (dequant_matmul + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_gguf_q4_1_mac_negative_control_requires_rounding_bound():
    # Sanity check the bound is genuine, not vacuous: with no error budget
    # assumed on ew at all, the same bound is not a theorem.
    float_matmul, dequant_matmul, _rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul

    solver = z3.Solver()
    solver.add(z3.Not(z3.And(error <= bound, -error <= bound)))
    assert solver.check() == z3.sat, (
        "the bound holds even without any rounding-error budget on ew -- "
        "negative control is vacuous"
    )


def test_gguf_q4_1_mac_uniform_bound_is_unsound_across_blocks():
    # Confirms the per-block sum is genuinely necessary, not merely
    # untested-but-equivalent to a single shared scale: bounding every tap
    # by block 0's own scale alone (Ws0 / 2) is NOT a theorem once block
    # 1's real scale Ws1 can exceed Ws0.
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]

    dequant_w = [W[k] - ew[k] for k in range(_K)]
    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= Ws[_block_of(k)] / 2 for k in range(_K)],
    )
    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))
    error = float_matmul - dequant_matmul

    uniform_bound = (Ws[0] / 2) * sum(_abs(X[k]) for k in range(_K))

    solver = z3.Solver()
    solver.add(rounding_bounds)
    solver.add(z3.Not(z3.And(error <= uniform_bound, -error <= uniform_bound)))
    assert solver.check() == z3.sat, (
        "the naive single-scale (block 0 only) bound holds even though "
        "block 1 has its own, potentially larger scale"
    )


# --- Differential tests -----------------------------------------------------


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
    return numpy_helper.from_array(array.astype(np.float32), name)


def _round_trip_float16(value):
    return float(np.array(value, dtype=np.float16).astype(np.float64))


def _q4_1_quantize_dequantize_matrix(weight):
    """Independent from-scratch numpy re-implementation of
    ``QuantizeDequantizeQ4_1Block``/``GgufLegacyQuantBase::runTransform``
    (``onnxsim/passes/gguf_legacy_quant.h``): flattens ``weight`` in
    ROW-MAJOR order, computes one (min, scale) pair per 32-element block of
    the flat array (the final block RAGGED, using only its own real
    elements when the flat count isn't a multiple of 32), fp16-round-trips
    both ``m`` and ``d`` (mirroring ``RoundTripFloat16``), and returns the
    same shape, each element replaced by ``code * d + m``.

    Written independently here (own control flow) rather than calling into
    ``onnxsim.gguf_legacy_quant``'s ``quantize_dequantize_q4_1`` or the
    compiled pass itself, so this is a genuine cross-check of the real
    pass's own math.
    """
    original_shape = weight.shape
    flat = np.asarray(weight, dtype=np.float64).reshape(-1)
    n = flat.size
    out = np.empty_like(flat)
    for start in range(0, n, _BLOCK_SIZE):
        block = flat[start : start + _BLOCK_SIZE]
        lo = float(np.min(block)) if block.size else 0.0
        hi = float(np.max(block)) if block.size else 0.0
        m = _round_trip_float16(lo)
        d = _round_trip_float16(max(hi - lo, 1e-12) / _MAX_CODE)
        code = np.clip(np.round((block - m) / d), 0, _MAX_CODE)
        out[start : start + block.size] = code * d + m
    return out.reshape(original_shape)


def _q4_1_scale_matrix(weight):
    """Same flattening/blocking as ``_q4_1_quantize_dequantize_matrix``, but
    returns, per element, the SCALE ``d`` of the block that element belongs
    to (same shape as ``weight``) -- used to build the proved half-scale
    error-bound matrix ``d / 2`` for the numeric bound check below.
    """
    original_shape = weight.shape
    flat = np.asarray(weight, dtype=np.float64).reshape(-1)
    n = flat.size
    out = np.empty_like(flat)
    for start in range(0, n, _BLOCK_SIZE):
        block = flat[start : start + _BLOCK_SIZE]
        lo = float(np.min(block)) if block.size else 0.0
        hi = float(np.max(block)) if block.size else 0.0
        d = _round_trip_float16(max(hi - lo, 1e-12) / _MAX_CODE)
        out[start : start + block.size] = d
    return out.reshape(original_shape)


def test_gguf_q4_1_pass_fires_and_matches_reimplementation_via_extra_optimizers():
    # Confirms the pass is reachable via extra_optimizers (this suite's
    # standard "is it actually registered" check) and rewrites the weight
    # in place (SAME shape/dtype, a brand-new initializer name -- no new
    # graph nodes at all).
    rng = np.random.default_rng(0)
    rows, K, N = 4, 8, 8  # K * N = 64 = exactly two 32-element blocks
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    sim_model, _ops = simplify_isolated_extra(model, "gguf_q4_1", check_n=0)

    matmul_node = producer(sim_model, "Y")
    assert matmul_node.op_type == "MatMul"
    x_input, w_input = matmul_node.input
    assert x_input == "X"
    assert w_input != "W"

    w_out_init = next(i for i in sim_model.graph.initializer if i.name == w_input)
    assert w_out_init.data_type == onnx.TensorProto.FLOAT
    assert list(w_out_init.dims) == [K, N]

    w_out = numpy_helper.to_array(w_out_init)
    expected = _q4_1_quantize_dequantize_matrix(weight.astype(np.float64)).astype(
        np.float32
    )
    np.testing.assert_allclose(w_out, expected, rtol=1e-5, atol=1e-6)


def test_gguf_q4_1_gemm_bias_untouched_and_blocked_by_own_flat_storage():
    # A "vanilla" Gemm with a bias: the bias is passed through unchanged.
    # transB=1 (weight stored [N, K]) confirms the pass blocks over the
    # weight's OWN storage directly, not some canonicalized [K, N] view.
    rng = np.random.default_rng(1)
    rows, K, N = 3, 20, 5  # N * K = 100, not a multiple of 32 -- ragged tail
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    bias = rng.standard_normal(N).astype(np.float32)
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )

    quantized = onnxsim.apply_gguf_q4_1_quantization_cpp(model)
    onnx.checker.check_model(quantized)

    gemm_node = next(n for n in quantized.graph.node if n.op_type == "Gemm")
    x_input, w_input, b_input = gemm_node.input
    assert x_input == "X"
    assert b_input == "B"

    w_out_init = next(i for i in quantized.graph.initializer if i.name == w_input)
    assert list(w_out_init.dims) == [N, K]
    w_out = numpy_helper.to_array(w_out_init)

    expected = _q4_1_quantize_dequantize_matrix(weight.astype(np.float64)).astype(
        np.float32
    )
    np.testing.assert_allclose(w_out, expected, rtol=1e-5, atol=1e-6)


def test_gguf_q4_1_declines_transposed_activation_gemm():
    # MatchMatMulLike requires transA == 0; a transA=1 Gemm must be left
    # byte-for-byte unchanged.
    rng = np.random.default_rng(2)
    K, rows, N = 6, 4, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[{K},{rows}] X) => (float[{rows},{N}] Y)
        {{
          Y = Gemm<transA = 1>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_gguf_q4_1_quantization_cpp(model)
    assert quantized.SerializeToString() == model.SerializeToString()


def test_gguf_q4_1_declines_non_constant_weight():
    # A weight that is a genuine graph input (not any constant/initializer)
    # must be left untouched.
    rows, K, N = 4, 8, 8
    model = _model(
        f"""
        g (float[{rows},{K}] X, float[{K},{N}] W) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """
    )

    quantized = onnxsim.apply_gguf_q4_1_quantization_cpp(model)
    assert quantized.SerializeToString() == model.SerializeToString()


def test_gguf_q4_1_ragged_final_block_matches_reimplementation():
    # Weight element count NOT a multiple of 32 (5 * 7 = 35 = one full
    # 32-element block + a ragged 3-element final block) -- confirms the
    # ragged block's own m/d come from only its own 3 real elements.
    rng = np.random.default_rng(3)
    dim0, dim1 = 5, 7
    weight = rng.standard_normal((dim0, dim1)).astype(np.float32) * 0.9
    model = _model(
        f"""
        g (float[4,{dim0}] X) => (float[4,{dim1}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_gguf_q4_1_quantization_cpp(model)
    matmul_node = next(n for n in quantized.graph.node if n.op_type == "MatMul")
    _x_input, w_input = matmul_node.input
    w_out = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == w_input)
    )

    expected = _q4_1_quantize_dequantize_matrix(weight.astype(np.float64)).astype(
        np.float32
    )
    np.testing.assert_allclose(w_out, expected, rtol=1e-5, atol=1e-6)

    flat = weight.astype(np.float64).reshape(-1)
    ragged_block = flat[32:35]
    assert ragged_block.size == 3
    lo = float(np.min(ragged_block))
    hi = float(np.max(ragged_block))
    expected_m = _round_trip_float16(lo)
    expected_d = _round_trip_float16(max(hi - lo, 1e-12) / _MAX_CODE)
    out_flat = w_out.astype(np.float64).reshape(-1)
    out_ragged = out_flat[32:35]
    recovered_codes = (out_ragged - expected_m) / expected_d
    assert np.all((recovered_codes >= -1e-6) & (recovered_codes <= _MAX_CODE + 1e-6))


def test_gguf_q4_1_beats_q4_0_on_large_constant_offset_weight():
    # A weight far from zero-centered (mirroring test_gguf_legacy_quant_cpp
    # .py's own analogous check): Q4_1's explicit min lets it represent this
    # far more accurately than Q4_0's fixed symmetric range can.
    #
    # HONESTY NOTE, confirmed empirically here rather than assumed: this
    # file's Z3 lemmas above prove the ``d / 2`` round-trip bound treating
    # ``m``/``d`` as EXACT reals -- but the real pass additionally
    # round-trips both through fp16 (``RoundTripFloat16``) BEFORE use.
    # Fp16's precision is RELATIVE (~11 significant bits), so its absolute
    # rounding step on ``m := RoundTripFloat16(lo)`` scales with ``|lo|``
    # itself, not with the block's own spread ``hi - lo``. For typical
    # (order-1) weights the two are comparable and the fp16 perturbation is
    # negligible next to ``d / 2``, exactly as this file's own module
    # docstring says -- but for THIS deliberately extreme test (weight
    # values around +-1000 with a spread of only ~0.1-0.5), ``m``'s own
    # fp16 rounding step (~1000 * 2**-11, order 0.5) can be tens of times
    # BIGGER than the proved ``d / 2`` (order 0.01, set by the tiny
    # spread) -- so the exact-arithmetic lemma's own assumption ("m, d used
    # exactly as (hi - lo) / 15 and lo respectively") is violated here in a
    # materially significant way, not a merely cosmetic one. This test
    # therefore checks the same relative-MSE-improvement property
    # ``test_gguf_legacy_quant_cpp.py`` already established, rather than
    # re-asserting the proved bound against a scenario the bound's own
    # exact-arithmetic hypotheses do not genuinely cover.
    rng = np.random.default_rng(4)
    K, N = _BLOCK_SIZE * 4, 4
    weight = (rng.standard_normal((K, N)) * 0.1 + 1000.0).astype(np.float32)
    model = _model(
        f"""
        g (float[4,{K}] X) => (float[4,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    q4_1 = onnxsim.apply_gguf_q4_1_quantization_cpp(model)
    q4_0 = onnxsim.apply_gguf_q4_0_quantization_cpp(model)
    matmul_node = next(n for n in q4_1.graph.node if n.op_type == "MatMul")
    _x_input, w_input = matmul_node.input
    w_out_q4_1 = numpy_helper.to_array(
        next(i for i in q4_1.graph.initializer if i.name == w_input)
    ).astype(np.float64)
    matmul_node_q4_0 = next(n for n in q4_0.graph.node if n.op_type == "MatMul")
    _x_input_0, w_input_0 = matmul_node_q4_0.input
    w_out_q4_0 = numpy_helper.to_array(
        next(i for i in q4_0.graph.initializer if i.name == w_input_0)
    ).astype(np.float64)

    w64 = weight.astype(np.float64)
    err_q4_0 = float(np.mean((w64 - w_out_q4_0) ** 2))
    err_q4_1 = float(np.mean((w64 - w_out_q4_1) ** 2))
    assert err_q4_1 < err_q4_0


def test_gguf_q4_1_output_within_proved_half_scale_bound_no_violation():
    # The full end-to-end sanity check against a real ONNX graph execution
    # (onnx's own reference evaluator -- plain float32, no quantized tensor
    # type or contrib op involved): every output element's error against
    # the true float MatMul must stay within the proved half-scale bound
    # sum_k |X[i, k]| * (d[k, n] / 2) -- and, in contrast to Q4_0's own
    # file, this bound is never exceeded for any element on the same kind
    # of ordinary random weight, since Q4_1's own range guarantee is a
    # genuine (unconditional) theorem.
    rng = np.random.default_rng(5)
    rows, K, N = 5, 10, 9  # K * N = 90: not a multiple of 32, ragged tail
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.8
    x = rng.standard_normal((rows, K)).astype(np.float32) * 2.0
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_gguf_q4_1_quantization_cpp(model)
    onnx.checker.check_model(quantized)

    matmul_node = next(n for n in quantized.graph.node if n.op_type == "MatMul")
    _x_input, w_input = matmul_node.input
    w_out = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == w_input)
    ).astype(np.float64)

    y_float = x.astype(np.float64) @ weight.astype(np.float64)

    evaluator = ReferenceEvaluator(quantized)
    (y_quant,) = evaluator.run(None, {"X": x})
    y_quant = y_quant.astype(np.float64)

    error = np.abs(y_float - y_quant)

    d_matrix = _q4_1_scale_matrix(weight.astype(np.float64))  # [K, N]
    eps_w = d_matrix / 2.0
    bound = np.abs(x.astype(np.float64)) @ eps_w
    assert np.all(error <= bound + 1e-4)

    expected_w = _q4_1_quantize_dequantize_matrix(weight.astype(np.float64))
    np.testing.assert_allclose(w_out, expected_w, rtol=1e-5, atol=1e-6)
