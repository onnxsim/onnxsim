"""Formal check for GgufQ4_0 (opt-in; onnxsim's own
``onnxsim/passes/gguf_legacy_quant.h``, C++ port of
``onnxsim.gguf_legacy_quant``'s own ``apply_gguf_q4_0_quantization``):
llama.cpp's legacy GGUF "Q4_0" block-quant format, the SIMPLEST member of
the format family this suite already covers via ``gguf_kquant``/``iq4_nl``
-- a plain, uncalibrated, SYMMETRIC 4-bit AFFINE quantizer (no
super-block/sub-block structure, no fixed codebook), so ``weight_only_
quantize_int8_block_matmul`` (READ THAT FILE FIRST -- the closest affine,
block-wise structural precedent) and ``iq4_nl`` (READ THAT FILE SECOND --
same flat, channel-agnostic block layout with a ragged final block) are
this file's two closest precedents, combined. Confirmed line-by-line from
``gguf_legacy_quant_detail::QuantizeDequantizeQ4_0Block`` (``gguf_legacy_
quant.h``), not assumed from any paraphrase: it rewrites::

    Y = MatMul(X, W) [+ bias]      W constant, 2-D, float32

into::

    Y = MatMul(X, W') [+ bias]     W' -- SAME shape/dtype as W, no new
                                     graph nodes at all (like IQ4_NL, unlike
                                     the DequantizeLinear-based INT8/INT4
                                     block passes): every element of W is
                                     replaced, in a brand-new initializer,
                                     by its own 32-element-block Q4_0
                                     quantize-dequantize round trip:
                                       d       = max(|block|) / 8
                                       code    = clip(round(value / d) + 8,
                                                       0, 15)
                                       dequant = (code - 8) * d

Only the common, unambiguous shape is matched -- a MatMul, or a Gemm with
transA=0, alpha=1 and beta=1 (bias, if any, left untouched) -- whose weight
(input 1) is a constant 2-D float32 tensor, confirmed by reading
``GgufLegacyQuantBase::patternMatchPredicate``/``runTransform`` and the
shared ``MatchMatMulLike`` (``quantize_matmul_common.h``) line by line:
exactly the same scope ``any_precision_llm.h``/``iq4_nl.h`` use, with no
``K % block_size == 0`` gate (see "ragged final block" below).

``d`` is additionally round-tripped through ``ggml_half`` (IEEE754
binary16) before use (``RoundTripFloat16``, mirroring
``gguf_legacy_quant.py``'s own ``.astype(np.float16).astype(np.float64)``
step, since real Q4_0 stores its scale as fp16) -- a real but tiny
(<= ~2^-11 relative) perturbation of the *value* of ``d`` versus the exact
``max(|block|) / 8``, orthogonal to, and utterly dwarfed by, the far
larger structural finding below. This file's Z3 lemmas reason about ``d``
as an exact free real (mirroring the fp16 codec's own already-verified
status elsewhere in this repo -- ``tests/test_formal_verify_quantize_fp16.
py`` -- rather than re-modeling fp16 rounding here); the differential
tests, which exercise the real compiled pass, naturally include this
perturbation and confirm it changes nothing material.

--------------------------------------------------------------------------
THE SURPRISE: Q4_0's own asymmetric code range makes the naive "d/2"
round-trip bound UNSOUND -- the correct universal per-element bound is the
FULL scale ``d``, not half of it
--------------------------------------------------------------------------

Every other affine block-quant file in this suite (``weight_only_quantize_
int8_block_matmul``, ``_int4_matmul``, ``_matmul_nbits``, ...) proves a
clean ``|W - Wdq| <= scale / 2`` bound and, as a *separate* lemma, that the
block's own max-based scale keeps every element's code within its valid
range before rounding/clipping ever needs to trigger -- so clipping is a
structurally impossible corner case there, never actually invoked. This
task's own instructions anticipated the same might be checked here for
Q4_0's "code - 8" centering. It is checked below, directly in Z3 rather
than assumed -- and it turns out FALSE in general:

Q4_0's 4-bit code ``[0, 15]``, shifted by the fixed bias 8, represents the
*signed* range ``code - 8 in [-8, 7]`` -- an ASYMMETRIC range (one more
negative code than positive), the ordinary two's-complement-style
asymmetry of an N-bit signed integer. But ``d`` is computed from
``max(|block|)`` -- the block's largest *magnitude*, blind to sign. If the
element achieving that magnitude is POSITIVE (call it ``v = max_abs`` for
concreteness), then ``round(v / d) = round(8) = 8`` EXACTLY (no rounding
ambiguity: ``v / d = max_abs / (max_abs / 8) = 8`` on the nose), giving
``code_raw = 8 + 8 = 16`` -- one past the valid range's top, ``kMaxCode =
15``, forcing ``std::min(..., 15)`` to actually clip. The dequantized
result is then ``(15 - 8) * d = 7d = 0.875 * max_abs``, NOT ``max_abs`` --
an error of exactly ``d`` (not ``d/2``) for that element. (The mirror case,
the block's magnitude-maximizing element being NEGATIVE, maps to
``code = 0`` exactly -- already the valid range's bottom, so no clipping is
needed there, and that element round-trips EXACTLY. The asymmetry only
bites on the positive side -- confirmed directly below, not merely
asserted.)

This is not a rare, adversarially-constructed edge case: it is invoked by
ANY block whose largest-magnitude element happens to be positive --
routinely about half of all blocks for realistic (roughly sign-symmetric)
weight data, confirmed empirically in the differential tests below via a
constructed positive-extreme block (near-``d``-sized error) contrasted with
a constructed negative-extreme block (exactly-zero error at that element).
``test_gguf_q4_0_no_clipping_range_guarantee_is_not_a_theorem`` below finds
this exact ``code_raw = 16`` witness with Z3 (mirroring this suite's own
"confirmed, not assumed" discipline for a claim that turns out FALSE, e.g.
``test_iq4_nl_codebook_unrestricted_domain_claim_is_false``'s own analogous
shape), and ``test_gguf_q4_0_half_scale_bound_is_not_a_theorem`` confirms
the direct consequence: the naive ``d/2`` bound is NOT valid in general.

What IS universally true, proved directly below accounting for the ``[0,
15]`` clamp explicitly (not assuming it is never reached):
``|W - Wdq| <= d`` for every element, tight (the positive-extreme element
above realizes exactly ``d``, not less). A companion conditional lemma
confirms the "expected" ``d/2`` bound DOES hold whenever the clamp truly
isn't invoked -- so the naive bound is not simply wrong everywhere, only at
this specific, real, unavoidable saturation boundary. The single-operand
MAC-bound composition below therefore uses the full scale ``Ws`` (NOT
``Ws / 2``, unlike every other affine block-quant file in this suite) as
each tap's own per-block error budget -- proved sound -- with a dedicated
negative control confirming the naive ``Ws / 2`` per-tap MAC bound is
UNSOUND, the format-specific analogue of this suite's usual "uniform bound
is unsound across blocks" control.

--------------------------------------------------------------------------
Block layout: flat, row-major, ragged final block -- identical to IQ4_NL's
--------------------------------------------------------------------------

Confirmed from ``GgufLegacyQuantBase::runTransform``'s own ``start/count``
loop (``gguf_legacy_quant.h``): blocks are laid out over the weight's own
FLATTENED, ROW-MAJOR storage -- there is no ``channel_axis``/``weight_
transposed`` branch at all, exactly ``iq4_nl.h``'s own layout (see that
file's own wrinkle (3)), NOT the per-(output-channel, K-block) layout
``weight_only_quantize_int8_block_matmul``/``_int4_matmul`` use. A ragged
final block (weight element count not a multiple of 32) is quantized using
only its own real elements (``count = std::min(kBlockSize, numel -
start)``), mathematically identical to the header's own documented
zero-pad-then-discard argument (a zero can never be a block's own
``max(|.|)`` unless the whole block is already all-zero).

--------------------------------------------------------------------------
Differential tests
--------------------------------------------------------------------------

Built via ``onnx.parser`` (per ``CLAUDE.md``) with ``numpy_helper.
from_array`` for random weight initializers. This pass has TWO reachable
entry points, both exercised below, mirroring ``test_formal_verify_iq4_nl.
py``'s own convention: the opt-in optimizer name ``"gguf_q4_0"`` via
``simplify_isolated_extra`` (confirming the pass is actually registered --
``onnxsim.onnxsim_cpp2py_export._list_other_optimizers()`` -- and reachable
that way), and the dedicated Python entry point
``onnxsim.apply_gguf_q4_0_quantization_cpp`` (``onnxsim/quantize_entry.
cpp``'s ``ApplyGgufQ4_0``) -- used for the detailed numeric checks since it
hands back a single self-contained rewrite with no other pass's side
effects to account for.

No ``com.microsoft`` contrib op and no quantized ONNX tensor type is ever
involved (confirmed from ``runTransform``: the new initializer is plain
``TensorProto_DataType_FLOAT``, same shape as the original weight) -- so,
per this task's own instruction, the numeric checks below use ``onnx.
reference.ReferenceEvaluator`` and direct initializer inspection, never an
onnxruntime ``InferenceSession``.

Confirmed below: only MatMul/vanilla-Gemm (not a ``transA=1`` Gemm, and not
a non-constant weight) is matched; ``W'`` has the same shape/dtype as
``W``; a fresh, independent from-scratch numpy reimplementation of
``QuantizeDequantizeQ4_0Block`` (including its own fp16 round-trip of
``d``, built here, NOT imported from ``onnxsim.gguf_legacy_quant`` or
calling back into the C++ pass) matches the real compiled pass's output
closely; the ragged final block case; a value of exactly 0 always
round-trips to exactly 0 (Q4_0 is symmetric, zero-point-free); the
surprising positive-vs-negative extreme-element asymmetry, demonstrated
directly against the real compiled pass, not just the Python reference;
and the real end-to-end output stays within the proved full-scale ``d``
bound (and, separately, is shown to occasionally EXCEED the naive ``d/2``
bound the way every other affine block-quant file's own output never
does).
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
_Q4_0_BIAS = 8


def _abs(v):
    return z3.If(v >= 0, v, -v)


# --- 1. The surprise: the naive "no clipping" range guarantee is FALSE ------


def test_gguf_q4_0_no_clipping_range_guarantee_is_not_a_theorem():
    # Every other affine block-quant file in this suite proves its own
    # max-based scale keeps every element's pre-clip code within its valid
    # range -- checked here for Q4_0 rather than assumed, and found FALSE:
    # a real, satisfiable witness has code_raw = round(v/d) + 8 landing at
    # 16, one past kMaxCode = 15, even though v is a perfectly ordinary
    # in-block element (|v| <= 8d = max_abs). Z3 finds this rather than it
    # being merely asserted -- mirroring test_iq4_nl_codebook_unrestricted_
    # domain_claim_is_false's own "confirm the surprising negative finding"
    # discipline.
    v, d = z3.Reals("v d")
    n = z3.Int("n")  # n := round(v / d), modeled by its defining property
    hypotheses = z3.And(
        d > 0,
        v >= -_Q4_0_BIAS * d,
        v <= _Q4_0_BIAS * d,  # |v| <= max_abs = 8d, the block's own scale invariant
        _abs(v / d - z3.ToReal(n)) <= z3.RealVal(1) / 2,
    )
    code_raw = n + _Q4_0_BIAS

    solver = z3.Solver()
    solver.add(hypotheses)
    solver.add(z3.Not(z3.And(code_raw >= 0, code_raw <= _MAX_CODE)))
    assert solver.check() == z3.sat, (
        "code_raw = round(v/d) + 8 stays within [0, 15] for every in-block "
        "v -- Q4_0's max-abs-based scale would then never need to clip, "
        "contradicting this file's own documented surprise"
    )
    model = solver.model()
    # The witness should indeed be the positive-extreme boundary: n = 8
    # (round(v/d) = 8), i.e. v sits at (or essentially at) +max_abs.
    assert model[n].as_long() == _Q4_0_BIAS


def test_gguf_q4_0_negative_extreme_needs_no_clipping():
    # The mirror-image confirmation: the SAME construction, but forcing the
    # block's magnitude-maximizing element to be the NEGATIVE one (v = -8d)
    # -- code_raw = round(-8) + 8 = 0 exactly, already the valid range's own
    # bottom. No clip is ever needed on this side, so the asymmetry
    # genuinely only bites the positive extreme, not both.
    v, d = z3.Reals("v d")
    n = z3.Int("n")
    hypotheses = z3.And(
        d > 0,
        v == -_Q4_0_BIAS * d,
        _abs(v / d - z3.ToReal(n)) <= z3.RealVal(1) / 2,
    )
    code_raw = n + _Q4_0_BIAS
    prove(z3.Implies(hypotheses, z3.And(code_raw >= 0, code_raw <= _MAX_CODE)))


# --- 2. The correct universal bound: |v - dequant| <= d (not d/2) ----------


def _q4_0_round_trip(v, d, n):
    """Builds (code, dequant) for one element from Z3 vocabulary: n :=
    round(v / d) (a free Int constrained by its defining |.| <= 1/2
    property at the call site), clamped exactly as QuantizeDequantizeQ4_0Block
    does (``std::min(std::max(code, 0.0), 15.0)`` applied to an already-
    integer ``code``).
    """
    code_raw = n + _Q4_0_BIAS
    code = z3.If(code_raw < 0, 0, z3.If(code_raw > _MAX_CODE, _MAX_CODE, code_raw))
    dequant = (z3.ToReal(code) - _Q4_0_BIAS) * d
    return dequant


def test_gguf_q4_0_full_scale_bound_is_a_theorem():
    # The genuinely correct, UNCONDITIONAL per-element bound, modeling the
    # [0, 15] clamp explicitly rather than assuming it is unreachable: for
    # every v with |v| <= max_abs = 8d (the block's own scale invariant),
    # |v - dequant| <= d. This is what test_gguf_q4_0_no_clipping_range_
    # guarantee_is_not_a_theorem's own witness makes necessary -- d/2 would
    # be violated there, but the full scale d is not.
    v, d = z3.Reals("v d")
    n = z3.Int("n")
    hypotheses = z3.And(
        d > 0,
        v >= -_Q4_0_BIAS * d,
        v <= _Q4_0_BIAS * d,
        _abs(v / d - z3.ToReal(n)) <= z3.RealVal(1) / 2,
    )
    dequant = _q4_0_round_trip(v, d, n)
    error = v - dequant
    prove(z3.Implies(hypotheses, z3.And(error <= d, -error <= d)))


def test_gguf_q4_0_half_scale_bound_is_not_a_theorem():
    # The direct consequence of the surprise above: the "usual" half-scale
    # uniform-quantizer bound this suite's other affine block-quant files
    # all enjoy is genuinely NOT valid for Q4_0 -- Z3 must find a real
    # counterexample (not merely fail to prove it), using the exact same
    # hypotheses as the full-scale theorem above.
    v, d = z3.Reals("v d")
    n = z3.Int("n")
    hypotheses = z3.And(
        d > 0,
        v >= -_Q4_0_BIAS * d,
        v <= _Q4_0_BIAS * d,
        _abs(v / d - z3.ToReal(n)) <= z3.RealVal(1) / 2,
    )
    dequant = _q4_0_round_trip(v, d, n)
    error = v - dequant

    solver = z3.Solver()
    solver.add(hypotheses)
    solver.add(z3.Not(z3.And(error <= d / 2, -error <= d / 2)))
    assert solver.check() == z3.sat, (
        "|v - dequant| <= d/2 holds for every in-block v -- Q4_0 would then "
        "have no saturation asymmetry at all, contradicting this file's own "
        "documented surprise"
    )


def test_gguf_q4_0_half_scale_bound_holds_absent_saturation():
    # Q4_0's naive bound is not simply "wrong everywhere" -- it is exactly
    # right whenever the [0, 15] clamp truly isn't invoked (code_raw already
    # in range), the ordinary round-to-nearest argument every other affine
    # file in this suite relies on. This isolates precisely which half of
    # the story is Q4_0-specific (the boundary) and which is the usual
    # uniform-quantizer algebra (everywhere else).
    v, d = z3.Reals("v d")
    n = z3.Int("n")
    code_raw = n + _Q4_0_BIAS
    hypotheses = z3.And(
        d > 0,
        _abs(v / d - z3.ToReal(n)) <= z3.RealVal(1) / 2,
        code_raw >= 0,
        code_raw <= _MAX_CODE,  # no clip needed -- the excluded case above
    )
    dequant = (z3.ToReal(code_raw) - _Q4_0_BIAS) * d
    error = v - dequant
    prove(z3.Implies(hypotheses, z3.And(error <= d / 2, -error <= d / 2)))


# --- 3. Composed single-operand MAC bound: eps_w := Ws (full scale) --------

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
    nonlinear-arithmetic hang) but with THIS format's own per-tap budget:
    the FULL block scale ``Ws[block]``, not ``Ws[block] / 2`` -- the
    surprise proved above.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]

    dequant_w = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= Ws[_block_of(k)] for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))

    bound = sum(Ws[_block_of(k)] * _abs(X[k]) for k in range(_K))

    return float_matmul, dequant_matmul, rounding_bounds, bound


def test_gguf_q4_0_mac_error_is_bounded():
    # Given W's own per-BLOCK full-scale rounding bound and X completely
    # unchanged, the true float dot product and the one computed against
    # the dequantized weight cannot differ by more than
    # sum_k Ws[block_of(k)] * |X[i, k]|.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_gguf_q4_0_mac_bias_variant_error_is_bounded():
    # This pass never touches Gemm's bias C; adding the same Bias(n) to both
    # the true and the dequantized computation leaves their difference, and
    # therefore the bound, unchanged.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_matmul + bias) - (dequant_matmul + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_gguf_q4_0_mac_negative_control_requires_rounding_bound():
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


def test_gguf_q4_0_mac_naive_half_scale_bound_is_unsound():
    # THIS format's own, distinctive negative control (in place of the
    # other affine files' "uniform bound is unsound across blocks" test,
    # which is about a different failure mode): if one instead used the
    # naive Ws / 2 per-tap budget every OTHER affine block-quant file in
    # this suite gets away with, that claim is NOT a theorem here -- Z3
    # finds a real counterexample, the MAC-level echo of test_gguf_q4_0_
    # half_scale_bound_is_not_a_theorem above.
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]

    dequant_w = [W[k] - ew[k] for k in range(_K)]
    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= Ws[_block_of(k)] for k in range(_K)],
    )
    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))
    error = float_matmul - dequant_matmul

    naive_half_bound = sum((Ws[_block_of(k)] / 2) * _abs(X[k]) for k in range(_K))

    solver = z3.Solver()
    solver.add(rounding_bounds)
    solver.add(z3.Not(z3.And(error <= naive_half_bound, -error <= naive_half_bound)))
    assert solver.check() == z3.sat, (
        "the naive half-scale (Ws / 2) per-tap MAC bound holds even though "
        "Q4_0's own saturation boundary can make a tap's real error reach "
        "the FULL scale Ws -- this format's own full-scale composition is "
        "not actually load-bearing, which would be wrong"
    )


def test_gguf_q4_0_mac_uniform_bound_is_unsound_across_blocks():
    # This suite's usual "one shared scale across independent blocks" is
    # ALSO unsound here (on top of the half-vs-full surprise above): bound
    # every tap by block 0's own full scale Ws0 alone, then let block 1's
    # real scale Ws1 exceed it.
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]

    dequant_w = [W[k] - ew[k] for k in range(_K)]
    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= Ws[_block_of(k)] for k in range(_K)],
    )
    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))
    error = float_matmul - dequant_matmul

    uniform_bound = Ws[0] * sum(_abs(X[k]) for k in range(_K))

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


def _round_trip_float16(values):
    return values.astype(np.float16).astype(np.float64)


def _q4_0_quantize_dequantize_matrix(weight):
    """Independent from-scratch numpy re-implementation of
    ``QuantizeDequantizeQ4_0Block``/``GgufLegacyQuantBase::runTransform``
    (``onnxsim/passes/gguf_legacy_quant.h``): flattens ``weight`` in
    ROW-MAJOR order, computes one symmetric scale per 32-element block of
    the flat array (the final block RAGGED, using only its own real
    elements when the flat count isn't a multiple of 32), fp16-round-trips
    that scale (mirroring ``RoundTripFloat16``), and returns the same
    shape, each element replaced by ``(code - 8) * d``.

    Written independently here (own control flow) rather than calling into
    ``onnxsim.gguf_legacy_quant``'s ``quantize_dequantize_q4_0`` or the
    compiled pass itself, so this is a genuine cross-check of the real
    pass's own math.
    """
    original_shape = weight.shape
    flat = np.asarray(weight, dtype=np.float64).reshape(-1)
    n = flat.size
    out = np.empty_like(flat)
    for start in range(0, n, _BLOCK_SIZE):
        block = flat[start : start + _BLOCK_SIZE]
        max_abs = float(np.max(np.abs(block))) if block.size else 0.0
        d = _round_trip_float16(np.array(max(max_abs, 1e-12) / _Q4_0_BIAS))
        d = float(d)
        code = np.clip(np.round(block / d) + _Q4_0_BIAS, 0, _MAX_CODE)
        out[start : start + block.size] = (code - _Q4_0_BIAS) * d
    return out.reshape(original_shape)


def test_gguf_q4_0_pass_fires_via_extra_optimizers_and_rewires_weight():
    # Confirms the pass is reachable via extra_optimizers (this suite's
    # standard "is it actually registered" check, via _list_other_
    # optimizers under the hood) and rewrites the weight in place (SAME
    # shape/dtype, a brand-new initializer name -- no new graph nodes at
    # all).
    #
    # UNLIKE iq4_nl's own analogous test, this one deliberately does NOT
    # assert exact equality against a single quantize_dequantize
    # application: onnxsim's own fixed-point driver (the ``extra_
    # optimizers`` path) re-runs an opt-in pass repeatedly until
    # convergence or ``ONNXSIM_FIXED_POINT_ITERS`` (default 50), and IQ4_NL
    # is naturally IDEMPOTENT under repeated re-quantization (a codebook
    # value's own block max-abs is always exactly 1.0 x its own scale, so
    # re-deriving the scale reproduces the same scale, and the nearest-
    # codebook search of an exact codebook entry returns itself). Q4_0 is
    # NOT idempotent, precisely because of this file's own documented
    # saturation surprise: a block whose positive-signed extremum got
    # clipped to code 15 (dequant 7d, not 8d) has a SMALLER quantized
    # max-abs than the original block, so re-quantizing shrinks the scale
    # again next iteration -- a genuine, repeated-application drift, unlike
    # every idempotent pass elsewhere in this suite. Exact-value matches
    # against a SINGLE application are checked below via the dedicated
    # ``apply_gguf_q4_0_quantization_cpp`` entry point instead (which does
    # not iterate), matching this file's own other differential tests.
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

    sim_model, _ops = simplify_isolated_extra(model, "gguf_q4_0", check_n=0)

    matmul_node = producer(sim_model, "Y")
    assert matmul_node.op_type == "MatMul"
    x_input, w_input = matmul_node.input
    assert x_input == "X"
    assert w_input != "W"

    w_out_init = next(i for i in sim_model.graph.initializer if i.name == w_input)
    assert w_out_init.data_type == onnx.TensorProto.FLOAT
    assert list(w_out_init.dims) == [K, N]

    w_out = numpy_helper.to_array(w_out_init)
    assert not np.array_equal(w_out, weight)
    # Still at most 16 distinct levels per 32-element block, however many
    # times the pass was re-applied.
    flat = w_out.astype(np.float64).reshape(-1)
    for start in range(0, flat.size, _BLOCK_SIZE):
        block = flat[start : start + _BLOCK_SIZE]
        assert len(np.unique(block)) <= _MAX_CODE + 1


def test_gguf_q4_0_single_application_matches_reimplementation_exactly():
    # The exact-value counterpart to the fixed-point-aware test above,
    # using the same weight but the dedicated (non-iterating) entry point
    # ``apply_gguf_q4_0_quantization_cpp`` -- a genuine single application,
    # so it can be compared exactly against the fresh reimplementation.
    rng = np.random.default_rng(0)
    K, N = 8, 8
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[4,{K}] X) => (float[4,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_gguf_q4_0_quantization_cpp(model)
    matmul_node = next(n for n in quantized.graph.node if n.op_type == "MatMul")
    _x_input, w_input = matmul_node.input
    w_out = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == w_input)
    )
    expected = _q4_0_quantize_dequantize_matrix(weight.astype(np.float64)).astype(
        np.float32
    )
    np.testing.assert_allclose(w_out, expected, rtol=1e-5, atol=1e-6)


def test_gguf_q4_0_gemm_bias_untouched_and_blocked_by_own_flat_storage():
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

    quantized = onnxsim.apply_gguf_q4_0_quantization_cpp(model)
    onnx.checker.check_model(quantized)

    gemm_node = next(n for n in quantized.graph.node if n.op_type == "Gemm")
    x_input, w_input, b_input = gemm_node.input
    assert x_input == "X"
    assert b_input == "B"

    w_out_init = next(i for i in quantized.graph.initializer if i.name == w_input)
    assert list(w_out_init.dims) == [N, K]
    w_out = numpy_helper.to_array(w_out_init)

    expected = _q4_0_quantize_dequantize_matrix(weight.astype(np.float64)).astype(
        np.float32
    )
    np.testing.assert_allclose(w_out, expected, rtol=1e-5, atol=1e-6)


def test_gguf_q4_0_declines_transposed_activation_gemm():
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

    quantized = onnxsim.apply_gguf_q4_0_quantization_cpp(model)
    assert quantized.SerializeToString() == model.SerializeToString()


def test_gguf_q4_0_declines_non_constant_weight():
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

    quantized = onnxsim.apply_gguf_q4_0_quantization_cpp(model)
    assert quantized.SerializeToString() == model.SerializeToString()


def test_gguf_q4_0_zero_maps_to_zero_exactly():
    # Q4_0 has no separate min/zero-point: a value of exactly 0 must
    # round-trip to exactly 0 regardless of the rest of the block, since
    # code = 8 maps to (8 - 8) * d == 0 exactly for any d.
    rng = np.random.default_rng(3)
    K, N = _BLOCK_SIZE, 4
    weight = rng.standard_normal((K, N)).astype(np.float32)
    weight[0, 0] = 0.0
    model = _model(
        f"""
        g (float[4,{K}] X) => (float[4,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_gguf_q4_0_quantization_cpp(model)
    matmul_node = next(n for n in quantized.graph.node if n.op_type == "MatMul")
    _x_input, w_input = matmul_node.input
    w_out = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == w_input)
    )
    assert w_out[0, 0] == 0.0


def test_gguf_q4_0_ragged_final_block_matches_reimplementation():
    # Weight element count NOT a multiple of 32 (5 * 7 = 35 = one full
    # 32-element block + a ragged 3-element final block).
    rng = np.random.default_rng(4)
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

    quantized = onnxsim.apply_gguf_q4_0_quantization_cpp(model)
    matmul_node = next(n for n in quantized.graph.node if n.op_type == "MatMul")
    _x_input, w_input = matmul_node.input
    w_out = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == w_input)
    )

    expected = _q4_0_quantize_dequantize_matrix(weight.astype(np.float64)).astype(
        np.float32
    )
    np.testing.assert_allclose(w_out, expected, rtol=1e-5, atol=1e-6)

    flat = weight.astype(np.float64).reshape(-1)
    ragged_block = flat[32:35]
    assert ragged_block.size == 3
    expected_d = float(
        _round_trip_float16(np.array(max(np.max(np.abs(ragged_block)), 1e-12) / 8.0))
    )
    out_flat = w_out.astype(np.float64).reshape(-1)
    out_ragged = out_flat[32:35]
    recovered_codes = np.round(out_ragged / expected_d + _Q4_0_BIAS)
    assert np.all((recovered_codes >= 0) & (recovered_codes <= _MAX_CODE))


def test_gguf_q4_0_positive_vs_negative_extreme_asymmetry_on_real_pass():
    # The surprise, confirmed against the REAL COMPILED pass (not just the
    # Z3 abstraction or the Python reference): a block whose magnitude-
    # maximizing element is POSITIVE loses a full extra half-scale of
    # precision at that element (error close to d, not d/2) versus the
    # mirror block whose magnitude-maximizing element is NEGATIVE (which
    # round-trips that element exactly).
    K, N = _BLOCK_SIZE, 1
    rng = np.random.default_rng(5)

    weight_pos = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    weight_pos[0, 0] = 1.0  # the block's own positive extreme
    weight_neg = weight_pos.copy()
    weight_neg[0, 0] = -1.0  # mirror: same magnitude, negative sign

    def _quantized_first_element(weight):
        model = _model(
            f"""
            g (float[2,{K}] X) => (float[2,{N}] Y)
            {{
              Y = MatMul(X, W)
            }}
            """,
            [_f32(weight, "W")],
        )
        quantized = onnxsim.apply_gguf_q4_0_quantization_cpp(model)
        matmul_node = next(n for n in quantized.graph.node if n.op_type == "MatMul")
        _x_input, w_input = matmul_node.input
        w_out = numpy_helper.to_array(
            next(i for i in quantized.graph.initializer if i.name == w_input)
        )
        return float(w_out[0, 0])

    dequant_pos = _quantized_first_element(weight_pos)
    dequant_neg = _quantized_first_element(weight_neg)

    max_abs = float(np.max(np.abs(weight_pos.astype(np.float64))))
    assert max_abs == 1.0
    d = max_abs / _Q4_0_BIAS  # 0.125, up to fp16 round-trip

    # Positive extreme: clipped, error close to the FULL scale d (~0.125),
    # nowhere near the naive d/2 (~0.0625) this suite's other affine files
    # would guarantee.
    error_pos = abs(1.0 - dequant_pos)
    assert error_pos > 0.75 * d, (
        f"expected the positive extreme's error ({error_pos}) to approach "
        f"the full scale d ({d}), confirming the saturation surprise"
    )

    # Negative extreme: exact, error 0 (up to fp16 round-trip of d itself).
    error_neg = abs(-1.0 - dequant_neg)
    assert error_neg < 1e-3, (
        f"expected the negative extreme to round-trip essentially exactly, "
        f"got error {error_neg}"
    )


def test_gguf_q4_0_output_within_proved_full_scale_bound():
    # The full end-to-end sanity check against a real ONNX graph execution
    # (onnx's own reference evaluator -- plain float32, no quantized tensor
    # type or contrib op involved): every output element's error against
    # the true float MatMul must stay within the proved full-scale bound
    # sum_k |X[i, k]| * d[k, n]. (This ordinary MatMul sums many taps per
    # output element, so positive- and negative-signed per-tap errors
    # routinely partially cancel in the aggregate -- unlike the single,
    # isolated tap the next test constructs on purpose, this one does NOT
    # also assert a half-scale-bound violation, since aggregate
    # cancellation across an ordinary random weight's many taps need not
    # reproduce the single-element surprise at the summed level.)
    rng = np.random.default_rng(6)
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

    quantized = onnxsim.apply_gguf_q4_0_quantization_cpp(model)
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

    # Per-element full scale d[k, n]: recompute directly from the weight's
    # own flat blocks (same layout as the reimplementation above).
    flat_w = weight.astype(np.float64).reshape(-1)
    d_flat = np.empty_like(flat_w)
    for start in range(0, flat_w.size, _BLOCK_SIZE):
        block = flat_w[start : start + _BLOCK_SIZE]
        max_abs = float(np.max(np.abs(block)))
        d_flat[start : start + block.size] = float(
            _round_trip_float16(np.array(max(max_abs, 1e-12) / _Q4_0_BIAS))
        )
    d_matrix = d_flat.reshape(K, N)

    bound_full = np.abs(x.astype(np.float64)) @ d_matrix  # sum_k |x| * d[k, n]
    assert np.all(error <= bound_full + 1e-4)

    expected_w = _q4_0_quantize_dequantize_matrix(weight.astype(np.float64))
    np.testing.assert_allclose(w_out, expected_w, rtol=1e-5, atol=1e-6)


def test_gguf_q4_0_end_to_end_output_exceeds_naive_half_scale_bound():
    # Confirms the bound genuinely needed to be the full scale, not merely
    # "not yet tightened", at the level of a real ONNX graph's OUTPUT (via
    # ReferenceEvaluator), not just a single initializer element: a single,
    # isolated tap (X's every other column zeroed out) ties the saturation
    # surprise directly to Y itself, deliberately avoiding the aggregate
    # cross-tap cancellation the previous, ordinary-random-weight test
    # would otherwise dilute it with.
    K, N = _BLOCK_SIZE, 1
    weight = np.full((K, N), 0.05, dtype=np.float32)
    weight[0, 0] = 1.0  # this block's own positive extreme
    x = np.zeros((1, K), dtype=np.float32)
    x[0, 0] = 1.0  # isolates tap k=0 exactly: Y = weight[0, 0] alone
    model = _model(
        f"""
        g (float[1,{K}] X) => (float[1,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_gguf_q4_0_quantization_cpp(model)
    onnx.checker.check_model(quantized)

    y_float = x.astype(np.float64) @ weight.astype(np.float64)
    evaluator = ReferenceEvaluator(quantized)
    (y_quant,) = evaluator.run(None, {"X": x})
    error = float(np.abs(y_float - y_quant.astype(np.float64))[0, 0])

    max_abs = float(np.max(np.abs(weight.astype(np.float64))))
    d = max_abs / _Q4_0_BIAS
    assert error > d / 2, (
        f"expected the isolated positive-extreme tap's own output error "
        f"({error}) to exceed the naive d/2 bound ({d / 2}), confirming "
        "the saturation surprise is visible at the graph's own output, not "
        "just in the raw initializer"
    )
    assert error <= d + 1e-6  # still within the proved, correct full-scale bound
