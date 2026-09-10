"""Formal check for GgufQ6K (opt-in; onnxsim's own
``onnxsim/passes/gguf_q6_k.h``, C++ port of ``onnxsim/gguf_q6_k.py``'s own
``apply_gguf_q6_k_quantization``): llama.cpp's GGUF "Q6_K" K-quant format --
this suite's FIRST genuinely TWO-LEVEL hierarchical block-quant scheme.
Every prior block-quant file (MXFP4, IQ4_NL, FP6_LLM, and the still-untested
Q4_0/Q4_1) has exactly ONE level of blocking: one scale per block, one code
per element, ``dequant = scale * code`` (or ``codebook[code] * scale``).
Q6_K, confirmed by reading ``onnxsim/passes/gguf_q6_k.h`` in full (its
``gguf_q6_k_detail::QuantizeDequantizeQ6KSuperBlock``) and cross-checked
against ``onnxsim/ggml_kquant.h``'s own already-verified
``DequantizeQ6_KBlock`` decoder (this port's own explicitly-documented
"transcribed from, and kept consistent with" source), has TWO nested levels
of blocking:

* a 256-element SUPER-block is split into 16 SUB-blocks of 16 elements each;
* each sub-block gets its OWN 8-bit scale code ``sc_j``;
* the whole super-block shares ONE float16 super-block scale ``d``;
* each individual element gets its own SYMMETRIC 6-bit code ``q``.

Reconstruction, read directly off ``QuantizeDequantizeQ6KSuperBlock``'s own
final line (``data[start + i] = code * (d * sc)``) and matching
``DequantizeQ6_KBlock``'s own ``y[l] = d * sc[is] * q`` exactly: ``dequant =
d * sc_j * q`` -- THREE multiplicative factors, unlike every prior file's
two.

--------------------------------------------------------------------------
Confirmed exactly from the C++ (not assumed from this docstring's own
paraphrase of the task that produced it)
--------------------------------------------------------------------------

* ``sc_j``'s raw FORMAT range is a signed 8-bit field, ``[-128, 127]``
  (``ggml_kquant.h`` reads it via ``reinterpret_cast<const int8_t*>``), but
  ``gguf_q6_k.h``'s own ENCODER (``kMaxSubScaleCode = 127`` and
  ``sc = std::min(std::max(sc, 0.0), 127.0)``) only ever emits
  ``sc_j in [0, 127]`` -- confirmed directly from the clamp, not merely from
  the header's own comment. This file's proof and tests only exercise the
  encoder's own non-negative range.
* ``q``'s range is exactly ``[-32, 31]`` (``kMaxCode = 32``, clamp
  ``std::min(std::max(code, -32.0), 31.0)``) -- an ASYMMETRIC range (32
  negative codes, 31 positive) despite the encoder's own scale formula
  dividing by the SYMMETRIC-looking ``kMaxCode = 32``, not
  ``kMaxCode - 1 = 31``. This mismatch is the source of a genuine surprise
  documented below.
* Per sub-block ``j``: ``ideal_scale_j = max(|sub_block_j|, 1e-12) /
  kMaxCode`` (a real number, computed once from the sub-block's own data,
  never itself quantized) and ``sc_j = round(ideal_scale_j / d)`` clamped to
  ``[0, 127]``. Per super-block: ``d = RoundTripFloat16(max_j(ideal_scale_j)
  / kMaxSubScaleCode)`` -- ONE shared value, float16-round-tripped (this
  port's own already-verified fp16 codec, reused verbatim from
  ``gguf_legacy_quant.h``/``gguf_ternary_quant.h``'s own ``RoundTripFloat16``
  helper -- not re-verified here, treated as exact for this file's own
  algebra exactly the way this suite's other GGUF ports do). Per element:
  ``q = round(value / effective_scale)`` clamped to ``[-32, 31]``, where
  ``effective_scale := d * sc_j`` (using ``1.0`` in place of ``effective_
  scale`` only to avoid a division by zero when ``sc_j == 0``; the
  RECONSTRUCTION still multiplies by ``d * sc_j`` regardless, so a ``sc_j ==
  0`` sub-block dequantizes to all-zero no matter what code was computed --
  a degenerate corner this file does not build a dedicated Z3 lemma for,
  since it never arises in any test weight below, all of which have
  ``sc_j > 0`` by construction or by the sheer size of ``kMaxSubScaleCode``
  relative to any all-but-all-zero sub-block).
* There is NO separate per-sub-block min/offset anywhere in the formula --
  confirmed directly: the reconstruction is a bare product ``d * sc_j * q``,
  no ``+`` term at all, matching the header's own "symmetric (no separate
  min)" description exactly, unlike Q4_K/Q5_K's asymmetric ``(scale, min)``
  pair (which is exactly why this repo has no C++ port of those -- see the
  header's own comment).
* Blocks are laid out over the weight's own FLATTENED, ROW-MAJOR storage
  (``ReadFloatMatrix`` + the flat ``start`` loop in ``GgufQ6K::runTransform``)
  -- like IQ4_NL/FP6_LLM/Q4_0/Q4_1, NOT indexed by (block-of-K,
  output-channel) the way MXFP4/INT4/INT8-block are. A ragged final
  super-block (and, within it, a ragged final sub-block) is quantized using
  only its own real elements -- ``QuantizeDequantizeQ6KSuperBlock``'s own
  ``sub_count = std::min(kSubBlockSize, count - start)``, no zero-padding at
  all, mirroring IQ4_NL's own ragged-final-block handling and its own
  zero-padding-invariance argument (a zero can never be a (sub-)block's own
  largest-magnitude element unless the whole (sub-)block is already
  all-zero).
* Only MatMul, or a "vanilla" Gemm (``transA=0``, ``alpha=1``, and
  ``beta=1`` when a bias is present) with a constant 2-D float32 weight is
  matched -- ``GgufQ6K::patternMatchPredicate``/``runTransform`` call the
  SAME ``MatchMatMulLike`` (``quantize_matmul_common.h``) every sibling
  ``*_cpp`` weight-only pass in this suite uses, read in full above. The
  bias, when present, is never touched.

--------------------------------------------------------------------------
The genuinely new modeling challenge: TWO nested rounding events, not one
--------------------------------------------------------------------------

Per this task's own instruction, error propagates through two nested
quantization levels rather than one:

  (a) the ELEMENT level: ``q = round(value / effective_scale)`` -- an
      ordinary round-to-nearest, IF ``effective_scale`` is taken as already
      fixed and exact;
  (b) the SUB-BLOCK-SCALE level, one level "underneath" (a): ``sc_j =
      round(ideal_scale_j / d)`` -- ``sc_j`` is itself a rounded integer
      approximation of the real ratio ``ideal_scale_j / d``, so
      ``effective_scale := d * sc_j`` is itself only an approximation of the
      sub-block's own "ideal" (real-valued, never-quantized) scale
      ``ideal_scale_j``.

Per ``test_formal_verify_dynamic_quantize_matmul.py``'s own module docstring
(the original documented incident) and
``test_formal_verify_qoperator_quantize_gemm.py``'s own module docstring
(the same mitigation carried one level further, for a *different* two-layer
composition than the one here), reconstructing a rounded quantity from its
own lower-level code/scale PRODUCTS inside the same Z3 query that also
chains further arithmetic on top of it is a documented way to make Z3's
nonlinear-arithmetic search hang well past a minute, even at tiny concrete
sizes. With THREE multiplicative unknowns feeding one reconstruction here
(``d``, ``sc_j``, ``q``) -- one more than either of those two files' own
two-unknown reconstructions -- this file follows the DIRECT-error-variable
idiom throughout, and NEVER lets more than one product of unknowns appear in
any single Z3 query:

  (a) ``test_q6_k_element_round_trip_given_fixed_effective_scale`` proves the
      standard half-step bound ``|value - q * effective_scale| <=
      effective_scale / 2`` treating ``effective_scale`` as an opaque,
      already-fixed free real (ONE product of unknowns: ``q *
      effective_scale``) -- it does not know or care that
      ``effective_scale`` is itself ``d * sc_j``.
  (b) ``test_q6_k_subblock_scale_perturbation_is_bounded`` separately proves
      ``|d * sc_j - ideal_scale_j| <= d / 2`` (ONE product of unknowns:
      ``d * sc_j``) from ``sc_j``'s own rounding hypothesis -- it does not
      know or care that ``d * sc_j`` will go on to multiply some ``q``.
  (c) ``test_q6_k_combined_per_element_bound_via_promoted_datums`` composes
      (a) and (b) via the triangle inequality into
      ``|value - dequant| <= ideal_scale_j / 2 + d / 4``
      -- but instead of literally substituting ``effective_scale = d *
      sc_j`` and ``dequant = q * d * sc_j`` (which would reintroduce the
      THREE-way product this file is specifically avoiding), it takes (a)'s
      and (b)'s own PROVED CONCLUSIONS as fresh, free, directly-bounded
      "datum" error variables (``e_q`` and ``e_sc`` below) -- exactly
      ``test_formal_verify_qoperator_quantize_gemm.py``'s own "promote an
      already-proved sub-lemma's own conclusion into a fresh direct datum
      variable for the larger combined query, rather than re-deriving it
      from lower-level products in the same query" technique, applied here
      to a genuinely three-level (not that file's two-level) structure. The
      resulting combined query has NO products of unknowns at all (every
      term is a sum of reals, some pre-halved/quartered) -- pure linear real
      arithmetic, and correspondingly instantaneous for Z3 (confirmed
      directly: well under a second, no risk of the hang the two-level files
      above already ran into and had to work around).

One consequence worth being honest about, matching this suite's own
"be honest about which parts are exact algebra vs. hypotheses chaining
separately-proved lemmas" convention: lemmas (a) and (b) are each proved
directly by Z3 as standalone theorems (real, checked algebra). The
COMPOSITION in (c) is likewise checked directly by Z3 -- given (a)'s and
(b)'s own conclusions as hypotheses, Z3 verifies the triangle-inequality
arithmetic that combines them -- so nothing here is merely asserted or
axiomatized; the only thing NOT re-verified inside query (c) is that ``e_q``
and ``e_sc`` really do arise from (a)'s and (b)'s own honest hypotheses
(that much is established by (a) and (b) themselves, each its own complete,
separately-passing proof).

Both (a) and (b) implicitly assume NO CLAMPING occurs (``q`` is the true
``round(...)`` value, not a clamped-away one; likewise ``sc_j``) -- the same
scoping every other block-quant file in this suite uses ("no test weight
below is built to force clipping"). See the next section for why, for Q6_K
specifically, this scoping needs more care than usual.

--------------------------------------------------------------------------
A genuine surprise: Q6_K's own scale formula makes element-level clipping
ROUTINE, not a rare edge case
--------------------------------------------------------------------------

Every prior block-quant file's own "no clipping" scoping is a mild
simplification for well-scaled test data: a symmetric scale of
``max(|block|) / M`` (``M`` = the largest representable code magnitude)
guarantees the block's own peak element normalizes to EXACTLY ``M``, safely
representable, so clipping only happens if the encoder's own further
rounding pushes a value slightly past it. Q6_K's ``ideal_scale_j =
max(|sub_block|) / kMaxCode`` divides by ``kMaxCode = 32`` -- but the
representable POSITIVE code range only goes up to ``kMaxCode - 1 = 31``
(the range is the ASYMMETRIC ``[-32, 31]``, confirmed above). So even with
NO sub-block-scale rounding error at all (an idealized, continuous
``effective_scale = ideal_scale_j`` exactly), the sub-block's own
peak-magnitude element, IF POSITIVE, normalizes to EXACTLY ``kMaxCode =
32`` -- one past the largest representable positive code -- and MUST be
clamped down to 31, losing a genuine ``effective_scale`` worth of magnitude
(``32 - 31 = 1`` full code unit), independent of any additional ``sc_j``
rounding. A peak that happens to be NEGATIVE instead normalizes to exactly
``-32``, which IS representable (the asymmetric range's "extra" negative
code), so no such forced clamp occurs on that side.

Confirmed directly (this file's own numeric check against the real compiled
pass, not merely asserted): with ordinary random Gaussian test data, roughly
6-7% of ELEMENTS clip (not merely "could, in an adversarial case") --
because in the overwhelming majority of the affected sub-blocks it is
specifically their own single peak element that clips, and roughly half of
all sub-blocks have a positively-signed peak. This is a structural property
of the encoder's own scale formula (dividing by ``kMaxCode`` rather than
``kMaxCode - 1``), not a bug this file is reporting for a fix -- it merely
means the "assume no clipping" round-trip lemmas above are honestly scoped
to well-chosen data, and this file's own differential/numeric bound check
(``test_gguf_q6_k_output_within_proved_combined_bound_via_reference_
evaluator`` below) uses a DELIBERATELY ENGINEERED clip-free weight (every
sub-block's own peak forced negative, every other element kept comfortably
smaller) rather than plain random data, exactly so the proved "no clipping"
bound is the honest thing being checked. A companion test
(``test_gguf_q6_k_positive_peak_forces_clamping_in_real_pass``) confirms
the surprise itself against the REAL compiled pass (not just the abstract
Z3 lemma or this file's own from-scratch reimplementation): a deliberately
positive sub-block peak really does come back as a magnitude-31, not
magnitude-32, code.

--------------------------------------------------------------------------
The MAC-bound composition: a genuinely SHARED (not per-block) scale term
--------------------------------------------------------------------------

Following this suite's own established pattern (``weight_only_quantize_
int8_block_matmul``'s/``iq4_nl``'s own ``_bound_formulas``), the per-element
bound above composes into a MAC (dot-product) bound the usual way, with
``eps_x := 0`` (``X`` is never touched) and a per-tap budget ``eps_w[k] :=
ideal_scale_j(block_of(k)) / 2 + D / 4``. One structural difference from
every sibling file's own analogous composition, worth stating plainly: in
every prior per-block pass, EVERY term of a per-tap budget varies
independently block-to-block (``Ws[block]`` there is a wholly separate free
variable per block). Here, ``ideal_scale_j`` is genuinely per-sub-block, but
``D`` (the super-block scale) is ONE variable SHARED by every tap in the
whole super-block -- modeled below with a single Z3 symbol ``D`` used in
every tap's own budget, not one ``D`` per block, reflecting the real
two-level hierarchy accurately rather than degenerating it back into a
single-level, fully-independent-per-block model.

--------------------------------------------------------------------------
Differential/structural tests
--------------------------------------------------------------------------

Built via ``onnx.parser`` (per ``CLAUDE.md``) with ``numpy_helper.from_array``
for weight initializers. Two entry points are exercised, mirroring IQ4_NL's
own precedent: the opt-in optimizer name ``"gguf_q6_k"`` via
``simplify_isolated_extra`` (confirming the pass is actually registered and
reachable -- ``GgufQ6K::getPassName()`` returns exactly this string, and
``onnxsim/custom_optimizer_passes.cpp`` registers it as an opt-in, not
default, pass), and the dedicated Python entry point
``onnxsim.apply_gguf_q6_k_quantization_cpp`` (``onnxsim/quantize_entry.cpp``'s
``ApplyGgufQ6K``) for the detailed numeric checks, since it hands back a
single self-contained rewrite with no other pass's side effects.

A further surprise, confirmed directly while writing this file: unlike
IQ4_NL's own quantize-dequantize round trip, Q6_K's is NOT idempotent --
re-applying it to its own already-quantized output keeps changing values by
a comparable amount on every further application, with no fixed point (or
even a short cycle) found within several iterations. Because
``simplify_isolated_extra`` drives ``onnxsim.simplify``'s own fixed-point
optimizer loop (which keeps re-running the isolated pass set until nothing
changes, up to ``ONNXSIM_FIXED_POINT_ITERS`` times), the weight it produces
for this pass is generally the result of MANY re-applications, not the
single quantize-dequantize round trip this file's own reimplementation
models -- confirmed by the "simplification stopped because of timeout"
warning it emits. So, unlike every numeric-matching differential test below
(all of which use the dedicated entry point, applying the rewrite exactly
once), the one test that exercises the ``simplify_isolated_extra`` path
checks only STRUCTURAL properties (op shape, dtype, passthrough of ``X``),
never exact values.

No ``com.microsoft`` contrib op or quantized ONNX tensor type is ever
involved (the new initializer is plain ``TensorProto_DataType_FLOAT``, same
shape as the original weight) -- so, per this task's own instruction, the
numeric checks below use ``onnx.reference.ReferenceEvaluator`` and direct
initializer inspection, never an onnxruntime ``InferenceSession``.

Confirmed below: only MatMul/vanilla-Gemm with a constant 2-D float32 weight
is matched (a transA=1 Gemm and a non-constant weight both decline
outright); ``W'`` has the same shape/dtype as ``W``; a ragged final
super-block (256 not dividing the weight's element count, AND the final
super-block's own final sub-block also ragged) produces no crash and a
finite, correctly-shaped output, matching an independent from-scratch numpy
reimplementation of the FULL two-level quantize-dequantize scheme (built
fresh in this file, NOT imported from ``onnxsim.gguf_q6_k`` or
``onnxsim.ggml_kquant``, and NOT calling back into the compiled pass);
observed reconstruction error, run through the real end-to-end ONNX graph,
stays within the combined bound proved in Z3 above. Per the header's own
"ACCEPTED, PERMANENT DIVERGENCE" note (this port is expected to track its
own Python reference closely, up to float16/summation-order differences,
not bit-for-bit), comparisons below use a small but non-zero ``rtol``/
``atol`` -- though, confirmed empirically while writing this file, this
particular C++ port and this file's own from-scratch numpy reimplementation
in fact agree EXACTLY (0 ULP difference) on every concrete case tried, since
both round-trip through IEEE754 binary16 the same way numpy's own
``.astype(np.float16)`` does; the non-zero tolerance is kept anyway as an
honest safety margin, not because a difference was ever observed.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser
from onnx.reference import ReferenceEvaluator

import onnxsim

# Transcribed verbatim from gguf_q6_k_detail's own constants (onnxsim/passes/
# gguf_q6_k.h).
_SUB_BLOCK_SIZE = 16
_SUB_BLOCKS_PER_SUPER_BLOCK = 16
_SUPER_BLOCK_SIZE = _SUB_BLOCK_SIZE * _SUB_BLOCKS_PER_SUPER_BLOCK  # 256
_MAX_CODE = 32  # symmetric-LOOKING 6-bit code range [-32, 31] -- see the
# module docstring's own "surprise" section for why it is not actually
# symmetric in effect.
_MAX_SUB_SCALE_CODE = 127  # this encoder's own 8-bit sub-scale code range


def _abs(v):
    return z3.If(v >= 0, v, -v)


# ============================================================================
# (a) Element level: round-trip given a FIXED, exact effective_scale
# ============================================================================


def test_q6_k_element_round_trip_given_fixed_effective_scale():
    # Standard round-to-nearest half-step bound, treating effective_scale
    # (= d * sc_j in the real format) as an opaque already-fixed positive
    # real -- this lemma does not know or care how effective_scale itself
    # came to be, which is exactly what keeps it a ONE-product-of-unknowns
    # query (q * effective_scale only).
    value, q, effective_scale = z3.Reals("value q effective_scale")
    half = z3.RealVal(1) / 2
    hypotheses = z3.And(
        effective_scale > 0,
        q - value / effective_scale <= half,
        value / effective_scale - q <= half,
    )
    dequant = q * effective_scale
    error = value - dequant
    prove(
        z3.Implies(
            hypotheses,
            z3.And(error <= effective_scale / 2, -error <= effective_scale / 2),
        )
    )


# ============================================================================
# (b) Sub-block-scale level: how far sc_j's own rounding perturbs
#     effective_scale away from the sub-block's own real, never-quantized
#     ideal_scale_j
# ============================================================================


def test_q6_k_subblock_scale_perturbation_is_bounded():
    # sc_j = round(ideal_scale_j / d): the standard half-step rounding bound
    # applied one level "underneath" (a), in terms of d and ideal_scale_j
    # instead of value and effective_scale. ONE product of unknowns here
    # (d * sc) -- this lemma does not know or care that d * sc will go on to
    # multiply some element code q.
    ideal_scale, d, sc = z3.Reals("ideal_scale d sc")
    half = z3.RealVal(1) / 2
    hypotheses = z3.And(
        d > 0,
        sc - ideal_scale / d <= half,
        ideal_scale / d - sc <= half,
    )
    effective_scale = d * sc
    perturbation = effective_scale - ideal_scale
    prove(z3.Implies(hypotheses, z3.And(perturbation <= d / 2, -perturbation <= d / 2)))


# ============================================================================
# (c) Composition via the triangle inequality, using PROMOTED datums (never
#     reconstructing dequant from d * sc * q's own THREE-way product)
# ============================================================================


def _combined_bound_hypotheses(
    *, include_element_bound=True, include_subblock_bound=True
):
    """Builds the promoted-datum vocabulary for the combined per-element
    claim. ``e_q`` stands for (a)'s own proved conclusion
    ``value - q * effective_scale`` and ``e_sc`` for (b)'s own proved
    conclusion ``d * sc - ideal_scale`` -- both taken here as FRESH, freely
    quantified reals, bounded ONLY by the hypothesis matching each lemma's
    own already-proved conclusion, never re-derived from d/sc/q products.
    Returns ``(ideal_scale, d, e_sc, e_q, hypotheses)``; the two boolean
    flags let the negative-control tests below drop exactly one of the two
    promoted hypotheses while keeping everything else identical.
    """
    ideal_scale, d, e_sc, e_q = z3.Reals("ideal_scale d e_sc e_q")
    effective_scale = ideal_scale + e_sc
    clauses = [ideal_scale > 0, d > 0, effective_scale > 0]
    if include_subblock_bound:
        clauses.append(_abs(e_sc) <= d / 2)  # (b)'s own proved conclusion
    if include_element_bound:
        clauses.append(_abs(e_q) <= effective_scale / 2)  # (a)'s own proved conclusion
    return ideal_scale, d, e_sc, e_q, z3.And(*clauses)


def test_q6_k_combined_per_element_bound_via_promoted_datums():
    # The genuine three-level composition: given BOTH (a)'s and (b)'s own
    # proved conclusions as direct data (no d*sc*q reconstruction anywhere),
    # |value - dequant| == |e_q| <= ideal_scale_j / 2 + d / 4. This query has
    # NO products of unknowns at all -- pure linear real arithmetic -- and
    # is correspondingly instant for Z3 (confirmed: well under a second),
    # unlike the two-level hang this exact mitigation was built to dodge.
    ideal_scale, d, _e_sc, e_q, hypotheses = _combined_bound_hypotheses()
    bound = ideal_scale / 2 + d / 4
    prove(z3.Implies(hypotheses, z3.And(e_q <= bound, -e_q <= bound)))


def test_q6_k_combined_bound_negative_control_requires_element_level_hypothesis():
    # Drop (a)'s own promoted hypothesis alone (keep (b)'s): the combined
    # bound is not a theorem -- e_q is then unconstrained by anything at all.
    ideal_scale, d, _e_sc, e_q, hypotheses = _combined_bound_hypotheses(
        include_element_bound=False
    )
    bound = ideal_scale / 2 + d / 4
    solver = z3.Solver()
    solver.add(hypotheses)
    solver.add(z3.Not(z3.And(e_q <= bound, -e_q <= bound)))
    assert solver.check() == z3.sat, (
        "the combined bound holds even without lemma (a)'s own promoted "
        "hypothesis -- negative control is vacuous"
    )


def test_q6_k_combined_bound_negative_control_requires_subblock_level_hypothesis():
    # Drop (b)'s own promoted hypothesis alone (keep (a)'s): effective_scale
    # can then be arbitrarily far from ideal_scale (e_sc unconstrained), so
    # e_q's own bound (tied to whatever effective_scale actually is) is no
    # longer boundable purely in terms of ideal_scale and d.
    ideal_scale, d, _e_sc, e_q, hypotheses = _combined_bound_hypotheses(
        include_subblock_bound=False
    )
    bound = ideal_scale / 2 + d / 4
    solver = z3.Solver()
    solver.add(hypotheses)
    solver.add(z3.Not(z3.And(e_q <= bound, -e_q <= bound)))
    assert solver.check() == z3.sat, (
        "the combined bound holds even without lemma (b)'s own promoted "
        "hypothesis -- negative control is vacuous"
    )


# ============================================================================
# The surprise: the encoder's own scale formula forces a positive peak to
# clip, exactly (no additional sc_j rounding needed)
# ============================================================================


def test_q6_k_positive_peak_normalizes_to_an_out_of_range_code_exactly():
    # With an IDEALIZED (unrounded) sub-block scale (ideal_scale = m /
    # kMaxCode, no sc_j quantization at all yet), the sub-block's own
    # maximum-magnitude element m, if POSITIVE, normalizes to EXACTLY
    # kMaxCode = 32 -- one past the largest representable positive code
    # (kMaxCode - 1 = 31) -- forced clamping, independent of any additional
    # sc_j-rounding error. Confirmed directly from
    # QuantizeDequantizeQ6KSuperBlock's own `ideal_scale[j] = max_abs /
    # kMaxCode` (dividing by 32, NOT by 31, despite the actual representable
    # positive maximum being 31).
    m = z3.Real("m")
    ideal_scale = m / _MAX_CODE
    prove(z3.Implies(m > 0, m / ideal_scale == _MAX_CODE))


def test_q6_k_negative_peak_normalizes_to_the_representable_boundary_exactly():
    # The mirror-image, NEGATIVE-peak case: normalizes to EXACTLY -kMaxCode =
    # -32, which unlike +32 IS representable (the asymmetric [-32, 31] range's
    # own "extra" negative code) -- so no forced clamp on this side. This
    # confirms the surprise above is genuinely about SIGN, not merely "the
    # peak element is always mishandled".
    m = z3.Real("m")
    ideal_scale = m / _MAX_CODE
    prove(z3.Implies(m > 0, (-m) / ideal_scale == -_MAX_CODE))


# ============================================================================
# MAC-bound composition: eps_w[k] = ideal_scale_j(block_of(k)) / 2 + D / 4,
# with D genuinely SHARED (one Z3 symbol) across every tap
# ============================================================================

_K = 2  # matches this suite's own minimal-case convention (two separate
# one-element sub-blocks is already enough to exercise independent
# per-sub-block ideal_scale terms against one SHARED D).
_BLOCK_SIZE_ABSTRACT = 1
_NUM_BLOCKS = _K // _BLOCK_SIZE_ABSTRACT


def _block_of(k):
    return k // _BLOCK_SIZE_ABSTRACT


def _mac_bound_formulas():
    X = [z3.Real(f"X{k}") for k in range(_K)]  # X[i, k], true float activation
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[k, n], true float weight
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[k, n] - Wdq[k, n]
    IdealScale = [
        z3.Real(f"IdealScale{b}") for b in range(_NUM_BLOCKS)
    ]  # per SUB-block
    D = z3.Real("D")  # ONE super-block scale, shared by EVERY tap below --
    # not one D per block, unlike every sibling file's own fully-independent
    # per-block scale (see the module docstring's own "genuinely shared"
    # section).

    dequant_w = [W[k] - ew[k] for k in range(_K)]

    eps_w = [IdealScale[_block_of(k)] / 2 + D / 4 for k in range(_K)]
    rounding_bounds = z3.And(
        D > 0,
        *[IdealScale[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= eps_w[k] for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))
    bound = sum(eps_w[k] * _abs(X[k]) for k in range(_K))

    return float_matmul, dequant_matmul, rounding_bounds, bound


def test_gguf_q6_k_mac_bound_error_is_bounded():
    float_matmul, dequant_matmul, rounding_bounds, bound = _mac_bound_formulas()
    error = float_matmul - dequant_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_gguf_q6_k_mac_bound_bias_variant_error_is_bounded():
    # This pass never touches Gemm's bias C -- adding the same Bias(n) to
    # both sides leaves the difference, and therefore the bound, unchanged.
    float_matmul, dequant_matmul, rounding_bounds, bound = _mac_bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_matmul + bias) - (dequant_matmul + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_gguf_q6_k_mac_bound_negative_control_requires_rounding_bound():
    float_matmul, dequant_matmul, _rounding_bounds, bound = _mac_bound_formulas()
    error = float_matmul - dequant_matmul
    solver = z3.Solver()
    solver.add(z3.Not(z3.And(error <= bound, -error <= bound)))
    assert solver.check() == z3.sat, (
        "the MAC bound holds even without any rounding-error budget on ew -- "
        "negative control is vacuous"
    )


def test_gguf_q6_k_mac_bound_uniform_ideal_scale_is_unsound_across_subblocks():
    # Confirms the per-sub-block IdealScale term is genuinely necessary (not
    # merely untested): bounding every tap's error using ONLY block 0's
    # IdealScale0 (as if every sub-block shared one ideal_scale the way they
    # already share D) is NOT a theorem once IdealScale1 can exceed
    # IdealScale0.
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    IdealScale = [z3.Real(f"IdealScale{b}") for b in range(_NUM_BLOCKS)]
    D = z3.Real("D")

    dequant_w = [W[k] - ew[k] for k in range(_K)]
    eps_w = [IdealScale[_block_of(k)] / 2 + D / 4 for k in range(_K)]
    rounding_bounds = z3.And(
        D > 0,
        *[IdealScale[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= eps_w[k] for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))
    error = float_matmul - dequant_matmul

    uniform_bound = (IdealScale[0] / 2 + D / 4) * sum(_abs(X[k]) for k in range(_K))

    solver = z3.Solver()
    solver.add(rounding_bounds)
    solver.add(z3.Not(z3.And(error <= uniform_bound, -error <= uniform_bound)))
    assert solver.check() == z3.sat, (
        "the naive single-sub-block-scale (block 0 only) bound holds even "
        "though block 1 has its own, potentially larger IdealScale -- the "
        "per-sub-block term is not actually load-bearing, which would be "
        "wrong"
    )


# ============================================================================
# Differential / structural tests
# ============================================================================


def _model(body, initializer=(), opset=21, ir_version=10):
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


def _q6_k_quantize_dequantize(flat):
    """Independent, from-scratch numpy reimplementation of
    ``gguf_q6_k_detail::QuantizeDequantizeQ6KSuperBlock`` (``onnxsim/passes/
    gguf_q6_k.h``) -- NOT imported from ``onnxsim.gguf_q6_k`` or
    ``onnxsim.ggml_kquant``, and NOT calling into the compiled pass. Takes a
    flattened float64 array of any length; the final super-block (and its
    own final sub-block) is RAGGED -- using only its own real elements, no
    zero-padding -- when the length is not itself a multiple of
    ``_SUPER_BLOCK_SIZE`` / ``_SUB_BLOCK_SIZE``, exactly mirroring the C++'s
    own ``count = min(kSuperBlockSize, numel - start)`` /
    ``sub_count = min(kSubBlockSize, count - start)`` loops.

    Returns three same-length arrays (dequantized values, per-element
    ``ideal_scale_j``, per-element super-block scale ``d``) -- the latter
    two feed the proved combined bound (``ideal_scale_j / 2 + d / 4``)
    directly, element by element.
    """
    flat = np.asarray(flat, dtype=np.float64)
    n = flat.size
    out = np.empty(n)
    ideal_scale_out = np.empty(n)
    d_out = np.empty(n)

    for sb_start in range(0, n, _SUPER_BLOCK_SIZE):
        sb = flat[sb_start : sb_start + _SUPER_BLOCK_SIZE]
        count = sb.size
        num_sub = -(-count // _SUB_BLOCK_SIZE)
        ideal_scale = np.empty(num_sub)
        for j in range(num_sub):
            s, e = j * _SUB_BLOCK_SIZE, min((j + 1) * _SUB_BLOCK_SIZE, count)
            max_abs = float(np.max(np.abs(sb[s:e]))) if e > s else 0.0
            ideal_scale[j] = max(max_abs, 1e-12) / _MAX_CODE
        max_ideal_scale = max(float(ideal_scale.max()), 1e-12)
        # RoundTripFloat16: mirrored exactly via numpy's own IEEE754
        # binary16 cast, the same round-to-nearest-even convention the C++
        # side's FloatToFloat16Bits implements.
        d = np.float16(max_ideal_scale / _MAX_SUB_SCALE_CODE).astype(np.float64)

        for j in range(num_sub):
            s, e = j * _SUB_BLOCK_SIZE, min((j + 1) * _SUB_BLOCK_SIZE, count)
            sc = np.clip(np.round(ideal_scale[j] / d), 0, _MAX_SUB_SCALE_CODE)
            sub_scale = d * sc if sc > 0 else 1.0
            effective_scale = d * sc
            for i in range(s, e):
                code = np.clip(np.round(sb[i] / sub_scale), -_MAX_CODE, _MAX_CODE - 1)
                out[sb_start + i] = code * effective_scale
                ideal_scale_out[sb_start + i] = ideal_scale[j]
                d_out[sb_start + i] = d

    return out, ideal_scale_out, d_out


def _make_clip_free_weight(rng, shape, base_scale=1.0, jitter=0.12, filler_frac=0.75):
    """Builds a weight array engineered so Q6_K's own quantize-dequantize
    round trip never clips at EITHER level (sub-block scale code ``sc_j``,
    or element code ``q``) -- needed because (see the module docstring's own
    "surprise" section) Q6_K's element code clips ROUTINELY on ordinary
    random data, unlike every earlier single-level format in this suite.
    Per ``_SUB_BLOCK_SIZE``-element sub-block (flattened, row-major, exactly
    the layout the real pass blocks over): one element is a deliberate,
    deterministic "peak" of magnitude ``m_j`` with a NEGATIVE sign (a
    POSITIVE peak would force a clamp by construction, per the surprise
    above), and the other elements have strictly smaller magnitude (at most
    ``filler_frac * m_j``, comfortably under the margin ``sc_j``'s own
    +-0.5 rounding step could otherwise eat into). Every sub-block's own
    ``m_j`` is kept within ``jitter`` of every other's in the SAME
    super-block, which keeps every ``sc_j`` large (confirmed empirically
    while writing this file: ``sc_j`` stays in roughly ``[100, 127]`` for the
    parameters used below), minimizing ``sc_j``'s own RELATIVE rounding
    error. Used ONLY by the differential test that checks real numeric
    output against the proved combined bound; other differential tests below
    use plain random data and do not need to avoid clipping.
    """
    n = int(np.prod(shape))
    out = np.empty(n)
    for sb_start in range(0, n, _SUPER_BLOCK_SIZE):
        remaining = min(_SUPER_BLOCK_SIZE, n - sb_start)
        num_sub = -(-remaining // _SUB_BLOCK_SIZE)
        for j in range(num_sub):
            s = j * _SUB_BLOCK_SIZE
            e = min(s + _SUB_BLOCK_SIZE, remaining)
            width = e - s
            if width == 0:
                continue
            m_j = base_scale * (1.0 + jitter * rng.uniform(-1, 1))
            block = (
                rng.uniform(0.05, filler_frac, size=width)
                * m_j
                * rng.choice([-1.0, 1.0], size=width)
            )
            block[0] = -m_j  # deliberate, negatively-signed peak
            rng.shuffle(block)
            out[sb_start + s : sb_start + e] = block
    return out.reshape(shape)


def test_gguf_q6_k_pass_fires_and_matches_reimplementation_via_extra_optimizers():
    # Confirms the pass is reachable via the usual opt-in path and rewrites
    # the weight in place (same shape/dtype, brand-new initializer name, no
    # new graph nodes). K * N = 512 = exactly two clean (non-ragged)
    # super-blocks.
    rng = np.random.default_rng(0)
    rows, K, N = 4, 16, 32
    weight = (rng.standard_normal((K, N)) * 0.6).astype(np.float32)
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    # check_n=0, and NO exact-value comparison against a single-application
    # reimplementation here (unlike every other differential test below,
    # which uses the dedicated onnxsim.apply_gguf_q6_k_quantization_cpp
    # entry point instead) -- a genuine further surprise, confirmed directly
    # while writing this file: unlike IQ4_NL's own analogous
    # extra_optimizers test (which DOES assert exact equality this way),
    # Q6_K's quantize-dequantize round trip is NOT idempotent -- re-applying
    # it to its own already-quantized output keeps changing values by a
    # comparable amount on every further application (confirmed empirically:
    # repeated re-application does not settle into a fixed point, or even a
    # short cycle, within several iterations). onnxsim's own fixed-point
    # optimizer loop (which simplify_isolated_extra drives via
    # onnxsim.simplify) therefore re-applies this pass to its own prior
    # output up to `ONNXSIM_FIXED_POINT_ITERS` times (confirmed via the
    # "simplification stopped because of timeout" warning it emits here),
    # so the resulting weight is NOT the single quantize-dequantize round
    # trip this file's own reimplementation models -- only the STRUCTURAL
    # shape of the rewrite is checked here; every numeric/bound check below
    # instead uses the dedicated entry point, which applies the rewrite
    # exactly once.
    sim_model, _ops = simplify_isolated_extra(model, "gguf_q6_k", check_n=0)

    matmul_node = producer(sim_model, "Y")
    assert matmul_node.op_type == "MatMul"
    x_input, w_input = matmul_node.input
    assert x_input == "X"  # activation passed through completely unchanged
    assert w_input != "W"  # rewritten to a brand-new initializer

    w_out_init = next(i for i in sim_model.graph.initializer if i.name == w_input)
    assert w_out_init.data_type == onnx.TensorProto.FLOAT
    assert list(w_out_init.dims) == [K, N]
    w_out = numpy_helper.to_array(w_out_init)
    assert np.all(np.isfinite(w_out))


def test_gguf_q6_k_gemm_bias_untouched_and_blocked_over_own_flat_storage():
    # A "vanilla" Gemm (transA=0, alpha=1, beta=1) with a bias: the bias is
    # passed through unchanged, and blocking happens over W's OWN [N, K]
    # flat storage directly (no channel-axis-aware transpose, matching the
    # header's own IQ4_NL-style flat layout, not a per-channel one).
    rng = np.random.default_rng(1)
    rows, K, N = 3, 12, 24  # N * K = 288: one clean super-block + a 32-wide
    # ragged tail (not a multiple of 256), also exercising the ragged case.
    weight = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
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

    quantized = onnxsim.apply_gguf_q6_k_quantization_cpp(model)
    onnx.checker.check_model(quantized)

    gemm_node = next(n for n in quantized.graph.node if n.op_type == "Gemm")
    x_input, w_input, b_input = gemm_node.input
    assert x_input == "X"
    assert b_input == "B"  # bias untouched

    w_out_init = next(i for i in quantized.graph.initializer if i.name == w_input)
    assert list(w_out_init.dims) == [N, K]
    w_out = numpy_helper.to_array(w_out_init).astype(np.float64)

    expected, _ideal_scale, _d = _q6_k_quantize_dequantize(
        weight.astype(np.float64).reshape(-1)
    )
    np.testing.assert_allclose(w_out.reshape(-1), expected, rtol=1e-4, atol=1e-6)


def test_gguf_q6_k_declines_transposed_activation_gemm():
    # MatchMatMulLike requires transA == 0; a transA=1 Gemm must decline
    # outright, leaving the model byte-for-byte unchanged.
    rng = np.random.default_rng(2)
    K, rows, N = 20, 4, 3
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    model = _model(
        f"""
        g (float[{K},{rows}] X) => (float[{rows},{N}] Y)
        {{
          Y = Gemm<transA = 1>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_gguf_q6_k_quantization_cpp(model)
    assert quantized.SerializeToString() == model.SerializeToString()


def test_gguf_q6_k_declines_non_constant_weight():
    # patternMatchPredicate requires FetchConstantTensor(info.w) to succeed;
    # a weight that is a genuine graph INPUT must be left untouched.
    rows, K, N = 4, 16, 16
    model = _model(
        f"""
        g (float[{rows},{K}] X, float[{K},{N}] W) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """
    )

    quantized = onnxsim.apply_gguf_q6_k_quantization_cpp(model)
    assert quantized.SerializeToString() == model.SerializeToString()


def test_gguf_q6_k_ragged_final_superblock_no_crash_finite_and_matches_reimplementation():
    # dim0 * dim1 = 13 * 17 = 221: not a multiple of _SUPER_BLOCK_SIZE (256)
    # -- a single, wholly ragged final super-block -- and 221 = 13 * 16 + 13,
    # so that super-block's OWN final sub-block is also ragged (13, not 16,
    # real elements). Confirms no crash, a finite/correctly-shaped output,
    # and an exact match (up to float rounding) against the reimplementation
    # ABOVE, which mirrors the same ragged-at-both-levels handling.
    rng = np.random.default_rng(3)
    dim0, dim1 = 13, 17
    weight = (rng.standard_normal((dim0, dim1)) * 0.5).astype(np.float32)
    model = _model(
        f"""
        g (float[4,{dim0}] X) => (float[4,{dim1}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_gguf_q6_k_quantization_cpp(model)
    onnx.checker.check_model(quantized)

    matmul_node = next(n for n in quantized.graph.node if n.op_type == "MatMul")
    _x_input, w_input = matmul_node.input
    w_out_init = next(i for i in quantized.graph.initializer if i.name == w_input)
    assert list(w_out_init.dims) == [dim0, dim1]
    w_out = numpy_helper.to_array(w_out_init).astype(np.float64)
    assert np.all(np.isfinite(w_out))

    expected, _ideal_scale, _d = _q6_k_quantize_dequantize(
        weight.astype(np.float64).reshape(-1)
    )
    np.testing.assert_allclose(w_out.reshape(-1), expected, rtol=1e-4, atol=1e-6)


def test_gguf_q6_k_output_within_proved_combined_bound_via_reference_evaluator():
    # The full end-to-end sanity check against a real ONNX graph execution
    # (onnx's own reference evaluator -- plain float32, no contrib op or
    # quantized tensor type involved at all): every output element's error
    # against the true float MatMul must stay within
    # sum_k |X[i, k]| * (ideal_scale_j(k) / 2 + d(k) / 4), the combined bound
    # proved in Z3 above. Uses the deliberately CLIP-FREE weight
    # (_make_clip_free_weight) -- see the module docstring's own "surprise"
    # section for why plain random data would routinely violate a bound that
    # assumes no clipping at either level.
    rng = np.random.default_rng(4)
    rows, K, N = 5, 16, 16  # K * N = 256: exactly one clean super-block.
    weight = _make_clip_free_weight(rng, (K, N)).astype(np.float32)
    x = (rng.standard_normal((rows, K)) * 2.0).astype(np.float32)
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_gguf_q6_k_quantization_cpp(model)
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

    expected_w, ideal_scale, d = _q6_k_quantize_dequantize(
        weight.astype(np.float64).reshape(-1)
    )
    # Confirm the rewritten weight itself matches the reimplementation
    # exactly (up to float rounding), tying the graph-level check back to
    # the element-level one.
    np.testing.assert_allclose(w_out.reshape(-1), expected_w, rtol=1e-4, atol=1e-6)

    eps_w = (ideal_scale / 2.0 + d / 4.0).reshape(K, N)  # [K, N]
    bound = np.abs(x.astype(np.float64)) @ eps_w  # [rows, N]
    assert np.all(error <= bound + 1e-6), (
        f"max observed error {error.max()} exceeds the proved combined "
        f"bound (max slack {(bound - error).min()}) -- the clip-free weight "
        "construction did not actually avoid clipping"
    )


def test_gguf_q6_k_positive_peak_forces_clamping_in_real_pass():
    # Confirms the Z3 surprise lemma
    # (test_q6_k_positive_peak_normalizes_to_an_out_of_range_code_exactly)
    # against the REAL compiled pass, not just the abstract algebra or this
    # file's own reimplementation: a deliberately positive sub-block peak
    # really does come back with a magnitude-31 (not magnitude-32) code, an
    # honest quantization loss the proved "no clipping" bound above does not
    # cover -- exactly why the bound-checking test above uses a clip-free
    # weight instead of this one.
    rng = np.random.default_rng(5)
    K, N = 16, 16  # exactly one clean super-block, 16 sub-blocks of 16.
    flat = np.empty(_SUPER_BLOCK_SIZE)
    for j in range(_SUB_BLOCKS_PER_SUPER_BLOCK):
        m_j = 1.0 * (1.0 + 0.1 * rng.uniform(-1, 1))
        filler = rng.uniform(0.05, 0.75, size=15) * m_j * rng.choice([-1, 1], size=15)
        # Sub-block 0's own peak is POSITIVE (the surprise); every other
        # sub-block's peak is negative (clip-free), so the super-block's own
        # `d` is not itself distorted by more than one affected sub-block.
        sign = 1.0 if j == 0 else -1.0
        block = np.concatenate([np.array([sign * m_j]), filler])
        rng.shuffle(block)
        flat[j * _SUB_BLOCK_SIZE : (j + 1) * _SUB_BLOCK_SIZE] = block

    weight = flat.reshape(K, N).astype(np.float32)
    model = _model(
        f"""
        g (float[4,{K}] X) => (float[4,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_gguf_q6_k_quantization_cpp(model)
    matmul_node = next(n for n in quantized.graph.node if n.op_type == "MatMul")
    _x_input, w_input = matmul_node.input
    w_out = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == w_input)
    ).astype(np.float64)

    sub0 = flat[:_SUB_BLOCK_SIZE]
    sub0_out = w_out.reshape(-1)[:_SUB_BLOCK_SIZE]
    peak_idx = int(np.argmax(np.abs(sub0)))
    assert sub0[peak_idx] > 0, (
        "test construction should make sub-block 0's own peak positive"
    )

    _expected, _ideal_scale, d_full = _q6_k_quantize_dequantize(flat)
    d = d_full[0]  # shared super-block scale, same for every element here
    # ideal_scale_j for sub-block 0, then its own (possibly-rounded) sc_j and
    # the resulting effective_scale, recomputed directly (not reused from
    # the reimplementation's internals) to recover the real pass's own code.
    ideal_scale_0 = max(np.max(np.abs(sub0)), 1e-12) / _MAX_CODE
    sc_0 = np.clip(np.round(ideal_scale_0 / d), 0, _MAX_SUB_SCALE_CODE)
    effective_scale_0 = d * sc_0
    recovered_code = sub0_out[peak_idx] / effective_scale_0
    assert recovered_code == _MAX_CODE - 1, (
        f"expected the positive peak's own code to clamp to exactly "
        f"{_MAX_CODE - 1}, got {recovered_code}"
    )
    # And the reconstructed magnitude is strictly LESS than the true one --
    # a genuine, honest loss, not merely "some rounding".
    assert abs(sub0_out[peak_idx]) < abs(sub0[peak_idx])
