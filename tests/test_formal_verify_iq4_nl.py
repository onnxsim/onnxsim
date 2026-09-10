"""Formal check for IQ4NL (opt-in; onnxsim's own ``onnxsim/passes/iq4_nl.h``,
C++ port of ``onnxsim/iq4_nl.py``'s own ``apply_iq4_nl_quantization``): like
``weight_only_quantize_mxfp4_matmul`` (READ that file's own test FIRST, this
file only documents what differs -- it is this suite's closest structural
precedent: a FIXED, NON-UNIFORM 4-bit codebook plus a per-block scale), this
is weight-only, data-free quantization of a MatMul/"vanilla" Gemm's constant
weight -- the activation ``X`` is left completely untouched. It rewrites::

    Y = MatMul(X, W) [+ bias]      W constant, 2-D, float32

into::

    Y = MatMul(X, W') [+ bias]     W' -- SAME shape/dtype as W, no new graph
                                     nodes at all (unlike MXFP4's Cast/Gather/
                                     Reshape/Mul chain): every element of W is
                                     replaced, in a brand-new initializer, by
                                     its own 32-element-block IQ4_NL
                                     quantize-dequantize round trip:
                                       dequant = codebook[nearest_code] * scale
                                       scale   = max(|block|) / max(|codebook|)

Only the common, unambiguous shape is matched -- a MatMul, or a Gemm with
transA=0, alpha=1 and beta=1 (bias, if any, left untouched) -- whose weight
(input 1) is a constant 2-D float32 tensor, confirmed by reading
``IQ4NL::patternMatchPredicate``/``runTransform`` (``iq4_nl.h``) line by
line: unlike MXFP4, there is no ``K % block_size == 0`` gate at all -- see
"ragged final block" below for why none is needed.

**Codebook provenance -- an honesty note, transcribed from ``iq4_nl.h``'s own
top-of-file comment (which itself points at ``onnxsim/iq4_nl.py``'s
docstring), not overclaimed here.** llama.cpp's real IQ4_NL format has a
fixed codebook of its own (``kvalues_iq4nl`` in llama.cpp's source), but
neither ``iq4_nl.h`` nor ``iq4_nl.py`` was written against a verified copy of
that table: this repository's own GGUF-decoding code (``ggml_kquant.h``,
``gguf_reconstruct.py``) covers the K-quant family (Q4_K et al.), not any
I-quant format, and a whole-tree grep for ``"IQ4_NL"``/``"kvalues_iq4nl"``
before this change turned up nothing to transcribe from or cross-check
against. So **the 16 values transcribed below are NOT llama.cpp's own
table** -- they are ``iq4_nl.py``'s own, independently and *computationally*
derived codebook (a generalized Lloyd-Max / 1-D k-means scalar quantizer run
to convergence against a standard normal distribution, deterministic, no
random sampling -- see that module's ``_lloyd_max_gaussian_codebook`` and its
own top-of-file docstring for the full derivation), which ``iq4_nl.h``
transcribes verbatim into its own ``iq4_nl_detail::Codebook()`` and which
``tests/test_iq4_nl_cpp.py`` already checks against ``onnxsim.IQ4_NL_CODEBOOK``
directly. It is plausibly similar in shape to llama.cpp's real table (both are
16-level, symmetric, denser near zero) but is not claimed, and must not be
assumed here, to be numerically identical to it.

--------------------------------------------------------------------------
1. The codebook's own worst-case half-gap (genuinely computed, not assumed)
--------------------------------------------------------------------------

The 16 values (``iq4_nl_detail::Codebook()``, ``iq4_nl.h``) are already
sorted ascending and exactly antisymmetric (no value at 0, since 16 is
even), with ``max(|codebook|) == 1.0`` at both ends. Unlike MXFP4's codebook
(exact dyadic literals), these are irrational-looking Lloyd-Max outputs --
this file computes the sorted consecutive gaps IN CODE below (``_GAPS``),
rather than hand-typing them, per this task's own instruction not to
hand-wave the worst case. The result, confirmed by the theorem below rather
than assumed going in: the codebook is DENSEST near zero (a Lloyd-Max
quantizer fit to a Gaussian source over-resolves the high-density center) and
COARSEST at its own two extremes -- the largest gap (``_MAX_GAP``, computed,
not hardcoded) occurs at BOTH ends (index 0, between the two most negative
entries, and its mirror at the two most positive), exactly the same
"worst case is at the extremes, not the center" shape
``test_formal_verify_weight_only_quantize_mxfp4_matmul.py`` found for its own
(differently-shaped, power-of-two) codebook. The worst-case half-gap
(``_HALF_GAP := _MAX_GAP / 2``) is what ``test_iq4_nl_codebook_worst_case_
half_gap_is_a_theorem`` proves is a valid universal bound over the codebook's
own range ``[-1, 1]``, and ``test_iq4_nl_codebook_worst_case_half_gap_lower_
bound_is_tight`` confirms it is not a loose overestimate (a concrete ``x``
sitting exactly between the two extreme entries has EVERY codebook distance
at or above ``_HALF_GAP``, up to double-precision rounding of the
Lloyd-Max-derived literals themselves -- see that test's own comment on why
exact equality isn't asserted the way MXFP4's clean dyadic-literal midpoint
allows).

Like MXFP4, an UNRESTRICTED "for every real x" version of this claim is
FALSE -- ``test_iq4_nl_codebook_unrestricted_domain_claim_is_false`` finds a
genuine Z3 witness (``x`` a bit above 2, well outside ``[-1, 1]``) rather
than merely asserting one exists. The ``[-1, 1]`` domain restriction lines up
with wrinkle (2) below by construction, not by coincidence: it is exactly the
range ``element / scale`` (i.e. ``NearestCodebookValue``'s own argument) is
guaranteed to land in, given how ``scale`` is chosen.

--------------------------------------------------------------------------
2. The block scale: an EXACT LINEAR RATIO, not MXFP4's power-of-two ceiling
--------------------------------------------------------------------------

This is the key structural difference from MXFP4 (confirmed from
``iq4_nl_detail::QuantizeDequantizeBlock``, ``iq4_nl.h``, line by line):
``scale := max(|block|) / max(|codebook|)`` -- a plain division, not
``2 ** ceil(log2(...))``. MXFP4's ceiling-based scale needs its own
INEQUALITY argument (``scale >= m / 6`` from the ceiling's own definition) to
show ``m / scale <= 6.0``, because a power-of-two scale only ever OVER-covers
the codebook's range, never lands exactly on it. IQ4_NL's scale is instead
constructed so the block's own largest-magnitude element maps to EXACTLY
``max(|codebook|)`` -- an EQUALITY, not an inequality -- so
``test_iq4_nl_scale_keeps_block_within_codebook_range`` below is a single
direct-algebra lemma (``element / scale = element * max_abs_codebook / m``,
and ``|element| <= m`` bounds the whole thing by ``max_abs_codebook``
immediately), genuinely simpler than MXFP4's ceiling lemma, with no
``log2``/exponent modeling needed at all -- confirming the task's own
prediction about this format's proof shape.

Combining (1) and (2) gives the pass's own per-element bound::

    |W[k, n] - Wdq[k, n]| <= scale(block_of(k, n)) * _HALF_GAP

proved as ``test_iq4_nl_per_element_error_bound_is_codebook_half_gap_times_
scale`` below, mirroring MXFP4's own composition corollary but with THIS
format's own ``_HALF_GAP`` constant (not ``1.0``) and THIS format's own
scale construction (an equality, not MXFP4's inequality) as the hypothesis.

--------------------------------------------------------------------------
3. A genuinely different block layout: flat storage, no output-channel axis
--------------------------------------------------------------------------

Unlike every other per-block pass in this suite (MXFP4, INT4, INT8-block --
all indexed by ``(block-of-K, output-channel)``), IQ4_NL's blocks are laid
out over the weight's own FLATTENED, ROW-MAJOR storage, "whatever 2-D
shape/layout it already has" (``iq4_nl.h``'s own header comment,
confirmed by ``ReadFloatMatrix`` + the flat ``start/count`` loop in
``runTransform``) -- there is no ``channel_axis``/``weight_transposed``
branch anywhere in this pass at all, unlike ``quantize_matmul_common.h``'s
other quantizers. A consequence worth stating plainly: a single 32-element
block can span MULTIPLE output columns (whenever the weight's own second
dimension isn't itself a multiple of 32), so the natural per-element error
matrix this file's differential tests build is NOT "one scale broadcast down
a whole output channel" the way MXFP4's/INT4's is -- it is one scale per
32-consecutive-FLAT-element run, reshaped back to ``W``'s own 2-D shape.  The
abstract Z3 MAC-bound lemma below (``_bound_formulas``, reused in structure,
down-sized to ``_K = 2``/``_BLOCK_SIZE = 1`` from ``weight_only_quantize_
mxfp4_matmul``'s own per-block generalization) does not need to encode this
distinction -- it only assumes each tap's own error is bounded by SOME
per-tap scale, never that the scale is constant across a whole output
column -- but the differential/numeric tests below build the real per-element
scale matrix explicitly, precisely because it does not follow the
per-channel shape every other block-quant file's own numpy reference does.

**Ragged final block.** Because blocking is over the weight's own flat
element count (``numel = dim0 * dim1``), which need not be a multiple of 32,
``runTransform``'s loop uses ``count = min(kBlockSize, numel - start)`` for
the last block -- no ``K % block_size == 0`` gate is needed (or present) at
all, unlike MXFP4/INT4. The header comment argues this ragged-real-elements-
only approach is mathematically IDENTICAL to ``iq4_nl.py``'s own
zero-pad-then-discard approach (a zero can never be a block's own
largest-magnitude element unless the whole block is already all-zero, in
which case both approaches floor to the same epsilon-derived scale) --
``test_iq4_nl_ragged_final_block_matches_reimplementation`` below exercises a
weight shape whose element count is NOT a multiple of 32 and confirms the
real pass's output against an independent from-scratch reimplementation that
also does NOT zero-pad (mirroring the C++ exactly, not ``iq4_nl.py``'s own
Python reference, so this is a genuine independent check of the pass's own
ragged-block handling, not a round-trip against the pad-based Python
reference it is documented to agree with).

--------------------------------------------------------------------------
Differential tests
--------------------------------------------------------------------------

Built via ``onnx.parser`` (per ``CLAUDE.md``) with ``numpy_helper.from_array``
for the (random, so not text-literal-expressible) weight initializer. This
pass has TWO reachable entry points, both exercised below: the opt-in
optimizer name ``"iq4_nl"`` via ``simplify_isolated_extra`` (confirming the
pass is actually registered and reachable the way every other file in this
suite's own convention checks), and the dedicated Python entry point
``onnxsim.apply_iq4_nl_quantization_cpp`` (``onnxsim/quantize_entry.cpp``'s
``ApplyIQ4NL``, mirroring ``apply_quarot_cpp``'s own dedicated-entry-point
precedent in ``test_formal_verify_quarot.py``) -- used for the detailed
numeric checks below since it hands back a single self-contained rewrite
with no other pass's side effects to account for.

No ``com.microsoft`` contrib op and no quantized ONNX tensor type is ever
involved (confirmed from ``runTransform``: the new initializer is plain
``TensorProto_DataType_FLOAT``, same shape as the original weight, and the
surrounding MatMul/Gemm node is never touched beyond its weight input) -- so,
per this task's own instruction, the numeric checks below use
``onnx.reference.ReferenceEvaluator`` and direct initializer inspection,
never an onnxruntime ``InferenceSession``.

Confirmed below: only MatMul/vanilla-Gemm (not, e.g., a Gemm with ``transA =
1``, and not a non-constant weight) is matched; ``W'`` has the same
shape/dtype as ``W``; the actual quantized values equal an independent
from-scratch Python/numpy reimplementation of ``iq4_nl_detail::
QuantizeDequantizeBlock`` (built fresh here, NOT imported from
``onnxsim.iq4_nl`` or calling back into the C++ pass, so this is a genuine
cross-check); every dequantized value equals ``scale * codebook[i]`` for
some codebook index ``i``, exactly (up to float32 rounding); the ragged
final block case; and the real end-to-end ``ReferenceEvaluator`` output
stays within the proved ``scale * _HALF_GAP`` per-element bound against the
true float MatMul.

Every bound-checking differential test uses a genuinely lossy comparison
(this suite's usual ``check_n=0`` for ``simplify_isolated_extra``, and no
``check_ok``-style equivalence check at all for the dedicated entry point,
which -- like ``apply_quarot_cpp`` -- returns a rewritten model directly with
no built-in equivalence check of its own).
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser
from onnx.reference import ReferenceEvaluator

import onnxsim

# Transcribed verbatim from iq4_nl_detail::Codebook() (onnxsim/passes/
# iq4_nl.h), NOT re-derived from memory of any published llama.cpp table --
# see this module's own docstring for the full provenance/honesty note. Also
# matches onnxsim.IQ4_NL_CODEBOOK (onnxsim/iq4_nl.py) byte-for-byte, but
# transcribed independently here (not imported) so the differential tests
# below are a genuine cross-check of iq4_nl.h's own literal array, not a
# round-trip against the same Python constant.
_IQ4_NL_CODEBOOK = [
    -1.0,
    -0.757290980404998,
    -0.5923241681050287,
    -0.45994900549399614,
    -0.34508882475608954,
    -0.24055736084608642,
    -0.1421631738251121,
    -0.04704566832784618,
    0.04704566832784618,
    0.1421631738251121,
    0.24055736084608642,
    0.34508882475608954,
    0.45994900549399614,
    0.5923241681050287,
    0.757290980404998,
    1.0,
]
_MAX_ABS_CODEBOOK = max(abs(c) for c in _IQ4_NL_CODEBOOK)
assert _MAX_ABS_CODEBOOK == 1.0

_BLOCK_SIZE = 32

# The codebook's own true worst-case half-gap, computed from its OWN sorted
# consecutive gaps (never hand-waved as "half the smallest gap" -- see this
# module's own docstring, and weight_only_quantize_mxfp4_matmul's own file
# for why that naive assumption is generally wrong).
_SORTED_CODEBOOK = sorted(_IQ4_NL_CODEBOOK)
_GAPS = [b - a for a, b in zip(_SORTED_CODEBOOK, _SORTED_CODEBOOK[1:])]
_MAX_GAP = max(_GAPS)
_MAX_GAP_INDEX = _GAPS.index(_MAX_GAP)
_HALF_GAP = _MAX_GAP / 2.0
_MIDPOINT_AT_WORST_GAP = (
    _SORTED_CODEBOOK[_MAX_GAP_INDEX] + _SORTED_CODEBOOK[_MAX_GAP_INDEX + 1]
) / 2.0

# Sanity-check the shape this module's docstring claims, in plain Python
# before any Z3 is invoked: the codebook is densest near zero, coarsest at
# its own two extremes, and the two largest gaps (at both ends, by symmetry)
# tie for the overall worst case.
assert _MAX_GAP_INDEX == 0, (
    "expected the worst-case gap at the negative extreme (index 0); the "
    "codebook's own density profile may have changed"
)
assert abs(_GAPS[-1] - _MAX_GAP) < 1e-12, "the two extremes should tie by symmetry"
assert _GAPS == sorted(_GAPS[: len(_GAPS) // 2], reverse=True) + sorted(
    _GAPS[len(_GAPS) // 2 :]
), "expected gaps to shrink toward the center and grow back out symmetrically"


def _abs(v):
    return z3.If(v >= 0, v, -v)


# --- 1. The codebook's own worst-case half-gap -------------------------------


def test_iq4_nl_codebook_worst_case_half_gap_is_a_theorem():
    # For every real x within the codebook's own range [-1, 1] (the domain
    # NearestCodebookValue is always actually called on -- see wrinkle (2)
    # below for why), some codebook entry is within _HALF_GAP of x. _HALF_GAP
    # is derived from the codebook's own real sorted gaps above, not assumed.
    x = z3.Real("x")
    in_range = z3.And(x >= -_MAX_ABS_CODEBOOK, x <= _MAX_ABS_CODEBOOK)
    prove(
        z3.Implies(
            in_range,
            z3.Or(*[_abs(x - z3.RealVal(c)) <= _HALF_GAP for c in _IQ4_NL_CODEBOOK]),
        ),
        msg=f"{_HALF_GAP} is not a valid worst-case half-gap bound for the IQ4_NL codebook",
    )


def test_iq4_nl_codebook_worst_case_half_gap_lower_bound_is_tight():
    # Companion to the theorem above: _HALF_GAP is not a loose overestimate.
    # x sitting exactly between the codebook's own two most extreme entries
    # (the worst gap, computed above -- NOT MXFP4's clean dyadic x = 5.0) has
    # every codebook distance AT OR ABOVE _HALF_GAP. Unlike MXFP4's exact
    # dyadic-literal midpoint, this codebook's values are Lloyd-Max floating-
    # point outputs, so the midpoint computed in ordinary float64 arithmetic
    # only matches _HALF_GAP up to ~1e-16 rounding (confirmed below, not
    # asserted to be bit-exact) -- still utterly negligible next to the
    # ~0.1-scale margins this bound is about.
    distances = [abs(_MIDPOINT_AT_WORST_GAP - c) for c in _IQ4_NL_CODEBOOK]
    assert abs(min(distances) - _HALF_GAP) < 1e-9, distances
    assert all(d >= _HALF_GAP - 1e-9 for d in distances)

    # A genuine Z3 confirmation of tightness, mirroring MXFP4's own
    # "threshold just below the proved bound is NOT valid" check: 0.121 is
    # comfortably below _HALF_GAP (~0.121354...), so Z3 must find some real x
    # (the worst-gap midpoint above is itself a witness) whose every codebook
    # distance exceeds 0.121.
    threshold = 0.121
    assert threshold < _HALF_GAP
    xx = z3.Real("x")
    solver = z3.Solver()
    solver.add(
        z3.And(*[_abs(xx - z3.RealVal(c)) > threshold for c in _IQ4_NL_CODEBOOK])
    )
    assert solver.check() == z3.sat, (
        f"every codebook entry is within {threshold} of every real x -- the "
        f"worst-case half-gap would then be < {_HALF_GAP}, contradicting the "
        "midpoint computation above"
    )


def test_iq4_nl_codebook_unrestricted_domain_claim_is_false():
    # SURPRISE (caught by Z3, mirroring MXFP4's own analogous finding, not
    # assumed going in): an UNRESTRICTED "for every real x" version of the
    # theorem above is FALSE. A genuine witness, found by Z3 rather than
    # merely asserted to exist: x a bit above the codebook's own max
    # magnitude of 1.0, where nothing bounds the distance once x leaves
    # [-1, 1] -- the domain restriction in the theorem above is load-bearing,
    # not a weakening of convenience.
    x = z3.Real("x")
    solver = z3.Solver()
    solver.add(z3.And(*[_abs(x - z3.RealVal(c)) > _HALF_GAP for c in _IQ4_NL_CODEBOOK]))
    assert solver.check() == z3.sat, (
        "every real x is within _HALF_GAP of some codebook entry -- the "
        "domain restriction to [-1, 1] would then not be load-bearing"
    )
    (x_val,) = (solver.model()[d] for d in solver.model() if str(d) == "x")
    x_float = float(x_val.as_fraction())
    assert abs(x_float) > _MAX_ABS_CODEBOOK, (
        f"expected a witness outside the codebook's own range, got x = {x_float}"
    )


# --- 2. The block scale: exact linear ratio, not a power-of-two ceiling -----


def test_iq4_nl_scale_keeps_block_within_codebook_range():
    # scale := max(|block|) / max(|codebook|), i.e. m / M for m := max_abs
    # of the block (assumed > 0; the all-zero-block/epsilon-floor case is a
    # degenerate corner iq4_nl_detail::QuantizeDequantizeBlock handles
    # separately and trivially -- every element is already exactly 0 then).
    # For ANY element of that block, |element| <= m by definition of m as
    # the block's own max-abs, so:
    #   |element / scale| = |element| * M / m <= m * M / m = M
    # A single direct-algebra step -- no ceiling/log2 modeling needed at all,
    # genuinely simpler than MXFP4's inequality-based ceiling lemma (see this
    # module's own docstring).
    element, m, big_m, scale = z3.Reals("element m big_m scale")
    hypotheses = z3.And(
        m > 0,
        big_m > 0,
        scale == m / big_m,
        _abs(element) <= m,
    )
    prove(z3.Implies(hypotheses, _abs(element / scale) <= big_m))


def test_iq4_nl_scale_construction_is_an_equality_not_merely_a_bound():
    # The genuinely different construction vs. MXFP4's ceiling (an
    # INEQUALITY: scale >= m / M): here the block's own largest-magnitude
    # element maps to EXACTLY M (max(|codebook|)), by direct substitution --
    # scale is defined so that the ELEMENT ACHIEVING m itself normalizes to
    # exactly M, not merely "within M". This is what makes wrinkle (1)'s
    # codebook lemma tight at the boundary meaningful: the boundary is
    # actually reached, not just approached.
    m, big_m = z3.Reals("m big_m")
    scale = m / big_m
    hypotheses = z3.And(m > 0, big_m > 0)
    prove(z3.Implies(hypotheses, m / scale == big_m))


# --- Composition: codebook half-gap x block scale = per-element bound ------


def test_iq4_nl_per_element_error_bound_is_codebook_half_gap_times_scale():
    # The corollary tying (1) and (2) into the single-element claim the MAC
    # bound below is built on, mirroring MXFP4's own analogous corollary but
    # with THIS format's own _HALF_GAP constant (not 1.0):
    # normalized := W / scale lands within _HALF_GAP of some codebook entry c
    # (taken as a hypothesis here, valid precisely because lemma (2) above
    # guarantees normalized never leaves the codebook's own [-1, 1] range),
    # and Wdq := c * scale, so
    #   |W - Wdq| = |normalized * scale - c * scale| = |normalized - c| * scale
    #             <= _HALF_GAP * scale.
    w, scale, normalized, c = z3.Reals("w scale normalized c")
    hypotheses = z3.And(
        scale > 0,
        normalized == w / scale,
        _abs(normalized - c) <= _HALF_GAP,
    )
    wdq = c * scale
    error = w - wdq
    bound = _HALF_GAP * scale
    prove(z3.Implies(hypotheses, z3.And(error <= bound, -error <= bound)))


# --- The per-block MAC bound (eps_w := _HALF_GAP * Ws[block], not Ws/2) -----

_K = 2  # matches quantized_mac_bound's / weight_only_quantize_mxfp4_matmul's
# own minimal-case convention: two SEPARATE one-element blocks is already the
# smallest layout exercising independent per-block scales.
_BLOCK_SIZE_ABSTRACT = 1  # each tap is its own block: Ws0 for tap 0, a
# genuinely different Ws1 for tap 1. (Named distinctly from the real format's
# _BLOCK_SIZE = 32 above -- this is only the abstract Z3 model's own
# minimal-case block size, unrelated to the real pass's constant.)
_NUM_BLOCKS = _K // _BLOCK_SIZE_ABSTRACT


def _block_of(k):
    return k // _BLOCK_SIZE_ABSTRACT


def _bound_formulas():
    """Z3 vocabulary for the single-operand, PER-BLOCK bounded-error claim,
    reused-in-structure (down-sized to ``_K = 2``) from
    ``weight_only_quantize_mxfp4_matmul``'s own per-block generalization:
    ``X`` has no error term at all (never quantized), only ``W`` does, via
    the free per-tap error variable ``ew``, with one scale variable PER BLOCK
    (``Ws[b]``). The constant multiplying each block's scale is THIS format's
    own ``_HALF_GAP`` (not MXFP4's ``1.0`` or the affine passes' ``0.5``).
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # X[i, k], true float activation
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[k, n], true float weight
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[k, n] - Wdq[k, n]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]  # per-block scale

    dequant_w = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= _HALF_GAP * Ws[_block_of(k)] for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))

    bound = sum((_HALF_GAP * Ws[_block_of(k)]) * _abs(X[k]) for k in range(_K))

    return float_matmul, dequant_matmul, rounding_bounds, bound


def test_iq4_nl_error_is_bounded():
    # The genuine bounded-error claim: given W's own per-BLOCK rounding
    # bound and X completely unchanged, the true float dot product and the
    # one computed against the dequantized weight cannot differ by more than
    # sum_k (_HALF_GAP * Ws[block_of(k)]) * |X[i, k]|.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_iq4_nl_bias_variant_error_is_bounded():
    # The "+ Bias" branch (a Gemm with a bias input): this pass never
    # touches Gemm's bias C at all, so adding the same Bias(n) to both the
    # true and the dequantized computation leaves their difference -- and
    # therefore the bound on it -- unchanged.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_matmul + bias) - (dequant_matmul + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_iq4_nl_negative_control_requires_rounding_bound():
    # Sanity check that the bound proved above is genuine, not vacuous: with
    # no error budget assumed on ew at all (only Ws0, Ws1 > 0), the same
    # bound is not a theorem -- Z3 must find a real counterexample.
    float_matmul, dequant_matmul, _rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul

    solver = z3.Solver()
    solver.add(z3.Not(z3.And(error <= bound, -error <= bound)))
    assert solver.check() == z3.sat, (
        "the bound holds even without any rounding-error budget on ew -- "
        "negative control is vacuous"
    )


def test_iq4_nl_uniform_bound_is_unsound_across_blocks():
    # Reproduces weight_only_quantize_mxfp4_matmul's own "naive single-
    # shared-scale bound is unsound across blocks" negative control, for
    # THIS format's own _HALF_GAP constant: if one instead (incorrectly)
    # bounded every tap's error using block 0's scale alone
    # (_HALF_GAP * Ws0), that claim is NOT a theorem once block 1's real
    # scale Ws1 can exceed Ws0 -- a genuine possibility here precisely
    # because (wrinkle 3 above) IQ4_NL's blocks have no shared output-channel
    # structure forcing scales to line up.
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]

    dequant_w = [W[k] - ew[k] for k in range(_K)]
    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= _HALF_GAP * Ws[_block_of(k)] for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))
    error = float_matmul - dequant_matmul

    uniform_bound = (_HALF_GAP * Ws[0]) * sum(_abs(X[k]) for k in range(_K))

    solver = z3.Solver()
    solver.add(rounding_bounds)
    solver.add(z3.Not(z3.And(error <= uniform_bound, -error <= uniform_bound)))
    assert solver.check() == z3.sat, (
        "the naive single-scale (block 0 only) bound holds even though "
        "block 1 has its own, potentially larger scale -- this pass's "
        "per-block sum is not actually load-bearing, which would be wrong"
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


def _nearest_iq4_nl_value(normalized):
    """Independent from-scratch numpy nearest-codebook-value search (not
    imported from onnxsim.iq4_nl): for each value, the codebook entry
    closest to it (ties broken toward the lower index, matching
    NearestCodebookValue's own strict '<' comparison over the ascending-
    sorted codebook, which numpy.argmin's own first-occurrence tie-break
    over the same ascending array reproduces identically).
    """
    codebook = np.asarray(_IQ4_NL_CODEBOOK, dtype=np.float64)
    diffs = np.abs(normalized[..., np.newaxis] - codebook[np.newaxis, ...])
    return codebook[np.argmin(diffs, axis=-1)]


def _iq4_nl_quantize_dequantize_matrix(weight):
    """Independent from-scratch numpy re-implementation of
    ``iq4_nl_detail::QuantizeDequantizeBlock``/``IQ4NL::runTransform``
    (``onnxsim/passes/iq4_nl.h``): flattens ``weight`` in ROW-MAJOR order
    (matching ``ReadFloatMatrix``'s own on-disk layout -- there is no
    channel_axis/transpose branch in the real pass at all, see this module's
    own docstring wrinkle (3)), computes one scale per 32-element block of
    the FLAT array (the final block RAGGED -- using only its own real
    elements, NOT zero-padded -- when the flat element count is not itself a
    multiple of 32, exactly matching ``runTransform``'s own
    ``count = min(kBlockSize, numel - start)`` loop, NOT ``iq4_nl.py``'s own
    zero-pad-then-discard Python reference), and returns the same shape,
    each element replaced by ``codebook[nearest_code] * scale``.

    Written independently here (own control flow, own loop) rather than
    calling into ``onnxsim.iq4_nl``'s ``quantize_dequantize_iq4_nl`` or the
    compiled pass itself, so this is a genuine cross-check of the real
    pass's own math, not a round-trip against either.
    """
    original_shape = weight.shape
    flat = np.asarray(weight, dtype=np.float64).reshape(-1)
    n = flat.size
    out = np.empty_like(flat)
    for start in range(0, n, _BLOCK_SIZE):
        block = flat[start : start + _BLOCK_SIZE]
        max_abs = float(np.max(np.abs(block))) if block.size else 0.0
        scale = max(max_abs, 1e-12) / _MAX_ABS_CODEBOOK
        normalized = block / scale
        nearest = _nearest_iq4_nl_value(normalized)
        out[start : start + block.size] = nearest * scale
    return out.reshape(original_shape)


def _iq4_nl_scale_matrix(weight):
    """Same flattening/blocking as ``_iq4_nl_quantize_dequantize_matrix``,
    but returns, per element, the SCALE of the block that element belongs
    to (same shape as ``weight``) -- used to build the real per-element
    error-bound matrix ``scale * _HALF_GAP`` for the numeric bound check
    below, since (wrinkle 3) a block's scale is not simply "per output
    channel" here.
    """
    original_shape = weight.shape
    flat = np.asarray(weight, dtype=np.float64).reshape(-1)
    n = flat.size
    out = np.empty_like(flat)
    for start in range(0, n, _BLOCK_SIZE):
        block = flat[start : start + _BLOCK_SIZE]
        max_abs = float(np.max(np.abs(block))) if block.size else 0.0
        scale = max(max_abs, 1e-12) / _MAX_ABS_CODEBOOK
        out[start : start + block.size] = scale
    return out.reshape(original_shape)


def test_iq4_nl_pass_fires_and_matches_reimplementation_via_extra_optimizers():
    # Confirms the pass is reachable via the usual extra_optimizers path
    # (this suite's standard "is it actually registered" check) and rewrites
    # the weight in place (SAME shape/dtype, a brand-new initializer name --
    # no new graph nodes at all, unlike MXFP4's Cast/Gather/Reshape/Mul
    # chain). K * N = 64 = exactly two 32-element blocks.
    rng = np.random.default_rng(0)
    rows, K, N = 4, 8, 8
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

    sim_model, _ops = simplify_isolated_extra(model, "iq4_nl", check_n=0)

    matmul_node = producer(sim_model, "Y")
    assert matmul_node.op_type == "MatMul"
    x_input, w_input = matmul_node.input
    assert x_input == "X"  # activation passed through completely unchanged
    assert w_input != "W"  # rewritten to a brand-new initializer

    w_out_init = next(i for i in sim_model.graph.initializer if i.name == w_input)
    assert w_out_init.data_type == onnx.TensorProto.FLOAT
    assert list(w_out_init.dims) == [K, N]

    w_out = numpy_helper.to_array(w_out_init)
    expected = _iq4_nl_quantize_dequantize_matrix(weight.astype(np.float64)).astype(
        np.float32
    )
    np.testing.assert_allclose(w_out, expected, rtol=1e-5, atol=1e-6)


def test_iq4_nl_gemm_bias_untouched_and_blocked_by_own_flat_storage():
    # A "vanilla" Gemm (transA=0, alpha=1, beta=1) with a bias: the bias is
    # passed through unchanged. transB=1 here (weight stored as [N, K],
    # PyTorch nn.Linear layout) -- confirms the pass blocks over the
    # weight's OWN [N, K] flat storage directly (wrinkle 3: no
    # weight_transposed-aware channel logic the way MXFP4/INT4 have), not
    # some canonicalized [K, N] view.
    rng = np.random.default_rng(1)
    rows, K, N = 3, 20, 5  # N * K = 100, not a multiple of 32 -- also
    # exercises a ragged final block in the same test.
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

    quantized = onnxsim.apply_iq4_nl_quantization_cpp(model)
    onnx.checker.check_model(quantized)

    gemm_node = next(n for n in quantized.graph.node if n.op_type == "Gemm")
    x_input, w_input, b_input = gemm_node.input
    assert x_input == "X"
    assert b_input == "B"  # bias untouched

    w_out_init = next(i for i in quantized.graph.initializer if i.name == w_input)
    assert list(w_out_init.dims) == [N, K]
    w_out = numpy_helper.to_array(w_out_init)

    # Blocked over the weight's OWN [N, K] storage, not transposed to [K, N]
    # first.
    expected = _iq4_nl_quantize_dequantize_matrix(weight.astype(np.float64)).astype(
        np.float32
    )
    np.testing.assert_allclose(w_out, expected, rtol=1e-5, atol=1e-6)


def test_iq4_nl_declines_transposed_activation_gemm():
    # MatchMatMulLike (quantize_matmul_common.h) requires transA == 0;
    # patternMatchPredicate must therefore decline a transA=1 Gemm outright,
    # leaving the model byte-for-byte unchanged.
    rng = np.random.default_rng(2)
    K, rows, N = 6, 4, 3  # X stored as [K, rows] so X^T @ W is well-typed.
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

    quantized = onnxsim.apply_iq4_nl_quantization_cpp(model)
    assert quantized.SerializeToString() == model.SerializeToString()


def test_iq4_nl_declines_non_constant_weight():
    # patternMatchPredicate requires FetchConstantTensor(info.w) to succeed;
    # a weight that is a genuine graph INPUT (not any constant/initializer at
    # all) must be left untouched.
    rows, K, N = 4, 8, 8
    model = _model(
        f"""
        g (float[{rows},{K}] X, float[{K},{N}] W) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """
    )

    quantized = onnxsim.apply_iq4_nl_quantization_cpp(model)
    assert quantized.SerializeToString() == model.SerializeToString()


def test_iq4_nl_ragged_final_block_matches_reimplementation():
    # Weight element count NOT a multiple of 32 (5 * 7 = 35 = one full
    # 32-element block + a ragged 3-element final block) -- confirms the
    # real pass's own ragged handling (real elements only, no zero-padding)
    # against the same reimplementation, and that the ragged block's own
    # scale comes from only its own 3 real elements.
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

    quantized = onnxsim.apply_iq4_nl_quantization_cpp(model)
    matmul_node = next(n for n in quantized.graph.node if n.op_type == "MatMul")
    _x_input, w_input = matmul_node.input
    w_out = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == w_input)
    )

    expected = _iq4_nl_quantize_dequantize_matrix(weight.astype(np.float64)).astype(
        np.float32
    )
    np.testing.assert_allclose(w_out, expected, rtol=1e-5, atol=1e-6)

    # The ragged final block (elements 32..34, flat row-major) has its own
    # scale from only those 3 real elements -- confirmed directly, not just
    # via the reimplementation above.
    flat = weight.astype(np.float64).reshape(-1)
    ragged_block = flat[32:35]
    assert ragged_block.size == 3
    expected_scale = max(np.max(np.abs(ragged_block)), 1e-12) / _MAX_ABS_CODEBOOK
    out_flat = w_out.astype(np.float64).reshape(-1)
    out_ragged = out_flat[32:35]
    # Every value in the ragged block is exactly scale * some codebook entry.
    recovered_codes = out_ragged / expected_scale
    nearest = _nearest_iq4_nl_value(recovered_codes)
    np.testing.assert_allclose(nearest, recovered_codes, atol=1e-4)


def test_iq4_nl_dequantized_values_are_exact_codebook_times_scale():
    # Every dequantized weight value equals scale(block) * (some codebook
    # entry) exactly (up to float32 rounding) -- the crux structural check
    # that the real pass's output actually IS a codebook lookup, not merely
    # "close to" one.
    rng = np.random.default_rng(4)
    rows, K, N = 6, 9, 11  # K * N = 99: two full blocks + a 3-element ragged
    # tail with the added texture of a non-square shape.
    weight = rng.standard_normal((K, N)).astype(np.float32) * 1.3
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_iq4_nl_quantization_cpp(model)
    matmul_node = next(n for n in quantized.graph.node if n.op_type == "MatMul")
    _x_input, w_input = matmul_node.input
    w_out = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == w_input)
    ).astype(np.float64)

    scale = _iq4_nl_scale_matrix(weight.astype(np.float64))
    codes = w_out / scale
    nearest = _nearest_iq4_nl_value(codes)
    np.testing.assert_allclose(codes, nearest, atol=1e-4)

    # And the observed per-element error stays within the proved bound.
    error = np.abs(weight.astype(np.float64) - w_out)
    bound = scale * _HALF_GAP
    assert np.all(error <= bound + 1e-6)


def test_iq4_nl_output_within_proved_bound_via_reference_evaluator():
    # The full end-to-end sanity check against a real ONNX graph execution
    # (onnx's own reference evaluator -- no onnxruntime needed, since this
    # rewrite is plain float32 with no quantized tensor type or contrib op
    # involved at all): every output element's error against the true float
    # MatMul must stay within sum_k |X[i, k]| * scale[k, n] * _HALF_GAP,
    # where scale[k, n] is the REAL per-element scale matrix (wrinkle 3:
    # not a per-output-channel broadcast, since a block can span columns).
    rng = np.random.default_rng(5)
    rows, K, N = 5, 10, 9  # K * N = 90: not a multiple of 32, exercising a
    # ragged final block inside the end-to-end check too.
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

    quantized = onnxsim.apply_iq4_nl_quantization_cpp(model)
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

    scale = _iq4_nl_scale_matrix(weight.astype(np.float64))  # [K, N]
    eps_w = scale * _HALF_GAP  # [K, N], per-element (not per-channel) error budget
    bound = np.abs(x.astype(np.float64)) @ eps_w  # [rows, N]
    assert np.all(error <= bound + 1e-4)

    # Also confirm the rewritten weight itself matches the independent
    # reimplementation exactly (up to float32 rounding), tying the graph-
    # level check back to the element-level one.
    expected_w = _iq4_nl_quantize_dequantize_matrix(weight.astype(np.float64))
    np.testing.assert_allclose(w_out, expected_w, rtol=1e-5, atol=1e-6)
