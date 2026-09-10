"""Formal check for Fp6Llm (opt-in; onnxsim's own ``onnxsim/passes/fp6_llm.h``,
"FP6-LLM", Xia et al. 2024): the same "only the constant weight is
quantized, the activation ``X`` is left completely untouched, no
calibration data" design as every other weight-only-quantization file in
this suite -- READ ``test_formal_verify_weight_only_quantize_mxfp4_matmul.py``
FIRST, this file only documents what differs -- but FP6's own codebook is
built by a GENUINELY DIFFERENT construction than MXFP4's fixed, hand-typed
16-value E2M1 codebook: FP6 E3M2's 64 codes are ENUMERATED from ordinary
exponent/mantissa floating-point arithmetic (confirmed by reading
``Fp6Codebook()`` in ``passes/fp6_llm.h`` line by line, not assumed to
match MXFP4's shape or any recalled constant), and its per-block scale is
a plain LINEAR ratio (``max(|block|) / 28.0``, GGUF/IQ4_NL's own
convention), NOT MXFP4's power-of-two ``ceil(log2(...))`` scale -- so
this file's lemma (2) below is a different (and, it turns out, a
*strictly stronger*) shape than MXFP4's own.

It rewrites ``Y = MatMul(X, W)`` (or a "vanilla" Gemm -- ``transA=0``,
``alpha=1``, ``beta=1``, bias left untouched -- the same scope
``gguf_legacy_quant.h``/``iq4_nl.h`` use, confirmed via
``quantize_matmul_common.h``'s own ``MatchMatMulLike``) -- ``W`` a
constant 2-D FLOAT32 tensor -- by replacing every element with its own
64-element-block FP6 E3M2 quantize-dequantize round trip, laid out over
``W``'s own flattened, ROW-MAJOR storage (NOT per (output-channel,
K-block) the way MXFP4/INT4 are) -- a ragged final block (when the
weight's element count isn't itself a multiple of 64) is quantized using
only its own real elements, since appending zero padding can never
change a block's own ``max(|.|)`` unless the whole block is already
all-zero (``passes/fp6_llm.h``'s own comment). Unlike every
``DequantizeLinear``/``Cast``-``Gather``-``Reshape``-``Mul``-based pass
in this suite, there is no dequantization graph at all: ``runTransform``
computes the round trip once, at pass-transform time, and writes it
straight back as a new plain FLOAT32 initializer -- so this file's
differential tests inspect a rewritten initializer directly, no
``com.microsoft`` op, opset-21 sub-8-bit tensor type, or ONNX Runtime
execution needed anywhere.

**1. The codebook is FIXED but NON-UNIFORM, and built by enumeration, not
a hand-typed table.** E3M2 (1 sign, 3 exponent, 2 mantissa bits, bias 3 --
derived in ``passes/fp6_llm.h``'s own top comment from
``ml_dtypes.float6_e3m2fn``'s empirically-measured max finite magnitude
of 28.0) gives, per code, ``value = (1 + mantissa / 4) * 2**(exponent -
3)`` for a normal code (``exponent`` code != 0) or ``value = (mantissa /
4) * 2**-2`` for a subnormal one (``exponent`` code 0, matching the
standard float definition's ``2**(1 - bias)`` subnormal scale). This
file builds that exact enumeration fresh, in Python
(``_build_fp6_e3m2_codebook`` below -- mirroring ``Fp6Codebook()``'s own
triple loop over sign/exponent/mantissa, not imported from
``onnxsim.fp6_llm`` or ``ml_dtypes``), rather than typing out 64 literal
``z3.RealVal`` constants by hand the way the MXFP4 file's tiny 16-entry
codebook could -- then feeds the resulting CONCRETE list into Z3's own
``z3.Or`` disjunction, exactly like the MXFP4 file does with its own
hand-typed list. Sign gives +/-, and +0/-0 collapse to one entry (0 and
-0 are the same real, IEEE-754-style), so 64 raw codes yield 63 DISTINCT
reals -- confirmed by construction (``len(...) == 63``), the same
"one-slot-collapse" shape the MXFP4 file's own 16-slots/15-distinct
codebook has, just at a different size.

Sorted, the positive half is::

    0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,   (subnormals)
    0.5, 0.625, 0.75, 0.875,                                  (exp code 1)
    1, 1.25, 1.5, 1.75,                                       (exp code 2)
    2, 2.5, 3, 3.5,                                           (exp code 3)
    4, 5, 6, 7,                                                (exp code 4)
    8, 10, 12, 14,                                             (exp code 5)
    16, 20, 24, 28                                             (exp code 6)

and the 31 gaps between consecutive positive/zero entries are, in order,
8 copies of 0.0625, 4 of 0.125, 4 of 0.25, 4 of 0.5, 4 of 1.0, 4 of 2.0,
and 3 of **4.0** -- computed by ``_build_fp6_e3m2_codebook``'s own sorted
output, not asserted by hand (see the diff of consecutive entries this
module's own test derives). Just like MXFP4, the gaps are geometrically
widening with the exponent, NOT constant, so "nearest codebook value"
error is bounded by half the LARGEST gap (4.0), not half the smallest
(0.0625, which would give 0.03125 -- wrong, and much too tight, exactly
the naive mistake MXFP4's docstring warns about). The TIGHT worst-case
half-gap is therefore **2.0**, achieved at every midpoint of the format's
three widest-spaced (highest-exponent) intervals: ``x = 18`` (halfway
between 16 and 20), ``x = 22`` (between 20 and 24), and ``x = 26``
(between 24 and 28) -- and their negatives -- each exactly distance 2.0
from its two nearest codebook entries, with the next-nearest entry twice
as far (e.g. at ``x = 18``, entries 16 and 20 are each 2.0 away, 14 and
24 are each 4.0 away). ``test_fp6_codebook_worst_case_half_gap_of_two_is_a_theorem``
proves ``forall x in [-28, 28], exists c in codebook, |x - c| <= 2.0``
over the full 63-entry codebook;
``test_fp6_codebook_worst_case_half_gap_lower_bound_is_tight`` confirms
2.0 is not a loose overestimate at ``x = 18``; and
``test_fp6_codebook_naive_half_smallest_gap_bound_is_unsound`` is this
file's own version of the MXFP4 docstring's "surprise caught by Z3"
moment -- 0.03125 (half the SMALLEST gap, the number a reader skimming
only the subnormal region might reach for) is not a valid universal
bound at all; Z3 finds a genuine counterexample within the very domain
the real bound is proved over.

SURPRISE, caught by Z3 rather than assumed while writing this file
(mirroring the MXFP4 file's own "x = -8" moment, not merely repeating
it): an UNRESTRICTED "for every real x" version of the half-gap claim is
also FALSE here -- Z3's own counterexample search finds points such as
``x = 40``, far outside [-28, 28], where nothing bounds the distance at
all. The domain restriction is exactly what lemma (2) below guarantees
``normalized := W / scale`` always lands in before ``NearestFp6Value`` is
ever called on it (``test_fp6_codebook_unrestricted_domain_breaks_the_bound``).

**2. The block scale is a plain LINEAR ratio, NOT a power of two --
confirmed from ``passes/fp6_llm.h`` itself, not assumed to match
MXFP4's convention:** ``QuantizeDequantizeFp6Block`` computes ``scale =
max(max_abs, 1e-12) / Fp6Max()`` (``Fp6Max() == 28.0``, the codebook's
own max magnitude) -- an ordinary division, matching GGUF/IQ4_NL's own
scale convention, not MXFP4's ``2**ceil(log2(max_abs / 6.0))``. This
makes lemma (2) here a *simpler and strictly tighter* claim than MXFP4's
own ceil-vs-floor lemma: there is no rounding of the scale itself at all,
so for any block with ``max_abs > 0`` (at or above the ``1e-12`` floor),
the block's own largest-magnitude element lands EXACTLY at the
codebook's max magnitude (``max_abs / scale == 28.0``, an equality, not
merely ``<= 28.0``) -- unlike MXFP4's ceil-based scale, which only
guarantees ``<=`` and can leave the block underutilizing the codebook's
range by up to 2x. ``test_fp6_linear_scale_keeps_block_within_codebook_range``
proves the ``<=`` direction (the one the codebook lemma's domain
restriction actually needs, stated generally for every element of the
block, not just its max); ``test_fp6_linear_scale_lands_the_blocks_own_max_exactly_at_codebook_max``
proves the exact-equality property as its own explicit corollary, to be
honest about how this scale formula is a strictly tighter animal than
MXFP4's rather than silently reusing "guarantees <=" language that would
undersell it. (The ``max(..., 1e-12)`` floor only matters for an
all-zero block, where it keeps ``scale`` strictly positive -- both lemmas
below are stated to cover that edge with an explicit ``z3.If``, not
assumed away.)

Combining (1) and (2) gives the pass's overall per-element bound::

    |W[k, n] - Wdq[k, n]| <= 2.0 * scale[block_of(flat_index(k, n))]

-- the codebook's own worst-case half-gap (2.0) TIMES the block's own
linear scale, composed in ``test_fp6_per_element_error_bound_is_codebook_half_gap_times_scale``
exactly the same way the MXFP4 file's own corollary composes its ``1.0
* Ws``.

This is still the single-operand (``eps_x := 0``) collapse of
``quantized_mac_bound``'s general MAC bound
(``test_formal_verify_quantized_mac_bound.py``), with a genuinely
PER-BLOCK ``eps_w := 2.0 * scale[block_of(k)]`` term -- the per-block-SUM
structure below is reused (down-sized to ``_K = 2``, ``_BLOCK_SIZE = 1``,
the same minimal-case convention ``weight_only_quantize_mxfp4_matmul``'s
own file uses: two separate one-element blocks is already the smallest
layout exercising independent per-block scales; FP6's own real block
size is 64, see ``_FP6_BLOCK_SIZE`` below and the differential tests)
from ``test_formal_verify_weight_only_quantize_mxfp4_matmul.py``'s own
per-block generalization, including its own "a naive single-shared-scale
bound is unsound across blocks" negative control -- but the constant
multiplying the scale is this file's own ``2.0``, derived from the
codebook lemma above, NOT MXFP4's ``1.0``. Every bound-proving query uses
the DIRECT-ERROR-VARIABLE idiom (a free ``ew`` bounded directly by ``2.0
* scale[block]``, not reconstructed from codebook-value/scale
multiplicands inside the same query) for the same Z3-hang-avoidance
reason the MXFP4 file's own docstring cites.

Differential tests build a plain float ``MatMul``/``Gemm`` via
``onnx.parser`` (per ``CLAUDE.md``) with ``numpy_helper.from_array`` for
random weight initializers (per ``CLAUDE.md``'s own "keep large/random
arrays out of text literals" guidance). Both the STRUCTURAL checks (only
MatMul/vanilla-Gemm with a constant 2-D float32 weight matches -- a 4-D
Conv weight and a non-vanilla Gemm, ``transA=1`` or ``alpha != 1``, both
decline outright, left completely untouched; ``W'`` has the same
shape/dtype as ``W``; a Gemm's bias passes through unchanged) and the
NUMERIC crux check run the real compiled pass through its dedicated
Python entry point, ``onnxsim.apply_fp6_llm_quantization_cpp``
(confirmed by reading ``onnxsim/quantize_entry.cpp``'s ``ApplyFp6Llm``
and ``onnxsim/onnx_simplifier.py``'s wrapper of the same name --
mirroring ``quantize_weight_only_mxfp4_cpp``'s own dedicated entry
point), NOT ``simplify_isolated_extra`` -- SURPRISE found empirically
while writing this file, not assumed: unlike every ``Cast``/``Gather``/
``Reshape``/``Mul``-chain-based pass in this suite (MXFP4, INT4, ...),
this pass's rewrite output -- a plain FLOAT32 initializer -- is ITSELF
again a constant 2-D float32 weight, exactly what
``patternMatchPredicate`` matches. Driving it through
``onnxsim.simplify``'s own outer Python fixed-point loop (which keeps
re-applying every requested "other" pass until the whole graph's
fingerprint stops changing) therefore re-quantizes an already-quantized
weight over and over; float64->float32->float64 rounding keeps the
fingerprint from ever exactly repeating, so the loop runs to its
50-iteration cap (with a printed timeout warning) instead of converging,
and -- combined with constant folding -- was observed to actually delete
the original ``W`` initializer this file's structural tests need to
check for. The dedicated entry point avoids this entirely (like
``ApplyFp6Llm``'s own single ``OptimizeFixed`` call, and matching
``tests/test_fp6_llm_cpp.py``'s own established way of exercising this
pass), and is used throughout instead. Unlike MXFP4, there is no
``Cast``/``Gather``/``Reshape``/``Mul`` chain to walk at all -- the
matched node's weight input is simply rewired to a new plain FLOAT32
initializer, inspected directly. For the numeric check, a real random
weight matrix is run through the pass and compared against an
INDEPENDENT from-scratch numpy re-implementation of
``QuantizeDequantizeFp6Block`` built in this file
(``_quantize_dequantize_fp6_reference``, using this file's own
``_build_fp6_e3m2_codebook`` and a numpy port of
``NearestFp6Value``'s own ``lower_bound``-style tie-breaking -- not
imported from ``onnxsim.fp6_llm`` or calling back into the C++ pass) --
this is the genuine cross-check of whether this file's Z3 modeling
actually matches the real C++ implementation, run on both a clean
(numel divisible by 64) and a ragged (not divisible by 64) shape.
Every dequantized weight value is also confirmed to equal
``scale * (some codebook entry)`` exactly (up to float32 rounding), and
the observed error is confirmed to stay within the proved ``2.0 *
scale`` per-element bound. A final test cross-checks this file's own
from-scratch codebook against ``ml_dtypes.float6_e3m2fn`` directly (a
dense sweep's own unique output values, not merely a round-trip of the
63 values already in hand) -- the same independent verification
``passes/fp6_llm.h``'s own top comment and ``tests/test_fp6_llm_cpp.py``
already perform, reproduced here fresh rather than trusted secondhand.
"""

import bisect

import ml_dtypes
import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
from _formal_verify_common import prove, z3
from onnx import parser

import onnxsim


def _abs(v):
    return z3.If(v >= 0, v, -v)


# --- 0. The codebook, built fresh from the same enumeration Fp6Codebook() --
# uses (sign, exponent-code, mantissa-code triples via ordinary float
# arithmetic) -- NOT imported from onnxsim.fp6_llm or ml_dtypes, so the
# cross-checks below are genuine, not round-trips against the same code.


def _build_fp6_e3m2_codebook():
    exp_bits, mant_bits, bias = 3, 2, 3
    exp_codes = 1 << exp_bits
    mant_codes = 1 << mant_bits
    values = []
    for sign_code in (0, 1):
        for e in range(exp_codes):
            for m in range(mant_codes):
                if e == 0:
                    v = (m / mant_codes) * 2.0 ** (1 - bias)
                else:
                    v = (1.0 + m / mant_codes) * 2.0 ** (e - bias)
                values.append(-v if sign_code else v)
    return sorted(set(values))


_FP6_E3M2_CODEBOOK = _build_fp6_e3m2_codebook()
_FP6_MAX_MAGNITUDE = _FP6_E3M2_CODEBOOK[-1]
_FP6_BLOCK_SIZE = 64  # kBlockSize in passes/fp6_llm.h


def test_fp6_codebook_has_63_distinct_entries_and_max_28():
    # 64 raw (sign, exponent, mantissa) codes, +0/-0 collapsing to one --
    # exactly the same "one-slot-collapse" shape MXFP4's 16-slot codebook
    # has (15 distinct), just a different size. Cross-checked against
    # passes/fp6_llm.h's own cited ml_dtypes.float6_e3m2fn max (28.0), not
    # merely re-asserted.
    assert len(_FP6_E3M2_CODEBOOK) == 63
    assert _FP6_MAX_MAGNITUDE == 28.0
    assert _FP6_E3M2_CODEBOOK[0] == -28.0
    # Symmetric around zero (every positive code has a sign-flipped twin).
    assert _FP6_E3M2_CODEBOOK == sorted(-v for v in _FP6_E3M2_CODEBOOK)


def test_fp6_codebook_gaps_are_not_constant_and_widest_gap_is_four():
    # The genuinely new claim vs. any uniform-step scheme: consecutive
    # gaps geometrically widen with the exponent rather than staying
    # constant. Derived directly from the sorted codebook's own
    # differences, not hand-typed.
    gaps = sorted(
        {
            round(_FP6_E3M2_CODEBOOK[i + 1] - _FP6_E3M2_CODEBOOK[i], 10)
            for i in range(len(_FP6_E3M2_CODEBOOK) - 1)
        }
    )
    assert gaps == [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    assert max(gaps) == 4.0
    assert min(gaps) == 0.0625


def test_fp6_python_codebook_matches_ml_dtypes_float6_e3m2fn_independently():
    # Cross-checks this file's own from-scratch enumeration against
    # ml_dtypes' reference type directly -- not merely a round-trip of the
    # 63 values already in hand: a dense sweep's own set of UNIQUE cast
    # outputs must equal this file's codebook exactly, confirming both
    # completeness (every code this file predicts is really produced) and
    # soundness (ml_dtypes produces no code this file's enumeration
    # missed). Mirrors passes/fp6_llm.h's own top-comment claim and
    # tests/test_fp6_llm_cpp.py's cross-check, reproduced fresh here.
    sweep = np.linspace(-40.0, 40.0, 200_001).astype(np.float32)
    cast = sweep.astype(ml_dtypes.float6_e3m2fn).astype(np.float64)
    ml_dtypes_codebook = sorted(set(np.unique(cast).tolist()))
    assert ml_dtypes_codebook == _FP6_E3M2_CODEBOOK
    finfo = ml_dtypes.finfo(ml_dtypes.float6_e3m2fn)
    assert float(finfo.max) == _FP6_MAX_MAGNITUDE


# --- 1. The codebook's own worst-case half-gap ------------------------------


def test_fp6_codebook_worst_case_half_gap_of_two_is_a_theorem():
    # "Nearest codebook value" rounding error is bounded by half the
    # LARGEST gap (4.0), not half the smallest (0.0625) -- 2.0. Proved
    # over the full, concrete 63-entry codebook (built above, not
    # abstracted inside Z3), for every real x WITHIN the codebook's own
    # range [-28, 28].
    x = z3.Real("x")
    in_range = z3.And(x >= -_FP6_MAX_MAGNITUDE, x <= _FP6_MAX_MAGNITUDE)
    prove(
        z3.Implies(
            in_range,
            z3.Or(*[_abs(x - z3.RealVal(c)) <= 2.0 for c in _FP6_E3M2_CODEBOOK]),
        ),
        msg="2.0 is not a valid worst-case half-gap bound for the FP6 E3M2 codebook",
    )


def test_fp6_codebook_worst_case_half_gap_lower_bound_is_tight():
    # 2.0 is not a loose overestimate: x = 18.0 sits exactly halfway
    # between codebook entries 16.0 and 20.0 -- one of the codebook's own
    # three widest adjacent gaps (4.0) -- each at distance 2.0; the
    # next-nearest entries (14.0, 24.0) are each twice as far (4.0). All
    # values are exact dyadic binary64 floats, so this arithmetic is
    # exact.
    x = 18.0
    distances = [abs(x - c) for c in _FP6_E3M2_CODEBOOK]
    assert min(distances) == 2.0, distances
    assert all(d >= 2.0 for d in distances)

    # Same conclusion as an actual Z3 counterexample search: 1.99 is NOT
    # a valid universal half-gap bound over the codebook's own domain,
    # even though 2.0 is (proved above).
    xx = z3.Real("x")
    solver = z3.Solver()
    solver.add(xx >= -_FP6_MAX_MAGNITUDE, xx <= _FP6_MAX_MAGNITUDE)
    solver.add(z3.And(*[_abs(xx - z3.RealVal(c)) > 1.99 for c in _FP6_E3M2_CODEBOOK]))
    assert solver.check() == z3.sat, (
        "every codebook entry is within 1.99 of every in-range real x -- the "
        "worst-case half-gap would then be < 2.0, contradicting the exact "
        "x = 18.0 computation above"
    )


def test_fp6_codebook_naive_half_smallest_gap_bound_is_unsound():
    # This file's own version of the MXFP4 docstring's "surprise caught
    # by Z3" moment: 0.03125 (half the SMALLEST gap, 0.0625, the number a
    # reader skimming only the near-zero subnormal region might reach
    # for) is NOT a valid universal half-gap bound at all, even restricted
    # to the codebook's own domain -- Z3 must find a genuine
    # counterexample (e.g. anything near x = 18.0, four codebook-gap
    # regions away from where the smallest gap lives).
    naive_half_gap = 0.03125
    x = z3.Real("x")
    solver = z3.Solver()
    solver.add(x >= -_FP6_MAX_MAGNITUDE, x <= _FP6_MAX_MAGNITUDE)
    solver.add(
        z3.And(*[_abs(x - z3.RealVal(c)) > naive_half_gap for c in _FP6_E3M2_CODEBOOK])
    )
    assert solver.check() == z3.sat, (
        "half the codebook's smallest gap (0.03125) holds as a universal "
        "in-range bound -- contradicts the widening-gap structure near the "
        "codebook's top exponent region"
    )


def test_fp6_codebook_unrestricted_domain_breaks_the_bound():
    # SURPRISE caught by Z3, not assumed going in (this file's own version
    # of the MXFP4 docstring's "x = -8" moment): an UNRESTRICTED "for
    # every real x" version of the half-gap claim is FALSE -- x = 40 sits
    # entirely outside the codebook's representable range [-28, 28], and
    # every codebook entry is more than 2.0 away from it. The domain
    # restriction is not a convenience weakening: lemma (2) below is
    # exactly what guarantees normalized := W / scale never leaves [-28,
    # 28] before NearestFp6Value is ever invoked on it.
    x = z3.Real("x")
    solver = z3.Solver()
    solver.add(z3.And(*[_abs(x - z3.RealVal(c)) > 2.0 for c in _FP6_E3M2_CODEBOOK]))
    assert solver.check() == z3.sat, (
        "every codebook entry is within 2.0 of every real x with no domain "
        "restriction at all -- the codebook lemma would then need no domain "
        "restriction, which is not how a finite codebook works"
    )
    model = solver.model()
    x_val = model[x]
    # Concrete sanity check with the actual witness Z3 found: sanity that
    # it is indeed outside the codebook's representable range.
    x_float = float(x_val.as_fraction())
    assert abs(x_float) > _FP6_MAX_MAGNITUDE


# --- 2. The linear block scale (max(|block|) / 28.0, not a power of two) ---


def test_fp6_linear_scale_keeps_block_within_codebook_range():
    # scale = max(max_abs, 1e-12) / 28.0 for max_abs = max(|block|) >= 0.
    # The max(..., 1e-12) floor (an all-zero block would otherwise divide
    # by zero) is modeled directly with z3.If, not assumed away. For any
    # element x with |x| <= max_abs, |x| / scale <= 28.0 follows from
    # scale >= max_abs / 28.0 (which holds in BOTH floor branches: when
    # max_abs >= 1e-12, scale == max_abs / 28.0 exactly; when max_abs <
    # 1e-12, scale == 1e-12 / 28.0 > max_abs / 28.0) and |x| <= max_abs.
    eps = z3.RealVal("1/1000000000000")  # 1e-12, exact in Z3's rationals
    max_mag = z3.RealVal(28.0)
    x, max_abs, scale = z3.Reals("x max_abs scale")
    hypotheses = z3.And(
        max_abs >= 0,
        _abs(x) <= max_abs,
        scale == z3.If(max_abs >= eps, max_abs, eps) / max_mag,
    )
    prove(z3.Implies(hypotheses, z3.And(scale > 0, _abs(x) / scale <= max_mag)))


def test_fp6_linear_scale_lands_the_blocks_own_max_exactly_at_codebook_max():
    # Explicit corollary, stated honestly rather than silently folded into
    # the <= lemma above: because this scale is an exact ratio (no
    # ceil/floor rounding of the scale itself, unlike MXFP4's power-of-two
    # scale), a block whose max_abs is already >= the 1e-12 floor has its
    # own largest-magnitude element land EXACTLY on the codebook's max
    # magnitude -- an equality, not merely a <=. This is a strictly
    # tighter guarantee than MXFP4's own ceil()-based scale, which can
    # leave a block underutilizing the codebook's range by up to 2x.
    max_mag = z3.RealVal(28.0)
    max_abs, scale = z3.Reals("max_abs scale")
    hypotheses = z3.And(
        max_abs >= z3.RealVal("1/1000000000000"), scale == max_abs / max_mag
    )
    prove(z3.Implies(hypotheses, max_abs / scale == max_mag))


# --- Composition: codebook half-gap x block scale = per-element bound ------


def test_fp6_per_element_error_bound_is_codebook_half_gap_times_scale():
    # The corollary tying the two lemmas above into the single-element
    # claim the MAC bound below is built on -- same shape as the MXFP4
    # file's own corollary, with this file's own 2.0 constant.
    # normalized := W / scale lands within 2.0 of SOME codebook entry c
    # (the domain-restricted codebook lemma's conclusion, valid precisely
    # because the linear-scale lemma above guarantees normalized never
    # leaves [-28, 28] in the first place -- taken here as a hypothesis
    # rather than re-derived symbolically in the same query, matching this
    # suite's usual style of chaining separately-proved lemmas), and
    # Wdq := c * scale, so
    #   |W - Wdq| = |normalized * scale - c * scale| = |normalized - c| * scale <= 2.0 * scale.
    w, scale, normalized, c = z3.Reals("w scale normalized c")
    hypotheses = z3.And(
        scale > 0,
        normalized == w / scale,
        _abs(normalized - c) <= 2.0,
    )
    wdq = c * scale
    error = w - wdq
    prove(z3.Implies(hypotheses, z3.And(error <= 2.0 * scale, -error <= 2.0 * scale)))


# --- The per-block MAC bound (eps_w := 2.0 * scale[block], not scale / 2) ---

_K = 2  # matches weight_only_quantize_mxfp4_matmul's own minimal-case
# convention: two SEPARATE one-element blocks is already the smallest
# layout exercising independent per-block scales. FP6's own real block
# size is 64 (_FP6_BLOCK_SIZE); see the differential tests below for that.
_BLOCK_SIZE = 1  # each tap is its own block: scale0 for tap 0, a
# genuinely different scale1 for tap 1.
_NUM_BLOCKS = _K // _BLOCK_SIZE


def _block_of(k):
    return k // _BLOCK_SIZE


def _bound_formulas():
    """Z3 vocabulary for the single-operand, PER-BLOCK bounded-error claim,
    reused-in-structure (down-sized to ``_K = 2``) from
    ``weight_only_quantize_mxfp4_matmul``'s own per-block generalization:
    ``X`` has no error term at all (never quantized), only ``W`` does, via
    the free per-tap error variable ``ew``, with one scale variable PER
    BLOCK. The constant multiplying each block's scale is THIS file's own
    ``2.0`` (the codebook's worst-case half-gap, proved above), not
    MXFP4's ``1.0``.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # X[i, k], true float activation
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[k, n], true float weight
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[k, n] - Wdq[k, n]
    scale = [z3.Real(f"scale{b}") for b in range(_NUM_BLOCKS)]  # per-block scale

    dequant_w = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        *[scale[b] > 0 for b in range(_NUM_BLOCKS)],
        # Each tap's error is bounded by ITS OWN block's scale, times the
        # codebook's own worst-case half-gap (2.0, NOT 0.5 or 1.0) --
        # scale0 for tap 0, a genuinely different scale1 for tap 1.
        *[_abs(ew[k]) <= 2.0 * scale[_block_of(k)] for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))

    bound = sum((2.0 * scale[_block_of(k)]) * _abs(X[k]) for k in range(_K))

    return float_matmul, dequant_matmul, rounding_bounds, bound


def test_fp6_llm_error_is_bounded():
    # The genuine bounded-error claim: given W's own per-BLOCK rounding
    # bound (|W[k, n] - Wdq[k, n]| <= 2.0 * scale[block_of(k), n]) and X
    # completely unchanged, the true float dot product and the one
    # computed against the dequantized weight cannot differ by more than
    # sum_k (2.0 * scale[block_of(k), n]) * |X[i, k]|.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_fp6_llm_bias_variant_error_is_bounded():
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


def test_fp6_llm_negative_control_requires_rounding_bound():
    # Sanity check that the bound proved above is genuine, not vacuous:
    # with no error budget assumed on ew at all (only scale0, scale1 > 0),
    # the same bound is not a theorem -- Z3 must find a real
    # counterexample.
    float_matmul, dequant_matmul, _rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul

    solver = z3.Solver()
    solver.add(z3.Not(z3.And(error <= bound, -error <= bound)))
    assert solver.check() == z3.sat, (
        "the bound holds even without any rounding-error budget on ew -- "
        "negative control is vacuous"
    )


def test_fp6_llm_uniform_bound_is_unsound_across_blocks():
    # Reproduces weight_only_quantize_mxfp4_matmul's own "naive
    # single-shared-scale bound is unsound across blocks" negative
    # control, for THIS file's own 2.0 * scale constant: if one instead
    # (incorrectly) bounded every tap's error using block 0's scale alone
    # (2.0 * scale0), that claim is NOT a theorem once block 1's real
    # scale scale1 can exceed scale0.
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    scale = [z3.Real(f"scale{b}") for b in range(_NUM_BLOCKS)]

    dequant_w = [W[k] - ew[k] for k in range(_K)]
    rounding_bounds = z3.And(
        *[scale[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= 2.0 * scale[_block_of(k)] for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))
    error = float_matmul - dequant_matmul

    # The naive/wrong claim: bound every tap's contribution using ONLY
    # block 0's scale scale0.
    uniform_bound = (2.0 * scale[0]) * sum(_abs(X[k]) for k in range(_K))

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


def _nearest_fp6_values(values):
    """Independent numpy port of ``NearestFp6Value``'s own
    ``std::lower_bound``-based nearest-neighbor search and tie-breaking
    (round-to-even is NOT what this does -- ties break toward the lower
    entry, matching the C++ ``(value - lo) <= (hi - value) ? lo : hi``).
    """
    codebook = np.asarray(_FP6_E3M2_CODEBOOK, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    flat_values = values.reshape(-1)
    flat_out = out.reshape(-1)
    for i, v in enumerate(flat_values):
        idx = bisect.bisect_left(_FP6_E3M2_CODEBOOK, v)
        if idx == 0:
            flat_out[i] = codebook[0]
        elif idx == len(codebook):
            flat_out[i] = codebook[-1]
        else:
            lo, hi = codebook[idx - 1], codebook[idx]
            flat_out[i] = lo if (v - lo) <= (hi - v) else hi
    return out


def _quantize_dequantize_fp6_reference(flat_weight, block_size=_FP6_BLOCK_SIZE):
    """Independent from-scratch numpy re-implementation of
    ``QuantizeDequantizeFp6Block`` (``passes/fp6_llm.h``): one linear
    scale (``max(|block|) / 28.0``, floored at ``1e-12``) per
    ``block_size``-element block of the FLATTENED, row-major input, each
    element snapped to its nearest FP6 E3M2 codebook value times that
    scale. A ragged final block (when the input's length isn't itself a
    multiple of ``block_size``) uses only its own real elements. Written
    fresh here (own control flow) rather than calling into
    ``onnxsim.fp6_llm``, so this is a genuine independent cross-check of
    the compiled C++ pass's own actual output.
    """
    flat = np.asarray(flat_weight, dtype=np.float64).reshape(-1)
    out = np.empty_like(flat)
    scales = []
    for start in range(0, flat.size, block_size):
        block = flat[start : start + block_size]
        max_abs = float(np.max(np.abs(block))) if block.size else 0.0
        scale = max(max_abs, 1e-12) / _FP6_MAX_MAGNITUDE
        out[start : start + block.size] = _nearest_fp6_values(block / scale) * scale
        scales.append(scale)
    return out, np.asarray(scales)


def _current_weight_and_name(model, weight_input_index=1):
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return numpy_helper.to_array(w_init), w_name


def test_fp6_llm_pass_fires_on_matmul_replacing_weight_in_place():
    # Plain MatMul: the matched node's weight input is rewired to a new,
    # same-shape/dtype FLOAT32 initializer -- no Cast/Gather/Reshape/Mul
    # chain at all, unlike MXFP4.
    rng = np.random.default_rng(0)
    rows, K, N = 4, 16, 8  # numel = 128 = 2 * _FP6_BLOCK_SIZE
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

    # Uses the dedicated Python entry point (onnxsim.quantize_entry.cpp's
    # ApplyFp6Llm), not simplify_isolated_extra: this pass writes a plain
    # FLOAT32 initializer directly (no wrapping Cast/Gather/Reshape/Mul the
    # way MXFP4 does), which is itself again a constant 2-D float32 weight
    # -- exactly what patternMatchPredicate matches -- so driving it through
    # onnxsim.simplify's own outer Python fixed-point loop (which keeps
    # re-applying every "other" pass, including this one, until the whole
    # graph's fingerprint stops changing) re-quantizes an already-quantized
    # weight over and over; float64->float32->float64 rounding keeps the
    # fingerprint from ever exactly repeating, so it runs to the loop's
    # 50-iteration cap instead of converging. The dedicated entry point
    # (like ``ApplyFp6Llm``'s own ``OptimizeFixed`` call) applies the named
    # pass to ITS OWN fixed point once, matching test_fp6_llm_cpp.py's own
    # established way of exercising this pass.
    quantized = onnxsim.apply_fp6_llm_quantization_cpp(model)
    onnx.checker.check_model(quantized)
    matmul_node = next(n for n in quantized.graph.node if n.op_type == "MatMul")
    x_input, w_input = matmul_node.input
    assert x_input == "X"  # activation passed through completely unchanged
    assert w_input != "W"  # rewired to a fresh initializer

    new_w, _name = _current_weight_and_name(quantized)
    assert new_w.shape == weight.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, weight)
    # The original initializer is left dangling in the graph, unused --
    # matching every other *_llm/*_quant pass's established convention.
    assert any(t.name == "W" for t in quantized.graph.initializer)


def test_fp6_llm_gemm_vanilla_matches_bias_untouched():
    rng = np.random.default_rng(1)
    rows, K, N = 3, 16, 4
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    bias = rng.standard_normal(N).astype(np.float32)
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )

    quantized = onnxsim.apply_fp6_llm_quantization_cpp(model)
    onnx.checker.check_model(quantized)
    gemm_node = next(n for n in quantized.graph.node if n.op_type == "Gemm")
    x_input, w_input, b_input = gemm_node.input
    assert x_input == "X"
    assert b_input == "B"  # bias untouched
    assert w_input != "W"


def test_fp6_llm_declines_for_non_2d_weight():
    # Conv's weight is 4-D -- patternMatchPredicate only matches a
    # constant 2-D float32 weight, so Conv must survive completely
    # untouched (only MatMul/vanilla-Gemm are in scope, per
    # quantize_matmul_common.h's own MatchMatMulLike).
    rng = np.random.default_rng(2)
    w = rng.standard_normal((2, 2, 3, 3)).astype(np.float32)
    model = _model(
        """
        g (float[1,2,8,8] X) => (float[1,2,6,6] Y)
        {
          Y = Conv(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    result = onnxsim.apply_fp6_llm_quantization_cpp(model)
    # Byte-identical: patternMatchPredicate declines outright, no rewrite at
    # all (matches test_fp6_llm_cpp.py's own test_cpp_skips_non_2d_weight).
    assert result.SerializeToString() == model.SerializeToString()


def test_fp6_llm_declines_for_non_vanilla_gemm():
    # transA=1 (and, separately, alpha != 1) fall outside MatchMatMulLike's
    # own "vanilla Gemm" scope -- both must survive completely untouched.
    rng = np.random.default_rng(3)
    K, N = 16, 4
    weight = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[{K},4] X) => (float[4,{N}] Y)
        {{
          Y = Gemm<transA = 1>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    result = onnxsim.apply_fp6_llm_quantization_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_fp6_llm_quantize_dequantize_matches_independent_python_reference():
    # The crux differential check: run the REAL compiled pass via its
    # dedicated Python entry point, and compare its output element-for-
    # element against this file's own from-scratch numpy reference
    # (_quantize_dequantize_fp6_reference), built from this file's own
    # codebook enumeration -- NOT by calling back into the C++ pass. Two
    # shapes: one with numel an exact multiple of 64 (clean blocks) and
    # one that is not (a ragged final block).
    for K, N, seed in [(16, 8, 4), (13, 11, 5)]:  # 128 (clean); 143 (ragged)
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 1.3
        model = _model(
            f"""
            g (float[5,{K}] X) => (float[5,{N}] Y)
            {{
              Y = MatMul(X, W)
            }}
            """,
            [_f32(weight, "W")],
        )

        quantized = onnxsim.apply_fp6_llm_quantization_cpp(model)
        onnx.checker.check_model(quantized)
        new_w, _name = _current_weight_and_name(quantized)
        assert new_w.shape == (K, N)
        assert new_w.dtype == np.float32

        expected_flat, scales = _quantize_dequantize_fp6_reference(
            weight.reshape(-1).astype(np.float64)
        )
        expected = expected_flat.reshape(K, N).astype(np.float32)
        np.testing.assert_allclose(new_w, expected, rtol=1e-5, atol=1e-6)

        # Every dequantized value equals scale * (some codebook entry)
        # exactly, up to float32 rounding -- confirms the output really
        # is a codebook-times-scale value, not merely numerically close.
        flat_new = new_w.reshape(-1).astype(np.float64)
        block_of = np.arange(flat_new.size) // _FP6_BLOCK_SIZE
        per_elem_scale = scales[block_of]
        normalized = np.divide(
            flat_new,
            per_elem_scale,
            out=np.zeros_like(flat_new),
            where=per_elem_scale > 0,
        )
        codebook = np.asarray(_FP6_E3M2_CODEBOOK)
        nearest_dist = np.min(np.abs(normalized[:, None] - codebook[None, :]), axis=1)
        assert np.all(nearest_dist < 1e-3), nearest_dist.max()

        # The observed error stays within the proved 2.0 * scale
        # per-element bound (plus a small float32 rounding slack).
        w64 = weight.reshape(-1).astype(np.float64)
        error = np.abs(w64 - flat_new)
        bound = 2.0 * per_elem_scale
        assert np.all(error <= bound + 1e-4), (error - bound).max()


def test_fp6_llm_zero_maps_to_zero():
    rng = np.random.default_rng(6)
    K, N = _FP6_BLOCK_SIZE, 4
    weight = rng.standard_normal((K, N)).astype(np.float32)
    weight[0, 0] = 0.0
    model = _model(
        f"""
        g (float[3,{K}] X) => (float[3,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    quantized = onnxsim.apply_fp6_llm_quantization_cpp(model)
    new_w, _name = _current_weight_and_name(quantized)
    assert new_w[0, 0] == 0.0
