"""Formal check for WeightOnlyQuantizeMXFP4MatMul (opt-in; onnxsim's own
``onnxsim/passes/weight_only_quantize_mxfp4_matmul.h`` and its shared helper
``onnxsim/passes/quantize_mxfp4_common.h``): the same "only the constant
weight is quantized, the activation ``X`` is left completely untouched, no
calibration data" design as every other file in this weight-only family --
READ ``test_formal_verify_weight_only_quantize_int4_matmul.py`` FIRST, this
file only documents what differs -- but this is OCP Microscaling MXFP4, a
GENUINELY DIFFERENT quantization scheme from every INT4/INT8/INT16 affine
scheme already in this suite, not just a narrower bit width or a different
op chain. It rewrites ``Y = MatMul(X, W)`` (or a "vanilla" Gemm, bias left
untouched) -- ``W`` a constant 2-D FLOAT32 tensor whose reduction dimension
``K`` is evenly divisible by ``kMXBlockSize = 32``, ``X`` FLOAT32 -- into::

    Wq, Ws := block-wise MXFP4 quantization of W (computed ONCE, at
              pass-transform time): Wq holds one UINT8 codebook index
              (0..15) per element, Ws one power-of-two FLOAT32 scale per
              (block-of-K, output-channel) pair
    Codes    = Cast(Wq, INT64)
    Gathered = Gather(Codebook, Codes, axis=0)      # Codebook: 16 E2M1 values
    Blocked  = Reshape(Gathered, <blocked shape>)
    ScaleB   = Reshape(Ws, <matching blocked shape, block dim singleton>)
    Scaled   = Mul(Blocked, ScaleB)
    Wdq      = Reshape(Scaled, W's original shape)
    Y        = MatMul(X, Wdq)

ONNX has no native MX tensor type, so -- unlike every ``DequantizeLinear``-
based pass in this suite -- the dequantization above is built from ordinary
opset-11+ ``Cast``/``Gather``/``Reshape``/``Mul``, confirmed by reading
``runTransform`` line by line (``weight_only_quantize_mxfp4_matmul.h``).

Two genuinely new wrinkles this file's proof needs that NO prior pass in
this suite's INT4/INT8/INT16 family has (confirmed by reading
``MXFP4Codebook()``, ``NearestMXFP4Code``, and
``TryQuantizeWeightBlockwiseMXFP4InPlace`` in ``quantize_mxfp4_common.h``
line by line, rather than assumed to work like affine round-to-nearest):

1. **The codebook is FIXED and NON-UNIFORM.** ``MXFP4Codebook()`` returns
   exactly ``{-6, -4, -3, -2, -1.5, -1, -0.5, -0, 0, 0.5, 1, 1.5, 2, 3, 4, 6}``
   -- 16 E2M1 bit patterns, fixed by the OCP MX spec, not fit to any data.
   Sorted (``-0`` and ``0`` collapse to the same real value, so there are 15
   DISTINCT reals in 16 slots): ``-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1,
   1.5, 2, 3, 4, 6``. The 14 gaps between adjacent sorted values are NOT
   constant -- computed by hand below, confirmed independently rather than
   assumed:

       -6 to -4: 2      -3 to -2:   1      -1 to -0.5: 0.5    0.5 to 1: 0.5
       -4 to -3: 1      -2 to -1.5: 0.5    -0.5 to 0:  0.5    1 to 1.5: 0.5
                                                                1.5 to 2: 0.5
        2 to 3: 1        3 to 4: 1          4 to 6: 2

   The largest gap is 2.0, at BOTH ends (-6/-4 and 4/6) -- so "nearest
   codebook value" rounding error is NOT bounded by a naive "half the
   smallest gap" (which would give 0.25, wrong) or any constant read off
   the codebook's small values alone. The TIGHT worst-case bound is half
   the LARGEST gap: 1.0, achieved at x = 5.0 (exactly between 4 and 6, both
   at distance 1.0, with the next-nearest entry -- 3.0 -- twice as far).
   ``test_mxfp4_codebook_worst_case_half_gap_of_one_is_a_theorem`` proves
   ``forall x in [-6, 6], exists c in codebook, |x - c| <= 1.0`` over the
   FULL 16-value codebook spelled out as literal ``z3.RealVal`` constants
   (not abstracted); ``test_mxfp4_codebook_worst_case_half_gap_lower_bound_
   is_tight`` confirms 1.0 is not a loose overestimate -- at x = 5.0 the
   nearest codebook entry is EXACTLY distance 1.0 away, so no constant
   below 1.0 can be substituted.

   SURPRISE, caught by Z3 rather than assumed while writing this file: an
   UNRESTRICTED "for every real x" version of this claim is actually
   FALSE -- Z3's own first counterexample was x = -8, far outside the
   codebook's representable range, where nothing bounds the distance at
   all. The ``[-6, 6]`` domain restriction is not an arbitrary weakening,
   though: it is exactly the range wrinkle (2) below guarantees
   ``normalized := W / Ws`` always lands in before ``NearestMXFP4Code`` is
   ever called on it, so the two lemmas' domains match by construction.

2. **The block scale is constrained to a POWER OF TWO**, chosen via
   ``scale = 2^ceil(log2(max_abs_in_block / 6.0))`` (``kMXFP4MaxMagnitude =
   6.0``, the codebook's own max magnitude) -- not ``max_abs / (some fixed
   divisor)`` the way every affine scheme in this suite works.
   ``std::ceil`` (not ``std::floor``) is what guarantees the block's own
   largest-magnitude element always lands within the codebook's
   representable range (``max_abs_in_block / scale <= 6.0``, no external
   clamping ever needed) -- confirmed as its own Z3 lemma
   (``test_mxfp4_ceil_power_of_two_scale_keeps_block_within_codebook_
   range``), with a companion negative control
   (``test_mxfp4_floor_based_alternative_does_not_guarantee_range``)
   exhibiting a concrete ``m`` for which the floor-based alternative -- the
   header comment's own "the floor-based alternative would silently clip"
   -- produces ``m / scale > 6.0``.

Combining (1) and (2) gives the pass's overall per-element bound::

    |W[k, n] - Wdq[k, n]| <= 1.0 * Ws[block_of(k), n]

-- the codebook's own worst-case half-gap (1.0) TIMES the block's own
power-of-two scale, NOT ``Ws / 2`` the way every affine weight-only pass in
this suite states it. This composition is its own explicit corollary,
``test_mxfp4_per_element_error_bound_is_codebook_half_gap_times_scale``, not
silently reused from another file's ``_bound_formulas``.

This is still the single-operand (``eps_x := 0``, ``X`` never quantized)
collapse of ``quantized_mac_bound``'s general MAC bound
(``test_formal_verify_quantized_mac_bound.py``), with a genuinely PER-BLOCK
``eps_w := 1.0 * Ws[block_of(k), n]`` term -- the per-block-SUM structure
below is reused (down-sized to ``_K = 2``, ``_BLOCK_SIZE = 1``, matching
``weight_only_quantize_matmul_nbits``'s own minimal-case convention: two
separate one-element blocks is already the smallest case that exercises
independent per-block scales) from ``test_formal_verify_weight_only_
quantize_int4_matmul.py``'s own per-block generalization, including its own
"a naive single-shared-scale bound is unsound across blocks" negative
control (``test_weight_only_quantize_mxfp4_matmul_uniform_bound_is_unsound_
across_blocks``) -- but the constant multiplying ``Ws`` is this file's own
``1.0``, derived from the codebook lemma above, NOT the affine passes'
``0.5``. Every bound-proving query below uses the DIRECT-ERROR-VARIABLE
idiom (a free ``ew`` bounded directly by ``1.0 * Ws[block]``, not
reconstructed from codebook-index/scale multiplicands inside the same
query) -- see ``test_formal_verify_dynamic_quantize_matmul.py``'s own module
docstring for why a combined nonlinear reconstruction is a documented Z3
hang risk (a single query hung past 90 seconds there at ``_K`` as small as
2). A bias-variant test (Gemm bias, untouched, cancels the same way as
every prior pass in this family) and the standard vacuity negative control
round out the Z3 side.

Differential tests build a plain float ``MatMul``/``Gemm`` via
``onnx.parser`` (per ``CLAUDE.md``) with ``K`` a multiple of 32
(``kMXBlockSize``) -- e.g. ``K = 64`` -- run the real pass alone via
``simplify_isolated_extra`` (``extra_optimizers=["weight_only_quantize_
mxfp4_matmul"]``) and, for the numeric bound check, via the dedicated
Python entry point ``onnxsim.quantize_weight_only_mxfp4_cpp`` (confirmed by
reading ``onnxsim/quantize_entry.cpp``'s ``QuantizeWeightOnlyMXFP4`` and
``onnxsim/onnx_simplifier.py``'s wrapper of the same name -- mirroring
``quantize_weight_only_int4``'s own dedicated entry point; a separate file,
``test_mx_quantization_cpp.py``, already differentially checks this entry
point against the pure-Python ``onnxsim.quantize_weight_only_mxfp4``
reference on its own terms, so the differential tests here focus on what
this file's proof actually needs: the exact node CHAIN and its shapes, the
codebook's literal values, an INDEPENDENT from-scratch numpy
re-implementation of the quantization scheme, and the proved per-element
bound against real ONNX Runtime output). Confirmed below: the exact
``Cast -> Gather -> Reshape -> Mul(., Reshape) -> Reshape`` chain fires,
walked backward from the graph output (mirroring every other opt-in-pass
differential test in this suite, since ``simplify_isolated_extra`` skips the
default dead-code-elimination pass); the ``Gather``'s codebook initializer
matches ``MXFP4Codebook()``'s 16 values exactly, in order; the two Reshape
shape conventions (``reduction_axis == 0`` for plain MatMul vs.
``reduction_axis == 1`` for Gemm's ``transB=1``) match the header comment
exactly; ``Wq``'s codes and ``Ws``'s per-block scales match an INDEPENDENT
Python/numpy re-implementation of ``TryQuantizeWeightBlockwiseMXFP4InPlace``
-- including the nearest-codebook-index search and the
``ceil(log2(...))`` scale formula, both implemented fresh here rather than
imported from ``onnxsim.mx_quantization``, so this is a genuine
cross-check, not a round-trip against the same code; every ``Ws`` value is
genuinely a power of two; ``K`` not divisible by 32 declines outright; and
the real ONNX Runtime output -- graph optimization EXPLICITLY DISABLED
(``so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_
ALL``, per this suite's now-established default; see
``tests/test_ort_matmul_nbits_workaround.py``'s docstring for the ORT
graph-optimization-fusion bug precedent this guards against) -- stays
within the proved per-block bound (``1.0 * Ws[block]`` per tap, not
``Ws / 2``) against the true float MatMul.

Every bound-checking differential test passes ``check_n=0`` to
``simplify_isolated_extra`` for the same reason every pass in this
quantization family does: this is a genuinely lossy rewrite, coarser than
onnxsim's own default random-input equivalence check's tolerance.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
import onnxsim.onnxsim_cpp2py_export as C
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim


def _simplify_isolated_extra_no_fold(model, *pass_names, check_n=0):
    """Like ``simplify_isolated_extra`` (``_formal_verify_common.py``), but
    with constant folding disabled (``skip_constant_folding=True``).

    This pass's whole rewrite -- Cast/Gather/Reshape/Mul/Reshape -- is built
    entirely from ordinary, universally-supported ops over ONLY constant
    inputs (the codebook, Wq, Ws and the shape initializers), unlike, say,
    the INT4 pass's ``DequantizeLinear`` (whose opset-21 int4 tensor type
    the installed onnxruntime constant-folding backend does not evaluate,
    so it survives ``simplify_isolated_extra`` unfolded). Confirmed
    empirically: with constant folding left on, onnxsim's own default
    folding step evaluates the whole chain right back down to a single
    plain float32 initializer, which would make it impossible to observe
    the chain shape this file's structural tests need to check at all --
    this is exactly what the differential tests here need to inspect, not a
    correctness problem with folding itself.
    """
    names = set(pass_names)
    all_other = set(C._list_other_optimizers())
    unknown = names - all_other
    assert not unknown, f"not an opt-in onnxsim optimizer pass: {sorted(unknown)}"
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        extra_optimizers=sorted(names),
        skipped_optimizers=sorted(C._list_optimizers()),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    return sim_model


# E2M1's own 16 bit patterns (quantize_mxfp4_common.h's MXFP4Codebook()),
# transcribed independently here (not imported from onnxsim.mx_quantization
# or the C++ header) so the differential tests below are a genuine
# cross-check of the pass's own literal initializer values, not a round-trip
# against the same constant.
_MXFP4_CODEBOOK = [
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]
_MXFP4_MAX_MAGNITUDE = 6.0
_MX_BLOCK_SIZE = 32


def _abs(v):
    return z3.If(v >= 0, v, -v)


# --- 1. The codebook's own worst-case half-gap ------------------------------


def test_mxfp4_codebook_worst_case_half_gap_of_one_is_a_theorem():
    # The genuinely new claim vs. every affine (uniform-step) scheme in this
    # suite: "nearest codebook value" rounding error is bounded not by half
    # the SMALLEST gap (0.25, near zero) but by half the LARGEST gap (2.0,
    # at the extremes) -- 1.0. Proved over the full, literal 16-value
    # codebook, for every real x WITHIN THE CODEBOOK'S OWN RANGE [-6, 6].
    #
    # SURPRISE caught by Z3, not assumed going in: an UNRESTRICTED "for
    # every real x" claim is actually FALSE -- Z3's first counterexample was
    # x = -8, entirely outside the codebook's representable range, where no
    # entry is anywhere near it (the codebook simply doesn't extend that
    # far, so nothing bounds the distance once x leaves [-6, 6]). The domain
    # restriction is not a weakening of convenience, though: this pass never
    # applies NearestMXFP4Code to an arbitrary real, only to normalized :=
    # W / Ws, and lemma (2) below (the ceil()-based scale selection) is
    # EXACTLY what guarantees normalized always lands in [-6, 6] before this
    # codebook lemma is ever invoked -- so the two lemmas' domains line up
    # by construction, not by coincidence.
    #
    # Free variable x is implicitly universally quantified by `prove`
    # (mirrors z3's own prove() semantics: valid iff the negation is
    # unsatisfiable) -- the same free-variable-plus-disjunction shape this
    # suite's `prove` helper already uses everywhere else, avoiding an
    # explicit ForAll/Exists quantifier alternation.
    x = z3.Real("x")
    in_range = z3.And(x >= -_MXFP4_MAX_MAGNITUDE, x <= _MXFP4_MAX_MAGNITUDE)
    prove(
        z3.Implies(
            in_range, z3.Or(*[_abs(x - z3.RealVal(c)) <= 1.0 for c in _MXFP4_CODEBOOK])
        ),
        msg="1.0 is not a valid worst-case half-gap bound for the MXFP4 codebook",
    )


def test_mxfp4_codebook_worst_case_half_gap_lower_bound_is_tight():
    # Negative-control-style companion to the theorem above: 1.0 is not a
    # loose overestimate. x = 5.0 sits exactly halfway between codebook
    # entries 4.0 and 6.0 -- the codebook's own largest adjacent gap (2.0,
    # computed by hand in the module docstring) -- each at distance 1.0; the
    # next-nearest entry (3.0) is twice as far (distance 2.0). All values
    # involved are exact dyadic (power-of-two-denominator) binary64 floats,
    # so this arithmetic is exact, not merely "close enough".
    x = 5.0
    distances = [abs(x - c) for c in _MXFP4_CODEBOOK]
    assert min(distances) == 1.0, distances
    assert all(d >= 1.0 for d in distances)

    # Same conclusion as an actual Z3 counterexample search: 0.99 is NOT a
    # valid universal half-gap bound (Z3 must find some x whose every
    # codebook distance exceeds 0.99) even though 1.0 is (proved above) --
    # confirming the constant genuinely cannot be tightened below 1.0.
    xx = z3.Real("x")
    solver = z3.Solver()
    solver.add(z3.And(*[_abs(xx - z3.RealVal(c)) > 0.99 for c in _MXFP4_CODEBOOK]))
    assert solver.check() == z3.sat, (
        "every codebook entry is within 0.99 of every real x -- the "
        "worst-case half-gap would then be < 1.0, contradicting the exact "
        "x = 5.0 computation above"
    )


# --- 2. The power-of-two block scale (ceil, not floor) ----------------------


def test_mxfp4_ceil_power_of_two_scale_keeps_block_within_codebook_range():
    # scale = 2 ** ceil(log2(m / 6.0)) for m = max_abs_in_block > 0. Modeled
    # by introducing `scale` as a free real standing for 2**e directly (Z3
    # never needs to reason about log2/exponentiation symbolically) and
    # encoding ceil's own two defining inequalities in terms of it:
    #   e >= log2(m / 6)       <=>  2**e >= m / 6       <=>  scale >= m / 6
    #   e <  log2(m / 6) + 1   <=>  2**(e-1) < m / 6     <=>  scale / 2 < m / 6
    # Only the first inequality is actually needed for this direction of the
    # claim (m / scale <= 6.0 follows from scale >= m / 6 by simple algebra);
    # the second is included anyway to state the full ceil() definition, per
    # the task's own framing, and to set up the floor-based negative control
    # below as its direct structural mirror.
    m, scale = z3.Reals("m scale")
    hypotheses = z3.And(m > 0, scale > 0, scale >= m / 6.0, scale / 2.0 < m / 6.0)
    prove(z3.Implies(hypotheses, m / scale <= 6.0))


def test_mxfp4_floor_based_alternative_does_not_guarantee_range():
    # Companion negative control: quantize_mxfp4_common.h's own header
    # comment ("ceil(), not floor(log2)-2, so max_abs / scale is always <=
    # 6.0") and mx_quantization.py's docstring both call out that a
    # floor-based alternative would silently clip. Modeled the same way as
    # the ceil lemma above but for scale_floor = 2 ** floor(log2(m / 6.0)):
    #   e <= log2(m / 6) < e + 1  <=>  scale_floor <= m / 6 < 2 * scale_floor
    # Under these hypotheses m / scale_floor can range anywhere in [6, 12) --
    # Z3 finds a genuine witness exceeding 6.0.
    m, scale_floor = z3.Reals("m scale_floor")
    hypotheses = z3.And(
        m > 0, scale_floor > 0, scale_floor <= m / 6.0, m / 6.0 < 2 * scale_floor
    )
    solver = z3.Solver()
    solver.add(hypotheses)
    solver.add(m / scale_floor > 6.0)
    assert solver.check() == z3.sat, (
        "the floor-based alternative always keeps m / scale <= 6.0 too -- "
        "contradicts quantize_mxfp4_common.h's own comment that ceil() "
        "(not floor()) is needed to avoid silently clipping"
    )

    # Concrete sanity check with clean literals, independent of Z3's own
    # model: m / 6 in [1, 2) (m in [6, 12)) with scale_floor = 2**0 = 1
    # satisfies the floor hypotheses while landing as high as just under
    # m / scale_floor = 12, comfortably past the codebook's max magnitude of
    # 6.0 -- exactly the "silent clipping" the header comment warns about.
    concrete_m, concrete_scale = 11.0, 1.0
    assert concrete_scale <= concrete_m / 6.0 < 2 * concrete_scale
    assert concrete_m / concrete_scale > 6.0


# --- Composition: codebook half-gap x block scale = per-element bound ------


def test_mxfp4_per_element_error_bound_is_codebook_half_gap_times_scale():
    # The corollary tying the two lemmas above into the single-element claim
    # the MAC bound below is built on. Unlike every affine pass in this
    # suite (bound = Ws / 2), this pass's per-element bound is Ws * 1.0:
    # normalized := W / Ws lands within 1.0 of SOME codebook entry c (the
    # DOMAIN-RESTRICTED codebook lemma's own conclusion -- valid precisely
    # because the ceil()-based scale lemma above guarantees normalized never
    # leaves the codebook's own [-6, 6] range in the first place -- taken
    # here as a hypothesis rather than re-derived symbolically in the same
    # query, matching this suite's usual style of chaining separately-proved
    # lemmas rather than handing Z3 one large combined nonlinear formula),
    # and Wdq := c * Ws, so
    #   |W - Wdq| = |normalized * Ws - c * Ws| = |normalized - c| * Ws <= Ws.
    w, ws, normalized, c = z3.Reals("w ws normalized c")
    hypotheses = z3.And(
        ws > 0,
        normalized == w / ws,
        _abs(normalized - c) <= 1.0,
    )
    wdq = c * ws
    error = w - wdq
    prove(z3.Implies(hypotheses, z3.And(error <= ws, -error <= ws)))


# --- The per-block MAC bound (eps_w := 1.0 * Ws[block], not Ws[block] / 2) --

_K = 2  # matches quantized_mac_bound's / weight_only_quantize_matmul_nbits's
# own minimal-case convention: two SEPARATE one-element blocks is already
# the smallest layout exercising independent per-block scales.
_BLOCK_SIZE = 1  # each tap is its own block: Ws0 for tap 0, a genuinely
# different Ws1 for tap 1.
_NUM_BLOCKS = _K // _BLOCK_SIZE


def _block_of(k):
    return k // _BLOCK_SIZE


def _bound_formulas():
    """Z3 vocabulary for the single-operand, PER-BLOCK bounded-error claim,
    reused-in-structure (down-sized to ``_K = 2``) from ``weight_only_
    quantize_int4_matmul``'s own per-block generalization: ``X`` has no
    error term at all (never quantized), only ``W`` does, via the free
    per-tap error variable ``ew``, with one scale variable PER BLOCK
    (``Ws[b]``). The constant multiplying each block's scale is THIS file's
    own ``1.0`` (the codebook's worst-case half-gap, proved above), not the
    affine passes' ``0.5``.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # X[i, k], true float activation
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[k, n], true float weight
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[k, n] - Wdq[k, n]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]  # per-block scale

    dequant_w = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        # Each tap's error is bounded by ITS OWN block's scale, times the
        # codebook's own worst-case half-gap (1.0, NOT 0.5) -- Ws0 for tap
        # 0, a genuinely different Ws1 for tap 1.
        *[_abs(ew[k]) <= 1.0 * Ws[_block_of(k)] for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))

    bound = sum((1.0 * Ws[_block_of(k)]) * _abs(X[k]) for k in range(_K))

    return float_matmul, dequant_matmul, rounding_bounds, bound


def test_weight_only_quantize_mxfp4_matmul_error_is_bounded():
    # The genuine bounded-error claim: given W's own per-BLOCK rounding
    # bound (|W[k, n] - Wdq[k, n]| <= 1.0 * Ws[block_of(k), n]) and X
    # completely unchanged, the true float dot product and the one computed
    # against the dequantized weight cannot differ by more than
    # sum_k (1.0 * Ws[block_of(k), n]) * |X[i, k]| -- quantized_mac_bound's
    # own bound with eps_x fixed to 0, with a per-tap eps_w := 1.0 * Ws.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_weight_only_quantize_mxfp4_matmul_bias_variant_error_is_bounded():
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


def test_weight_only_quantize_mxfp4_matmul_negative_control_requires_rounding_bound():
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


def test_weight_only_quantize_mxfp4_matmul_uniform_bound_is_unsound_across_blocks():
    # Reproduces weight_only_quantize_int4_matmul's own "naive single-shared-
    # scale bound is unsound across blocks" negative control, for THIS
    # file's own 1.0 * Ws constant: if one instead (incorrectly) bounded
    # every tap's error using block 0's scale alone (1.0 * Ws0), that claim
    # is NOT a theorem once block 1's real scale Ws1 can exceed Ws0.
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]

    dequant_w = [W[k] - ew[k] for k in range(_K)]
    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= 1.0 * Ws[_block_of(k)] for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))
    error = float_matmul - dequant_matmul

    # The naive/wrong claim: bound every tap's contribution using ONLY
    # block 0's scale Ws0.
    uniform_bound = (1.0 * Ws[0]) * sum(_abs(X[k]) for k in range(_K))

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


def _nearest_mxfp4_code(normalized):
    """Independent from-scratch numpy nearest-codebook-index search (not
    imported from onnxsim.mx_quantization): for each value, the index of the
    codebook entry closest to it.
    """
    codebook = np.asarray(_MXFP4_CODEBOOK, dtype=np.float64)
    diffs = np.abs(normalized[..., np.newaxis] - codebook[np.newaxis, ...])
    return np.argmin(diffs, axis=-1).astype(np.uint8)


def _quantize_weight_blockwise_mxfp4_in_place(weight, channel_axis, block_size):
    """Independent numpy re-implementation of
    ``TryQuantizeWeightBlockwiseMXFP4InPlace`` (quantize_mxfp4_common.h):
    block-wise MXFP4 quantization of ``weight`` *in its own layout* (no
    transpose) -- one power-of-two scale per (block-of-reduction-axis,
    channel) group, via ``scale = 2 ** ceil(log2(max_abs_in_block / 6.0))``,
    and one nearest-codebook-index code per element of ``weight / scale``.
    Written fresh here (own control flow, own moveaxis bookkeeping) rather
    than calling into ``onnxsim.mx_quantization``'s own helper, so this is a
    genuine independent cross-check.
    """
    reduction_axis = 1 - channel_axis
    k = weight.shape[reduction_axis]
    assert k % block_size == 0
    num_blocks = k // block_size

    w_r = np.moveaxis(weight, reduction_axis, 0).astype(np.float64)  # [K, C]
    max_abs = np.stack(
        [
            np.max(np.abs(w_r[b * block_size : (b + 1) * block_size]), axis=0)
            for b in range(num_blocks)
        ],
        axis=0,
    )  # [num_blocks, C]
    max_abs = np.maximum(max_abs, 1e-30)
    shared_exponent = np.ceil(np.log2(max_abs / _MXFP4_MAX_MAGNITUDE))
    scale = np.exp2(shared_exponent).astype(np.float32)  # a pure power of two

    scale_bcast = np.repeat(scale.astype(np.float64), block_size, axis=0)  # [K, C]
    normalized = w_r / scale_bcast
    codes_r = _nearest_mxfp4_code(normalized)  # [K, C]

    codes = np.moveaxis(codes_r, 0, reduction_axis)
    scale_out = np.moveaxis(scale, 0, reduction_axis)
    return codes, scale_out


def _walk_mxfp4_chain(model, w_input):
    """Walks the exact ``Reshape <- Mul(Reshape <-, Reshape <-) <- Gather <-
    Cast`` chain backward from a matched layer's (post-rewrite) weight input,
    returning the nodes and initializer names needed to check every piece of
    it: ``(cast, gather, reshape1, reshape2, mul, reshape3, wq_name,
    codebook_name, ws_name, blocked_shape, scale_shape)``.
    """
    reshape3 = producer(model, w_input)
    assert reshape3.op_type == "Reshape"
    mul_out, _orig_shape_name = reshape3.input

    mul = producer(model, mul_out)
    assert mul.op_type == "Mul"
    reshape1_out, reshape2_out = mul.input

    reshape1 = producer(model, reshape1_out)
    assert reshape1.op_type == "Reshape"
    gather_out, blocked_shape_name = reshape1.input

    reshape2 = producer(model, reshape2_out)
    assert reshape2.op_type == "Reshape"
    ws_name, scale_shape_name = reshape2.input

    gather = producer(model, gather_out)
    assert gather.op_type == "Gather"
    codebook_name, cast_out = gather.input

    cast = producer(model, cast_out)
    assert cast.op_type == "Cast"
    (wq_name,) = cast.input

    def _shape_of(name):
        init = next(i for i in model.graph.initializer if i.name == name)
        return list(numpy_helper.to_array(init))

    return {
        "cast": cast,
        "gather": gather,
        "reshape1": reshape1,
        "reshape2": reshape2,
        "mul": mul,
        "reshape3": reshape3,
        "wq_name": wq_name,
        "codebook_name": codebook_name,
        "ws_name": ws_name,
        "blocked_shape": _shape_of(blocked_shape_name),
        "scale_shape": _shape_of(scale_shape_name),
    }


def _int_attr(node, name):
    return next(a.i for a in node.attribute if a.name == name)


def test_weight_only_quantize_mxfp4_matmul_pass_fires_and_matches_chain():
    # Plain MatMul (K=64, exactly two 32-element blocks). Confirms the exact
    # Cast -> Gather -> Reshape -> Mul(., Reshape) -> Reshape chain fires
    # (walked backward from Y, since simplify_isolated_extra skips the
    # default dead-code pass); the Gather's codebook matches MXFP4Codebook()
    # exactly, in order; the reduction_axis == 0 shape convention
    # (blocked_shape = [num_blocks, block_size, N], scale_shape =
    # [num_blocks, 1, N]); and Wq/Ws match the independent re-implementation.
    rng = np.random.default_rng(0)
    rows, K, N = 4, 64, 3
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

    sim_model = _simplify_isolated_extra_no_fold(
        model, "weight_only_quantize_mxfp4_matmul"
    )

    matmul_node = producer(sim_model, "Y")
    assert matmul_node.op_type == "MatMul"
    x_input, w_input = matmul_node.input
    assert x_input == "X"  # activation passed through completely unchanged

    chain = _walk_mxfp4_chain(sim_model, w_input)
    assert _int_attr(chain["cast"], "to") == onnx.TensorProto.INT64
    assert _int_attr(chain["gather"], "axis") == 0

    num_blocks = K // _MX_BLOCK_SIZE
    assert chain["blocked_shape"] == [num_blocks, _MX_BLOCK_SIZE, N]
    assert chain["scale_shape"] == [num_blocks, 1, N]

    codebook_init = next(
        i for i in sim_model.graph.initializer if i.name == chain["codebook_name"]
    )
    np.testing.assert_array_equal(
        numpy_helper.to_array(codebook_init), np.asarray(_MXFP4_CODEBOOK, np.float32)
    )

    wq_init = next(i for i in sim_model.graph.initializer if i.name == chain["wq_name"])
    ws_init = next(i for i in sim_model.graph.initializer if i.name == chain["ws_name"])
    wq = numpy_helper.to_array(wq_init)
    ws = numpy_helper.to_array(ws_init)
    assert wq.shape == (K, N)
    assert ws.shape == (num_blocks, N)
    assert wq.min() >= 0
    assert wq.max() <= 15

    expected_wq, expected_ws = _quantize_weight_blockwise_mxfp4_in_place(
        weight, 1, _MX_BLOCK_SIZE
    )
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-5)

    # Every scale is genuinely a power of two.
    log2_ws = np.log2(ws.astype(np.float64))
    np.testing.assert_allclose(log2_ws, np.round(log2_ws), atol=1e-9)


def test_weight_only_quantize_mxfp4_matmul_gemm_transb_uses_other_shape_convention():
    # PyTorch nn.Linear layout: weight [N, K], Gemm(X, W, B, transB=1), so
    # channel_axis=0 and reduction_axis=1 -- the OTHER blocked/scale shape
    # convention (blocked_shape = [N, num_blocks, block_size], scale_shape =
    # [N, num_blocks, 1]), and the bias is passed through untouched.
    rng = np.random.default_rng(1)
    rows, K, N = 3, 64, 2
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

    sim_model = _simplify_isolated_extra_no_fold(
        model, "weight_only_quantize_mxfp4_matmul"
    )

    gemm_node = producer(sim_model, "Y")
    assert gemm_node.op_type == "Gemm"
    x_input, w_input, b_input = gemm_node.input
    assert x_input == "X"
    assert b_input == "B"  # bias untouched

    chain = _walk_mxfp4_chain(sim_model, w_input)
    num_blocks = K // _MX_BLOCK_SIZE
    assert chain["blocked_shape"] == [N, num_blocks, _MX_BLOCK_SIZE]
    assert chain["scale_shape"] == [N, num_blocks, 1]

    wq_init = next(i for i in sim_model.graph.initializer if i.name == chain["wq_name"])
    ws_init = next(i for i in sim_model.graph.initializer if i.name == chain["ws_name"])
    wq = numpy_helper.to_array(wq_init)
    ws = numpy_helper.to_array(ws_init)
    assert wq.shape == (N, K)
    assert ws.shape == (N, num_blocks)

    expected_wq, expected_ws = _quantize_weight_blockwise_mxfp4_in_place(
        weight, 0, _MX_BLOCK_SIZE
    )
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-5)


def test_weight_only_quantize_mxfp4_matmul_declines_when_k_not_divisible_by_block_size():
    # TryQuantizeWeightBlockwiseMXFP4InPlace (and this pass's own
    # patternMatchPredicate) requires K % kMXBlockSize == 0; K=50 must
    # survive untouched.
    rng = np.random.default_rng(2)
    rows, K, N = 4, 50, 3
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

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_mxfp4_matmul", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["MatMul"]


def test_weight_only_quantize_mxfp4_matmul_output_within_proved_bound_via_ort():
    # Differential check against real ONNX Runtime output, with graph
    # optimization EXPLICITLY DISABLED (see test_ort_matmul_nbits_
    # workaround.py's docstring for the ORT graph-optimization-fusion bug
    # precedent this guards against): every output element's error against
    # the true float MatMul must stay within the proved per-block bound
    # (1.0 * Ws[block_of(k), n] per tap, NOT Ws / 2).
    rng = np.random.default_rng(3)
    rows, K, N = 4, 64, 3
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

    quantized = onnxsim.quantize_weight_only_mxfp4_cpp(model)
    onnx.checker.check_model(quantized)

    matmul_node = next(n for n in quantized.graph.node if n.op_type == "MatMul")
    _x_input, w_input = matmul_node.input
    chain = _walk_mxfp4_chain(quantized, w_input)
    ws_init = next(i for i in quantized.graph.initializer if i.name == chain["ws_name"])
    ws = numpy_helper.to_array(ws_init)  # [K/32, N]
    num_blocks = K // _MX_BLOCK_SIZE
    assert ws.shape == (num_blocks, N)

    # Every scale is genuinely a power of two.
    log2_ws = np.log2(ws.astype(np.float64))
    np.testing.assert_allclose(log2_ws, np.round(log2_ws), atol=1e-9)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        quantized.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    (y_quant,) = sess.run(["Y"], {"X": x})

    y_float = x @ weight
    error = np.abs(y_float - y_quant)

    eps_w = 1.0 * ws  # [K/32, N]: eps_w[block_of(k), n], NOT ws / 2
    block_of_k = np.arange(K) // _MX_BLOCK_SIZE  # [K]
    per_tap_eps = eps_w[block_of_k, :]  # [K, N]
    bound = np.einsum("ik,kn->in", np.abs(x), per_tap_eps)
    assert np.all(error <= bound + 1e-4)
