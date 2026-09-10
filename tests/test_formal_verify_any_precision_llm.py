"""Formal check for AnyPrecisionLlm (opt-in; onnxsim's own
``onnxsim/passes/any_precision_llm.h``, C++ port of ``onnxsim/any_precision_llm.py``'s
own ``apply_any_precision_llm`` -- READ THAT MODULE'S DOCSTRING for the full
rationale and READ ``any_precision_llm.h`` IN FULL for the exact bisection
algorithm before touching this file).

This is Park et al., 2024, ICML 2024, "Any-Precision LLM: Low-Cost Deployment
of Multiple, Different-Sized LLMs" (https://arxiv.org/abs/2402.10517). It
rewrites ``Y = MatMul(X, W)`` (or a "vanilla" Gemm -- transA=0, alpha=1,
beta=1 -- bias left untouched) -- ``W`` a constant 2-D FLOAT32 tensor, ``X``
untouched -- into ``Y = MatMul(X, W')``, ``W'`` the same shape/dtype as ``W``
with every element replaced by its own (output-channel, K-block)
nested-bit-plane quantize-dequantize round trip.

GENUINELY DIFFERENT KIND OF CLAIM than every other quantization pass in this
suite (READ ``test_formal_verify_weight_only_quantize_int8_block_matmul.py``
FIRST for this suite's usual single-level-affine differential-test shape, and
``test_formal_verify_weight_only_quantize_mxfp4_matmul.py`` for a
fixed-codebook scheme's style -- but NEITHER file's central Z3 claim applies
here). Every other pass in this suite picks ONE bit-width/codebook and proves
a single-shot bounded-reconstruction-error MAC bound. This pass instead
builds a MAXIMUM-bit-width code ONCE, by repeatedly bisecting each bin at
that bin's OWN current min/max (confirmed directly from
``NestedBitplaneCodes`` in the header, not from prose: at each of ``max_bits``
rounds, every *existing* bin -- grouped by its current code -- is split at
``0.5 * (bin.min() + bin.max())``, appending one more low-order bit to every
member's code; a singleton bin just appends a trailing zero), and the
DEFINING, NOVEL property is that ANY lower bit-width's own code is recovered
EXACTLY from the max-depth code by a plain integer right-shift
(``max_codes >> (max_bits - bits)``, confirmed literally at the pass's own
call site: ``codes_b[i] = max_codes[i] >> (max_bits - bits)``) -- because a
bin is only ever refined (split into a subset of one of its own two halves),
never merged or reassigned across an old boundary, a ``k``-bit code's own bin
partition is a strict COARSENING of any deeper code's partition built from
the same tree. So this file's Z3 section is NOT another "bounded MAC error"
proof first and foremost -- it is a proof that TWO DIFFERENT WAYS of
computing a ``k``-bit code (truncate the max-depth code vs. run only the
first ``k`` bisection rounds) are the SAME function of the input, for every
free scalar value and every free pair of bin bounds. A secondary, more
familiar per-bit-width reconstruction-error bound is also proved (bin width
at bit-width ``b`` is at most the original range over ``2**b``, so an
idealized bin-CENTER reconstruction's worst-case error is at most half that,
shrinking monotonically as ``b`` grows) -- but see the honest caveat in the
next paragraph: that bound is proved for an idealized bin-CENTER
reconstruction, not for this pass's own actual per-bin-MEAN reconstruction,
confirmed by hand-constructing a counterexample while writing this file.

**Honest caveat found while writing this file, not assumed going in.** The
compiled pass reconstructs each element as the MEAN of its own bin's actual
member values (``DequantizeByBinMean``), not the bin's geometric CENTER --
the paper's own "maximum-likelihood constant reconstruction for a fixed
partition" (see any_precision_llm.py's own docstring on why: a per-bin mean
is guaranteed monotonically non-increasing in squared error as bins are
refined, by the law of total variance, whereas a single affine/linear fit is
not, since this scheme's bin boundaries are not laid out on any fixed
grid). The IDEALIZED bin-center bound (``error <= width / 2``) this file
proves in Z3 does NOT carry over as a per-element worst-case guarantee for
bin-MEAN reconstruction: a 3-member bin ``{lo, hi, hi}`` with the tracked
value ``v = lo`` reconstructs to ``mean = (lo + 2*hi) / 3``, giving error
``(2/3) * (hi - lo)`` -- strictly worse than the idealized ``(hi - lo) / 2``
bound. This is not a bug in the pass (the mean is still the best FIXED
constant for that bin in a mean-squared-error sense, and is what makes
reconstruction improve monotonically with more bits at all -- see the
Python module's own docstring), it just means this file's idealized
half-width bound is a *design-level* bound on what a bin-CENTER quantizer of
this same nested shape could guarantee, not a literal per-element theorem
about the shipped bin-MEAN pass. The differential section below therefore
checks the bin-mean pass's real per-element error against the always-true
FULL local bin width (not half of it), and separately confirms empirically
(typical random weight data, not adversarial) that real error sits well
inside the idealized half-range bound on average -- exactly the "well inside
the proved worst-case bound" pattern
``test_formal_verify_weight_only_quantize_int8_block_matmul.py`` already
uses for an unrelated reason (INT8's density), used here for a different,
more fundamental reason (mean vs. center reconstruction).

**Entry point used for every differential test below, and why.** Unlike
every other pass in this suite, this pass's own per-call parameters
(``bits``, ``max_bits``, ``block_size``) are NOT baked into the graph or
threaded through ``extra_optimizers`` at all -- ``quantize_entry.cpp``'s
``ApplyAnyPrecisionLlm`` sets three process-global mutable statics
(``AnyPrecisionLlmBits()`` etc., the same "global reference, reconfigured
per call" pattern ``QuarotSeed()``/``QuarotBlockSize()`` already use) right
before invoking ``OptimizeFixed`` with the registered pass by name. Reaching
this pass via ``simplify()``/``simplify_isolated_extra`` (i.e. via
``extra_optimizers=["any_precision_llm"]`` -- confirmed present in
``onnxsim.onnxsim_cpp2py_export._list_other_optimizers()``) would run it
with WHATEVER those statics were last set to by some earlier, unrelated call
in the same process -- an order-dependent hazard this file avoids entirely
by calling the dedicated nanobind entry point,
``onnxsim.apply_any_precision_llm_cpp(model, bits, max_bits, block_size)``
(``onnxsim/onnx_simplifier.py``'s wrapper of ``cpp2py_export.cc``'s
``apply_any_precision_llm``, mirroring ``apply_quarot_cpp``'s own dedicated-
entry-point precedent for a pass with per-call runtime parameters), which
sets the statics itself immediately before running -- deterministic
regardless of what else has run in this process. One structural test below
still confirms the pass genuinely IS registered under the opt-in name
(pinning the global state via an explicit no-op call first, documented
where it happens) since that registration is itself part of this pass's
contract.

Differential tests build a plain float ``MatMul``/``Gemm`` via
``onnx.parser`` (per ``CLAUDE.md``), with reasonably-sized weights (K=64,
several output channels) -- run purely through
``onnxsim.apply_any_precision_llm_cpp`` and inspected via
``numpy_helper``/direct initializer access; no ONNX Runtime session is
needed anywhere in this file (the rewrite stays in plain float32 MatMul/Gemm
the whole way, unlike a ``com.microsoft`` contrib-op quantization scheme).
Confirmed empirically: (a) the pass only fires on MatMul/vanilla-Gemm with a
constant 2-D float32 weight, matching the header's own scope, and declines
on transA!=0, alpha!=1, beta!=1 (with a bias), a non-constant weight, a
non-2-D (batched) weight, a non-FLOAT32 (double) weight, and any unrelated
op; (b) it handles a K NOT evenly divisible by ``block_size`` (a ragged last
block) correctly, unlike this suite's other blockwise passes
(``weight_only_quantize_int8_block_matmul``/``..._mxfp4_matmul``), confirmed
directly from the header's own ``for (start = 0; start < K; start +=
block_size)`` / ``end = min(start + block_size, K)`` loop, with no
divisibility check in ``patternMatchPredicate`` at all; (c) an independent,
from-scratch numpy reimplementation of ``NestedBitplaneCodes`` /
``DequantizeByBinMean`` (not imported from ``onnxsim.any_precision_llm`` --
a genuine cross-check) matches the compiled pass's actual output closely;
(d) real per-element reconstruction error stays within the always-true full
local-bin-width bound, and sits comfortably inside the idealized half-range
bound on typical (non-adversarial) random data; and, prioritized per this
pass's own most distinguishing property, (e) THE NESTING INVARIANT HOLDS ON
REAL, COMPILED-PASS-PRODUCED CODES: quantizing once to a high ``max_bits``
and materializing a lower ``bits`` gives (up to floating-point-summation-
order noise -- the header's own documented, accepted divergence from the
pure-Python port) the exact same float32 weight as quantizing directly with
``max_bits == bits`` -- checked against THREE different ceilings sharing one
low bit-width, not just two data points.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
import onnxsim.onnxsim_cpp2py_export as C
from _formal_verify_common import prove, z3
from onnx import parser

import onnxsim

# =============================================================================
# 1. Z3: the nesting invariant itself (this pass's genuinely novel claim)
# =============================================================================


def _bisect_step(v, lo, hi):
    """One within-bin bisection step (``NestedBitplaneCodes``'s own inner-loop
    body, specialized to a single already-isolated value ``v`` whose current
    bin is exactly ``[lo, hi]``): splits at the bin's own midpoint,
    ``0.5 * (lo + hi)`` -- exactly the header's own ``split`` formula, with
    ``lo``/``hi`` playing the role of that bin's own tracked min/max -- and
    returns ``(bit, new_lo, new_hi)`` for the half ``v`` falls into. Mirrors
    the header's own ``codes[i] = codes[i] * 2 + (values[i] >= split ? 1 :
    0)`` at the per-bit level: ``bit`` is that one appended low-order bit.
    """
    mid = (lo + hi) / 2
    above = v >= mid
    bit = z3.If(above, z3.IntVal(1), z3.IntVal(0))
    new_lo = z3.If(above, mid, lo)
    new_hi = z3.If(above, hi, mid)
    return bit, new_lo, new_hi


def _three_level_tree():
    """Builds the Z3 vocabulary for one free scalar ``v`` bisected 3 levels
    deep (``max_bits = 3``) from one free initial bin ``[lo0, hi0]`` --
    every level's bit/bounds are literally ``_bisect_step`` applied to the
    PREVIOUS level's own output, exactly as ``NestedBitplaneCodes`` builds
    one more bit per round from the current bin state, never looking ahead.
    Returns ``(domain, code1, code2, code3)``, the 1-, 2-, and 3-bit-wide
    codes built by running exactly that many bisection rounds.
    """
    v, lo0, hi0 = z3.Reals("v lo0 hi0")
    domain = z3.And(lo0 < hi0, v >= lo0, v <= hi0)

    bit1, lo1, hi1 = _bisect_step(v, lo0, hi0)
    bit2, lo2, hi2 = _bisect_step(v, lo1, hi1)
    bit3, _lo3, _hi3 = _bisect_step(v, lo2, hi2)

    code1 = bit1
    code2 = bit1 * 2 + bit2
    code3 = bit1 * 4 + bit2 * 2 + bit3  # the max_bits = 3 code
    return domain, code1, code2, code3


def test_nesting_right_shift_by_the_correct_amount_recovers_every_lower_bit_width_code():
    # THE claim: for a free value v and free initial bin [lo0, hi0], the
    # max_bits=3 code (code3, built by 3 full rounds of _bisect_step) right-
    # shifted by (max_bits - bits) reproduces EXACTLY the code that would
    # have been built by running only `bits` rounds directly (code1, code2)
    # -- not approximately, not "for most v", but as a Z3-proved identity
    # over every free v/lo0/hi0, forcing Z3 to check all 2**3 = 8 branches
    # of the nested If-conditions the three bisection rounds introduce.
    # Integer division by a positive literal (Z3's Int "/" is Euclidean,
    # which coincides with a plain right-shift for the always-nonnegative
    # codes built here) stands in for ">>".
    domain, code1, code2, code3 = _three_level_tree()
    prove(
        z3.Implies(
            domain,
            z3.And(
                code3 / 2 == code2,  # bits=2, max_bits=3: shift by 1
                code3 / 4 == code1,  # bits=1, max_bits=3: shift by 2
                code3 / 1 == code3,  # bits=3, max_bits=3: shift by 0 (identity)
            ),
        )
    )


def test_nesting_holds_for_every_max_bits_ceiling_sharing_a_low_bit_width():
    # A stronger form of the same claim: the 1-bit code recovered from the
    # max_bits=3 tree (code3 >> 2) must ALSO equal the 1-bit code recovered
    # from a max_bits=2 tree of the SAME v/lo0/hi0 (code2 >> 1) -- any deeper
    # ceiling you choose to build to, truncating down to bits=1 always lands
    # on the same value, since deeper rounds only ever refine bins that
    # bits=1's own two bins have already fixed. This is the "any precision"
    # part of the paper's name: one tree serves every ceiling, not just one
    # arbitrarily-chosen pair of (bits, max_bits).
    domain, code1, code2, code3 = _three_level_tree()
    prove(z3.Implies(domain, z3.And(code2 / 2 == code1, code3 / 4 == code1)))


# --- Negative controls -------------------------------------------------------


def test_negative_control_wrong_shift_amount_does_not_recover_the_lower_bit_code():
    # If the shift amount is wrong (shifting the max_bits=3 code by 1, as if
    # materializing bits=2, while actually comparing against the bits=1
    # code) the two need NOT agree -- Z3 must find a genuine counterexample,
    # confirming the exact shift amount (max_bits - bits) in the theorem
    # above is load-bearing, not an arbitrary choice that happens to work
    # for any shift.
    domain, code1, _code2, code3 = _three_level_tree()
    solver = z3.Solver()
    solver.add(domain)
    solver.add(code3 / 2 != code1)
    assert solver.check() == z3.sat, (
        "code3 >> 1 always equals code1 -- the shift amount (max_bits - bits) "
        "is not actually load-bearing, which would be wrong"
    )


def test_negative_control_independently_recomputed_code_neednt_match_the_nested_code():
    # The nesting theorem's own hypothesis is that BOTH codes come from the
    # SAME shared tree (the same v, built from the same initial bin [lo0,
    # hi0]). Re-deriving a "1-bit code" for the same v completely
    # independently -- e.g. from a genuinely different block's own [lo0,
    # hi0] statistics, as re-quantizing v from scratch against a different
    # block would -- need not agree with the nested code at all: Z3 finds a
    # v and two genuinely different bin ranges that disagree on which half v
    # falls into.
    v, lo0, hi0, alt_lo0, alt_hi0 = z3.Reals("v lo0 hi0 alt_lo0 alt_hi0")
    domain = z3.And(
        lo0 < hi0,
        v >= lo0,
        v <= hi0,
        alt_lo0 < alt_hi0,
        v >= alt_lo0,
        v <= alt_hi0,
    )
    bit1, _lo1, _hi1 = _bisect_step(v, lo0, hi0)
    alt_bit1, _alo1, _ahi1 = _bisect_step(v, alt_lo0, alt_hi0)

    solver = z3.Solver()
    solver.add(domain)
    solver.add(bit1 != alt_bit1)
    assert solver.check() == z3.sat, (
        "every independently-recomputed 1-bit code agrees with the nested "
        "one regardless of which block's own bin range it was built from -- "
        "the nesting theorem would then be trivial/vacuous, not a genuine "
        "consequence of sharing one tree"
    )


# =============================================================================
# 2. Z3: per-bit-width bin-width shrink and the (idealized) reconstruction
#    bound -- secondary claims, but each worth its own small lemma per the
#    task's own framing.
# =============================================================================


def test_bin_width_shrinks_by_at_most_half_each_bisection_level():
    # The GENERAL (not single-scalar-idealized) justification for "bin width
    # at bit-width b is at most the original range / 2**b": one bisection
    # round replaces a bin [lo, hi] with a bin [new_lo, new_hi] that is the
    # tight (member-min/member-max) bounds of EITHER the lower half (members
    # < mid, so bounded above by mid) or the upper half (members >= mid, so
    # bounded below by mid) -- confirmed from NestedBitplaneCodes's own
    # per-bin lo/hi recomputation, not assumed. Both shapes force the new
    # width to be at most half the old one, regardless of exactly where the
    # real member values inside that half happen to land (they may shrink
    # the bin far more than exactly half, if members cluster -- this lemma
    # only claims the upper bound the header's own construction guarantees).
    lo, hi, new_lo, new_hi = z3.Reals("lo hi new_lo new_hi")
    mid = (lo + hi) / 2
    lower_half = z3.And(lo <= new_lo, new_lo <= new_hi, new_hi <= mid)
    upper_half = z3.And(mid <= new_lo, new_lo <= new_hi, new_hi <= hi)
    hypotheses = z3.And(lo <= hi, z3.Or(lower_half, upper_half))
    prove(z3.Implies(hypotheses, new_hi - new_lo <= (hi - lo) / 2))


def test_three_level_bin_width_bound_composes_to_original_range_over_eight():
    # Composing the one-level lemma above 3 times (by simple transitivity --
    # Z3 does not need induction for a fixed, small depth, matching this
    # suite's usual "prove a fixed small number of concrete levels" style)
    # gives exactly the max_bits=3 case of "width_b <= width_0 / 2**b".
    w0, w1, w2, w3 = z3.Reals("w0 w1 w2 w3")
    hypotheses = z3.And(
        w0 >= 0,
        w1 >= 0,
        w2 >= 0,
        w1 <= w0 / 2,
        w2 <= w1 / 2,
        w3 <= w2 / 2,
    )
    prove(z3.Implies(hypotheses, w3 <= w0 / 8))


def test_idealized_bin_center_reconstruction_error_is_within_half_the_bin_width():
    # The idealized bound this file is honest about (see the module
    # docstring's caveat): reconstructing to the bin's own geometric CENTER
    # (not this pass's actual per-bin MEAN of real member values) bounds the
    # round-trip error by exactly half the bin's own width, for any v inside
    # it -- the standard uniform-quantizer error bound, restated here for
    # THIS scheme's own (possibly narrower-than-uniform, per the shrink
    # lemma above) bin width.
    lo, hi, v, recon = z3.Reals("lo hi v recon")
    hypotheses = z3.And(lo <= v, v <= hi, recon == (lo + hi) / 2)
    half_width = (hi - lo) / 2
    prove(
        z3.Implies(hypotheses, z3.And(v - recon <= half_width, recon - v <= half_width))
    )


def test_idealized_reconstruction_bound_shrinks_monotonically_with_more_bits():
    # bound(b) := original_range / 2**(b + 1) (combining the two lemmas
    # above): as b grows, the bound must shrink (or at least not grow) --
    # instantiated concretely for b = 0, 1, 2, 3 (bounds w0/2, w0/4, w0/8,
    # w0/16) rather than symbolic exponentiation, matching this suite's
    # usual small-concrete-chain style for a monotonicity corollary.
    w0 = z3.Real("w0")
    bound0, bound1, bound2, bound3 = w0 / 2, w0 / 4, w0 / 8, w0 / 16
    prove(
        z3.Implies(w0 > 0, z3.And(bound1 <= bound0, bound2 <= bound1, bound3 <= bound2))
    )


# =============================================================================
# 3. Z3: single-operand MAC bound corollary (the familiar shape every other
#    pass in this suite centers on), stated here as a downstream consequence
#    of the idealized per-element bound above, not this file's main claim.
# =============================================================================

_K = 2  # matches this suite's own minimal-case convention (quantized_mac_
# bound / weight_only_quantize_int8_block_matmul's own _K = 2): two SEPARATE
# one-element blocks is already the smallest layout exercising independent
# per-block bounds.
_BLOCK_SIZE = 1
_NUM_BLOCKS = _K // _BLOCK_SIZE


def _block_of(k):
    return k // _BLOCK_SIZE


def _abs(v):
    return z3.If(v >= 0, v, -v)


def _bound_formulas():
    """Z3 vocabulary for the single-operand, per-block bounded-error claim:
    X has no error term at all (never quantized), only W does, via the free
    per-tap error variable ew (the direct-error-variable idiom -- see
    test_formal_verify_dynamic_quantize_matmul.py's own module docstring for
    why this avoids a documented Z3 nonlinear-arithmetic hang). EpsW[b] is
    each block's own IDEALIZED per-element bound (original_range / 2 **
    (bits + 1), taken here as an opaque free positive real, matching this
    suite's own style of chaining separately-proved lemmas rather than
    re-deriving the range/2**bits arithmetic symbolically in the same
    query).
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    eps_w = [z3.Real(f"epsW{b}") for b in range(_NUM_BLOCKS)]

    dequant_w = [W[k] - ew[k] for k in range(_K)]
    rounding_bounds = z3.And(
        *[eps_w[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= eps_w[_block_of(k)] for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))
    bound = sum(eps_w[_block_of(k)] * _abs(X[k]) for k in range(_K))

    return float_matmul, dequant_matmul, rounding_bounds, bound


def test_any_precision_llm_mac_error_is_bounded_by_the_idealized_per_block_epsilon():
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_any_precision_llm_mac_bias_variant_error_is_bounded():
    # Gemm's bias C is never touched by this pass; it cancels out of the
    # error term algebraically, exactly as every affine weight-only pass's
    # own bias-variant proof in this suite shows.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_matmul + bias) - (dequant_matmul + bias)
    prove(
        z3.Implies(
            rounding_bounds,
            z3.And(error_with_bias <= bound, -error_with_bias <= bound),
        )
    )


def test_any_precision_llm_mac_uniform_bound_is_unsound_across_blocks():
    # This pass builds one INDEPENDENT bit-plane tree per (channel, K-block)
    # group -- two different blocks' own reconstructions have no relationship
    # to each other at all, so a single shared epsilon across blocks is
    # unsound the instant the two blocks' own idealized bounds differ,
    # mirroring every other blockwise pass's own analogous negative control
    # in this suite.
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    eps_w = [z3.Real(f"epsW{b}") for b in range(_NUM_BLOCKS)]

    dequant_w = [W[k] - ew[k] for k in range(_K)]
    rounding_bounds = z3.And(
        *[eps_w[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= eps_w[_block_of(k)] for k in range(_K)],
    )
    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))
    error = float_matmul - dequant_matmul

    uniform_bound = eps_w[0] * sum(_abs(X[k]) for k in range(_K))

    solver = z3.Solver()
    solver.add(rounding_bounds)
    solver.add(z3.Not(z3.And(error <= uniform_bound, -error <= uniform_bound)))
    assert solver.check() == z3.sat, (
        "the naive single-block-epsilon bound holds even though block 1 has "
        "its own, potentially larger epsilon -- this pass's per-block sum is "
        "not actually load-bearing, which would be wrong"
    )


# =============================================================================
# 4. Differential tests against the real compiled pass
# =============================================================================


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


def _f64(array, name):
    return numpy_helper.from_array(array.astype(np.float64), name)


def _current_weight(model, weight_input_index=1):
    # The pass (like every other weight-only pass in this suite) rewires the
    # matched node's weight input to a freshly created initializer, leaving
    # the original one dangling and unused in the graph -- so the node's own
    # CURRENT input name is the only reliable way to find the actual
    # (post-quantization) weight, not initializer list position or count.
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return numpy_helper.to_array(w_init)


# --- (a) predicate scope: fires / declines ----------------------------------


def test_pass_fires_on_plain_matmul_with_constant_2d_float_weight():
    rng = np.random.default_rng(0)
    rows, K, N = 4, 64, 16
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.6
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quant = onnxsim.apply_any_precision_llm_cpp(
        model, bits=4, max_bits=8, block_size=32
    )
    new_w = _current_weight(quant)
    assert new_w.shape == weight.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, weight)


def test_pass_fires_on_vanilla_gemm_transb_with_bias_untouched():
    rng = np.random.default_rng(1)
    rows, K, N = 3, 64, 8
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

    quant = onnxsim.apply_any_precision_llm_cpp(
        model, bits=4, max_bits=8, block_size=32
    )
    node = next(n for n in quant.graph.node if n.op_type == "Gemm")
    x_input, w_input, b_input = node.input
    assert x_input == "X"
    assert b_input == "B"

    new_w = _current_weight(quant)
    assert new_w.shape == weight.shape
    assert not np.array_equal(new_w, weight)
    bias_init = next(t for t in quant.graph.initializer if t.name == "B")
    np.testing.assert_array_equal(numpy_helper.to_array(bias_init), bias)


def test_pass_declines_gemm_with_nonzero_transA():
    weight = np.random.default_rng(2).standard_normal((8, 4)).astype(np.float32)
    model = _model(
        """
        g (float[8,4] X) => (float[4,4] Y)
        {
          Y = Gemm<transA = 1>(X, W)
        }
        """,
        [_f32(weight, "W")],
    )
    quant = onnxsim.apply_any_precision_llm_cpp(
        model, bits=4, max_bits=8, block_size=32
    )
    assert quant.SerializeToString() == model.SerializeToString()


def test_pass_declines_gemm_with_nondefault_alpha():
    weight = np.random.default_rng(3).standard_normal((8, 4)).astype(np.float32)
    model = _model(
        """
        g (float[4,8] X) => (float[4,4] Y)
        {
          Y = Gemm<alpha = 2.0>(X, W)
        }
        """,
        [_f32(weight, "W")],
    )
    quant = onnxsim.apply_any_precision_llm_cpp(
        model, bits=4, max_bits=8, block_size=32
    )
    assert quant.SerializeToString() == model.SerializeToString()


def test_pass_declines_gemm_with_nondefault_beta_when_bias_present():
    rng = np.random.default_rng(4)
    weight = rng.standard_normal((8, 4)).astype(np.float32)
    bias = rng.standard_normal(4).astype(np.float32)
    model = _model(
        """
        g (float[4,8] X) => (float[4,4] Y)
        {
          Y = Gemm<beta = 2.0>(X, W, B)
        }
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )
    quant = onnxsim.apply_any_precision_llm_cpp(
        model, bits=4, max_bits=8, block_size=32
    )
    assert quant.SerializeToString() == model.SerializeToString()


def test_pass_declines_non_constant_weight():
    # Both MatMul operands are graph inputs (neither is a constant), so
    # there is nothing to quantize ahead of time.
    model = _model(
        """
        g (float[4,8] X, float[8,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    quant = onnxsim.apply_any_precision_llm_cpp(
        model, bits=4, max_bits=8, block_size=32
    )
    assert quant.SerializeToString() == model.SerializeToString()


def test_pass_declines_non_2d_weight():
    # A batched MatMul with a 3-D constant weight falls outside "the common,
    # unambiguous shape" the header says this pass handles.
    weight = np.random.default_rng(5).standard_normal((2, 8, 4)).astype(np.float32)
    model = _model(
        """
        g (float[2,4,8] X) => (float[2,4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        [_f32(weight, "W")],
    )
    quant = onnxsim.apply_any_precision_llm_cpp(
        model, bits=4, max_bits=8, block_size=32
    )
    assert quant.SerializeToString() == model.SerializeToString()


def test_pass_declines_non_float32_weight():
    # patternMatchPredicate requires elem_type() == FLOAT specifically;
    # DOUBLE is a different, unhandled element type.
    weight = np.random.default_rng(6).standard_normal((8, 4))
    model = _model(
        """
        g (double[4,8] X) => (double[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        [_f64(weight, "W")],
    )
    quant = onnxsim.apply_any_precision_llm_cpp(
        model, bits=4, max_bits=8, block_size=32
    )
    assert quant.SerializeToString() == model.SerializeToString()


def test_pass_declines_unrelated_op():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    quant = onnxsim.apply_any_precision_llm_cpp(model)
    assert quant.SerializeToString() == model.SerializeToString()


def test_pass_registered_as_opt_in_optimizer_via_extra_optimizers():
    # Confirms the registration fact this file's module docstring relies on:
    # "any_precision_llm" is a genuine opt-in ("other") optimizer, reachable
    # via extra_optimizers, not just via the dedicated apply_any_precision_
    # llm_cpp entry point. The process-global bits/max_bits/block_size
    # statics are pinned to known values first via one explicit (discarded)
    # apply_any_precision_llm_cpp call, documenting the exact hazard this
    # file's other tests avoid by not using this path for anything numeric.
    assert "any_precision_llm" in C._list_other_optimizers()

    rng = np.random.default_rng(7)
    rows, K, N = 4, 64, 8
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.6
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    dummy = _model(
        """
        g (float[2,2] X) => (float[2,2] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        [_f32(np.zeros((2, 2), dtype=np.float32), "W")],
    )
    onnxsim.apply_any_precision_llm_cpp(dummy, bits=4, max_bits=8, block_size=32)

    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=0,
        extra_optimizers=["any_precision_llm"],
        skipped_optimizers=sorted(C._list_optimizers()),
    )
    assert check_ok
    new_w = _current_weight(sim_model)
    assert not np.array_equal(new_w, weight)


# --- (b) ragged last block ---------------------------------------------------


def test_pass_handles_ragged_last_block_not_evenly_divisible_by_block_size():
    # Unlike weight_only_quantize_int8_block_matmul/weight_only_quantize_
    # mxfp4_matmul (which decline outright on a non-divisible K),
    # NestedBitplaneCodes's own caller loop (`for (start = 0; start < K;
    # start += block_size) { end = min(start + block_size, K); ... }`) has
    # no divisibility requirement at all -- confirmed directly from the
    # header, with no special case needed since a ragged final block is
    # simply a smaller block. K=70 with the default block_size=32 gives
    # blocks of size 32, 32, 6.
    rng = np.random.default_rng(8)
    rows, K, N = 4, 70, 6
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.6
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    quant = onnxsim.apply_any_precision_llm_cpp(
        model, bits=4, max_bits=8, block_size=32
    )
    new_w = _current_weight(quant)
    assert new_w.shape == weight.shape
    assert not np.array_equal(new_w, weight)
    assert np.all(np.isfinite(new_w))


# --- Independent numpy reimplementation (not imported from
#     onnxsim.any_precision_llm) for (c)/(d)/(e) below ----------------------


def _bitplane_tree(values, max_bits):
    """From-scratch numpy reimplementation of any_precision_llm.h's own
    ``NestedBitplaneCodes``, generalized to return the code AND the tight
    (member-min/member-max) bin bounds at EVERY depth ``0..max_bits`` -- not
    just the deepest level -- since the reconstruction-bound checks below
    need the bin actually used for reconstruction at bit-width ``bits``,
    which is an ANCESTOR of the max-depth leaf, not the leaf itself.
    ``codes_by_depth[0]`` is the trivial single-bin (all-zero) code;
    ``codes_by_depth[d]`` for ``d >= 1`` is exactly what running only ``d``
    bisection rounds directly would produce.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    codes = np.zeros(n, dtype=np.int64)
    lo = np.full(n, values.min())
    hi = np.full(n, values.max())
    codes_by_depth = [codes.copy()]
    bounds_by_depth = [(lo.copy(), hi.copy())]

    for _ in range(max_bits):
        new_codes = codes.copy()
        new_lo = lo.copy()
        new_hi = hi.copy()
        for bin_id in np.unique(codes):
            mask = codes == bin_id
            bin_values = values[mask]
            if bin_values.size <= 1:
                new_codes[mask] = codes[mask] * 2
                continue
            split = 0.5 * (bin_values.min() + bin_values.max())
            idx = np.nonzero(mask)[0]
            above = bin_values >= split
            new_codes[idx] = codes[idx] * 2 + above.astype(np.int64)
            if np.any(~above):
                lower_idx = idx[~above]
                new_lo[lower_idx] = bin_values[~above].min()
                new_hi[lower_idx] = bin_values[~above].max()
            if np.any(above):
                upper_idx = idx[above]
                new_lo[upper_idx] = bin_values[above].min()
                new_hi[upper_idx] = bin_values[above].max()
        codes, lo, hi = new_codes, new_lo, new_hi
        codes_by_depth.append(codes.copy())
        bounds_by_depth.append((lo.copy(), hi.copy()))

    return codes_by_depth, bounds_by_depth


def _dequantize_by_bin_mean(values, codes):
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    for code in np.unique(codes):
        mask = codes == code
        out[mask] = values[mask].mean()
    return out


def _weight_nk(weight, weight_transposed):
    # Logical [N, K] (output-channel-first) view, matching the header's own
    # `at()` lambda / any_precision_llm.py's own `w_nk` convention.
    return weight if weight_transposed else weight.T


def _reference_quantize_row(row, bits, max_bits, block_size):
    """Independent reimplementation of ``_quantize_channel_nested`` (one
    output channel's own quantize-dequantize round trip): per block of
    ``block_size`` (ragged last block allowed), builds the depth-``max_bits``
    tree, truncates to ``bits``, and reconstructs by bin mean. Returns
    ``(recon, lo_at_bits, hi_at_bits)`` per element -- the bin bounds are the
    ones ACTUALLY used for reconstruction, at depth ``bits``.
    """
    k = row.shape[0]
    recon = np.empty(k)
    lo_out = np.empty(k)
    hi_out = np.empty(k)
    for start in range(0, k, block_size):
        end = min(start + block_size, k)
        block = row[start:end].astype(np.float64)
        codes_by_depth, bounds_by_depth = _bitplane_tree(block, max_bits)
        codes_at_bits = codes_by_depth[bits]
        lo_at_bits, hi_at_bits = bounds_by_depth[bits]
        recon[start:end] = _dequantize_by_bin_mean(block, codes_at_bits)
        lo_out[start:end] = lo_at_bits
        hi_out[start:end] = hi_at_bits
    return recon, lo_out, hi_out


# --- (c)/(e): the reimplementation's own internal nesting check ------------


def test_reference_reimplementation_itself_satisfies_the_nesting_invariant():
    # Before even touching the compiled pass: confirm THIS FILE's own
    # independent numpy reimplementation satisfies the same nesting
    # invariant proved symbolically above, on real (not 1-element-toy)
    # array data -- a from-scratch, purely empirical corroboration of the
    # Z3 proof, at a reasonably-sized block (64 elements).
    rng = np.random.default_rng(9)
    block = rng.standard_normal(64) * 0.7
    max_bits = 8
    codes_by_depth, _bounds = _bitplane_tree(block, max_bits)
    codes_max = codes_by_depth[max_bits]
    for bits in (1, 2, 3, 5, 8):
        direct = codes_by_depth[bits]
        shifted = codes_max >> (max_bits - bits)
        np.testing.assert_array_equal(
            shifted,
            direct,
            err_msg=f"nesting invariant violated at bits={bits} in the "
            f"reimplementation itself",
        )


# --- (c): compiled pass vs. independent reimplementation --------------------


def test_compiled_pass_matches_independent_reimplementation():
    rng = np.random.default_rng(10)
    rows, K, N, block_size = 4, 64, 6, 16
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

    for bits, max_bits in ((2, 8), (4, 8), (8, 8)):
        quant = onnxsim.apply_any_precision_llm_cpp(
            model, bits=bits, max_bits=max_bits, block_size=block_size
        )
        actual = _current_weight(quant).astype(np.float64)

        w_nk = _weight_nk(weight.astype(np.float64), weight_transposed=False)
        expected_nk = np.empty_like(w_nk)
        for i in range(w_nk.shape[0]):
            expected_nk[i, :], _lo, _hi = _reference_quantize_row(
                w_nk[i, :], bits, max_bits, block_size
            )
        expected = expected_nk.T  # back to [K, N] storage layout

        # Not bit-for-bit (hash-map bin grouping order vs. numpy's own
        # reduction order in the mean/min/max -- the header's own documented
        # divergence), but close: same partition, same reconstruction rule.
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"mismatch at bits={bits}, max_bits={max_bits}",
        )


# --- (d): per-element reconstruction error bounds ---------------------------


def test_reconstruction_error_is_within_the_full_local_bin_width():
    # The always-true, rigorous per-element bound (see the module docstring's
    # honest caveat about why HALF the width does not hold in general for
    # bin-MEAN reconstruction): both the true value and the bin mean lie
    # inside that same element's own final [lo, hi] (at the materialized
    # bit-width), so their difference cannot exceed the bin's own full
    # width. lo/hi are recomputed by the independent reimplementation above,
    # not read from the compiled pass (which does not expose them), and
    # cross-checked to actually correspond to the compiled pass's real
    # output via the previous test.
    rng = np.random.default_rng(11)
    rows, K, N, block_size = 4, 64, 5, 16
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.8
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    bits, max_bits = 4, 8
    quant = onnxsim.apply_any_precision_llm_cpp(
        model, bits=bits, max_bits=max_bits, block_size=block_size
    )
    actual = _current_weight(quant).astype(np.float64)

    w_nk = _weight_nk(weight.astype(np.float64), weight_transposed=False)
    error_nk = np.empty_like(w_nk)
    width_nk = np.empty_like(w_nk)
    for i in range(w_nk.shape[0]):
        row = w_nk[i, :]
        recon, lo, hi = _reference_quantize_row(row, bits, max_bits, block_size)
        error_nk[i, :] = np.abs(row - recon)
        width_nk[i, :] = hi - lo

    error = np.abs(actual.T - w_nk)  # actual is [K, N]; compare in [N, K]
    # error computed two ways above must agree (sanity: actual output really
    # is the reference reconstruction, established by the previous test);
    # the bound itself only needs error_nk/width_nk.
    np.testing.assert_allclose(error, error_nk, rtol=1e-4, atol=1e-4)
    assert np.all(error_nk <= width_nk + 1e-9)


def test_reconstruction_error_is_typically_well_inside_the_idealized_half_range_bound():
    # Typical-case (not adversarial) confirmation that, for ordinary random
    # weight data, real per-block reconstruction error sits comfortably
    # inside the idealized half-range bound this file's Z3 section proves
    # for a bin-CENTER reconstruction -- even though that bound is not a
    # per-element theorem for the pass's actual bin-MEAN reconstruction (see
    # the module docstring), it is, on average, not even close to violated
    # in practice: a per-block-mean-of-real-values reconstruction is a good
    # approximation of the block's own center for reasonably-distributed
    # data. Mirrors this suite's own "well inside the proved worst-case
    # bound" pattern (see test_formal_verify_weight_only_quantize_int8_
    # block_matmul.py) for a different underlying reason.
    rng = np.random.default_rng(12)
    rows, K, N, block_size = 4, 64, 8, 32
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.8
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    bits, max_bits = 4, 8
    quant = onnxsim.apply_any_precision_llm_cpp(
        model, bits=bits, max_bits=max_bits, block_size=block_size
    )
    actual = _current_weight(quant).astype(np.float64)
    w_nk = _weight_nk(weight.astype(np.float64), weight_transposed=False)
    error = np.abs(actual.T - w_nk)

    # The idealized bound at bits=4: original_block_range / 2**(bits + 1).
    num_blocks = -(-K // block_size)
    idealized_bound = np.empty_like(w_nk)
    for i in range(w_nk.shape[0]):
        row = w_nk[i, :]
        for b in range(num_blocks):
            start, end = b * block_size, min((b + 1) * block_size, K)
            block = row[start:end]
            block_range = block.max() - block.min()
            idealized_bound[i, start:end] = block_range / (2 ** (bits + 1))

    mean_error = error.mean()
    mean_bound = idealized_bound.T.mean()
    assert mean_error < 0.6 * mean_bound, (
        f"mean_error={mean_error:.6g} not comfortably inside the idealized "
        f"half-range bound mean_bound={mean_bound:.6g}"
    )


def test_reconstruction_error_improves_monotonically_with_more_bits():
    rng = np.random.default_rng(13)
    rows, K, N = 4, 64, 12
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    errors = []
    for bits in (1, 2, 4, 6, 8):
        quant = onnxsim.apply_any_precision_llm_cpp(
            model, bits=bits, max_bits=8, block_size=32
        )
        new_w = _current_weight(quant).astype(np.float64)
        errors.append(float(np.linalg.norm(new_w - weight.astype(np.float64))))

    for a, b in zip(errors, errors[1:]):
        assert b <= a + 1e-6
    assert errors[-1] < errors[0] * 0.5


# --- (e): the nesting invariant on REAL, compiled-pass-produced codes
#     (this pass's single most distinguishing empirical check) --------------


def test_nesting_holds_on_real_compiled_pass_output_across_three_ceilings():
    # The paper's own headline property, checked directly on the compiled
    # C++ pass's real float32 output rather than only symbolically: fixing
    # bits=3 and materializing it via THREE different max_bits ceilings
    # (3, 5, 8 -- max_bits=3 means "build directly to bits=3, no further
    # refinement", the other two mean "build deeper, then truncate") must
    # give the exact same quantized weight every time, up to floating-point
    # summation-order noise (the header's own documented, accepted
    # divergence -- hash-map bin grouping order, not numerically
    # significant since each bin's own split only ever looks at that bin's
    # own min/max). This is the single check that most directly
    # distinguishes this pass from every other quantizer in this suite: no
    # other pass in onnxsim has a "quantize once, deploy at multiple
    # precisions from the same artifact" contract to check at all.
    rng = np.random.default_rng(14)
    rows, K, N = 4, 64, 10
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.6
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    bits = 3
    results = {}
    for max_bits in (3, 5, 8):
        quant = onnxsim.apply_any_precision_llm_cpp(
            model, bits=bits, max_bits=max_bits, block_size=32
        )
        results[max_bits] = _current_weight(quant)

    baseline = results[3]
    for max_bits in (5, 8):
        np.testing.assert_allclose(
            results[max_bits],
            baseline,
            rtol=1e-5,
            atol=1e-6,
            err_msg=(
                f"bits={bits} materialized via max_bits={max_bits} does not "
                f"match the direct max_bits={bits} construction -- the "
                f"nesting invariant does not hold on real compiled-pass "
                f"output"
            ),
        )


def test_nesting_does_not_hold_between_genuinely_different_bit_widths():
    # Negative-control-style sanity check on real data: the SAME bits value
    # materialized from different max_bits ceilings must match (proved
    # above), but two GENUINELY DIFFERENT bits values (both built to the
    # same max_bits, i.e. two different truncations of the same tree) need
    # not produce the same weight -- confirming the previous test's equality
    # is a real, nontrivial confirmation of nesting, not an artifact of
    # apply_any_precision_llm_cpp ignoring its bits parameter altogether.
    rng = np.random.default_rng(15)
    rows, K, N = 4, 64, 10
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.6
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    low = _current_weight(
        onnxsim.apply_any_precision_llm_cpp(model, bits=2, max_bits=8, block_size=32)
    )
    high = _current_weight(
        onnxsim.apply_any_precision_llm_cpp(model, bits=7, max_bits=8, block_size=32)
    )
    assert not np.allclose(low, high, rtol=1e-5, atol=1e-6)


def test_pass_rejects_bits_outside_one_to_max_bits_range():
    model = _model(
        """
        g (float[4,8] X) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        [_f32(np.zeros((8, 4), dtype=np.float32), "W")],
    )
    try:
        onnxsim.apply_any_precision_llm_cpp(model, bits=9, max_bits=8)
        raised = False
    except Exception:
        raised = True
    assert raised

    try:
        onnxsim.apply_any_precision_llm_cpp(model, bits=0, max_bits=8)
        raised = False
    except Exception:
        raised = True
    assert raised
