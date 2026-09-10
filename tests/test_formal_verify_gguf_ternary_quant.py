"""Formal check for GgufTernaryQuant (opt-in; ``onnxsim/passes/
gguf_ternary_quant.h``, C++ port of ``onnxsim/gguf_ternary_quant.py``'s own
``apply_gguf_ternary_quantization``): BitNet b1.58's published "absmean"
ternary weight quantization (Ma et al. 2024, Section 2), as shipped by
llama.cpp's GGUF ``TQ1_0``/``TQ2_0`` tensor types. Read the header comment and
``QuantizeDequantizeTernaryBlock`` (``gguf_ternary_quant.h``) in full before
this file: for one 256-element block ``w``::

    d       = mean(|w|)                          -- an ABSMEAN scale
    code    = round(w / d), then clipped to [-1, 1]   -- in {-1, 0, 1}
    dequant = code * d

Like ``test_formal_verify_iq4_nl.py`` (READ that file first -- closest
structural precedent: a fixed codebook plus a per-block scale, flat
channel-agnostic block layout, ragged final block), this rewrites::

    Y = MatMul(X, W) [+ bias]      W constant, 2-D, float32
into
    Y = MatMul(X, W') [+ bias]     W' -- SAME shape/dtype, no new nodes,
                                     every element replaced by its own
                                     256-element-block ternary round trip

Only MatMul/vanilla-Gemm (transA=0, alpha=1, beta=1) with a constant 2-D
float32 weight is matched -- the same scope IQ4_NL/FP6_LLM/gguf_legacy_quant
use (confirmed from ``GgufTernaryQuant::patternMatchPredicate``: no opset
gate at all, unlike ``dynamic_quantize_ternary_matmul.py``'s
``DynamicQuantizeLinear``-based rewrite -- this one never introduces an
integer op, so there is nothing opset-sensitive to gate on).

--------------------------------------------------------------------------
The OTHER ternary file in this suite is NOT the same scheme
--------------------------------------------------------------------------

``test_formal_verify_dynamic_quantize_ternary_matmul.py`` (read first, this
suite's other ``{-1,0,+1} * scale`` file) proves something genuinely
different: ``TryQuantizeWeightTernaryKN`` (``quantize_matmul_common.h``) only
*detects* a weight that is ALREADY exactly ``{-s, 0, +s}`` per output column,
with ``s := max(|column|)`` -- a lossless structural match, ``W == Wq * Ws``
EXACTLY, no rounding at all, and the predicate declines outright otherwise.
This pass is the opposite: it always fires (given the shape match) and always
introduces real, lossy rounding -- it does not require the weight to already
be ternary, it *forces* every weight into the ternary codebook. And its scale
is ``mean(|block|)``, not ``max(|block|)`` -- the first MEAN-scaled fixed
codebook in this suite. That distinction is the entire point of this file;
see the next section.

--------------------------------------------------------------------------
1. Why a MEAN-based scale changes the correctness argument completely
--------------------------------------------------------------------------

Every other fixed-codebook file in this suite (MXFP4, IQ4_NL, and this
suite's other ternary file) constructs its scale as (a ceiling over, or
exactly) ``max(|block|)``, which gives, BY CONSTRUCTION, an unconditional
guarantee that ``element / scale`` never leaves the codebook's own natural
range -- that guarantee is what lets those files prove a clean, universal
per-element bound of ``half_gap * scale``.

``d := mean(|block|)`` has no such guarantee. ``mean(|block|) <= max(|block|)``
always (proved below as a genuine, general fact -- not assumed), with
equality only for a perfectly uniform-magnitude block; for any block with
real variation, ``d`` is *strictly less* than the block's own max, so some
element's ``|value / d|`` CAN exceed 1 -- in fact can be made arbitrarily
large (a block with one large element and many near-zero ones drives ``d``
toward zero while that one element's magnitude stays fixed). Concretely
(verified with real numbers below, not just asserted): the 2-element block
``[1.0, 0.0]`` has ``d = 0.5``, so the first element normalizes to ``2.0`` --
already outside ``[-1, 1]``.

**This is why the C++'s explicit clip is genuinely load-bearing here**, unlike
every prior codebook file where the analogous clip (if any) is a vacuous
safety net a max-based scale already makes unreachable. But -- and this is
the honest finding this file does NOT paper over -- clipping does not, and
cannot, turn a mean-based scale into a bounded-relative-error scheme the way
IQ4_NL's/MXFP4's max-based scale is. Working out the actual worst-case
arithmetic (below) shows:

* The naive half-gap bound ``|error| <= 0.5 * d`` holds ``iff`` the element's
  own normalized value lies in ``[-1.5, 1.5]`` (NOT ``[-1, 1]`` -- the tighter
  ``[-1, 1]`` domain the task's own framing suggests is true but not tight;
  the bound in fact survives out to ``1.5`` because ``code`` saturates at
  ``+-1``, and ``|1.5 - 1| == 0.5`` exactly).
* That domain is NOT guaranteed by the mean-based scale construction (unlike
  IQ4_NL's ``m / M`` ratio, which is an EQUALITY landing exactly at the
  codebook's own boundary). A concrete witness (the same ``[1.0, 0.0]`` block)
  has an element normalizing to ``2.0``, outside ``[-1.5, 1.5]``, for which
  the ACTUAL dequantization error (``0.5``) is double the naive bound
  (``0.25``) -- a real, load-bearing counterexample, not a corner case
  invented for the proof.
* Once an element's magnitude is unbounded relative to ``d`` (which the mean
  construction allows), so is its own dequantization error -- there is no
  universal constant ``K`` such that ``|error| <= K * d`` holds for every
  element of every possible block. This is proved as a genuine unboundedness
  result below (a Z3 witness parametrized by an arbitrary budget ``B``,
  mirroring this suite's "unrestricted-domain claim is false" idiom but taken
  one step further: not just "the ``[-1,1]``-restricted bound fails outside
  it", but "no restatement of the bound with a bigger constant would ever
  fix it either").
* This is not a pathological-input-only phenomenon: a quick empirical check
  with an ordinary standard-normal 256-element block (``rng.standard_normal``,
  seed 0) finds 64 of 256 elements (25%) already outside the safe
  ``[-1.5, 1.5]`` domain, with the worst per-element error over 5x the naive
  ``0.5 * d`` bound. Ordinary, unremarkable random data routinely exercises
  this file's own honest finding -- it is not an edge case.

So this file's MAC-level bound (mirroring ``quantized_mac_bound``'s and
IQ4_NL's own per-block generalization) is stated CONDITIONALLY, on an
explicit "well-scaled" hypothesis per tap (``|W[k]| <= 1.5 * Ws[block]``) that
IQ4_NL/MXFP4 get for free from their own scale construction and this pass
does not -- proved NOT automatic via a concrete counterexample, per this
task's own instruction not to force-fit a bound that isn't actually true. A
genuinely true (if far looser) UNIVERSAL per-element bound does still exist
and is proved too: ``|error| <= |W[k]| + Ws[block]`` (plain triangle
inequality on ``dequant = code * d`` with ``|code| <= 1``) -- the honest
answer to "is there any bound at all that always holds", once the clean
``K * d`` shape is shown to fail.

--------------------------------------------------------------------------
2. Clipping's actual role: not a safety net, but a global nearest-neighbor
--------------------------------------------------------------------------

The C++ computes ``code = round(w / d)`` THEN clips to ``[-1, 1]`` --
round-then-clip, not the docstring's ``round(clip(w / d, -1, 1))``
(clip-then-round) at first glance. These are in fact the SAME function here
(unlike the general case), and this file proves the reason why, rather than
merely asserting it: because the codebook's own two extreme values (``-1``,
``+1``) exactly coincide with the clip bounds, ``clip(round(x), -1, 1)`` is,
for EVERY real ``x`` with no domain restriction needed at all, exactly the
argmin over ``{-1, 0, 1}`` of ``|x - c|`` -- a genuinely different, and
cleaner, correctness fact than every prior codebook file's own "scale keeps
you in range" lemma: clipping here does not merely fail to hurt, it
IMPLEMENTS global nearest-neighbor search by construction, for any real
input. What clipping does NOT do is make that nearest-neighbor's own
resulting error small -- see (1) above; being "the best of 3 codes" and
"close to the true value" are different claims once the true value can be
far from all three.

--------------------------------------------------------------------------
3. Ragged final block: real count, not the padded 256 (mean is not
   zero-padding-invariant)
--------------------------------------------------------------------------

``QuantizeDequantizeTernaryBlock``'s own ``count`` parameter (confirmed from
the source: the sum is divided by ``static_cast<double>(count)``, and
``runTransform``'s loop passes ``count = std::min(kBlockSize, numel - start)``
for the final, possibly-short block) means the ragged final block's mean is
computed over ONLY its own real elements. This matters in a way it does not
for gguf_legacy_quant.h's/IQ4_NL's max-based scales: a max is invariant to
zero-padding (a zero can never become the new max unless the whole block was
already zero), but a *mean* is diluted by every phantom zero -- dividing by
the full 256 instead of the real count ``n`` would silently shrink ``d`` by a
factor of ``n / 256``, which for a small ragged tail (e.g. ``n = 5``) is a
factor of roughly 51x. The differential test below
(``test_gguf_ternary_quant_ragged_block_uses_real_count_not_full_block_size``)
is built to FAIL if the pass used the wrong divisor: it plants a ragged
5-element tail whose "correct" scale (``d = 2.0``, dividing by 5) and "wrong"
scale (``d = 0.0390625``, dividing by the full 256) are so different that the
two hypotheses predict completely different quantized outputs for those
elements, and confirms the real compiled pass matches the correct one.

--------------------------------------------------------------------------
4. Float16 round-trip of ``d``
--------------------------------------------------------------------------

``RoundTripFloat16`` (``gguf_ternary_quant.h``) round-trips ``d`` through
``ggml_half`` (real IEEE754 binary16, via this repo's own already-verified
``FloatToFloat16Bits``/``Float16BitsToFloat32``, not a hand-rolled
approximation), matching real TQ1_0/TQ2_0's on-disk storage. This is folded
into this suite's usual "tracks closely, not bit-identically" tolerance
language (see ``gguf_ternary_quant.h``'s own "ACCEPTED, PERMANENT DIVERGENCE"
note) rather than modeled as a separate Z3 error term: float16's own
worst-case relative rounding error is at most ``2**-11`` (see
``test_formal_verify_quantize_fp16.py``'s own corollary), utterly negligible
next to the codebook's own coarse ``0.5 * d`` spacing -- so it is exercised
numerically (the differential tests below apply ``np.float16`` to ``d``,
mirroring the C++ exactly) rather than added as its own Z3 hypothesis.

--------------------------------------------------------------------------
Differential tests
--------------------------------------------------------------------------

Built via ``onnx.parser`` (per ``CLAUDE.md``) with ``numpy_helper.from_array``
for random weight initializers. Entry points (both exercised, mirroring
IQ4_NL's own two-entry-point convention): the opt-in optimizer name
``"gguf_ternary_quant"`` via ``simplify_isolated_extra``, and the dedicated
Python entry point ``onnxsim.apply_gguf_ternary_quantization_cpp``
(``onnxsim/quantize_entry.cpp``'s ``ApplyGgufTernaryQuant``). No
``com.microsoft`` contrib op or quantized ONNX tensor type is involved (plain
float32 in, float32 out), so ``onnx.reference.ReferenceEvaluator`` is used for
end-to-end numeric checks, never onnxruntime.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser
from onnx.reference import ReferenceEvaluator

import onnxsim

_BLOCK_SIZE = 256  # llama.cpp's own TQ1_0/TQ2_0 super-block size


def _abs(v):
    return z3.If(v >= 0, v, -v)


# ============================================================================
# 1. The codebook's own worst-case half-gap -- and where it actually stops
#    holding (NOT the same as the [-1, 1] domain the naive framing suggests).
# ============================================================================


def test_gguf_ternary_quant_codebook_half_gap_on_unit_domain():
    # Literally the bound the task asks for first: for every real x in
    # [-1, 1], some code in {-1, 0, 1} is within 0.5 of it. TRUE, but (see
    # the next test) not the tightest domain for which it's true.
    x = z3.Real("x")
    in_range = z3.And(x >= -1, x <= 1)
    prove(
        z3.Implies(
            in_range,
            z3.Or(_abs(x - (-1)) <= 0.5, _abs(x - 0) <= 0.5, _abs(x - 1) <= 0.5),
        ),
        msg="0.5 is not a valid half-gap bound for {-1,0,1} on [-1,1]",
    )


def test_gguf_ternary_quant_half_gap_bound_actually_extends_to_one_point_five():
    # The ACTUAL tight domain: because code saturates at +-1 (the codebook's
    # own extremes), the 0.5 bound survives all the way out to +-1.5, half
    # again as wide as the naive [-1, 1] framing. This is the domain that
    # matters for the "well-scaled" hypothesis used in the MAC bound below.
    x = z3.Real("x")
    in_range = z3.And(x >= -1.5, x <= 1.5)
    prove(
        z3.Implies(
            in_range,
            z3.Or(_abs(x - (-1)) <= 0.5, _abs(x - 0) <= 0.5, _abs(x - 1) <= 0.5),
        ),
        msg="0.5 is not a valid half-gap bound for {-1,0,1} on [-1.5,1.5]",
    )


def test_gguf_ternary_quant_half_gap_bound_is_tight_at_one_point_five():
    # x = 1.5 (nearest code is 1): distances are 2.5, 1.5, 0.5 -- the bound
    # is achieved with equality, not slack, at the edge of the domain above.
    distances = [abs(1.5 - c) for c in (-1, 0, 1)]
    assert min(distances) == 0.5, distances


def test_gguf_ternary_quant_half_gap_bound_has_no_universal_widening():
    # SURPRISE this file does NOT paper over: unlike IQ4_NL/MXFP4 (where the
    # analogous domain restriction is FREE -- their scale construction
    # guarantees it), there is no way to just widen the bound's constant to
    # make it universal either: for ANY budget B > 0, the point x = B + 2 has
    # every codebook distance strictly greater than B (distance to the
    # NEAREST code, 1, is B + 1 > B) -- so the error genuinely grows without
    # bound as x moves away from the codebook, not merely "exceeds 0.5
    # sometimes". This is what makes clipping load-bearing for VALIDITY
    # (confining the output to {-1,0,1} at all) but not sufficient for a
    # bounded-relative-error GUARANTEE the way a max-based scale's domain
    # restriction is.
    b = z3.Real("B")
    x = b + 2
    prove(
        z3.Implies(
            b > 0,
            z3.And(_abs(x - 1) > b, _abs(x - 0) > b, _abs(x - (-1)) > b),
        ),
        msg="the half-gap error does not actually grow without bound away from the codebook",
    )


# ============================================================================
# 2. Clipping is a global nearest-neighbor, for every real x, not a
#    domain-restricted safety net.
# ============================================================================


def test_gguf_ternary_quant_clip_after_round_is_a_global_nearest_neighbor():
    # code := clip(round(x), -1, 1) -- exactly GgufTernaryQuant's own order
    # (QuantizeDequantizeTernaryBlock: round first, THEN clip), modeling
    # round() the same way this suite's other files do (any integer within
    # 0.5 of x, matching test_formal_verify_quantize_fp16.py's own
    # round-to-nearest characterization, not a specific tie-break rule).
    # Claim: for EVERY real x (no domain restriction at all -- this is what
    # makes it a genuinely different fact from every per-codebook "stays in
    # range" lemma before it in this suite), code is a global argmin over
    # the codebook -- i.e. clipping never does worse, and always exactly
    # matches, an unclipped nearest-of-{-1,0,1} lookup performed directly on
    # x. This is the formal confirmation that clip-then-round (the
    # docstring's own restatement) and round-then-clip (the C++'s actual
    # order) compute the same thing here: both reduce to this same
    # nearest-neighbor characterization.
    x = z3.Real("x")
    raw = z3.Int("raw")
    round_hypothesis = _abs(x - z3.ToReal(raw)) <= 0.5
    code = z3.If(raw > 1, 1, z3.If(raw < -1, -1, raw))  # clip(raw, -1, 1)
    code_r = z3.ToReal(code)
    prove(
        z3.Implies(
            round_hypothesis,
            z3.And(
                _abs(x - code_r) <= _abs(x - (-1)),
                _abs(x - code_r) <= _abs(x - 0),
                _abs(x - code_r) <= _abs(x - 1),
            ),
        ),
        msg="clip(round(x), -1, 1) is not always a nearest-of-{-1,0,1} lookup",
    )


def test_gguf_ternary_quant_round_then_clip_matches_clip_then_round_concretely():
    # Concrete-arithmetic cross-check (this suite's own "verified with real
    # numbers" idiom, mirroring gguf_ternary_quant.py's own docstring
    # convention) of the order-equivalence claim above, over a spread of
    # values including both sides of every decision boundary and well
    # outside the codebook's own range.
    for x in (-5.0, -1.5, -1.0, -0.6, -0.5, -0.4, 0.0, 0.4, 0.5, 0.6, 1.0, 1.5, 5.0):
        round_then_clip = min(max(round(x), -1.0), 1.0)
        clip_then_round = round(min(max(x, -1.0), 1.0))
        assert round_then_clip == clip_then_round, x


# ============================================================================
# 3. mean(|block|) <= max(|block|), and why that inequality can be strict
#    enough to break the "well-scaled" domain the bound above needs.
# ============================================================================


def test_gguf_ternary_quant_mean_is_never_more_than_max():
    # The general arithmetic fact underlying everything above: for any block
    # (modeled here with 2 elements -- generalizes to any size), the mean of
    # the absolute values never exceeds their max. This is why a mean-based
    # scale can only ever be <= a max-based one for the same block -- it
    # never over-covers the codebook's range the way IQ4_NL's scale
    # construction is designed to do exactly.
    w1, w2 = z3.Reals("w1 w2")
    mean = (_abs(w1) + _abs(w2)) / 2
    m = z3.If(_abs(w1) >= _abs(w2), _abs(w1), _abs(w2))
    prove(mean <= m)


def test_gguf_ternary_quant_well_scaled_domain_is_not_guaranteed_by_mean_scale():
    # Unlike IQ4_NL's test_iq4_nl_scale_keeps_block_within_codebook_range
    # (an unconditional EQUALITY-based guarantee), a mean-based scale gives
    # NO such guarantee -- a concrete, minimal (2-element) witness: block =
    # [1.0, 0.0], d = mean(|block|) = 0.5, and the first element's own
    # normalized value is 1.0 / 0.5 = 2.0, outside even the WIDENED
    # [-1.5, 1.5] safe domain above.
    w1, w2 = 1.0, 0.0
    d = (abs(w1) + abs(w2)) / 2
    assert d == 0.5
    normalized = w1 / d
    assert normalized == 2.0
    assert not (-1.5 <= normalized <= 1.5)

    # And the same fact as a genuine (non-degenerate) Z3 existence result,
    # not just this one hand-picked pair: some 2-element block has its own
    # first element more than 1.5x the block's own mean-abs scale.
    ww1, ww2 = z3.Reals("ww1 ww2")
    dd = (_abs(ww1) + _abs(ww2)) / 2
    solver = z3.Solver()
    solver.add(dd > 0)
    solver.add(_abs(ww1) > 1.5 * dd)
    assert solver.check() == z3.sat, (
        "every element of every block is automatically within 1.5x the "
        "block's own mean-abs scale -- the well-scaled hypothesis the MAC "
        "bound below relies on would then be free, which is false for a "
        "mean-based (as opposed to max-based) scale"
    )


def test_gguf_ternary_quant_naive_half_gap_bound_is_violated_by_a_real_block():
    # The central honest finding of this file, made fully concrete (not just
    # an abstract Z3 possibility): using the SAME [1.0, 0.0] block above,
    # actually run the pass's own quantization rule (round, then clip) and
    # confirm the naive "error <= 0.5 * d" bound genuinely fails for a real,
    # valid quantization outcome -- not a hypothetical one.
    block = np.array([1.0, 0.0])
    d = np.mean(np.abs(block))
    assert d == 0.5
    normalized = block / d
    code = np.clip(np.round(normalized), -1.0, 1.0)
    dequant = code * d
    np.testing.assert_array_equal(code, [1.0, 0.0])
    np.testing.assert_array_equal(dequant, [0.5, 0.0])

    error = np.abs(block - dequant)
    naive_bound = 0.5 * d
    assert naive_bound == 0.25
    assert error[0] == 0.5
    assert error[0] > naive_bound, (
        "expected the naive per-element bound to be violated by this "
        "block's own first (outlier) element"
    )


def test_gguf_ternary_quant_naive_bound_violated_routinely_on_ordinary_data():
    # Not a pathological-input-only phenomenon: an ordinary, unremarkable
    # standard-normal 256-element block already violates the naive bound on
    # roughly a quarter of its own elements, with errors several times the
    # naive bound -- documented honestly here (an observation, not a
    # universal claim) rather than swept under an "edge case" label.
    rng = np.random.default_rng(0)
    block = rng.standard_normal(_BLOCK_SIZE)
    d = np.mean(np.abs(block))
    normalized = block / d
    code = np.clip(np.round(normalized), -1.0, 1.0)
    dequant = code * d
    error = np.abs(block - dequant)
    bound = 0.5 * d

    violations = np.sum(error > bound)
    assert violations > 0, (
        "expected the naive 0.5 * d bound to be genuinely violated on "
        "ordinary gaussian data -- if this ever fails, the empirical claim "
        "in this file's own docstring needs to be revisited, not deleted"
    )
    # A meaningful fraction, not a single outlier: roughly a quarter of a
    # standard-normal block's own elements sit outside the +-1.5 domain
    # (P(|Z| > 1.5 * E|Z|) is not small for a Gaussian).
    assert violations > _BLOCK_SIZE // 10


def test_gguf_ternary_quant_trivial_universal_bound_always_holds():
    # The honest replacement for the false "K * d" bound: a plain triangle
    # inequality that IS universally true, for every real w, d > 0 and
    # code in {-1, 0, 1} -- much looser than any codebook file's usual
    # half-gap bound, but genuinely unconditional, unlike the 0.5 * d shape.
    w, d = z3.Reals("w d")
    code = z3.Real("code")
    hypotheses = z3.And(d > 0, z3.Or(code == -1, code == 0, code == 1))
    dequant = code * d
    error = w - dequant
    bound = _abs(w) + d
    prove(z3.Implies(hypotheses, z3.And(error <= bound, -error <= bound)))


# ============================================================================
# 4. The conditional per-element bound (well-scaled hypothesis -> 0.5 * d)
#    and its composition into a single-operand MAC bound.
# ============================================================================


def test_gguf_ternary_quant_well_scaled_element_error_is_half_d_bound():
    # The corollary tying (1)'s domain lemma to a per-element error claim,
    # mirroring IQ4_NL's own "per_element_error_bound_is_codebook_half_gap_
    # times_scale" corollary, but here the "normalized lands near some code"
    # hypothesis is taken directly (justified by the [-1.5, 1.5] domain lemma
    # above -- NOT unconditionally true the way IQ4_NL's is, see (3) above).
    w, d, code = z3.Reals("w d code")
    hypotheses = z3.And(
        d > 0,
        z3.Or(code == -1, code == 0, code == 1),
        _abs(w / d - code) <= 0.5,
    )
    dequant = code * d
    error = w - dequant
    bound = 0.5 * d
    prove(z3.Implies(hypotheses, z3.And(error <= bound, -error <= bound)))


_K = 2  # matches quantized_mac_bound's / IQ4_NL's own minimal-case
# convention: two separate one-element blocks is already the smallest layout
# exercising independent per-block scales.


def _bound_formulas():
    """Z3 vocabulary for the single-operand, per-block bounded-error claim.
    Structurally identical to IQ4_NL's own ``_bound_formulas`` (``ew[k]``
    bounded by ``half_gap * Ws[block]``, here ``half_gap = 0.5``), but the
    per-tap hypothesis ``|ew[k]| <= 0.5 * Ws[k]`` is, for THIS pass,
    conditional on the well-scaled domain proved above rather than automatic
    -- see the dedicated tests above for why it is not free here the way it
    is for IQ4_NL/MXFP4.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # X[i, k], true float activation
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[k, n], true float weight
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[k, n] - Wdq[k, n]
    Ws = [z3.Real(f"Ws{k}") for k in range(_K)]  # per-block scale (own block per tap)

    dequant_w = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        *[Ws[k] > 0 for k in range(_K)],
        *[_abs(ew[k]) <= 0.5 * Ws[k] for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))

    bound = sum((0.5 * Ws[k]) * _abs(X[k]) for k in range(_K))

    return float_matmul, dequant_matmul, rounding_bounds, bound


def test_gguf_ternary_quant_error_is_bounded_given_well_scaled_taps():
    # Given the per-tap dequantization bound (itself only justified when
    # every tap's own element is well-scaled relative to its block, per the
    # dedicated tests above), the true float dot product and the one
    # computed against the dequantized weight cannot differ by more than
    # sum_k (0.5 * Ws[k]) * |X[i, k]|.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_gguf_ternary_quant_bias_variant_error_is_bounded():
    # This pass never touches Gemm's bias C -- adding the same Bias(n) to
    # both the true and the dequantized computation leaves their difference,
    # and therefore the bound on it, unchanged.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_matmul + bias) - (dequant_matmul + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_gguf_ternary_quant_negative_control_requires_dequant_bound():
    # Sanity check that the bound above is genuine: with no per-tap
    # dequantization bound at all (only Ws[k] > 0), the same MAC-level bound
    # is not a theorem.
    float_matmul, dequant_matmul, _rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul

    solver = z3.Solver()
    solver.add(z3.Real("Ws0") > 0, z3.Real("Ws1") > 0)
    solver.add(z3.Not(z3.And(error <= bound, -error <= bound)))
    assert solver.check() == z3.sat, (
        "the bound holds even without any per-tap dequantization bound -- "
        "negative control is vacuous"
    )


def test_gguf_ternary_quant_uniform_bound_is_unsound_across_blocks():
    # Mirrors IQ4_NL's own analogous negative control: a block can span an
    # arbitrary run of the weight's flattened storage (wrinkle shared with
    # IQ4_NL's own block layout), so bounding every tap's error using block
    # 0's scale alone is unsound once block 1's own (independent) scale can
    # exceed it.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul
    Ws0 = z3.Real("Ws0")
    uniform_bound = (0.5 * Ws0) * sum(_abs(z3.Real(f"X{k}")) for k in range(_K))

    solver = z3.Solver()
    solver.add(rounding_bounds)
    solver.add(z3.Not(z3.And(error <= uniform_bound, -error <= uniform_bound)))
    assert solver.check() == z3.sat, (
        "the naive single-scale (block 0 only) bound holds even though "
        "block 1 has its own, potentially larger scale"
    )


# ============================================================================
# Differential tests
# ============================================================================


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


def _quantize_dequantize_ternary_reimpl(weight):
    """Independent from-scratch numpy reimplementation of
    ``QuantizeDequantizeTernaryBlock``/``GgufTernaryQuant::runTransform``
    (``gguf_ternary_quant.h``): flattens ``weight`` in ROW-MAJOR order
    (matching ``ReadFloatMatrix``), one scale ``d = mean(|block|)`` per
    256-element block of the FLAT array (the final block RAGGED -- using
    only its own real elements, divided by its own real count, NOT the
    zero-padded 256 -- exactly matching ``runTransform``'s own
    ``count = min(kBlockSize, numel - start)`` loop), ``d`` round-tripped
    through float16 (matching ``RoundTripFloat16``), ``code = clip(round(w /
    d), -1, 1)``, ``dequant = code * d``. Written independently here (own
    loop, own control flow), not calling into ``onnxsim.gguf_ternary_quant``
    or the compiled pass itself.
    """
    original_shape = weight.shape
    flat = np.asarray(weight, dtype=np.float64).reshape(-1)
    n = flat.size
    out = np.empty_like(flat)
    for start in range(0, n, _BLOCK_SIZE):
        block = flat[start : start + _BLOCK_SIZE]
        count = block.size  # the block's own REAL element count -- never the
        # zero-padded 256, since a mean (unlike a max) is diluted by padding.
        d = max(float(np.sum(np.abs(block))) / count, 1e-12)
        d = float(np.float16(d))  # real TQ1_0/TQ2_0 stores d as fp16
        code = np.clip(np.round(block / d), -1.0, 1.0)
        out[start : start + count] = code * d
    return out.reshape(original_shape)


def _block_scale_reimpl(weight):
    """Same blocking as above, but returns the per-element block scale
    (same shape as ``weight``) -- used to build the real per-element error
    budget for the numeric bound checks below.
    """
    original_shape = weight.shape
    flat = np.asarray(weight, dtype=np.float64).reshape(-1)
    n = flat.size
    out = np.empty_like(flat)
    for start in range(0, n, _BLOCK_SIZE):
        block = flat[start : start + _BLOCK_SIZE]
        count = block.size
        d = max(float(np.sum(np.abs(block))) / count, 1e-12)
        d = float(np.float16(d))
        out[start : start + count] = d
    return out.reshape(original_shape)


def _current_weight(model, weight_input_index=1):
    # The pass rewires the matched node's weight input to a freshly created
    # initializer, leaving the original one dangling -- so the node's own
    # current input name is the only reliable way to find the actual
    # (post-quantization) weight.
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    return numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == w_name)
    )


def test_gguf_ternary_quant_matches_reimplementation_on_clean_multiple_of_block_size():
    # Single-shot application (no fixed-point re-iteration, unlike
    # simplify_isolated_extra -- see the next test's own note) on a weight
    # whose element count is an EXACT multiple of 256 (no ragged block at
    # all): confirms the pass's real output matches the independent
    # reimplementation exactly (up to float32/float16 rounding) for the
    # "clean" case, complementing the ragged-block-specific test below.
    rng = np.random.default_rng(7)
    rows, K, N = 4, 32, 16  # K * N = 512 = exactly two full 256-element blocks
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

    quantized = onnxsim.apply_gguf_ternary_quantization_cpp(model)
    onnx.checker.check_model(quantized)
    w_out = _current_weight(quantized)
    assert w_out.shape == (K, N)
    assert w_out.dtype == np.float32

    expected = _quantize_dequantize_ternary_reimpl(weight.astype(np.float64)).astype(
        np.float32
    )
    np.testing.assert_allclose(w_out, expected, rtol=1e-5, atol=1e-6)

    # Every 256-element block of the flattened output has at most 3 distinct
    # values (a shared +-d/0 per block), and the two blocks here have their
    # own independent scales (not forced to match).
    flat = w_out.reshape(-1)
    for start in range(0, flat.size, _BLOCK_SIZE):
        block = flat[start : start + _BLOCK_SIZE]
        assert len(np.unique(block)) <= 3


def test_gguf_ternary_quant_pass_fires_via_extra_optimizers():
    # Confirms the pass is reachable via extra_optimizers (this suite's
    # standard "is it actually registered" check) and rewrites the weight in
    # place (same shape/dtype, brand-new initializer name, no new nodes).
    # K * N = 2 * 256 = 512 = exactly two full 256-element blocks.
    #
    # NOTE: unlike IQ4_NL/MXFP4's max-based scale, this pass's mean-based
    # scale is NOT idempotent under repeated re-quantization -- re-running it
    # on its own already-ternary output generally shrinks the scale further
    # (many elements are now exactly 0, pulling the mean down), so
    # onnxsim's own fixed-point loop (onnxsim.simplify's default iteration,
    # which simplify_isolated_extra goes through) keeps re-matching and
    # drifting the weight toward zero over iterations instead of reaching a
    # stable point after one rewrite -- a genuine, if incidental, structural
    # difference from every max-scaled codebook pass in this suite, and a
    # reason (beyond this suite's usual dedicated-entry-point preference) to
    # check exact reimplementation-matching VALUES only through the
    # single-shot onnxsim.apply_gguf_ternary_quantization_cpp entry point
    # below, not through this iterated path. This test therefore only checks
    # structural properties that survive any number of iterations.
    rng = np.random.default_rng(0)
    rows, K, N = 4, 32, 16
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

    sim_model, _ops = simplify_isolated_extra(model, "gguf_ternary_quant", check_n=0)

    matmul_node = producer(sim_model, "Y")
    assert matmul_node.op_type == "MatMul"
    x_input, w_input = matmul_node.input
    assert x_input == "X"
    assert w_input != "W"

    w_out_init = next(i for i in sim_model.graph.initializer if i.name == w_input)
    assert w_out_init.data_type == onnx.TensorProto.FLOAT
    assert list(w_out_init.dims) == [K, N]

    # Structural sanity, robust to however many fixed-point iterations ran:
    # every 256-element block of the flattened output has at most 3 distinct
    # values (a shared +-d/0 per block).
    w_out = numpy_helper.to_array(w_out_init)
    flat = w_out.reshape(-1)
    for start in range(0, flat.size, _BLOCK_SIZE):
        block = flat[start : start + _BLOCK_SIZE]
        assert len(np.unique(block)) <= 3


def test_gguf_ternary_quant_declines_transposed_activation_gemm():
    # MatchMatMulLike requires transA == 0; patternMatchPredicate must
    # therefore decline a transA=1 Gemm outright.
    rng = np.random.default_rng(1)
    K, rows, N = 260, 4, 3  # X stored as [K, rows] so X^T @ W is well-typed.
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

    quantized = onnxsim.apply_gguf_ternary_quantization_cpp(model)
    assert quantized.SerializeToString() == model.SerializeToString()


def test_gguf_ternary_quant_declines_non_constant_weight():
    # patternMatchPredicate requires FetchConstantTensor(info.w) to succeed;
    # a weight that is a genuine graph input must be left untouched.
    rows, K, N = 4, 260, 3
    model = _model(
        f"""
        g (float[{rows},{K}] X, float[{K},{N}] W) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """
    )

    quantized = onnxsim.apply_gguf_ternary_quantization_cpp(model)
    assert quantized.SerializeToString() == model.SerializeToString()


def test_gguf_ternary_quant_declines_non_2d_weight():
    # patternMatchPredicate requires w_t->sizes().size() == 2; a Conv's 4-D
    # weight (this C++ port, unlike gguf_ternary_quant.py's own
    # include_conv option, never matches Conv at all -- see this module's
    # own docstring) must be left completely untouched.
    rng = np.random.default_rng(2)
    w = rng.standard_normal((2, 4, 4, 4)).astype(np.float32)
    model = _model(
        """
        g (float[1,2,8,8] X) => (float[1,2,5,5] Y)
        {
          Y = Conv(X, W)
        }
        """,
        [_f32(w, "W")],
    )

    quantized = onnxsim.apply_gguf_ternary_quantization_cpp(model)
    assert quantized.SerializeToString() == model.SerializeToString()


def test_gguf_ternary_quant_gemm_bias_untouched_and_blocked_by_own_flat_storage():
    # A "vanilla" Gemm (transA=0, alpha=1, beta=1) with a bias: the bias is
    # passed through unchanged. transB=1 (weight stored as [N, K]) confirms
    # blocking happens over the weight's OWN [N, K] flat storage directly,
    # not some canonicalized [K, N] view.
    rng = np.random.default_rng(3)
    rows, K, N = 3, 20, 15  # N * K = 300, not a multiple of 256 -- exercises
    # a ragged final block in the same test.
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.4
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

    quantized = onnxsim.apply_gguf_ternary_quantization_cpp(model)
    onnx.checker.check_model(quantized)

    gemm_node = next(n for n in quantized.graph.node if n.op_type == "Gemm")
    x_input, w_input, b_input = gemm_node.input
    assert x_input == "X"
    assert b_input == "B"  # bias untouched

    w_out_init = next(i for i in quantized.graph.initializer if i.name == w_input)
    assert list(w_out_init.dims) == [N, K]
    w_out = numpy_helper.to_array(w_out_init)

    expected = _quantize_dequantize_ternary_reimpl(weight.astype(np.float64)).astype(
        np.float32
    )
    np.testing.assert_allclose(w_out, expected, rtol=1e-5, atol=1e-6)


def test_gguf_ternary_quant_ragged_block_uses_real_count_not_full_block_size():
    # THE single most important differential check in this file (see this
    # module's own docstring, section 3): a weight whose flat element count
    # is 256 + 5 = 261, with the ragged 5-element tail set to values whose
    # "correct" scale (divide by the real count, 5) and "wrong" scale
    # (divide by the full 256) are wildly different -- 2.0 vs. 0.0390625,
    # a ~51x gap -- so the two hypotheses predict completely different
    # quantized outputs for those 5 elements. This test FAILS if the C++
    # ever divided a ragged block's sum by the full 256 instead of its own
    # real count.
    dim0, dim1 = 3, 87  # dim0 * dim1 = 261 = 256 + 5
    rng = np.random.default_rng(4)
    weight = (rng.standard_normal((dim0, dim1)) * 0.3).astype(np.float32)
    # Flat row-major indices 256..260 are row 2, columns 82..86 (2*87 = 174,
    # 174 + 82 = 256): the ragged tail, set to a clean, exactly
    # fp16-representable value.
    weight[2, 82:87] = 2.0
    model = _model(
        f"""
        g (float[4,{dim0}] X) => (float[4,{dim1}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_gguf_ternary_quantization_cpp(model)
    w_out = _current_weight(quantized)

    flat = weight.astype(np.float64).reshape(-1)
    ragged_tail = flat[256:261]
    np.testing.assert_array_equal(ragged_tail, [2.0] * 5)

    # The "correct" (real-count) scale for this tail: mean(|tail|) = 2.0,
    # exactly representable in float16, so the whole round trip is exact:
    # normalized = 1.0 -> code = 1 -> dequant = 2.0 (recovers the input
    # exactly, since it already sits exactly at the codebook boundary).
    correct_d = np.sum(np.abs(ragged_tail)) / ragged_tail.size
    assert correct_d == 2.0
    expected_correct = np.full(5, 2.0)

    # The "wrong" (full-256-divisor) scale a naive port might use instead:
    wrong_d = np.sum(np.abs(ragged_tail)) / _BLOCK_SIZE
    assert wrong_d == 0.0390625
    normalized_wrong = ragged_tail / wrong_d
    code_wrong = np.clip(np.round(normalized_wrong), -1.0, 1.0)
    expected_wrong = code_wrong * wrong_d
    np.testing.assert_array_equal(expected_wrong, [0.0390625] * 5)

    out_tail = w_out.reshape(-1)[256:261].astype(np.float64)
    np.testing.assert_allclose(out_tail, expected_correct, atol=1e-3)
    assert not np.allclose(out_tail, expected_wrong, atol=1e-3), (
        "the pass's own ragged-block output matches the WRONG (full-256-"
        "divisor) hypothesis instead of the correct real-count one"
    )

    # And the pass matches the independent reimplementation over the whole
    # weight, including the first (full) block.
    expected_full = _quantize_dequantize_ternary_reimpl(
        weight.astype(np.float64)
    ).astype(np.float32)
    np.testing.assert_allclose(w_out, expected_full, rtol=1e-5, atol=1e-6)


def test_gguf_ternary_quant_dequantized_values_are_exact_code_times_scale():
    # Every dequantized weight value equals scale(block) * (some code in
    # {-1, 0, 1}) exactly (up to float32/float16 rounding) -- the crux
    # structural check that the real pass's output actually IS a ternary
    # codebook lookup, not merely "close to" one.
    rng = np.random.default_rng(5)
    rows, K, N = 4, 17, 19  # K * N = 323 = 256 + 67: a full block plus a
    # sizable ragged tail.
    weight = rng.standard_normal((K, N)).astype(np.float32) * 1.1
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_gguf_ternary_quantization_cpp(model)
    w_out = _current_weight(quantized).astype(np.float64)

    scale = _block_scale_reimpl(weight.astype(np.float64))
    codes = np.divide(w_out, scale, out=np.zeros_like(w_out), where=scale > 0)
    rounded = np.round(codes)
    np.testing.assert_allclose(codes, rounded, atol=1e-3)
    assert np.all(np.abs(rounded) <= 1.0 + 1e-6)


def test_gguf_ternary_quant_output_close_to_reimplementation_via_reference_evaluator():
    # The full end-to-end check against a real ONNX graph execution (onnx's
    # own reference evaluator -- plain float32, no quantized tensor type or
    # contrib op involved). Documents the ACTUAL observed behavior honestly:
    # some individual weight elements DO exceed the naive 0.5 * d bound (this
    # file's own central finding), but the trivial universal bound
    # (|W| + scale) always holds, and the pass's output matches the
    # independent reimplementation closely.
    rng = np.random.default_rng(6)
    rows, K, N = 5, 24, 12  # K * N = 288 = 256 + 32
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

    quantized = onnxsim.apply_gguf_ternary_quantization_cpp(model)
    onnx.checker.check_model(quantized)
    w_out = _current_weight(quantized).astype(np.float64)

    expected_w = _quantize_dequantize_ternary_reimpl(weight.astype(np.float64))
    np.testing.assert_allclose(w_out, expected_w, rtol=1e-5, atol=1e-6)

    scale = _block_scale_reimpl(weight.astype(np.float64))
    weight64 = weight.astype(np.float64)
    per_element_error = np.abs(weight64 - w_out)
    naive_bound = 0.5 * scale
    trivial_bound = np.abs(weight64) + scale

    # The honest finding, confirmed against the REAL compiled pass's own
    # output, not just the reimplementation: the naive bound is genuinely
    # violated for some elements of ordinary random data ...
    assert np.any(per_element_error > naive_bound)
    # ... but the trivial universal bound never is.
    assert np.all(per_element_error <= trivial_bound + 1e-9)

    evaluator = ReferenceEvaluator(quantized)
    (y_quant,) = evaluator.run(None, {"X": x})
    y_float = x.astype(np.float64) @ weight64
    error = np.abs(y_float - y_quant.astype(np.float64))

    # MAC-level trivial bound (sum of the per-element trivial bound times
    # |X|) -- always true, composed the same way the conditional bound
    # would be, but without needing the well-scaled hypothesis.
    bound = np.abs(x.astype(np.float64)) @ trivial_bound
    assert np.all(error <= bound + 1e-6)


def test_gguf_ternary_quant_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_gguf_ternary_quantization_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
