"""Formal check for QOperatorQuantizeGemm (opt-in; onnxsim's own
``onnxsim/passes/qoperator_quantize_gemm.h``): the fully-general ``Gemm``
analogue of ``qoperator_quantize_matmul.h``'s ``QLinearMatMul`` rewrite
(``test_formal_verify_qoperator_quantize_matmul.py`` is this file's primary
template for the two-layer MAC-bound + output-round-trip composition -- read
it first) that ALSO needs ``qoperator_quantize_conv.h``'s own
INT32-bias-with-fixed-scale technique
(``test_formal_verify_qoperator_quantize_conv.py`` is this file's template for
the "bias quantized ahead of time to a fixed, non-calibrated scale" third
layer -- read it too). It targets ONNX Runtime's ``com.microsoft`` contrib op
``QGemm``, which -- unlike ``QLinearMatMul`` -- keeps ``transA``/``transB``/
``alpha`` as its OWN attributes, so this pass handles Gemm's FULL generality
(any ``transA``, ``transB``, ``alpha != 1``), not just the "vanilla"
(``transA=0``, ``alpha=1``) case ``qoperator_quantize_matmul`` handles::

    Aq = QuantizeLinear(A, As, Azp)                        -- As/Azp: CALIBRATED
    Yq = QGemm(Aq, As, Azp, Bq, Bs, Bzp, [Cq,] Ys, Yzp,
               transA=ta, transB=tb, alpha=al)              -- true int8 compute
    Y  = DequantizeLinear(Yq, Ys, Yzp)                      -- Ys/Yzp: CALIBRATED

``B`` is quantized per output channel (INT8, symmetric) IN ITS OWN STORAGE
LAYOUT via ``QuantizeWeightPerChannelInPlace`` -- ``channel_axis = transB != 0
? 0 : 1`` -- reusing the SAME helper ``qoperator_quantize_matmul``'s own
vanilla-Gemm case already uses (confirmed by reading both files' ``runTransform``
above): unlike ``QLinearMatMul``, which has no transpose attribute of its own
and so always reads its weight into a fixed ``[K, N]`` layout via
``QuantizeWeightPerChannelKN``, ``QGemm`` keeps ``transB`` as its own
attribute, so no forced transpose is needed here -- this part is template
reuse, not genuinely new content.

Soundness claim -- three layers of error, PLUS a new alpha-scaling wrinkle
============================================================================
Layer 1 (reused verbatim from ``qoperator_quantize_matmul``): the internal
int8 accumulation -- ``QuantizeLinear``/``QGemm`` reading ``Aq``/``Bq`` back at
``As``/``Bs``, before ``alpha`` is applied and before ``QGemm``'s own output is
re-quantized -- is mechanically identical to that pass's ``MatMul(Adq,
Bdq)``, so the exact same ``quantized_mac_bound`` instance
(``_bound_formulas()`` below, an exact copy of the matmul template's) bounds
how far the raw int8 dot product (before alpha, before bias, before output
round-trip -- call it ``quantized_matmul``) is from the true float dot product
(``float_matmul``), for one fixed output position: ``_K`` stands for the
FLATTENED, POST-``transA``/``transB`` reduction dimension -- i.e. ``X``/``W``
below are ``A'[i, :]``/``B'[:, n]`` (the LOGICAL operands after resolving
``transA``/``transB``, exactly the way ``Gemm``'s own spec defines
``A' = transA ? A^T : A``), not literal un-transposed rows/columns of the
graph's ``A``/``B`` inputs -- the same "flattened, not shape-specific"
adaptation ``qoperator_quantize_conv`` makes for a convolution's receptive
field::

    |A'@B'[i, n] - quantized_matmul[i, n]| <=
        (As / 2) * sum_k |B'[k, n]| + (Bs[n] / 2) * sum_k |A'[i, k]|
        + K * (As / 2) * (Bs[n] / 2)                                 -- "bound1"

Layer 1.5 -- the alpha-scaling wrinkle, genuinely new to this pass
--------------------------------------------------------------------
``QGemm``'s own documented contract multiplies the dequantized int8
accumulator by ``alpha`` -- confirmed textually from this pass's own
``runTransform``: ``Bq``'s bias-scale formula is ``bias_scale[n] = alpha *
a_scale * b_scale[n]`` (an EXPLICIT, separate ``alpha`` factor multiplying the
two quantization scales, not folded into either scale itself), which is only
sensible if ``QGemm``'s own raw pre-bias result is ``alpha * quantized_matmul``
(so that adding the ``alpha``-scaled bias in the SAME units, ``bias_scale``,
needs no further rescale -- exactly ``qoperator_quantize_conv``'s own
``bias_scale := x_scale * w_scale[c]`` reasoning, with the extra ``alpha``
factor this pass's ``alpha`` attribute makes necessary). So the value to
compare against the true ``alpha * A'@B'`` is ``alpha * quantized_matmul``,
and the natural claim is ``alpha`` times layer 1's own bound::

    |alpha * A'@B'[i, n] - alpha * quantized_matmul[i, n]| <= alpha * bound1   -- ???

That is FALSE in general. ``MatchGemmQuantizable`` (``qoperator_quantize_gemm.h``,
read in full above) places NO constraint whatsoever on ``info.alpha``'s sign or
magnitude -- it only ever calls ``GetValueFromAttrWithDefault(n, kalpha, 1.0)``
and later writes it straight onto the new node's ``alpha`` attribute; the only
attribute this pass rejects a Gemm over is ``beta`` (only when a bias is also
present, see below), never ``alpha``. So ``alpha`` can be negative (or zero),
and the bound has to survive that: multiplying an inequality
``|e| <= b`` (``b >= 0``) through by a possibly-negative ``alpha`` needs
``|alpha| * b``, not ``alpha * b`` -- confirmed as this file's own negative
control (``test_qoperator_quantize_gemm_alpha_scaled_bound_requires_abs_value``)
right after the lemma that proves the correct (``_abs``) version.

Layer 2 (reused verbatim from ``qoperator_quantize_matmul``'s own technique):
``Yq`` is itself a quantized representation of the raw, pre-output-quantization
result (``alpha * quantized_matmul``, or ``alpha * quantized_matmul + Bias``
when a bias is present -- see layer 3), into the fixed calibrated ``(Ys,
Yzp)`` range, with its own round-trip bound ``Ys / 2``.

Layer 3 (reused from ``qoperator_quantize_conv``'s technique, but with a
genuine difference in its own scale formula): when ``C`` is present,
``runTransform`` quantizes it AHEAD OF TIME into a fixed, non-calibrated INT32
tensor with zero_point 0 and per-column scale
``bias_scale[n] = alpha * a_scale * b_scale[n]`` -- unlike
``qoperator_quantize_matmul``'s Gemm-bias handling (which keeps ``C`` in FLOAT
and adds it back AFTER the output's own dequantization, so it cancels out of
the combined-bound error term algebraically and contributes nothing new -- see
that file's own bias-variant test), ``QGemm`` has no float fallback for its
bias, exactly like ``QLinearConv``. The difference from
``qoperator_quantize_conv``'s own bias-scale formula
(``x_scale * w_scale[c]``, no ``alpha`` factor at all, since ``QLinearConv``
has no ``alpha`` attribute to fold in) is exactly that extra ``alpha`` factor
-- stated and tested explicitly below
(``test_qoperator_quantize_gemm_bias_scale_includes_alpha``), not silently
reused. That extra factor also means ``bias_scale`` inherits ``alpha``'s
unrestricted sign: ``qoperator_quantize_conv``'s own bias round-trip lemma
safely uses ``bias_scale / 2`` (no ``abs``) because ITS ``bias_scale`` is a
product of two quantities BOTH always positive (an activation scale and a
per-channel weight scale). This pass's ``bias_scale`` can be NEGATIVE, so its
own round-trip lemma below is stated for a ``bias_scale`` of unrestricted
(nonzero) sign, with bound ``_abs(bias_scale) / 2`` -- confirmed as its own
negative control the same way the alpha lemma's is.

Composing all three layers: a genuinely new Z3-tractability wrinkle
======================================================================
Every sibling file in this family (``qoperator_quantize_matmul``,
``qoperator_quantize_conv``) composes its layers by handing Z3 ONE query that
both (a) reconstructs the raw MAC result from the direct-error-variable
``ex``/``ew`` idiom AND (b) adds the further round-trip/bias error terms on
top, and that query stays tractable (confirmed passing in this suite already).
Once ``alpha`` enters the picture, that no longer holds: confirmed directly
while writing this file (concrete timings from this environment) --

* the alpha-scaled MAC bound ALONE (``ex``/``ew`` reconstruction, multiplied
  through by a free ``alpha``, ``_abs(alpha) * bound1`` on the right) proves in
  about 25 seconds -- still within this suite's normal range;
* adding just ONE more linear term on top of that SAME ``ex``/``ew``-plus-
  ``alpha`` reconstruction -- the output's own round-trip error ``eout``, the
  cheapest possible extra term -- pushes the identical query past 100 seconds
  with no result (confirmed directly, no result up to that cutoff);
* the fully combined query (alpha-scaled MAC + bias round-trip + output
  round-trip, all in one ``ex``/``ew``-reconstructing query) hangs the same
  way.

This is exactly the nonlinear-blowup risk
``test_formal_verify_dynamic_quantize_matmul.py``'s own module docstring
documents and mitigates (see it for the original incident writeup) --
multiplying the already-nonlinear MAC reconstruction by one more free variable
(``alpha``) pushes the SAME kind of query past what this suite's `_K=2`
convention keeps tractable elsewhere. The fix here follows that file's own
mitigation one level further: rather than reconstruct the alpha-scaled MAC
layer's error from ``ex``/``ew`` inside the combined query, the combined
queries below introduce it as its OWN free, direct error variable ``e_mac``
(standing for ``alpha * (float_matmul - quantized_matmul)``), bounded DIRECTLY
by the hypothesis ``_abs(e_mac) <= _abs(alpha) * bound1`` -- exactly the
conclusion ``test_qoperator_quantize_gemm_alpha_scaled_mac_bound_holds`` proves
standalone, used here as a datum rather than re-derived. ``bound1`` itself is
still ``_bound_formulas()``'s own real formula (in ``As``, ``Bs``, and the
flattened ``A'``/``B'`` operands) -- not an opaque placeholder -- so the
combined bound is the literal same expression the standalone lemma proves; only
the ``ex``/``ew``-level reconstruction of ``quantized_matmul``/``float_matmul``
is left out of the combined query (confirmed: with that reconstruction
dropped, the combined queries below check in well under a second). This keeps
every individual Z3 query in the direct-error-variable shape this suite's
mitigation for this exact failure mode requires (``e_mac``, like ``eout`` and
``ebias``, is a free variable bounded directly by its own hypothesis, never
reconstructed from separate code/scale multiplicands in the SAME query) while
Z3 -- not hand algebra -- still verifies the triangle-inequality composition of
all three layers.

Predicate differences from ``qoperator_quantize_matmul``
============================================================
``QGemm`` has no ``beta`` attribute of its own -- its bias-scale formula
implicitly assumes ``beta = 1`` -- so ``MatchGemmQuantizable`` declines a Gemm
whose bias is present AND whose ``beta != 1.0``
(``if (beta != 1.0) return false;``, read directly above), the SAME
restriction ``quantize_matmul_common.h``'s ``MatchMatMulLike`` already applies
for the vanilla-Gemm case. And only a 1-D ``C`` of EXACTLY length ``N`` (the
common, unambiguous per-column-bias case) is handled; any other ``C`` shape is
left alone entirely. Both are confirmed below as dedicated differential tests
(unlike ``qoperator_quantize_conv``'s bias-shape restriction, which is about
``C`` not being a CONSTANT float32 ``[Cout]`` tensor at all, this pass's
restriction is specifically about ``beta`` and about ``C``'s exact shape/rank
being anything other than 1-D-length-``N``).

Confirming QGemm's CPU kernel before relying on it
======================================================
``tests/test_qoperator_quantize_gemm.py`` (this repo's own non-formal coverage
of this pass) already runs the real quantized graph -- vanilla, no-bias,
and the full ``transA=1``/``transB=1``/``alpha=2.5``-with-bias case -- through
``onnxruntime.InferenceSession`` successfully (confirmed passing in this
environment: ``python3 -m pytest tests/test_qoperator_quantize_gemm.py -q``).
This file's own numeric-bound differential test below additionally disables
graph optimization (``ORT_DISABLE_ALL``) -- confirmed empirically to still run
correctly -- per this suite's now-established precedent for exactly this
"a QDQ/QOperator-shaped chain might get fused into a different code path" risk
(see ``tests/test_ort_matmul_nbits_workaround.py``'s docstring for the real ORT
graph-optimization-fusion bug this guards against).

Invoking the pass
===================
Unlike ``qoperator_quantize_matmul``/``qoperator_quantize_conv`` (which share
ONE entry point, ``quantize_qoperator``, running
``OptimizeFixed(["qoperator_quantize_matmul", "qoperator_quantize_conv"])``),
this pass has its OWN dedicated nanobind entry point,
``onnxsim.onnxsim_cpp2py_export.quantize_qoperator_gemm(model_bytes,
activation_ranges)`` -- confirmed by reading ``QuantizeQOperatorGemm`` in
``onnxsim/quantize_entry.cpp`` (``OptimizeFixed(model, {"qoperator_quantize_gemm"})``)
and its nanobind binding in ``onnxsim/cpp2py_export.cc``. It already isolates
exactly this one pass, so, as with the sibling files' own entry points, no
extra ``skipped_optimizers``/``simplify_isolated_extra`` isolation is needed.
``activation_ranges`` must supply a calibrated range for BOTH ``A``'s own name
and the Gemm node's own output name (``"Y"`` for the single-node models built
below) -- the same "quantizes its own output too" requirement every pass in
this ``qoperator_quantize_*`` family has.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
import onnxsim.onnxsim_cpp2py_export as C
import pytest
from _formal_verify_common import producer, prove, z3
from onnx import parser

assert "qoperator_quantize_gemm" in C._list_other_optimizers()

_K = 2  # concrete number of contraction taps -- matches quantized_mac_bound's/
# qoperator_quantize_matmul's/qoperator_quantize_conv's own choice; see their
# docstrings for why (enough to exercise the cross-tap sum, empirically the
# largest tractable value for this shape of nonlinear Z3 query).


def _abs(v):
    return z3.If(v >= 0, v, -v)


def _bound_formulas():
    """Layer 1's Z3 vocabulary -- an exact copy of
    ``test_formal_verify_qoperator_quantize_matmul.py``'s own
    ``_bound_formulas`` (same direct-error-variable ``ex``/``ew`` idiom, same
    ``_K``): ``QGemm``'s internal int8 accumulation (before ``alpha`` is
    applied, before any bias, before ``QGemm``'s own output is re-quantized)
    is mechanically identical to that pass's ``MatMul(Adq, Bdq)``, so the same
    lemma applies unchanged. ``X``/``W`` here stand for the FLATTENED,
    post-``transA``/``transB`` logical operands (``A'[i, :]``/``B'[:, n]``),
    not literal un-transposed rows/columns of the graph's own ``A``/``B``
    inputs -- see this file's module docstring.

    Returns ``(float_matmul, quantized_matmul, rounding_bounds, bound)``.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # A'[i, k], true float activation
    W = [z3.Real(f"W{k}") for k in range(_K)]  # B'[k, n], true float weight
    ex = [z3.Real(f"ex{k}") for k in range(_K)]  # A'[i, k] - Adq[i, k]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # B'[k, n] - Bdq[k, n]
    As = z3.Real("As")  # calibrated per-tensor activation scale
    Bs = z3.Real("Bs")  # this pass's per-output-channel weight scale Bs[n]

    Adq = [X[k] - ex[k] for k in range(_K)]
    Bdq = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        As > 0,
        Bs > 0,
        *[_abs(ex[k]) <= As / 2 for k in range(_K)],
        *[_abs(ew[k]) <= Bs / 2 for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    quantized_matmul = sum(Adq[k] * Bdq[k] for k in range(_K))

    bound = (
        (As / 2) * sum(_abs(W[k]) for k in range(_K))
        + (Bs / 2) * sum(_abs(X[k]) for k in range(_K))
        + _K * (As / 2) * (Bs / 2)
    )

    return float_matmul, quantized_matmul, rounding_bounds, bound


def test_qoperator_quantize_gemm_layer1_error_is_bounded():
    # Layer 1: unchanged from qoperator_quantize_matmul's own proof -- QGemm's
    # raw int8 accumulation (before alpha, before bias, before the output's
    # own re-quantization) obeys the exact same quantized_mac_bound instance.
    float_matmul, quantized_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - quantized_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_qoperator_quantize_gemm_alpha_scaled_mac_bound_holds():
    # Layer 1.5, genuinely new to this pass: QGemm's raw pre-bias,
    # pre-output-quantization result is `alpha * quantized_matmul`, to be
    # compared against the true `alpha * float_matmul` (see this file's module
    # docstring for the textual evidence -- Bq's own bias-scale formula
    # already includes `alpha` explicitly, which only makes sense if the raw
    # MAC result QGemm accumulates before adding the bias is itself
    # alpha-scaled). Since MatchGemmQuantizable applies NO constraint to
    # info.alpha's sign, `alpha` here is a free Real of UNRESTRICTED sign, and
    # the bound needs `_abs(alpha)`, not `alpha` -- confirmed insufficient
    # (without abs) as this file's own negative control right below.
    float_matmul, quantized_matmul, rounding_bounds, bound1 = _bound_formulas()
    alpha = z3.Real("alpha")
    alpha_error = alpha * (float_matmul - quantized_matmul)
    alpha_bound = _abs(alpha) * bound1
    prove(
        z3.Implies(
            rounding_bounds,
            z3.And(alpha_error <= alpha_bound, -alpha_error <= alpha_bound),
        )
    )


def test_qoperator_quantize_gemm_alpha_scaled_bound_requires_abs_value():
    # Sanity check that _abs(alpha) is genuinely required, not a conservative-
    # but-avoidable strengthening: with the SIGNED multiplier `alpha * bound1`
    # instead, the previous lemma's own hypotheses are satisfiable together
    # with a violation of the signed-bound claim -- Z3 finds a real
    # counterexample (a negative alpha, exactly the case MatchGemmQuantizable
    # never rules out).
    float_matmul, quantized_matmul, rounding_bounds, bound1 = _bound_formulas()
    alpha = z3.Real("alpha")
    alpha_error = alpha * (float_matmul - quantized_matmul)
    signed_bound = alpha * bound1
    solver = z3.Solver()
    solver.add(rounding_bounds)
    solver.add(
        z3.Not(z3.And(alpha_error <= signed_bound, -alpha_error <= signed_bound))
    )
    assert solver.check() == z3.sat, (
        "the alpha-scaled bound holds even with the signed `alpha * bound1` "
        "multiplier -- negative control is vacuous (alpha's sign must matter)"
    )


def test_qoperator_quantize_gemm_bias_round_trip_holds():
    # Layer 3's own round-trip lemma, reusing qoperator_quantize_conv's
    # round-trip-lemma-then-triangle-inequality technique for the bias, but
    # with an explicit difference from that file's version: conv's
    # bias_scale = x_scale * w_scale[c] is a product of two quantities BOTH
    # always positive, so its own lemma safely bounds by bias_scale / 2 (no
    # abs needed). This pass's bias_scale[n] = alpha * a_scale * b_scale[n]
    # carries the EXTRA alpha factor conv's formula has no analogue for, and
    # -- per this file's alpha-sign finding above -- that makes bias_scale
    # itself able to be NEGATIVE. So this lemma is stated for a bias_scale of
    # unrestricted (nonzero) sign, with bound _abs(bias_scale) / 2 -- still
    # exactly the round(x / scale) rounding shape with zero_point fixed at 0
    # (`n` models round(bias / bias_scale); "no clipping" means it's exactly
    # the INT32 code QGemm reads back, runTransform's own clamp not hit).
    bias, bias_scale = z3.Reals("bias bias_scale")
    n = z3.Int("n")  # round(bias / bias_scale)
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        bias_scale != 0,
        n - (bias / bias_scale) <= half,
        (bias / bias_scale) - n <= half,
    )
    dequant_bias = n * bias_scale  # zero_point is fixed at 0

    error = bias - dequant_bias
    prove(
        z3.Implies(
            hypotheses,
            z3.And(error <= _abs(bias_scale) / 2, -error <= _abs(bias_scale) / 2),
        )
    )


def test_qoperator_quantize_gemm_bias_round_trip_requires_abs_value():
    # Symmetric negative control to the alpha one above: the SIGNED
    # bias_scale / 2 bound is not sufficient once bias_scale can be negative
    # -- Z3 finds a real counterexample (a negative bias_scale).
    bias, bias_scale = z3.Reals("bias bias_scale")
    n = z3.Int("n")
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        bias_scale != 0,
        n - (bias / bias_scale) <= half,
        (bias / bias_scale) - n <= half,
    )
    dequant_bias = n * bias_scale
    error = bias - dequant_bias
    signed_bound = bias_scale / 2

    solver = z3.Solver()
    solver.add(hypotheses)
    solver.add(z3.Not(z3.And(error <= signed_bound, -error <= signed_bound)))
    assert solver.check() == z3.sat, (
        "the bias round-trip bound holds even with the signed bias_scale / 2 "
        "bound -- negative control is vacuous (bias_scale's sign must matter)"
    )


def test_qoperator_quantize_gemm_combined_bound_holds():
    # The no-bias combined claim: Y (DequantizeLinear(Yq, Ys, Yzp)) is
    # `alpha * quantized_matmul`'s own further round trip through a SECOND
    # QuantizeLinear/DequantizeLinear-shaped step, `eout := (alpha *
    # quantized_matmul) - Y`, bounded by `Ys / 2` exactly like any other
    # single-tensor round trip in this suite. As explained at length in this
    # file's module docstring, the alpha-scaled MAC layer's own error is
    # modeled here as its OWN free, direct error variable `e_mac` (standing
    # for `alpha * (float_matmul - quantized_matmul)`), bounded DIRECTLY by
    # `_abs(e_mac) <= _abs(alpha) * bound1` -- exactly the conclusion
    # test_qoperator_quantize_gemm_alpha_scaled_mac_bound_holds proves
    # standalone -- rather than reconstructed from ex/ew inside this same
    # query (confirmed to hang past 100s otherwise). Given both hypotheses,
    # Z3 -- not hand algebra -- confirms the triangle-inequality composition:
    # |alpha * float_matmul - Y| <= _abs(alpha) * bound1 + Ys / 2.
    _, _, _, bound1 = _bound_formulas()
    As = z3.Real("As")
    Bs = z3.Real("Bs")
    alpha = z3.Real("alpha")
    Ys = z3.Real("Ys")  # calibrated per-tensor OUTPUT scale
    eout = z3.Real("eout")
    true_scaled = z3.Real("true_scaled")  # alpha * float_matmul
    y_raw = z3.Real("y_raw")  # alpha * quantized_matmul, QGemm's raw result
    e_mac = true_scaled - y_raw
    y = y_raw - eout

    alpha_mac_bound = z3.And(As > 0, Bs > 0, _abs(e_mac) <= _abs(alpha) * bound1)
    output_round_trip = z3.And(Ys > 0, _abs(eout) <= Ys / 2)
    hypotheses = z3.And(alpha_mac_bound, output_round_trip)

    combined_bound = _abs(alpha) * bound1 + Ys / 2
    error = true_scaled - y
    prove(
        z3.Implies(
            hypotheses, z3.And(error <= combined_bound, -error <= combined_bound)
        )
    )


def test_qoperator_quantize_gemm_combined_bound_bias_variant():
    # Layer 3 added on top: with a bias present, QGemm's own int32
    # accumulator sums the alpha-scaled raw dot product with the
    # pre-quantized bias, both already expressed in the SAME "alpha * As *
    # Bs" units (bias_scale[n] := alpha * As * Bs[n] is exactly why no extra
    # rescale step is needed -- the same reasoning qoperator_quantize_conv's
    # own bias variant uses, with the extra alpha factor folded in). UNLIKE
    # qoperator_quantize_matmul's Gemm-bias variant (where Bias cancels out
    # algebraically because it rides through in float, untouched, entirely
    # outside the quantized round trip), this ebias term does NOT cancel --
    # it is a genuine third additive error budget, since the bias itself is
    # quantized here, exactly like qoperator_quantize_conv's own bias variant.
    # Z3 confirms the three-way triangle-inequality composition:
    # |(alpha * float_matmul + Bias) - Y_with_bias| <=
    #     _abs(alpha) * bound1 + _abs(bias_scale) / 2 + Ys / 2.
    _, _, _, bound1 = _bound_formulas()
    As = z3.Real("As")
    Bs = z3.Real("Bs")
    alpha = z3.Real("alpha")
    Ys = z3.Real("Ys")
    eout = z3.Real("eout")
    bias_scale = z3.Real("bias_scale")
    ebias = z3.Real("ebias")  # Bias[n] - dequant(Bias_q)[n]
    Bias = z3.Real("Bias")
    true_scaled = z3.Real("true_scaled")  # alpha * float_matmul
    y_raw_no_bias = z3.Real("y_raw_no_bias")  # alpha * quantized_matmul
    e_mac = true_scaled - y_raw_no_bias

    y_raw_with_bias = y_raw_no_bias + (Bias - ebias)
    y_with_bias = y_raw_with_bias - eout
    float_result = true_scaled + Bias
    error = float_result - y_with_bias

    hypotheses = z3.And(
        As > 0,
        Bs > 0,
        _abs(e_mac) <= _abs(alpha) * bound1,
        Ys > 0,
        _abs(eout) <= Ys / 2,
        bias_scale != 0,
        _abs(ebias) <= _abs(bias_scale) / 2,
    )
    combined_bound = _abs(alpha) * bound1 + _abs(bias_scale) / 2 + Ys / 2
    prove(
        z3.Implies(
            hypotheses, z3.And(error <= combined_bound, -error <= combined_bound)
        )
    )


def test_qoperator_quantize_gemm_negative_control_requires_alpha_scaled_mac_bound():
    # The no-bias combined bound genuinely needs the alpha-scaled MAC bound
    # hypothesis, not just the output round-trip one: with the output round
    # trip assumed but NO budget at all on e_mac, the combined claim is not a
    # theorem -- Z3 must find a real counterexample.
    _, _, _, bound1 = _bound_formulas()
    As = z3.Real("As")
    Bs = z3.Real("Bs")
    alpha = z3.Real("alpha")
    Ys = z3.Real("Ys")
    eout = z3.Real("eout")
    true_scaled = z3.Real("true_scaled")
    y_raw = z3.Real("y_raw")
    y = y_raw - eout

    combined_bound = _abs(alpha) * bound1 + Ys / 2
    error = true_scaled - y

    solver = z3.Solver()
    solver.add(As > 0, Bs > 0, Ys > 0, _abs(eout) <= Ys / 2)  # no bound on e_mac
    solver.add(z3.Not(z3.And(error <= combined_bound, -error <= combined_bound)))
    assert solver.check() == z3.sat, (
        "the combined bound holds even without the alpha-scaled MAC bound "
        "hypothesis -- negative control is vacuous"
    )


def test_qoperator_quantize_gemm_negative_control_requires_output_bound():
    # Symmetric negative control: with the alpha-scaled MAC bound assumed but
    # NO budget at all on the output's own round-trip error (eout free), the
    # no-bias combined claim is likewise not a theorem.
    _, _, _, bound1 = _bound_formulas()
    As = z3.Real("As")
    Bs = z3.Real("Bs")
    alpha = z3.Real("alpha")
    Ys = z3.Real("Ys")
    eout = z3.Real("eout")
    true_scaled = z3.Real("true_scaled")
    y_raw = z3.Real("y_raw")
    e_mac = true_scaled - y_raw
    y = y_raw - eout

    combined_bound = _abs(alpha) * bound1 + Ys / 2
    error = true_scaled - y

    solver = z3.Solver()
    solver.add(As > 0, Bs > 0, _abs(e_mac) <= _abs(alpha) * bound1)  # no eout bound
    solver.add(z3.Not(z3.And(error <= combined_bound, -error <= combined_bound)))
    assert solver.check() == z3.sat, (
        "the combined bound holds even without the output's own round-trip "
        "bound -- negative control is vacuous"
    )


def test_qoperator_quantize_gemm_negative_control_requires_bias_bound():
    # The bias-specific negative control: with the alpha-scaled MAC bound AND
    # the output's own round-trip bound both assumed, but NO budget at all on
    # the bias's own rounding error (ebias free), the combined-with-bias claim
    # is not a theorem either -- confirming the third error term is a genuine
    # additional hypothesis, not one already implied by the other two.
    _, _, _, bound1 = _bound_formulas()
    As = z3.Real("As")
    Bs = z3.Real("Bs")
    alpha = z3.Real("alpha")
    Ys = z3.Real("Ys")
    eout = z3.Real("eout")
    bias_scale = z3.Real("bias_scale")
    ebias = z3.Real("ebias")
    Bias = z3.Real("Bias")
    true_scaled = z3.Real("true_scaled")
    y_raw_no_bias = z3.Real("y_raw_no_bias")
    e_mac = true_scaled - y_raw_no_bias

    y_with_bias = y_raw_no_bias + (Bias - ebias) - eout
    float_result = true_scaled + Bias
    error = float_result - y_with_bias

    combined_bound = _abs(alpha) * bound1 + _abs(bias_scale) / 2 + Ys / 2

    solver = z3.Solver()
    solver.add(
        As > 0,
        Bs > 0,
        _abs(e_mac) <= _abs(alpha) * bound1,
        Ys > 0,
        _abs(eout) <= Ys / 2,
        bias_scale != 0,  # no bound on ebias
    )
    solver.add(z3.Not(z3.And(error <= combined_bound, -error <= combined_bound)))
    assert solver.check() == z3.sat, (
        "the combined-with-bias bound holds even without the bias's own "
        "rounding-error hypothesis -- negative control is vacuous"
    )


def _model(body, initializer=(), opset=13, ir_version=10):
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


def _quantize_qoperator_gemm(model, activation_ranges):
    """Invokes the real compiled pass directly via the nanobind-exposed
    ``quantize_qoperator_gemm(model_bytes, activation_ranges)`` -- this pass's
    OWN dedicated entry point (unlike ``qoperator_quantize_matmul``/
    ``qoperator_quantize_conv``'s shared ``quantize_qoperator``), running
    ``OptimizeFixed(["qoperator_quantize_gemm"])`` -- see this file's module
    docstring. It already isolates exactly this pass, so no extra
    ``skipped_optimizers``/``simplify_isolated_extra`` isolation is needed.
    """
    out = onnx.ModelProto()
    out.ParseFromString(
        C.quantize_qoperator_gemm(model.SerializeToString(), activation_ranges)
    )
    return out


def _quantize_weight_per_channel_in_place(weight, channel_axis):
    """Independent numpy re-implementation of
    ``QuantizeWeightPerChannelInPlace`` (quantize_matmul_common.h) for a 2-D
    weight quantized IN ITS OWN LAYOUT (no transpose): per-channel
    (``channel_axis`` 0 or 1) symmetric INT8 quantization, scale =
    max(|channel|) / 127 (or 1.0 for an all-zero channel), codes =
    round(w / scale) clipped to [-127, 127]. Identical formula to
    ``test_formal_verify_qoperator_quantize_matmul.py``'s own
    ``_quantize_weight_per_channel_kn`` helper except no transpose is ever
    applied here -- this pass reads ``B`` in its own storage layout regardless
    of ``transB``.
    """
    reduce_axis = 1 - channel_axis
    scale = np.max(np.abs(weight), axis=reduce_axis)
    scale = np.where(scale > 0, scale / 127.0, 1.0).astype(np.float32)
    scale_bcast = scale[:, np.newaxis] if channel_axis == 0 else scale[np.newaxis, :]
    codes = np.clip(np.round(weight / scale_bcast), -127, 127).astype(np.int8)
    return codes, scale


def _expected_asymmetric_uint8_quant_params(min_val, max_val):
    """Independent re-implementation of ``ComputeAsymmetricUint8QuantParams``
    (static_quantize_matmul.h), in float32 to match the pass's own arithmetic
    precision -- identical to every sibling formal-verify file's helper of the
    same name, since this pass reads the exact same global/function for both
    its activation AND its output.
    """
    lo = min(np.float32(0.0), np.float32(min_val))
    hi = max(np.float32(0.0), np.float32(max_val))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zero_point = int(np.clip(np.round(-lo / scale), 0, 255))
    return np.float32(scale), zero_point


def _quantize_gemm_bias_int32(bias, alpha, a_scale, b_scale):
    """Independent numpy re-implementation of ``runTransform``'s bias
    quantization (``qoperator_quantize_gemm.h``): per-output-column scale
    ``bias_scale[n] = alpha * a_scale * b_scale[n]`` (the EXTRA ``alpha``
    factor ``qoperator_quantize_conv``'s own bias formula has no analogue
    for), zero_point fixed at 0, codes = round(bias[n] / bias_scale[n])
    clipped to the exact INT32 range -- mirrors the C++'s own
    ``std::clamp`` against ``std::numeric_limits<int32_t>::min()/max()`` done
    in ``double`` (unlike ``qoperator_quantize_conv``'s bias clamp, which
    narrows to the largest float32 value <= INT32_MAX since ITS clamp is done
    in float32; this pass's clamp is done in double, which represents the
    full int32 range exactly, so no analogous narrowing is needed here).
    """
    bias_scale = np.float64(alpha) * np.float64(a_scale) * b_scale.astype(np.float64)
    q = np.round(bias.astype(np.float64) / bias_scale)
    q = np.clip(q, -2147483648.0, 2147483647.0)
    return q.astype(np.int64).astype(np.int32), bias_scale


def test_qoperator_quantize_gemm_pass_fires_and_matches_scheme():
    # Build a plain, vanilla-shaped (transA=0, transB=0, alpha=1, no bias)
    # Gemm and run the real pass with calibration ranges for BOTH A and the
    # node's own output "Y", chosen to straddle 0 so both Azp/Yzp come out
    # genuinely nonzero -- mirroring qoperator_quantize_matmul's own
    # analogous test.
    rng = np.random.default_rng(0)
    rows, K, N = 4, 5, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = Gemm(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    a_range = (-5.0, 10.0)
    y_range = (-3.0, 12.0)
    quantized = _quantize_qoperator_gemm(model, {"X": a_range, "Y": y_range})

    # No float Gemm left anywhere -- the genuinely distinguishing structural
    # check vs. QDQ format (static_quantize_matmul always keeps the original
    # Gemm node).
    op_types = {n.op_type for n in quantized.graph.node}
    assert "Gemm" not in op_types

    # Walk the chain backward from the real graph output:
    # DequantizeLinear(Yq) <- QGemm(Aq, ...) <- QuantizeLinear(X).
    dq_node = producer(quantized, "Y")
    assert dq_node.op_type == "DequantizeLinear"
    qgemm_node = producer(quantized, dq_node.input[0])
    assert qgemm_node.op_type == "QGemm"
    assert qgemm_node.domain == "com.microsoft"
    domains = {o.domain for o in quantized.opset_import}
    assert "com.microsoft" in domains

    assert len(qgemm_node.input) == 9
    assert qgemm_node.input[6] == ""  # no bias -> empty-string placeholder

    trans_a = next(a.i for a in qgemm_node.attribute if a.name == "transA")
    trans_b = next(a.i for a in qgemm_node.attribute if a.name == "transB")
    alpha_attr = next(a.f for a in qgemm_node.attribute if a.name == "alpha")
    assert trans_a == 0
    assert trans_b == 0
    assert alpha_attr == pytest.approx(1.0)

    assert dq_node.input[1:] == [
        qgemm_node.input[7],
        qgemm_node.input[8],
    ]  # Ys, Yzp shared

    ql_node = producer(quantized, qgemm_node.input[0])
    assert ql_node.op_type == "QuantizeLinear"
    assert ql_node.input[0] == "X"
    assert ql_node.input[1:] == qgemm_node.input[1:3]  # As, Azp shared

    init = {i.name: i for i in quantized.graph.initializer}

    a_scale = numpy_helper.to_array(init[ql_node.input[1]])
    a_zp = numpy_helper.to_array(init[ql_node.input[2]])
    expected_a_scale, expected_a_zp = _expected_asymmetric_uint8_quant_params(*a_range)
    assert a_zp.dtype == np.uint8
    assert int(a_zp) == expected_a_zp
    assert expected_a_zp != 0, "test calibration range must exercise a nonzero Azp"
    np.testing.assert_allclose(float(a_scale), float(expected_a_scale), rtol=1e-6)

    y_scale = numpy_helper.to_array(init[dq_node.input[1]])
    y_zp = numpy_helper.to_array(init[dq_node.input[2]])
    expected_y_scale, expected_y_zp = _expected_asymmetric_uint8_quant_params(*y_range)
    assert y_zp.dtype == np.uint8
    assert int(y_zp) == expected_y_zp
    assert expected_y_zp != 0, "test calibration range must exercise a nonzero Yzp"
    np.testing.assert_allclose(float(y_scale), float(expected_y_scale), rtol=1e-6)

    bq = numpy_helper.to_array(init[qgemm_node.input[3]])
    bs = numpy_helper.to_array(init[qgemm_node.input[4]])
    bzp = numpy_helper.to_array(init[qgemm_node.input[5]])
    expected_bq, expected_bs = _quantize_weight_per_channel_in_place(
        weight, channel_axis=1
    )
    np.testing.assert_array_equal(bq, expected_bq)
    np.testing.assert_allclose(bs, expected_bs, rtol=1e-6)
    assert bzp.dtype == np.int8
    assert bzp.shape == (N,)
    np.testing.assert_array_equal(bzp, np.zeros(N, dtype=np.int8))


def test_qoperator_quantize_gemm_bias_scale_includes_alpha():
    # The genuinely new structural content vs. qoperator_quantize_conv's own
    # bias handling: bias_scale[n] = alpha * a_scale * b_scale[n] carries an
    # EXTRA alpha factor conv's own formula (x_scale * w_scale[c], no alpha)
    # has no analogue for. Isolated here with transA=0/transB=0 (so only the
    # bias-scale formula, not the transpose logic, is under test) but
    # alpha=2.0 != 1, so the extra factor is actually exercised.
    rng = np.random.default_rng(1)
    K, N, rows = 6, 4, 3
    alpha = 2.0
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.6
    bias = rng.standard_normal(N).astype(np.float32) * 1.5
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = Gemm<alpha = {alpha}>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )

    a_range = (-4.0, 6.0)
    y_range = (-10.0, 10.0)
    quantized = _quantize_qoperator_gemm(model, {"X": a_range, "Y": y_range})

    qgemm_node = next(n for n in quantized.graph.node if n.op_type == "QGemm")
    assert qgemm_node.input[6] != ""  # bias present, not the empty placeholder

    init = {i.name: i for i in quantized.graph.initializer}
    a_scale = float(numpy_helper.to_array(init[qgemm_node.input[1]]))
    bs = numpy_helper.to_array(init[qgemm_node.input[4]])
    bias_q = numpy_helper.to_array(init[qgemm_node.input[6]])

    assert bias_q.dtype == np.int32
    expected_bias_q, _bias_scale = _quantize_gemm_bias_int32(bias, alpha, a_scale, bs)
    np.testing.assert_array_equal(bias_q, expected_bias_q)

    # And: WITHOUT the extra alpha factor (i.e. qoperator_quantize_conv's own
    # formula, x_scale * w_scale[n], as if alpha were silently dropped) the
    # SAME bias values quantize to visibly DIFFERENT codes here (alpha=2.0
    # != 1) -- concretely confirming this pass's formula really does carry
    # the extra factor, not just reuse conv's unchanged.
    conv_style_bias_q, _ = _quantize_gemm_bias_int32(bias, 1.0, a_scale, bs)
    assert not np.array_equal(bias_q, conv_style_bias_q)


def test_qoperator_quantize_gemm_full_generality_matches_combined_bound():
    # The genuinely general case QLinearMatMul (and qoperator_quantize_matmul)
    # cannot handle at all: transA=1 AND transB=1 AND alpha != 1 (2.5), with a
    # bias. Confirms QGemm's own transA/transB/alpha attributes match the
    # original Gemm's, the weight/bias quantization matches independent numpy
    # re-implementations, and the real onnxruntime output (graph optimization
    # EXPLICITLY DISABLED, per this suite's established default -- see this
    # file's module docstring) stays within the proved combined
    # (alpha-scaled MAC bound + bias round-trip + output round-trip) bound.
    rng = np.random.default_rng(2)
    K, M, N = 16, 3, 8
    alpha = 2.5
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.6  # [N, K], transB=1
    bias = rng.standard_normal(N).astype(np.float32) * 0.5
    x = rng.standard_normal((K, M)).astype(np.float32) * 1.3  # [K, M], transA=1

    model = _model(
        f"""
        g (float[{K},{M}] X) => (float[{M},{N}] Y)
        {{
          Y = Gemm<transA = 1, transB = 1, alpha = {alpha}>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )

    a_prime = x.T  # [M, K] -- the logical, post-transA operand
    b_prime = weight.T  # [K, N] -- the logical, post-transB operand
    y_float = alpha * (a_prime @ b_prime) + bias[np.newaxis, :]

    a_min, a_max = float(x.min()), float(x.max())
    y_min, y_max = float(y_float.min()), float(y_float.max())
    quantized = _quantize_qoperator_gemm(
        model, {"X": (a_min, a_max), "Y": (y_min, y_max)}
    )

    qgemm_node = next(n for n in quantized.graph.node if n.op_type == "QGemm")
    trans_a = next(a.i for a in qgemm_node.attribute if a.name == "transA")
    trans_b = next(a.i for a in qgemm_node.attribute if a.name == "transB")
    alpha_attr = next(a.f for a in qgemm_node.attribute if a.name == "alpha")
    assert trans_a == 1
    assert trans_b == 1
    assert alpha_attr == pytest.approx(alpha)

    init = {i.name: i for i in quantized.graph.initializer}
    a_scale, _a_zp = _expected_asymmetric_uint8_quant_params(a_min, a_max)
    y_scale, _y_zp = _expected_asymmetric_uint8_quant_params(y_min, y_max)

    bq = numpy_helper.to_array(init[qgemm_node.input[3]])
    bs = numpy_helper.to_array(init[qgemm_node.input[4]])
    # transB != 0 -> channel_axis = 0 (weight's own [N, K] layout, no
    # transpose): confirms the "no forced transpose" claim from this file's
    # module docstring.
    expected_bq, expected_bs = _quantize_weight_per_channel_in_place(
        weight, channel_axis=0
    )
    np.testing.assert_array_equal(bq, expected_bq)
    np.testing.assert_allclose(bs, expected_bs, rtol=1e-6)

    bias_q = numpy_helper.to_array(init[qgemm_node.input[6]])
    expected_bias_q, bias_scale = _quantize_gemm_bias_int32(bias, alpha, a_scale, bs)
    np.testing.assert_array_equal(bias_q, expected_bias_q)

    # Graph optimization disabled: see this file's module docstring and
    # `tests/test_ort_matmul_nbits_workaround.py`'s docstring for the ORT
    # graph-optimization-fusion bug precedent this guards against.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        quantized.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    (y_quant,) = sess.run(None, {"X": x})

    eps_a = float(a_scale) / 2.0
    eps_b = bs / 2.0  # shape [N]
    eps_y = float(y_scale) / 2.0
    eps_bias = np.abs(bias_scale) / 2.0  # shape [N]

    bound1 = (
        eps_a * np.abs(b_prime).sum(axis=0)[np.newaxis, :]
        + eps_b[np.newaxis, :] * np.abs(a_prime).sum(axis=1)[:, np.newaxis]
        + K * eps_a * eps_b[np.newaxis, :]
    )
    combined_bound = abs(alpha) * bound1 + eps_bias[np.newaxis, :] + eps_y

    error = np.abs(y_float - y_quant)
    assert np.all(error <= combined_bound + 1e-6)

    # Also confirm the rewrite is meaningfully close (not merely "within a
    # loose worst-case bound") for these well-scaled inputs -- the actual
    # rigorous check is the proved combined worst-case bound above, already
    # asserted. A looser tolerance than the plain-MatMul template's own test:
    # quantizing the weight, the alpha-scaled activation AND the bias
    # compounds three independent INT8/INT32 rounding sources.
    np.testing.assert_allclose(y_quant, y_float, rtol=0.2, atol=0.3)


def test_qoperator_quantize_gemm_declines_with_only_activation_range():
    # patternMatchPredicate requires calibrated ranges for BOTH the
    # activation AND the node's own output -- the same requirement every
    # qoperator_quantize_* pass has. Supplying only X's range (no entry for
    # "Y") must leave the Gemm completely untouched.
    rng = np.random.default_rng(3)
    rows, K, N = 4, 5, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = Gemm(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = _quantize_qoperator_gemm(model, {"X": (-5.0, 10.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Gemm"]


def test_qoperator_quantize_gemm_declines_beta_not_one_with_bias():
    # This pass's own structural decline condition (item 4 in this file's
    # task): QGemm has no beta attribute of its own -- its bias-scale formula
    # implicitly assumes beta=1 -- so MatchGemmQuantizable declines a Gemm
    # whose bias is present AND whose beta != 1.0, even with both calibrated
    # ranges available.
    rng = np.random.default_rng(4)
    rows, K, N = 4, 5, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.7
    bias = rng.standard_normal(N).astype(np.float32)
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = Gemm<beta = 0.5>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )

    quantized = _quantize_qoperator_gemm(model, {"X": (-5.0, 10.0), "Y": (-3.0, 12.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Gemm"]


def test_qoperator_quantize_gemm_declines_non_length_n_bias():
    # The other half of this pass's own structural decline condition: only a
    # 1-D C of EXACTLY length N is handled; a 2-D C (here [rows, N], matching
    # Y's own shape but not the "common, unambiguous per-column bias" case)
    # declines, even with both calibrated ranges available.
    rng = np.random.default_rng(5)
    rows, K, N = 4, 5, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.7
    c2d = rng.standard_normal((rows, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = Gemm(X, W, C)
        }}
        """,
        [_f32(weight, "W"), _f32(c2d, "C")],
    )

    quantized = _quantize_qoperator_gemm(model, {"X": (-5.0, 10.0), "Y": (-3.0, 12.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Gemm"]
