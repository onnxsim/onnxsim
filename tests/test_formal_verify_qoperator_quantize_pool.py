"""Formal check for QOperatorQuantizePool (opt-in; onnxsim's own
``onnxsim/passes/qoperator_quantize_pool.h``): the pooling sibling of
``qoperator_quantize_softmax.h``'s ``QLinearSoftmax`` rewrite and the
structural twin of ``qoperator_quantize_matmul.h``'s single-calibrated-output
shape (``test_formal_verify_qoperator_quantize_matmul.py`` is this file's
template for the two-layer bound-composition structure below). It rewrites a
standalone ``Y = AveragePool(X, kernel_shape=k, ...)`` or
``Y = GlobalAveragePool(X)`` into ONNX Runtime's "com.microsoft" contrib ops::

    Xq = QuantizeLinear(X, Xs, Xzp)                                  -- CALIBRATED
    Yq = QLinear{AveragePool,GlobalAveragePool}(Xq, Xs, Xzp, Ys, Yzp,
                 <original attributes>, channels_last=0)             -- true int8
    Y  = DequantizeLinear(Yq, Ys, Yzp)                                -- CALIBRATED

Every attribute the original node has (``kernel_shape``, ``pads``,
``strides``, ``ceil_mode``, ``count_include_pad``, ``auto_pad``) is copied
onto the rewritten node unchanged via ``copyAttributes``; ``channels_last`` is
always explicitly set to 0 (onnxsim only ever produces NCHW graphs); a node
carrying a ``dilations`` attribute (standard ONNX opset 19+) is left
untouched entirely, since ``QLinearAveragePool``'s own ONNX Runtime kernel
rejects that attribute outright -- confirmed from the header's own comment.
Only a node with exactly 1 float32 input, and a calibrated range for BOTH
that input's name and the node's own output name, is matched.

Why this file is UNLIKE ``qoperator_quantize_softmax`` (a genuinely thin Z3
file, see that file's own docstring for the honest "no closed-form bound"
precedent): Softmax is a nonlinear, whole-axis reduction with no algebraic
error-propagation formula this repo could derive. Average pooling, by
contrast, IS linear -- for one output position it is literally
``AveragePool(X)[pos] = (1/K) * sum_{k in window} X[k]``, exactly
``quantized_mac_bound``'s own (``test_formal_verify_quantized_mac_bound.py``)
MAC-bound shape, with every "weight" pinned to the KNOWN, EXACT constant
``1/K`` -- a compile-time-known averaging divisor, never quantized,
never approximated, never anything a calibration range or a weight-scale
initializer touches at all.

The mirror image of ``weight_only_quantize_matmul``'s own special case
=======================================================================
``quantized_mac_bound``'s general two-operand lemma::

    eps_x * sum_k |W[k]| + eps_w * sum_k |X[k]| + K * eps_w * eps_x

collapses here with ``eps_w := 0`` -- the constant ``1/K`` has ZERO
quantization error, exactly like ``dynamic_quantize_ternary_matmul.py``'s own
``eps_w := 0`` collapse for a structurally-exact weight (that file's own
closest precedent for this "bake eps_w := 0 in from the start, no free ``ew``
variable at all" formulation -- see its docstring). The crucial difference
from that file: THERE, ``eps_w := 0`` holds because a *learned* ternary
weight *happens* to be exactly representable (an empirical, per-model
property ``TryQuantizeWeightTernaryKN`` must check and can fail); HERE,
``eps_w := 0`` holds *by construction*, unconditionally, for every model --
the averaging divisor ``1/K`` is a Python-computed/compile-time-known
constant baked into the very definition of "average", never a quantity that
could be approximately ternary or approximately anything else. There is
nothing to check; there is nothing that could fail.

This is also, in a precise sense, the MIRROR IMAGE of
``weight_only_quantize_matmul.py``'s own ``eps_x := 0`` special case
(``test_formal_verify_weight_only_quantize_matmul.py``): there, the
ACTIVATION ``X`` is exact (never quantized at all) and the WEIGHT ``W``
carries the only error budget (``eps_w := Ws / 2``), leaving a bound
``(Ws / 2) * sum_k |X[k]|``. Here it is the opposite operand that is exact:
the "weight" (the constant ``1/K`` averaging factor) is exact, and it is the
ACTIVATION ``X`` -- quantized via ``Xq`` -- that carries the only error
budget (``eps_x := Xs / 2``), leaving a bound ``eps_x * sum_k |W[k]|
= (Xs / 2) * sum_k |1/K| = (Xs / 2) * K * (1/K) = Xs / 2``. The two passes
each collapse the exact same general MAC bound along the OPPOSITE one of its
two free error terms.

The single most interesting conclusion this file's proof reaches
==================================================================
Because the "weight" here is not merely exact but SHRINKS as exactly
``1/K``, ``sum_k |W[k]|`` telescopes to EXACTLY 1 for every window size ``K``
-- so the collapsed bound ``eps_x * sum_k |W[k]|`` is EXACTLY ``eps_x =
Xs / 2``, independent of ``K``. This is genuinely different from every other
MAC-bound file in this suite: every one of them has a bound that grows WITH
``K`` (the ``K * eps_x * eps_w`` cross term, and the ``eps_w * sum|X|`` /
``eps_x * sum|W|`` terms, all scale with the number of taps, since an
ordinary weight matrix's entries do not shrink as the contraction dimension
grows). Here they don't, because the "weight" shrinks in exact lockstep with
the sum's length. Composing this with the OUTPUT's own round trip (the same
``Ys / 2`` second layer every other ``qoperator_quantize_*`` file in this
suite adds, via the triangle inequality) gives a combined bound of
``Xs / 2 + Ys / 2`` -- ENTIRELY INDEPENDENT of the pooling window size,
whether that window is a 2x2 ``AveragePool`` or a ``GlobalAveragePool`` over
a thousand spatial positions. Put plainly: average pooling is itself an
ERROR-REDUCING operation on the quantization noise it consumes -- independent
per-tap quantization errors get AVERAGED, not accumulated the way a MAC's
weighted sum accumulates them when the weights don't shrink to compensate.
The ``_pool_bound_formulas``/``test_..._bound_collapses_...`` pair below
proves this cancellation explicitly, at several concrete window sizes (not
hand-waved as "obviously true because K cancels algebraically"): each
concrete instantiation still builds the genuine ``K``-term sum before
dividing it back down, so Z3 -- not the test author -- confirms the
cancellation is exact at every one of those window sizes. Z3 has no native
way to quantify over an unbounded, symbolic ``K`` in this sum-of-``K``-terms
shape, so "independent of ``K``" is established the way this suite
establishes any claim about an unbounded family: by proving the SAME bound
at a growing sequence of concrete ``K`` values, exactly mirroring the sibling
differential test's real-onnxruntime confirmation of the same fact (window
sizes 9 and 4096) later in this file.

Differential tests build standalone ``AveragePool``/``GlobalAveragePool``
models via ``onnx.parser`` (per ``CLAUDE.md``) and run the real pass through
the nanobind-exposed
``onnxsim.onnxsim_cpp2py_export.quantize_qoperator_pool(model_bytes,
activation_ranges)`` -- the dedicated Pool entry point (see
``QuantizeQOperatorPool`` in ``onnxsim/onnxsim.h`` and
``onnxsim/quantize_entry.cpp``; isolates exactly
``OptimizeFixed(["qoperator_quantize_pool"])``), the same "no calibration
*data* need be fabricated, only calibrated ranges" reasoning as every other
file in this family. ``QLinearAveragePool``/``QLinearGlobalAveragePool`` are
confirmed to have working ONNX Runtime CPU kernels empirically before relying
on them anywhere else in this file, per this suite's own precedent
(``test_formal_verify_qoperator_quantize_softmax.py``'s docstring). Every
``InferenceSession`` built from a quantized graph explicitly disables graph
optimization (``ORT_DISABLE_ALL``) -- see
``tests/test_ort_matmul_nbits_workaround.py``'s docstring for this suite's
own precedent of a real ORT graph-optimization-fusion bug that changes a
QDQ/QOperator-shaped chain's actual executed computation -- and every
onnxruntime-execution differential test below uses reasonably sized tensors
(not tiny 2x2 spatial extents), since this suite has previously found and
fixed a real ONNX Runtime quantized-kernel edge case that manifested only in
CI, only for very small/irregular shapes, on a structurally similar
QOperator-format chain (``test_formal_verify_qoperator_quantize_matmul.py``'s
own docstring).

Structural/decline tests confirm: attributes (``kernel_shape``, ``pads``,
``strides``, ``ceil_mode``, ``count_include_pad``, ``auto_pad``) are carried
over unchanged via ``copyAttributes`` for both node kinds;
``channels_last=0`` is always set; a ``dilations`` attribute (opset 19+)
makes the pass decline outright; the "com.microsoft" opset import is added
exactly once, whether or not it was already present; and a missing
calibrated range for either the input or the output leaves the node
completely untouched.

The genuine crux differential test (the single most distinguishing empirical
confirmation this file can offer) builds a SMALL-window ``AveragePool``
(``kernel_shape=[3, 3]``, 9 taps) and a LARGE-window ``GlobalAveragePool``
over a much bigger spatial extent (4096 taps -- a >450x larger window) and
confirms the REAL quantized error, measured against ONNX Runtime's own float
execution, does NOT grow with window size -- consistent with the
K-independent ``Xs / 2 + Ys / 2`` bound proved above, and the opposite of
what any MAC-bound file in this suite would show for an ordinary
(non-shrinking) weight.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
import onnxsim.onnxsim_cpp2py_export as C
import pytest
from _formal_verify_common import producer, prove, z3
from onnx import parser

# Window sizes exercised by the "independent of K" proofs below -- unlike
# every other MAC-bound file in this suite (which fixes a single small
# concrete _K, e.g. 2, since the query's cost or the point being made does
# not depend on varying it), this file's whole point is that the bound is the
# SAME across window sizes, so it is checked at a growing sequence of them.
# The formulas below are purely linear (no product of two free error
# variables the way a two-quantized-operand MAC bound has), so there is no
# nonlinear-blowup risk in going considerably larger than this suite's usual
# _K = 2.
_WINDOW_SIZES = (1, 2, 3, 5, 8, 16)


def _abs(v):
    return z3.If(v >= 0, v, -v)


# ---------------------------------------------------------------------------
# 1. Z3 content
# ---------------------------------------------------------------------------


def _pool_bound_formulas(k):
    """Builds the Z3 vocabulary for the average-pooling bounded-error claim
    at a fixed but arbitrary window size ``k``: ``quantized_mac_bound``'s own
    general two-operand lemma, with the "weight" pinned to the EXACT constant
    ``1/k`` (``eps_w := 0`` baked in from the start -- no free ``ew`` variable
    anywhere, mirroring ``dynamic_quantize_ternary_matmul.py``'s own
    "more honest to state the special case directly" choice -- see module
    docstring) and only the activation carrying a rounding-error budget
    (``ex``, the direct-error-variable idiom every sibling file in this
    family uses).

    ``bound`` is deliberately built as the literal K-term sum
    ``(Xs / 2) * sum_i |1/k|`` rather than pre-simplified to ``Xs / 2`` in
    Python -- the whole point of this file's proof is that Z3 itself
    confirms that sum telescopes down to exactly ``Xs / 2``, not that the
    test author asserts it does.

    Returns ``(float_avg, dequant_avg, rounding_bounds, bound)``.
    """
    X = [z3.Real(f"X{i}") for i in range(k)]  # X[pos, i], true float activation tap
    ex = [z3.Real(f"ex{i}") for i in range(k)]  # X[i] - Xdq[i], per-tap direct error
    Xs = z3.Real("Xs")  # calibrated per-tensor activation scale
    w = z3.RealVal(1) / k  # the EXACT, compile-time-known averaging divisor 1/K --
    # never quantized, never approximated: this is eps_w := 0 by construction,
    # not a special case that could fail to hold for some model.

    Xdq = [X[i] - ex[i] for i in range(k)]

    rounding_bounds = z3.And(
        Xs > 0,
        *[_abs(ex[i]) <= Xs / 2 for i in range(k)],
    )

    float_avg = sum(w * X[i] for i in range(k))
    dequant_avg = sum(w * Xdq[i] for i in range(k))

    # quantized_mac_bound's general bound eps_x * sum|W| + eps_w * sum|X|
    # + K * eps_w * eps_x, with eps_w := 0 zeroing the last two terms,
    # leaving only eps_x * sum_i |w| -- the literal K-term sum, not yet
    # collapsed.
    bound = (Xs / 2) * sum(_abs(w) for _ in range(k))

    return float_avg, dequant_avg, rounding_bounds, bound


@pytest.mark.parametrize("k", _WINDOW_SIZES)
def test_qoperator_quantize_pool_average_error_is_bounded(k):
    # The genuine bounded-error claim at window size k: given the
    # activation's own per-tap rounding bound (|X[i] - Xdq[i]| <= Xs / 2) and
    # the averaging divisor held EXACT, the true float average and the
    # average of the dequantized taps cannot differ by more than `bound`
    # (which Z3 must independently confirm equals Xs / 2 -- see the next
    # test -- despite being handed here as the un-simplified K-term sum).
    float_avg, dequant_avg, rounding_bounds, bound = _pool_bound_formulas(k)
    error = float_avg - dequant_avg
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


@pytest.mark.parametrize("k", _WINDOW_SIZES)
def test_qoperator_quantize_pool_bound_collapses_to_half_xs_independent_of_window_size(
    k,
):
    # The single most interesting conclusion this file reaches (module
    # docstring): the K-term sum `(Xs / 2) * sum_i |1/k|` -- built literally,
    # not pre-simplified -- collapses to EXACTLY `Xs / 2` regardless of k,
    # because the per-tap `1/k` weighting means the k-fold sum of `Xs/2`-
    # scaled terms divides back down to exactly `Xs/2`. Unlike every MAC-
    # bound file in this suite (whose bound's cross term and per-operand sums
    # all grow WITH k), this bound is IDENTICAL at k=1 and at k=16 -- checked
    # here as its own explicit Z3 equality, not inferred from the error bound
    # above holding at each k separately.
    _float_avg, _dequant_avg, _rounding_bounds, bound = _pool_bound_formulas(k)
    Xs = z3.Real("Xs")
    prove(bound == Xs / 2)


def test_qoperator_quantize_pool_negative_control_requires_round_trip():
    # Standard negative control: dropping the per-tap round-trip hypothesis
    # entirely (keeping only Xs > 0) breaks the claim -- Z3 must find a real
    # counterexample, confirming the bound is not vacuously true regardless
    # of X's own quantization error. A single representative window size (4)
    # suffices; this is about hypothesis necessity, not window-size
    # independence (already established above).
    k = 4
    float_avg, dequant_avg, _rounding_bounds, bound = _pool_bound_formulas(k)
    Xs = z3.Real("Xs")
    error = float_avg - dequant_avg

    solver = z3.Solver()
    solver.add(Xs > 0)  # no per-tap |ex[i]| <= Xs / 2 bound at all
    solver.add(z3.Not(z3.And(error <= bound, -error <= bound)))
    assert solver.check() == z3.sat, (
        "the bound holds even without any per-tap rounding-error budget on "
        "the activation -- negative control is vacuous"
    )


def _combined_bound_formulas(k):
    """Layer 2, composed with layer 1 exactly as
    ``test_formal_verify_qoperator_quantize_matmul.py``'s own
    ``combined_bound`` tests do: the pass's real output ``Y`` is
    ``DequantizeLinear(Yq, Ys, Yzp)``, itself a further round trip on top of
    the raw dequantized average (``dequant_avg``, playing the role that
    file's ``Y_raw`` plays) into the fixed calibrated ``(Ys, Yzp)`` range,
    modeled the same direct-error-variable way (``eout``). Z3 -- not hand
    algebra -- confirms the triangle-inequality composition.

    Returns ``(float_avg, Y, hypotheses, combined_bound)``.
    """
    float_avg, dequant_avg, rounding_bounds, bound1 = _pool_bound_formulas(k)
    Ys = z3.Real("Ys")  # calibrated per-tensor OUTPUT scale
    eout = z3.Real("eout")  # dequant_avg[pos] - Y[pos], the output's own round trip
    Y = dequant_avg - eout

    output_round_trip = z3.And(Ys > 0, _abs(eout) <= Ys / 2)
    hypotheses = z3.And(rounding_bounds, output_round_trip)
    combined_bound = bound1 + Ys / 2

    return float_avg, Y, hypotheses, combined_bound


@pytest.mark.parametrize("k", _WINDOW_SIZES)
def test_qoperator_quantize_pool_combined_bound_holds(k):
    float_avg, Y, hypotheses, combined_bound = _combined_bound_formulas(k)
    error = float_avg - Y
    prove(
        z3.Implies(
            hypotheses, z3.And(error <= combined_bound, -error <= combined_bound)
        )
    )


@pytest.mark.parametrize("k", _WINDOW_SIZES)
def test_qoperator_quantize_pool_combined_bound_is_half_xs_plus_half_ys(k):
    # The composed conclusion, independent of window size: Xs/2 (layer 1,
    # collapsed) + Ys/2 (layer 2, the output's own ordinary round trip) --
    # NOT Xs/2 + Ys/2 scaled by, or added to, any K-dependent term.
    _float_avg, _Y, _hyp, combined_bound = _combined_bound_formulas(k)
    Xs, Ys = z3.Reals("Xs Ys")
    prove(combined_bound == Xs / 2 + Ys / 2)


def test_qoperator_quantize_pool_negative_control_requires_layer1_bound():
    # Sanity check the combined bound genuinely needs LAYER 1's rounding
    # hypothesis, not just the output round-trip one: with the output's own
    # round-trip bound assumed but no budget at all on the activation's
    # per-tap error, the combined claim is not a theorem.
    k = 4
    X = [z3.Real(f"X{i}") for i in range(k)]
    ex = [z3.Real(f"ex{i}") for i in range(k)]
    Xs = z3.Real("Xs")
    Ys = z3.Real("Ys")
    eout = z3.Real("eout")
    w = z3.RealVal(1) / k

    Xdq = [X[i] - ex[i] for i in range(k)]
    float_avg = sum(w * X[i] for i in range(k))
    dequant_avg = sum(w * Xdq[i] for i in range(k))
    Y = dequant_avg - eout

    bound1 = (Xs / 2) * sum(_abs(w) for _ in range(k))
    combined_bound = bound1 + Ys / 2
    error = float_avg - Y

    solver = z3.Solver()
    solver.add(Xs > 0, Ys > 0, _abs(eout) <= Ys / 2)  # output bound only
    solver.add(z3.Not(z3.And(error <= combined_bound, -error <= combined_bound)))
    assert solver.check() == z3.sat, (
        "the combined bound holds even without layer 1's rounding hypothesis "
        "-- negative control is vacuous"
    )


def test_qoperator_quantize_pool_negative_control_requires_output_bound():
    # Symmetric negative control: layer 1's rounding hypothesis assumed but
    # no budget at all on the output's own round-trip error (eout free).
    k = 4
    float_avg, dequant_avg, rounding_bounds, bound1 = _pool_bound_formulas(k)
    Ys = z3.Real("Ys")
    eout = z3.Real("eout")
    Y = dequant_avg - eout
    combined_bound = bound1 + Ys / 2
    error = float_avg - Y

    solver = z3.Solver()
    solver.add(rounding_bounds, Ys > 0)  # no bound on eout at all
    solver.add(z3.Not(z3.And(error <= combined_bound, -error <= combined_bound)))
    assert solver.check() == z3.sat, (
        "the combined bound holds even without the output's own round-trip "
        "bound -- negative control is vacuous"
    )


# ---------------------------------------------------------------------------
# 2. Differential / structural content
# ---------------------------------------------------------------------------


def _model(body, opset=13, ir_version=10, extra_imports=""):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}{extra_imports}]
        >
        {body}
        """
    )


def _quantize_qoperator_pool(model, activation_ranges):
    """Invokes the real compiled pass directly via the nanobind-exposed
    ``quantize_qoperator_pool(model_bytes, activation_ranges)`` -- the
    dedicated Pool entry point (module docstring). Runs ``OptimizeFixed``
    with exactly ``["qoperator_quantize_pool"]``.
    """
    out = onnx.ModelProto()
    out.ParseFromString(
        C.quantize_qoperator_pool(model.SerializeToString(), activation_ranges)
    )
    return out


def _expected_asymmetric_uint8_quant_params(min_val, max_val):
    """Independent re-implementation of ``ComputeAsymmetricUint8QuantParams``
    (static_quantize_matmul.h), in float32 to match the pass's own arithmetic
    precision -- identical to every sibling ``qoperator_quantize_*`` file's
    own helper of the same name, since this pass reads the exact same
    global/function for both its activation AND its output.
    """
    lo = min(np.float32(0.0), np.float32(min_val))
    hi = max(np.float32(0.0), np.float32(max_val))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zero_point = int(np.clip(np.round(-lo / scale), 0, 255))
    return np.float32(scale), zero_point


def _disable_opt_session(model_bytes):
    # Disable graph optimization explicitly: by default onnxruntime can
    # fuse/transform a QDQ/QOperator-shaped chain like this pass produces
    # into a different, hardware-specific code path than the literal node
    # chain this file's proofs reason about -- see
    # tests/test_ort_matmul_nbits_workaround.py's docstring for this suite's
    # existing precedent of a real ORT graph-optimization fusion bug of
    # exactly this shape.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model_bytes, sess_options=so, providers=["CPUExecutionProvider"]
    )


def test_qoperator_quantize_pool_qlinear_kernels_exist():
    # Confirmed empirically before relying on either op anywhere else in this
    # file (this suite's own precedent for a contrib op used by a
    # differential test): minimal standalone QuantizeLinear -> QLinear{
    # AveragePool,GlobalAveragePool} -> DequantizeLinear models actually run
    # on onnxruntime's CPU provider.
    avg_model = _model(
        """
        g (float[1,2,6,6] X) => (float[1,2,2,2] Y)
        <float Xs = {0.05}, uint8 Xzp = {20}, float Ys = {0.02}, uint8 Yzp = {10}>
        {
          Xq = QuantizeLinear(X, Xs, Xzp)
          Yq = com.microsoft.QLinearAveragePool<
            kernel_shape=[3, 3], strides=[3, 3], channels_last=0
          >(Xq, Xs, Xzp, Ys, Yzp)
          Y = DequantizeLinear(Yq, Ys, Yzp)
        }
        """,
        extra_imports=', "com.microsoft": 1',
    )
    sess = _disable_opt_session(avg_model.SerializeToString())
    x = np.linspace(-1.0, 1.0, 72, dtype=np.float32).reshape(1, 2, 6, 6)
    (y,) = sess.run(None, {"X": x})
    assert y.shape == (1, 2, 2, 2)
    assert np.all(np.isfinite(y))

    global_model = _model(
        """
        g (float[1,2,6,6] X) => (float[1,2,1,1] Y)
        <float Xs = {0.05}, uint8 Xzp = {20}, float Ys = {0.02}, uint8 Yzp = {10}>
        {
          Xq = QuantizeLinear(X, Xs, Xzp)
          Yq = com.microsoft.QLinearGlobalAveragePool<channels_last=0>(
            Xq, Xs, Xzp, Ys, Yzp
          )
          Y = DequantizeLinear(Yq, Ys, Yzp)
        }
        """,
        extra_imports=', "com.microsoft": 1',
    )
    sess2 = _disable_opt_session(global_model.SerializeToString())
    (y2,) = sess2.run(None, {"X": x})
    assert y2.shape == (1, 2, 1, 1)
    assert np.all(np.isfinite(y2))


def test_qoperator_quantize_pool_average_pool_pass_fires_and_matches_scheme():
    # Build a plain float AveragePool with a non-default attribute set and
    # run the real pass with calibration ranges for BOTH X and the node's own
    # output, each straddling 0 so both Xzp/Yzp come out genuinely nonzero.
    model = _model(
        """
        g (float[1,3,8,8] X) => (float[1,3,5,5] Y)
        {
          Y = AveragePool<
            kernel_shape = [3, 3], strides = [2, 2], pads = [1, 1, 1, 1],
            ceil_mode = 1, count_include_pad = 1
          >(X)
        }
        """
    )

    x_range = (-5.0, 10.0)
    y_range = (-3.0, 12.0)
    quantized = _quantize_qoperator_pool(model, {"X": x_range, "Y": y_range})

    # No float AveragePool left anywhere.
    op_types = {n.op_type for n in quantized.graph.node}
    assert "AveragePool" not in op_types

    # Walk the chain backward from the real graph output:
    # DequantizeLinear(Yq) <- QLinearAveragePool(Xq, ...) <- QuantizeLinear(X).
    dq_node = producer(quantized, "Y")
    assert dq_node.op_type == "DequantizeLinear"
    qlap_node = producer(quantized, dq_node.input[0])
    assert qlap_node.op_type == "QLinearAveragePool"
    assert qlap_node.domain == "com.microsoft"
    assert dq_node.input[1:] == list(qlap_node.input[3:5])  # Ys, Yzp shared

    ql_node = producer(quantized, qlap_node.input[0])
    assert ql_node.op_type == "QuantizeLinear"
    assert ql_node.input[0] == "X"
    assert ql_node.input[1:] == list(qlap_node.input[1:3])  # Xs, Xzp shared

    attrs = {a.name: a for a in qlap_node.attribute}
    assert list(attrs["kernel_shape"].ints) == [3, 3]
    assert list(attrs["strides"].ints) == [2, 2]
    assert list(attrs["pads"].ints) == [1, 1, 1, 1]
    assert attrs["ceil_mode"].i == 1
    assert attrs["count_include_pad"].i == 1
    assert attrs["channels_last"].i == 0  # always 0 (NCHW), never carried over

    # com.microsoft opset import added, version 1.
    ms_imports = [o for o in quantized.opset_import if o.domain == "com.microsoft"]
    assert len(ms_imports) == 1
    assert ms_imports[0].version == 1

    init = {i.name: i for i in quantized.graph.initializer}

    x_scale = numpy_helper.to_array(init[ql_node.input[1]])
    x_zp = numpy_helper.to_array(init[ql_node.input[2]])
    expected_x_scale, expected_x_zp = _expected_asymmetric_uint8_quant_params(*x_range)
    assert x_zp.dtype == np.uint8
    assert int(x_zp) == expected_x_zp
    assert expected_x_zp != 0, "test calibration range must exercise a nonzero Xzp"
    np.testing.assert_allclose(float(x_scale), float(expected_x_scale), rtol=1e-6)

    y_scale = numpy_helper.to_array(init[dq_node.input[1]])
    y_zp = numpy_helper.to_array(init[dq_node.input[2]])
    expected_y_scale, expected_y_zp = _expected_asymmetric_uint8_quant_params(*y_range)
    assert y_zp.dtype == np.uint8
    assert int(y_zp) == expected_y_zp
    assert expected_y_zp != 0, "test calibration range must exercise a nonzero Yzp"
    np.testing.assert_allclose(float(y_scale), float(expected_y_scale), rtol=1e-6)


def test_qoperator_quantize_pool_global_average_pool_pass_fires_and_matches_scheme():
    # GlobalAveragePool: no attributes of its own, so copyAttributes is a
    # no-op, but channels_last=0 must still be set explicitly.
    model = _model(
        """
        g (float[1,4,6,6] X) => (float[1,4,1,1] Y)
        {
          Y = GlobalAveragePool(X)
        }
        """
    )

    x_range = (-4.0, 6.0)
    y_range = (-1.0, 2.0)
    quantized = _quantize_qoperator_pool(model, {"X": x_range, "Y": y_range})

    op_types = {n.op_type for n in quantized.graph.node}
    assert "GlobalAveragePool" not in op_types
    assert op_types == {
        "QuantizeLinear",
        "QLinearGlobalAveragePool",
        "DequantizeLinear",
    }

    dq_node = producer(quantized, "Y")
    assert dq_node.op_type == "DequantizeLinear"
    qlgap_node = producer(quantized, dq_node.input[0])
    assert qlgap_node.op_type == "QLinearGlobalAveragePool"
    assert qlgap_node.domain == "com.microsoft"

    attrs = {a.name: a for a in qlgap_node.attribute}
    assert attrs["channels_last"].i == 0

    ms_imports = [o for o in quantized.opset_import if o.domain == "com.microsoft"]
    assert len(ms_imports) == 1
    assert ms_imports[0].version == 1


def test_qoperator_quantize_pool_auto_pad_attribute_carried_over():
    # auto_pad (mutually exclusive with an explicit pads attribute in
    # standard ONNX AveragePool) must also be threaded through unchanged.
    model = _model(
        """
        g (float[1,2,7,7] X) => (float[1,2,7,7] Y)
        {
          Y = AveragePool<kernel_shape = [3, 3], auto_pad = "SAME_UPPER">(X)
        }
        """
    )
    quantized = _quantize_qoperator_pool(model, {"X": (-2.0, 2.0), "Y": (-1.0, 1.0)})
    qlap_node = producer(quantized, producer(quantized, "Y").input[0])
    assert qlap_node.op_type == "QLinearAveragePool"
    attrs = {a.name: a for a in qlap_node.attribute}
    assert attrs["auto_pad"].s == b"SAME_UPPER"


def test_qoperator_quantize_pool_com_microsoft_import_not_duplicated():
    # If "com.microsoft" is already present in opset_import (e.g. a model
    # that already went through this pass, or already used another contrib
    # op), the pass must not add a second entry.
    model = _model(
        """
        g (float[1,2,4,4] X) => (float[1,2,2,2] Y)
        {
          Y = AveragePool<kernel_shape = [2, 2], strides = [2, 2]>(X)
        }
        """,
        extra_imports=', "com.microsoft": 1',
    )
    quantized = _quantize_qoperator_pool(model, {"X": (-3.0, 3.0), "Y": (-3.0, 3.0)})
    ms_imports = [o for o in quantized.opset_import if o.domain == "com.microsoft"]
    assert len(ms_imports) == 1
    assert ms_imports[0].version == 1
    # And the rewrite still actually fired.
    assert producer(quantized, "Y").op_type == "DequantizeLinear"


def test_qoperator_quantize_pool_declines_with_dilations_attribute():
    # Standard ONNX AveragePool gained an optional `dilations` attribute in
    # opset 19; QLinearAveragePool's own ONNX Runtime kernel does not accept
    # it ("Unrecognized attribute: dilations for operator
    # QLinearAveragePool" is a real, observed error per the header's own
    # comment) -- so a node carrying it must be left completely untouched,
    # even with both calibrated ranges present.
    model = _model(
        """
        g (float[1,2,4,4] X) => (float[1,2,3,3] Y)
        {
          Y = AveragePool<kernel_shape = [2, 2], dilations = [1, 1]>(X)
        }
        """,
        opset=19,
    )
    quantized = _quantize_qoperator_pool(model, {"X": (-3.0, 3.0), "Y": (-3.0, 3.0)})
    assert [n.op_type for n in quantized.graph.node] == ["AveragePool"]


def test_qoperator_quantize_pool_declines_with_only_activation_range():
    # patternMatchPredicate requires calibrated ranges for BOTH the
    # activation AND the node's own output. Supplying only X's range must
    # leave the AveragePool completely untouched.
    model = _model(
        """
        g (float[1,2,4,4] X) => (float[1,2,2,2] Y)
        {
          Y = AveragePool<kernel_shape = [2, 2], strides = [2, 2]>(X)
        }
        """
    )
    quantized = _quantize_qoperator_pool(model, {"X": (-3.0, 3.0)})
    assert [n.op_type for n in quantized.graph.node] == ["AveragePool"]


def test_qoperator_quantize_pool_declines_with_only_output_range():
    # Symmetric: only Y's range supplied, no entry for X.
    model = _model(
        """
        g (float[1,2,4,4] X) => (float[1,2,2,2] Y)
        {
          Y = AveragePool<kernel_shape = [2, 2], strides = [2, 2]>(X)
        }
        """
    )
    quantized = _quantize_qoperator_pool(model, {"Y": (-3.0, 3.0)})
    assert [n.op_type for n in quantized.graph.node] == ["AveragePool"]


def test_qoperator_quantize_pool_declines_non_float_input():
    # Only a float32 input is matched.
    model = _model(
        """
        g (float16[1,2,4,4] X) => (float16[1,2,1,1] Y)
        {
          Y = GlobalAveragePool(X)
        }
        """
    )
    quantized = _quantize_qoperator_pool(model, {"X": (-3.0, 3.0), "Y": (-3.0, 3.0)})
    assert [n.op_type for n in quantized.graph.node] == ["GlobalAveragePool"]


# ---------------------------------------------------------------------------
# 3. The genuine crux: real quantized error does not grow with window size
# ---------------------------------------------------------------------------


def _real_pool_quantized_error(body, shape, rng):
    """Runs the real pass on a standalone pooling model, executes both the
    original float graph and the quantized graph through onnxruntime (graph
    optimization disabled -- see module docstring), and returns
    ``(max_abs_error, proved_bound)`` where ``proved_bound`` is this file's
    own ``Xs / 2 + Ys / 2`` combined bound computed from the SAME calibrated
    ranges the pass itself was given (the actual observed (min, max) of X and
    of the true float output, so neither round-trip lemma's no-saturation
    side condition is violated).
    """
    model = _model(body)
    x = rng.uniform(-4.0, 4.0, size=shape).astype(np.float32)

    float_sess = _disable_opt_session(model.SerializeToString())
    (y_float,) = float_sess.run(None, {"X": x})

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y_float.min()), float(y_float.max())
    quantized = _quantize_qoperator_pool(
        model, {"X": (x_min, x_max), "Y": (y_min, y_max)}
    )

    quant_sess = _disable_opt_session(quantized.SerializeToString())
    (y_quant,) = quant_sess.run(None, {"X": x})

    x_scale, _x_zp = _expected_asymmetric_uint8_quant_params(x_min, x_max)
    y_scale, _y_zp = _expected_asymmetric_uint8_quant_params(y_min, y_max)
    bound = float(x_scale) / 2.0 + float(y_scale) / 2.0

    error = np.abs(y_float.astype(np.float64) - y_quant.astype(np.float64))
    assert np.all(error <= bound + 1e-6), (
        f"real quantized pooling error exceeds the proved Xs/2 + Ys/2 bound "
        f"(max error {error.max()}, bound {bound})"
    )
    return float(error.max()), bound


def test_qoperator_quantize_pool_real_quantized_error_does_not_grow_with_window_size():
    # The single most distinguishing empirical confirmation this file can
    # offer (module docstring): a SMALL window (AveragePool, kernel_shape=
    # [3, 3], K=9 taps) and a LARGE window (GlobalAveragePool over a much
    # bigger spatial extent, K=4096 taps -- a >450x larger window) are run
    # through the real pass and real onnxruntime execution. If the proved
    # bound's K-independence were an artifact of the Z3 formulation rather
    # than a real property of this pass's actual computation, the large-
    # window case's real error would be expected to shrink drastically
    # (pure random-error averaging, no bound at all) or, for a naively wrong
    # implementation with a per-tap error that doesn't cancel, to grow --
    # either way, NOT track the same small, K-independent Xs/2 + Ys/2 bound
    # this file's proof derives for both cases alike.
    rng = np.random.default_rng(0)

    small_body = """
    g (float[1,4,15,15] X) => (float[1,4,5,5] Y)
    {
      Y = AveragePool<kernel_shape = [3, 3], strides = [3, 3]>(X)
    }
    """
    max_err_small, bound_small = _real_pool_quantized_error(
        small_body, (1, 4, 15, 15), rng
    )

    large_body = """
    g (float[1,4,64,64] X) => (float[1,4,1,1] Y)
    {
      Y = GlobalAveragePool(X)
    }
    """
    max_err_large, bound_large = _real_pool_quantized_error(
        large_body, (1, 4, 64, 64), rng
    )

    # Both individually respect their own K-independent bound (asserted
    # inside the helper already); the genuinely distinguishing claim is that
    # the LARGE window's real error is not meaningfully bigger than the SMALL
    # window's, despite pooling over 455x as many taps.
    assert max_err_large < 5 * max_err_small + 1e-6, (
        f"quantized pooling error grew with window size (K=9 max error "
        f"{max_err_small}, bound {bound_small}; K=4096 max error "
        f"{max_err_large}, bound {bound_large}) -- inconsistent with the "
        f"proved K-independent Xs/2 + Ys/2 bound"
    )
