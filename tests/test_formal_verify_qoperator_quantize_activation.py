"""Formal check for QOperatorQuantizeActivation (opt-in; onnxsim's own
``onnxsim/passes/qoperator_quantize_activation.h``): the unary-activation
sibling of ``qoperator_quantize_elementwise.h``'s QLinearAdd/QLinearMul
rewrite. It rewrites a standalone ``Y = Sigmoid(X)`` or
``Y = LeakyRelu(X, alpha=a)`` into ONNX Runtime's "com.microsoft" contrib ops
``QLinearSigmoid``/``QLinearLeakyRelu``::

    Xq = QuantizeLinear(X, Xs, Xzp)                              -- CALIBRATED
    Yq = QLinear{Sigmoid,LeakyRelu}(Xq, Xs, Xzp, Ys, Yzp)        -- true int8
    Y  = DequantizeLinear(Yq, Ys, Yzp)                            -- CALIBRATED

Only a Sigmoid or LeakyRelu with exactly 1 input, float32, is matched; a node
is only rewritten when both its input's name and its own output's name have
a calibrated range -- confirmed from ``patternMatchPredicate`` in the header.
``LeakyRelu``'s ``alpha`` attribute (default 0.01, per the ONNX spec and this
pass's own ``GetValueFromAttrWithDefault(n, kalpha, 0.01)``) is carried over
onto ``QLinearLeakyRelu`` unchanged.

Why this file's content is shaped DIFFERENTLY from
``test_formal_verify_qoperator_quantize_softmax.py``
====================================================================================
That sibling file honestly documents that ``Softmax`` -- a nonlinear,
whole-axis reduction -- has NO closed-form bound this repo could derive, so
its real content is a thin structural lemma plus heavy opset-semantics
differential weight. Sigmoid and LeakyRelu are different: both are
**pointwise** nonlinearities with a genuine, provable Lipschitz constant, so
THIS file has real bound content to derive and prove, in the same
"quantization-error propagates through an operator's own Lipschitz constant"
spirit as ``test_formal_verify_quantized_mac_bound.py``'s dot-product bound
-- just for a single scalar nonlinearity instead of a reduction.

Sigmoid's own Lipschitz constant: 1/4
--------------------------------------
``sigmoid(x) = 1/(1+e^-x)``; its derivative is
``sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))``, i.e. ``s * (1 - s)`` for
``s = sigmoid(x) in [0, 1]``. Z3 cannot reason about ``sigmoid``/``exp``
symbolically, but the elementary algebraic fact
``s * (1 - s) = 0.25 - (s - 0.5)^2 <= 0.25`` (maximized at ``s = 0.5``, i.e.
``x = 0``) is exactly the kind of ground polynomial inequality Z3 verifies
directly -- proved below as its own standalone lemma
(``test_qoperator_quantize_activation_sigmoid_derivative_bound_lemma``)
before being used as a datum anywhere else. (Algebraically the bound in fact
holds for ANY real ``s``, not just ``[0, 1]`` -- the ``[0, 1]`` hypothesis is
kept anyway since ``s`` here stands for ``sigmoid(x)``, which is only ever
known to lie in that range, and matching the derivative's real domain of
interest makes the lemma's role in what follows clearer.)

By the mean value theorem, a derivative bounded by ``1/4`` everywhere means
``sigmoid`` itself is ``1/4``-Lipschitz: ``|sigmoid(x1) - sigmoid(x2)| <=
0.25 * |x1 - x2|`` for ALL real ``x1``, ``x2``. Z3 verifies only the
ALGEBRAIC step above (``s*(1-s) <= 1/4``); the mean-value-theorem step
connecting a pointwise derivative bound to a global Lipschitz inequality is
real analysis, taken here as a stated, standard mathematical fact (an
explicit hypothesis/axiom in the Z3 queries below that use it), not
something Z3 derives from first principles. This is the same
axiom-then-Z3-composition split ``test_formal_verify_dynamic_quantize_
matmul.py``'s module docstring documents for its own accumulator identity
plus rounding-bound composition -- and, like that file, the composed claim
below uses the DIRECT-ERROR-VARIABLE idiom (a free real standing for an
error quantity, bounded directly by a hypothesis, rather than re-derived
from lower-level quantized-code arithmetic inside the same query) throughout.

Given that axiom, this pass's full two-layer soundness claim is::

    |Sigmoid(X) - Y| <= 0.25 * (Xs / 2) + Ys / 2

-- X's own QuantizeLinear/DequantizeLinear round-trip error (``Xs / 2``),
shrunk by Sigmoid's own Lipschitz constant as it propagates through the
nonlinearity, PLUS the pass's own output round trip (``Ys / 2``, exactly the
same shape every other file in this family re-derives for its own output).
Proved below (``test_qoperator_quantize_activation_sigmoid_composed_bound_
holds``) via three direct-error variables -- ``ex`` (``X``'s round-trip
error, bounded by ``Xs / 2``), ``esig`` (``sigmoid(X) - sigmoid(Xdq)``,
bounded by ``0.25 * |ex|`` -- the Lipschitz axiom instantiated at the pair
``(X, Xdq)``, whose separation is exactly ``ex``), ``eout`` (the output's own
round-trip error, bounded by ``Ys / 2``) -- composed via the triangle
inequality. A negative control confirms the composition genuinely needs the
Lipschitz link between ``ex`` and ``esig``: with ``esig`` left otherwise
unconstrained (only ``eout``/``ex`` bounded), the combined claim is no
longer a theorem.

LeakyRelu's own Lipschitz constant: ``max(1, |alpha|)``
--------------------------------------------------------
``LeakyRelu(x) = x`` if ``x >= 0`` else ``alpha * x`` -- piecewise LINEAR, so
unlike Sigmoid, Z3 verifies its Lipschitz claim end to end with no external
axiom: modeled via ``z3.If`` for the piecewise branch, with Z3's own
case-split covering all four sign combinations of a free ``x1``/``x2``
pair, for a free real ``alpha`` (``test_qoperator_quantize_activation_
leaky_relu_lipschitz_case_split``). The same two-layer composed bound then
holds with ``max(1, |alpha|)`` in place of ``0.25``
(``test_qoperator_quantize_activation_leaky_relu_composed_bound_holds``),
with the same negative control for the Lipschitz link.

Two more negative controls confirm the ``max(1, |alpha|)`` constant itself is
load-bearing, not just "any constant will do":
``test_qoperator_quantize_activation_leaky_relu_default_alpha_lipschitz_
is_1_not_alpha`` confirms that for this pass's own DEFAULT ``alpha=0.01``,
the Lipschitz constant is exactly 1 (dominated by the ``x >= 0`` branch's
unit slope), NOT ``0.01`` -- a naive "the Lipschitz constant is ``alpha``"
guess would be far too tight for this pass's own default, most common case.
``test_qoperator_quantize_activation_leaky_relu_needs_max_1_alpha_not_
just_1`` gives a concrete counterexample (``alpha=5``, ``x1=1``, ``x2=-1``)
where a naive constant-1 bound (ignoring ``alpha`` entirely) fails outright,
confirming the ``|alpha|`` branch of ``max(1, |alpha|)`` is equally
load-bearing whenever ``|alpha| > 1``.

Differential tests invoke the real compiled pass directly via the
nanobind-exposed ``onnxsim.onnxsim_cpp2py_export.quantize_qoperator_
activation(model_bytes, activation_ranges)`` entry point (see
``QuantizeQOperatorActivation`` in ``onnxsim/onnxsim.h`` and
``onnxsim/quantize_entry.cpp``, exposed in ``onnxsim/cpp2py_export.cc``) --
the same "no calibration *data* need be fabricated, only calibrated ranges"
reasoning as every other file in this family, isolating exactly
``qoperator_quantize_activation``. ``QLinearSigmoid``'s and
``QLinearLeakyRelu``'s own ONNX Runtime CPU kernels are confirmed to exist
empirically (minimal standalone models) before anything else in this file
relies on them, per this suite's established precedent
(``test_formal_verify_qoperator_quantize_softmax.py``'s docstring). Every
``InferenceSession`` explicitly disables graph optimization
(``ORT_DISABLE_ALL``) -- see ``tests/test_ort_matmul_nbits_workaround.py``'s
docstring for this suite's precedent of a real ORT graph-optimization-fusion
bug changing which node chain actually executes -- and every ORT-execution
differential test below uses a reasonably sized (64-element) tensor, not a
tiny 2x2 one, per this suite's own recent finding of an ONNX Runtime
quantized-kernel edge case that manifested only in CI, only for very
small/irregular tensor shapes (see
``test_formal_verify_qoperator_quantize_matmul.py``'s own docstring for that
precedent). The actual quantized output is confirmed to stay within the
proved Lipschitz-scaled combined bound against the TRUE float
Sigmoid/LeakyRelu (computed via numpy, not re-derived from onnxruntime's own
plain-float kernel, since a real closed-form bound -- unlike Softmax's -- is
exactly what this file is able to derive and check against).

Structural/decline tests confirm: the "com.microsoft" opset import is added
once, not duplicated; a missing calibrated range for EITHER the input or the
node's own output leaves the node untouched; a non-float32 input is
declined; and, with a Sigmoid and a LeakyRelu both present but only one
calibrated, the pass rewrites exactly the calibrated one, confirming the
per-node granularity of ``patternMatchPredicate``.
"""

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim.onnxsim_cpp2py_export as C
from _formal_verify_common import prove, z3
from onnx import parser


def _abs(v):
    return z3.If(v >= 0, v, -v)


# ---------------------------------------------------------------------------
# 1. Z3 content -- Sigmoid
# ---------------------------------------------------------------------------


def test_qoperator_quantize_activation_sigmoid_derivative_bound_lemma():
    # sigmoid'(x) = s * (1 - s) for s = sigmoid(x) in [0, 1]. This is plain
    # algebra Z3 verifies directly: s*(1-s) = 0.25 - (s-0.5)^2 <= 0.25,
    # maximized at s = 0.5 (i.e. x = 0, where sigmoid(0) = 0.5). This is the
    # ONLY step in Sigmoid's Lipschitz argument Z3 actually proves from first
    # principles -- see module docstring for why the mean-value-theorem step
    # connecting this derivative bound to a global Lipschitz inequality is
    # taken as a stated axiom instead, in the tests that use it below.
    s = z3.Real("s")
    prove(z3.Implies(z3.And(s >= 0, s <= 1), s * (1 - s) <= z3.RealVal(1) / 4))


def test_qoperator_quantize_activation_sigmoid_x_round_trip_is_sound():
    # This pass's own activation-side round trip, re-derived in this file's
    # own vocabulary per this suite's per-file convention (same shape as
    # test_formal_verify_qoperator_quantize_softmax.py's own X round-trip
    # lemma): QuantizeLinear(X, Xs, Xzp) then DequantizeLinear back recovers
    # X to within Xs/2, given no saturation and a free (arbitrary) zero
    # point.
    X, Xs, Xzp = z3.Reals("X Xs Xzp")
    n = z3.Int("n")  # round(X / Xs): some integer within 0.5 of X / Xs
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(Xs > 0, n - X / Xs <= half, X / Xs - n <= half)
    xq = n + Xzp
    xdq = (xq - Xzp) * Xs

    error = X - xdq
    prove(z3.Implies(hypotheses, z3.And(error <= Xs / 2, -error <= Xs / 2)))


def test_qoperator_quantize_activation_y_round_trip_is_sound():
    # The symmetric claim for this pass's own output round trip: Y (after
    # this pass fires) is DequantizeLinear(QLinear{Sigmoid,LeakyRelu}(...),
    # Ys, Yzp) -- the QLinear op itself computes directly in int8 with no
    # float intermediate, so whatever int8 code it produces is, from the
    # perspective of this round-trip lemma alone, dequantized the same
    # QuantizeLinear/DequantizeLinear-shaped way X's own round trip is above.
    # Shared verbatim between the Sigmoid and LeakyRelu composed-bound proofs
    # below, since both rewrite into the same output-quantization shape.
    Yraw, Ys, Yzp = z3.Reals("Yraw Ys Yzp")
    n = z3.Int("n")
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(Ys > 0, n - Yraw / Ys <= half, Yraw / Ys - n <= half)
    yq = n + Yzp
    ydq = (yq - Yzp) * Ys

    error = Yraw - ydq
    prove(z3.Implies(hypotheses, z3.And(error <= Ys / 2, -error <= Ys / 2)))


def test_qoperator_quantize_activation_sigmoid_composed_bound_holds():
    # The full two-layer claim: |Sigmoid(X) - Y| <= 0.25*(Xs/2) + Ys/2.
    #
    # Direct-error-variable idiom throughout (see module docstring):
    #   ex    := X - Xdq                       (X's own round-trip error)
    #   esig  := sigmoid(X) - sigmoid(Xdq)      (propagated through Sigmoid)
    #   eout  := sigmoid(Xdq) - Y               (the output's own round trip)
    # so Sigmoid(X) - Y = esig + eout exactly (telescoping), and the claim
    # reduces to bounding |esig + eout| by triangle inequality, GIVEN:
    #   |ex|   <= Xs / 2      -- the X round-trip lemma above
    #   |esig| <= 0.25 * |ex| -- the mean-value-theorem axiom (module
    #                            docstring) instantiated at the pair
    #                            (X, Xdq), whose separation is exactly ex
    #   |eout| <= Ys / 2      -- the Y round-trip lemma above
    # Z3 then only has to chain "0.25 * |ex| <= 0.25 * (Xs/2)" (multiplying
    # a hypothesis by a positive literal constant) with the triangle
    # inequality -- ordinary linear arithmetic, not a search over sigmoid's
    # own (unavailable) symbolic definition.
    ex, esig, eout, Xs, Ys = z3.Reals("ex esig eout Xs Ys")
    hypotheses = z3.And(
        Xs > 0,
        Ys > 0,
        _abs(ex) <= Xs / 2,
        _abs(esig) <= z3.RealVal(1) / 4 * _abs(ex),
        _abs(eout) <= Ys / 2,
    )
    bound = z3.RealVal(1) / 4 * (Xs / 2) + Ys / 2
    prove(z3.Implies(hypotheses, _abs(esig + eout) <= bound))


def test_qoperator_quantize_activation_sigmoid_composed_bound_needs_lipschitz_link():
    # Negative control: drop the Lipschitz link between esig and ex --
    # i.e. assume esig could be arbitrarily large relative to ex, keeping
    # only ex's and eout's own round-trip bounds -- and the composed claim
    # is no longer a theorem. Confirms the 0.25 scaling factor in the
    # composed bound genuinely comes from, and needs, the Lipschitz axiom
    # above, rather than holding vacuously from the round-trip bounds alone.
    ex, esig, eout, Xs, Ys = z3.Reals("ex esig eout Xs Ys")
    bound = z3.RealVal(1) / 4 * (Xs / 2) + Ys / 2

    solver = z3.Solver()
    solver.add(Xs > 0, Ys > 0, _abs(ex) <= Xs / 2, _abs(eout) <= Ys / 2)
    solver.add(z3.Not(_abs(esig + eout) <= bound))
    assert solver.check() == z3.sat, (
        "the composed Sigmoid bound holds even without the esig-ex "
        "Lipschitz link -- negative control is vacuous"
    )


# ---------------------------------------------------------------------------
# 2. Z3 content -- LeakyRelu
# ---------------------------------------------------------------------------


def _leaky_relu(x, alpha):
    return z3.If(x >= 0, x, alpha * x)


def _leaky_lipschitz_constant(alpha):
    return z3.If(_abs(alpha) >= 1, _abs(alpha), z3.RealVal(1))


def test_qoperator_quantize_activation_leaky_relu_lipschitz_case_split():
    # Unlike Sigmoid, LeakyRelu is piecewise LINEAR, so Z3 verifies its own
    # Lipschitz claim end to end -- no external axiom needed. z3.If encodes
    # the piecewise definition directly; Z3's own case-split handles all
    # four sign combinations of the free x1/x2 pair (both >= 0, both < 0,
    # and the two mixed-sign cases) for a free real alpha.
    x1, x2, alpha = z3.Reals("x1 x2 alpha")
    lipschitz = _leaky_lipschitz_constant(alpha)
    prove(
        _abs(_leaky_relu(x1, alpha) - _leaky_relu(x2, alpha))
        <= lipschitz * _abs(x1 - x2)
    )


def test_qoperator_quantize_activation_leaky_relu_default_alpha_lipschitz_is_1_not_alpha():
    # Tightness check for this pass's own default alpha=0.01 (module
    # docstring): whenever |alpha| <= 1 (which includes the default 0.01),
    # max(1, |alpha|) collapses to exactly 1, dominated by the x >= 0
    # branch's own unit slope -- NOT alpha itself, and in particular not
    # 0.01. A pass author who assumed "the Lipschitz constant is alpha"
    # would badly underestimate the true worst-case error for this pass's
    # single most common configuration.
    alpha = z3.Real("alpha")
    lipschitz = _leaky_lipschitz_constant(alpha)
    prove(z3.Implies(z3.And(alpha >= -1, alpha <= 1), lipschitz == 1))

    # And the concrete default value satisfies that hypothesis.
    default_alpha = 0.01
    assert -1 <= default_alpha <= 1


def test_qoperator_quantize_activation_leaky_relu_needs_max_1_alpha_not_just_1():
    # Negative control: a concrete counterexample showing a naive
    # constant-1 Lipschitz bound (ignoring alpha entirely) fails once
    # |alpha| > 1 -- confirming max(1, |alpha|)'s own |alpha| branch is
    # genuinely load-bearing, not just a defensive max() that never actually
    # matters. alpha=5, x1=1, x2=-1: LeakyRelu(1, 5) = 1,
    # LeakyRelu(-1, 5) = -5, so |diff| = 6 against |x1 - x2| = 2 -- a ratio
    # of 3, which the correct max(1, |alpha|) = 5 bound accommodates but a
    # naive constant of 1 does not.
    alpha_val, x1_val, x2_val = 5.0, 1.0, -1.0

    def leaky(x):
        return x if x >= 0 else alpha_val * x

    diff = abs(leaky(x1_val) - leaky(x2_val))
    naive_bound = 1.0 * abs(x1_val - x2_val)
    correct_bound = max(1.0, abs(alpha_val)) * abs(x1_val - x2_val)

    assert diff > naive_bound, (
        "the naive constant-1 Lipschitz bound unexpectedly holds for "
        "alpha=5 -- negative control is vacuous"
    )
    assert diff <= correct_bound

    # Same fact confirmed generically via Z3 (some alpha/x1/x2 with
    # |alpha| > 1 violates the constant-1 bound), not merely for this one
    # concrete triple.
    x1, x2, alpha = z3.Reals("x1 x2 alpha")
    solver = z3.Solver()
    solver.add(_abs(alpha) > 1)
    solver.add(
        z3.Not(
            _abs(_leaky_relu(x1, alpha) - _leaky_relu(x2, alpha)) <= 1 * _abs(x1 - x2)
        )
    )
    assert solver.check() == z3.sat, (
        "the constant-1 Lipschitz bound holds for every |alpha| > 1 -- "
        "negative control is vacuous"
    )


def test_qoperator_quantize_activation_leaky_relu_composed_bound_holds():
    # The full two-layer claim, LeakyRelu's analogue of Sigmoid's own
    # composed bound above: |LeakyRelu(X) - Y| <= max(1,|alpha|)*(Xs/2) +
    # Ys/2. Same direct-error-variable idiom and same telescoping
    # (eleaky + eout), but the Lipschitz link (|eleaky| <= L * |ex|) is
    # itself a proven Z3 theorem (the case-split above), not an external
    # axiom -- included here as a hypothesis anyway, exactly mirroring
    # Sigmoid's composed-bound query shape, since composing "L*|ex| is a
    # theorem" with "|ex| <= Xs/2" to get "L*|ex| <= L*(Xs/2)" is itself a
    # nonlinear step (multiplying an inequality by a free, only
    # sign-constrained L) worth handing to Z3 explicitly rather than doing
    # by hand.
    ex, eleaky, eout, Xs, Ys, alpha = z3.Reals("ex eleaky eout Xs Ys alpha")
    lipschitz = _leaky_lipschitz_constant(alpha)
    hypotheses = z3.And(
        Xs > 0,
        Ys > 0,
        _abs(ex) <= Xs / 2,
        _abs(eleaky) <= lipschitz * _abs(ex),
        _abs(eout) <= Ys / 2,
    )
    bound = lipschitz * (Xs / 2) + Ys / 2
    prove(z3.Implies(hypotheses, _abs(eleaky + eout) <= bound))


def test_qoperator_quantize_activation_leaky_relu_composed_bound_needs_lipschitz_link():
    # Negative control mirroring Sigmoid's own: drop the Lipschitz link
    # between eleaky and ex, and the composed claim is no longer a theorem.
    ex, eleaky, eout, Xs, Ys, alpha = z3.Reals("ex eleaky eout Xs Ys alpha")
    lipschitz = _leaky_lipschitz_constant(alpha)
    bound = lipschitz * (Xs / 2) + Ys / 2

    solver = z3.Solver()
    solver.add(Xs > 0, Ys > 0, _abs(ex) <= Xs / 2, _abs(eout) <= Ys / 2)
    solver.add(z3.Not(_abs(eleaky + eout) <= bound))
    assert solver.check() == z3.sat, (
        "the composed LeakyRelu bound holds even without the eleaky-ex "
        "Lipschitz link -- negative control is vacuous"
    )


# ---------------------------------------------------------------------------
# 3. Differential / structural content
# ---------------------------------------------------------------------------


def _model(body, opset=17, ir_version=10, extra_imports=""):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}{extra_imports}]
        >
        {body}
        """
    )
    return model


def _quantize_qoperator_activation(model, activation_ranges):
    """Invokes the real compiled pass directly via the nanobind-exposed
    ``quantize_qoperator_activation(model_bytes, activation_ranges)`` --
    module docstring. Runs ``OptimizeFixed`` with exactly
    ``["qoperator_quantize_activation"]``.
    """
    out = onnx.ModelProto()
    out.ParseFromString(
        C.quantize_qoperator_activation(model.SerializeToString(), activation_ranges)
    )
    return out


def _producer(model, output_name):
    return next(n for n in model.graph.node if output_name in n.output)


def _expected_asymmetric_uint8_quant_params(min_val, max_val):
    """Independent re-implementation of ``ComputeAsymmetricUint8QuantParams``
    (static_quantize_matmul.h), in float32 to match the pass's own arithmetic
    precision -- identical to this suite's other qoperator_quantize_* files'
    own helper of the same name.
    """
    lo = min(np.float32(0.0), np.float32(min_val))
    hi = max(np.float32(0.0), np.float32(max_val))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zero_point = int(np.clip(np.round(-lo / scale), 0, 255))
    return np.float32(scale), zero_point


def _disable_opt_session(model_bytes):
    # Disable graph optimization explicitly -- see module docstring /
    # tests/test_ort_matmul_nbits_workaround.py's docstring for this suite's
    # existing precedent of a real ORT graph-optimization fusion bug.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model_bytes, sess_options=so, providers=["CPUExecutionProvider"]
    )


def test_qoperator_quantize_activation_qlinear_sigmoid_kernel_exists():
    # Confirmed empirically before relying on it anywhere else in this file
    # (this suite's own precedent, test_formal_verify_qoperator_quantize_
    # softmax.py's docstring): a minimal standalone QuantizeLinear ->
    # QLinearSigmoid -> DequantizeLinear model actually runs on
    # onnxruntime's CPU provider.
    model = _model(
        """
        g (float[2,4] X) => (float[2,4] Y)
        <float Xs = {0.05}, uint8 Xzp = {20}, float Ys = {0.01}, uint8 Yzp = {0}>
        {
          Xq = QuantizeLinear(X, Xs, Xzp)
          Yq = com.microsoft.QLinearSigmoid(Xq, Xs, Xzp, Ys, Yzp)
          Y = DequantizeLinear(Yq, Ys, Yzp)
        }
        """,
        extra_imports=', "com.microsoft": 1',
    )
    sess = _disable_opt_session(model.SerializeToString())
    x = np.linspace(-3.0, 3.0, 8, dtype=np.float32).reshape(2, 4)
    (y,) = sess.run(None, {"X": x})
    assert y.shape == (2, 4)
    assert np.all(np.isfinite(y))


def test_qoperator_quantize_activation_qlinear_leaky_relu_kernel_exists():
    # Same, for QLinearLeakyRelu (with a non-default alpha, to also confirm
    # the attribute itself is accepted by the kernel).
    model = _model(
        """
        g (float[2,4] X) => (float[2,4] Y)
        <float Xs = {0.05}, uint8 Xzp = {20}, float Ys = {0.02}, uint8 Yzp = {50}>
        {
          Xq = QuantizeLinear(X, Xs, Xzp)
          Yq = com.microsoft.QLinearLeakyRelu<alpha=2.0>(Xq, Xs, Xzp, Ys, Yzp)
          Y = DequantizeLinear(Yq, Ys, Yzp)
        }
        """,
        extra_imports=', "com.microsoft": 1',
    )
    sess = _disable_opt_session(model.SerializeToString())
    x = np.linspace(-3.0, 3.0, 8, dtype=np.float32).reshape(2, 4)
    (y,) = sess.run(None, {"X": x})
    assert y.shape == (2, 4)
    assert np.all(np.isfinite(y))


def test_qoperator_quantize_activation_sigmoid_pass_fires_and_matches_scheme():
    model = _model(
        """
        g (float[4,5] X) => (float[4,5] Y)
        {
          Y = Sigmoid(X)
        }
        """
    )
    x_range = (-5.0, 3.0)
    y_range = (0.0, 1.0)
    quantized = _quantize_qoperator_activation(model, {"X": x_range, "Y": y_range})

    op_types = {n.op_type for n in quantized.graph.node}
    assert "Sigmoid" not in op_types

    dq_node = _producer(quantized, "Y")
    assert dq_node.op_type == "DequantizeLinear"
    qlop_node = _producer(quantized, dq_node.input[0])
    assert qlop_node.op_type == "QLinearSigmoid"
    assert qlop_node.domain == "com.microsoft"
    assert dq_node.input[1:] == list(qlop_node.input[3:5])  # Ys, Yzp shared

    ql_node = _producer(quantized, qlop_node.input[0])
    assert ql_node.op_type == "QuantizeLinear"
    assert ql_node.input[0] == "X"
    assert ql_node.input[1:] == list(qlop_node.input[1:3])  # Xs, Xzp shared

    ms_imports = [o for o in quantized.opset_import if o.domain == "com.microsoft"]
    assert len(ms_imports) == 1
    assert ms_imports[0].version == 1

    init = {i.name: i for i in quantized.graph.initializer}
    x_scale = onnx.numpy_helper.to_array(init[ql_node.input[1]])
    x_zp = onnx.numpy_helper.to_array(init[ql_node.input[2]])
    expected_x_scale, expected_x_zp = _expected_asymmetric_uint8_quant_params(*x_range)
    assert x_zp.dtype == np.uint8
    assert int(x_zp) == expected_x_zp
    assert expected_x_zp != 0, "test calibration range must exercise a nonzero Xzp"
    np.testing.assert_allclose(float(x_scale), float(expected_x_scale), rtol=1e-6)


def test_qoperator_quantize_activation_leaky_relu_pass_fires_and_alpha_carried_over():
    # Non-default alpha=2.0 must be threaded through unchanged onto
    # QLinearLeakyRelu.
    model = _model(
        """
        g (float[4,5] X) => (float[4,5] Y)
        {
          Y = LeakyRelu<alpha=2.0>(X)
        }
        """
    )
    quantized = _quantize_qoperator_activation(
        model, {"X": (-5.0, 3.0), "Y": (-10.0, 3.0)}
    )
    dq_node = _producer(quantized, "Y")
    qlop_node = _producer(quantized, dq_node.input[0])
    assert qlop_node.op_type == "QLinearLeakyRelu"
    assert qlop_node.domain == "com.microsoft"
    attrs = {a.name: a for a in qlop_node.attribute}
    assert attrs["alpha"].f == 2.0


def test_qoperator_quantize_activation_leaky_relu_default_alpha_materialized():
    # A LeakyRelu node with NO explicit alpha attribute (relying on the ONNX
    # spec's own default) must still get an explicit alpha=0.01 on
    # QLinearLeakyRelu -- confirmed empirically that the source node's
    # attribute list is genuinely empty before the rewrite, i.e. this is a
    # real default lookup (GetValueFromAttrWithDefault), not an accidental
    # pass-through of an attribute that was already there.
    model = _model(
        """
        g (float[4,5] X) => (float[4,5] Y)
        {
          Y = LeakyRelu(X)
        }
        """
    )
    assert list(model.graph.node[0].attribute) == []

    quantized = _quantize_qoperator_activation(
        model, {"X": (-5.0, 3.0), "Y": (-0.05, 3.0)}
    )
    dq_node = _producer(quantized, "Y")
    qlop_node = _producer(quantized, dq_node.input[0])
    assert qlop_node.op_type == "QLinearLeakyRelu"
    attrs = {a.name: a for a in qlop_node.attribute}
    # attrs["alpha"].f is a float32 FLOAT attribute; 0.01 isn't exactly
    # representable in float32, hence the tolerance.
    np.testing.assert_allclose(attrs["alpha"].f, 0.01, rtol=1e-6)


def test_qoperator_quantize_activation_com_microsoft_import_not_duplicated():
    model = _model(
        """
        g (float[2,4] X) => (float[2,4] Y)
        {
          Y = Sigmoid(X)
        }
        """,
        extra_imports=', "com.microsoft": 1',
    )
    quantized = _quantize_qoperator_activation(
        model, {"X": (-3.0, 3.0), "Y": (0.0, 1.0)}
    )
    ms_imports = [o for o in quantized.opset_import if o.domain == "com.microsoft"]
    assert len(ms_imports) == 1
    assert ms_imports[0].version == 1
    assert _producer(quantized, "Y").op_type == "DequantizeLinear"


def test_qoperator_quantize_activation_declines_with_only_activation_range():
    model = _model(
        """
        g (float[2,4] X) => (float[2,4] Y)
        {
          Y = Sigmoid(X)
        }
        """
    )
    quantized = _quantize_qoperator_activation(model, {"X": (-3.0, 3.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Sigmoid"]


def test_qoperator_quantize_activation_declines_with_only_output_range():
    model = _model(
        """
        g (float[2,4] X) => (float[2,4] Y)
        {
          Y = LeakyRelu<alpha=0.3>(X)
        }
        """
    )
    quantized = _quantize_qoperator_activation(model, {"Y": (-1.0, 3.0)})
    assert [n.op_type for n in quantized.graph.node] == ["LeakyRelu"]


def test_qoperator_quantize_activation_declines_non_float_input():
    model = _model(
        """
        g (float16[2,4] X) => (float16[2,4] Y)
        {
          Y = Sigmoid(X)
        }
        """
    )
    quantized = _quantize_qoperator_activation(
        model, {"X": (-3.0, 3.0), "Y": (0.0, 1.0)}
    )
    assert [n.op_type for n in quantized.graph.node] == ["Sigmoid"]


def test_qoperator_quantize_activation_selective_firing_per_node():
    # Two independent nodes, only one calibrated: the pass must rewrite
    # exactly the calibrated one and leave the other untouched, confirming
    # patternMatchPredicate's per-node (not per-model) granularity.
    model = _model(
        """
        g (float[8] X1, float[8] X2) => (float[8] Y1, float[8] Y2)
        {
          Y1 = Sigmoid(X1)
          Y2 = LeakyRelu<alpha=2.0>(X2)
        }
        """
    )
    quantized = _quantize_qoperator_activation(
        model, {"X1": (-3.0, 3.0), "Y1": (0.0, 1.0)}
    )
    op_types = [n.op_type for n in quantized.graph.node]
    assert "LeakyRelu" in op_types  # Y2's branch untouched
    assert "Sigmoid" not in op_types  # Y1's branch rewritten
    assert _producer(quantized, "Y1").op_type == "DequantizeLinear"
    assert _producer(quantized, "Y2").op_type == "LeakyRelu"


# ---------------------------------------------------------------------------
# 4. Numeric-bound differential tests -- the genuine crux of this file
# ---------------------------------------------------------------------------

# 64 elements: reasonably sized, not a tiny 2x2 -- see module docstring for
# why (a real ONNX Runtime quantized-kernel edge case that only manifested
# in CI for very small/irregular tensor shapes).
_N_ELEMENTS = 64


def test_qoperator_quantize_activation_sigmoid_output_within_proved_bound():
    # Both calibration ranges are set to the actual observed (min, max) of X
    # and of the TRUE float Sigmoid output, so nothing clips -- the
    # round-trip lemmas' explicit side condition, for both tensors.
    rng = np.random.default_rng(0)
    x = rng.uniform(-6.0, 6.0, size=_N_ELEMENTS).astype(np.float32)
    y_float = (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)

    model = _model(
        f"""
        g (float[{_N_ELEMENTS}] X) => (float[{_N_ELEMENTS}] Y)
        {{
          Y = Sigmoid(X)
        }}
        """
    )
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y_float.min()), float(y_float.max())
    quantized = _quantize_qoperator_activation(
        model, {"X": (x_min, x_max), "Y": (y_min, y_max)}
    )

    sess = _disable_opt_session(quantized.SerializeToString())
    (y_quant,) = sess.run(None, {"X": x})

    x_scale, _x_zp = _expected_asymmetric_uint8_quant_params(x_min, x_max)
    y_scale, _y_zp = _expected_asymmetric_uint8_quant_params(y_min, y_max)
    eps_x = float(x_scale) / 2.0
    eps_y = float(y_scale) / 2.0
    bound = 0.25 * eps_x + eps_y  # the proved combined bound

    error = np.abs(y_float - y_quant)
    assert np.all(error <= bound + 1e-6), (
        f"max abs error {error.max()} exceeds the proved Lipschitz-scaled "
        f"combined bound {bound}"
    )

    # And genuinely close -- not merely within a loose worst-case bound --
    # for these well-scaled inputs, consistent with the small (single-digit
    # percent of the calibrated range) UINT8 quantization steps involved.
    np.testing.assert_allclose(y_quant, y_float, rtol=0.05, atol=0.02)


def test_qoperator_quantize_activation_leaky_relu_default_alpha_output_within_proved_bound():
    # Same shape, LeakyRelu at this pass's own DEFAULT alpha=0.01 -- so the
    # Lipschitz constant here is exactly 1 (test_qoperator_quantize_
    # activation_leaky_relu_default_alpha_lipschitz_is_1_not_alpha above),
    # not 0.01.
    rng = np.random.default_rng(1)
    x = rng.uniform(-6.0, 6.0, size=_N_ELEMENTS).astype(np.float32)
    alpha = 0.01
    y_float = np.where(x >= 0, x, alpha * x).astype(np.float32)

    model = _model(
        f"""
        g (float[{_N_ELEMENTS}] X) => (float[{_N_ELEMENTS}] Y)
        {{
          Y = LeakyRelu<alpha={alpha}>(X)
        }}
        """
    )
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y_float.min()), float(y_float.max())
    quantized = _quantize_qoperator_activation(
        model, {"X": (x_min, x_max), "Y": (y_min, y_max)}
    )

    sess = _disable_opt_session(quantized.SerializeToString())
    (y_quant,) = sess.run(None, {"X": x})

    x_scale, _x_zp = _expected_asymmetric_uint8_quant_params(x_min, x_max)
    y_scale, _y_zp = _expected_asymmetric_uint8_quant_params(y_min, y_max)
    eps_x = float(x_scale) / 2.0
    eps_y = float(y_scale) / 2.0
    lipschitz = max(1.0, abs(alpha))
    assert lipschitz == 1.0
    bound = lipschitz * eps_x + eps_y

    error = np.abs(y_float - y_quant)
    assert np.all(error <= bound + 1e-6), (
        f"max abs error {error.max()} exceeds the proved combined bound "
        f"{bound} (alpha={alpha}, Lipschitz constant {lipschitz})"
    )


def test_qoperator_quantize_activation_leaky_relu_nondefault_alpha_output_within_proved_bound():
    # alpha=2.0 exercises the max(1, |alpha|) branch genuinely -- the
    # Lipschitz constant here is 2.0, not 1, so the proved bound scales
    # X's own round-trip error up accordingly.
    rng = np.random.default_rng(2)
    x = rng.uniform(-6.0, 6.0, size=_N_ELEMENTS).astype(np.float32)
    alpha = 2.0
    y_float = np.where(x >= 0, x, alpha * x).astype(np.float32)

    model = _model(
        f"""
        g (float[{_N_ELEMENTS}] X) => (float[{_N_ELEMENTS}] Y)
        {{
          Y = LeakyRelu<alpha={alpha}>(X)
        }}
        """
    )
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y_float.min()), float(y_float.max())
    quantized = _quantize_qoperator_activation(
        model, {"X": (x_min, x_max), "Y": (y_min, y_max)}
    )

    sess = _disable_opt_session(quantized.SerializeToString())
    (y_quant,) = sess.run(None, {"X": x})

    x_scale, _x_zp = _expected_asymmetric_uint8_quant_params(x_min, x_max)
    y_scale, _y_zp = _expected_asymmetric_uint8_quant_params(y_min, y_max)
    eps_x = float(x_scale) / 2.0
    eps_y = float(y_scale) / 2.0
    lipschitz = max(1.0, abs(alpha))
    assert lipschitz == 2.0
    bound = lipschitz * eps_x + eps_y

    error = np.abs(y_float - y_quant)
    assert np.all(error <= bound + 1e-6), (
        f"max abs error {error.max()} exceeds the proved combined bound "
        f"{bound} (alpha={alpha}, Lipschitz constant {lipschitz})"
    )

    # And confirm a naive constant-1 bound (ignoring alpha) would genuinely
    # have been too tight here -- the real quantized error actually exceeds
    # it -- mirroring the Z3-level negative control above
    # (test_qoperator_quantize_activation_leaky_relu_needs_max_1_alpha_not_
    # just_1) with a real quantized execution instead of a symbolic one.
    naive_bound = 1.0 * eps_x + eps_y
    assert naive_bound < bound
    assert error.max() > naive_bound, (
        "this alpha=2.0 case no longer distinguishes the naive constant-1 "
        "bound from the correct max(1,|alpha|) one -- strengthen the test "
        "input so the naive bound is genuinely too tight"
    )
