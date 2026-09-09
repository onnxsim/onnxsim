"""Formal check for CrossLayerEqualization (opt-in;
``onnxsim/passes/cross_layer_equalization.h``): the data-free weight-
equalization preprocessing step for quantization from "Data-Free
Quantization Through Weight Equalization and Bias Correction" (Nagel et al.,
2019). It matches ``Conv1 -> [activation] -> Conv2`` where the activation, if
present, is ``Relu``/``PRelu``/``LeakyRelu`` (or absent entirely -- the
identity), both convs have ``group == 1`` and constant FLOAT32 weights (plus
an optional constant FLOAT32 bias on Conv1 only), and Conv1's output-channel
count ``C`` equals Conv2's input-channel count. Per shared channel ``c`` it
computes ``r1[c] = max(|W1[c, ...]|)``, ``r2[c] = max(|W2[:, c, ...]|)``,
``S[c] = sqrt(r1[c] / r2[c])`` (left at 1 if either range is exactly 0), and
rewrites::

    W1'[c, ...] = W1[c, ...] / S[c];  b1'[c] = b1[c] / S[c]
    W2'[:, c, ...] = W2[:, c, ...] * S[c]

Neither Conv node is destroyed -- only their weight/bias *inputs* change
(``NodeDestroyType::DestroyZero``).

The genuine mathematical content, spelled out in the header comment: the
composed function ``x -> Conv2(activation(Conv1(x)))`` is *exactly* unchanged
by this rewrite, for ANY positive per-channel scale vector ``S`` -- not just
the specific ``sqrt(r1/r2)`` one the pass happens to pick (that particular
choice only affects how *balanced* the resulting ranges are, not
correctness). Three facts combine to give this:

1. Conv is linear per output channel in its own weight/bias for that
   channel: scaling channel ``c``'s own weight slice and bias by ``1/S``
   scales that channel's Conv1 output by ``1/S``. Modeled here as an axiom
   about an uninterpreted ``Conv1Out(weight, bias) -> output`` function
   (mirroring ``test_formal_verify_fuse_bn_into_conv.py``'s and
   ``test_formal_verify_fuse_matmul_add_bias_into_gemm.py``'s own
   dot-product-level treatment of this same underlying fact, abstracted one
   level further here since CLE's actual content is about chaining three
   such facts together, not any single one of them) rather than re-derived
   from Conv's own windowed-sum semantics.
2. A positive-homogeneous-degree-1 activation ``f`` satisfies
   ``f(a*x) = a*f(x)`` for every ``a > 0`` -- true of Relu, PRelu,
   LeakyRelu, and trivially of the identity (no activation at all). Stated
   as an EXPLICIT axiom, parametrized over an uninterpreted ``f`` so one
   proof covers all of those activations via a single abstract hypothesis,
   and ground-instantiated at exactly the point this proof needs
   (``a = 1/S``, ``x =`` Conv1's channel-``c`` output) rather than left
   universally quantified -- this suite's established idiom (see
   ``test_formal_verify_fuse_consecutive_log_softmax.py``'s and
   ``test_formal_verify_replace_einsum_with_matmul.py``'s own ground
   instantiations) for avoiding Z3 quantifier-instantiation fragility.
3. Conv2's output is linear (jointly) in input-channel ``c``'s own weight
   slice and its own input value: scaling the weight by ``S`` and dividing
   that channel's (post-activation) input by ``S`` leaves that channel's
   contribution to Conv2's output unchanged. Modeled as an axiom about an
   uninterpreted ``Contrib2(weight, channel_value) -> contribution``
   function, with Conv2's *total* output represented as that one channel's
   contribution plus an uninterpreted ``rest`` term standing in for every
   other channel's contribution -- unaffected by this rewrite, since only
   channel ``c``'s weights/bias are ever touched.

Chaining these: Conv1' channel ``c``'s output is ``(1/S)`` times Conv1's,
the activation passes that scale through unchanged by homogeneity, and
Conv2' channel-``c``'s weight contribution absorbs the ``S`` back via
linearity -- so the final sum (that channel's contribution, now restored to
its original value, plus every untouched other channel's contribution) is
byte-for-byte the same expression as before the rewrite. A negative-control
test confirms this is genuine, load-bearing content: drop the homogeneity
axiom (leaving ``f`` a totally free uninterpreted function, standing in for
a non-pass-through activation like Sigmoid) and Z3 finds a real
counterexample -- exactly why the pass restricts its activation match to
Relu/PRelu/LeakyRelu/no-activation.

Differential tests build a small, hand-verifiable ``Conv1 -> Relu -> Conv2``
chain (and a no-activation variant) via ``onnx.parser.parse_model()`` with
``r1 = [4.0, 1.0]``, ``r2 = [1.0, 4.0]`` chosen so ``S = sqrt(r1/r2) =
[2.0, 0.5]`` and every rewritten weight/bias value is a round number, and
confirm the real compiled pass rewrites Conv1's weight/bias and Conv2's
weight to those exact values, that the end-to-end numeric output is
unchanged (both via onnxsim's own ``simplify_isolated_extra`` equivalence
check and an explicit onnxruntime comparison), that a non-pass-through
activation (Sigmoid) or a grouped Conv2 makes the predicate decline
entirely, and that an already-balanced pair (``S[c] == 1`` for every ``c``)
reports no change at all -- confirming ``RescaleAndApply``'s own
``any_change``/``kConvergedTol`` early-return, which is what lets onnxsim's
fixed-point pass driver converge instead of looping forever re-applying a
no-op.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser


def _formulas():
    """Shared Z3 vocabulary for the general (activation present) soundness
    proof. Returns ``(axioms, original_total, new_total, f, consumer)``.

    ``S``, ``w1c``, ``b1c``, ``w2c`` and ``rest`` are all free Z3 variables,
    implicitly universally quantified by ``prove()`` -- covering every
    channel weight/bias value, every OTHER channels' contribution, and every
    positive scale ``S`` in one proof, with no ``ForAll`` needed anywhere.
    """
    S, w1c, b1c, w2c, rest = z3.Reals("S w1c b1c w2c rest")
    Conv1Out = z3.Function("Conv1Out", z3.RealSort(), z3.RealSort(), z3.RealSort())
    Contrib2 = z3.Function("Contrib2", z3.RealSort(), z3.RealSort(), z3.RealSort())
    f = z3.Function("f", z3.RealSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    y = Conv1Out(w1c, b1c)  # Conv1's original channel-c output
    act_c = f(y)  # activation(Conv1)'s original channel-c output

    # Fact 1: Conv1's channel-c output, as a function of only that channel's
    # own weight slice and bias, scales by 1/S when the weight/bias do.
    linear1_axiom = Conv1Out(w1c / S, b1c / S) == y / S

    # Fact 2: f is positive-homogeneous of degree 1 (f(a*x) = a*f(x) for
    # a > 0), ground-instantiated at a = 1/S (positive since S > 0) and
    # x = y -- exactly the point the chained proof needs.
    homogeneity_axiom = f(y / S) == act_c / S

    # Fact 3: Conv2's channel-c contribution is jointly linear in its own
    # weight slice and its own (post-activation) input value -- scaling one
    # by S and the other by 1/S cancels exactly.
    linear2_axiom = Contrib2(w2c * S, act_c / S) == Contrib2(w2c, act_c)

    side = S > 0
    axioms = z3.And(linear1_axiom, homogeneity_axiom, linear2_axiom, side)

    # Conv2's TOTAL output: channel c's own contribution plus every other
    # channel's (the free `rest` term -- untouched by this rewrite, since
    # only channel c's weights/bias are ever modified).
    original_total = Contrib2(w2c, act_c) + rest
    new_total = Contrib2(w2c * S, f(Conv1Out(w1c / S, b1c / S))) + rest

    return axioms, original_total, new_total, f, consumer


def test_cross_layer_equalization_is_sound():
    axioms, original_total, new_total, _f, consumer = _formulas()

    prove(z3.Implies(axioms, original_total == new_total))
    prove(z3.Implies(axioms, consumer(original_total) == consumer(new_total)))


def test_cross_layer_equalization_no_activation_is_sound():
    # The "activation == nullptr" branch: Conv1 feeds Conv2 directly (the
    # identity function is positive-homogeneous too, but this models the
    # actual no-activation graph shape directly rather than instantiating
    # the general proof's abstract f at the identity).
    S, w1c, b1c, w2c, rest = z3.Reals("S w1c b1c w2c rest")
    Conv1Out = z3.Function("Conv1Out", z3.RealSort(), z3.RealSort(), z3.RealSort())
    Contrib2 = z3.Function("Contrib2", z3.RealSort(), z3.RealSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    y = Conv1Out(w1c, b1c)
    linear1_axiom = Conv1Out(w1c / S, b1c / S) == y / S
    linear2_axiom = Contrib2(w2c * S, y / S) == Contrib2(w2c, y)
    axioms = z3.And(linear1_axiom, linear2_axiom, S > 0)

    original_total = Contrib2(w2c, y) + rest
    new_total = Contrib2(w2c * S, Conv1Out(w1c / S, b1c / S)) + rest

    prove(z3.Implies(axioms, original_total == new_total))
    prove(z3.Implies(axioms, consumer(original_total) == consumer(new_total)))


def test_cross_layer_equalization_negative_control_requires_homogeneity_axiom():
    # Sanity check that the proof above is genuine, not vacuous: drop the
    # homogeneity axiom entirely (f left a totally free uninterpreted
    # function -- standing in for a non-pass-through activation like
    # Sigmoid, which is not positive-homogeneous), keeping only the two
    # Conv-linearity facts, and Z3 must find a real counterexample. This is
    # exactly why the pass restricts its activation match to
    # Relu/PRelu/LeakyRelu/no-activation.
    S, w1c, b1c, w2c, rest = z3.Reals("S w1c b1c w2c rest")
    Conv1Out = z3.Function("Conv1Out", z3.RealSort(), z3.RealSort(), z3.RealSort())
    Contrib2 = z3.Function("Contrib2", z3.RealSort(), z3.RealSort(), z3.RealSort())
    f = z3.Function("f", z3.RealSort(), z3.RealSort())

    y = Conv1Out(w1c, b1c)
    act_c = f(y)
    linear1_axiom = Conv1Out(w1c / S, b1c / S) == y / S
    linear2_axiom = Contrib2(w2c * S, act_c / S) == Contrib2(w2c, act_c)
    axioms_without_homogeneity = z3.And(linear1_axiom, linear2_axiom, S > 0)

    original_total = Contrib2(w2c, act_c) + rest
    new_total = Contrib2(w2c * S, f(Conv1Out(w1c / S, b1c / S))) + rest

    solver = z3.Solver()
    solver.add(axioms_without_homogeneity)
    solver.add(z3.Not(original_total == new_total))
    assert solver.check() == z3.sat, (
        "the rewrite still holds without the positive-homogeneity axiom -- "
        "negative control is vacuous"
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
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


# r1 = [4.0, 1.0] (Conv1's per-output-channel weight range), r2 = [1.0, 4.0]
# (Conv2's per-input-channel weight range) -> S = sqrt(r1/r2) = [2.0, 0.5],
# chosen so every rewritten weight/bias value below is a round number,
# hand-verifiable without a calculator.
_W1 = np.array([[[[4.0]]], [[[1.0]]]], dtype=np.float32)  # [2,1,1,1]
_B1 = np.array([2.0, 1.0], dtype=np.float32)
_W2 = np.array([[[[1.0]], [[4.0]]]], dtype=np.float32)  # [1,2,1,1]
_EXPECTED_W1 = np.array([[[[2.0]]], [[[2.0]]]], dtype=np.float32)
_EXPECTED_B1 = np.array([1.0, 2.0], dtype=np.float32)
_EXPECTED_W2 = np.array([[[[2.0]], [[2.0]]]], dtype=np.float32)


def _assert_end_to_end_output_unchanged(model, sim_model):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, 2, 2)).astype(np.float32)
    orig_sess = ort.InferenceSession(model.SerializeToString())
    new_sess = ort.InferenceSession(sim_model.SerializeToString())
    (orig_out,) = orig_sess.run(None, {"X": x})
    (new_out,) = new_sess.run(None, {"X": x})
    np.testing.assert_allclose(orig_out, new_out, rtol=1e-5, atol=1e-6)


def test_cross_layer_equalization_pass_fires_and_rescales_correctly():
    model = _model(
        """
        g (float[1,1,2,2] X) => (float[1,1,2,2] Y)
        {
          c1 = Conv(X, W1, B1)
          a = Relu(c1)
          Y = Conv(a, W2)
        }
        """,
        [_f32(_W1, "W1"), _f32(_B1, "B1"), _f32(_W2, "W2")],
    )

    # check_n=3 (the default): the pass's headline correctness claim is that
    # the end-to-end function is UNCHANGED, so onnxsim's own random-input
    # equivalence check must pass, not just be skipped.
    sim_model, ops = simplify_isolated_extra(model, "cross_layer_equalization")
    assert ops["Conv"] == 2
    assert ops["Relu"] == 1

    conv1 = producer(sim_model, "c1")
    conv2 = producer(sim_model, "Y")
    by_name = {
        init.name: numpy_helper.to_array(init) for init in sim_model.graph.initializer
    }
    np.testing.assert_allclose(by_name[conv1.input[1]], _EXPECTED_W1, rtol=1e-6)
    np.testing.assert_allclose(by_name[conv1.input[2]], _EXPECTED_B1, rtol=1e-6)
    np.testing.assert_allclose(by_name[conv2.input[1]], _EXPECTED_W2, rtol=1e-6)

    # The single most important differential check: an explicit onnxruntime
    # comparison of the ORIGINAL vs REWRITTEN model's numeric output, on top
    # of onnxsim's own internal equivalence check already asserted above.
    _assert_end_to_end_output_unchanged(model, sim_model)


def test_cross_layer_equalization_pass_fires_without_activation():
    # Conv1 -> Conv2 directly (activation == nullptr) -- the identity is
    # positive-homogeneous too, so the exact same S/rewrite applies.
    model = _model(
        """
        g (float[1,1,2,2] X) => (float[1,1,2,2] Y)
        {
          c1 = Conv(X, W1, B1)
          Y = Conv(c1, W2)
        }
        """,
        [_f32(_W1, "W1"), _f32(_B1, "B1"), _f32(_W2, "W2")],
    )

    sim_model, ops = simplify_isolated_extra(model, "cross_layer_equalization")
    assert ops["Conv"] == 2

    conv1 = producer(sim_model, "c1")
    conv2 = producer(sim_model, "Y")
    by_name = {
        init.name: numpy_helper.to_array(init) for init in sim_model.graph.initializer
    }
    np.testing.assert_allclose(by_name[conv1.input[1]], _EXPECTED_W1, rtol=1e-6)
    np.testing.assert_allclose(by_name[conv1.input[2]], _EXPECTED_B1, rtol=1e-6)
    np.testing.assert_allclose(by_name[conv2.input[1]], _EXPECTED_W2, rtol=1e-6)

    _assert_end_to_end_output_unchanged(model, sim_model)


def test_cross_layer_equalization_declines_non_pass_through_activation():
    # Sigmoid is not positive-homogeneous, so it isn't in
    # IsPassThroughActivation's allow-list (Relu/PRelu/LeakyRelu only) --
    # the predicate must decline outright, leaving both convs' weights
    # completely untouched.
    model = _model(
        """
        g (float[1,1,2,2] X) => (float[1,1,2,2] Y)
        {
          c1 = Conv(X, W1, B1)
          a = Sigmoid(c1)
          Y = Conv(a, W2)
        }
        """,
        [_f32(_W1, "W1"), _f32(_B1, "B1"), _f32(_W2, "W2")],
    )

    sim_model, ops = simplify_isolated_extra(model, "cross_layer_equalization")
    assert ops["Conv"] == 2
    assert ops["Sigmoid"] == 1

    conv1 = producer(sim_model, "c1")
    conv2 = producer(sim_model, "Y")
    by_name = {
        init.name: numpy_helper.to_array(init) for init in sim_model.graph.initializer
    }
    np.testing.assert_allclose(by_name[conv1.input[1]], _W1)
    np.testing.assert_allclose(by_name[conv1.input[2]], _B1)
    np.testing.assert_allclose(by_name[conv2.input[1]], _W2)


def test_cross_layer_equalization_declines_grouped_conv2():
    # group != 1 on Conv2 breaks the clean 1:1 input/output channel
    # correspondence CLE's per-channel bookkeeping assumes --
    # ValidateConvWeights's HasSingleGroup check fails for conv2, so
    # TryMatch declines before even inspecting conv1.
    w2_grouped = np.array([[[[1.0]]], [[[4.0]]]], dtype=np.float32)  # [2,1,1,1]
    model = _model(
        """
        g (float[1,1,2,2] X) => (float[1,2,2,2] Y)
        {
          c1 = Conv(X, W1, B1)
          Y = Conv<group = 2>(c1, W2)
        }
        """,
        [_f32(_W1, "W1"), _f32(_B1, "B1"), _f32(w2_grouped, "W2")],
    )

    sim_model, ops = simplify_isolated_extra(model, "cross_layer_equalization")
    assert ops["Conv"] == 2

    conv1 = producer(sim_model, "c1")
    conv2 = producer(sim_model, "Y")
    by_name = {
        init.name: numpy_helper.to_array(init) for init in sim_model.graph.initializer
    }
    np.testing.assert_allclose(by_name[conv1.input[1]], _W1)
    np.testing.assert_allclose(by_name[conv1.input[2]], _B1)
    np.testing.assert_allclose(by_name[conv2.input[1]], w2_grouped)


def test_cross_layer_equalization_already_balanced_reports_no_change():
    # r1 == r2 for every channel ([4.0, 1.0] on both sides) -> S == 1
    # everywhere -- RescaleAndApply's own any_change/kConvergedTol
    # early-return must report no change at all, which is what lets
    # onnxsim's fixed-point driver converge instead of looping forever
    # re-applying a no-op.
    w2_balanced = np.array([[[[4.0]], [[1.0]]]], dtype=np.float32)  # [1,2,1,1]
    model = _model(
        """
        g (float[1,1,2,2] X) => (float[1,1,2,2] Y)
        {
          c1 = Conv(X, W1, B1)
          a = Relu(c1)
          Y = Conv(a, W2)
        }
        """,
        [_f32(_W1, "W1"), _f32(_B1, "B1"), _f32(w2_balanced, "W2")],
    )

    sim_model, ops = simplify_isolated_extra(model, "cross_layer_equalization")
    assert ops["Conv"] == 2

    conv1 = producer(sim_model, "c1")
    conv2 = producer(sim_model, "Y")
    # No rewrite happened at all -- not even a byte-identical replacement --
    # so the weight/bias inputs still name the ORIGINAL initializers.
    assert conv1.input[1] == "W1"
    assert conv1.input[2] == "B1"
    assert conv2.input[1] == "W2"
