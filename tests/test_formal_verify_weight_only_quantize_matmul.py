"""Formal check for WeightOnlyQuantizeMatMul (opt-in; onnxsim's own
``onnxsim/passes/weight_only_quantize_matmul.h``): unlike
``dynamic_quantize_matmul.h``
(``test_formal_verify_dynamic_quantize_matmul.py``) and
``static_quantize_matmul.h``, ONLY the constant weight is quantized -- the
activation ``X`` is left completely untouched, with no
``DynamicQuantizeLinear``/``QuantizeLinear``/``DequantizeLinear`` on the
activation side and no calibration data of any kind. It rewrites
``Y = MatMul(X, W)`` (or a "vanilla" ``Gemm``, bias left untouched) --- ``W``
a constant 2-D FLOAT32 tensor, ``X`` FLOAT32 -- into::

    Wq, Ws := per-output-channel symmetric INT8 quantization of W (computed
              ONCE, at pass-transform time, from W's static values)
    Wdq = DequantizeLinear(Wq, Ws, axis=<channel_axis>)   # zero_point=0
    Y   = MatMul(X, Wdq)

``channel_axis`` is ``W``'s own output-channel axis in its own (untransposed)
storage layout: 0 when Gemm's ``transB`` made ``W`` ``[N, K]``, else 1 (``W``
already ``[K, N]``, MatMul's native layout) -- see the header comment.

This is a *single-operand* special case of the same MAC bounded-error lemma
``test_formal_verify_quantized_mac_bound.py`` establishes and
``test_formal_verify_dynamic_quantize_matmul.py`` reuses for its own
(two-quantized-operand) rewrite: here ``X`` has NO error at all (``eps_x =
0`` in ``quantized_mac_bound``'s own vocabulary, since ``X[i, k] ==
X[i, k]`` trivially -- it is never quantized), so ``quantized_mac_bound``'s
general bound::

    eps_x * sum_k |W[k, n]| + eps_w * sum_k |X[i, k]| + K * eps_w * eps_x

collapses, with ``eps_x := 0`` and ``eps_w := Ws[n] / 2`` (the standard
``DequantizeLinear(QuantizeLinear(...))`` round-trip bound
``test_formal_verify_quantize_round_trip.py`` proves), to just::

    |MatMul(X, W)[i, n] - MatMul(X, Wdq)[i, n]| <= (Ws[n] / 2) * sum_k |X[i, k]|

-- no ``Xs``-driven term (there is no ``Xs``: ``X`` is never quantized) and no
cross term at all (that term is exactly ``K * eps_w * eps_x``, and ``eps_x =
0`` zeroes it). Given the substitution is this direct, the proof below states
and proves this single-operand bound as one straightforward Z3 query from the
start, rather than splitting an identity lemma from a bound lemma the way
``test_formal_verify_dynamic_quantize_matmul.py`` had to
(``dynamic_quantize_matmul`` has TWO quantized operands feeding one nonlinear
node chain -- ``Cast<float>(Acc) * (Xs * Ws)`` -- whose combined nonlinear
Z3 query hung past a minute; there is no such combined nonlinear chain here,
since ``X`` never gets a quantized/scale representation to multiply against
in the first place). This was confirmed EMPIRICALLY (see the module's git
history / commit message for the measured wall-clock time) before writing
this file's structure around it, precisely to avoid re-discovering that
exact hang by trial and error: the direct-error-variable formulation
(``ew``, mirroring ``quantized_mac_bound``'s own ``ew``/``ex`` idiom, with no
``ex`` at all since there is nothing for it to model) is used from the start,
never the ``Wq``/``Ws``-multiplicand form.

A second, genuinely new piece of content this pass's proof needs that
``dynamic_quantize_matmul``'s did not: that pass folds its dequantization
into a plain ``Mul`` (``Cast<float>(Acc) * (Xs * Ws)``), so its own proof
never has to reason about ``DequantizeLinear`` itself. This pass instead
leaves an actual ``DequantizeLinear(Wq, Ws, axis=channel_axis)`` node in the
graph, so a short separate test below confirms that op's own defining
per-channel-axis formula (``Wdq[k, n] = Wq[k, n] * Ws[n]`` when ``axis``
selects the output-channel dimension, implicit ``zero_point = 0``) is
*exactly* the quantity the rounding bound is stated about -- i.e. that
treating ``Wdq`` as "some per-channel dequantization satisfying the round-
trip bound" (what the main proof below does, via the free ``ew`` error
variable) is not begging the question: it really is what
``DequantizeLinear``'s own semantics produce, given ``Wq``'s defining
round-to-nearest property (``test_formal_verify_quantize_round_trip.py``'s
own hypothesis shape).

A negative-control test confirms the bound is not vacuous: with no rounding-
error budget on ``ew`` at all, the same claim is not a theorem -- Z3 finds a
real counterexample.

No consumer-composition step is added (this suite's usual substitution-
safety idiom, e.g. ``test_formal_verify_eliminate_identity.py``): per
``test_formal_verify_dynamic_quantize_matmul.py``'s own reasoning for a
numeric *bound* (not an equality) -- an arbitrary ``consumer`` need not be
Lipschitz, so "``|a - b| <= bound``" implies nothing in general about
"``|consumer(a) - consumer(b)|``" -- and that reasoning applies identically
here; ``quantized_mac_bound`` itself, the template for this shape of claim,
does not compose with a consumer either.

Differential tests build a plain float ``MatMul``/``Gemm`` via
``onnx.parser`` (per ``CLAUDE.md``), run the real pass alone via
``simplify_isolated_extra`` (and, for the numeric-bound check, via
:func:`onnxsim.quantize_weight_only`, which applies exactly this rewrite --
see its own docstring), and confirm: the exact ``MatMul(X,
DequantizeLinear(Wq, Ws, axis=<channel_axis>))`` chain fires with the
expected ``channel_axis`` (1 for plain MatMul, 0 for Gemm's ``transB=1``);
``Wq``/``Ws`` match an independent numpy re-implementation of
``QuantizeWeightPerChannelInPlace``'s formula; ``X`` is passed through
completely unchanged (``MatMul``'s first input is still literally the graph
input ``X``, the single most distinguishing behavioral difference from
``dynamic_quantize_matmul``); the actual runtime output stays within the
bound proved above (checked via onnxruntime, with ``Ws`` read directly from
the static initializer -- no ``DynamicQuantizeLinear`` runtime step to
expose here, unlike ``dynamic_quantize_matmul``'s own equivalent test); and a
pre-opset-13 model (``DequantizeLinear``'s per-channel ``axis`` attribute
needs opset >= 13) is declined outright.

The firing/bound tests below pass ``check_n=0`` to ``simplify_isolated_extra``
-- confirmed empirically, not assumed -- since this really is a lossy INT8
rewrite: onnxsim's own random-input equivalence check (its default
``check_n=3``, tolerance ``rtol=1e-4``/``atol=1e-5``) fails on the quantized
output at that tolerance, the same reasoning both prior files in this family
document for their own ``check_n=0``.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

_K = 2  # concrete number of contraction taps -- matches quantized_mac_bound's
# own _K; enough to exercise the cross-tap sum, and this shape of query has
# not shown the nonlinear-blowup risk larger _K hits elsewhere in this suite.


def _abs(v):
    return z3.If(v >= 0, v, -v)


def _bound_formulas():
    """Builds the Z3 vocabulary for the single-operand bounded-error claim:
    ``X`` has no error term at all (never quantized), only ``W`` does, via
    the free per-tap error variable ``ew`` (mirroring
    ``quantized_mac_bound``'s own ``ew``/``ex`` idiom, with no ``ex`` since
    there is nothing for it to model). Returns ``(float_matmul,
    dequant_matmul, rounding_bounds, bound)``.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # X[i, k], true float activation
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[k, n], true float weight
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[k, n] - Wdq[k, n]
    Ws = z3.Real("Ws")  # this pass's per-output-channel scale Ws[n]

    dequant_w = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        Ws > 0,
        *[_abs(ew[k]) <= Ws / 2 for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))

    bound = (Ws / 2) * sum(_abs(X[k]) for k in range(_K))

    return float_matmul, dequant_matmul, rounding_bounds, bound


def test_weight_only_quantize_matmul_error_is_bounded():
    # The genuine bounded-error claim: given W's own per-channel rounding
    # bound (|W[k, n] - Wdq[k, n]| <= Ws[n] / 2) and X completely unchanged,
    # the true float dot product and the one computed against the
    # dequantized weight cannot differ by more than (Ws[n] / 2) * sum_k
    # |X[i, k]| -- quantized_mac_bound's own bound with eps_x fixed to 0.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_weight_only_quantize_matmul_bias_variant_error_is_bounded():
    # The "+ Bias" branch (a Gemm with a bias input): this pass never
    # touches Gemm's bias C at all, so adding the same Bias(n) to both the
    # true and the dequantized computation leaves their difference -- and
    # therefore the bound on it -- unchanged; Bias cancels out of the error
    # term algebraically.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_matmul + bias) - (dequant_matmul + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_weight_only_quantize_matmul_negative_control_requires_rounding_bound():
    # Sanity check that the bound proved above is genuine, not vacuous: with
    # no error budget assumed on ew at all (only Ws > 0), the same bound is
    # not a theorem -- Z3 must find a real counterexample, confirming the
    # rounding bound on ew is load-bearing rather than the two computations
    # always agreeing (within `bound`) regardless.
    float_matmul, dequant_matmul, _rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul

    solver = z3.Solver()
    solver.add(z3.Not(z3.And(error <= bound, -error <= bound)))
    assert solver.check() == z3.sat, (
        "the bound holds even without any rounding-error budget on ew -- "
        "negative control is vacuous"
    )


def test_weight_only_quantize_matmul_dequantizelinear_axis_semantics_matches_rounding_bound():
    # New content this pass's proof needs that dynamic_quantize_matmul's
    # does not: dynamic_quantize_matmul folds its dequantization into a
    # plain Mul, never leaving a DequantizeLinear node to reason about; this
    # pass leaves an actual DequantizeLinear(Wq, Ws, axis=channel_axis) node.
    # Confirms that op's own defining per-channel-axis formula
    # (Wdq[k, n] = Wq[k, n] * Ws[n], implicit zero_point=0, for the axis
    # that selects the output-channel dimension) is exactly the quantity the
    # rounding bound above assumes -- i.e. that Wq being "some integer code
    # within 0.5 of W[k, n] / Ws[n]" (round-to-nearest, no saturation,
    # test_formal_verify_quantize_round_trip.py's own hypothesis shape) is
    # what makes DequantizeLinear's literal output satisfy
    # |W[k, n] - Wdq[k, n]| <= Ws[n] / 2 -- not assumed for free.
    w, ws = z3.Reals("w ws")
    wq = z3.Int("wq")  # round(w / ws): some integer within 0.5 of w / ws.
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        ws > 0,
        wq - w / ws <= half,
        w / ws - wq <= half,
    )
    # DequantizeLinear(Wq, Ws, axis=<channel axis>)'s own defining formula
    # for one element on that axis: zero_point implicit 0 (symmetric).
    wdq = z3.ToReal(wq) * ws

    error = w - wdq
    prove(z3.Implies(hypotheses, z3.And(error <= ws / 2, -error <= ws / 2)))


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


def _dequantizelinear_node(model, output_name):
    node = producer(model, output_name)
    assert node.op_type == "DequantizeLinear"
    return node


def _axis(node):
    return next(a.i for a in node.attribute if a.name == "axis")


def _quantize_weight_per_channel_in_place(weight, channel_axis):
    """Independent numpy re-implementation of
    ``QuantizeWeightPerChannelInPlace`` (quantize_matmul_common.h):
    per-``channel_axis`` symmetric INT8 quantization *in weight's own
    layout* (no transpose), scale = max(|channel|) / 127 (or 1.0 for an
    all-zero channel), codes = round(w / scale) clipped to [-127, 127].
    """
    reduce_axis = 1 - channel_axis
    scale = np.max(np.abs(weight), axis=reduce_axis)
    scale = np.where(scale > 0, scale / 127.0, 1.0).astype(np.float32)
    scale_bcast = np.expand_dims(scale, axis=reduce_axis)
    codes = np.clip(np.round(weight / scale_bcast), -127, 127).astype(np.int8)
    return codes, scale


def test_weight_only_quantize_matmul_pass_fires_and_matches_scheme():
    # Build a plain float MatMul and run the real weight_only_quantize_matmul
    # rewrite alone, then confirm both the node chain's shape (MatMul(X,
    # DequantizeLinear(Wq, Ws, axis=1))) and the quantized weight's exact
    # values.
    rng = np.random.default_rng(0)
    rows, K, N = 4, 5, 3
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

    # check_n=0: see the module docstring for why the built-in random-input
    # check (tolerance rtol=1e-4/atol=1e-5) does not apply to a genuinely
    # lossy INT8 rewrite.
    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_matmul", check_n=0
    )

    matmul_node = producer(sim_model, "Y")
    assert matmul_node.op_type == "MatMul"
    x_input, w_input = matmul_node.input
    # X (the activation) is passed through completely unchanged -- the
    # single most distinguishing behavioral difference from
    # dynamic_quantize_matmul, which replaces X with a DynamicQuantizeLinear
    # output.
    assert x_input == "X"

    dql_node = _dequantizelinear_node(sim_model, w_input)
    assert _axis(dql_node) == 1
    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in sim_model.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in sim_model.graph.initializer if i.name == ws_name)
    wq = numpy_helper.to_array(wq_init)
    ws = numpy_helper.to_array(ws_init)
    assert wq.shape == (K, N)
    assert ws.shape == (N,)

    expected_wq, expected_ws = _quantize_weight_per_channel_in_place(weight, 1)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_weight_only_quantize_matmul_gemm_transb_uses_channel_axis_zero():
    # PyTorch nn.Linear layout: weight [N, K], Gemm(X, W, B, transB=1). The
    # header comment's "axis 0 when Gemm's transB made W [N, K]" -- confirm
    # channel_axis differs from the plain-MatMul case above, and Wq/Ws match
    # the same independent numpy scheme with channel_axis=0.
    rng = np.random.default_rng(1)
    rows, K, N = 3, 4, 2
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

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_matmul", check_n=0
    )

    gemm_node = producer(sim_model, "Y")
    assert gemm_node.op_type == "Gemm"
    x_input, w_input, b_input = gemm_node.input
    assert x_input == "X"
    assert b_input == "B"  # bias untouched

    dql_node = _dequantizelinear_node(sim_model, w_input)
    assert _axis(dql_node) == 0
    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in sim_model.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in sim_model.graph.initializer if i.name == ws_name)
    wq = numpy_helper.to_array(wq_init)
    ws = numpy_helper.to_array(ws_init)
    assert wq.shape == (N, K)
    assert ws.shape == (N,)

    expected_wq, expected_ws = _quantize_weight_per_channel_in_place(weight, 0)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_weight_only_quantize_matmul_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring quantized_mac_bound's/dynamic_quantize_
    # matmul's own analogous tests: run the real quantized graph through
    # onnxruntime and confirm every output element's error against the true
    # float MatMul stays within the bound the proof above derives. Unlike
    # dynamic_quantize_matmul's version of this test, there is no runtime
    # DynamicQuantizeLinear step to expose -- Ws is a static initializer,
    # read directly.
    rng = np.random.default_rng(2)
    rows, K, N = 4, 6, 3
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

    quantized = onnxsim.quantize_weight_only(model)
    dql_node = next(n for n in quantized.graph.node if n.op_type == "DequantizeLinear")
    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in quantized.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in quantized.graph.initializer if i.name == ws_name)
    ws = numpy_helper.to_array(ws_init)
    assert numpy_helper.to_array(wq_init).shape == (K, N)
    assert ws.shape == (N,)

    sess = ort.InferenceSession(quantized.SerializeToString())
    (y_quant,) = sess.run(["Y"], {"X": x})

    y_float = x @ weight
    error = np.abs(y_float - y_quant)

    eps_w = ws / 2.0  # shape [N]
    bound = eps_w[np.newaxis, :] * np.abs(x).sum(axis=1)[:, np.newaxis]
    assert np.all(error <= bound + 1e-6)

    # Also confirm the rewrite is meaningfully close (not merely "within a
    # loose worst-case bound") for these well-scaled inputs -- consistent
    # with INT8 quantization, not bitwise equal. Mirrors
    # test_dynamic_quantize_matmul_output_is_close_to_float_within_proved_
    # bound's own reasoning for why atol is loosened beyond a pure-rtol
    # check.
    np.testing.assert_allclose(y_quant, y_float, rtol=0.05, atol=1e-2)


def test_weight_only_quantize_matmul_declines_pre_opset13():
    # DequantizeLinear's per-channel `axis` attribute needs opset >= 13
    # (patternMatchPredicate's first check). A plain MatMul at an older
    # opset must survive untouched.
    rng = np.random.default_rng(3)
    rows, K, N = 4, 5, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
        opset=12,
        ir_version=8,
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_matmul", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["MatMul"]
