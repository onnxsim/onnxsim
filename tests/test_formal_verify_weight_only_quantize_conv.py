"""Formal check for WeightOnlyQuantizeConv (opt-in; onnxsim's own
``onnxsim/passes/weight_only_quantize_conv.h``): the Conv-layer sibling of
``weight_only_quantize_matmul.h``
(``test_formal_verify_weight_only_quantize_matmul.py``, this file's direct
template) -- same "only the constant weight is quantized, the activation is
left completely untouched" design, no ``DynamicQuantizeLinear``/
``QuantizeLinear``/``DequantizeLinear`` on the activation side, no
calibration data of any kind. It rewrites ``Y = Conv(X, W)`` (optional bias,
a third input, left completely untouched) -- ``W`` a constant FLOAT32
tensor, rank >= 3 (``[Cout, Cin/groups, k...]``), ``X`` FLOAT32 -- into::

    Wq, Ws := per-output-channel symmetric INT8 quantization of W (computed
              ONCE, at pass-transform time, from W's static values --
              QuantizeConvWeightPerOutputChannel, quantize_conv_common.h)
    Wdq = DequantizeLinear(Wq, Ws, axis=0)   # zero_point=0
    Y   = Conv(X, Wdq)

The ONE structural difference from the MatMul/Gemm version: Conv's weight
layout ``[Cout, Cin/groups, k...]`` ALWAYS puts the output channel on axis 0,
so unlike Gemm (whose ``transB`` attribute picks between weight stored as
``[N, K]`` or ``[K, N]``, giving ``weight_only_quantize_matmul.h`` a
``channel_axis`` choice to make) there is no transposed-layout case here --
``axis=0`` unconditionally. Every other structural point below is identical
to the template file's own reasoning, restated in Conv's vocabulary.

This is a *single-operand* special case of the same MAC bounded-error lemma
``test_formal_verify_quantized_mac_bound.py`` establishes: here ``X`` has NO
error at all (``eps_x = 0`` in ``quantized_mac_bound``'s own vocabulary,
since ``X`` is never quantized), so ``quantized_mac_bound``'s general bound::

    eps_x * sum_k |W[k, n]| + eps_w * sum_k |X[i, k]| + K * eps_w * eps_x

collapses, with ``eps_x := 0`` and ``eps_w := Ws[c] / 2`` (the standard
``DequantizeLinear(QuantizeLinear(...))`` round-trip bound
``test_formal_verify_quantize_round_trip.py`` proves), to just::

    |Conv(X, W)[..., c, ...] - Conv(X, Wdq)[..., c, ...]|
        <= (Ws[c] / 2) * sum_k |X_patch[k]|

-- no cross term (``K * eps_w * eps_x`` with ``eps_x = 0``) and no
``Xs``-driven term (``X`` is never quantized).

Modeling Conv's actual windowed-sum/im2col semantics in Z3 is not needed, any
more than the MatMul version needed to model MatMul's own indexing beyond a
concrete contraction dimension: once a spatial output position and output
channel ``c`` are fixed, that one output element is exactly a dot product of
the receptive-field patch of ``X`` (``Cin/groups * prod(kernel dims)``
values, flattened) against ``W[c, ...]`` (flattened the same way) -- i.e.
precisely the "one output element is a linear combination of weight and
input values" shape ``quantized_mac_bound`` already models via a small
concrete contraction dimension ``_K``. So, exactly as in the template file,
``_K`` below stands in for that one output position's receptive-field-times-
weight contraction (``Cin/groups * prod(kernel dims)`` in a real Conv),
not for anything specific to convolution's sliding-window structure -- the
proof reuses the template file's exact Z3 formulation (the direct-error-
variable ``ew`` idiom, no ``Wq``/``Ws``-multiplicand reconstruction), with no
nonlinear-blowup risk expected here either, for the same reason: there is no
combined nonlinear chain of two quantized operands to hit (``X`` never gets a
quantized/scale representation to multiply against in the first place).

A short separate test, near-identical to the template's own version, confirms
``DequantizeLinear``'s own defining per-channel-axis formula
(``Wdq[c, ...] = Wq[c, ...] * Ws[c]`` for the axis that selects the output
channel, implicit ``zero_point = 0``) is exactly the quantity the rounding
bound above assumes -- i.e. that treating ``Wdq`` as "some per-channel
dequantization satisfying the round-trip bound" (what the main proof does,
via the free ``ew`` error variable) is not begging the question.

A negative-control test confirms the bound is not vacuous: with no rounding-
error budget on ``ew`` at all, the same claim is not a theorem -- Z3 finds a
real counterexample.

No consumer-composition step is added, matching the template file's own
reasoning: an arbitrary ``consumer`` need not be Lipschitz, so a numeric
*bound* (not an equality) implies nothing in general about
``|consumer(a) - consumer(b)|``.

Differential tests build a plain float ``Conv`` (and a bias variant) via
``onnx.parser`` (per ``CLAUDE.md``), run the real pass alone via
``simplify_isolated_extra`` (and, for the numeric-bound check, via
:func:`onnxsim.quantize_weight_only`, which applies this rewrite among
others -- see its own docstring), and confirm: the exact ``Conv(X,
DequantizeLinear(Wq, Ws, axis=0))`` chain fires; ``Wq``/``Ws`` match an
independent numpy re-implementation of
``QuantizeConvWeightPerOutputChannel``'s formula (reducing over axes 1, 2, 3
of a ``[Cout, Cin, kH, kW]`` weight, per output channel); ``X`` and any bias
are passed through completely unchanged -- Conv's OTHER inputs untouched, the
same distinguishing check the template file makes for MatMul's ``X``; the
actual runtime output stays within the bound proved above (checked via
onnxruntime, with ``Ws`` read directly from the static initializer); and a
pre-opset-13 model (``DequantizeLinear``'s per-channel ``axis`` attribute
needs opset >= 13) is declined outright.

The firing/bound tests below pass ``check_n=0`` to ``simplify_isolated_extra``
-- confirmed empirically, not assumed -- since this really is a lossy INT8
rewrite: onnxsim's own random-input equivalence check (its default
``check_n=3``, tolerance ``rtol=1e-4``/``atol=1e-5``) fails on the quantized
output at that tolerance, the same reasoning the template file documents for
its own ``check_n=0``.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

_K = 2  # concrete number of contraction taps -- matches quantized_mac_bound's
# own _K (and the MatMul template's), standing in for one output position's
# flattened receptive-field-times-weight contraction
# (Cin/groups * prod(kernel dims)) in a real Conv; enough to exercise the
# cross-tap sum, and this shape of query has not shown the nonlinear-blowup
# risk larger _K hits elsewhere in this suite.


def _abs(v):
    return z3.If(v >= 0, v, -v)


def _bound_formulas():
    """Builds the Z3 vocabulary for the single-operand bounded-error claim:
    ``X`` has no error term at all (never quantized), only ``W`` does, via
    the free per-tap error variable ``ew`` (mirroring
    ``quantized_mac_bound``'s own ``ew``/``ex`` idiom, with no ``ex`` since
    there is nothing for it to model). Returns ``(float_conv, dequant_conv,
    rounding_bounds, bound)``.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # one output position's
    # flattened receptive-field patch of X, true float values
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[c, ...] flattened, true
    # float weight for output channel c
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[c, k] - Wdq[c, k]
    Ws = z3.Real("Ws")  # this pass's per-output-channel scale Ws[c]

    dequant_w = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        Ws > 0,
        *[_abs(ew[k]) <= Ws / 2 for k in range(_K)],
    )

    float_conv = sum(X[k] * W[k] for k in range(_K))
    dequant_conv = sum(X[k] * dequant_w[k] for k in range(_K))

    bound = (Ws / 2) * sum(_abs(X[k]) for k in range(_K))

    return float_conv, dequant_conv, rounding_bounds, bound


def test_weight_only_quantize_conv_error_is_bounded():
    # The genuine bounded-error claim: given W's own per-channel rounding
    # bound (|W[c, k] - Wdq[c, k]| <= Ws[c] / 2) and X completely unchanged,
    # the true float dot product (one output element's receptive-field
    # contraction) and the one computed against the dequantized weight
    # cannot differ by more than (Ws[c] / 2) * sum_k |X_patch[k]| --
    # quantized_mac_bound's own bound with eps_x fixed to 0.
    float_conv, dequant_conv, rounding_bounds, bound = _bound_formulas()
    error = float_conv - dequant_conv
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_weight_only_quantize_conv_bias_variant_error_is_bounded():
    # Conv's own optional bias (a third input, a per-output-channel additive
    # term the pass never touches): adding the same Bias(c) to both the true
    # and the dequantized computation leaves their difference -- and
    # therefore the bound on it -- unchanged; Bias cancels out of the error
    # term algebraically.
    float_conv, dequant_conv, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_conv + bias) - (dequant_conv + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_weight_only_quantize_conv_negative_control_requires_rounding_bound():
    # Sanity check that the bound proved above is genuine, not vacuous: with
    # no error budget assumed on ew at all (only Ws > 0), the same bound is
    # not a theorem -- Z3 must find a real counterexample, confirming the
    # rounding bound on ew is load-bearing rather than the two computations
    # always agreeing (within `bound`) regardless.
    float_conv, dequant_conv, _rounding_bounds, bound = _bound_formulas()
    error = float_conv - dequant_conv

    solver = z3.Solver()
    solver.add(z3.Not(z3.And(error <= bound, -error <= bound)))
    assert solver.check() == z3.sat, (
        "the bound holds even without any rounding-error budget on ew -- "
        "negative control is vacuous"
    )


def test_weight_only_quantize_conv_dequantizelinear_axis_semantics_matches_rounding_bound():
    # This pass leaves an actual DequantizeLinear(Wq, Ws, axis=0) node in the
    # graph (unlike a pass that folds dequantization into a plain Mul).
    # Confirms that op's own defining per-channel-axis formula (Wdq[c, ...] =
    # Wq[c, ...] * Ws[c], implicit zero_point=0, for the axis that selects
    # the output-channel dimension) is exactly the quantity the rounding
    # bound above assumes -- i.e. that Wq being "some integer code within 0.5
    # of W[c, k] / Ws[c]" (round-to-nearest, no saturation,
    # test_formal_verify_quantize_round_trip.py's own hypothesis shape) is
    # what makes DequantizeLinear's literal output satisfy
    # |W[c, k] - Wdq[c, k]| <= Ws[c] / 2 -- not assumed for free.
    w, ws = z3.Reals("w ws")
    wq = z3.Int("wq")  # round(w / ws): some integer within 0.5 of w / ws.
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        ws > 0,
        wq - w / ws <= half,
        w / ws - wq <= half,
    )
    # DequantizeLinear(Wq, Ws, axis=0)'s own defining formula for one element
    # on the output-channel axis: zero_point implicit 0 (symmetric).
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


def _quantize_conv_weight_per_output_channel(weight):
    """Independent numpy re-implementation of
    ``QuantizeConvWeightPerOutputChannel`` (quantize_conv_common.h):
    per-output-channel (axis 0) symmetric INT8 quantization of a Conv weight
    ``[Cout, Cin/groups, k...]`` -- scale = max(|W[c, ...]|) / 127 (or 1.0
    for an all-zero channel), codes = round(W[c, ...] / scale) clipped to
    [-127, 127], reducing over every axis but 0.
    """
    reduce_axes = tuple(range(1, weight.ndim))
    scale = np.max(np.abs(weight), axis=reduce_axes)
    scale = np.where(scale > 0, scale / 127.0, 1.0).astype(np.float32)
    scale_bcast = scale.reshape((-1,) + (1,) * (weight.ndim - 1))
    codes = np.clip(np.round(weight / scale_bcast), -127, 127).astype(np.int8)
    return codes, scale


def test_weight_only_quantize_conv_pass_fires_and_matches_scheme():
    # Build a plain float Conv and run the real weight_only_quantize_conv
    # rewrite alone, then confirm both the node chain's shape (Conv(X,
    # DequantizeLinear(Wq, Ws, axis=0))) and the quantized weight's exact
    # values.
    rng = np.random.default_rng(0)
    cout, cin, kh, kw = 2, 1, 3, 3
    hw = 4
    weight = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X) => (float[1,{cout},{hw - kh + 1},{hw - kw + 1}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    # check_n=0: see the module docstring for why the built-in random-input
    # check (tolerance rtol=1e-4/atol=1e-5) does not apply to a genuinely
    # lossy INT8 rewrite.
    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_conv", check_n=0
    )

    conv_node = producer(sim_model, "Y")
    assert conv_node.op_type == "Conv"
    x_input, w_input = conv_node.input
    # X (the activation) is passed through completely unchanged -- the
    # single most distinguishing behavioral difference from a full
    # activation-quantizing rewrite.
    assert x_input == "X"

    dql_node = _dequantizelinear_node(sim_model, w_input)
    assert _axis(dql_node) == 0
    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in sim_model.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in sim_model.graph.initializer if i.name == ws_name)
    wq = numpy_helper.to_array(wq_init)
    ws = numpy_helper.to_array(ws_init)
    assert wq.shape == (cout, cin, kh, kw)
    assert ws.shape == (cout,)

    expected_wq, expected_ws = _quantize_conv_weight_per_output_channel(weight)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_weight_only_quantize_conv_bias_is_untouched():
    # Conv's own optional bias input (input 2): confirm it survives the
    # rewrite completely unchanged, exactly like X -- only the weight input
    # is ever replaced.
    rng = np.random.default_rng(1)
    cout, cin, kh, kw = 2, 1, 3, 3
    hw = 4
    weight = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.5
    bias = rng.standard_normal(cout).astype(np.float32)
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X) => (float[1,{cout},{hw - kh + 1},{hw - kw + 1}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_conv", check_n=0
    )

    conv_node = producer(sim_model, "Y")
    assert conv_node.op_type == "Conv"
    x_input, w_input, b_input = conv_node.input
    assert x_input == "X"
    assert b_input == "B"  # bias untouched

    dql_node = _dequantizelinear_node(sim_model, w_input)
    assert _axis(dql_node) == 0
    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in sim_model.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in sim_model.graph.initializer if i.name == ws_name)
    wq = numpy_helper.to_array(wq_init)
    ws = numpy_helper.to_array(ws_init)
    assert wq.shape == (cout, cin, kh, kw)
    assert ws.shape == (cout,)

    expected_wq, expected_ws = _quantize_conv_weight_per_output_channel(weight)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_weight_only_quantize_conv_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring quantized_mac_bound's/weight_only_
    # quantize_matmul's own analogous tests: run the real quantized graph
    # through onnxruntime and confirm every output element's error against
    # the true float Conv stays within the bound the proof above derives.
    # Ws is a static initializer, read directly -- no runtime quantization
    # step on the activation side to expose.
    rng = np.random.default_rng(2)
    cout, cin, kh, kw = 3, 2, 3, 3
    hw = 6
    oh, ow = hw - kh + 1, hw - kw + 1
    weight = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.8
    x = rng.standard_normal((1, cin, hw, hw)).astype(np.float32) * 2.0
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X) => (float[1,{cout},{oh},{ow}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W)
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
    assert numpy_helper.to_array(wq_init).shape == (cout, cin, kh, kw)
    assert ws.shape == (cout,)

    sess = ort.InferenceSession(quantized.SerializeToString())
    (y_quant,) = sess.run(["Y"], {"X": x})

    # True float Conv output, computed independently via a brute-force
    # sliding-window sum (no padding, stride 1 -- matches the model above).
    y_float = np.zeros((1, cout, oh, ow), dtype=np.float64)
    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                patch = x[0, :, i : i + kh, j : j + kw].astype(np.float64)
                y_float[0, c, i, j] = np.sum(patch * weight[c].astype(np.float64))

    error = np.abs(y_float - y_quant)

    eps_w = ws / 2.0  # shape [cout]
    # For each output channel c and spatial position, the bound is
    # (Ws[c] / 2) * sum_k |X_patch[k]| over that position's receptive field.
    bound = np.zeros((1, cout, oh, ow), dtype=np.float64)
    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                patch = x[0, :, i : i + kh, j : j + kw].astype(np.float64)
                bound[0, c, i, j] = eps_w[c] * np.sum(np.abs(patch))

    assert np.all(error <= bound + 1e-6)

    # Also confirm the rewrite is meaningfully close (not merely "within a
    # loose worst-case bound") for these well-scaled inputs -- consistent
    # with INT8 quantization, not bitwise equal. Mirrors the MatMul
    # template's own reasoning for why atol is loosened beyond a pure-rtol
    # check; Conv's larger per-output-element contraction depth
    # (cin * kh * kw = 18 here, vs. the MatMul template's K = 6) accumulates
    # more quantization error per element, so atol is loosened a bit further
    # than the template's own 1e-2.
    np.testing.assert_allclose(y_quant, y_float, rtol=0.05, atol=5e-2)


def test_weight_only_quantize_conv_declines_pre_opset13():
    # DequantizeLinear's per-channel `axis` attribute needs opset >= 13
    # (patternMatchPredicate's first check). A plain Conv at an older opset
    # must survive untouched.
    rng = np.random.default_rng(3)
    cout, cin, kh, kw = 2, 1, 3, 3
    hw = 4
    weight = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X) => (float[1,{cout},{hw - kh + 1},{hw - kw + 1}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W)
        }}
        """,
        [_f32(weight, "W")],
        opset=12,
        ir_version=8,
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_conv", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["Conv"]
