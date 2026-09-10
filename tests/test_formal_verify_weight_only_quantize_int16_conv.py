"""Formal check for WeightOnlyQuantizeInt16Conv (opt-in; onnxsim's own
``onnxsim/passes/weight_only_quantize_int16_conv.h``): the Conv-layer sibling
of ``weight_only_quantize_int16_matmul.h``, itself the INT16 upgrade of
``weight_only_quantize_matmul.h`` -- related to ``weight_only_quantize_int16_
matmul.h`` exactly the way ``weight_only_quantize_conv.h``
(``test_formal_verify_weight_only_quantize_conv.py``, this file's primary
Conv-shaped template) relates to ``weight_only_quantize_matmul.h``. As this
file was being written, ``tests/test_formal_verify_weight_only_quantize_
int16_matmul.py`` (the MatMul/Gemm sibling) did not yet exist, so the
INT8-vs-INT16 scale-ratio corollary below is derived directly from
``weight_only_quantize_int16_conv.h`` and ``weight_only_quantize_conv.h``
rather than adapted from that file -- see ``test_formal_verify_static_
quantize_int16_conv.py``'s own analogous UINT8-vs-UINT16 corollary (read
alongside its own template pair, ``test_formal_verify_static_quantize_
conv.py``/``_matmul.py``) for the general shape this file's corollary
follows instead.

It rewrites ``Y = Conv(X, W)`` (optional bias, a third input, left
completely untouched) -- ``W`` a constant FLOAT32 tensor, rank >= 3
(``[Cout, Cin/groups, k...]``), ``X`` FLOAT32 -- into::

    Wq, Ws := per-output-channel symmetric INT16 quantization of W (computed
              ONCE, at pass-transform time, from W's static values --
              QuantizeConvWeightPerOutputChannelInt16, quantize_conv_common.h)
    Wdq = DequantizeLinear(Wq, Ws, axis=0)   # zero_point=0
    Y   = Conv(X, Wdq)

Needs opset >= 21 (INT16 QuantizeLinear/DequantizeLinear support), unlike the
INT8 template's opset >= 13.

The ONE structural difference from the MatMul/Gemm version, restated from the
Conv template's own docstring: Conv's weight layout ``[Cout, Cin/groups,
k...]`` ALWAYS puts the output channel on axis 0, so unlike Gemm (whose
``transB`` attribute picks between weight stored as ``[N, K]`` or ``[K, N]``)
there is no transposed-layout case here -- ``axis=0`` unconditionally.

Soundness claim
================
Exactly like the INT8 Conv template, this is a *single-operand* special case
of the same MAC bounded-error lemma ``test_formal_verify_quantized_mac_
bound.py`` establishes: ``X`` has NO error at all (``eps_x = 0``, since ``X``
is never quantized), so the general bound collapses, with ``eps_x := 0`` and
``eps_w := Ws[c] / 2`` (the standard ``DequantizeLinear(QuantizeLinear(...))``
round-trip bound), to::

    |Conv(X, W)[..., c, ...] - Conv(X, Wdq)[..., c, ...]|
        <= (Ws[c] / 2) * sum_k |X_patch[k]|

Modeling Conv's actual windowed-sum/im2col semantics in Z3 is not needed, for
the same reason the Conv template gives: once a spatial output position and
output channel ``c`` are fixed, that one output element is exactly a dot
product of the receptive-field patch of ``X`` (``Cin/groups * prod(kernel
dims)`` values, flattened) against ``W[c, ...]`` (flattened the same way) --
the "one output element is a linear combination of weight and input values"
shape ``quantized_mac_bound`` already models via a small concrete contraction
dimension ``_K``. ``_K`` below is therefore ``2``, matching ``quantized_mac_
bound``'s/the Conv template's own convention -- standing in for that one
output position's flattened receptive-field-times-weight contraction, not
for anything specific to convolution's sliding-window structure or to which
integer width (INT8 or INT16) produced a caller's particular ``Ws``: the core
bound query below is stated with ``Ws`` an entirely free ``Ws > 0`` real,
never given a defining formula in terms of ``max(|w|)`` at all, so its
hypotheses cannot possibly depend on which divisor (127 or 32767) produced
it. This file's own ``_bound_formulas`` reuses the Conv template's exact Z3
formulation (the direct-error-variable ``ew`` idiom -- see below), confirming
this by construction rather than merely asserting it.

Avoiding the reconstructed-dequantized-value shape
====================================================
As ``test_formal_verify_dynamic_quantize_matmul.py``'s own module docstring
documents in full, reconstructing a dequantized value from separate
quantized-code/scale multiplicands inside a bound-proving Z3 query has
repeatedly caused Z3 to hang past 90s in this suite. Every Z3 query below
uses the direct-error-variable (``ew``) idiom instead -- ``W[k] - Wdq[k]`` is
a free Real bounded directly by its own rounding hypothesis, exactly like
every other bound-proving file in this suite, including this file's own
template.

The genuinely new, pass-specific content
==========================================
``test_weight_only_quantize_int16_conv_scale_is_127_over_32767_times_int8_
scale`` / ``..._is_strictly_finer_than_int8_scale`` below: for the SAME
weight channel, this pass's INT16 scale is exactly ``127 / 32767`` times
``weight_only_quantize_conv``'s own INT8 scale, and strictly finer (smaller)
whenever the channel isn't all-zero -- both are literally
``max(|W[c, ...]|)`` (the same numerator) divided by a different, but fixed,
integer-range constant (32767 vs. 127) -- mirroring ``test_formal_verify_
static_quantize_int16_conv.py``'s own analogous UINT8-vs-UINT16 corollary for
the activation scale.

On top of the abstract scale-ratio corollary, ``test_weight_only_quantize_
int16_conv_preserves_typical_weights_an_outlier_would_destroy_in_int8`` below
is a concrete differential test built directly from this pass's own header
doc comment / ``onnxsim.quantize_weight_only_int16``'s own docstring
motivation: a channel with one extreme-outlier weight and several small
"typical" weights is exactly the case ``onnxsim.estimate_quantization_
precision`` flags (a channel's ``max(|w|) / median(|w|)`` ratio past 127) and
recommends INT16 for -- INT8's coarse step (``max(|w|) / 127``) rounds the
small values to within one step of zero (i.e. loses most of their relative
precision, or collapses several distinct small values to the same code or to
0 outright), while INT16's ~258x finer step preserves their relative
magnitudes meaningfully.

Conv's own optional bias is untouched by this pass -- a per-output-channel
additive term that cancels out of the error term algebraically, exactly like
the Conv template's own bias-variant test.

No consumer-composition step is added, matching this suite's usual reasoning
for a numeric *bound* (as opposed to an *equality*) claim: an arbitrary
consumer need not be Lipschitz, so a bound on
``|Conv(X, W) - Conv(X, Wdq)|`` does not in general bound anything about
``|consumer(Conv(X, W)) - consumer(Conv(X, Wdq))|``.

Differential tests build a plain float ``Conv`` (and a bias variant) via
``onnx.parser`` (per ``CLAUDE.md``), run the real pass alone via
``simplify_isolated_extra`` (and, for the numeric-bound check, via
:func:`onnxsim.quantize_weight_only_int16`), and confirm: the exact
``Conv(X, DequantizeLinear(Wq, Ws, axis=0))`` chain fires; ``Wq`` is truly
INT16 and matches an independent numpy re-implementation of
``QuantizeConvWeightPerOutputChannelInt16``'s formula; ``X`` and any bias are
passed through completely unchanged; ``axis=0`` unconditionally; the actual
runtime output stays within the bound proved above (checked via
onnxruntime, WITH graph optimization explicitly disabled -- see the
per-test comment for why); and a pre-opset-21 model is declined outright,
even at an opset where ``weight_only_quantize_conv`` itself would already
fire.

The firing/bound tests below pass ``check_n=0`` to ``simplify_isolated_extra``
-- confirmed empirically, not assumed -- since this really is a lossy INT16
rewrite: onnxsim's own random-input equivalence check (its default
``check_n=3``, tolerance ``rtol=1e-4``/``atol=1e-5``) fails on the quantized
output at that tolerance for some inputs, the same reasoning the Conv
template documents for its own (coarser) INT8 rewrite's ``check_n=0``.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

_K = 2  # concrete number of contraction taps -- matches quantized_mac_bound's
# own _K (and both the INT8 Conv template's and the general lemma's), standing
# in for one output position's flattened receptive-field-times-weight
# contraction (Cin/groups * prod(kernel dims)) in a real Conv; enough to
# exercise the cross-tap sum, and this shape of query has not shown the
# nonlinear-blowup risk larger _K hits elsewhere in this suite.


def _abs(v):
    return z3.If(v >= 0, v, -v)


def _bound_formulas():
    """Builds the Z3 vocabulary for the single-operand bounded-error claim:
    ``X`` has no error term at all (never quantized), only ``W`` does, via
    the free per-tap error variable ``ew`` (mirroring ``quantized_mac_
    bound``'s own ``ew``/``ex`` idiom, with no ``ex`` since there is nothing
    for it to model). Byte-for-byte identical in shape to the INT8 Conv
    template's own ``_bound_formulas`` -- ``Ws`` is left an entirely free
    ``Ws > 0`` real with no defining formula, which is exactly how this file
    confirms the bound's hypotheses don't depend on which divisor (127 or
    32767) produced a caller's particular ``Ws``: this query would remain
    sound for literally any positive ``Ws``. Returns ``(float_conv,
    dequant_conv, rounding_bounds, bound)``.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # one output position's
    # flattened receptive-field patch of X, true float values
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[c, ...] flattened, true
    # float weight for output channel c
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[c, k] - Wdq[c, k]
    Ws = z3.Real("Ws")  # this pass's per-output-channel INT16 scale Ws[c]

    dequant_w = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        Ws > 0,
        *[_abs(ew[k]) <= Ws / 2 for k in range(_K)],
    )

    float_conv = sum(X[k] * W[k] for k in range(_K))
    dequant_conv = sum(X[k] * dequant_w[k] for k in range(_K))

    bound = (Ws / 2) * sum(_abs(X[k]) for k in range(_K))

    return float_conv, dequant_conv, rounding_bounds, bound


def test_weight_only_quantize_int16_conv_error_is_bounded():
    # The genuine bounded-error claim: given W's own per-channel rounding
    # bound (|W[c, k] - Wdq[c, k]| <= Ws[c] / 2) and X completely unchanged,
    # the true float dot product (one output element's receptive-field
    # contraction) and the one computed against the dequantized weight
    # cannot differ by more than (Ws[c] / 2) * sum_k |X_patch[k]| --
    # quantized_mac_bound's own bound with eps_x fixed to 0. Identical in
    # shape to the INT8 Conv template's own version -- Ws is free, so this
    # holds regardless of which integer width produced it.
    float_conv, dequant_conv, rounding_bounds, bound = _bound_formulas()
    error = float_conv - dequant_conv
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_weight_only_quantize_int16_conv_bias_variant_error_is_bounded():
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


def test_weight_only_quantize_int16_conv_negative_control_requires_rounding_bound():
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


def test_weight_only_quantize_int16_conv_dequantizelinear_axis_semantics_matches_rounding_bound():
    # This pass leaves an actual DequantizeLinear(Wq, Ws, axis=0) node in the
    # graph (unlike a pass that folds dequantization into a plain Mul).
    # Confirms that op's own defining per-channel-axis formula (Wdq[c, ...] =
    # Wq[c, ...] * Ws[c], implicit zero_point=0, for the axis that selects
    # the output-channel dimension) is exactly the quantity the rounding
    # bound above assumes -- i.e. that Wq being "some integer code within 0.5
    # of W[c, k] / Ws[c]" (round-to-nearest, no saturation,
    # test_formal_verify_quantize_round_trip.py's own hypothesis shape) is
    # what makes DequantizeLinear's literal output satisfy
    # |W[c, k] - Wdq[c, k]| <= Ws[c] / 2 -- not assumed for free. This lemma
    # has nothing INT16-specific in it (`wq` is a free, unbounded-above
    # integer -- INT16's [-32767, 32767] saturation range never appears),
    # so it is identical in shape to the INT8 template's own version.
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


def test_weight_only_quantize_int16_conv_scale_is_127_over_32767_times_int8_scale():
    # The genuinely pass-specific corollary: for the SAME weight channel
    # (same max(|w|) numerator), this pass's INT16 scale and
    # weight_only_quantize_conv's own INT8 scale are related by an EXACT
    # ratio, 127 / 32767 -- both are literally the same numerator divided by
    # a different, but fixed, constant, so the ratio identity follows from
    # real-number algebra alone with no side condition at all (in particular
    # this holds even in the degenerate max_abs_w == 0 case -- both passes
    # substitute a scale of 1.0 there instead of dividing, a case the
    # strict-inequality corollary below excludes).
    max_abs_w = z3.Real("max_abs_w")
    Ws_8 = max_abs_w / 127
    Ws_16 = max_abs_w / 32767
    prove(Ws_16 * 32767 == Ws_8 * 127)


def test_weight_only_quantize_int16_conv_scale_is_strictly_finer_than_int8_scale():
    # This pass's whole point, as a checked theorem rather than folklore: for
    # any non-all-zero channel (max_abs_w > 0 -- QuantizeConvWeightPerOutput
    # ChannelInt16's/QuantizeConvWeightPerOutputChannel's shared "scale = 1.0
    # if max_abs_w == 0" substitution means this hypothesis always holds
    # whenever a real division actually occurs), the INT16 scale is STRICTLY
    # smaller (finer) than the INT8 scale for the identical channel -- i.e.
    # this pass's round-trip error budget (Ws / 2) is strictly tighter than
    # weight_only_quantize_conv's own, for every weight value in that
    # channel, not merely on average.
    max_abs_w = z3.Real("max_abs_w")
    Ws_8 = max_abs_w / 127
    Ws_16 = max_abs_w / 32767
    prove(z3.Implies(max_abs_w > 0, Ws_16 < Ws_8))


def _model(body, initializer=(), opset=21, ir_version=10):
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


def _quantize_conv_weight_per_output_channel_int16(weight):
    """Independent numpy re-implementation of
    ``QuantizeConvWeightPerOutputChannelInt16`` (quantize_conv_common.h):
    per-output-channel (axis 0) symmetric INT16 quantization of a Conv
    weight ``[Cout, Cin/groups, k...]`` -- scale = max(|W[c, ...]|) / 32767
    (or 1.0 for an all-zero channel), codes = round(W[c, ...] / scale)
    clipped to [-32767, 32767], reducing over every axis but 0. Same
    formula/reduction axes as the INT8 template's own re-implementation,
    just with 32767 instead of 127.
    """
    reduce_axes = tuple(range(1, weight.ndim))
    scale = np.max(np.abs(weight), axis=reduce_axes)
    scale = np.where(scale > 0, scale / 32767.0, 1.0).astype(np.float32)
    scale_bcast = scale.reshape((-1,) + (1,) * (weight.ndim - 1))
    codes = np.clip(np.round(weight / scale_bcast), -32767, 32767).astype(np.int16)
    return codes, scale


def test_weight_only_quantize_int16_conv_pass_fires_and_matches_scheme():
    # Build a plain float Conv and run the real weight_only_quantize_int16_
    # conv rewrite alone, then confirm both the node chain's shape (Conv(X,
    # DequantizeLinear(Wq, Ws, axis=0))) and the quantized weight's exact
    # values -- and that Wq is truly INT16, not INT8.
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
    # check (tolerance rtol=1e-4/atol=1e-5) does not apply to this genuinely
    # lossy INT16 rewrite.
    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int16_conv", check_n=0
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
    assert wq.dtype == np.int16, "quantized weight must be INT16, not INT8"
    assert wq.shape == (cout, cin, kh, kw)
    assert ws.shape == (cout,)

    expected_wq, expected_ws = _quantize_conv_weight_per_output_channel_int16(weight)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_weight_only_quantize_int16_conv_bias_is_untouched():
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
        model, "weight_only_quantize_int16_conv", check_n=0
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
    assert wq.dtype == np.int16
    assert wq.shape == (cout, cin, kh, kw)
    assert ws.shape == (cout,)

    expected_wq, expected_ws = _quantize_conv_weight_per_output_channel_int16(weight)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_weight_only_quantize_int16_conv_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring the INT8 Conv template's own analogous
    # test: run the real quantized graph through onnxruntime and confirm
    # every output element's error against the true float Conv stays within
    # the bound the proof above derives. Ws is a static initializer, read
    # directly -- no runtime quantization step on the activation side to
    # expose.
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

    quantized = onnxsim.quantize_weight_only_int16(model)
    dql_node = next(n for n in quantized.graph.node if n.op_type == "DequantizeLinear")
    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in quantized.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in quantized.graph.initializer if i.name == ws_name)
    ws = numpy_helper.to_array(ws_init)
    wq = numpy_helper.to_array(wq_init)
    assert wq.dtype == np.int16
    assert wq.shape == (cout, cin, kh, kw)
    assert ws.shape == (cout,)

    # Graph optimization explicitly disabled: onnxruntime's default
    # optimization level can fuse certain quantized-graph shapes into a
    # hardware-specific fused kernel -- a DIFFERENT code path than the
    # literal node chain this pass's proof reasons about. See
    # `tests/test_ort_matmul_nbits_workaround.py`'s docstring for this
    # suite's precedent of a real ORT graph-optimization fusion bug of
    # exactly this shape, and the fix applied across the
    # test_formal_verify_static_quantize_conv.py family. Disabling
    # optimization executes the graph exactly as the pass produced it.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        quantized.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
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
    # with INT16 quantization's much finer step, tighter than the INT8
    # template's own tolerance.
    np.testing.assert_allclose(y_quant, y_float, rtol=1e-3, atol=1e-3)


def test_weight_only_quantize_int16_conv_preserves_typical_weights_an_outlier_would_destroy_in_int8():
    # The concrete, non-abstract version of this pass's whole reason for
    # existing: a channel with one extreme outlier and several small
    # "typical" weights, exactly the case onnxsim.estimate_quantization_
    # precision flags (max(|w|) / median(|w|) ratio past 127) and
    # onnxsim.quantize_weight_only_int16's own docstring names as the
    # motivating scenario for this pass. Built as a single-output-channel,
    # 1x1-kernel Conv (Cout=1) so the whole flattened weight vector IS the
    # one channel -- no cross-channel scale dilution to reason about.
    cout, cin, kh, kw = 1, 16, 1, 1
    hw = 3
    typical = np.full((cin - 1,), 0.05, dtype=np.float32)  # 15 "typical" taps
    outlier = np.float32(100.0)  # one extreme outlier
    weight = np.concatenate([[outlier], typical]).reshape(cout, cin, kh, kw)

    # INT8 scale for this channel: max(|w|) / 127 = 100 / 127 =~ 0.787 -- more
    # than 15x bigger than the typical weight (0.05) itself, so INT8 rounds
    # every typical weight to 0 outright.
    scale8 = np.float32(outlier) / 127.0
    code8_typical = np.round(typical / scale8)
    assert np.all(code8_typical == 0), (
        "test setup: INT8 must actually lose the typical weights to 0 for "
        "this comparison to be meaningful"
    )

    # INT16 scale for the identical channel: max(|w|) / 32767 =~ 0.00305 --
    # small enough that each typical weight rounds to a nonzero code whose
    # relative error against the true value is small.
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X) => (float[1,{cout},{hw},{hw}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int16_conv", check_n=0
    )
    conv_node = producer(sim_model, "Y")
    dql_node = _dequantizelinear_node(sim_model, conv_node.input[1])
    wq_name, ws_name = dql_node.input
    wq = numpy_helper.to_array(
        next(i for i in sim_model.graph.initializer if i.name == wq_name)
    )
    ws = numpy_helper.to_array(
        next(i for i in sim_model.graph.initializer if i.name == ws_name)
    )
    assert wq.dtype == np.int16
    np.testing.assert_allclose(float(ws[0]), float(outlier / 32767.0), rtol=1e-6)

    wq_typical = wq.reshape(-1)[1:]
    # Not lost to 0: every typical weight's INT16 code is nonzero (unlike the
    # INT8 codes computed above, which are all exactly 0).
    assert np.all(wq_typical != 0), (
        "INT16 must preserve the typical weights as nonzero codes"
    )

    # Each typical weight's *relative* reconstruction error stays small under
    # INT16 (the INT8 reconstruction of the SAME weight is 100% relative
    # error -- it rounds to exactly 0 -- since 0.05 is well under half of
    # INT8's own scale8 =~ 0.787 step).
    wdq_typical = wq_typical.astype(np.float64) * float(ws[0])
    relative_error16 = np.abs(
        wdq_typical - typical.astype(np.float64)
    ) / typical.astype(np.float64)
    assert np.all(relative_error16 < 0.1), (
        "INT16 should preserve each typical weight's relative magnitude to "
        "within 10%, unlike INT8's total loss (rounds to 0) for this channel"
    )


def test_weight_only_quantize_int16_conv_declines_pre_opset21_even_though_int8_would_fire():
    # This pass's own opset floor (INT16 QuantizeLinear/DequantizeLinear
    # needs opset >= 21) is HIGHER than weight_only_quantize_conv's (opset
    # >= 13): build the model at opset 13 -- a version at which
    # weight_only_quantize_conv itself would already fire (see
    # test_formal_verify_weight_only_quantize_conv.py's own equivalent
    # negative test, which uses opset 12 for ITS floor) -- and confirm THIS
    # pass still declines and leaves the graph completely untouched.
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
        opset=13,
        ir_version=8,
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int16_conv", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["Conv"]


def test_weight_only_quantize_int16_conv_declines_pre_opset21():
    # Same opset-floor check as above, restated at opset 20 -- one below this
    # pass's own floor -- to directly exercise the ">= 21" boundary itself.
    rng = np.random.default_rng(4)
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
        opset=20,
        ir_version=9,
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int16_conv", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["Conv"]
