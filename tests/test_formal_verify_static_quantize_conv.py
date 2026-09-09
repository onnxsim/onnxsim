"""Formal check for StaticQuantizeConv (opt-in; onnxsim's own
``onnxsim/passes/static_quantize_conv.h``): the Conv-layer sibling of
``static_quantize_matmul.h`` (``test_formal_verify_static_quantize_matmul.py``,
this file's direct template) -- same QDQ format, same calibrated asymmetric
activation quantization, same per-output-channel symmetric weight
quantization. It rewrites ``Y = Conv(X, W)`` (optional bias, a third input,
left completely untouched) -- ``W`` a constant FLOAT32 tensor, rank >= 3
(``[Cout, Cin/groups, k...]``), ``X`` FLOAT32 -- into QDQ format, leaving the
Conv node itself untouched and rewiring only its inputs::

    Xq  = QuantizeLinear(X, Xs, Xzp)          -- Xs/Xzp: CALIBRATED, fixed
    Xdq = DequantizeLinear(Xq, Xs, Xzp)
    Wdq = DequantizeLinear(Wq, Ws, axis=0)    -- symmetric, zp=0
    Y   = Conv(Xdq, Wdq)

The ONE structural difference from the MatMul/Gemm version -- exactly the
same simplification ``weight_only_quantize_conv.h`` makes relative to its own
MatMul sibling: Conv's weight layout ``[Cout, Cin/groups, k...]`` ALWAYS puts
the output channel on axis 0, so unlike Gemm (whose ``transB`` attribute
picks between weight stored as ``[N, K]`` or ``[K, N]``) there is no
transposed-layout case here -- ``axis=0`` unconditionally. Every other
structural point below, including the shared calibration-range global
(``StaticQuantizationCalibrationRanges``, defined in
``static_quantize_matmul.h`` and included by this pass's own header), is
identical to the template file's own reasoning, restated in Conv's
vocabulary.

Soundness claim
================
Exactly like ``static_quantize_matmul``, and unlike ``weight_only_quantize_
conv`` (whose bound collapses to a single-operand special case, ``eps_x :=
0``, since ``X`` is never quantized there), this pass quantizes BOTH
operands -- ``quantized_mac_bound``'s own general two-operand shape, with
``eps_x := Xs / 2`` and ``eps_w := Ws[c] / 2`` both genuinely nonzero::

    |Conv(X, W)[..., c, ...] - Conv(Xdq, Wdq)[..., c, ...]| <=
        (Xs / 2) * sum_k |W[c, k]| + (Ws[c] / 2) * sum_k |X_patch[k]|
        + K * (Xs / 2) * (Ws[c] / 2)

Modeling Conv's actual windowed-sum/im2col semantics in Z3 is not needed, any
more than the MatMul version needed to model MatMul's own indexing beyond a
concrete contraction dimension: once a spatial output position and output
channel ``c`` are fixed, that one output element is exactly a dot product of
the receptive-field patch of ``X`` (``Cin/groups * prod(kernel dims)``
values, flattened) against ``W[c, ...]`` (flattened the same way) -- i.e.
precisely the "one output element is a linear combination of weight and
input values" shape ``quantized_mac_bound`` already models via a small
concrete contraction dimension ``_K``. So, exactly as in the MatMul template
file and in ``weight_only_quantize_conv``'s own analogous docstring, ``_K``
below stands in for that one output position's receptive-field-times-weight
contraction (``Cin/groups * prod(kernel dims)`` in a real Conv), not for
anything specific to convolution's sliding-window structure. The proof
reuses the MatMul template's exact Z3 formulation -- the direct-error-
variable (``ex``/``ew``) idiom, built from the start to avoid the known
nonlinear-blowup risk (see that file's own docstring for the investigation
that established this) -- with no changes needed to the vocabulary itself,
only to the surrounding prose.

Carrying the nonzero zero-point through the derivation
========================================================
As in the MatMul template, the per-element hypothesis the bound proof needs
is ``X``'s own round-trip bound, ``|X[i, k] - Xdq[i, k]| <= Xs / 2``. This
file reproves that round-trip lemma here, in its own ``Xs``/``Xzp``
vocabulary, with ``Xzp`` a genuinely free (and, in the differential tests
below, genuinely nonzero) variable -- the algebra shows exactly why the bound
doesn't care: dequantizing subtracts back out exactly the zero-point that
quantizing added, for ANY zero-point value, as long as nothing clips. That
"as long as nothing clips" matters -- the same explicit side condition
``test_formal_verify_quantize_round_trip.py`` calls out for its own lemma --
and, as in the MatMul template, the differential tests below calibrate from
the actual data range being tested (so nothing clips there) rather than
asserting it away.

No consumer-composition step is added here, matching this suite's usual
reasoning for a numeric *bound* (as opposed to an *equality*) claim: an
arbitrary consumer need not be Lipschitz, so a bound on
``|Conv(X, W) - Conv(Xdq, Wdq)|`` does not in general bound anything about
``|consumer(Conv(X, W)) - consumer(Conv(Xdq, Wdq))|``.

Invoking the pass with a specific calibrated range
====================================================
As the MatMul template file's own docstring explains in full: the actual
nanobind-exposed entry point ``onnxsim.onnxsim_cpp2py_export.quantize_static
(model_bytes, activation_ranges)`` takes a caller-supplied ``{tensor name:
(min, max)}`` dict directly and runs ``OptimizeFixed`` with exactly
``["static_quantize_matmul", "static_quantize_conv"]`` -- i.e. it already
isolates this pass family (confirmed here to include ``static_quantize_
conv`` via ``onnxsim.onnxsim_cpp2py_export._list_other_optimizers()``) with
no extra ``skipped_optimizers``/``simplify_isolated_extra`` machinery, and
applies the rewrite with no shape inference or other simplification
alongside it. The differential tests below call it directly, exactly like
the MatMul template does, and check the proved bound directly against
onnxruntime's own execution of the rewritten graph rather than onnxsim's own
random-input equivalence check.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
import onnxsim.onnxsim_cpp2py_export as C
from _formal_verify_common import producer, prove, z3
from onnx import parser

assert "static_quantize_conv" in C._list_other_optimizers()

_K = 2  # concrete number of contraction taps -- see module docstring for why
# this matches quantized_mac_bound's/static_quantize_matmul's own _K rather
# than a larger value.


def _abs(v):
    return z3.If(v >= 0, v, -v)


def test_static_quantize_conv_activation_round_trip_holds_for_any_zero_point():
    # Carries a genuinely free (including nonzero) Xzp through the standard
    # QuantizeLinear/DequantizeLinear round-trip derivation, in this file's
    # own Xs/Xzp vocabulary -- a property of QuantizeLinear/DequantizeLinear
    # themselves, not of Conv vs. MatMul, so this is near-identical to the
    # MatMul template's own version of this lemma. `n` models
    # QuantizeLinear's round(x / Xs + Xzp): *some* integer within 0.5 of it
    # (any correct rounding rule, not just one specific tie-break), and the
    # "no clipping" side condition is: `n` itself (not some separately-
    # clamped code) is what DequantizeLinear reads back -- i.e.
    # round(x / Xs + Xzp) already lands in [0, 255] without needing to be
    # clamped there.
    x, Xs, Xzp = z3.Reals("x Xs Xzp")
    n = z3.Int("n")  # round(x / Xs + Xzp), not yet known to be an integer code
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        Xs > 0,
        n - (x / Xs + Xzp) <= half,
        (x / Xs + Xzp) - n <= half,
    )
    # Not clipped: the quantized code is exactly `n` (no saturation), so
    # dequantizing is Xs * (n - Xzp) -- Xzp cancels exactly regardless of its
    # value, positive, negative, or zero.
    xdq = (n - Xzp) * Xs

    error = x - xdq
    prove(z3.Implies(hypotheses, z3.And(error <= Xs / 2, -error <= Xs / 2)))


def _bound_formulas():
    """Builds the Z3 vocabulary for the genuine two-operand bounded-error
    claim -- the MatMul template's exact formulation, restated in Conv's
    vocabulary: ``X``/``W`` stand for one output position's flattened
    receptive-field patch and the corresponding output channel's flattened
    weight, and each operand's dequantization ERROR is a free Real bounded
    directly by its own rounding hypothesis, rather than being re-derived
    from separate quantized-code/scale/zero-point multiplicands. As in the
    MatMul template, no separate ring-identity lemma is needed to connect
    this to the pass's own node chain -- ``Y = Conv(Xdq, Wdq)`` IS exactly
    the elementwise-dequantized dot product ``quantized_conv`` below, for the
    one output element this contraction models, with no integer-accumulator
    rescale step in between.

    Returns ``(float_conv, quantized_conv, rounding_bounds, bound)``.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # one output position's
    # flattened receptive-field patch of X, true float values
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[c, ...] flattened, true
    # float weight for output channel c
    ex = [z3.Real(f"ex{k}") for k in range(_K)]  # X[k] - Xdq[k]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[k] - Wdq[k]
    Xs = z3.Real("Xs")  # calibrated per-tensor activation scale
    Ws = z3.Real("Ws")  # this pass's per-output-channel weight scale Ws[c]

    Xdq = [X[k] - ex[k] for k in range(_K)]
    Wdq = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        Xs > 0,
        Ws > 0,
        *[_abs(ex[k]) <= Xs / 2 for k in range(_K)],
        *[_abs(ew[k]) <= Ws / 2 for k in range(_K)],
    )

    float_conv = sum(X[k] * W[k] for k in range(_K))
    quantized_conv = sum(Xdq[k] * Wdq[k] for k in range(_K))  # Conv(Xdq, Wdq)'s
    # one output element, this contraction's receptive-field-times-weight dot
    # product

    bound = (
        (Xs / 2) * sum(_abs(W[k]) for k in range(_K))
        + (Ws / 2) * sum(_abs(X[k]) for k in range(_K))
        + _K * (Xs / 2) * (Ws / 2)
    )

    return float_conv, quantized_conv, rounding_bounds, bound


def test_static_quantize_conv_error_is_bounded():
    # The genuine bounded-error claim, a direct instantiation of
    # quantized_mac_bound's own lemma with both eps_x := Xs / 2 and
    # eps_w := Ws / 2 nonzero (unlike weight_only_quantize_conv's
    # single-operand eps_x := 0 special case): given each operand's own
    # rounding bound, the true float dot product (one output element's
    # receptive-field contraction) and Conv(Xdq, Wdq) -- the pass's actual
    # literal output -- cannot differ by more than `bound`.
    float_conv, quantized_conv, rounding_bounds, bound = _bound_formulas()
    error = float_conv - quantized_conv
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_static_quantize_conv_bias_variant_error_is_bounded():
    # Conv's own optional bias (a third input, a per-output-channel additive
    # term the pass never touches): adding the same Bias(c) to both the true
    # and the quantized computation leaves their difference -- and therefore
    # the bound on it -- unchanged, since Bias cancels out of the error term
    # algebraically.
    float_conv, quantized_conv, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_conv + bias) - (quantized_conv + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_static_quantize_conv_negative_control_requires_rounding_bounds():
    # Sanity check that the bound proved above is genuine, not vacuous:
    # without ANY error budget (no rounding-error hypothesis at all), the
    # exact-equality claim float_conv == quantized_conv is not a theorem --
    # Z3 must find a real counterexample, confirming the rounding really
    # does introduce error rather than the two computations always
    # coinciding regardless.
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ex = [z3.Real(f"ex{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    Xdq = [X[k] - ex[k] for k in range(_K)]
    Wdq = [W[k] - ew[k] for k in range(_K)]
    float_conv = sum(X[k] * W[k] for k in range(_K))
    quantized_conv = sum(Xdq[k] * Wdq[k] for k in range(_K))

    solver = z3.Solver()
    solver.add(z3.Not(float_conv == quantized_conv))
    assert solver.check() == z3.sat, (
        "the float and quantized computations always agree even without "
        "any rounding-error budget -- negative control is vacuous"
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


def _quantize_static(model, activation_ranges):
    """Invokes the real compiled pass directly via the nanobind-exposed
    ``quantize_static(model_bytes, activation_ranges)`` -- see the module
    docstring for why this, rather than
    ``onnxsim.calibration.quantize_static`` or ``simplify_isolated_extra``,
    is used here.
    """
    out = onnx.ModelProto()
    out.ParseFromString(C.quantize_static(model.SerializeToString(), activation_ranges))
    return out


def _quantize_conv_weight_per_output_channel(weight):
    """Independent numpy re-implementation of
    ``QuantizeConvWeightPerOutputChannel`` (quantize_conv_common.h):
    per-output-channel (axis 0) symmetric INT8 quantization of a Conv weight
    ``[Cout, Cin/groups, k...]`` -- scale = max(|W[c, ...]|) / 127 (or 1.0
    for an all-zero channel), codes = round(W[c, ...] / scale) clipped to
    [-127, 127], reducing over every axis but 0. Same helper
    ``test_formal_verify_weight_only_quantize_conv.py`` already validates
    this formula with.
    """
    reduce_axes = tuple(range(1, weight.ndim))
    scale = np.max(np.abs(weight), axis=reduce_axes)
    scale = np.where(scale > 0, scale / 127.0, 1.0).astype(np.float32)
    scale_bcast = scale.reshape((-1,) + (1,) * (weight.ndim - 1))
    codes = np.clip(np.round(weight / scale_bcast), -127, 127).astype(np.int8)
    return codes, scale


def _expected_asymmetric_uint8_quant_params(min_val, max_val):
    """Independent re-implementation of ``ComputeAsymmetricUint8QuantParams``
    (static_quantize_matmul.h, shared by this pass), in float32 to match the
    pass's own arithmetic precision. Same helper
    ``test_formal_verify_static_quantize_matmul.py`` already validates this
    formula with.
    """
    lo = min(np.float32(0.0), np.float32(min_val))
    hi = max(np.float32(0.0), np.float32(max_val))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zero_point = int(np.clip(np.round(-lo / scale), 0, 255))
    return np.float32(scale), zero_point


def _axis(node):
    return next(a.i for a in node.attribute if a.name == "axis")


def test_static_quantize_conv_pass_fires_and_matches_scheme():
    # Build a plain float Conv and run the real pass with a calibration
    # range chosen so the zero-point comes out genuinely NONZERO -- min=-5,
    # max=10 straddles 0, so lo=-5 (not widened) and
    # zero_point = round(5 / scale) = 85 != 0. This is the single most
    # important differential check for the asymmetric-quantization content
    # this pass introduces that weight_only_quantize_conv doesn't have.
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

    quantized = _quantize_static(model, {"X": (-5.0, 10.0)})

    # Walk the chain backward from the real graph output: Conv (untouched
    # node) <- DequantizeLinear(Xq) <- QuantizeLinear(X), and
    # Conv <- DequantizeLinear(Wq, Ws, axis=0).
    conv_node = producer(quantized, "Y")
    assert conv_node.op_type == "Conv"
    xdq_name, wdq_name = conv_node.input

    xdq_node = producer(quantized, xdq_name)
    assert xdq_node.op_type == "DequantizeLinear"
    xq_node = producer(quantized, xdq_node.input[0])
    assert xq_node.op_type == "QuantizeLinear"
    assert xq_node.input[0] == "X"
    # QuantizeLinear and its matching DequantizeLinear share the same
    # (scale, zero_point) initializers -- the pass builds exactly one of
    # each and feeds both nodes from them.
    assert xq_node.input[1:] == xdq_node.input[1:]

    wdq_node = producer(quantized, wdq_name)
    assert wdq_node.op_type == "DequantizeLinear"
    assert len(wdq_node.input) == 2  # symmetric: no zero_point input at all
    assert _axis(wdq_node) == 0  # Conv: axis 0 unconditionally, no transposed case

    init = {i.name: i for i in quantized.graph.initializer}
    x_scale = numpy_helper.to_array(init[xq_node.input[1]])
    x_zp = numpy_helper.to_array(init[xq_node.input[2]])
    expected_scale, expected_zp = _expected_asymmetric_uint8_quant_params(-5.0, 10.0)
    assert x_zp.dtype == np.uint8
    assert int(x_zp) == expected_zp
    assert expected_zp != 0, "test calibration range must exercise a nonzero zero-point"
    np.testing.assert_allclose(float(x_scale), float(expected_scale), rtol=1e-6)

    wq = numpy_helper.to_array(init[wdq_node.input[0]])
    ws = numpy_helper.to_array(init[wdq_node.input[1]])
    expected_wq, expected_ws = _quantize_conv_weight_per_output_channel(weight)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_static_quantize_conv_bias_is_untouched():
    # Conv's own optional bias input (input 2): confirm it survives the
    # rewrite completely unchanged -- only X and W are ever replaced, exactly
    # like the pass's own doc comment states.
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

    quantized = _quantize_static(model, {"X": (-5.0, 10.0)})

    conv_node = producer(quantized, "Y")
    assert conv_node.op_type == "Conv"
    xdq_name, wdq_name, b_input = conv_node.input
    assert b_input == "B"  # bias untouched, still fed directly by name

    xdq_node = producer(quantized, xdq_name)
    assert xdq_node.op_type == "DequantizeLinear"
    xq_node = producer(quantized, xdq_node.input[0])
    assert xq_node.op_type == "QuantizeLinear"
    assert xq_node.input[0] == "X"

    wdq_node = producer(quantized, wdq_name)
    assert wdq_node.op_type == "DequantizeLinear"
    assert _axis(wdq_node) == 0

    init = {i.name: i for i in quantized.graph.initializer}
    wq = numpy_helper.to_array(init[wdq_node.input[0]])
    ws = numpy_helper.to_array(init[wdq_node.input[1]])
    expected_wq, expected_ws = _quantize_conv_weight_per_output_channel(weight)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_static_quantize_conv_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring the MatMul template's/weight_only_
    # quantize_conv's own analogous tests: run the real quantized graph
    # through onnxruntime and confirm every output element's error against
    # the true float Conv stays within the bound proved above. The
    # calibration range is set to X's own actual (min, max) so nothing
    # clips -- the round-trip lemma's explicit side condition.
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

    x_min, x_max = float(x.min()), float(x.max())
    quantized = _quantize_static(model, {"X": (x_min, x_max)})

    sess = ort.InferenceSession(quantized.SerializeToString())
    (y_quant,) = sess.run(None, {"X": x})

    x_scale, _x_zp = _expected_asymmetric_uint8_quant_params(x_min, x_max)
    _wq, ws = _quantize_conv_weight_per_output_channel(weight)
    eps_x = float(x_scale) / 2.0
    eps_w = ws / 2.0  # shape [cout]
    K = cin * kh * kw  # this concrete model's actual per-position contraction depth

    # True float Conv output and the per-element bound, computed independently
    # via a brute-force sliding-window sum (no padding, stride 1 -- matches
    # the model above).
    y_float = np.zeros((1, cout, oh, ow), dtype=np.float64)
    bound = np.zeros((1, cout, oh, ow), dtype=np.float64)
    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                patch = x[0, :, i : i + kh, j : j + kw].astype(np.float64)
                w_c = weight[c].astype(np.float64)
                y_float[0, c, i, j] = np.sum(patch * w_c)
                bound[0, c, i, j] = (
                    eps_x * np.sum(np.abs(w_c))
                    + eps_w[c] * np.sum(np.abs(patch))
                    + K * eps_x * eps_w[c]
                )

    error = np.abs(y_float - y_quant)
    assert np.all(error <= bound + 1e-6)

    # Also confirm the rewrite is meaningfully close (not merely "within a
    # loose worst-case bound") for these well-scaled inputs -- consistent
    # with combined UINT8 (activation) + INT8 (weight) quantization, not
    # bitwise equal. atol is loosened further than either template file's own
    # (0.01 for MatMul's activation+weight case, 0.05 for Conv's weight-only
    # case): this test combines BOTH Conv's larger per-output-element
    # contraction depth (cin * kh * kw = 18 here) AND two-operand
    # (activation + weight) quantization error, accumulating more error per
    # element than either template alone. The actual rigorous check is the
    # proved worst-case bound above, already asserted.
    np.testing.assert_allclose(y_quant, y_float, rtol=0.05, atol=0.2)


def test_static_quantize_conv_declines_without_calibrated_range():
    # patternMatchPredicate's calibration-range check: an activation tensor
    # with no entry in activation_ranges is left completely alone.
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
    )

    quantized = _quantize_static(model, {})
    assert [n.op_type for n in quantized.graph.node] == ["Conv"]


def test_static_quantize_conv_declines_pre_opset13():
    # DequantizeLinear's per-channel `axis` attribute needs opset >= 13
    # (patternMatchPredicate's opset check). A plain Conv at an older opset
    # must survive untouched even with a calibrated range available.
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
        opset=12,
        ir_version=8,
    )

    quantized = _quantize_static(model, {"X": (-5.0, 10.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Conv"]
