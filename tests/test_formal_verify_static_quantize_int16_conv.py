"""Formal check for StaticQuantizeInt16Conv (opt-in; onnxsim's own
``onnxsim/passes/static_quantize_int16_conv.h``): the Conv-layer sibling of
``static_quantize_int16_matmul.h``, itself the "W8A16" variant of
``static_quantize_matmul.h`` -- weight stays INT8 per-output-channel
symmetric, but the activation widens to UINT16 (instead of UINT8) via
``ComputeAsymmetricUint16QuantParams``, needing opset >= 21 (UINT16
QuantizeLinear/DequantizeLinear support) rather than ``static_quantize_conv.h``'s
opset >= 13. It rewrites ``Y = Conv(X, W)`` (optional bias, a third input,
left completely untouched) -- ``W`` a constant FLOAT32 tensor, rank >= 3
(``[Cout, Cin/groups, k...]``), ``X`` FLOAT32 -- into QDQ format, leaving the
Conv node itself untouched and rewiring only its inputs::

    Xq  = QuantizeLinear(X, Xs, Xzp)          -- Xs/Xzp: CALIBRATED, fixed, uint16
    Xdq = DequantizeLinear(Xq, Xs, Xzp)
    Wdq = DequantizeLinear(Wq, Ws, axis=0)    -- symmetric, zp=0, int8
    Y   = Conv(Xdq, Wdq)

The ONE structural difference from ``static_quantize_int16_matmul.h`` --
exactly the same simplification ``static_quantize_conv.h`` makes relative to
``static_quantize_matmul.h``: Conv's weight layout ``[Cout, Cin/groups,
k...]`` ALWAYS puts the output channel on axis 0, so unlike Gemm (whose
``transB`` attribute picks between weight stored as ``[N, K]`` or ``[K, N]``)
there is no transposed-layout case here -- ``axis=0`` unconditionally. This
file is this suite's Conv-shaped adaptation of
``test_formal_verify_static_quantize_int16_matmul.py`` in exactly the way
``test_formal_verify_static_quantize_conv.py`` is the Conv-shaped adaptation
of ``test_formal_verify_static_quantize_matmul.py`` -- if that MatMul-sibling
test file exists alongside this one, its general activation-scale
bound/corollary content is this file's primary template; either way, the
Z3 content below is derived directly, following ``test_formal_verify_static_
quantize_conv.py``'s own bound machinery.

Soundness claim
================
Exactly like ``static_quantize_conv`` and ``static_quantize_int16_matmul``,
and unlike ``weight_only_quantize_conv`` (whose bound collapses to a
single-operand special case, ``eps_x := 0``, since ``X`` is never quantized
there), this pass quantizes BOTH operands -- ``quantized_mac_bound``'s own
general two-operand shape, with ``eps_x := Xs / 2`` and ``eps_w := Ws[c] / 2``
both genuinely nonzero::

    |Conv(X, W)[..., c, ...] - Conv(Xdq, Wdq)[..., c, ...]| <=
        (Xs / 2) * sum_k |W[c, k]| + (Ws[c] / 2) * sum_k |X_patch[k]|
        + K * (Xs / 2) * (Ws[c] / 2)

with ``Xs`` here this pass's own (finer) calibrated UINT16 activation scale,
``(hi - lo) / 65535`` rather than ``static_quantize_conv``'s ``(hi - lo) /
255``. Modeling Conv's actual windowed-sum/im2col semantics in Z3 is not
needed, exactly as ``test_formal_verify_static_quantize_conv.py``'s own
docstring explains: once a spatial output position and output channel ``c``
are fixed, that one output element is exactly a dot product of the
receptive-field patch of ``X`` (``Cin/groups * prod(kernel dims)`` values,
flattened) against ``W[c, ...]`` (flattened the same way) -- i.e. precisely
the "one output element is a linear combination of weight and input values"
shape ``quantized_mac_bound`` already models via a small concrete contraction
dimension ``_K``.

The general bound proof (``quantized_mac_bound``'s own lemma, and this file's
``test_static_quantize_int16_conv_error_is_bounded``) is stated with ``Xs``
as an entirely free ``Xs > 0`` real -- it is never given a defining formula
in terms of ``hi``/``lo`` at all, so its hypotheses cannot possibly depend on
*which* divisor (255 or 65535) produced a caller's particular ``Xs``. The
core bound query therefore needs no new nonlinear content versus
``static_quantize_conv``'s own version -- confirmed directly below by reusing
that exact same query, unedited apart from renaming, rather than merely
asserted.

The genuinely new, pass-specific content is the UINT16-vs-UINT8 scale-ratio
corollary (``test_static_quantize_int16_conv_uint16_scale_is_255_over_65535_
times_uint8_scale`` / ``..._is_strictly_finer_than_uint8_scale`` below):
for the SAME calibrated range, this pass's UINT16 scale is exactly
``255 / 65535`` times ``static_quantize_conv``'s own UINT8 scale, and is
strictly finer (smaller) whenever the range is non-degenerate -- mirroring
how ``test_formal_verify_quantize_fp16.py``/``_bf16.py``/``_fp8.py`` each add
their own format's worst-case relative-precision corollary on top of a
shared general round-to-nearest bound.

Carrying the nonzero zero-point through the derivation
========================================================
As in the Conv template, the per-element hypothesis the bound proof needs is
``X``'s own round-trip bound, ``|X[i, k] - Xdq[i, k]| <= Xs / 2``. This file
reproves that round-trip lemma here, with ``Xzp`` a genuinely free (and, in
the differential tests below, genuinely nonzero) variable -- the algebra
shows exactly why the bound doesn't care: dequantizing subtracts back out
exactly the zero-point that quantizing added, for ANY zero-point value, as
long as nothing clips. That "as long as nothing clips" matters -- the same
explicit side condition ``test_formal_verify_quantize_round_trip.py`` calls
out for its own lemma -- and, as in the Conv template, the differential tests
below calibrate from the actual data range being tested (so nothing clips
there) rather than asserting it away.

No consumer-composition step is added here, matching this suite's usual
reasoning for a numeric *bound* (as opposed to an *equality*) claim: an
arbitrary consumer need not be Lipschitz, so a bound on
``|Conv(X, W) - Conv(Xdq, Wdq)|`` does not in general bound anything about
``|consumer(Conv(X, W)) - consumer(Conv(Xdq, Wdq))|``.

Avoiding the reconstructed-dequantized-value shape
====================================================
As ``test_formal_verify_dynamic_quantize_matmul.py``'s own module docstring
documents in full, reconstructing a dequantized value from separate
quantized-code/zero-point multiplicands inside a bound-proving Z3 query has
repeatedly caused Z3 to hang past 90s in this suite. Every Z3 query below
uses the direct-error-variable (``ex``/``ew``) idiom instead -- each
operand's dequantization error is a free Real bounded directly by its own
rounding hypothesis -- exactly like every other bound-proving file in this
suite, including this file's own two templates.

Invoking the pass with a specific calibrated range
====================================================
As ``test_formal_verify_static_quantize_conv.py``'s own docstring explains in
full for ``quantize_static``: the actual nanobind-exposed entry point for
THIS pass family is ``onnxsim.onnxsim_cpp2py_export.quantize_static_int16
(model_bytes, activation_ranges)`` (see ``QuantizeStaticInt16`` in
``onnxsim/quantize_entry.cpp``), which takes a caller-supplied ``{tensor
name: (min, max)}`` dict directly and runs ``OptimizeFixed`` with exactly
``["static_quantize_int16_matmul", "static_quantize_int16_conv"]`` -- i.e. it
already isolates this pass family (confirmed here to include
``static_quantize_int16_conv`` via ``onnxsim.onnxsim_cpp2py_export.
_list_other_optimizers()``) with no extra ``skipped_optimizers``/
``simplify_isolated_extra`` machinery, and applies the rewrite with no shape
inference or other simplification alongside it. The differential tests below
call it directly, exactly like the Conv template does with
``quantize_static``, and check the proved bound directly against
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

assert "static_quantize_int16_conv" in C._list_other_optimizers()

_K = 2  # concrete number of contraction taps -- see module docstring for why
# this matches quantized_mac_bound's/static_quantize_conv's own _K rather
# than a larger value.


def _abs(v):
    return z3.If(v >= 0, v, -v)


def test_static_quantize_int16_conv_activation_round_trip_holds_for_any_zero_point():
    # Carries a genuinely free (including nonzero) Xzp through the standard
    # QuantizeLinear/DequantizeLinear round-trip derivation. This lemma is a
    # property of QuantizeLinear/DequantizeLinear themselves -- not of Conv
    # vs. MatMul, nor of UINT8 vs. UINT16 -- so it is identical in shape to
    # both templates' own version, just with the "not clipped" side condition
    # implicitly meaning "lands in [0, 65535]" instead of "[0, 255]" (the
    # actual numeric bound 65535 never appears in the Z3 query itself: `n` is
    # left as a free, unbounded-above integer, exactly as both templates
    # leave it).
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
    claim -- unchanged from ``static_quantize_conv``'s own version: ``X``/
    ``W`` stand for one output position's flattened receptive-field patch and
    the corresponding output channel's flattened weight, and each operand's
    dequantization ERROR is a free Real bounded directly by its own rounding
    hypothesis, rather than being re-derived from separate quantized-code/
    scale/zero-point multiplicands. ``Xs`` is left as a free ``Xs > 0`` real
    with no defining formula -- this is exactly how the module docstring's
    claim that the bound's hypotheses don't depend on Xs's divisor (255 vs.
    65535) is confirmed: this query is byte-for-byte identical in shape to
    the UINT8 pass's own, and would remain so for literally any positive
    Xs. As in both template files, no separate ring-identity lemma is needed
    to connect this to the pass's own node chain -- ``Y = Conv(Xdq, Wdq)`` IS
    exactly the elementwise-dequantized dot product ``quantized_conv`` below,
    for the one output element this contraction models, with no integer-
    accumulator rescale step in between.

    Returns ``(float_conv, quantized_conv, rounding_bounds, bound)``.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # one output position's
    # flattened receptive-field patch of X, true float values
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[c, ...] flattened, true
    # float weight for output channel c
    ex = [z3.Real(f"ex{k}") for k in range(_K)]  # X[k] - Xdq[k]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[k] - Wdq[k]
    Xs = z3.Real("Xs")  # this pass's own calibrated per-tensor UINT16
    # activation scale -- free, with no defining formula: see the docstring
    # above for why the bound proof cannot depend on Xs's divisor.
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


def test_static_quantize_int16_conv_error_is_bounded():
    # The genuine bounded-error claim, a direct instantiation of
    # quantized_mac_bound's own lemma with both eps_x := Xs / 2 and
    # eps_w := Ws / 2 nonzero: given each operand's own rounding bound, the
    # true float dot product (one output element's receptive-field
    # contraction) and Conv(Xdq, Wdq) -- the pass's actual literal output --
    # cannot differ by more than `bound`. Xs stands for THIS pass's finer
    # UINT16 scale, but as the docstring/`_bound_formulas` explain, the query
    # itself is identical to static_quantize_conv's UINT8 version.
    float_conv, quantized_conv, rounding_bounds, bound = _bound_formulas()
    error = float_conv - quantized_conv
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_static_quantize_int16_conv_bias_variant_error_is_bounded():
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


def test_static_quantize_int16_conv_negative_control_requires_rounding_bounds():
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


def test_static_quantize_int16_conv_uint16_scale_is_255_over_65535_times_uint8_scale():
    # The genuinely pass-specific corollary: for the SAME calibrated
    # [lo, hi] range (post zero-widening, i.e. lo <= 0 <= hi already holds,
    # exactly as ComputeAsymmetricUint8QuantParams/ComputeAsymmetricUint16
    # QuantParams both produce before dividing), this pass's UINT16 scale
    # and static_quantize_conv's own UINT8 scale are related by an EXACT
    # ratio, 255 / 65535 -- both are literally the same numerator (hi - lo)
    # divided by a different, but fixed, constant, so the ratio identity
    # follows from real-number algebra alone with no side condition at all
    # (in particular this holds even in the degenerate hi == lo case, unlike
    # the strict-inequality corollary below).
    hi, lo = z3.Reals("hi lo")
    Xs_8 = (hi - lo) / 255
    Xs_16 = (hi - lo) / 65535
    prove(Xs_16 * 65535 == Xs_8 * 255)


def test_static_quantize_int16_conv_uint16_scale_is_strictly_finer_than_uint8_scale():
    # This pass's whole point, as a checked theorem rather than folklore:
    # for any non-degenerate calibrated range (hi > lo -- guaranteed by
    # ComputeAsymmetricUint8QuantParams's/ComputeAsymmetricUint16QuantParams's
    # shared "hi = lo + 1 if hi <= lo" widening step, so this hypothesis
    # always holds for either pass's actual computed (lo, hi)), the UINT16
    # activation scale is STRICTLY smaller (finer) than the UINT8 scale for
    # the identical range -- i.e. this pass's round-trip error budget
    # (Xs / 2) is strictly tighter than static_quantize_conv's own, for every
    # activation value, not merely on average.
    hi, lo = z3.Reals("hi lo")
    Xs_8 = (hi - lo) / 255
    Xs_16 = (hi - lo) / 65535
    prove(z3.Implies(hi > lo, Xs_16 < Xs_8))


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


def _quantize_static_int16(model, activation_ranges):
    """Invokes the real compiled pass directly via the nanobind-exposed
    ``quantize_static_int16(model_bytes, activation_ranges)`` -- see the
    module docstring for why this, rather than
    ``onnxsim.quantize_static_int16`` or ``simplify_isolated_extra``, is used
    here.
    """
    out = onnx.ModelProto()
    out.ParseFromString(
        C.quantize_static_int16(model.SerializeToString(), activation_ranges)
    )
    return out


def _quantize_conv_weight_per_output_channel(weight):
    """Independent numpy re-implementation of
    ``QuantizeConvWeightPerOutputChannel`` (quantize_conv_common.h):
    per-output-channel (axis 0) symmetric INT8 quantization of a Conv weight
    ``[Cout, Cin/groups, k...]`` -- scale = max(|W[c, ...]|) / 127 (or 1.0
    for an all-zero channel), codes = round(W[c, ...] / scale) clipped to
    [-127, 127], reducing over every axis but 0. Same formula
    ``test_formal_verify_static_quantize_conv.py`` and
    ``test_formal_verify_weight_only_quantize_conv.py`` already validate --
    this pass's weight scheme is bit-identical to ``static_quantize_conv``'s
    own, unaffected by the activation-side UINT8-vs-UINT16 change.
    """
    reduce_axes = tuple(range(1, weight.ndim))
    scale = np.max(np.abs(weight), axis=reduce_axes)
    scale = np.where(scale > 0, scale / 127.0, 1.0).astype(np.float32)
    scale_bcast = scale.reshape((-1,) + (1,) * (weight.ndim - 1))
    codes = np.clip(np.round(weight / scale_bcast), -127, 127).astype(np.int8)
    return codes, scale


def _expected_asymmetric_uint16_quant_params(min_val, max_val):
    """Independent re-implementation of ``ComputeAsymmetricUint16QuantParams``
    (static_quantize_matmul.h, shared by this pass), in float32 to match the
    pass's own arithmetic precision."""
    lo = min(np.float32(0.0), np.float32(min_val))
    hi = max(np.float32(0.0), np.float32(max_val))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(65535.0)
    zero_point = int(np.clip(np.round(-lo / scale), 0, 65535))
    return np.float32(scale), zero_point


def _expected_asymmetric_uint8_quant_params(min_val, max_val):
    """Independent re-implementation of ``ComputeAsymmetricUint8QuantParams``
    -- static_quantize_conv's own scheme -- used below only to compute the
    concrete "what static_quantize_conv would have done for this exact case"
    numeric comparison, not to check this pass's own output."""
    lo = min(np.float32(0.0), np.float32(min_val))
    hi = max(np.float32(0.0), np.float32(max_val))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zero_point = int(np.clip(np.round(-lo / scale), 0, 255))
    return np.float32(scale), zero_point


def _axis(node):
    return next(a.i for a in node.attribute if a.name == "axis")


def test_static_quantize_int16_conv_pass_fires_and_matches_scheme():
    # Build a plain float Conv and run the real pass with a calibration
    # range chosen so the zero-point comes out genuinely NONZERO -- min=-5,
    # max=10 straddles 0, so lo=-5 (not widened) and
    # zero_point = round(5 / scale) != 0. This is the single most important
    # differential check for the asymmetric-quantization content this pass
    # introduces beyond weight-only quantization, and confirms the activation
    # is UINT16 (not UINT8) -- the one genuine behavioral difference from
    # static_quantize_conv.
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

    quantized = _quantize_static_int16(model, {"X": (-5.0, 10.0)})

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
    expected_scale, expected_zp = _expected_asymmetric_uint16_quant_params(-5.0, 10.0)
    assert x_zp.dtype == np.uint16, "activation zero_point must be UINT16, not UINT8"
    assert int(x_zp) == expected_zp
    assert expected_zp != 0, "test calibration range must exercise a nonzero zero-point"
    np.testing.assert_allclose(float(x_scale), float(expected_scale), rtol=1e-6)

    xq_value_info = next(
        vi for vi in quantized.graph.value_info if vi.name == xq_node.output[0]
    )
    assert xq_value_info.type.tensor_type.elem_type == onnx.TensorProto.UINT16, (
        "quantized activation tensor must be UINT16, not UINT8"
    )

    wq = numpy_helper.to_array(init[wdq_node.input[0]])
    ws = numpy_helper.to_array(init[wdq_node.input[1]])
    expected_wq, expected_ws = _quantize_conv_weight_per_output_channel(weight)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)
    assert wq.dtype == np.int8  # weight scheme unaffected by the activation change


def test_static_quantize_int16_conv_bias_is_untouched():
    # Conv's own optional bias input (input 2): confirm it survives the
    # rewrite completely unchanged -- only X and W are ever replaced, exactly
    # like the pass's own doc comment states (unlike qoperator_quantize_conv,
    # which does rewrite a Conv's bias into a quantized form -- this pass
    # never touches Conv's bias at all).
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

    quantized = _quantize_static_int16(model, {"X": (-5.0, 10.0)})

    conv_node = producer(quantized, "Y")
    assert conv_node.op_type == "Conv"
    xdq_name, wdq_name, b_input = conv_node.input
    assert b_input == "B"  # bias untouched, still fed directly by name

    # The bias initializer itself, byte for byte, is also untouched.
    init = {i.name: i for i in quantized.graph.initializer}
    np.testing.assert_array_equal(numpy_helper.to_array(init["B"]), bias)

    xdq_node = producer(quantized, xdq_name)
    assert xdq_node.op_type == "DequantizeLinear"
    xq_node = producer(quantized, xdq_node.input[0])
    assert xq_node.op_type == "QuantizeLinear"
    assert xq_node.input[0] == "X"

    wdq_node = producer(quantized, wdq_name)
    assert wdq_node.op_type == "DequantizeLinear"
    assert _axis(wdq_node) == 0

    wq = numpy_helper.to_array(init[wdq_node.input[0]])
    ws = numpy_helper.to_array(init[wdq_node.input[1]])
    expected_wq, expected_ws = _quantize_conv_weight_per_output_channel(weight)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_static_quantize_int16_conv_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring the Conv template's own analogous test:
    # run the real quantized graph through onnxruntime and confirm every
    # output element's error against the true float Conv stays within the
    # bound proved above (with this pass's own, finer, UINT16 eps_x). The
    # calibration range is set to X's own actual (min, max) so nothing
    # clips -- the round-trip lemma's explicit side condition. On top of
    # that, this also computes what static_quantize_conv's own UINT8 bound
    # (and actual UINT8 output) would have been for the IDENTICAL inputs/
    # weight/range, and confirms this pass's actual error is meaningfully
    # tighter.
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
    quantized = _quantize_static_int16(model, {"X": (x_min, x_max)})

    # Graph optimization disabled: onnxruntime can apply hardware-specific
    # graph transforms to a quantized Conv graph like this one, a different
    # code path than the literal node chain this pass's proof reasons about
    # -- see `tests/test_ort_matmul_nbits_workaround.py`'s docstring for this
    # suite's existing precedent of a real ORT graph-optimization fusion bug
    # of exactly this shape. Disabling optimization executes the graph
    # exactly as the pass produced it.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        quantized.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    (y_quant,) = sess.run(None, {"X": x})

    x_scale16, _x_zp16 = _expected_asymmetric_uint16_quant_params(x_min, x_max)
    _wq, ws = _quantize_conv_weight_per_output_channel(weight)
    eps_x16 = float(x_scale16) / 2.0
    eps_w = ws / 2.0  # shape [cout]
    K = cin * kh * kw  # this concrete model's actual per-position contraction depth

    # True float Conv output and this pass's per-element bound, computed
    # independently via a brute-force sliding-window sum (no padding,
    # stride 1 -- matches the model above).
    y_float = np.zeros((1, cout, oh, ow), dtype=np.float64)
    bound16 = np.zeros((1, cout, oh, ow), dtype=np.float64)
    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                patch = x[0, :, i : i + kh, j : j + kw].astype(np.float64)
                w_c = weight[c].astype(np.float64)
                y_float[0, c, i, j] = np.sum(patch * w_c)
                bound16[0, c, i, j] = (
                    eps_x16 * np.sum(np.abs(w_c))
                    + eps_w[c] * np.sum(np.abs(patch))
                    + K * eps_x16 * eps_w[c]
                )

    error16 = np.abs(y_float - y_quant)
    assert np.all(error16 <= bound16 + 1e-6)

    # Also confirm the rewrite is meaningfully close (not merely "within a
    # loose worst-case bound") for these well-scaled inputs -- consistent
    # with combined UINT16 (activation) + INT8 (weight) quantization, not
    # bitwise equal.
    np.testing.assert_allclose(y_quant, y_float, rtol=0.05, atol=0.2)

    # What static_quantize_conv's own UINT8 scheme would have done for the
    # IDENTICAL inputs/weight/range: a concrete numeric comparison, not just
    # the abstract Z3 scale-ratio corollary above. Same weight, same
    # dequantized-activation round trip, just with the coarser 8-bit scale.
    x_scale8, x_zp8 = _expected_asymmetric_uint8_quant_params(x_min, x_max)
    x_code8 = np.clip(np.round(x / x_scale8 + x_zp8), 0, 255)
    x_dq8 = (x_code8 - x_zp8) * x_scale8  # float32 QDQ round trip, UINT8-width

    y_quant8_wouldbe = np.zeros((1, cout, oh, ow), dtype=np.float64)
    bound8_wouldbe = np.zeros((1, cout, oh, ow), dtype=np.float64)
    eps_x8 = float(x_scale8) / 2.0
    wq8, _ws8 = _quantize_conv_weight_per_output_channel(weight)
    w_dq8 = wq8.astype(np.float64) * ws.reshape((-1, 1, 1, 1)).astype(np.float64)
    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                patch_dq8 = x_dq8[0, :, i : i + kh, j : j + kw].astype(np.float64)
                patch = x[0, :, i : i + kh, j : j + kw].astype(np.float64)
                w_c = weight[c].astype(np.float64)
                y_quant8_wouldbe[0, c, i, j] = np.sum(patch_dq8 * w_dq8[c])
                bound8_wouldbe[0, c, i, j] = (
                    eps_x8 * np.sum(np.abs(w_c))
                    + eps_w[c] * np.sum(np.abs(patch))
                    + K * eps_x8 * eps_w[c]
                )

    # Confirms the Z3 corollary's real-world consequence: this pass's
    # UINT16-activation worst-case bound is strictly (and substantially)
    # tighter than the UINT8 bound would be for this identical case, and the
    # actual achieved error tracks that -- this pass's real error stays well
    # under what would have been the UINT8 worst case.
    assert np.all(bound16 < bound8_wouldbe)
    assert np.max(error16) < np.max(bound8_wouldbe)
    error8_wouldbe = np.abs(y_float - y_quant8_wouldbe)
    assert np.all(error8_wouldbe <= bound8_wouldbe + 1e-6)
    # The actual achieved UINT16 error is meaningfully smaller than the
    # actual achieved UINT8-would-be error, not just its worst-case bound.
    assert np.max(error16) < np.max(error8_wouldbe)


def test_static_quantize_int16_conv_declines_without_calibrated_range():
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

    quantized = _quantize_static_int16(model, {})
    assert [n.op_type for n in quantized.graph.node] == ["Conv"]


def test_static_quantize_int16_conv_declines_pre_opset21_even_though_static_quantize_conv_would_fire():
    # This pass's own opset floor (UINT16 QuantizeLinear/DequantizeLinear
    # needs opset >= 21) is HIGHER than static_quantize_conv's (opset >= 13):
    # build the model at opset 13 -- a version at which static_quantize_conv
    # itself would already fire (see
    # test_formal_verify_static_quantize_conv.py's own equivalent negative
    # test, which uses opset 12 for ITS floor) -- with a calibrated range
    # available, and confirm THIS pass still declines and leaves the graph
    # completely untouched.
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
        opset=13,
        ir_version=8,
    )

    quantized = _quantize_static_int16(model, {"X": (-5.0, 10.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Conv"]


def test_static_quantize_int16_conv_declines_pre_opset21():
    # Same opset-floor check as above, restated at opset 20 -- one below this
    # pass's own floor rather than static_quantize_conv's -- to directly
    # exercise the ">= 21" boundary itself.
    rng = np.random.default_rng(5)
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

    quantized = _quantize_static_int16(model, {"X": (-5.0, 10.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Conv"]
