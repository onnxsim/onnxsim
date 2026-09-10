"""Formal check for QOperatorQuantizeConv (opt-in; onnxsim's own
``onnxsim/passes/qoperator_quantize_conv.h``): the Conv-shaped sibling of
``qoperator_quantize_matmul.h`` (``test_formal_verify_qoperator_quantize_
matmul.py`` is this file's primary template for the two-layer bound
structure -- read it first) and the QOperator-format sibling of
``static_quantize_conv.h`` (``test_formal_verify_static_quantize_conv.py`` is
this file's template for adapting a MatMul-shaped bound to Conv -- read it
too). It rewrites ``Y = Conv(X, W[, B])`` (``W`` constant float32, rank >= 3;
``B`` optional) into a single ``QLinearConv`` op -- no float ``Conv`` left in
the graph at all::

    Xq = QuantizeLinear(X, Xs, Xzp)                             -- CALIBRATED
    Yq = QLinearConv(Xq, Xs, Xzp, Wq, Ws, Wzp, Ys, Yzp[, Bq])    -- true int8
    Y  = DequantizeLinear(Yq, Ys, Yzp)                           -- CALIBRATED

``Wq``/``Ws`` use ``QuantizeConvWeightPerOutputChannel`` (axis 0 -- Conv's
``[Cout, Cin/groups, k...]`` layout gives no other choice, unlike Gemm's
``transB``-dependent layout ambiguity), exactly like ``static_quantize_conv``.
``X``/``Y`` both use a single (scalar) calibrated scale/zero-point, exactly
like ``qoperator_quantize_matmul``. Needs opset >= 10 (``QLinearConv``'s own
minimum), and BOTH the activation's and the node's own output's name need a
calibrated range in ``StaticQuantizationCalibrationRanges()`` -- the same
"quantizes its own output too" requirement ``qoperator_quantize_matmul`` has,
for the same reason (no float Conv survives to absorb a dequantization step).

Soundness claim -- three layers of error
==========================================
Layer 1 (reused verbatim from ``static_quantize_conv``'s own Conv-shaped
bound, itself the MatMul template's exact formulation): once a spatial output
position and output channel ``c`` are fixed, that one output element is
exactly a dot product of the receptive-field patch of ``X``
(``Cin/groups * prod(kernel dims)`` values, flattened) against ``W[c, ...]``
(flattened the same way) -- modeled, as in every sibling file, via a small
concrete contraction dimension ``_K`` standing in for that flattened
receptive field rather than anything specific to convolution's sliding-window
structure::

    |Conv(X, W)[..., c, ...] - Y_raw[..., c, ...]| <=
        (Xs / 2) * sum_k |W[c, k]| + (Ws[c] / 2) * sum_k |X_patch[k]|
        + K * (Xs / 2) * (Ws[c] / 2)                                -- "bound1"

where ``Y_raw`` is the value ``Y`` would take if ``QLinearConv``'s own output
weren't itself re-quantized (i.e. the raw int8-conv-and-dequantize result).
Same direct-error-variable (``ex``/``ew``) idiom as every sibling file, shown
tractable at this ``_K`` throughout this suite; the same nonlinear-blowup
risk that idiom avoids (see ``test_formal_verify_dynamic_quantize_matmul.py``
for the original incident writeup) applies here unchanged.

Layer 2 (reused verbatim from ``qoperator_quantize_matmul``'s own technique):
``Yq`` is ITSELF a quantized representation of ``Y_raw``, into the fixed
calibrated ``(Ys, Yzp)`` range, with its own round-trip bound
``|Y_raw - Y| <= Ys / 2``. Composing layers 1 and 2 via the triangle
inequality is checked as its own explicit Z3 query
(``test_qoperator_quantize_conv_combined_bound_holds``), exactly like the
MatMul template: ``|Conv(X, W) - Y| <= bound1 + Ys / 2``.

Layer 3 (genuinely new to THIS pass -- the content this file adds beyond
template-copying): unlike ``qoperator_quantize_matmul``'s Gemm bias, which
stays float and is added back AFTER the output's own dequantization (so it
cancels out of the combined-bound error term algebraically, contributing
nothing new -- see that file's own bias-variant test), ``QLinearConv``'s
optional bias input has no float fallback at all. When present, ``runTransform``
quantizes it AHEAD OF TIME into a fixed, non-calibrated per-output-channel
INT32 tensor, with ``bias_scale[c] = x_scale * w_scale[c]`` and zero_point
implicitly 0 -- ``QLinearConv``'s own documented contract for its bias input
(see the ``QLinearConv_ver10`` schema comment ``qoperator_quantize_conv.h``
itself references, and that header's own doc comment above ``runTransform``).
This is NOT a free/calibrated round-trip like the activation or output: it is
an exact-formula-derived quantization whose own rounding error is bounded by
``|Bias[c] - dequant(Bias_q)[c]| <= bias_scale[c] / 2 = (Xs * Ws[c]) / 2`` --
a THIRD error term, additive via the same triangle-inequality-as-a-Z3-query
technique layer 2 already uses, folded into the bias variant of the combined
bound (``test_qoperator_quantize_conv_combined_bound_bias_variant`` below).
Because the bias itself is quantized here (unlike Gemm's untouched float
bias), this term does NOT cancel algebraically -- it is a genuine additional
additive error budget the combined bound must carry whenever a bias is
present, which a negative control below
(``test_qoperator_quantize_conv_negative_control_requires_bias_bound``)
confirms is not vacuous: dropping just the bias's own rounding-error
hypothesis (while keeping both other hypotheses) breaks the combined-with-bias
bound.

Predicate difference from ``qoperator_quantize_matmul``
==========================================================
Because there is no float fallback for a ``QLinearConv`` bias (unlike Gemm's,
which rides through untouched in float regardless of whether it's constant),
a Conv whose bias is present but NOT a constant float32 ``[Cout]`` tensor
cannot be rewritten at all and is left alone entirely --
``patternMatchPredicate``'s own distinguishing decline condition versus
``qoperator_quantize_matmul`` (whose Gemm-bias case has no such restriction).
Confirmed below as a differential test
(``test_qoperator_quantize_conv_declines_with_non_constant_bias``).

Confirming QLinearConv's CPU kernel before relying on it
===========================================================
``tests/test_qoperator_quantize_conv.py`` (this repo's own non-formal
coverage of this pass) already runs the real quantized graph -- both without
a bias (``test_quantize_conv``) and with a constant bias baked into
``QLinearConv``'s 9th input (``test_quantize_conv_with_constant_bias``) --
through ``onnxruntime.InferenceSession`` successfully; re-running that file
here (``python3 -m pytest tests/test_qoperator_quantize_conv.py -q``) confirms
it still passes in this environment. So, unlike some other ops in this suite
that need a hand-simulated fallback for a missing onnxruntime kernel (see
e.g. the FLOAT8/BFLOAT16 workarounds elsewhere in this suite), ``QLinearConv``
has a working CPU kernel here, bias input included, and the numeric-bound
differential tests below run the real quantized graph through onnxruntime
rather than a hand-simulated int8 computation.

Invoking the pass
===================
As in ``qoperator_quantize_matmul``'s own docstring: the nanobind-exposed
entry point is ``onnxsim.onnxsim_cpp2py_export.quantize_qoperator(model_bytes,
activation_ranges)``, running ``OptimizeFixed(["qoperator_quantize_matmul",
"qoperator_quantize_conv"])`` -- so it already isolates this pass family, and
``activation_ranges`` must supply a calibrated range for BOTH the Conv's
activation input's name and the Conv node's own output name (``"Y"`` for the
single-node models built below).
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
import onnxsim.onnxsim_cpp2py_export as C
from _formal_verify_common import producer, prove, z3
from onnx import parser

assert "qoperator_quantize_conv" in C._list_other_optimizers()

_K = 2  # concrete number of contraction taps -- matches quantized_mac_bound's/
# static_quantize_conv's/qoperator_quantize_matmul's own choice; see their
# docstrings for why (enough to exercise the cross-tap sum, empirically the
# largest tractable value for this shape of nonlinear Z3 query).


def _abs(v):
    return z3.If(v >= 0, v, -v)


def _bound_formulas():
    """Layer 1's Z3 vocabulary -- an exact copy of
    ``test_formal_verify_static_quantize_conv.py``'s own ``_bound_formulas``
    (same direct-error-variable ``ex``/``ew`` idiom, same ``_K``, same
    per-tap sum over the flattened receptive field): ``QLinearConv``'s
    internal int8 accumulation, before its own output is re-quantized, is
    mechanically identical to ``static_quantize_conv``'s literal
    ``Conv(Xdq, Wdq)``, so the same lemma applies unchanged.
    ``quantized_conv`` here plays the role of ``Y_raw`` -- the raw
    int8-conv-and-dequantize result, before the output's own extra round
    trip. Unlike the matmul/static_quantize_conv templates, ``Xs``/``Ws`` are
    returned too -- this file's bias variant needs them directly for the
    THIRD error term (``bias_scale[c] := Xs * Ws[c]``), not just baked into
    ``bound``.

    Returns ``(float_conv, quantized_conv, rounding_bounds, bound, Xs, Ws)``.
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
    quantized_conv = sum(Xdq[k] * Wdq[k] for k in range(_K))  # Y_raw[..., c, ...]

    bound = (
        (Xs / 2) * sum(_abs(W[k]) for k in range(_K))
        + (Ws / 2) * sum(_abs(X[k]) for k in range(_K))
        + _K * (Xs / 2) * (Ws / 2)
    )

    return float_conv, quantized_conv, rounding_bounds, bound, Xs, Ws


def test_qoperator_quantize_conv_layer1_error_is_bounded():
    # Layer 1: the same bound static_quantize_conv's own proof establishes
    # for its literal Conv(Xdq, Wdq) applies unchanged to this pass's raw
    # int8-conv-and-dequantize result Y_raw (quantized_conv here) -- the
    # value before QLinearConv's own output gets re-quantized. Reused here
    # (rather than imported) so this file is self-contained, matching this
    # suite's per-file convention of each proof carrying its own Z3
    # vocabulary.
    float_conv, quantized_conv, rounding_bounds, bound, _Xs, _Ws = _bound_formulas()
    error = float_conv - quantized_conv
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_qoperator_quantize_conv_combined_bound_holds():
    # Layer 2, reused verbatim from qoperator_quantize_matmul's own
    # technique: Y (the pass's actual final output, DequantizeLinear(Yq, Ys,
    # Yzp)) is Y_raw's own further round-trip through a SECOND
    # QuantizeLinear/DequantizeLinear-shaped step -- modeled the same
    # direct-error-variable way, `eout := Y_raw - Y`, bounded by `Ys / 2`.
    # Given BOTH layer 1's rounding hypotheses (bounding
    # |float_conv - Y_raw|) and this output round-trip hypothesis (bounding
    # |Y_raw - Y|), Z3 -- not hand algebra -- confirms the triangle-
    # inequality composition: |float_conv - Y| <= bound1 + Ys / 2.
    float_conv, quantized_conv, rounding_bounds, bound1, _Xs, _Ws = _bound_formulas()
    Ys = z3.Real("Ys")  # calibrated per-tensor OUTPUT scale
    eout = z3.Real("eout")  # Y_raw[..., c, ...] - Y[..., c, ...]
    Y = quantized_conv - eout

    output_round_trip = z3.And(Ys > 0, _abs(eout) <= Ys / 2)
    hypotheses = z3.And(rounding_bounds, output_round_trip)

    combined_bound = bound1 + Ys / 2
    error = float_conv - Y
    prove(
        z3.Implies(
            hypotheses, z3.And(error <= combined_bound, -error <= combined_bound)
        )
    )


def test_qoperator_quantize_conv_bias_round_trip_holds():
    # Grounds the THIRD error term from first principles: the bias's own
    # quantization is NOT a calibrated round-trip like X/Y -- it's an
    # exact-formula derivation from a FIXED scale, bias_scale[c] :=
    # x_scale * w_scale[c] (QLinearConv's own documented contract for its
    # bias input -- the QLinearConv_ver10 schema comment
    # qoperator_quantize_conv.h itself references), zero_point pinned at 0
    # (never calibrated, unlike Xzp/Yzp). Still, its rounding error obeys
    # exactly the same round(x / scale) shape as the X/Y round-trip lemma
    # with zero_point fixed at 0 -- `n` models round(bias / bias_scale), and
    # "no clipping" means `n` is exactly the INT32 code QLinearConv reads
    # back (runTransform's clamp to the INT32 range is not hit).
    bias, bias_scale = z3.Reals("bias bias_scale")
    n = z3.Int("n")  # round(bias / bias_scale), not yet known to be an integer code
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        bias_scale > 0,
        n - (bias / bias_scale) <= half,
        (bias / bias_scale) - n <= half,
    )
    dequant_bias = n * bias_scale  # zero_point is fixed at 0, unlike X/Y

    error = bias - dequant_bias
    prove(
        z3.Implies(
            hypotheses, z3.And(error <= bias_scale / 2, -error <= bias_scale / 2)
        )
    )


def test_qoperator_quantize_conv_combined_bound_bias_variant():
    # The genuinely new claim this pass needs on top of the MatMul template's
    # two layers: when a bias is present, QLinearConv's own int32 accumulator
    # sums the raw quantized dot product with the pre-quantized bias, both
    # already expressed in the SAME "Xs * Ws" units (bias_scale[c] :=
    # Xs * Ws[c] is exactly why no extra rescale step is needed) -- so, once
    # converted back to real units, the raw (pre-output-quantization) result
    # is quantized_conv + dequant(Bias), i.e. quantized_conv + (Bias -
    # ebias), with `ebias` bounded by bias_scale / 2 per the round-trip lemma
    # just proved above. UNLIKE qoperator_quantize_matmul's Gemm-bias variant
    # (where Bias cancels out algebraically because it rides through in
    # float, untouched, entirely outside the quantized round-trip), this
    # ebias term does NOT cancel -- it is a genuine third additive error
    # budget, since the bias itself is quantized here. Z3 confirms the
    # three-way triangle-inequality composition:
    # |(float_conv + Bias) - Y_with_bias| <= bound1 + bias_scale/2 + Ys/2.
    float_conv, quantized_conv, rounding_bounds, bound1, Xs, Ws = _bound_formulas()
    Ys = z3.Real("Ys")
    eout = z3.Real("eout")  # Y_raw_with_bias[..., c, ...] - Y_with_bias[..., c, ...]
    ebias = z3.Real("ebias")  # Bias[c] - dequant(Bias_q)[c]
    Bias = z3.Real("Bias")

    Y_raw_with_bias = quantized_conv + (Bias - ebias)
    Y_with_bias = Y_raw_with_bias - eout

    bias_scale_bound = Xs * Ws / 2  # bias_scale[c] = x_scale * w_scale[c]
    hypotheses = z3.And(
        rounding_bounds,
        Ys > 0,
        _abs(eout) <= Ys / 2,
        _abs(ebias) <= bias_scale_bound,
    )
    combined_bound = bound1 + bias_scale_bound + Ys / 2
    error = (float_conv + Bias) - Y_with_bias
    prove(
        z3.Implies(
            hypotheses, z3.And(error <= combined_bound, -error <= combined_bound)
        )
    )


def test_qoperator_quantize_conv_negative_control_requires_layer1_bound():
    # Sanity check that the (no-bias) combined bound genuinely needs LAYER
    # 1's rounding hypotheses, not just the output round-trip one: with the
    # output's own round-trip bound assumed but NO budget at all on the
    # internal accumulation error, the combined claim is not a theorem -- Z3
    # must find a real counterexample.
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ex = [z3.Real(f"ex{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    Xs = z3.Real("Xs")
    Ws = z3.Real("Ws")
    Ys = z3.Real("Ys")
    eout = z3.Real("eout")

    Xdq = [X[k] - ex[k] for k in range(_K)]
    Wdq = [W[k] - ew[k] for k in range(_K)]
    float_conv = sum(X[k] * W[k] for k in range(_K))
    quantized_conv = sum(Xdq[k] * Wdq[k] for k in range(_K))
    Y = quantized_conv - eout

    bound1 = (
        (Xs / 2) * sum(_abs(W[k]) for k in range(_K))
        + (Ws / 2) * sum(_abs(X[k]) for k in range(_K))
        + _K * (Xs / 2) * (Ws / 2)
    )
    combined_bound = bound1 + Ys / 2
    error = float_conv - Y

    solver = z3.Solver()
    solver.add(Xs > 0, Ws > 0, Ys > 0, _abs(eout) <= Ys / 2)  # output bound only
    solver.add(z3.Not(z3.And(error <= combined_bound, -error <= combined_bound)))
    assert solver.check() == z3.sat, (
        "the combined bound holds even without layer 1's rounding hypotheses "
        "-- negative control is vacuous"
    )


def test_qoperator_quantize_conv_negative_control_requires_output_bound():
    # Symmetric negative control: with layer 1's rounding hypotheses assumed
    # but NO budget at all on the output's own round-trip error (eout free),
    # the combined claim is likewise not a theorem.
    float_conv, quantized_conv, rounding_bounds, bound1, _Xs, _Ws = _bound_formulas()
    Ys = z3.Real("Ys")
    eout = z3.Real("eout")
    Y = quantized_conv - eout
    combined_bound = bound1 + Ys / 2
    error = float_conv - Y

    solver = z3.Solver()
    solver.add(rounding_bounds, Ys > 0)  # no bound on eout at all
    solver.add(z3.Not(z3.And(error <= combined_bound, -error <= combined_bound)))
    assert solver.check() == z3.sat, (
        "the combined bound holds even without the output's own round-trip "
        "bound -- negative control is vacuous"
    )


def test_qoperator_quantize_conv_negative_control_requires_bias_bound():
    # The bias-specific negative control this file's docstring promises: with
    # layer 1's rounding hypotheses AND the output's own round-trip bound
    # both assumed, but NO budget at all on the bias's own rounding error
    # (ebias free), the combined-with-bias claim is not a theorem either --
    # confirming the third error term is a genuine additional hypothesis, not
    # one already implied by the other two.
    float_conv, quantized_conv, rounding_bounds, bound1, Xs, Ws = _bound_formulas()
    Ys = z3.Real("Ys")
    eout = z3.Real("eout")
    ebias = z3.Real("ebias")
    Bias = z3.Real("Bias")
    Y_with_bias = quantized_conv + (Bias - ebias) - eout

    bias_scale_bound = Xs * Ws / 2
    combined_bound = bound1 + bias_scale_bound + Ys / 2
    error = (float_conv + Bias) - Y_with_bias

    solver = z3.Solver()
    solver.add(rounding_bounds, Ys > 0, _abs(eout) <= Ys / 2)  # no bound on ebias
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


def _quantize_qoperator(model, activation_ranges):
    """Invokes the real compiled pass directly via the nanobind-exposed
    ``quantize_qoperator(model_bytes, activation_ranges)`` -- see this file's
    and ``test_formal_verify_qoperator_quantize_matmul.py``'s module
    docstrings for why. Runs ``OptimizeFixed(["qoperator_quantize_matmul",
    "qoperator_quantize_conv"])``, so no extra
    ``skipped_optimizers``/``simplify_isolated_extra`` isolation is needed.
    """
    out = onnx.ModelProto()
    out.ParseFromString(
        C.quantize_qoperator(model.SerializeToString(), activation_ranges)
    )
    return out


def _quantize_conv_weight_per_output_channel(weight):
    """Independent numpy re-implementation of
    ``QuantizeConvWeightPerOutputChannel`` (quantize_conv_common.h):
    per-output-channel (axis 0) symmetric INT8 quantization of a Conv weight
    ``[Cout, Cin/groups, k...]`` -- scale = max(|W[c, ...]|) / 127 (or 1.0
    for an all-zero channel), codes = round(W[c, ...] / scale) clipped to
    [-127, 127], reducing over every axis but 0. Identical to
    ``test_formal_verify_static_quantize_conv.py``'s own helper of the same
    name (both passes share this exact weight-quantization scheme).
    """
    reduce_axes = tuple(range(1, weight.ndim))
    scale = np.max(np.abs(weight), axis=reduce_axes)
    scale = np.where(scale > 0, scale / 127.0, 1.0).astype(np.float32)
    scale_bcast = scale.reshape((-1,) + (1,) * (weight.ndim - 1))
    codes = np.clip(np.round(weight / scale_bcast), -127, 127).astype(np.int8)
    return codes, scale


def _expected_asymmetric_uint8_quant_params(min_val, max_val):
    """Independent re-implementation of ``ComputeAsymmetricUint8QuantParams``
    (static_quantize_matmul.h, shared by this pass for both its activation
    AND its output), in float32 to match the pass's own arithmetic precision.
    Identical to ``test_formal_verify_static_quantize_conv.py``'s/
    ``test_formal_verify_qoperator_quantize_matmul.py``'s own helper of the
    same name.
    """
    lo = min(np.float32(0.0), np.float32(min_val))
    hi = max(np.float32(0.0), np.float32(max_val))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zero_point = int(np.clip(np.round(-lo / scale), 0, 255))
    return np.float32(scale), zero_point


def _quantize_conv_bias_int32(bias, x_scale, w_scale):
    """Independent numpy re-implementation of ``runTransform``'s bias
    quantization (``qoperator_quantize_conv.h``): per-output-channel scale
    ``bias_scale[c] = x_scale * w_scale[c]``, zero_point fixed at 0 (never
    calibrated), codes = round(bias[c] / bias_scale[c]) clipped to the INT32
    range representable exactly in float32 -- mirrors the C++ clamp to
    ``[-2147483648.0f, 2147483520.0f]`` (the largest float32 value
    <= INT32_MAX; 2147483647 itself isn't exactly representable as float32).
    """
    bias_scale = (np.float32(x_scale) * w_scale.astype(np.float32)).astype(np.float32)
    q = np.round(bias.astype(np.float32) / bias_scale)
    q = np.clip(q, -2147483648.0, 2147483520.0)
    return q.astype(np.int64).astype(np.int32), bias_scale


def test_qoperator_quantize_conv_pass_fires_and_matches_scheme():
    # Build a plain float Conv (no bias) and run the real pass with
    # calibration ranges for BOTH X and the node's own output "Y", each
    # chosen to straddle 0 so BOTH Xzp and Yzp come out genuinely nonzero --
    # mirroring qoperator_quantize_matmul's own analogous test, exercised
    # here for Conv.
    rng = np.random.default_rng(0)
    cout, cin, kh, kw = 3, 2, 3, 3
    hw = 6
    oh, ow = hw - kh + 1, hw - kw + 1
    weight = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.6
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X) => (float[1,{cout},{oh},{ow}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    x_range = (-5.0, 10.0)
    y_range = (-3.0, 12.0)
    quantized = _quantize_qoperator(model, {"X": x_range, "Y": y_range})

    # No float Conv left anywhere -- the genuinely distinguishing structural
    # check vs. QDQ format (static_quantize_conv always keeps the original
    # Conv node).
    op_types = {n.op_type for n in quantized.graph.node}
    assert "Conv" not in op_types

    # Walk the chain backward from the real graph output:
    # DequantizeLinear(Yq) <- QLinearConv(Xq, ...) <- QuantizeLinear(X).
    dq_node = producer(quantized, "Y")
    assert dq_node.op_type == "DequantizeLinear"
    qlconv_node = producer(quantized, dq_node.input[0])
    assert qlconv_node.op_type == "QLinearConv"
    assert len(qlconv_node.input) == 8  # no bias in this model
    assert dq_node.input[1:] == [
        qlconv_node.input[6],
        qlconv_node.input[7],
    ]  # Ys, Yzp shared

    ql_node = producer(quantized, qlconv_node.input[0])
    assert ql_node.op_type == "QuantizeLinear"
    assert ql_node.input[0] == "X"
    assert ql_node.input[1:] == qlconv_node.input[1:3]  # Xs, Xzp shared

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

    wq = numpy_helper.to_array(init[qlconv_node.input[3]])
    ws = numpy_helper.to_array(init[qlconv_node.input[4]])
    wzp = numpy_helper.to_array(init[qlconv_node.input[5]])
    expected_wq, expected_ws = _quantize_conv_weight_per_output_channel(weight)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)
    assert wzp.dtype == np.int8
    assert wzp.shape == (cout,)
    np.testing.assert_array_equal(wzp, np.zeros(cout, dtype=np.int8))


def test_qoperator_quantize_conv_bias_is_quantized_into_qlinearconv():
    # The genuinely new structural content this pass has over
    # qoperator_quantize_matmul: a constant float32 [Cout] bias is quantized
    # AHEAD OF TIME into QLinearConv's own 9th (optional) input, as INT32,
    # with per-channel scale x_scale * w_scale[c] and zero_point 0 -- NOT
    # added back in float via a separate Add the way Gemm's bias is.
    rng = np.random.default_rng(1)
    cout, cin, kh, kw = 4, 2, 3, 3
    hw = 6
    oh, ow = hw - kh + 1, hw - kw + 1
    weight = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.5
    bias = rng.standard_normal(cout).astype(np.float32) * 3.0
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X) => (float[1,{cout},{oh},{ow}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )

    x_range = (-5.0, 10.0)
    y_range = (-3.0, 12.0)
    quantized = _quantize_qoperator(model, {"X": x_range, "Y": y_range})

    op_types = [n.op_type for n in quantized.graph.node]
    assert "Add" not in op_types  # no float Add for the bias, unlike Gemm's case
    assert op_types.count("QLinearConv") == 1

    qlconv_node = next(n for n in quantized.graph.node if n.op_type == "QLinearConv")
    assert len(qlconv_node.input) == 9  # ..., y_scale, y_zero_point, B

    init = {i.name: i for i in quantized.graph.initializer}
    x_scale = float(numpy_helper.to_array(init[qlconv_node.input[1]]))
    ws = numpy_helper.to_array(init[qlconv_node.input[4]])
    bias_q = numpy_helper.to_array(init[qlconv_node.input[8]])

    assert bias_q.dtype == np.int32
    expected_bias_q, _bias_scale = _quantize_conv_bias_int32(bias, x_scale, ws)
    np.testing.assert_array_equal(bias_q, expected_bias_q)


def test_qoperator_quantize_conv_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring qoperator_quantize_matmul's/
    # static_quantize_conv's own numeric-bound tests: run the real quantized
    # graph through onnxruntime (QLinearConv has a working CPU kernel in this
    # build -- confirmed empirically both by this file's module docstring and
    # by tests/test_qoperator_quantize_conv.py's own passing tests) and
    # confirm every output element's error against the true float Conv stays
    # within the COMBINED (two-layer, no-bias) bound proved above. Both
    # calibration ranges are set to the actual observed (min, max) of X and
    # of the true float output so nothing clips.
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

    K = cin * kh * kw  # this concrete model's actual per-position contraction depth
    y_float = np.zeros((1, cout, oh, ow), dtype=np.float64)
    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                patch = x[0, :, i : i + kh, j : j + kw].astype(np.float64)
                y_float[0, c, i, j] = np.sum(patch * weight[c].astype(np.float64))

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y_float.min()), float(y_float.max())
    quantized = _quantize_qoperator(model, {"X": (x_min, x_max), "Y": (y_min, y_max)})

    # Graph optimization disabled: onnxruntime can apply hardware-specific
    # graph transforms (e.g. layout transforms for quantized Conv) that are a
    # different code path than the literal node chain this pass's proof
    # reasons about -- see `tests/test_ort_matmul_nbits_workaround.py`'s
    # docstring for this suite's existing precedent of a real ORT
    # graph-optimization fusion bug of exactly this shape. Disabling
    # optimization executes the graph exactly as the pass produced it.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        quantized.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    (y_quant,) = sess.run(None, {"X": x})

    x_scale, _x_zp = _expected_asymmetric_uint8_quant_params(x_min, x_max)
    y_scale, _y_zp = _expected_asymmetric_uint8_quant_params(y_min, y_max)
    _wq, ws = _quantize_conv_weight_per_output_channel(weight)

    eps_x = float(x_scale) / 2.0
    eps_w = ws / 2.0  # shape [cout]
    eps_y = float(y_scale) / 2.0

    bound = np.zeros((1, cout, oh, ow), dtype=np.float64)
    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                patch = x[0, :, i : i + kh, j : j + kw].astype(np.float64)
                w_c = weight[c].astype(np.float64)
                bound[0, c, i, j] = (
                    eps_x * np.sum(np.abs(w_c))
                    + eps_w[c] * np.sum(np.abs(patch))
                    + K * eps_x * eps_w[c]
                    + eps_y
                )

    error = np.abs(y_float - y_quant)
    assert np.all(error <= bound + 1e-6)

    np.testing.assert_allclose(y_quant, y_float, rtol=0.1, atol=0.2)


def test_qoperator_quantize_conv_output_with_bias_is_close_to_float_within_proved_bound():
    # The bias variant of the previous test -- exercises the THIRD error
    # term (bias_scale[c] / 2) this file's proof adds on top of the two-layer
    # bound, checked against onnxruntime's own execution of the real
    # quantized graph (bias included, riding inside QLinearConv's 9th input).
    rng = np.random.default_rng(3)
    cout, cin, kh, kw = 3, 2, 3, 3
    hw = 6
    oh, ow = hw - kh + 1, hw - kw + 1
    weight = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.8
    bias = rng.standard_normal(cout).astype(np.float32) * 2.0
    x = rng.standard_normal((1, cin, hw, hw)).astype(np.float32) * 2.0
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X) => (float[1,{cout},{oh},{ow}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )

    K = cin * kh * kw
    y_float = np.zeros((1, cout, oh, ow), dtype=np.float64)
    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                patch = x[0, :, i : i + kh, j : j + kw].astype(np.float64)
                y_float[0, c, i, j] = (
                    np.sum(patch * weight[c].astype(np.float64)) + bias[c]
                )

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y_float.min()), float(y_float.max())
    quantized = _quantize_qoperator(model, {"X": (x_min, x_max), "Y": (y_min, y_max)})

    sess = ort.InferenceSession(quantized.SerializeToString())
    (y_quant,) = sess.run(None, {"X": x})

    x_scale, _x_zp = _expected_asymmetric_uint8_quant_params(x_min, x_max)
    y_scale, _y_zp = _expected_asymmetric_uint8_quant_params(y_min, y_max)
    _wq, ws = _quantize_conv_weight_per_output_channel(weight)
    _bias_q, bias_scale = _quantize_conv_bias_int32(bias, float(x_scale), ws)

    eps_x = float(x_scale) / 2.0
    eps_w = ws / 2.0  # shape [cout]
    eps_y = float(y_scale) / 2.0
    eps_bias = bias_scale / 2.0  # shape [cout] -- the THIRD error term

    bound = np.zeros((1, cout, oh, ow), dtype=np.float64)
    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                patch = x[0, :, i : i + kh, j : j + kw].astype(np.float64)
                w_c = weight[c].astype(np.float64)
                bound[0, c, i, j] = (
                    eps_x * np.sum(np.abs(w_c))
                    + eps_w[c] * np.sum(np.abs(patch))
                    + K * eps_x * eps_w[c]
                    + eps_bias[c]
                    + eps_y
                )

    error = np.abs(y_float - y_quant)
    assert np.all(error <= bound + 1e-6)

    np.testing.assert_allclose(y_quant, y_float, rtol=0.1, atol=0.3)


def test_qoperator_quantize_conv_declines_with_only_activation_range():
    # patternMatchPredicate requires calibrated ranges for BOTH the
    # activation AND the node's own output -- the same requirement
    # qoperator_quantize_matmul has. Supplying only X's range (no entry for
    # "Y") must leave the Conv completely untouched.
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
    )

    quantized = _quantize_qoperator(model, {"X": (-5.0, 10.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Conv"]


def test_qoperator_quantize_conv_declines_with_non_constant_bias():
    # This pass's own distinguishing decline condition versus
    # qoperator_quantize_matmul's Gemm bias (which has no such restriction,
    # since it stays float and untouched): QLinearConv has no float fallback
    # for its bias, so a Conv whose bias is present but NOT a constant
    # float32 [Cout] tensor -- here, a graph input -- cannot be rewritten and
    # is left alone entirely, even with both calibrated ranges available.
    rng = np.random.default_rng(5)
    cout, cin, kh, kw = 2, 1, 3, 3
    hw = 4
    weight = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X, float[{cout}] B) => (float[1,{cout},{hw - kh + 1},{hw - kw + 1}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W, B)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = _quantize_qoperator(model, {"X": (-5.0, 10.0), "Y": (-3.0, 12.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Conv"]


def test_qoperator_quantize_conv_declines_pre_opset10():
    # QLinearConv needs opset >= 10 (patternMatchPredicate's opset check); a
    # plain Conv at an older opset must survive untouched even with both
    # calibrated ranges available.
    rng = np.random.default_rng(6)
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
        opset=9,
        ir_version=7,
    )

    quantized = _quantize_qoperator(model, {"X": (-5.0, 10.0), "Y": (-3.0, 12.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Conv"]
