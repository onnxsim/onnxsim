"""Formal check for StaticQuantizeInt16MatMul (opt-in; onnxsim's own
``onnxsim/passes/static_quantize_int16_matmul.h``): the "W8A16" sibling of
``static_quantize_matmul.h`` (this file's primary template --
``test_formal_verify_static_quantize_matmul.py`` -- already proves the exact
QDQ-format bound for that UINT8-activation pass; read it in full first). The
weight stays INT8, quantized per output channel, symmetric, via the exact
same ``QuantizeWeightPerChannelInPlace`` call the UINT8 pass uses -- nothing
about the weight scheme changes here. The only difference is the
*activation*: instead of ``ComputeAsymmetricUint8QuantParams`` (scale divisor
255), this pass calls ``ComputeAsymmetricUint16QuantParams`` (scale divisor
65535), an 8x finer calibrated affine step aimed at activations unusually
sensitive to the QuantizeLinear/DequantizeLinear round trip (e.g. post-
softmax attention scores). It also requires opset >= 21 (UINT16
QuantizeLinear/DequantizeLinear support), unlike the UINT8 pass's opset >= 13.

Is the bound proof genuinely new Z3 content, or a re-instantiation?
====================================================================
``static_quantize_matmul``'s own ``_bound_formulas`` helper (and this file's
copy of it, below) builds ``quantized_mac_bound``'s general two-operand lemma
purely from a free per-tap scale variable ``Xs`` and a per-tap rounding
hypothesis ``|ex[k]| <= Xs / 2`` -- the proof's hypotheses never mention 255,
65535, or any other concrete divisor; the divisor only ever enters where a
concrete scale ``Xs`` is *computed* from a calibrated range, which happens on
the C++ side (``ComputeAsymmetricUint16QuantParams``) and in this file's own
differential tests (``_expected_asymmetric_uint16_quant_params`` below), not
inside the bound lemma's Z3 vocabulary itself. So the bound-proving queries
below are mechanically identical to the UINT8 file's -- this is confirmed,
not assumed, precisely because ``_bound_formulas`` here is copied verbatim
and still closes with the same hypotheses. What line up differently is:

1. ``static_quantize_int16_matmul``-specific declination behavior (opset >=
   21 rather than >= 13) -- tested structurally against the real pass below,
   including the concrete case where the UINT8 pass would fire (opset 13)
   but this one still declines.
2. A genuinely new corollary, absent from the UINT8 file entirely: for the
   SAME calibrated range, this pass's UINT16 activation scale is *exactly*
   1/257 of what the UINT8 pass's scale would be for that same range --
   ``255 * 257 == 65535`` exactly, so ``Xs_8 == 257 * Xs_16`` is not an
   approximation but an exact integer identity (``test_static_quantize_
   int16_matmul_activation_scale_is_257x_finer_than_uint8`` below). Since the
   bound's activation-error term is ``eps_x * sum|W|`` with ``eps_x = Xs /
   2``, this directly shows that term is 257x smaller here, all else equal
   -- the concrete payoff of "W8A16" over "W8A8" for the activand fed
   through the standard analysis.

Both scale formulas apply the identical widening step first (``lo = min(0,
min_val)``, ``hi = max(0, max_val)``) before dividing by their respective
divisor, so the corollary below works directly with that shared, already-
widened ``(lo, hi)`` pair rather than re-deriving the widening itself (which
neither differs between, nor is specific to, either pass).

Everything else -- the direct-error-variable (``ex``/``ew``) idiom, ``_K =
2``, the round-trip lemma carrying a free zero-point, the Gemm "+ Bias"
cancellation, the vacuity negative control -- mirrors the UINT8 file exactly;
see it for the fuller exposition of each. As that file's own docstring notes
(re-emphasized in ``test_formal_verify_dynamic_quantize_matmul.py``'s
incident writeup), every bound-proving query here is built from the start in
the direct-error-variable idiom -- never reconstructed from separate
quantized-code/zero-point multiplicands -- which is what keeps these queries
fast (~20-25s) instead of hanging.

Invoking the pass with a specific calibrated range
====================================================
Exactly mirroring the UINT8 file's own reasoning: the nanobind-exposed entry
point one layer under ``onnxsim.calibration.quantize_static_int16`` is
``onnxsim.onnxsim_cpp2py_export.quantize_static_int16(model_bytes,
activation_ranges)`` (see ``onnxsim/cpp2py_export.cc``) -- the exact
analogue of ``quantize_static`` the UINT8 file's own differential tests use,
just calling ``QuantizeStaticInt16`` (``onnxsim/quantize_entry.cpp``), which
runs ``OptimizeFixed`` with exactly ``["static_quantize_int16_matmul",
"static_quantize_int16_conv"]``. This already isolates this pass family with
no ``extra_optimizers``/``simplify_isolated_extra`` machinery needed, and
takes a caller-supplied ``{name: (min, max)}`` dict directly rather than
requiring calibration *data* to be fabricated -- so no dedicated "isolate
just this one pass" helper needed to be discovered or improvised; it already
existed, symmetric with the UINT8 pass's own.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
import onnxsim.onnxsim_cpp2py_export as C
from _formal_verify_common import prove, z3
from onnx import parser

_K = 2  # matches quantized_mac_bound's/static_quantize_matmul's own _K.


def _abs(v):
    return z3.If(v >= 0, v, -v)


def test_static_quantize_int16_matmul_activation_round_trip_holds_for_any_zero_point():
    # Identical in shape to static_quantize_matmul's own version of this
    # lemma (down to the "no clipping" side condition) -- QuantizeLinear's
    # round-trip bound is agnostic to which integer type backs the code and
    # to Xzp's value, so this pass's UINT16 Xzp is covered by the exact same
    # derivation, just reproved here in this file's own vocabulary per this
    # suite's per-file convention.
    x, Xs, Xzp = z3.Reals("x Xs Xzp")
    n = z3.Int("n")  # round(x / Xs + Xzp)
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        Xs > 0,
        n - (x / Xs + Xzp) <= half,
        (x / Xs + Xzp) - n <= half,
    )
    xdq = (n - Xzp) * Xs
    error = x - xdq
    prove(z3.Implies(hypotheses, z3.And(error <= Xs / 2, -error <= Xs / 2)))


def test_static_quantize_int16_matmul_activation_scale_is_257x_finer_than_uint8():
    # The genuinely new corollary this file adds on top of the UINT8 file's
    # machinery: for the SAME already-widened calibrated range [lo, hi],
    # this pass's UINT16 scale (divisor 65535) is EXACTLY 1/257 of what
    # static_quantize_matmul's UINT8 scale (divisor 255) would be for that
    # same range -- 255 * 257 == 65535 exactly, so this is an exact integer
    # identity, not merely "smaller" or "approximately 8x finer per code,
    # 257x finer per relative step". Whenever hi > lo (guaranteed by both
    # ComputeAsymmetric*QuantParams's shared widen-then-clamp-degenerate-
    # range step, `if (hi <= lo) hi = lo + 1`), the UINT16 scale is also
    # strictly smaller -- the activation's own contribution to the bound
    # (eps_x = Xs / 2) shrinks by the same exact factor.
    lo, hi = z3.Reals("lo hi")
    Xs16 = (hi - lo) / 65535
    Xs8 = (hi - lo) / 255
    prove(
        z3.Implies(
            hi > lo,
            z3.And(Xs8 == 257 * Xs16, Xs16 < Xs8, Xs16 > 0, Xs8 > 0),
        )
    )


def _bound_formulas():
    """Verbatim copy of static_quantize_matmul's own helper -- see that
    file's docstring for the full derivation. Kept as a literal copy (not a
    shared import) per this suite's per-file self-containment convention;
    its being unchanged is itself the point, see this file's module
    docstring for why no new nonlinear query is needed for the bound itself.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # X[i, k], true float activation
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[k, n], true float weight
    ex = [z3.Real(f"ex{k}") for k in range(_K)]  # X[i, k] - Xdq[i, k]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[k, n] - Wdq[k, n]
    Xs = z3.Real("Xs")  # calibrated per-tensor activation scale (UINT16 here)
    Ws = z3.Real("Ws")  # this pass's per-output-channel weight scale Ws[n]

    Xdq = [X[k] - ex[k] for k in range(_K)]
    Wdq = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        Xs > 0,
        Ws > 0,
        *[_abs(ex[k]) <= Xs / 2 for k in range(_K)],
        *[_abs(ew[k]) <= Ws / 2 for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    quantized_matmul = sum(Xdq[k] * Wdq[k] for k in range(_K))  # MatMul(Xdq, Wdq)[i, n]

    bound = (
        (Xs / 2) * sum(_abs(W[k]) for k in range(_K))
        + (Ws / 2) * sum(_abs(X[k]) for k in range(_K))
        + _K * (Xs / 2) * (Ws / 2)
    )

    return float_matmul, quantized_matmul, rounding_bounds, bound


def test_static_quantize_int16_matmul_error_is_bounded():
    # Same instantiation of quantized_mac_bound's general lemma as the UINT8
    # file's own test -- see module docstring for why this is a genuine
    # re-confirmation and not an unjustified copy: the lemma's hypotheses
    # never depend on which divisor produced Xs, only that Xs > 0 and each
    # ex[k] is bounded by Xs / 2.
    float_matmul, quantized_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - quantized_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_static_quantize_int16_matmul_bias_variant_error_is_bounded():
    # The Gemm "+ Bias" case: this pass never rewrites input index 2 either,
    # so Bias cancels out of the error term exactly as it does for the
    # UINT8 pass.
    float_matmul, quantized_matmul, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_matmul + bias) - (quantized_matmul + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_static_quantize_int16_matmul_negative_control_requires_rounding_bounds():
    # Without any rounding-error budget at all, exact equality is not a
    # theorem -- confirms the bound genuinely depends on the rounding
    # hypotheses rather than holding vacuously.
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ex = [z3.Real(f"ex{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    Xdq = [X[k] - ex[k] for k in range(_K)]
    Wdq = [W[k] - ew[k] for k in range(_K)]
    float_matmul = sum(X[k] * W[k] for k in range(_K))
    quantized_matmul = sum(Xdq[k] * Wdq[k] for k in range(_K))

    solver = z3.Solver()
    solver.add(z3.Not(float_matmul == quantized_matmul))
    assert solver.check() == z3.sat, (
        "the float and quantized computations always agree even without "
        "any rounding-error budget -- negative control is vacuous"
    )


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
    module docstring for why this is the isolating entry point (exactly
    ``QuantizeStaticInt16``, running only
    ``["static_quantize_int16_matmul", "static_quantize_int16_conv"]``).
    """
    out = onnx.ModelProto()
    out.ParseFromString(
        C.quantize_static_int16(model.SerializeToString(), activation_ranges)
    )
    return out


def _quantize_static_uint8(model, activation_ranges):
    """Same, but the sibling UINT8 pass (``quantize_static`` /
    ``QuantizeStatic``) -- used only as a concrete comparison point for the
    differential tests below (never as this file's subject under test).
    """
    out = onnx.ModelProto()
    out.ParseFromString(C.quantize_static(model.SerializeToString(), activation_ranges))
    return out


def _producer(model, output_name):
    return next(n for n in model.graph.node if output_name in n.output)


def _quantize_weight_per_channel(weight, channel_axis=1):
    """Independent numpy re-implementation of
    ``QuantizeWeightPerChannelInPlace`` (quantize_matmul_common.h), general
    enough to cover both MatMul's untransposed layout (``channel_axis=1``,
    weight ``[K, N]``) and Gemm's ``transB=1`` layout (``channel_axis=0``,
    weight ``[N, K]``): per-channel symmetric INT8 quantization, scale =
    max(|channel|) / 127 (or 1.0 for an all-zero channel), codes =
    round(w / scale) clipped to [-127, 127]. This is exactly the same
    scheme static_quantize_matmul.h uses -- this pass's weight handling is
    bit-identical to it, see the structural test below.
    """
    reduce_axis = 1 - channel_axis
    scale = np.max(np.abs(weight), axis=reduce_axis)
    scale = np.where(scale > 0, scale / 127.0, 1.0).astype(np.float32)
    scale_bcast = np.expand_dims(scale, axis=reduce_axis)
    codes = np.clip(np.round(weight / scale_bcast), -127, 127).astype(np.int8)
    return codes, scale


def _expected_asymmetric_uint16_quant_params(min_val, max_val):
    """Independent re-implementation of ``ComputeAsymmetricUint16QuantParams``
    (static_quantize_matmul.h), in float32 to match the pass's own
    arithmetic precision.
    """
    lo = min(np.float32(0.0), np.float32(min_val))
    hi = max(np.float32(0.0), np.float32(max_val))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(65535.0)
    zero_point = int(np.clip(np.round(-lo / scale), 0, 65535))
    return np.float32(scale), zero_point


def _expected_asymmetric_uint8_quant_params(min_val, max_val):
    """Same, but ``ComputeAsymmetricUint8QuantParams`` -- used only as a
    comparison point (see module docstring).
    """
    lo = min(np.float32(0.0), np.float32(min_val))
    hi = max(np.float32(0.0), np.float32(max_val))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zero_point = int(np.clip(np.round(-lo / scale), 0, 255))
    return np.float32(scale), zero_point


def _node_attr(node, name):
    return next(a.i for a in node.attribute if a.name == name)


def test_static_quantize_int16_matmul_pass_fires_and_matches_scheme():
    # Structural check: Xq is UINT16 (not UINT8), the weight quantization is
    # bit-identical to static_quantize_matmul's own INT8 per-channel scheme
    # (checked here by running BOTH passes on the same weight/range and
    # comparing their Wq/Ws outputs directly, not just each against its own
    # independent numpy re-implementation), and the activation scale
    # relationship the corollary above proves in the abstract
    # (Xs_8 == 257 * Xs_16) holds concretely for the real compiled passes'
    # own numbers too.
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

    quantized16 = _quantize_static_int16(model, {"X": (-5.0, 10.0)})
    # opset 21 also satisfies the UINT8 pass's own opset >= 13 requirement,
    # so this comparison run is a like-for-like isolation of the sibling
    # pass, not an opset-mismatched one.
    quantized8 = _quantize_static_uint8(model, {"X": (-5.0, 10.0)})

    matmul_node = _producer(quantized16, "Y")
    assert matmul_node.op_type == "MatMul"
    xdq_name, wdq_name = matmul_node.input

    xdq_node = _producer(quantized16, xdq_name)
    assert xdq_node.op_type == "DequantizeLinear"
    xq_node = _producer(quantized16, xdq_node.input[0])
    assert xq_node.op_type == "QuantizeLinear"
    assert xq_node.input[0] == "X"
    assert xq_node.input[1:] == xdq_node.input[1:]

    wdq_node = _producer(quantized16, wdq_name)
    assert wdq_node.op_type == "DequantizeLinear"
    assert len(wdq_node.input) == 2  # symmetric: no zero_point input
    assert _node_attr(wdq_node, "axis") == 1  # MatMul, untransposed: axis 1

    init16 = {i.name: i for i in quantized16.graph.initializer}
    x_scale16 = numpy_helper.to_array(init16[xq_node.input[1]])
    x_zp16 = numpy_helper.to_array(init16[xq_node.input[2]])
    expected_scale16, expected_zp16 = _expected_asymmetric_uint16_quant_params(
        -5.0, 10.0
    )
    assert x_zp16.dtype == np.uint16
    assert int(x_zp16) == expected_zp16
    assert expected_zp16 != 0, (
        "test calibration range must exercise a nonzero zero-point"
    )
    np.testing.assert_allclose(float(x_scale16), float(expected_scale16), rtol=1e-6)

    wq16 = numpy_helper.to_array(init16[wdq_node.input[0]])
    ws16 = numpy_helper.to_array(init16[wdq_node.input[1]])
    expected_wq, expected_ws = _quantize_weight_per_channel(weight, channel_axis=1)
    np.testing.assert_array_equal(wq16, expected_wq)
    np.testing.assert_allclose(ws16, expected_ws, rtol=1e-6)

    # Bit-identical to the sibling UINT8 pass's own weight quantization.
    matmul_node8 = _producer(quantized8, "Y")
    wdq_node8 = _producer(quantized8, matmul_node8.input[1])
    init8 = {i.name: i for i in quantized8.graph.initializer}
    wq8 = numpy_helper.to_array(init8[wdq_node8.input[0]])
    ws8 = numpy_helper.to_array(init8[wdq_node8.input[1]])
    np.testing.assert_array_equal(wq16, wq8)
    np.testing.assert_array_equal(ws16, ws8)

    # The proved corollary, concretely: this pass's real activation scale is
    # exactly 1/257 of the sibling UINT8 pass's real activation scale for
    # the identical calibrated range.
    xdq_node8 = _producer(quantized8, matmul_node8.input[0])
    xq_node8 = _producer(quantized8, xdq_node8.input[0])
    x_scale8 = numpy_helper.to_array(init8[xq_node8.input[1]])
    np.testing.assert_allclose(float(x_scale8), 257.0 * float(x_scale16), rtol=1e-6)


def test_static_quantize_int16_matmul_gemm_transb_bias_variant():
    # "vanilla Gemm handled identically, bias untouched", per this pass's
    # own doc comment: PyTorch's nn.Linear layout, weight [N, K],
    # transB=1 -- channel_axis becomes 0, and Bias is passed through
    # unchanged (same initializer name still feeds the Gemm node).
    rng = np.random.default_rng(5)
    rows, K, N = 3, 4, 6
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.6
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

    quantized = _quantize_static_int16(model, {"X": (0.0, 4.0)})
    gemm_node = _producer(quantized, "Y")
    assert gemm_node.op_type == "Gemm"
    xdq_name, wdq_name, bias_name = gemm_node.input
    assert bias_name == "B"  # bias input left completely untouched

    wdq_node = _producer(quantized, wdq_name)
    assert wdq_node.op_type == "DequantizeLinear"
    assert _node_attr(wdq_node, "axis") == 0  # transB: W is [N, K], channel axis 0

    init = {i.name: i for i in quantized.graph.initializer}
    wq = numpy_helper.to_array(init[wdq_node.input[0]])
    ws = numpy_helper.to_array(init[wdq_node.input[1]])
    expected_wq, expected_ws = _quantize_weight_per_channel(weight, channel_axis=0)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)

    xdq_node = _producer(quantized, xdq_name)
    xq_node = _producer(quantized, xdq_node.input[0])
    x_zp = numpy_helper.to_array(init[xq_node.input[2]])
    assert x_zp.dtype == np.uint16


def test_static_quantize_int16_matmul_output_is_close_to_float_within_proved_bound():
    # Differential check: run the real UINT16-activation quantized graph
    # through onnxruntime and confirm every output element's error against
    # the true float MatMul stays within the bound proved above -- and is
    # meaningfully tighter than what the sibling UINT8 pass's bound (and its
    # real numeric error) would be for the identical inputs/weight/range.
    rng = np.random.default_rng(1)
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

    x_min, x_max = float(x.min()), float(x.max())
    quantized16 = _quantize_static_int16(model, {"X": (x_min, x_max)})
    quantized8 = _quantize_static_uint8(model, {"X": (x_min, x_max)})

    # Graph optimization disabled: onnxruntime can fuse a QDQ-shaped MatMul
    # chain like this one into a hardware-specific code path different from
    # the literal node chain each pass's proof reasons about -- see
    # `tests/test_ort_matmul_nbits_workaround.py`'s docstring for this
    # suite's existing precedent of a real ORT graph-optimization fusion bug
    # of exactly this shape. Disabling optimization executes each graph
    # exactly as its pass produced it.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess16 = ort.InferenceSession(
        quantized16.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    (y_quant16,) = sess16.run(None, {"X": x})
    sess8 = ort.InferenceSession(
        quantized8.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    (y_quant8,) = sess8.run(None, {"X": x})

    x_scale16, _x_zp16 = _expected_asymmetric_uint16_quant_params(x_min, x_max)
    x_scale8, _x_zp8 = _expected_asymmetric_uint8_quant_params(x_min, x_max)
    _wq, ws = _quantize_weight_per_channel(weight, channel_axis=1)

    y_float = x @ weight
    error16 = np.abs(y_float - y_quant16)
    error8 = np.abs(y_float - y_quant8)

    eps_x16 = float(x_scale16) / 2.0
    eps_x8 = float(x_scale8) / 2.0
    eps_w = ws / 2.0  # shape [N], identical for both passes

    def _bound(eps_x):
        return (
            eps_x * np.abs(weight).sum(axis=0)[np.newaxis, :]
            + eps_w[np.newaxis, :] * np.abs(x).sum(axis=1)[:, np.newaxis]
            + K * eps_x * eps_w[np.newaxis, :]
        )

    bound16 = _bound(eps_x16)
    bound8 = _bound(eps_x8)

    assert np.all(error16 <= bound16 + 1e-6)
    # The concrete comparison this pass exists for: its own worst-case bound
    # is dramatically tighter than the UINT8 sibling's would be for the
    # identical case, purely from the finer activation step (eps_x16 is
    # exactly eps_x8 / 257, matching the proved corollary).
    assert np.all(bound16 < bound8)
    np.testing.assert_allclose(eps_x8, 257.0 * eps_x16, rtol=1e-6)
    # And the REAL onnxruntime-executed error, not just the worst-case
    # bound, is also meaningfully smaller in practice for this well-scaled
    # case.
    assert np.linalg.norm(error16) < np.linalg.norm(error8)

    # Loosened beyond a pure worst-case-bound tolerance -- the weight's own
    # INT8 quantization (unchanged from the UINT8 pass) still dominates the
    # error here, so this is a "meaningfully close" sanity check, not the
    # rigorous claim; the actual rigorous check is the proved bound above.
    np.testing.assert_allclose(y_quant16, y_float, rtol=0.03, atol=1e-2)


def test_static_quantize_int16_matmul_declines_without_calibrated_range():
    rng = np.random.default_rng(2)
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

    quantized = _quantize_static_int16(model, {})
    assert [n.op_type for n in quantized.graph.node] == ["MatMul"]


def test_static_quantize_int16_matmul_declines_at_opset13_though_uint8_pass_fires_there():
    # The pass-specific opset asymmetry this file's docstring calls out:
    # UINT16 QuantizeLinear/DequantizeLinear needs opset >= 21, so this pass
    # leaves a plain MatMul alone at opset 13 -- even though, for the exact
    # same model and calibrated range, the sibling UINT8 pass (opset >= 13
    # only) DOES fire there. Concretely demonstrating both sides (rather
    # than only the decline) is what confirms this is a genuine, pass-
    # specific opset gate and not merely "old opsets are unsupported by
    # this whole family".
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
        opset=13,
        ir_version=8,
    )

    declined = _quantize_static_int16(model, {"X": (-5.0, 10.0)})
    assert [n.op_type for n in declined.graph.node] == ["MatMul"]

    fired = _quantize_static_uint8(model, {"X": (-5.0, 10.0)})
    fired_ops = [n.op_type for n in fired.graph.node]
    assert "QuantizeLinear" in fired_ops
    assert "DequantizeLinear" in fired_ops
