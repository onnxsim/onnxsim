"""Formal check for StaticQuantizeMatMul (opt-in; onnxsim's own
``onnxsim/passes/static_quantize_matmul.h``): the calibration-based sibling
of ``dynamic_quantize_matmul.h``
(``test_formal_verify_dynamic_quantize_matmul.py`` is this file's closest
template for the overall Z3-vocabulary shape) and
``weight_only_quantize_matmul.h``. It rewrites ``Y = MatMul(X, W)`` (or a
"vanilla" ``Gemm``, bias left untouched) -- ``W`` a constant 2-D FLOAT32
tensor, ``X`` FLOAT32 -- into QDQ ("quantize/dequantize") format, leaving the
MatMul/Gemm node itself untouched and rewiring only its inputs::

    Xq  = QuantizeLinear(X, Xs, Xzp)          -- Xs/Xzp: CALIBRATED, fixed
    Xdq = DequantizeLinear(Xq, Xs, Xzp)
    Wdq = DequantizeLinear(Wq, Ws, axis=<channel_axis>)   -- symmetric, zp=0
    Y   = MatMul(Xdq, Wdq)

``Wq``/``Ws`` are the same per-output-channel symmetric INT8 weight
quantization every sibling pass in this family uses
(``QuantizeWeightPerChannelInPlace``). The genuinely new wrinkle here --
absent from every other pass this suite has proved so far, all of which use
a *symmetric* (``zero_point = 0``) scheme -- is ``Xs``/``Xzp``: the
activation's scale and zero-point are a *calibrated*, fixed ``(min, max)``
range (supplied by the caller, e.g. via ``onnxsim.calibration.quantize_static``
running representative data through the float model), quantized
*asymmetrically* to UINT8 via ``ComputeAsymmetricUint8QuantParams``:

    lo    = min(0, min_val)         hi = max(0, max_val)   # widened to include 0
    scale = (hi - lo) / 255
    zero_point = clamp(round(-lo / scale), 0, 255)

Widening the calibrated range to always include 0 (so a zero activation is
exactly representable) is what makes ``zero_point`` genuinely nonzero in
general -- e.g. a post-ReLU-like one-sided range ``[2, 10]`` still gets
``lo = 0`` (no shift needed), but an range like ``[-5, 10]`` that already
straddles 0 gives ``zero_point = round(5 / scale) = 85 != 0``. Every prior
pass in this family (``dynamic_quantize_matmul``, ``weight_only_quantize_
matmul``) only ever has ``zero_point = 0`` (dynamic quantization's own
``DynamicQuantizeLinear`` *does* compute an asymmetric zero-point at
runtime, but that pass's own proof deliberately works with its per-tap error
variables directly, never a literal zero-point term -- see below for why
this file does carry one explicitly).

Soundness claim
================
Unlike ``weight_only_quantize_matmul`` (whose bound collapses to a
single-operand special case, ``eps_x := 0``, since ``X`` is never quantized
there), this pass quantizes BOTH operands -- exactly ``quantized_mac_bound``'s
own general two-operand shape, with ``eps_x := Xs / 2`` and
``eps_w := Ws[n] / 2`` both genuinely nonzero::

    |MatMul(X, W)[i, n] - MatMul(Xdq, Wdq)[i, n]| <=
        (Xs / 2) * sum_k |W[k, n]| + (Ws[n] / 2) * sum_k |X[i, k]|
        + K * (Xs / 2) * (Ws[n] / 2)

This is actually a MORE direct instance of ``quantized_mac_bound``'s lemma
than ``dynamic_quantize_matmul``'s own proof is: that pass's literal node
chain computes ``Cast<float>(Acc) * (Xs * Ws)`` (an integer accumulator
rescaled after the fact), so its proof needs a separate ring-identity lemma
connecting that node chain to the elementwise-dequantized dot product before
the bound lemma even applies. Here, ``Y = MatMul(Xdq, Wdq)`` -- the pass's
own literal node chain -- computes the elementwise-dequantized dot product
directly, with no separate integer accumulator/rescale step, so
``quantized_mac_bound``'s bound lemma applies to the pass's actual output
with no intermediate identity lemma needed at all. The proof below is
therefore a direct, self-contained (own Z3 vocabulary, per this suite's
convention) instantiation of that already-proved lemma at a concrete
``_K = 2`` (matching ``quantized_mac_bound``'s and ``dynamic_quantize_
matmul``'s own choice -- enough to exercise the cross-tap sum, and this
shape of nonlinear query was already shown, in ``dynamic_quantize_matmul``'s
own investigation, to blow up well past 60s at ``_K`` this small if the
formulation is wrong), built from the start in the direct-error-variable
(``ex``/``ew``) idiom ``quantized_mac_bound`` itself uses -- not the
Xq/Xzp/Wq-multiplicand shape ``dynamic_quantize_matmul``'s FIRST (buggy,
since-fixed) attempt used, which is exactly what caused that blowup there.

Carrying the nonzero zero-point through the derivation
========================================================
The per-element hypothesis the bound proof above needs is ``X``'s own
round-trip bound, ``|X[i, k] - Xdq[i, k]| <= Xs / 2``. This is
``test_formal_verify_quantize_round_trip.py``'s own lemma (which is already
zero-point-generic there -- its proof shows ``zero_point`` cancels exactly
regardless of value), but since every OTHER pass in this file family only
ever instantiates it at ``zero_point = 0``, this file reproves it here, in
its own ``Xs``/``Xzp`` vocabulary, with ``Xzp`` a genuinely free (and, in the
differential tests below, genuinely nonzero) variable -- so this proof
actually carries a nonzero zero-point term through its own derivation rather
than assuming the symmetric case away. The algebra shows exactly why the
bound doesn't care: dequantizing subtracts back out exactly the zero-point
that quantizing added, for ANY zero-point value, as long as nothing clips.

That "as long as nothing clips" matters: ``QuantizeLinear``'s round-trip
bound of ``scale / 2`` holds only while the true value's quantized code
lands inside UINT8's ``[0, 255]`` range without being clamped -- a real
value outside the calibrated ``[min, max]`` range (calibration data that
doesn't cover the actual data seen at inference) CAN clip, and clipping can
introduce unbounded error (the same explicit side condition
``test_formal_verify_quantize_round_trip.py`` calls out for its own lemma).
This file's round-trip lemma below states "no clipping" as an explicit
hypothesis rather than silently assuming it, and the differential tests
calibrate from the actual data range being tested (so nothing clips there
either) rather than asserting it away.

No consumer-composition step is added here, matching this suite's usual
reasoning for a numeric *bound* (as opposed to an *equality*) claim: an
arbitrary consumer need not be Lipschitz, so a bound on
``|MatMul(X, W) - MatMul(Xdq, Wdq)|`` does not in general bound anything
about ``|consumer(MatMul(X, W)) - consumer(MatMul(Xdq, Wdq))|``.
``quantized_mac_bound``, ``dynamic_quantize_matmul`` and
``weight_only_quantize_matmul`` all skip this step for the same reason.

Invoking the pass with a specific calibrated range
====================================================
The typical end-user entry point, ``onnxsim.calibration.quantize_static``,
derives the calibration range itself by running representative
``calibration_data`` through the float model -- there is no argument letting
a caller hand it a specific, exact ``(min, max)`` per tensor directly. One
layer down, though, is the actual nanobind-exposed entry point that function
calls: ``onnxsim.onnxsim_cpp2py_export.quantize_static(model_bytes,
activation_ranges)``, where ``activation_ranges`` is precisely a ``{tensor
name: (min, max)}`` dict -- exactly the calibrated-range shape
``StaticQuantizationCalibrationRanges()`` (this pass's own C++ global,
overwritten by this call) stores. This is more direct and deterministic for
a small test than fabricating calibration *data* that happens to produce a
desired range, so the differential tests below call it directly. It runs
``OptimizeFixed`` with exactly ``["static_quantize_matmul",
"static_quantize_conv"]`` -- i.e. it already isolates this pass family
without any extra ``skipped_optimizers``/``simplify_isolated_extra``
machinery, and, unlike ``onnxsim.simplify``, applies the rewrite with no
shape inference or other simplification alongside it, so this suite's usual
``simplify_isolated_extra``/``check_n=0`` pattern isn't used here -- there is
no ``onnxsim.simplify`` random-input equivalence check to run in the first
place. The differential numeric-bound test instead checks the proved bound
directly against onnxruntime's own execution of the rewritten graph.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
import onnxsim.onnxsim_cpp2py_export as C
from _formal_verify_common import producer, prove, z3
from onnx import parser

_K = 2  # concrete number of contraction taps -- see module docstring for why
# this matches quantized_mac_bound's/dynamic_quantize_matmul's own _K rather
# than a larger value.


def _abs(v):
    return z3.If(v >= 0, v, -v)


def test_static_quantize_matmul_activation_round_trip_holds_for_any_zero_point():
    # Carries a genuinely free (including nonzero) Xzp through the standard
    # QuantizeLinear/DequantizeLinear round-trip derivation, in this file's
    # own Xs/Xzp vocabulary -- test_formal_verify_quantize_round_trip.py
    # proves the same fact (also zero-point-generic there), but every OTHER
    # pass in this family only ever instantiates it at zero_point = 0, so
    # this file reproves it here rather than silently assuming the
    # symmetric case. `n` models QuantizeLinear's round(x / Xs + Xzp): *some*
    # integer within 0.5 of it (any correct rounding rule, not just one
    # specific tie-break), and the "no clipping" side condition is: `n`
    # itself (not some separately-clamped code) is what DequantizeLinear
    # reads back -- i.e. round(x / Xs + Xzp) already lands in [0, 255]
    # without needing to be clamped there.
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
    claim, mirroring ``quantized_mac_bound``'s own ``ew``/``ex`` idiom (and
    ``dynamic_quantize_matmul``'s final, fixed formulation of it) exactly:
    each operand's dequantization ERROR is a free Real bounded directly by
    its own rounding hypothesis, rather than being re-derived from separate
    quantized-code/scale/zero-point multiplicands. Unlike
    ``dynamic_quantize_matmul``, no separate ring-identity lemma is needed to
    connect this to the pass's own node chain -- ``Y = MatMul(Xdq, Wdq)`` IS
    exactly the elementwise-dequantized dot product ``quantized_matmul``
    below, with no integer-accumulator rescale step in between.

    Returns ``(float_matmul, quantized_matmul, rounding_bounds, bound)``.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # X[i, k], true float activation
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[k, n], true float weight
    ex = [z3.Real(f"ex{k}") for k in range(_K)]  # X[i, k] - Xdq[i, k]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[k, n] - Wdq[k, n]
    Xs = z3.Real("Xs")  # calibrated per-tensor activation scale
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


def test_static_quantize_matmul_error_is_bounded():
    # The genuine bounded-error claim, a direct instantiation of
    # quantized_mac_bound's own lemma with both eps_x := Xs / 2 and
    # eps_w := Ws / 2 nonzero (unlike weight_only_quantize_matmul's
    # single-operand eps_x := 0 special case): given each operand's own
    # rounding bound, the true float dot product and MatMul(Xdq, Wdq) --
    # the pass's actual literal output -- cannot differ by more than `bound`.
    float_matmul, quantized_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - quantized_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_static_quantize_matmul_bias_variant_error_is_bounded():
    # The Gemm "+ Bias" case: the pass leaves Bias completely untouched
    # (runTransform never rewrites input index 2), so adding the same
    # Bias(n) to both the true and the quantized computation leaves their
    # difference -- and therefore the bound on it -- unchanged, since Bias
    # cancels out of the error term algebraically.
    float_matmul, quantized_matmul, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_matmul + bias) - (quantized_matmul + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_static_quantize_matmul_negative_control_requires_rounding_bounds():
    # Sanity check that the bound proved above is genuine, not vacuous:
    # without ANY error budget (no rounding-error hypothesis at all), the
    # exact-equality claim float_matmul == quantized_matmul is not a
    # theorem -- Z3 must find a real counterexample, confirming the
    # rounding really does introduce error rather than the two computations
    # always coinciding regardless.
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
    is used here: it takes a caller-supplied ``{name: (min, max)}`` dict
    directly, is exactly this pass family's own isolation (``OptimizeFixed``
    with only ``static_quantize_matmul``/``static_quantize_conv``), and
    needs no calibration *data* to be fabricated.
    """
    out = onnx.ModelProto()
    out.ParseFromString(C.quantize_static(model.SerializeToString(), activation_ranges))
    return out


def _quantize_weight_per_channel(weight):
    """Independent numpy re-implementation of
    ``QuantizeWeightPerChannelInPlace`` (quantize_matmul_common.h) for a
    plain (non-transposed) MatMul weight: per-output-column (axis 1)
    symmetric INT8 quantization, scale = max(|column|) / 127 (or 1.0 for an
    all-zero column), codes = round(w / scale) clipped to [-127, 127] --
    the same formula (and, for this axis, the same numbers)
    ``QuantizeWeightPerChannelKN`` uses, which
    ``test_formal_verify_dynamic_quantize_matmul.py`` already validates for
    the sibling dynamic pass.
    """
    scale = np.max(np.abs(weight), axis=0)
    scale = np.where(scale > 0, scale / 127.0, 1.0).astype(np.float32)
    codes = np.clip(np.round(weight / scale[np.newaxis, :]), -127, 127).astype(np.int8)
    return codes, scale


def _expected_asymmetric_uint8_quant_params(min_val, max_val):
    """Independent re-implementation of ``ComputeAsymmetricUint8QuantParams``
    (static_quantize_matmul.h), in float32 to match the pass's own
    arithmetic precision.
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


def test_static_quantize_matmul_pass_fires_and_matches_scheme():
    # Build a plain float MatMul and run the real pass with a calibration
    # range chosen so the zero-point comes out genuinely NONZERO -- min=-5,
    # max=10 straddles 0, so lo=-5 (not widened) and
    # zero_point = round(5 / scale) = 85 != 0. This is the single most
    # important differential check for the asymmetric-quantization content
    # this pass introduces that its symmetric-only siblings don't have.
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

    quantized = _quantize_static(model, {"X": (-5.0, 10.0)})

    # Walk the chain backward from the real graph output: MatMul (untouched
    # node) <- DequantizeLinear(Xq) <- QuantizeLinear(X), and
    # MatMul <- DequantizeLinear(Wq, Ws, axis).
    matmul_node = producer(quantized, "Y")
    assert matmul_node.op_type == "MatMul"
    xdq_name, wdq_name = matmul_node.input

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
    assert _node_attr(wdq_node, "axis") == 1  # MatMul, untransposed: axis 1 ([K, N])

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
    expected_wq, expected_ws = _quantize_weight_per_channel(weight)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_static_quantize_matmul_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring quantized_mac_bound's/dynamic_quantize_
    # matmul's own test_..._matches_onnxruntime: run the real quantized
    # graph through onnxruntime and confirm every output element's error
    # against the true float MatMul stays within the bound proved above.
    # The calibration range is set to X's own actual (min, max) so nothing
    # clips -- the round-trip lemma's explicit side condition.
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
    quantized = _quantize_static(model, {"X": (x_min, x_max)})

    # Graph optimization disabled: by default onnxruntime silently fuses this
    # QDQ chain (QuantizeLinear -> DequantizeLinear x2 -> MatMul) into a
    # hardware-specific `MatMulIntegerToFloat` contrib kernel (confirmed via
    # `sess_options.optimized_model_filepath`) -- a different code path than
    # the literal node chain this pass's proof reasons about, and one whose
    # own numeric behavior isn't part of this claim (see
    # `onnxsim/ort_matmul_nbits_workaround.py`'s docstring and
    # `tests/test_ort_matmul_nbits_workaround.py` for this suite's existing
    # precedent of a real ORT graph-optimization fusion bug of exactly this
    # shape). Disabling optimization here executes the graph exactly as the
    # pass produced it.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        quantized.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    (y_quant,) = sess.run(None, {"X": x})

    x_scale, _x_zp = _expected_asymmetric_uint8_quant_params(x_min, x_max)
    _wq, ws = _quantize_weight_per_channel(weight)

    y_float = x @ weight
    error = np.abs(y_float - y_quant)

    eps_x = float(x_scale) / 2.0
    eps_w = ws / 2.0  # shape [N]
    bound = (
        eps_x * np.abs(weight).sum(axis=0)[np.newaxis, :]
        + eps_w[np.newaxis, :] * np.abs(x).sum(axis=1)[:, np.newaxis]
        + K * eps_x * eps_w[np.newaxis, :]
    )
    assert np.all(error <= bound + 1e-6)

    # Also confirm the rewrite is meaningfully close (not merely "within a
    # loose worst-case bound") for these well-scaled inputs -- a few percent
    # relative error, consistent with INT8/UINT8 quantization, not bitwise
    # equal. atol is loosened beyond a pure-rtol bound (one output element
    # near zero would otherwise need a tighter tolerance for a small
    # *absolute* error that is still a large *relative* one); the actual
    # rigorous check is the proved worst-case bound above, already asserted.
    np.testing.assert_allclose(y_quant, y_float, rtol=0.05, atol=1e-2)


def test_static_quantize_matmul_declines_without_calibrated_range():
    # patternMatchPredicate's calibration-range check: an activation tensor
    # with no entry in activation_ranges is left completely alone -- the
    # simplest, and most asymmetric-quantization-specific, of the pass's
    # decline conditions (dynamic_quantize_matmul has no such condition at
    # all, since it never needs calibration).
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

    quantized = _quantize_static(model, {})
    assert [n.op_type for n in quantized.graph.node] == ["MatMul"]


def test_static_quantize_matmul_declines_pre_opset13():
    # DequantizeLinear's per-channel `axis` attribute needs opset >= 13
    # (patternMatchPredicate's opset check) -- unlike dynamic_quantize_matmul,
    # which only needs opset >= 11 (DynamicQuantizeLinear, no per-channel
    # axis on its own dequantization). A plain MatMul at an older opset must
    # survive untouched even with a calibrated range available.
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

    quantized = _quantize_static(model, {"X": (-5.0, 10.0)})
    assert [n.op_type for n in quantized.graph.node] == ["MatMul"]
