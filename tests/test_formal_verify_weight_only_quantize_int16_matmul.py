"""Formal check for WeightOnlyQuantizeInt16MatMul (opt-in; onnxsim's own
``onnxsim/passes/weight_only_quantize_int16_matmul.h``): structurally
IDENTICAL to ``weight_only_quantize_matmul.h``'s own INT8 scheme (this file's
primary template -- ``test_formal_verify_weight_only_quantize_matmul.py``
already proves the single-operand bound and every supporting lemma for that
pass; read it in full first). Only the constant weight is quantized (``X`` is
never touched, no calibration data, no ``DynamicQuantizeLinear``/
``QuantizeLinear`` on the activation side), via
``QuantizeWeightPerChannelInPlaceInt16`` instead of
``QuantizeWeightPerChannelInPlace``: scale = ``max(|w|) / 32767`` per output
channel instead of INT8's ``/ 127``. It also requires opset >= 21 (INT16
``QuantizeLinear``/``DequantizeLinear`` support), unlike the INT8 pass's
opset >= 13 floor.

Is the bound proof genuinely new Z3 content, or a re-instantiation?
====================================================================
The INT8 file's own ``_bound_formulas`` helper (and this file's copy of it,
below) builds ``quantized_mac_bound``'s single-operand collapse
(``eps_x := 0``, since ``X`` is never quantized) purely from a free
per-channel scale variable ``Ws`` and a per-tap rounding hypothesis
``|ew[k]| <= Ws / 2`` -- the hypotheses never mention 127, 32767, or any other
concrete divisor; the divisor only ever enters where a concrete ``Ws`` is
*computed* from a channel's ``max(|w|)``, which happens on the C++ side
(``QuantizeWeightPerChannelInPlaceInt16``) and in this file's own
differential tests (``_quantize_weight_per_channel_in_place_int16`` below),
never inside the bound lemma's Z3 vocabulary itself. So the bound-proving
queries below are mechanically identical to the INT8 file's -- confirmed, not
assumed, precisely because ``_bound_formulas`` here is copied verbatim from
it and still closes with the same hypotheses, i.e. the core per-tap rounding
bound proof does not depend on which integer width (INT8/INT16) produced the
scale, only on the free ``Ws > 0`` / ``|ew[k]| <= Ws / 2`` hypothesis shape.
What line up differently is:

1. ``weight_only_quantize_int16_matmul``-specific declination behavior
   (opset >= 21 rather than >= 13) -- tested structurally against the real
   pass below, including the concrete case where the INT8 pass would fire
   (opset 13) but this one still declines.
2. A genuinely new corollary, absent from the INT8 file entirely: for the
   SAME weight column (same ``max(|w|)`` numerator), this pass's INT16 scale
   is *exactly* ``127 / 32767`` times what the INT8 pass's scale would be for
   that column -- an exact rational identity (unlike
   ``test_formal_verify_static_quantize_int16_matmul.py``'s UINT8-vs-UINT16
   corollary, where ``255 * 257 == 65535`` exactly makes the ratio a whole
   number; here ``32767`` is not an exact multiple of ``127``
   (``127 * 258 == 32766``), so the honest statement is the exact fraction
   ``127 / 32767``, proved as a Z3 identity rather than approximated), and
   strictly finer (smaller) whenever the column is not all-zero
   (``test_weight_only_quantize_int16_matmul_weight_scale_is_127_over_32767x_int8_scale``
   below).
3. A concrete, adversarial differential test showing why this pass exists at
   all, per the header's own doc comment and ``onnxsim.estimate_quantization_
   precision``'s ``max(|w|) / median(|w|)`` outlier-ratio motivation
   (``precision_estimator.py``): a weight column with one extreme outlier and
   several small-magnitude "typical" values, chosen so INT8's coarse
   ``max/127`` step rounds every one of those typical values to exactly zero
   (complete loss, not just degraded precision), while INT16's ``max/32767``
   step preserves their sign and most of their relative magnitude
   (``test_weight_only_quantize_int16_matmul_outlier_column_int16_preserves_typical_weights_int8_loses``
   below).

Everything else -- the direct-error-variable (``ew``, no ``ex`` at all since
there is nothing for it to model) idiom, ``_K = 2``, the
``DequantizeLinear``-axis-semantics lemma, the Gemm "+ Bias" cancellation, the
vacuity negative control -- mirrors the INT8 file exactly; see it for the
fuller exposition of each, and
``test_formal_verify_dynamic_quantize_matmul.py``'s own module docstring for
why the direct-error-variable idiom (never reconstructed from separate
code/scale multiplicands) is used from the start rather than discovered by
trial and error after a slow query.

Differential tests build a plain float ``MatMul``/``Gemm`` via ``onnx.parser``
(per ``CLAUDE.md``), run the real pass alone via ``simplify_isolated_extra``
(and, for the numeric-bound check, via :func:`onnxsim.quantize_weight_only_int16`,
which applies exactly this rewrite -- see its own docstring in
``onnx_simplifier.py``; no dedicated Python entry point search was needed
beyond confirming that one already exists), and confirm: the exact
``MatMul(X, DequantizeLinear(Wq, Ws, axis=<channel_axis>))`` chain fires with
the expected ``channel_axis``; ``Wq`` is truly INT16 and matches an
independent numpy re-implementation of
``QuantizeWeightPerChannelInPlaceInt16``'s formula; ``X`` is passed through
unchanged; the actual runtime output (onnxruntime, graph optimization
explicitly disabled -- see the note below) stays within the bound proved
above; and a pre-opset-21 model is declined outright, even at opset 13 where
the sibling INT8 pass fires.

A note on disabling onnxruntime graph optimization
====================================================
Every ``InferenceSession`` built here to check a numeric bound against real
execution explicitly sets
``so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL``.
This suite previously found and fixed a real bug
(``tests/test_formal_verify_static_quantize_matmul.py``,
``tests/test_ort_matmul_nbits_workaround.py``'s own docstring) where
onnxruntime's default optimization level silently fuses a
``QuantizeLinear``/``DequantizeLinear``/``MatMul``-shaped graph into a
hardware-specific ``MatMulIntegerToFloat`` contrib kernel -- a DIFFERENT code
path than what these proofs reason about. This pass's own shape (a plain
``DequantizeLinear``-only weight-only rewrite, no ``QuantizeLinear`` on the
activation side at all) is less likely to trigger that specific fusion, but
optimization is disabled defensively anyway: it costs nothing, and the pass
leaves the ``MatMul``/``Gemm`` node itself in the graph for onnxruntime to
see.

The firing/bound tests below pass ``check_n=0`` to ``simplify_isolated_extra``
for the same reason the INT8 file's own tests do: this is a genuinely lossy
rewrite, and onnxsim's own random-input equivalence check (default
``check_n=3``, tolerance ``rtol=1e-4``/``atol=1e-5``) is tighter than a
quantized rewrite -- even INT16's -- can guarantee in general.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

_K = 2  # concrete number of contraction taps -- matches quantized_mac_bound's
# (and the INT8 sibling file's) own _K.


def _abs(v):
    return z3.If(v >= 0, v, -v)


def _bound_formulas():
    """Verbatim copy of weight_only_quantize_matmul's own helper (the
    single-operand ``eps_x := 0`` collapse of quantized_mac_bound's general
    lemma) -- see that file's docstring for the full derivation. Kept as a
    literal copy (not a shared import) per this suite's per-file
    self-containment convention; its being unchanged is itself the point --
    see this file's module docstring for why no new nonlinear query is
    needed for the bound itself, only for the width-comparison corollary and
    the differential outlier test below.
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


def test_weight_only_quantize_int16_matmul_error_is_bounded():
    # Same instantiation of quantized_mac_bound's single-operand collapse as
    # the INT8 sibling's own test -- see module docstring for why this is a
    # genuine re-confirmation, not an unjustified copy: the lemma's
    # hypotheses never depend on which integer width (or divisor) produced
    # Ws, only that Ws > 0 and each ew[k] is bounded by Ws / 2.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_weight_only_quantize_int16_matmul_bias_variant_error_is_bounded():
    # The "+ Bias" branch (a Gemm with a bias input): this pass never
    # touches Gemm's bias C at all (identical to the INT8 sibling), so
    # adding the same Bias(n) to both the true and the dequantized
    # computation leaves their difference -- and therefore the bound on
    # it -- unchanged.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_matmul + bias) - (dequant_matmul + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_weight_only_quantize_int16_matmul_negative_control_requires_rounding_bound():
    # Sanity check that the bound proved above is genuine, not vacuous: with
    # no error budget assumed on ew at all (only Ws > 0), the same bound is
    # not a theorem -- Z3 must find a real counterexample.
    float_matmul, dequant_matmul, _rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul

    solver = z3.Solver()
    solver.add(z3.Not(z3.And(error <= bound, -error <= bound)))
    assert solver.check() == z3.sat, (
        "the bound holds even without any rounding-error budget on ew -- "
        "negative control is vacuous"
    )


def test_weight_only_quantize_int16_matmul_dequantizelinear_axis_semantics_matches_rounding_bound():
    # This pass leaves an actual DequantizeLinear(Wq, Ws, axis=channel_axis)
    # node in the graph (INT16 code, still implicit zero_point=0 --
    # symmetric). Confirms DequantizeLinear's own defining per-channel-axis
    # formula (Wdq[k, n] = Wq[k, n] * Ws[n]) is exactly the quantity the
    # rounding bound above assumes -- i.e. that Wq being "some integer code
    # within 0.5 of W[k, n] / Ws[n]" (round-to-nearest, no saturation) is
    # what makes DequantizeLinear's literal output satisfy
    # |W[k, n] - Wdq[k, n]| <= Ws[n] / 2 -- not assumed for free. Nothing in
    # this derivation is INT8/INT16-specific: it is stated purely in terms
    # of a free integer code wq and a free positive scale ws.
    w, ws = z3.Reals("w ws")
    wq = z3.Int("wq")  # round(w / ws): some integer within 0.5 of w / ws.
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        ws > 0,
        wq - w / ws <= half,
        w / ws - wq <= half,
    )
    wdq = z3.ToReal(wq) * ws

    error = w - wdq
    prove(z3.Implies(hypotheses, z3.And(error <= ws / 2, -error <= ws / 2)))


def test_weight_only_quantize_int16_matmul_weight_scale_is_127_over_32767x_int8_scale():
    # The genuinely new corollary this file adds on top of the INT8 file's
    # machinery: for the SAME weight column (same numerator m = max(|w|)),
    # this pass's INT16 scale (divisor 32767) is EXACTLY 127 / 32767 times
    # what weight_only_quantize_matmul's INT8 scale (divisor 127) would be
    # for that same column. Unlike test_formal_verify_static_quantize_int16_
    # matmul.py's UINT8-vs-UINT16 corollary (255 * 257 == 65535 exactly, a
    # whole-number ratio), 32767 is NOT an exact multiple of 127
    # (127 * 258 == 32766, one short) -- so the honest, exact statement is
    # the rational identity Ws16 == (127 / 32767) * Ws8, proved here as a Z3
    # identity rather than approximated as "about 258x finer". Whenever the
    # column is not all-zero (m > 0, guaranteed by both quantization
    # functions' shared "scale = max(|w|) / divisor, or 1.0 for an all-zero
    # channel" convention), the INT16 scale is also strictly smaller -- the
    # weight's own contribution to the bound (eps_w = Ws / 2) shrinks by the
    # same exact factor.
    m = z3.Real("m")  # max(|w|) for one output channel, shared numerator
    Ws16 = m / 32767
    Ws8 = m / 127
    ratio = z3.RealVal(127) / 32767
    prove(
        z3.Implies(
            m > 0,
            z3.And(Ws16 == ratio * Ws8, Ws16 < Ws8, Ws16 > 0, Ws8 > 0),
        )
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


def _dequantizelinear_node(model, output_name):
    node = producer(model, output_name)
    assert node.op_type == "DequantizeLinear"
    return node


def _axis(node):
    return next(a.i for a in node.attribute if a.name == "axis")


def _quantize_weight_per_channel_in_place_int16(weight, channel_axis):
    """Independent numpy re-implementation of
    ``QuantizeWeightPerChannelInPlaceInt16`` (quantize_matmul_common.h):
    per-``channel_axis`` symmetric INT16 quantization *in weight's own
    layout* (no transpose), scale = max(|channel|) / 32767 (or 1.0 for an
    all-zero channel), codes = round(w / scale) clipped to [-32767, 32767].
    """
    reduce_axis = 1 - channel_axis
    scale = np.max(np.abs(weight), axis=reduce_axis)
    scale = np.where(scale > 0, scale / 32767.0, 1.0).astype(np.float32)
    scale_bcast = np.expand_dims(scale, axis=reduce_axis)
    codes = np.clip(np.round(weight / scale_bcast), -32767, 32767).astype(np.int16)
    return codes, scale


def _quantize_weight_per_channel_in_place_int8(weight, channel_axis):
    """Same, but weight_only_quantize_matmul's own INT8 scheme -- used only
    as a concrete comparison point for the outlier differential test below
    (never as this file's subject under test).
    """
    reduce_axis = 1 - channel_axis
    scale = np.max(np.abs(weight), axis=reduce_axis)
    scale = np.where(scale > 0, scale / 127.0, 1.0).astype(np.float32)
    scale_bcast = np.expand_dims(scale, axis=reduce_axis)
    codes = np.clip(np.round(weight / scale_bcast), -127, 127).astype(np.int8)
    return codes, scale


def test_weight_only_quantize_int16_matmul_pass_fires_and_matches_scheme():
    # Build a plain float MatMul and run the real weight_only_quantize_
    # int16_matmul rewrite alone, then confirm both the node chain's shape
    # (MatMul(X, DequantizeLinear(Wq, Ws, axis=1))) and the quantized
    # weight's exact values -- and that Wq is truly INT16, not INT8.
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
    # check does not apply to a genuinely lossy quantization rewrite.
    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int16_matmul", check_n=0
    )

    matmul_node = producer(sim_model, "Y")
    assert matmul_node.op_type == "MatMul"
    x_input, w_input = matmul_node.input
    # X (the activation) is passed through completely unchanged.
    assert x_input == "X"

    dql_node = _dequantizelinear_node(sim_model, w_input)
    assert _axis(dql_node) == 1
    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in sim_model.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in sim_model.graph.initializer if i.name == ws_name)
    wq = numpy_helper.to_array(wq_init)
    ws = numpy_helper.to_array(ws_init)
    assert wq.dtype == np.int16, "weight must be quantized to INT16, not INT8"
    assert wq.shape == (K, N)
    assert ws.shape == (N,)

    expected_wq, expected_ws = _quantize_weight_per_channel_in_place_int16(weight, 1)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_weight_only_quantize_int16_matmul_gemm_transb_uses_channel_axis_zero():
    # PyTorch nn.Linear layout: weight [N, K], Gemm(X, W, B, transB=1). The
    # header comment's "axis 0 when Gemm's transB made W [N, K]" -- confirm
    # channel_axis differs from the plain-MatMul case above, and Wq/Ws match
    # the same independent numpy scheme with channel_axis=0, and Bias is
    # left untouched.
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
        model, "weight_only_quantize_int16_matmul", check_n=0
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
    assert wq.dtype == np.int16
    assert wq.shape == (N, K)
    assert ws.shape == (N,)

    expected_wq, expected_ws = _quantize_weight_per_channel_in_place_int16(weight, 0)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_weight_only_quantize_int16_matmul_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring the INT8 sibling's own analogous test:
    # run the real quantized graph through onnxruntime (graph optimization
    # explicitly disabled -- see module docstring) and confirm every output
    # element's error against the true float MatMul stays within the bound
    # the proof above derives, and is meaningfully close for well-scaled
    # inputs (not merely "within a loose worst-case bound").
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

    quantized = onnxsim.quantize_weight_only_int16(model)
    dql_node = next(n for n in quantized.graph.node if n.op_type == "DequantizeLinear")
    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in quantized.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in quantized.graph.initializer if i.name == ws_name)
    ws = numpy_helper.to_array(ws_init)
    wq_arr = numpy_helper.to_array(wq_init)
    assert wq_arr.dtype == np.int16
    assert wq_arr.shape == (K, N)
    assert ws.shape == (N,)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        quantized.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    (y_quant,) = sess.run(["Y"], {"X": x})

    y_float = x @ weight
    error = np.abs(y_float - y_quant)

    eps_w = ws / 2.0  # shape [N]
    bound = eps_w[np.newaxis, :] * np.abs(x).sum(axis=1)[:, np.newaxis]
    assert np.all(error <= bound + 1e-6)

    # Also confirm the rewrite is meaningfully close -- consistent with
    # INT16's much finer step than INT8's, tolerances tightened accordingly
    # relative to the INT8 sibling's own analogous assertion.
    np.testing.assert_allclose(y_quant, y_float, rtol=1e-3, atol=1e-3)


def test_weight_only_quantize_int16_matmul_declines_below_opset21_though_int8_pass_fires_there():
    # INT16 QuantizeLinear/DequantizeLinear needs opset >= 21, so this pass
    # leaves a plain MatMul alone at opset 13 -- even though, for the exact
    # same model, the sibling INT8 pass (opset >= 13 only) DOES fire there.
    # Demonstrating both sides (rather than only the decline) is what
    # confirms this is a genuine, pass-specific opset gate and not merely
    # "old opsets are unsupported by this whole family".
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

    declined, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int16_matmul", check_n=0
    )
    assert [n.op_type for n in declined.graph.node] == ["MatMul"]

    fired, fired_ops = simplify_isolated_extra(
        model, "weight_only_quantize_matmul", check_n=0
    )
    assert fired_ops["DequantizeLinear"] >= 1
    fired_matmul = producer(fired, "Y")
    assert fired_matmul.op_type == "MatMul"


def test_weight_only_quantize_int16_matmul_outlier_column_int16_preserves_typical_weights_int8_loses():
    # The concrete payoff this pass exists for, per the header's own doc
    # comment and precision_estimator.py's max(|w|) / median(|w|) outlier-
    # ratio motivation: a single output channel with one extreme-outlier
    # weight and several small-magnitude "typical" weights. INT8's coarse
    # max/127 step is set entirely by the outlier, so every typical weight
    # here rounds to EXACTLY zero -- complete loss, not merely degraded
    # precision -- while INT16's max/32767 step (~258x finer) preserves
    # each typical weight's sign and most of its relative magnitude. This is
    # shown on the REAL compiled passes, not just the abstract scale ratio
    # proved above.
    outlier = 50.0
    typical = np.array([0.1, 0.15, -0.12, 0.08], dtype=np.float32)
    weight = np.concatenate([[outlier], typical]).astype(np.float32).reshape(-1, 1)
    K, N = weight.shape
    rows = 2
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    int16_model, _ops16 = simplify_isolated_extra(
        model, "weight_only_quantize_int16_matmul", check_n=0
    )
    int8_model, _ops8 = simplify_isolated_extra(
        model, "weight_only_quantize_matmul", check_n=0
    )

    dql16 = next(n for n in int16_model.graph.node if n.op_type == "DequantizeLinear")
    dql8 = next(n for n in int8_model.graph.node if n.op_type == "DequantizeLinear")

    def _wq_ws(model, dql):
        wq_name, ws_name = dql.input
        wq_init = next(i for i in model.graph.initializer if i.name == wq_name)
        ws_init = next(i for i in model.graph.initializer if i.name == ws_name)
        return numpy_helper.to_array(wq_init), numpy_helper.to_array(ws_init)

    wq16, ws16 = _wq_ws(int16_model, dql16)
    wq8, ws8 = _wq_ws(int8_model, dql8)
    assert wq16.dtype == np.int16
    assert wq8.dtype == np.int8

    # Sanity: this really is the SAME weight column feeding both passes,
    # matching the corollary's shared-numerator hypothesis.
    np.testing.assert_allclose(float(ws8[0]), outlier / 127.0, rtol=1e-6)
    np.testing.assert_allclose(float(ws16[0]), outlier / 32767.0, rtol=1e-6)

    typical_codes_int8 = wq8[1:, 0]
    typical_codes_int16 = wq16[1:, 0]

    # INT8: every typical weight rounds to exactly zero -- total loss.
    np.testing.assert_array_equal(typical_codes_int8, np.zeros(4, dtype=np.int8))

    # INT16: every typical weight keeps a nonzero code with the correct
    # sign, and its dequantized value stays within a small relative error
    # of the true weight -- meaningfully preserved, not merely "less lossy".
    assert np.all(typical_codes_int16 != 0)
    np.testing.assert_array_equal(np.sign(typical_codes_int16), np.sign(typical))

    dequant16 = typical_codes_int16.astype(np.float32) * float(ws16[0])
    relative_error_int16 = np.abs(dequant16 - typical) / np.abs(typical)
    assert np.all(relative_error_int16 < 0.05), (
        f"INT16 should preserve typical weights to within 5% relative error, "
        f"got {relative_error_int16}"
    )

    # INT8's dequantized value for every typical weight is a flat zero:
    # 100% relative error, the complete-loss failure mode this pass exists
    # to fix.
    dequant8 = typical_codes_int8.astype(np.float32) * float(ws8[0])
    np.testing.assert_array_equal(dequant8, np.zeros(4, dtype=np.float32))
