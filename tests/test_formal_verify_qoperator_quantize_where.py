"""Formal check for QOperatorQuantizeWhere (opt-in; onnxsim's own
``onnxsim/passes/qoperator_quantize_where.h``): the ternary-select analogue
of ``qoperator_quantize_elementwise.h``'s ``QLinearAdd``/``QLinearMul``
rewrite, using ONNX Runtime's "com.microsoft" contrib op ``QLinearWhere``
instead. It rewrites ``Z = Where(Cond, X, Y)`` (``Cond``: bool; ``X``, ``Y``:
both RUNTIME, non-constant float32 tensors -- a constant operand is declined,
see ``patternMatchPredicate``) into::

    Xq = QuantizeLinear(X, Xs, Xzp)                              -- CALIBRATED
    Yq = QuantizeLinear(Y, Ys, Yzp)                              -- CALIBRATED
    Zq = QLinearWhere(Cond, Xq, Xs, Xzp, Yq, Ys, Yzp, Zs, Zzp)   -- Cond passes
                                                    through unquantized (bool)
    Z  = DequantizeLinear(Zq, Zs, Zzp)                           -- CALIBRATED

Soundness claim -- a genuinely different SHAPE from every other file in this
suite so far
=============================================================================
Every other quantized-op file in this suite proves a MAC-bound claim (a
reduction/contraction: MatMul, Conv, Gemm, Attention -- error accumulates
across a contraction dimension via ``quantized_mac_bound``'s cross-tap sum).
``Where`` is not a reduction at all -- it is a per-element SELECTION -- so
there is no MAC-bound content here. The correct claim is a per-element
CASE-SPLIT composition: for any single output element, either ``Cond`` is
true there (the true output equals ``X`` at that element, and -- per
``QLinearWhere``'s own contrib-op semantics, confirmed by how this pass's
``runTransform`` plumbs ``Xq``/``Xs``/``Xzp`` straight into ``QLinearWhere``
as the "true" operand triple, mirroring ``Where``'s own ONNX semantics of
selecting the WHOLE typed value (not reinterpreting bits) at each position --
the quantized computation selects ``Xq`` (dequantized at ``Xs``/``Xzp``) at
that element) or ``Cond`` is false there (symmetric, with ``Y``/``Yq``/``Ys``/
``Yzp``). Either way the composed error is exactly that OPERAND's own
round-trip error, PLUS the output ``Z``'s own round-trip error, via the
triangle inequality -- and, crucially, NOT the other operand's error budget,
since ``QLinearWhere`` never reads the unselected operand's quantized value
into the selected branch's output.

This file formalizes that as a genuine Z3 case-split
(``test_qoperator_quantize_where_case_split_bound_holds`` below): one output
element is modeled with a free boolean ``cond`` (the true/false value of
``Cond`` at that position), free reals ``x``, ``y`` (the true float values of
``X``, ``Y`` there), free direct round-trip error variables ``ex := x -
dequant(Xq)`` bounded by ``Xs / 2`` and ``ey := y - dequant(Yq)`` bounded by
``Ys / 2`` (the same direct-error-variable idiom
``test_formal_verify_dynamic_quantize_matmul.py``'s own module docstring
explains -- reused here even though the nonlinear-blowup risk that idiom
guards against is much lower for a per-element, non-summed claim like this
one with no cross-tap sum at all), and a free output round-trip error ``ez``
bounded by ``Zs / 2``. The TRUE output is ``z_true := z3.If(cond, x, y)``.
``QLinearWhere``'s own int8 select-then-implicit-dequant result (before the
output's own re-quantization) is modeled as ``y_raw := z3.If(cond, x - ex, y
- ey)`` -- selecting the SAME branch's dequantized operand ``cond`` selects
for the true value -- and the pass's actual final output is ``y_raw - ez``.
Z3 proves, in one ``prove()`` call (letting Z3's own ``If``-reasoning
discharge both the ``cond`` true and ``cond`` false cases, rather than two
separate hand-written per-branch proofs)::

    |z_true - (y_raw - ez)| <= z3.If(cond, Xs / 2, Ys / 2) + Zs / 2

i.e. the error bound genuinely depends on WHICH branch was selected: only the
selected operand's own round-trip budget enters, composed via the triangle
inequality with the output's own round-trip -- never both operands' budgets,
and never the wrong one.

A negative control
(``test_qoperator_quantize_where_negative_control_wrong_branch_budget``)
confirms this dependence is genuine, not vacuous: a bound that always charges
``Xs / 2`` regardless of ``cond``'s actual value is NOT sound whenever
``cond`` is false and ``Ys`` exceeds ``Xs`` -- Z3 finds an explicit
counterexample (checked via ``solver.check() == z3.sat``, not ``prove()``,
since this claim is meant to fail).

The standard round-trip lemma
(``test_qoperator_quantize_where_round_trip_lemma_is_sound``, mirroring
``test_formal_verify_quantize_round_trip.py``'s own proof in this file's own
vocabulary, per this suite's convention of every quantization file reproving
it since this pass's error terms are built from it) confirms
``QuantizeLinear``/``DequantizeLinear``'s own defining formula matches the
``|x - xdq| <= scale / 2`` hypothesis used throughout, for a free zero point
and no clipping/saturation.

No consumer-composition step is needed here, matching this suite's usual
reasoning for a numeric *bound* claim (as opposed to an *equality* claim) --
see ``static_quantize_matmul``'s own docstring for why.

Differential tests build a ``Where(Cond, X, Y)`` with ``X``, ``Y`` both
float32 GRAPH INPUTS (not constants -- this pass's own scope requirement) via
``onnx.parser``, establish calibrated ranges for ``X``, ``Y``, AND the node's
own output (all three required by ``patternMatchPredicate``, confirmed by
reading it above), and run the real pass through the nanobind-exposed
``onnxsim.onnxsim_cpp2py_export.quantize_qoperator_where(model_bytes,
activation_ranges)`` (the ``Where``-specific analogue of
``test_formal_verify_qoperator_quantize_matmul.py``'s own
``quantize_qoperator`` entry point, confirmed by reading
``QuantizeQOperatorWhere`` in ``onnxsim/quantize_entry.cpp`` and its
"quantize_qoperator_where" nanobind binding in ``onnxsim/cpp2py_export.cc``).

``QLinearWhere`` DOES have a working ONNX Runtime CPU kernel in this
environment -- confirmed empirically (a minimal standalone quantized-Where
model, run through a graph-optimization-disabled ``InferenceSession``,
before writing this file) -- so the numeric-bound differential test runs the
real quantized graph through onnxruntime rather than a hand-simulated int8
fallback. As with every ``InferenceSession`` this suite constructs for a
quantized graph, graph optimization is disabled explicitly
(``ORT_DISABLE_ALL``): this suite previously found and fixed a real bug where
ONNX Runtime's default optimization level silently fuses certain
quantized-graph shapes into a different, hardware-specific code path than
what these proofs reason about (see
``tests/test_ort_matmul_nbits_workaround.py``'s docstring for the
precedent) -- disabled here from the start rather than risking rediscovering
the same class of bug.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
import onnxsim.onnxsim_cpp2py_export as C
from _formal_verify_common import producer, prove, z3
from onnx import parser


def _abs(v):
    return z3.If(v >= 0, v, -v)


def test_qoperator_quantize_where_round_trip_lemma_is_sound():
    # Standard QuantizeLinear/DequantizeLinear round-trip lemma, reproved in
    # this file's own vocabulary (this suite's convention -- see
    # test_formal_verify_quantize_round_trip.py for the original, reused
    # here verbatim in spirit): round(v) is modeled as *some* integer n
    # within 0.5 of v (true for round-half-to-even and any other correct
    # nearest-integer rule), and with no saturation (the quantized code
    # stays in range -- a real, separate failure mode of too-narrow
    # calibration, not something scale/2 ever bounds), zero_point cancels
    # exactly on dequantization.
    x, scale, zero_point = z3.Reals("x scale zero_point")
    n = z3.Int("n")
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        scale > 0,
        n - x / scale <= half,
        x / scale - n <= half,
    )
    quantized_code = n + zero_point
    dequantized = (quantized_code - zero_point) * scale

    error = x - dequantized
    prove(z3.Implies(hypotheses, z3.And(error <= scale / 2, -error <= scale / 2)))


def test_qoperator_quantize_where_case_split_bound_holds():
    # The pass's genuinely new claim: a per-element CASE SPLIT, not a
    # MAC-bound reduction -- there is no contraction dimension in Where at
    # all. cond is a free bool standing for Cond's value at one output
    # element; x, y are the true float X/Y values there; ex/ey are direct
    # round-trip error variables for Xq/Yq (the same idiom
    # dynamic_quantize_matmul's docstring explains, though the nonlinear-
    # blowup risk that idiom guards against barely applies to a query this
    # small -- no sum, no cross-tap products); ez is the OUTPUT's own
    # round-trip error. y_raw is QLinearWhere's own int8 select-then-
    # implicit-dequant result (before the output's own re-quantization):
    # z3.If(cond, x - ex, y - ey), i.e. it selects the SAME branch's
    # dequantized operand that the true computation selects. The pass's
    # actual final output is y_raw - ez (one more round trip, through the
    # output's own QuantizeLinear/DequantizeLinear pair inside QLinearWhere
    # + the trailing DequantizeLinear node).
    #
    # Z3's own If-reasoning discharges BOTH the cond=True and cond=False
    # cases inside this single prove() call -- this is not two separate
    # hand-written per-branch proofs glued together.
    cond = z3.Bool("cond")
    x, y = z3.Reals("x y")
    ex, ey, ez = z3.Reals("ex ey ez")
    Xs, Ys, Zs = z3.Reals("Xs Ys Zs")

    z_true = z3.If(cond, x, y)
    y_raw = z3.If(cond, x - ex, y - ey)
    z_quant = y_raw - ez

    hypotheses = z3.And(
        Xs > 0,
        Ys > 0,
        Zs > 0,
        _abs(ex) <= Xs / 2,
        _abs(ey) <= Ys / 2,
        _abs(ez) <= Zs / 2,
    )
    # The bound itself depends on cond: only the SELECTED operand's own
    # round-trip budget enters, never the other operand's.
    bound = z3.If(cond, Xs / 2, Ys / 2) + Zs / 2
    error = z_true - z_quant
    prove(z3.Implies(hypotheses, z3.And(error <= bound, -error <= bound)))


def test_qoperator_quantize_where_negative_control_wrong_branch_budget():
    # Confirms the case-split bound genuinely depends on selecting the RIGHT
    # branch's error budget: a naive bound that always charges Xs/2 (X's own
    # budget), regardless of cond's actual value, is NOT sound whenever cond
    # is False and Ys exceeds Xs -- Z3 must find a real counterexample
    # (checked via solver.check() == z3.sat, since this claim is meant to
    # fail, unlike every prove() call in this file).
    cond = z3.Bool("cond")
    x, y = z3.Reals("x y")
    ex, ey, ez = z3.Reals("ex ey ez")
    Xs, Ys, Zs = z3.Reals("Xs Ys Zs")

    z_true = z3.If(cond, x, y)
    y_raw = z3.If(cond, x - ex, y - ey)
    z_quant = y_raw - ez

    hypotheses = z3.And(
        Xs > 0,
        Ys > 0,
        Zs > 0,
        _abs(ex) <= Xs / 2,
        _abs(ey) <= Ys / 2,
        _abs(ez) <= Zs / 2,
    )
    naive_bound = Xs / 2 + Zs / 2  # WRONG: ignores cond, always charges X's budget
    error = z_true - z_quant

    solver = z3.Solver()
    solver.add(hypotheses)
    solver.add(cond == False)  # noqa: E712 -- z3 BoolRef, not a Python bool
    solver.add(Ys > Xs)  # the branch actually taken needs a bigger budget than Xs/2
    solver.add(z3.Not(z3.And(error <= naive_bound, -error <= naive_bound)))
    assert solver.check() == z3.sat, (
        "the naive 'always charge X's budget' bound holds even when Y's "
        "branch is selected and needs a bigger budget -- negative control "
        "is vacuous"
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


def _quantize_qoperator_where(model, activation_ranges):
    """Invokes the real compiled pass directly via the nanobind-exposed
    ``quantize_qoperator_where(model_bytes, activation_ranges)`` -- the
    ``Where``-specific analogue of
    ``test_formal_verify_qoperator_quantize_matmul.py``'s own
    ``quantize_qoperator``. It runs ``OptimizeFixed`` with exactly
    ``["qoperator_quantize_where"]``, so no extra
    ``skipped_optimizers``/``simplify_isolated_extra`` isolation is needed.
    """
    out = onnx.ModelProto()
    out.ParseFromString(
        C.quantize_qoperator_where(model.SerializeToString(), activation_ranges)
    )
    return out


def _expected_asymmetric_uint8_quant_params(min_val, max_val):
    """Independent re-implementation of ``ComputeAsymmetricUint8QuantParams``
    (static_quantize_matmul.h), in float32 to match the pass's own arithmetic
    precision -- identical to
    ``test_formal_verify_qoperator_quantize_matmul.py``'s own helper of the
    same name, since this pass reads the exact same function for X, Y, AND
    the output.
    """
    lo = min(np.float32(0.0), np.float32(min_val))
    hi = max(np.float32(0.0), np.float32(max_val))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zero_point = int(np.clip(np.round(-lo / scale), 0, 255))
    return np.float32(scale), zero_point


def test_qoperator_quantize_where_pass_fires_and_matches_scheme():
    # Build a plain float Where with X, Y both graph inputs (non-constant --
    # this pass's own scope requirement) and run the real pass with
    # calibration ranges for X, Y, AND the node's own output "Z", each
    # chosen to straddle 0 so every zero point comes out genuinely nonzero.
    model = _model(
        """
        g (bool[4,8] Cond, float[4,8] X, float[4,8] Y) => (float[4,8] Z)
        {
          Z = Where(Cond, X, Y)
        }
        """
    )

    x_range = (-5.0, 10.0)
    y_range = (-4.0, 8.0)
    z_range = (-5.0, 10.0)
    quantized = _quantize_qoperator_where(
        model, {"X": x_range, "Y": y_range, "Z": z_range}
    )

    op_types = {n.op_type for n in quantized.graph.node}
    assert "Where" not in op_types
    assert op_types == {"QuantizeLinear", "QLinearWhere", "DequantizeLinear"}

    # Walk the chain backward from the real graph output:
    # DequantizeLinear(Zq) <- QLinearWhere(Cond, Xq, ..., Yq, ...) <-
    # QuantizeLinear(X)/QuantizeLinear(Y).
    dq_node = producer(quantized, "Z")
    assert dq_node.op_type == "DequantizeLinear"
    qlw_node = producer(quantized, dq_node.input[0])
    assert qlw_node.op_type == "QLinearWhere"
    assert qlw_node.domain == "com.microsoft"
    assert dq_node.input[1:] == [qlw_node.input[7], qlw_node.input[8]]  # Zs, Zzp shared

    # Cond passes straight through UNQUANTIZED: still the original bool
    # graph input, untouched by any Quantize/Dequantize node.
    assert qlw_node.input[0] == "Cond"
    cond_value_info = {vi.name: vi for vi in quantized.graph.input}["Cond"]
    assert cond_value_info.type.tensor_type.elem_type == onnx.TensorProto.BOOL
    assert "Cond" not in {n.output[0] for n in quantized.graph.node}

    xq_node = producer(quantized, qlw_node.input[1])
    assert xq_node.op_type == "QuantizeLinear"
    assert xq_node.input[0] == "X"
    assert xq_node.input[1:] == qlw_node.input[2:4]  # Xs, Xzp shared

    yq_node = producer(quantized, qlw_node.input[4])
    assert yq_node.op_type == "QuantizeLinear"
    assert yq_node.input[0] == "Y"
    assert yq_node.input[1:] == qlw_node.input[5:7]  # Ys, Yzp shared

    domains = {o.domain for o in quantized.opset_import}
    assert "com.microsoft" in domains

    init = {i.name: i for i in quantized.graph.initializer}

    x_scale = numpy_helper.to_array(init[xq_node.input[1]])
    x_zp = numpy_helper.to_array(init[xq_node.input[2]])
    expected_x_scale, expected_x_zp = _expected_asymmetric_uint8_quant_params(*x_range)
    assert x_zp.dtype == np.uint8
    assert int(x_zp) == expected_x_zp
    assert expected_x_zp != 0, "test calibration range must exercise a nonzero Xzp"
    np.testing.assert_allclose(float(x_scale), float(expected_x_scale), rtol=1e-6)

    y_scale = numpy_helper.to_array(init[yq_node.input[1]])
    y_zp = numpy_helper.to_array(init[yq_node.input[2]])
    expected_y_scale, expected_y_zp = _expected_asymmetric_uint8_quant_params(*y_range)
    assert y_zp.dtype == np.uint8
    assert int(y_zp) == expected_y_zp
    assert expected_y_zp != 0, "test calibration range must exercise a nonzero Yzp"
    np.testing.assert_allclose(float(y_scale), float(expected_y_scale), rtol=1e-6)

    z_scale = numpy_helper.to_array(init[dq_node.input[1]])
    z_zp = numpy_helper.to_array(init[dq_node.input[2]])
    expected_z_scale, expected_z_zp = _expected_asymmetric_uint8_quant_params(*z_range)
    assert z_zp.dtype == np.uint8
    assert int(z_zp) == expected_z_zp
    assert expected_z_zp != 0, "test calibration range must exercise a nonzero Zzp"
    np.testing.assert_allclose(float(z_scale), float(expected_z_scale), rtol=1e-6)


def test_qoperator_quantize_where_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring qoperator_quantize_matmul's own numeric-
    # bound test: run the real quantized graph through onnxruntime
    # (QLinearWhere has a working CPU kernel in this build, confirmed
    # empirically -- see module docstring) with a Cond array that has BOTH
    # True and False entries, so both branches of the case-split are
    # genuinely exercised numerically, not just proved abstractly. Every
    # output element's error against the true float Where stays within the
    # proved per-branch bound: Xs/2 or Ys/2 (whichever branch Cond actually
    # selected at that element) plus Zs/2.
    rng = np.random.default_rng(1)
    shape = (4, 8)
    cond = rng.random(shape) > 0.5
    assert cond.any() and not cond.all(), "test Cond must exercise both branches"
    x = rng.standard_normal(shape).astype(np.float32) * 2.0
    y = rng.standard_normal(shape).astype(np.float32) * 3.0 + 1.0
    model = _model(
        """
        g (bool[4,8] Cond, float[4,8] X, float[4,8] Y) => (float[4,8] Z)
        {
          Z = Where(Cond, X, Y)
        }
        """
    )

    z_float = np.where(cond, x, y)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    z_min, z_max = float(z_float.min()), float(z_float.max())
    quantized = _quantize_qoperator_where(
        model, {"X": (x_min, x_max), "Y": (y_min, y_max), "Z": (z_min, z_max)}
    )

    # Graph optimization disabled: by default onnxruntime can fuse/transform
    # a quantized-graph shape like this one into a hardware-specific code
    # path different from the literal node chain this pass's proof reasons
    # about -- see `tests/test_ort_matmul_nbits_workaround.py`'s docstring
    # for this suite's existing precedent of a real ORT graph-optimization
    # fusion bug of exactly this shape. Disabling optimization executes the
    # graph exactly as the pass produced it.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        quantized.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    (z_quant,) = sess.run(None, {"Cond": cond, "X": x, "Y": y})

    x_scale, _x_zp = _expected_asymmetric_uint8_quant_params(x_min, x_max)
    y_scale, _y_zp = _expected_asymmetric_uint8_quant_params(y_min, y_max)
    z_scale, _z_zp = _expected_asymmetric_uint8_quant_params(z_min, z_max)

    error = np.abs(z_float - z_quant)
    per_element_operand_budget = np.where(
        cond, float(x_scale) / 2.0, float(y_scale) / 2.0
    )
    bound = per_element_operand_budget + float(z_scale) / 2.0
    assert np.all(error <= bound + 1e-6)

    # Also confirm the rewrite is meaningfully close (not merely "within a
    # loose worst-case bound") for these well-scaled inputs -- consistent
    # with UINT8 quantization on X, Y, AND the output, not bitwise equal.
    # The actual rigorous check is the proved per-branch worst-case bound
    # above, already asserted.
    np.testing.assert_allclose(z_quant, z_float, rtol=0.1, atol=2e-2)


def test_qoperator_quantize_where_declines_constant_operand():
    # patternMatchPredicate's own distinguishing decline condition: a
    # constant X or Y operand is declined outright, even with a calibrated
    # range available for every tensor name -- it should be quantized from
    # its own static values instead of force-fed through the calibration
    # harness as if it varied at inference time.
    const_y = _f32(np.random.default_rng(2).standard_normal((4, 8)), "Y")
    model = _model(
        """
        g (bool[4,8] Cond, float[4,8] X) => (float[4,8] Z)
        {
          Z = Where(Cond, X, Y)
        }
        """,
        initializer=[const_y],
    )

    quantized = _quantize_qoperator_where(
        model, {"X": (-5.0, 10.0), "Y": (-3.0, 3.0), "Z": (-5.0, 10.0)}
    )
    assert [n.op_type for n in quantized.graph.node] == ["Where"]


def test_qoperator_quantize_where_declines_with_only_two_of_three_ranges():
    # patternMatchPredicate requires calibrated ranges for X, Y, AND the
    # node's own output -- all three, confirmed by reading the predicate.
    # Supplying only two of the three must leave the Where completely
    # untouched.
    model = _model(
        """
        g (bool[4,8] Cond, float[4,8] X, float[4,8] Y) => (float[4,8] Z)
        {
          Z = Where(Cond, X, Y)
        }
        """
    )

    quantized = _quantize_qoperator_where(model, {"X": (-5.0, 10.0), "Y": (-4.0, 8.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Where"]
