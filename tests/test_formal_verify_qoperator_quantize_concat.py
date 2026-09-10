"""Formal check for QOperatorQuantizeConcat (opt-in; onnxsim's own
``onnxsim/passes/qoperator_quantize_concat.h``): the variadic-select analogue
of ``qoperator_quantize_elementwise.h``'s ``QLinearAdd``/``QLinearMul``
rewrite, using ONNX Runtime's "com.microsoft" contrib op ``QLinearConcat``
instead. It rewrites ``Z = Concat(A0, A1, ..., Am, axis=ax)`` (every ``Ai``:
RUNTIME, non-constant float32 -- a constant operand is declined, see
``patternMatchPredicate``) into::

    A0q = QuantizeLinear(A0, A0s, A0zp)                     -- CALIBRATED
    A1q = QuantizeLinear(A1, A1s, A1zp)                     -- CALIBRATED
    ...
    Amq = QuantizeLinear(Am, Ams, Amzp)                     -- CALIBRATED
    Zq  = QLinearConcat(Zs, Zzp,
                        A0q, A0s, A0zp, A1q, A1s, A1zp, ..., Amq, Ams, Amzp,
                        axis=ax)                             -- true int8 compute
    Z   = DequantizeLinear(Zq, Zs, Zzp)                     -- CALIBRATED

Soundness claim -- a GENERALIZATION of ``QOperatorQuantizeWhere``'s own
per-element case split, from 2 branches to N
=============================================================================
``test_formal_verify_qoperator_quantize_where.py`` proves that ``Where`` --
a per-element SELECTION between exactly 2 branches, chosen by a *data*
condition ``Cond`` -- has an error bound that depends only on the SELECTED
branch's own round-trip budget: ``z3.If(cond, Xs / 2, Ys / 2) + Zs / 2``.
``Concat`` is the same shape of claim generalized to N branches, but with one
structural difference that actually makes the proof *simpler*: a ``Concat``
output position's "selected branch" -- i.e. which input ``Ai`` a given output
element along the concat axis came from -- is determined purely by the
POSITION'S INDEX along that axis (a static, structural fact about how
``Concat`` lays out its inputs end-to-end), never by a runtime data value the
way ``Where``'s ``Cond`` is. So there is nothing to case-split on inside Z3 at
all: for an arbitrary but FIXED representative input ``Ai`` (a stand-in for
"whichever input this output position happens to come from" -- since nothing
below is index-specific, the identical derivation applies uniformly to EVERY
input, for every ``i`` in ``[0, m]``, by the same argument), this file states
the pass's central lemma directly, with no ``z3.If``::

    |A_i[pos] - Z[pos]| <= Ais / 2 + Zs / 2

where ``Ais`` is THAT input's own calibrated scale -- exactly mirroring
Where's "only the selected branch's budget enters, plus the output's own
round trip" conclusion, minus the case split Where needed only because its
selection was data-dependent.

This file formalizes that as a direct-error-variable Z3 lemma
(``test_qoperator_quantize_concat_representative_input_bound_holds`` below,
reusing the exact idiom
``test_formal_verify_dynamic_quantize_matmul.py``'s own module docstring
explains, and that Where's own file reuses in turn): a free real ``a_i`` (the
true float value of ``Ai`` at one output position), a free direct round-trip
error variable ``e_i := a_i - dequant(Aiq)`` bounded by ``Ais / 2``, and a
free output round-trip error ``e_out`` bounded by ``Zs / 2``. ``QLinearConcat``
copies the already-quantized ``Aiq`` code straight into the output buffer at
that position with NO recompute (concatenation moves int8 codes verbatim; it
never re-quantizes at a different scale mid-copy) and the trailing
``DequantizeLinear`` only re-quantizes/dequantizes through ``Zs``/``Zzp`` --
so the pass's actual final output at that position is modeled as
``(a_i - e_i) - e_out``. Z3 proves, via the same round-trip-lemma-then-
triangle-inequality technique Where's file uses (adapted from "2 named
branches x/y" down to "1 representative input Ai", since the argument is
identical for every input by construction)::

    |a_i - ((a_i - e_i) - e_out)| <= Ais / 2 + Zs / 2

i.e. the error bound genuinely depends on WHICH input a given output position
came from: only that input's own round-trip budget enters, composed via the
triangle inequality with the output's own round trip -- never another
input's budget.

A negative control
(``test_qoperator_quantize_concat_negative_control_wrong_input_budget``)
confirms this dependence is genuine, not vacuous -- the N-input generalization
of Where's own "wrong branch budget" control, done concretely with just 2 of
the N inputs (already sufficient to show the budgets are genuinely
input-specific, not a shared global constant): two inputs with DIFFERENT
calibrated scales ``As0 != As1``, a bound that charges input 1's budget
(``As1 / 2``) against an error that actually came from input 0's own
round trip (bounded only by ``As0 / 2``) is NOT sound whenever ``As1 < As0``
-- Z3 must find an explicit counterexample (``solver.check() == z3.sat``, not
``prove()``, since this claim is meant to fail).

The standard round-trip lemma
(``test_qoperator_quantize_concat_round_trip_lemma_is_sound``, mirroring
``test_formal_verify_quantize_round_trip.py``'s own proof and
``QOperatorQuantizeWhere``'s own reproof of it, per this suite's convention of
every quantization file reproving it since this pass's error terms are built
from it) confirms ``QuantizeLinear``/``DequantizeLinear``'s own defining
formula matches the ``|x - xdq| <= scale / 2`` hypothesis used throughout, for
a free zero point and no clipping/saturation.

No consumer-composition step is needed here, matching this suite's usual
reasoning for a numeric *bound* claim (as opposed to an *equality* claim) --
see ``static_quantize_matmul``'s own docstring for why.

A genuinely new STRUCTURAL piece Concat has that Where does not
=============================================================================
``QLinearWhere``'s schema puts the output scale/zero-point LAST (``Cond, Xq,
Xs, Xzp, Yq, Ys, Yzp, Zs, Zzp``). ``QLinearConcat``'s schema instead puts
``Y_scale``/``Y_zero_point`` as its FIRST TWO inputs, confirmed directly from
``runTransform``'s own input-building order (``qlop->addInput(z_scale_v);
qlop->addInput(z_zp_v);`` BEFORE the per-input loop that appends each
``Aiq, Ais, Aizp`` triple) -- getting this backwards would silently build a
graph ONNX Runtime either rejects outright or, worse, silently misinterprets
(e.g. treating some operand's own scale/zero-point as the output's). The
structural differential test below checks this input ordering directly
against the real compiled pass's output, not just against the doc comment.

Differential tests build a ``Concat`` with 3 non-constant float32 inputs
(each given a DIFFERENTLY calibrated range, so their scales genuinely
differ) via ``onnx.parser``, establish calibrated ranges for every input AND
the node's own output (all required by ``patternMatchPredicate``, confirmed
by reading it above), and run the real pass through the nanobind-exposed
``onnxsim.onnxsim_cpp2py_export.quantize_qoperator_concat(model_bytes,
activation_ranges)`` (the ``Concat``-specific analogue of
``test_formal_verify_qoperator_quantize_where.py``'s own
``quantize_qoperator_where``, confirmed by reading ``QuantizeQOperatorConcat``
in ``onnxsim/quantize_entry.cpp`` and its "quantize_qoperator_concat"
nanobind binding in ``onnxsim/cpp2py_export.cc``).

``QLinearConcat`` DOES have a working ONNX Runtime CPU kernel in this
environment -- confirmed empirically (a minimal standalone quantized-Concat
model, run through a graph-optimization-disabled ``InferenceSession``, before
writing this file) -- so the numeric-bound differential test runs the real
quantized graph through onnxruntime rather than a hand-simulated int8
fallback. As with every ``InferenceSession`` this suite constructs for a
quantized graph, graph optimization is disabled explicitly
(``ORT_DISABLE_ALL``): this suite previously found and fixed a real bug where
ONNX Runtime's default optimization level silently fuses certain
quantized-graph shapes into a different, hardware-specific code path than
what these proofs reason about (see
``tests/test_ort_matmul_nbits_workaround.py``'s docstring for the
precedent) -- disabled here from the start rather than risking rediscovering
the same class of bug. The numeric-bound differential test also uses
REASONABLY SIZED tensors (not tiny 2-/3-element inputs): this suite recently
found and fixed a real ONNX Runtime quantized-kernel edge case that only
manifested in CI, not locally, for very small/irregular shapes -- avoided
here from the start rather than risking rediscovering it.
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


def test_qoperator_quantize_concat_round_trip_lemma_is_sound():
    # Standard QuantizeLinear/DequantizeLinear round-trip lemma, reproved in
    # this file's own vocabulary (this suite's convention -- see
    # test_formal_verify_quantize_round_trip.py for the original, and
    # QOperatorQuantizeWhere's own file for the most recent reuse): round(v)
    # is modeled as *some* integer n within 0.5 of v (true for
    # round-half-to-even and any other correct nearest-integer rule), and
    # with no saturation (the quantized code stays in range -- a real,
    # separate failure mode of too-narrow calibration, not something
    # scale/2 ever bounds), zero_point cancels exactly on dequantization.
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


def test_qoperator_quantize_concat_representative_input_bound_holds():
    # The pass's genuinely new claim relative to Where: an N-way selection
    # among Concat's inputs, generalizing Where's 2-way per-element case
    # split -- but WITHOUT a z3.If, because Concat's own "which input did
    # this output position come from" fact is purely STRUCTURAL (determined
    # by the position's index along the concat axis), never data-dependent
    # the way Where's Cond is. So a single representative input Ai suffices:
    # a_i is the true float value of Ai at one output position; e_i is a
    # direct round-trip error variable for Aiq (the same idiom
    # dynamic_quantize_matmul's docstring explains, reused verbatim by
    # QOperatorQuantizeWhere's own file); e_out is the OUTPUT's own
    # round-trip error, through the trailing DequantizeLinear's Zs/Zzp.
    # QLinearConcat copies the already-quantized Aiq code straight into the
    # output buffer at that position (concatenation moves int8 codes
    # verbatim -- it never re-quantizes at a different scale mid-copy), so
    # the pass's actual final output there is (a_i - e_i) - e_out: one round
    # trip through Ai's own QuantizeLinear, then one more through the
    # trailing DequantizeLinear's Zs/Zzp.
    #
    # Nothing below mentions which index i is -- the SAME statement, with
    # the SAME proof, holds for every one of the m+1 inputs by construction,
    # which is exactly the sense in which this generalizes Where's 2-branch
    # case split to N branches without needing an explicit case split at
    # all.
    a_i = z3.Real("a_i")
    e_i, e_out = z3.Reals("e_i e_out")
    Ais, Zs = z3.Reals("Ais Zs")

    a_i_true = a_i
    z_quant = (a_i - e_i) - e_out

    hypotheses = z3.And(
        Ais > 0,
        Zs > 0,
        _abs(e_i) <= Ais / 2,
        _abs(e_out) <= Zs / 2,
    )
    # The bound depends only on THIS input's own scale, never another
    # input's -- there is no other input in scope in this lemma at all.
    bound = Ais / 2 + Zs / 2
    error = a_i_true - z_quant
    prove(z3.Implies(hypotheses, z3.And(error <= bound, -error <= bound)))


def test_qoperator_quantize_concat_negative_control_wrong_input_budget():
    # Confirms the representative-input bound genuinely depends on charging
    # THAT position's own source input's scale, not some OTHER input's --
    # the N-input generalization of Where's own "wrong branch budget"
    # control, done concretely with 2 of the N inputs (already sufficient:
    # if borrowing one wrong input's budget is unsound, the claim that any
    # given position's bound depends only on its OWN source input is
    # genuine, not vacuous).
    #
    # An output position sourced from input 0 has error e0 bounded only by
    # As0 / 2 (input 0's own budget). Charging it against input 1's budget
    # (As1 / 2) instead is NOT sound whenever As1 < As0 -- input 0's actual
    # round-trip error can exceed what As1/2 allows for.
    a0 = z3.Real("a0")
    e0, e_out = z3.Reals("e0 e_out")
    As0, As1, Zs = z3.Reals("As0 As1 Zs")

    z_true = a0
    z_quant = (a0 - e0) - e_out

    hypotheses = z3.And(
        As0 > 0,
        As1 > 0,
        Zs > 0,
        _abs(e0) <= As0 / 2,
        _abs(e_out) <= Zs / 2,
    )
    wrong_bound = (
        As1 / 2 + Zs / 2
    )  # WRONG: input 0's error charged against input 1's budget
    error = z_true - z_quant

    solver = z3.Solver()
    solver.add(hypotheses)
    solver.add(As1 < As0)  # the budget actually needed (As0/2) exceeds the borrowed one
    solver.add(z3.Not(z3.And(error <= wrong_bound, -error <= wrong_bound)))
    assert solver.check() == z3.sat, (
        "bounding one input's error using a DIFFERENT input's scale budget "
        "holds even when that budget is smaller than the input's own -- "
        "negative control is vacuous"
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


def _quantize_qoperator_concat(model, activation_ranges):
    """Invokes the real compiled pass directly via the nanobind-exposed
    ``quantize_qoperator_concat(model_bytes, activation_ranges)`` -- the
    ``Concat``-specific analogue of
    ``test_formal_verify_qoperator_quantize_where.py``'s own
    ``quantize_qoperator_where``. It runs ``OptimizeFixed`` with exactly
    ``["qoperator_quantize_concat"]``, so no extra
    ``skipped_optimizers``/``simplify_isolated_extra`` isolation is needed.
    """
    out = onnx.ModelProto()
    out.ParseFromString(
        C.quantize_qoperator_concat(model.SerializeToString(), activation_ranges)
    )
    return out


def _expected_asymmetric_uint8_quant_params(min_val, max_val):
    """Independent re-implementation of ``ComputeAsymmetricUint8QuantParams``
    (static_quantize_matmul.h), in float32 to match the pass's own arithmetic
    precision -- identical to
    ``test_formal_verify_qoperator_quantize_where.py``'s own helper of the
    same name, since this pass reads the exact same function for every input
    AND the output.
    """
    lo = min(np.float32(0.0), np.float32(min_val))
    hi = max(np.float32(0.0), np.float32(max_val))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zero_point = int(np.clip(np.round(-lo / scale), 0, 255))
    return np.float32(scale), zero_point


def test_qoperator_quantize_concat_pass_fires_and_matches_scheme():
    # Build a plain float Concat with 3 inputs, all graph inputs
    # (non-constant -- this pass's own scope requirement), and run the real
    # pass with calibration ranges for every input AND the node's own output
    # "Z" -- each range chosen to straddle 0 (so every zero point comes out
    # genuinely nonzero) and to genuinely DIFFER across inputs (so their
    # scales are pairwise distinct, matching this file's central claim that
    # each input's own scale -- not some shared one -- governs its own
    # segment's error).
    model = _model(
        """
        g (float[2,3] A0, float[2,4] A1, float[2,5] A2) => (float[2,12] Z)
        {
          Z = Concat<axis = 1>(A0, A1, A2)
        }
        """
    )

    a0_range = (-5.0, 10.0)
    a1_range = (-4.0, 8.0)
    a2_range = (-2.0, 20.0)
    z_range = (-5.0, 20.0)
    quantized = _quantize_qoperator_concat(
        model, {"A0": a0_range, "A1": a1_range, "A2": a2_range, "Z": z_range}
    )

    op_types = {n.op_type for n in quantized.graph.node}
    assert "Concat" not in op_types
    assert op_types == {"QuantizeLinear", "QLinearConcat", "DequantizeLinear"}

    # Walk the chain backward from the real graph output:
    # DequantizeLinear(Zq) <- QLinearConcat(Zs, Zzp, A0q, ..., A1q, ..., A2q, ...)
    # <- QuantizeLinear(A0)/QuantizeLinear(A1)/QuantizeLinear(A2).
    dq_node = producer(quantized, "Z")
    assert dq_node.op_type == "DequantizeLinear"
    qlc_node = producer(quantized, dq_node.input[0])
    assert qlc_node.op_type == "QLinearConcat"
    assert qlc_node.domain == "com.microsoft"

    # The genuinely new structural piece vs. QLinearWhere: Y_scale/
    # Y_zero_point are QLinearConcat's FIRST TWO inputs (runTransform's own
    # addInput order), not its last two -- confirmed directly against the
    # real compiled pass's output, not just the doc comment.
    assert len(qlc_node.input) == 2 + 3 * 3  # Zs, Zzp + 3 x (Aiq, Ais, Aizp)
    assert dq_node.input[1:] == list(qlc_node.input[0:2])  # Zs, Zzp shared

    axis_attr = next(a for a in qlc_node.attribute if a.name == "axis")
    assert axis_attr.i == 1

    init = {i.name: i for i in quantized.graph.initializer}
    input_names = ["A0", "A1", "A2"]
    input_ranges = [a0_range, a1_range, a2_range]
    scales = []
    for idx, (name, rng) in enumerate(zip(input_names, input_ranges)):
        base = 2 + 3 * idx
        aq_node = producer(quantized, qlc_node.input[base])
        assert aq_node.op_type == "QuantizeLinear"
        assert aq_node.input[0] == name
        # Each Aiq's own scale/zero-point, shared with the triple
        # QLinearConcat reads for that same input -- NOT the output's, and
        # NOT another input's.
        assert aq_node.input[1:] == list(qlc_node.input[base + 1 : base + 3])

        scale = numpy_helper.to_array(init[aq_node.input[1]])
        zp = numpy_helper.to_array(init[aq_node.input[2]])
        expected_scale, expected_zp = _expected_asymmetric_uint8_quant_params(*rng)
        assert zp.dtype == np.uint8
        assert int(zp) == expected_zp
        assert expected_zp != 0, "test calibration range must exercise a nonzero zp"
        np.testing.assert_allclose(float(scale), float(expected_scale), rtol=1e-6)
        scales.append(float(scale))

    # The whole point of this scheme test: every input's scale really is
    # pairwise distinct, so the differential test below actually exercises
    # "this segment's bound uses THIS input's own scale" rather than
    # incidentally passing because all scales happen to coincide.
    assert len(set(scales)) == 3

    z_scale = numpy_helper.to_array(init[dq_node.input[1]])
    z_zp = numpy_helper.to_array(init[dq_node.input[2]])
    expected_z_scale, expected_z_zp = _expected_asymmetric_uint8_quant_params(*z_range)
    assert z_zp.dtype == np.uint8
    assert int(z_zp) == expected_z_zp
    assert expected_z_zp != 0, "test calibration range must exercise a nonzero Zzp"
    np.testing.assert_allclose(float(z_scale), float(expected_z_scale), rtol=1e-6)

    domains = {o.domain for o in quantized.opset_import}
    assert "com.microsoft" in domains


def test_qoperator_quantize_concat_adds_com_microsoft_domain_once():
    # The rewrite is only reachable through "com.microsoft"; the model
    # already importing that domain (e.g. from an earlier, unrelated
    # QOperator rewrite) must not end up with it listed twice.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13, "com.microsoft": 1]
        >
        g (float[2,3] A0, float[2,4] A1) => (float[2,7] Z)
        {
          Z = Concat<axis = 1>(A0, A1)
        }
        """
    )
    quantized = _quantize_qoperator_concat(
        model, {"A0": (-5.0, 10.0), "A1": (-4.0, 8.0), "Z": (-5.0, 10.0)}
    )
    ms_domains = [o for o in quantized.opset_import if o.domain == "com.microsoft"]
    assert len(ms_domains) == 1
    assert ms_domains[0].version == 1


def test_qoperator_quantize_concat_declines_constant_operand():
    # patternMatchPredicate's own distinguishing decline condition: a
    # constant operand is declined outright, even with a calibrated range
    # available for every tensor name -- it should be quantized from its own
    # static values instead of force-fed through the calibration harness as
    # if it varied at inference time. This declines the WHOLE node, not just
    # the constant operand.
    const_a1 = _f32(np.random.default_rng(2).standard_normal((2, 4)), "A1")
    model = _model(
        """
        g (float[2,3] A0) => (float[2,7] Z)
        {
          Z = Concat<axis = 1>(A0, A1)
        }
        """,
        initializer=[const_a1],
    )

    quantized = _quantize_qoperator_concat(
        model, {"A0": (-5.0, 10.0), "A1": (-3.0, 3.0), "Z": (-5.0, 10.0)}
    )
    assert [n.op_type for n in quantized.graph.node] == ["Concat"]


def test_qoperator_quantize_concat_declines_missing_input_range():
    # patternMatchPredicate requires a calibrated range for EVERY input --
    # missing even one (here, A1's) leaves the Concat completely untouched.
    model = _model(
        """
        g (float[2,3] A0, float[2,4] A1) => (float[2,7] Z)
        {
          Z = Concat<axis = 1>(A0, A1)
        }
        """
    )

    quantized = _quantize_qoperator_concat(
        model, {"A0": (-5.0, 10.0), "Z": (-5.0, 10.0)}
    )
    assert [n.op_type for n in quantized.graph.node] == ["Concat"]


def test_qoperator_quantize_concat_declines_missing_output_range():
    # patternMatchPredicate also requires a calibrated range for the node's
    # own output -- every input's range being present is not enough.
    model = _model(
        """
        g (float[2,3] A0, float[2,4] A1) => (float[2,7] Z)
        {
          Z = Concat<axis = 1>(A0, A1)
        }
        """
    )

    quantized = _quantize_qoperator_concat(
        model, {"A0": (-5.0, 10.0), "A1": (-4.0, 8.0)}
    )
    assert [n.op_type for n in quantized.graph.node] == ["Concat"]


def test_qoperator_quantize_concat_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring qoperator_quantize_where's own numeric-
    # bound test, generalized to N inputs: run the real quantized graph
    # through onnxruntime (QLinearConcat has a working CPU kernel in this
    # build, confirmed empirically -- see module docstring) with 3
    # REASONABLY SIZED inputs (not tiny 2-/3-element tensors -- this suite
    # previously found a real ORT quantized-kernel edge case that only
    # manifested for very small/irregular shapes), each given a genuinely
    # DIFFERENT calibrated range so their scales differ.
    #
    # The single most important check here: sliced by which original input
    # produced it, EACH segment of the real quantized output stays within
    # the bound built from THAT segment's own source input's scale --
    # exactly the representative-input lemma proved above, checked
    # separately per input rather than against one pooled/worst-case scale.
    rng = np.random.default_rng(3)
    rows = 16
    widths = [20, 32, 24]  # concat-axis sizes per input -- reasonably sized
    axis = 1

    a0 = rng.standard_normal((rows, widths[0])).astype(np.float32) * 2.0
    a1 = rng.standard_normal((rows, widths[1])).astype(np.float32) * 5.0 + 3.0
    a2 = rng.standard_normal((rows, widths[2])).astype(np.float32) * 0.5 - 1.0
    inputs = [a0, a1, a2]

    model = _model(
        f"""
        g (float[{rows},{widths[0]}] A0, float[{rows},{widths[1]}] A1,
           float[{rows},{widths[2]}] A2) => (float[{rows},{sum(widths)}] Z)
        {{
          Z = Concat<axis = {axis}>(A0, A1, A2)
        }}
        """
    )

    z_float = np.concatenate(inputs, axis=axis)
    ranges = {
        f"A{i}": (float(arr.min()), float(arr.max())) for i, arr in enumerate(inputs)
    }
    ranges["Z"] = (float(z_float.min()), float(z_float.max()))
    quantized = _quantize_qoperator_concat(model, ranges)

    op_types = {n.op_type for n in quantized.graph.node}
    assert op_types == {"QuantizeLinear", "QLinearConcat", "DequantizeLinear"}

    dq_node = producer(quantized, "Z")
    qlc_node = producer(quantized, dq_node.input[0])
    assert qlc_node.op_type == "QLinearConcat"
    # Structural confirmation against the real compiled output (not just the
    # doc comment / scheme test above): QLinearConcat's first two inputs are
    # the OUTPUT's Zs/Zzp, shared verbatim with the trailing
    # DequantizeLinear -- never the first operand's own scale/zero-point.
    assert dq_node.input[1:] == list(qlc_node.input[0:2])
    first_aq = producer(quantized, qlc_node.input[2])
    assert qlc_node.input[0:2] != first_aq.input[1:3]

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
    (z_quant,) = sess.run(None, {"A0": a0, "A1": a1, "A2": a2})

    z_scale, _z_zp = _expected_asymmetric_uint8_quant_params(*ranges["Z"])
    error = np.abs(z_float - z_quant)

    # Check the bound SEPARATELY per input segment, using that segment's own
    # source input's scale -- the representative-input lemma's central
    # claim, generalized from "one arbitrary i" to "every i, checked
    # individually."
    offset = 0
    for i, width in enumerate(widths):
        seg = slice(offset, offset + width)
        a_scale, _a_zp = _expected_asymmetric_uint8_quant_params(*ranges[f"A{i}"])
        bound = float(a_scale) / 2.0 + float(z_scale) / 2.0
        seg_error = error[:, seg] if axis == 1 else error[seg, :]
        assert np.all(seg_error <= bound + 1e-6), (
            f"segment for input A{i} exceeded its own scale-derived bound"
        )
        offset += width

    # Also confirm the rewrite is meaningfully close (not merely "within a
    # loose worst-case bound") for these well-scaled inputs -- consistent
    # with UINT8 quantization on every input AND the output, not bitwise
    # equal. The actual rigorous check is the proved per-input worst-case
    # bound above, already asserted.
    np.testing.assert_allclose(z_quant, z_float, rtol=0.1, atol=2e-1)
