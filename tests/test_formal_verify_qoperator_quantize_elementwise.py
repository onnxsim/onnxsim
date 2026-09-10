"""Formal check for QOperatorQuantizeElementwise (opt-in; onnxsim's own
``onnxsim/passes/qoperator_quantize_elementwise.h``): the elementwise
analogue of ``qoperator_quantize_matmul.h``'s ``QLinearMatMul`` rewrite,
using ONNX Runtime's "com.microsoft" contrib ops ``QLinearAdd``/
``QLinearMul`` instead (standard ONNX has no quantized elementwise-binary op
at all -- see this pass's own header doc comment). It rewrites
``Z = Add(A, B)`` or ``Z = Mul(A, B)`` (``A``, ``B``: both RUNTIME,
non-constant float32 tensors -- a constant operand is declined, see
``patternMatchPredicate``) into::

    Aq = QuantizeLinear(A, As, Azp)                          -- CALIBRATED
    Bq = QuantizeLinear(B, Bs, Bzp)                          -- CALIBRATED
    Zq = QLinear{Add,Mul}(Aq, As, Azp, Bq, Bs, Bzp, Zs, Zzp) -- true int8 compute
    Z  = DequantizeLinear(Zq, Zs, Zzp)                       -- CALIBRATED

Unlike ``QLinearMatMul``/``QLinearConv``, neither operand plays a "weight"
role: BOTH ``A`` and ``B`` need a calibrated range, on top of the output
``Z``'s (QOperator format computes directly in int8, so its output must be
quantized too).

Soundness claim -- TWO genuinely different-in-kind operations, TWO different
bound shapes
=============================================================================
This single pass covers two ops whose error-propagation algebra is not the
same claim merely instantiated twice -- they differ in KIND, not just in
which operator symbol appears:

Add is exactly LINEAR (degree 1): ``Add(A, B) = A + B`` has no
multiplication anywhere, so its quantization error composes by a PLAIN
triangle inequality, no MAC-bound machinery at all. Writing the direct
round-trip error variables ``ea := A - Adq`` (bounded by ``As / 2``) and
``eb := B - Bdq`` (bounded by ``Bs / 2``), the raw (pre-output-rounding) sum
error is exactly ``(A + B) - (Adq + Bdq) = ea + eb``, bounded by
``As / 2 + Bs / 2``. This is genuinely NOT an instance of
``test_formal_verify_quantized_mac_bound.py``'s own lemma at all: that
lemma is for a PRODUCT ``X[k] * W[k]`` summed over taps, and there is no
product, no cross term, and no "K" anywhere in this derivation -- it is
strictly simpler, a purely-additive composition (``test_add_layer1_...``
below).

Mul is exactly ``quantized_mac_bound``'s own lemma at ``_K = 1``:
``Mul(A, B) = A * B`` IS a one-tap MAC (``sum_{k=1}^{1} A[k] * B[k]``), so
the exact same general two-operand bound applies unchanged, just with the
sum ranging over a single index -- there is no reduction dimension in Mul at
all (unlike MatMul/Conv/Gemm, where ``_K`` is a real contraction size this
suite fixes at 2 to exercise the cross-tap sum), so ``_K = 1`` here is not
an arbitrary small choice, it is Mul's actual, exact shape:

    |A * B - Adq * Bdq| <= (As / 2) * |B| + (Bs / 2) * |A|
                            + 1 * (As / 2) * (Bs / 2)

``test_mul_layer1_...`` below instantiates ``quantized_mac_bound``'s exact
``_bound_formulas`` shape (same direct-error-variable ``ea``/``eb`` idiom,
same sum-then-triangle-inequality structure) at ``_K = 1``, confirming by
construction that this is that file's own general lemma, not a
coincidentally-similar-looking one.

Both layers then compose with the output ``Z``'s own round-trip (``Zs / 2``)
via the triangle inequality -- the standard second layer every
``qoperator_quantize_*`` file in this suite has (see
``test_formal_verify_qoperator_quantize_matmul.py``'s own docstring for why
this composition is checked as its own explicit Z3 query rather than
assumed by hand). The two combined bounds, stated separately because they
are genuinely different claims:

    Add:  As / 2 + Bs / 2 + Zs / 2
    Mul:  (As / 2) * |B| + (Bs / 2) * |A| + (As / 2) * (Bs / 2) + Zs / 2

A genuinely interesting comparative point, directly visible in the two
bounds' own shapes (``test_bound_shapes_reflect_add_constant_mul_grows``
below makes it a concrete, deterministic check rather than just prose):
Add's combined bound is CONSTANT -- it does not mention ``A`` or ``B`` at
all, only the three scales -- while Mul's combined bound GROWS with ``|A|``
and ``|B|``. This is a real, structural difference in how worst-case
quantization error propagates through the two ops, not an artifact of how
tightly each bound happens to be stated. (The RAW per-element error observed
in the differential tests below is additionally noise-dominated by
quantization rounding, which is quasi-random per element -- so that raw
error does not by itself cleanly track this trend the way the two proved
*worst-case bounds* do; the bounds, not individual sampled errors, are what
exhibit the constant-vs-growing distinction cleanly.)

A negative-control pair per op confirms each combined bound genuinely needs
ALL of its own hypotheses: for both Add and Mul, dropping the round-trip
bound on any ONE of ``ea``, ``eb``, ``eout`` on its own breaks that op's own
combined claim (Z3 finds an explicit counterexample in each case).

No consumer-composition step is added, matching this family's usual
reasoning for a numeric *bound* (as opposed to an *equality*) claim -- see
``static_quantize_matmul``'s own docstring for why.

Differential tests build a plain float ``Add``/``Mul`` (two non-constant
float32 graph-input operands each) via ``onnx.parser`` and run the real pass
through the nanobind-exposed
``onnxsim.onnxsim_cpp2py_export.quantize_qoperator_elementwise(model_bytes,
activation_ranges)`` -- confirmed by reading ``QuantizeQOperatorElementwise``
in ``onnxsim/quantize_entry.cpp`` and its "quantize_qoperator_elementwise"
nanobind binding in ``onnxsim/cpp2py_export.cc``. Both ``QLinearAdd`` and
``QLinearMul`` have working ONNX Runtime CPU kernels in this environment --
confirmed empirically (minimal standalone quantized models, run through a
graph-optimization-disabled ``InferenceSession``, before writing this file;
also reproved as their own tests below) -- so the numeric-bound differential
tests run the real quantized graphs through onnxruntime. Tensors used for
those tests are deliberately NOT tiny (16x32, not 2x2): this suite
previously found and fixed a real ONNX Runtime quantized-kernel edge case
that only manifested in CI, not locally, for very small/irregular shapes
(see ``test_formal_verify_qoperator_quantize_matmul.py``'s own docstring and
``tests/test_ort_matmul_nbits_workaround.py`` for the precedent) -- a
reasonably sized shape sidesteps that class of risk from the start. As with
every ``InferenceSession`` this suite constructs for a quantized graph,
graph optimization is disabled explicitly (``ORT_DISABLE_ALL``).
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


# ---------------------------------------------------------------------------
# 1. Z3 proofs
# ---------------------------------------------------------------------------


def test_add_layer1_error_is_bounded_by_plain_triangle_inequality():
    # Add(A, B) = A + B is exactly linear -- NOT an instance of
    # quantized_mac_bound's own lemma (that lemma is for a PRODUCT, and Add
    # has no multiplication anywhere). ea/eb are the direct round-trip error
    # variables (A - Adq, B - Bdq); the raw (pre-output-rounding) sum error
    # is exactly ea + eb, bounded by plain triangle inequality alone -- no
    # cross term, no K, no nonlinear-blowup risk to mitigate at all.
    a, b = z3.Reals("a b")
    ea, eb = z3.Reals("ea eb")
    As, Bs = z3.Reals("As Bs")

    hypotheses = z3.And(As > 0, Bs > 0, _abs(ea) <= As / 2, _abs(eb) <= Bs / 2)

    adq = a - ea
    bdq = b - eb
    float_add = a + b
    quantized_add = adq + bdq  # == a + b - ea - eb == float_add - (ea + eb)

    error = float_add - quantized_add
    bound = As / 2 + Bs / 2
    prove(z3.Implies(hypotheses, z3.And(error <= bound, -error <= bound)))


def test_mul_layer1_error_is_bounded_by_mac_bound_at_k1():
    # Mul(A, B) = A * B is LITERALLY a one-tap MAC
    # (sum_{k=1}^{1} A[k] * B[k]) -- there is no "K" here at all, since Mul
    # only ever multiplies ONE pair of scalars per output element, not a
    # reduction over any dimension. This instantiates
    # test_formal_verify_quantized_mac_bound.py's own _bound_formulas shape
    # verbatim (same ea/eb direct-error-variable idiom, same
    # sum-then-triangle-inequality structure) at _K = 1, confirming by
    # construction that this really is that file's own general lemma, not a
    # coincidentally-similar-looking one -- just with the sum ranging over a
    # single index.
    _K = 1
    a = [z3.Real(f"a{k}") for k in range(_K)]
    b = [z3.Real(f"b{k}") for k in range(_K)]
    ea = [z3.Real(f"ea{k}") for k in range(_K)]  # a[k] - adq[k]
    eb = [z3.Real(f"eb{k}") for k in range(_K)]  # b[k] - bdq[k]
    As = z3.Real("As")  # calibrated A scale
    Bs = z3.Real("Bs")  # calibrated B scale

    adq = [a[k] - ea[k] for k in range(_K)]
    bdq = [b[k] - eb[k] for k in range(_K)]

    hypotheses = z3.And(
        As > 0,
        Bs > 0,
        *[_abs(ea[k]) <= As / 2 for k in range(_K)],
        *[_abs(eb[k]) <= Bs / 2 for k in range(_K)],
    )

    float_mul = sum(a[k] * b[k] for k in range(_K))
    quantized_mul = sum(adq[k] * bdq[k] for k in range(_K))

    bound = (
        (As / 2) * sum(_abs(b[k]) for k in range(_K))
        + (Bs / 2) * sum(_abs(a[k]) for k in range(_K))
        + _K * (As / 2) * (Bs / 2)
    )

    error = float_mul - quantized_mul
    prove(z3.Implies(hypotheses, z3.And(error <= bound, -error <= bound)))


def test_add_combined_bound_holds():
    # Layer 2: Z (the pass's actual final output) is the raw sum's own
    # further round trip through the output's QuantizeLinear/
    # DequantizeLinear pair inside QLinearAdd + the trailing
    # DequantizeLinear node, modeled the same direct-error-variable way --
    # eout := (Adq + Bdq) - Z, bounded by Zs / 2. Given both layer 1's
    # hypotheses (bounding |(a + b) - (adq + bdq)|) and this output
    # round-trip hypothesis, Z3 confirms the triangle-inequality
    # composition: |(a + b) - Z| <= (As/2 + Bs/2) + Zs/2.
    a, b = z3.Reals("a b")
    ea, eb, eout = z3.Reals("ea eb eout")
    As, Bs, Zs = z3.Reals("As Bs Zs")

    hypotheses = z3.And(
        As > 0,
        Bs > 0,
        Zs > 0,
        _abs(ea) <= As / 2,
        _abs(eb) <= Bs / 2,
        _abs(eout) <= Zs / 2,
    )

    z_raw = (a - ea) + (b - eb)
    z_final = z_raw - eout

    combined_bound = As / 2 + Bs / 2 + Zs / 2
    error = (a + b) - z_final
    prove(
        z3.Implies(
            hypotheses, z3.And(error <= combined_bound, -error <= combined_bound)
        )
    )


def test_mul_combined_bound_holds():
    # Layer 2 for Mul, exactly mirroring the Add case above but built on the
    # _K=1 MAC-bound layer 1 instead of plain addition.
    a, b = z3.Reals("a b")
    ea, eb, eout = z3.Reals("ea eb eout")
    As, Bs, Zs = z3.Reals("As Bs Zs")

    hypotheses = z3.And(
        As > 0,
        Bs > 0,
        Zs > 0,
        _abs(ea) <= As / 2,
        _abs(eb) <= Bs / 2,
        _abs(eout) <= Zs / 2,
    )

    z_raw = (a - ea) * (b - eb)
    z_final = z_raw - eout

    bound1 = (As / 2) * _abs(b) + (Bs / 2) * _abs(a) + (As / 2) * (Bs / 2)
    combined_bound = bound1 + Zs / 2
    error = (a * b) - z_final
    prove(
        z3.Implies(
            hypotheses, z3.And(error <= combined_bound, -error <= combined_bound)
        )
    )


def test_add_combined_bound_requires_each_hypothesis():
    # Negative control: dropping the round-trip bound on ANY ONE of ea, eb,
    # eout on its own breaks Add's own combined claim -- checked for each of
    # the three individually (not just "layer 1 as a whole" vs "output"),
    # since Add's layer 1 already has no cross term binding ea and eb
    # together: each is its own independent hypothesis.
    a, b = z3.Reals("a b")
    ea, eb, eout = z3.Reals("ea eb eout")
    As, Bs, Zs = z3.Reals("As Bs Zs")

    z_raw = (a - ea) + (b - eb)
    z_final = z_raw - eout
    combined_bound = As / 2 + Bs / 2 + Zs / 2
    error = (a + b) - z_final

    all_bounds = {
        "ea": _abs(ea) <= As / 2,
        "eb": _abs(eb) <= Bs / 2,
        "eout": _abs(eout) <= Zs / 2,
    }
    for dropped in all_bounds:
        solver = z3.Solver()
        solver.add(As > 0, Bs > 0, Zs > 0)
        for name, clause in all_bounds.items():
            if name != dropped:
                solver.add(clause)
        solver.add(z3.Not(z3.And(error <= combined_bound, -error <= combined_bound)))
        assert solver.check() == z3.sat, (
            f"Add's combined bound holds even without {dropped}'s round-trip "
            "bound -- negative control is vacuous"
        )


def test_mul_combined_bound_requires_each_hypothesis():
    # Symmetric negative control for Mul: dropping the round-trip bound on
    # any one of ea, eb, eout alone breaks Mul's own combined claim too.
    a, b = z3.Reals("a b")
    ea, eb, eout = z3.Reals("ea eb eout")
    As, Bs, Zs = z3.Reals("As Bs Zs")

    z_raw = (a - ea) * (b - eb)
    z_final = z_raw - eout
    bound1 = (As / 2) * _abs(b) + (Bs / 2) * _abs(a) + (As / 2) * (Bs / 2)
    combined_bound = bound1 + Zs / 2
    error = (a * b) - z_final

    all_bounds = {
        "ea": _abs(ea) <= As / 2,
        "eb": _abs(eb) <= Bs / 2,
        "eout": _abs(eout) <= Zs / 2,
    }
    for dropped in all_bounds:
        solver = z3.Solver()
        solver.add(As > 0, Bs > 0, Zs > 0)
        for name, clause in all_bounds.items():
            if name != dropped:
                solver.add(clause)
        solver.add(z3.Not(z3.And(error <= combined_bound, -error <= combined_bound)))
        assert solver.check() == z3.sat, (
            f"Mul's combined bound holds even without {dropped}'s round-trip "
            "bound -- negative control is vacuous"
        )


def test_bound_shapes_reflect_add_constant_mul_grows_with_magnitude():
    # The comparative point from the module docstring, made concrete: Add's
    # combined bound is a CONSTANT function of (As, Bs, Zs) alone -- it does
    # not reference A or B's actual values anywhere in its formula -- while
    # Mul's combined bound is a function that strictly grows with |A| and
    # |B|. This is a deterministic property of the two proved bound
    # FORMULAS themselves (not a sampled/noisy empirical observation), so it
    # is checked directly in plain Python rather than as a Z3 query.
    As, Bs, Zs = 0.1, 0.2, 0.3

    def add_bound(_a, _b):
        return As / 2 + Bs / 2 + Zs / 2

    def mul_bound(a, b):
        return (As / 2) * abs(b) + (Bs / 2) * abs(a) + (As / 2) * (Bs / 2) + Zs / 2

    # Add's bound is literally identical however large |a|/|b| get.
    assert add_bound(1.0, 1.0) == add_bound(1000.0, -1000.0) == add_bound(0.0, 0.0)

    # Mul's bound strictly increases as |a|/|b| grow.
    small = mul_bound(1.0, 1.0)
    large = mul_bound(1000.0, 1000.0)
    assert large > small
    assert large > 100 * small  # not just "larger", but unboundedly so


# ---------------------------------------------------------------------------
# 2. Differential / structural content
# ---------------------------------------------------------------------------


def _model(body, initializer=(), opset=13, ir_version=10, extra_imports=""):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}{extra_imports}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return numpy_helper.from_array(array.astype(np.float32), name)


def _quantize_qoperator_elementwise(model, activation_ranges):
    """Invokes the real compiled pass directly via the nanobind-exposed
    ``quantize_qoperator_elementwise(model_bytes, activation_ranges)``. Runs
    ``OptimizeFixed`` with exactly ``["qoperator_quantize_elementwise"]``, so
    no extra ``skipped_optimizers``/``simplify_isolated_extra`` isolation is
    needed (module docstring; ``QuantizeQOperatorElementwise`` in
    ``onnxsim/quantize_entry.cpp``).
    """
    out = onnx.ModelProto()
    out.ParseFromString(
        C.quantize_qoperator_elementwise(model.SerializeToString(), activation_ranges)
    )
    return out


def _expected_asymmetric_uint8_quant_params(min_val, max_val):
    """Independent re-implementation of ``ComputeAsymmetricUint8QuantParams``
    (static_quantize_matmul.h), in float32 to match the pass's own arithmetic
    precision -- identical to this suite's other ``qoperator_quantize_*``
    files' own helper of the same name, since this pass reads the exact same
    function for A, B, AND the output.
    """
    lo = min(np.float32(0.0), np.float32(min_val))
    hi = max(np.float32(0.0), np.float32(max_val))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zero_point = int(np.clip(np.round(-lo / scale), 0, 255))
    return np.float32(scale), zero_point


def _disable_opt_session(model_bytes):
    # Disable graph optimization explicitly: by default onnxruntime can
    # fuse/transform a QDQ/QOperator-shaped chain like this pass produces
    # into a different, hardware-specific code path than the literal node
    # chain this file's proofs reason about -- see
    # tests/test_ort_matmul_nbits_workaround.py's docstring for this suite's
    # existing precedent of a real ORT graph-optimization fusion bug of
    # exactly this shape.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model_bytes, sess_options=so, providers=["CPUExecutionProvider"]
    )


def test_qoperator_quantize_elementwise_qlinear_add_kernel_exists():
    # Confirmed empirically before relying on it anywhere else in this file
    # (this suite's own precedent for a contrib op used by a differential
    # test, e.g. test_formal_verify_qoperator_quantize_softmax.py's own
    # kernel-exists test): a minimal standalone
    # QuantizeLinear/QuantizeLinear -> QLinearAdd -> DequantizeLinear model
    # actually runs on onnxruntime's CPU provider.
    model = _model(
        """
        g (uint8[4] Aq, uint8[4] Bq) => (uint8[4] Zq)
        <float As = {0.1}, uint8 Azp = {10}, float Bs = {0.2}, uint8 Bzp = {20},
         float Zs = {0.3}, uint8 Zzp = {30}>
        {
          Zq = com.microsoft.QLinearAdd(Aq, As, Azp, Bq, Bs, Bzp, Zs, Zzp)
        }
        """,
        extra_imports=', "com.microsoft": 1',
    )
    sess = _disable_opt_session(model.SerializeToString())
    aq = np.array([10, 20, 30, 40], dtype=np.uint8)
    bq = np.array([20, 21, 22, 23], dtype=np.uint8)
    (zq,) = sess.run(None, {"Aq": aq, "Bq": bq})
    assert zq.dtype == np.uint8
    assert zq.shape == (4,)


def test_qoperator_quantize_elementwise_qlinear_mul_kernel_exists():
    # Symmetric kernel-exists check for QLinearMul.
    model = _model(
        """
        g (uint8[4] Aq, uint8[4] Bq) => (uint8[4] Zq)
        <float As = {0.1}, uint8 Azp = {10}, float Bs = {0.2}, uint8 Bzp = {20},
         float Zs = {0.3}, uint8 Zzp = {30}>
        {
          Zq = com.microsoft.QLinearMul(Aq, As, Azp, Bq, Bs, Bzp, Zs, Zzp)
        }
        """,
        extra_imports=', "com.microsoft": 1',
    )
    sess = _disable_opt_session(model.SerializeToString())
    aq = np.array([10, 20, 30, 40], dtype=np.uint8)
    bq = np.array([20, 21, 22, 23], dtype=np.uint8)
    (zq,) = sess.run(None, {"Aq": aq, "Bq": bq})
    assert zq.dtype == np.uint8
    assert zq.shape == (4,)


def _assert_pass_fires_and_matches_scheme(op, qlinear_op_type):
    # Shared structural check for both Add and Mul: both operands and the
    # output straddle 0 so every zero point comes out genuinely nonzero,
    # mirroring this suite's usual convention.
    rows, cols = 4, 8
    model = _model(
        f"""
        g (float[{rows},{cols}] A, float[{rows},{cols}] B) => (float[{rows},{cols}] Z)
        {{
          Z = {op}(A, B)
        }}
        """
    )

    a_range = (-5.0, 10.0)
    b_range = (-4.0, 8.0)
    z_range = (-5.0, 10.0)
    quantized = _quantize_qoperator_elementwise(
        model, {"A": a_range, "B": b_range, "Z": z_range}
    )

    # No float Add/Mul left anywhere.
    op_types = {n.op_type for n in quantized.graph.node}
    assert op not in op_types
    assert op_types == {"QuantizeLinear", qlinear_op_type, "DequantizeLinear"}

    # Walk the chain backward from the real graph output:
    # DequantizeLinear(Zq) <- QLinear{Add,Mul}(Aq, ..., Bq, ...) <-
    # QuantizeLinear(A)/QuantizeLinear(B).
    dq_node = producer(quantized, "Z")
    assert dq_node.op_type == "DequantizeLinear"
    qlop_node = producer(quantized, dq_node.input[0])
    assert qlop_node.op_type == qlinear_op_type
    assert qlop_node.domain == "com.microsoft"
    assert dq_node.input[1:] == list(qlop_node.input[6:8])  # Zs, Zzp shared

    aq_node = producer(quantized, qlop_node.input[0])
    assert aq_node.op_type == "QuantizeLinear"
    assert aq_node.input[0] == "A"
    assert aq_node.input[1:] == list(qlop_node.input[1:3])  # As, Azp shared

    bq_node = producer(quantized, qlop_node.input[3])
    assert bq_node.op_type == "QuantizeLinear"
    assert bq_node.input[0] == "B"
    assert bq_node.input[1:] == list(qlop_node.input[4:6])  # Bs, Bzp shared

    # com.microsoft opset import added, version 1.
    ms_imports = [o for o in quantized.opset_import if o.domain == "com.microsoft"]
    assert len(ms_imports) == 1
    assert ms_imports[0].version == 1

    init = {i.name: i for i in quantized.graph.initializer}

    a_scale = numpy_helper.to_array(init[aq_node.input[1]])
    a_zp = numpy_helper.to_array(init[aq_node.input[2]])
    expected_a_scale, expected_a_zp = _expected_asymmetric_uint8_quant_params(*a_range)
    assert a_zp.dtype == np.uint8
    assert int(a_zp) == expected_a_zp
    assert expected_a_zp != 0, "test calibration range must exercise a nonzero Azp"
    np.testing.assert_allclose(float(a_scale), float(expected_a_scale), rtol=1e-6)

    b_scale = numpy_helper.to_array(init[bq_node.input[1]])
    b_zp = numpy_helper.to_array(init[bq_node.input[2]])
    expected_b_scale, expected_b_zp = _expected_asymmetric_uint8_quant_params(*b_range)
    assert b_zp.dtype == np.uint8
    assert int(b_zp) == expected_b_zp
    assert expected_b_zp != 0, "test calibration range must exercise a nonzero Bzp"
    np.testing.assert_allclose(float(b_scale), float(expected_b_scale), rtol=1e-6)

    z_scale = numpy_helper.to_array(init[dq_node.input[1]])
    z_zp = numpy_helper.to_array(init[dq_node.input[2]])
    expected_z_scale, expected_z_zp = _expected_asymmetric_uint8_quant_params(*z_range)
    assert z_zp.dtype == np.uint8
    assert int(z_zp) == expected_z_zp
    assert expected_z_zp != 0, "test calibration range must exercise a nonzero Zzp"
    np.testing.assert_allclose(float(z_scale), float(expected_z_scale), rtol=1e-6)

    return quantized


def test_qoperator_quantize_elementwise_add_pass_fires_and_matches_scheme():
    _assert_pass_fires_and_matches_scheme("Add", "QLinearAdd")


def test_qoperator_quantize_elementwise_mul_pass_fires_and_matches_scheme():
    _assert_pass_fires_and_matches_scheme("Mul", "QLinearMul")


def _numeric_bound_differential_check(op, floatfn, combined_bound_fn, seed):
    # Reasonably sized tensors (16x32 = 512 elements, not tiny 2x2) -- see
    # module docstring for why: this suite previously found and fixed a real
    # ONNX Runtime quantized-kernel edge case specific to very small/
    # irregular shapes.
    rng = np.random.default_rng(seed)
    rows, cols = 16, 32
    a = rng.standard_normal((rows, cols)).astype(np.float32) * 3.0 + 1.0
    b = rng.standard_normal((rows, cols)).astype(np.float32) * 2.0 - 0.5

    model = _model(
        f"""
        g (float[{rows},{cols}] A, float[{rows},{cols}] B) => (float[{rows},{cols}] Z)
        {{
          Z = {op}(A, B)
        }}
        """
    )

    z_float = floatfn(a, b)
    # Calibration ranges set to the actual observed (min, max) of A, B, and
    # the true float output so nothing clips -- the round-trip lemmas'
    # explicit side condition, for all three tensors.
    a_min, a_max = float(a.min()), float(a.max())
    b_min, b_max = float(b.min()), float(b.max())
    z_min, z_max = float(z_float.min()), float(z_float.max())
    quantized = _quantize_qoperator_elementwise(
        model, {"A": (a_min, a_max), "B": (b_min, b_max), "Z": (z_min, z_max)}
    )

    sess = _disable_opt_session(quantized.SerializeToString())
    (z_quant,) = sess.run(None, {"A": a, "B": b})

    a_scale, _a_zp = _expected_asymmetric_uint8_quant_params(a_min, a_max)
    b_scale, _b_zp = _expected_asymmetric_uint8_quant_params(b_min, b_max)
    z_scale, _z_zp = _expected_asymmetric_uint8_quant_params(z_min, z_max)
    eps_a, eps_b, eps_z = (
        float(a_scale) / 2.0,
        float(b_scale) / 2.0,
        float(z_scale) / 2.0,
    )

    error = np.abs(z_float - z_quant)
    bound = combined_bound_fn(a, b, eps_a, eps_b, eps_z)
    assert np.all(error <= bound + 1e-6)
    return error, bound


def test_qoperator_quantize_elementwise_add_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring this suite's other qoperator_quantize_*
    # files: run the real quantized graph through onnxruntime (QLinearAdd
    # has a working CPU kernel, confirmed above) and confirm every output
    # element's error against the true float Add stays within the COMBINED
    # bound proved above -- which, per this file's own claim, is CONSTANT
    # across every element (does not depend on A/B's values at all).
    error, bound = _numeric_bound_differential_check(
        "Add",
        lambda a, b: a + b,
        lambda a, b, eps_a, eps_b, eps_z: eps_a + eps_b + eps_z,
        seed=0,
    )
    # The bound is a single scalar broadcast identically to every element --
    # confirming the "constant" half of the comparative claim structurally,
    # not just in the abstract Z3 formula.
    assert np.asarray(bound).ndim == 0 or np.unique(np.asarray(bound)).size == 1

    # Also confirm the rewrite is meaningfully close (not merely "within a
    # loose worst-case bound") for these well-scaled inputs.
    np.testing.assert_allclose(error, 0.0, atol=0.5)


def test_qoperator_quantize_elementwise_mul_output_is_close_to_float_within_proved_bound():
    # Symmetric differential check for Mul: the combined bound now varies
    # per element (it depends on |A[i]|/|B[i]| at that element), per this
    # file's own claim -- confirmed structurally below, not just proved
    # abstractly.
    error, bound = _numeric_bound_differential_check(
        "Mul",
        lambda a, b: a * b,
        lambda a, b, eps_a, eps_b, eps_z: (
            eps_a * np.abs(b) + eps_b * np.abs(a) + eps_a * eps_b + eps_z
        ),
        seed=1,
    )
    # Unlike Add's bound, Mul's genuinely varies across elements (it is not
    # a single repeated scalar) -- the "grows with magnitude" half of the
    # comparative claim, made concrete on the actual per-element bound array
    # this differential test computed (not just the abstract formula).
    assert np.unique(bound).size > 1

    np.testing.assert_allclose(error, 0.0, atol=0.5)


def test_qoperator_quantize_elementwise_declines_with_constant_operand():
    # patternMatchPredicate's own distinguishing scope restriction: a
    # constant operand (e.g. a per-channel bias added elementwise) is left
    # alone entirely, for both Add and Mul, even with all three calibrated
    # ranges present.
    rng = np.random.default_rng(2)
    w = rng.standard_normal((4, 8)).astype(np.float32)
    for op in ("Add", "Mul"):
        model = _model(
            f"""
            g (float[4,8] A) => (float[4,8] Z)
            {{
              Z = {op}(A, W)
            }}
            """,
            [_f32(w, "W")],
        )
        quantized = _quantize_qoperator_elementwise(
            model, {"A": (-5.0, 10.0), "W": (-3.0, 3.0), "Z": (-8.0, 13.0)}
        )
        assert [n.op_type for n in quantized.graph.node] == [op]


def test_qoperator_quantize_elementwise_sub_div_never_matched():
    # This pass's patternMatchPredicate only matches kAdd/kMul -- Sub and
    # Div (both binary, both float32, both plausibly non-constant) must
    # never be rewritten, even with every calibrated range present.
    for op in ("Sub", "Div"):
        model = _model(
            f"""
            g (float[4,8] A, float[4,8] B) => (float[4,8] Z)
            {{
              Z = {op}(A, B)
            }}
            """
        )
        quantized = _quantize_qoperator_elementwise(
            model, {"A": (-5.0, 10.0), "B": (-4.0, 8.0), "Z": (-13.0, 18.0)}
        )
        assert [n.op_type for n in quantized.graph.node] == [op]


def test_qoperator_quantize_elementwise_com_microsoft_import_not_duplicated():
    # If "com.microsoft" is already present in opset_import (e.g. a model
    # that already went through this pass, or already used another contrib
    # op), the pass must not add a second entry -- and the rewrite must
    # still actually fire.
    model = _model(
        """
        g (float[4,8] A, float[4,8] B) => (float[4,8] Z)
        {
          Z = Add(A, B)
        }
        """,
        extra_imports=', "com.microsoft": 1',
    )
    quantized = _quantize_qoperator_elementwise(
        model, {"A": (-5.0, 10.0), "B": (-4.0, 8.0), "Z": (-9.0, 18.0)}
    )
    ms_imports = [o for o in quantized.opset_import if o.domain == "com.microsoft"]
    assert len(ms_imports) == 1
    assert ms_imports[0].version == 1
    assert producer(quantized, "Z").op_type == "DequantizeLinear"


def test_qoperator_quantize_elementwise_declines_with_any_missing_range():
    # patternMatchPredicate requires calibrated ranges for ALL THREE of A,
    # B, and the node's own output -- missing any single one of the three
    # must leave the Add/Mul node completely untouched.
    model = _model(
        """
        g (float[4,8] A, float[4,8] B) => (float[4,8] Z)
        {
          Z = Add(A, B)
        }
        """
    )
    full_ranges = {"A": (-5.0, 10.0), "B": (-4.0, 8.0), "Z": (-9.0, 18.0)}
    for missing in ("A", "B", "Z"):
        ranges = {k: v for k, v in full_ranges.items() if k != missing}
        quantized = _quantize_qoperator_elementwise(model, ranges)
        assert [n.op_type for n in quantized.graph.node] == ["Add"], (
            f"rewrite fired despite missing {missing}'s calibrated range"
        )
