"""Formal check for QuantizeFp16 (opt-in; ``onnxsim/passes/quantize_fp16.h``):
converts every float32 initializer/``Constant``-node value in the top-level
graph to float16, round-to-nearest (ties away from zero), clamping any
magnitude beyond float16's largest finite value (65504) to +-65504 rather
than producing an infinity. With ``keep_io_types`` (the pass's own default,
read from the function-local static ``QuantizeFp16KeepIoTypes()``), the
graph's own external input/output types stay float32 via a boundary
``Cast``; with it false, inputs/outputs are redeclared float16 directly.
Full mechanics in the header comment -- read in full before this file.

Unlike a ``FullGraphBasedPass`` reached the usual way in this suite
(``simplify_isolated_extra`` + ``skipped_optimizers``/``extra_optimizers``),
``keep_io_types`` is a parameter ``OptimizeFixed``'s pass-name-list interface
has no way to carry -- it is a C++-side function-local static that onnxsim's
own higher-level Python entry point sets immediately before invoking the
pass (the same pattern ``dynamic_quantize_matmul``'s and
``static_quantize_matmul``'s own passes use for parameters
``simplify_isolated_extra`` can't plumb through). Grepping ``onnxsim/*.py``
turns up exactly that entry point: ``onnxsim.quantize_fp16(model,
keep_io_types=True)`` (``onnxsim/onnx_simplifier.py``), analogous to
``onnxsim.quantize_dynamic`` for ``dynamic_quantize_matmul``
(``test_formal_verify_dynamic_quantize_matmul.py``'s own template for this
situation) -- so every differential test below calls it directly with an
explicit ``keep_io_types`` value, rather than going through
``simplify_isolated_extra`` at all.

Formal content: this is fundamentally a *lossy, rounding* rewrite -- IEEE-754
float16 rounding, not integer quantization with a scale/zero-point, so
``quantized_mac_bound``'s dot-product-accumulation shape doesn't apply as-is
(there is no scale to factor out of a sum; the "grid" itself is float16's own
non-uniform, magnitude-dependent spacing). The genuinely new mathematical
content here is a round-to-nearest-with-clamping bound:

1. ``test_quantize_fp16_round_to_nearest_is_half_step_bound``: the general,
   scheme-agnostic shape of "round to nearest" -- for any real ``v`` and any
   grid spacing ``step > 0``, a value ``n`` steps from the origin that is
   *nearest* to ``v / step`` (within 0.5, exactly ``quantize_round_trip``'s
   own modeling of ``round()``, minus that proof's ``zero_point`` term, which
   float16 has no analogue of) reconstructs ``v`` to within ``step / 2``.
   This is the same general lemma ``test_formal_verify_quantize_round_trip.py``
   proves for uniform-affine quantization's ``round(x / scale)``, restated
   without the affine offset -- float16's rounding is symmetric about zero
   (ties away from zero, not toward some non-zero zero-point), so the same
   half-step argument applies with one fewer term.
2. ``test_quantize_fp16_worst_case_relative_bound_near_precision_limit``: the
   float16-specific corollary -- for a normalized value, the local step
   (ULP) is at most ``v * 2**-10`` (10 stored mantissa bits: within
   ``[2**e, 2**(e+1))`` the ULP is exactly ``2**(e-10)``, and ``v >= 2**e``,
   so ``ulp(v) / v <= 2**-10``), so the reconstruction error is at most
   ``v * 2**-11`` -- literally the general lemma above with ``step``
   instantiated to ``v * 2**-10``, kept as a separate ``prove()`` call (not
   derived from the first as a Python-level substitution) since it is itself
   a fully general, one-line Z3 query and there is no tractability reason
   here (unlike ``dynamic_quantize_matmul``'s deliberately-split lemmas) to
   avoid handing Z3 both steps at once.
3. ``test_quantize_fp16_clamp_saturates_to_max_magnitude``: the clamping
   design choice this pass's header comment calls out explicitly ("this pass
   never silently introduces a new infinity") -- for ``|v| > 65504``, the
   clamped value is *exactly* +-65504, not proportionally further off and
   not infinite. Definitional given the ``If``-expression clamp itself, but
   worth stating as its own claim since it is real, deliberate behavior a
   reader could otherwise assume works like ordinary saturating rounding
   (clamp *after* rounding, potentially landing one ULP beyond 65504) rather
   than clamp-then-round (this pass's actual order, per
   ``FloatToFloat16Bits``).
4. ``test_quantize_fp16_negative_control_needs_nearest_hypothesis``: without
   constraining ``n`` to be *near* ``v / step`` at all (only ``step > 0``),
   the half-step bound is not a theorem -- Z3 finds a real counterexample --
   confirming the bound genuinely depends on the rounding hypothesis rather
   than holding vacuously for any two reals.

No substitution/composition step (this suite's usual next move once a
per-element bound is established, e.g. ``quantized_mac_bound``'s
dot-product lemma) is attempted here, for the same reason
``dynamic_quantize_matmul``'s own docstring gives: an arbitrary downstream
consumer need not be Lipschitz, so a numeric *bound* on one value doesn't by
itself bound anything about a consumer applied to it (unlike an *equality*,
which trivially composes under any function). Every float32 value in the
graph -- weights via direct conversion, activations implicitly via ordinary
dtype propagation once every op's inputs are float16 -- gets this same
per-element bound uniformly, but turning that into a whole-graph bound would
need an op-by-op Lipschitz/error-propagation argument this pass's own
transform doesn't need and this file doesn't attempt; the differential tests
below instead check the real end-to-end numeric behavior on concrete models.

Differential tests build models via ``onnx.parser.parse_model()`` (per
``CLAUDE.md``) and call the real ``onnxsim.quantize_fp16`` entry point found
above, confirming: (1) with the pass's own default ``keep_io_types=True``, a
non-tie float32 weight becomes a bit-exact-correct float16 value (checked
against ``numpy``'s own ``.astype(np.float16)`` as an independent reference
-- valid for a non-tie value even though numpy's ties-to-even convention
differs from this pass's documented ties-away-from-zero, per ``CLAUDE.md``'s
guidance on this exact wrinkle) and boundary ``Cast`` nodes appear exactly
where the header comment says; (2) the overall numeric output, run through
onnxruntime, stays close to (not equal to) the unquantized model's output;
(3) a value hand-constructed to be an *exact* float16 tie confirms the
documented ties-away-from-zero rule specifically, independent of numpy;
(4) a value with magnitude > 65504 is clamped to exactly +-65504 (read back
from the raw float16 bytes), never inf/nan; (5) ``keep_io_types=False``
redeclares the graph's I/O float16 directly, with no boundary Cast at all.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import pytest
from _formal_verify_common import prove, z3
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")

_FLOAT16_MAX = 65504.0


def test_quantize_fp16_round_to_nearest_is_half_step_bound():
    # n stands for "the number of grid steps the rounded value sits at" --
    # constrained only by *being nearest* to v / step (within 0.5, the same
    # abstract modeling of round() quantize_round_trip's own proof uses,
    # true regardless of tie-breaking rule since a tie only ever occurs
    # exactly halfway, where both neighbors already satisfy the bound). No
    # zero_point term: float16's grid is symmetric about zero, unlike
    # uniform-affine quantization's offset code.
    v, step = z3.Reals("v step")
    n = z3.Int("n")
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(step > 0, n - v / step <= half, v / step - n <= half)
    rounded = n * step
    error = v - rounded
    prove(z3.Implies(hypotheses, z3.And(error <= step / 2, -error <= step / 2)))


def test_quantize_fp16_worst_case_relative_bound_near_precision_limit():
    # float16's own corollary of the general bound above: for a normalized
    # value v > 0, the local grid spacing (ULP) is at most v * 2**-10 (10
    # stored mantissa bits -- within [2**e, 2**(e+1)) the ULP is exactly
    # 2**(e-10), and v >= 2**e there, so ulp(v)/v <= 2**-10). Instantiating
    # the general lemma's `step` with that worst-case relative spacing gives
    # a reconstruction error of at most v * 2**-11 -- the "roughly half a
    # ULP, about a thousandth of the value" folk description of float16
    # precision, now as a checked bound rather than an approximation.
    v = z3.Real("v")
    n = z3.Int("n")
    half = z3.RealVal(1) / 2
    step = v * (z3.RealVal(1) / 1024)  # v * 2**-10

    hypotheses = z3.And(v > 0, n - v / step <= half, v / step - n <= half)
    rounded = n * step
    error = v - rounded
    bound = v * (z3.RealVal(1) / 2048)  # v * 2**-11
    prove(z3.Implies(hypotheses, z3.And(error <= bound, -error <= bound)))


def test_quantize_fp16_clamp_saturates_to_max_magnitude():
    # This pass's own documented design choice (the header comment: "this
    # pass never silently introduces a new infinity"): a value beyond
    # float16's largest finite magnitude clamps to *exactly* that magnitude,
    # not to something proportionally further off and not to infinity.
    # FloatToFloat16Bits clamps first and rounds second, so a value just
    # past 65504 cannot land one ULP beyond it either -- it becomes the
    # clamp bound itself, bit for bit.
    v = z3.Real("v")
    max_mag = z3.RealVal(_FLOAT16_MAX)
    clamped = z3.If(v > max_mag, max_mag, z3.If(v < -max_mag, -max_mag, v))
    prove(z3.Implies(v > max_mag, clamped == max_mag))
    prove(z3.Implies(v < -max_mag, clamped == -max_mag))


def test_quantize_fp16_negative_control_needs_nearest_hypothesis():
    # Sanity check that the bound above is genuine, not vacuous: with only
    # step > 0 assumed (dropping the "n is nearest to v/step" hypothesis
    # entirely), the half-step bound is not a theorem -- Z3 must find a real
    # counterexample where an arbitrary "quantized" value n * step differs
    # from v by more than step / 2.
    v, step = z3.Reals("v step")
    n = z3.Int("n")
    rounded = n * step
    error = v - rounded

    solver = z3.Solver()
    solver.add(step > 0)
    solver.add(z3.Not(z3.And(error <= step / 2, -error <= step / 2)))
    assert solver.check() == z3.sat, (
        "the half-step bound holds even without constraining n to be "
        "nearest to v/step -- negative control is vacuous"
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
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _node_input_initializer(model, op_type, input_index):
    # quantize_fp16 replaces a converted weight with a *new* initializer,
    # leaving the old float32 one orphaned in the model, so the initializer
    # actually feeding a node must be looked up by that node's current input
    # name, not by indexing graph.initializer blindly.
    node = next(n for n in model.graph.node if n.op_type == op_type)
    name = node.input[input_index]
    return next(init for init in model.graph.initializer if init.name == name)


def test_quantize_fp16_is_an_opt_in_pass():
    C = onnxsim.onnxsim_cpp2py_export
    assert "quantize_fp16" in C._list_other_optimizers()
    assert "quantize_fp16" not in C._list_optimizers()


def test_quantize_fp16_keep_io_types_rounds_weight_and_inserts_boundary_casts():
    # Non-tie weight values (arbitrary decimals essentially never land
    # exactly halfway between two representable float16 values) -- checked
    # bit-for-bit against numpy's own .astype(np.float16) as an independent
    # reference, valid here per CLAUDE.md's guidance even though numpy's own
    # tie-breaking rule (round-half-to-even) differs from this pass's
    # documented ties-away-from-zero, since neither value is a tie.
    w = np.array([0.1, -0.2, 1234.25, -6.7], dtype=np.float32)
    model = _model(
        """
        g (float[4] X) => (float[4] Y)
        {
          Y = Add(X, W)
        }
        """,
        initializer=[_f32(w, "W")],
    )

    quant = onnxsim.quantize_fp16(model)  # keep_io_types defaults to True
    onnx.checker.check_model(quant)

    # Boundary casts, right after the input and right before the output.
    assert [n.op_type for n in quant.graph.node] == ["Cast", "Add", "Cast"]
    input_cast, add_node, output_cast = quant.graph.node
    assert input_cast.input[0] == quant.graph.input[0].name
    assert add_node.input[0] == input_cast.output[0]
    assert output_cast.input[0] == add_node.output[0]
    assert output_cast.output[0] == quant.graph.output[0].name
    to_attr = {a.name: a.i for a in input_cast.attribute}
    assert to_attr["to"] == onnx.TensorProto.FLOAT16
    to_attr = {a.name: a.i for a in output_cast.attribute}
    assert to_attr["to"] == onnx.TensorProto.FLOAT

    # The graph's own declared I/O stays float32.
    assert quant.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert quant.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    w_init = _node_input_initializer(quant, "Add", 1)
    assert w_init.data_type == onnx.TensorProto.FLOAT16
    w_fp16 = numpy_helper.to_array(w_init)
    np.testing.assert_array_equal(
        w_fp16.view(np.uint16), w.astype(np.float16).view(np.uint16)
    )

    # Overall numeric output stays close to (not equal to) the original.
    x = np.array([1.0, -2.5, 100.0, 0.3], dtype=np.float32)
    orig_sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    quant_sess = ort.InferenceSession(
        quant.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (orig_y,) = orig_sess.run(None, {"X": x})
    (quant_y,) = quant_sess.run(None, {"X": x})
    assert not np.array_equal(orig_y, quant_y)  # genuinely lossy, not a no-op
    np.testing.assert_allclose(quant_y, orig_y, rtol=5e-3, atol=1e-2)


def test_quantize_fp16_ties_away_from_zero():
    # A value hand-constructed to be an EXACT float16 tie: within [1, 2),
    # float16's representable grid is 1 + k/1024 for integer k. Pick k=2
    # (even mantissa bits -- the "round to even" candidate) and its
    # neighbor k=3 (odd); the exact midpoint 1 + 2.5/1024 is a dyadic
    # fraction with denominator 2**11, representable exactly in float32 (and
    # in ordinary Python/double-precision arithmetic), so no incidental
    # rounding sneaks into building the test value itself.
    #
    # Round-half-to-even would pick k=2 (already even); this pass's
    # documented rule (ties away from zero) instead picks k=3 -- the
    # larger-magnitude neighbor -- for both signs. The expected results
    # (1 + 3/1024 and its negation) are themselves exactly representable in
    # float16, so no rounding is needed to compute them independently.
    low = 1.0 + 2 / 1024.0
    high = 1.0 + 3 / 1024.0
    tie = (low + high) / 2.0
    assert tie == 1.0 + 2.5 / 1024.0  # sanity: exact midpoint, no drift

    w = np.array([tie, -tie], dtype=np.float32)
    model = _model(
        """
        g (float[2] X) => (float[2] Y)
        {
          Y = Add(X, W)
        }
        """,
        initializer=[_f32(w, "W")],
    )

    quant = onnxsim.quantize_fp16(model)
    onnx.checker.check_model(quant)

    w_init = _node_input_initializer(quant, "Add", 1)
    w_fp16 = numpy_helper.to_array(w_init)
    assert w_fp16[0] == np.float16(high)
    assert w_fp16[1] == np.float16(-high)
    # Not the round-half-to-even answer -- confirms the tie was actually
    # resolved away from zero, not merely landing on the same value either
    # rule would produce.
    assert w_fp16[0] != np.float16(low)
    assert w_fp16[1] != np.float16(-low)


def test_quantize_fp16_clamps_out_of_range_weight():
    # A magnitude far beyond float16's largest finite value (65504) must
    # clamp to exactly +-65504, never inf/nan -- read back from the raw
    # float16 bytes directly, independent of numpy_helper.to_array's own
    # decoding path.
    w = np.array([1.0e10, -1.0e10, 3.0], dtype=np.float32)
    model = _model(
        """
        g (float[3] X) => (float[3] Y)
        {
          Y = Add(X, W)
        }
        """,
        initializer=[_f32(w, "W")],
    )

    quant = onnxsim.quantize_fp16(model)
    onnx.checker.check_model(quant)

    w_init = _node_input_initializer(quant, "Add", 1)
    w_bits = np.frombuffer(w_init.raw_data, dtype=np.float16)
    assert np.all(np.isfinite(w_bits))
    assert w_bits[0] == np.float16(_FLOAT16_MAX)
    assert w_bits[1] == np.float16(-_FLOAT16_MAX)
    assert w_bits[2] == np.float16(3.0)

    x = np.zeros(3, dtype=np.float32)
    (out,) = ort.InferenceSession(
        quant.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"X": x})
    assert np.all(np.isfinite(out))


def test_quantize_fp16_no_keep_io_types_redeclares_io_directly():
    w = np.array([0.1, -0.2, 1234.25, -6.7], dtype=np.float32)
    model = _model(
        """
        g (float[4] X) => (float[4] Y)
        {
          Y = Add(X, W)
        }
        """,
        initializer=[_f32(w, "W")],
    )

    quant = onnxsim.quantize_fp16(model, keep_io_types=False)
    onnx.checker.check_model(quant)

    # No boundary casts -- the graph's own I/O is redeclared float16 directly.
    assert [n.op_type for n in quant.graph.node] == ["Add"]
    assert quant.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert quant.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16

    x = np.array([1.0, -2.5, 100.0, 0.3], dtype=np.float32)
    x16 = x.astype(np.float16)
    orig_sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    quant_sess = ort.InferenceSession(
        quant.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (orig_y,) = orig_sess.run(None, {"X": x})
    (quant_y,) = quant_sess.run(None, {"X": x16})
    np.testing.assert_allclose(quant_y, orig_y, rtol=5e-3, atol=1e-2)
