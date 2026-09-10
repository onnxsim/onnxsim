"""Formal check for QuantizeBf16 (opt-in; ``onnxsim/passes/quantize_bf16.h``):
converts every float32 initializer/``Constant``-node value in the top-level
graph to bfloat16, round-to-nearest (ties away from zero -- see below). With
``keep_io_types`` (the pass's own default, read from the function-local
static ``QuantizeBf16KeepIoTypes()``), the graph's own external input/output
types stay float32 via a boundary ``Cast``; with it false, inputs/outputs are
redeclared bfloat16 directly. Full mechanics in the header comment -- read in
full before this file.

``quantize_bf16`` is ``quantize_fp16``'s sibling: same whole-graph, no-
calibration, floating-point-not-integer "quantization" design (constant
conversion via ``FetchConstantTensor``, the same ``keep_io_types`` boundary-
Cast mechanics, top-level-graph-only scope), reached through Python the same
way -- grepping ``onnxsim/*.py`` turns up ``onnxsim.quantize_bf16(model,
keep_io_types=True)`` (``onnxsim/onnx_simplifier.py``), analogous to
``onnxsim.quantize_fp16`` -- so every differential test below calls it
directly, the same as ``test_formal_verify_quantize_fp16.py`` does for its
own pass, rather than going through ``simplify_isolated_extra`` (which has no
way to plumb ``keep_io_types`` through).

Where bfloat16 genuinely differs from float16: it keeps float32's full 8-bit
exponent and narrows *only* the mantissa, to 7 stored bits (float16 narrows
both -- 5 exponent bits, 10 mantissa bits). So a bfloat16 value's bit pattern
is literally the top 16 bits of its float32 counterpart, rounded from the
discarded low 16 bits (``FloatToBFloat16Bits``: ``bits + 0x00008000``, then
truncate) -- there is no separate exponent-encoding step to get subtly wrong.
Because bfloat16's exponent range is *exactly* float32's, no finite float32
value can ever fall outside it: **no clamping and no subnormal handling
exist in this pass at all**, unlike ``quantize_fp16``'s explicit
clamp-to-65504. This is a genuine simplification of the proof, not an
omission -- there is no clamp lemma in this file for the same reason there is
no boat in a proof about trains: the mechanism this file verifies simply has
no clamp branch, and the header comment says so explicitly ("no finite
float32 value can overflow it").

Formal content, mirroring ``test_formal_verify_quantize_fp16.py``'s shape:

1. ``test_quantize_bf16_round_to_nearest_is_half_step_bound``: the same
   general, scheme-agnostic "round to nearest is a half-step bound" lemma
   ``test_formal_verify_quantize_fp16.py`` proves -- restated here
   independently (per this suite's convention of every file being
   self-contained even when a proof's *shape* is shared, the same way
   ``test_formal_verify_dynamic_quantize_matmul.py`` and
   ``test_formal_verify_weight_only_quantize_matmul.py`` each restate a
   closely related bound rather than importing one from the other): for any
   real ``v`` and any grid spacing ``step > 0``, a value ``n`` steps from the
   origin that is *nearest* to ``v / step`` (within 0.5) reconstructs ``v``
   to within ``step / 2``.
2. ``test_quantize_bf16_worst_case_relative_bound_near_precision_limit``:
   bfloat16's own corollary, with bfloat16's own (coarser) precision in
   place of float16's -- 7 stored mantissa bits, not 10, so the worst-case
   local step (ULP) for a normalized ``v > 0`` is ``v * 2**-7`` (within
   ``[2**e, 2**(e+1))`` the ULP is exactly ``2**(e-7)``, and ``v >= 2**e``
   there), giving a reconstruction-error bound of ``v * 2**-8`` --
   substantially coarser than float16's ``v * 2**-11``, matching bfloat16's
   well-known folk description of "2-3 decimal digits" versus float16's
   "3-4": bfloat16 spends the mantissa bits float16 keeps on exponent range
   instead, buying float32-equivalent dynamic range at the cost of
   precision.
3. No clamp-saturation lemma. Explicitly absent, not merely unwritten: see
   this docstring's second paragraph above for why (bfloat16's exponent
   range exactly matches float32's, so this pass's ``FloatToBFloat16Bits``
   has no clamp branch for a lemma to describe).
4. ``test_quantize_bf16_negative_control_needs_nearest_hypothesis``: same
   shape as ``quantize_fp16``'s own negative control -- with only
   ``step > 0`` assumed (dropping "n is nearest to v/step" entirely), the
   half-step bound is not a theorem; Z3 finds a real counterexample,
   confirming the bound genuinely depends on the rounding hypothesis.

No substitution/composition step is attempted here, for the same reason
``quantize_fp16``'s own docstring gives (and ``dynamic_quantize_matmul``'s
before that): an arbitrary downstream consumer need not be Lipschitz, so a
per-element numeric bound doesn't by itself bound anything about a consumer
applied to it. The differential tests below instead check real end-to-end
numeric behavior on concrete models.

Differential tests build models via ``onnx.parser.parse_model()`` (per
``CLAUDE.md``) and call the real ``onnxsim.quantize_bf16`` entry point found
above. One environment-specific wrinkle drives their shape: **this
environment's ONNX Runtime CPU build has essentially no BFLOAT16 compute
kernels** -- ``Add``/``Sub``/``Mul``/``Div``/``MatMul``/``Gemm``/``Sum``/
``Min``/``Max``/``Neg``/``Abs``/``Sqrt``/``ReduceSum``/``Softmax``/``Where``
all raise ``NOT_IMPLEMENTED`` for BFLOAT16 on ``CPUExecutionProvider``
(checked empirically for this file), across every opset tried -- only a
handful of pure data-movement ops (``Identity``, ``Reshape``, ``Transpose``,
``Concat``, ``Slice``, ``Gather``, ``Clip``, ``Expand``) have a registered
BFLOAT16 kernel here. So every differential model below uses ``Concat(X, W)``
as its one computational node in place of ``quantize_fp16``'s own
``Add(X, W)`` -- data movement is enough to exercise real per-element
bfloat16 rounding of both the weight and (via the boundary Cast) the
activation, without needing a BFLOAT16 arithmetic kernel that doesn't exist
here. Separately, numpy has no native bfloat16 dtype for ONNX Runtime's
Python bindings to map to (unlike float16, which numpy supports natively and
``quantize_fp16``'s own file feeds/reads directly) -- ``ml_dtypes`` (checked
available in this environment; a hand-rolled bit-truncation fallback is used
if not) supplies an independent reference bfloat16 conversion, and the
``keep_io_types=False`` numeric test feeds/reads raw ``uint16`` buffers via
``OrtValue.ortvalue_from_numpy_with_onnx_type``/``ctypes`` rather than a
native array dtype.

The tests: (1) ``quantize_bf16_is_an_opt_in_pass`` -- confirmed via
``_list_other_optimizers()``/``_list_optimizers()``. (2) with the pass's own
default ``keep_io_types=True``, a non-tie float32 weight becomes the correct
bfloat16 bit pattern (checked against the independent reference above) and
boundary ``Cast`` nodes appear exactly where the header comment says; the
overall numeric output, run through onnxruntime, stays close to (not equal
to) the unquantized model's. (3) A value hand-constructed to be an *exact*
bfloat16 tie -- included, since one is genuinely easy to find here (bfloat16's
7-bit mantissa makes ``1 + 2.5/128`` an exact midpoint, exactly the same
construction ``quantize_fp16``'s own tie test uses at float16's own
mantissa width) -- confirms ties-away-from-zero, checked against both a
hand-rolled reimplementation of ``FloatToBFloat16Bits``'s bit formula *and*
the real compiled pass directly (both independently verified while writing
this file to round the tie the same way: away from zero, matching the header
comment's claim at face value). (4) An extreme magnitude (``1e30`` --
well within float32's range, and (unlike ``quantize_fp16``'s old 65504
clamp point) nowhere near any limit bfloat16 has) converts cleanly to a
large but finite bfloat16 value of the same order of magnitude, never
clamped/saturated -- the single most distinguishing differential behavior
versus this pass's sibling. (5) ``keep_io_types=False`` redeclares I/O as
BFLOAT16 directly, with no boundary casts.
"""

import ctypes

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import pytest
from _formal_verify_common import prove, z3
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")

try:
    import ml_dtypes

    _HAS_ML_DTYPES = True
except ImportError:  # pragma: no cover - depends on the test environment
    ml_dtypes = None
    _HAS_ML_DTYPES = False


def test_quantize_bf16_round_to_nearest_is_half_step_bound():
    # n stands for "the number of grid steps the rounded value sits at" --
    # constrained only by *being nearest* to v / step (within 0.5, the same
    # abstract modeling of round() quantize_round_trip's own proof uses, true
    # regardless of tie-breaking rule since a tie only ever occurs exactly
    # halfway, where both neighbors already satisfy the bound). No
    # zero_point term: bfloat16's grid, like float16's, is symmetric about
    # zero, unlike uniform-affine quantization's offset code.
    v, step = z3.Reals("v step")
    n = z3.Int("n")
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(step > 0, n - v / step <= half, v / step - n <= half)
    rounded = n * step
    error = v - rounded
    prove(z3.Implies(hypotheses, z3.And(error <= step / 2, -error <= step / 2)))


def test_quantize_bf16_worst_case_relative_bound_near_precision_limit():
    # bfloat16's own corollary of the general bound above: for a normalized
    # value v > 0, the local grid spacing (ULP) is at most v * 2**-7 (7
    # stored mantissa bits -- within [2**e, 2**(e+1)) the ULP is exactly
    # 2**(e-7), and v >= 2**e there, so ulp(v)/v <= 2**-7). Instantiating the
    # general lemma's `step` with that worst-case relative spacing gives a
    # reconstruction error of at most v * 2**-8 -- coarser than float16's
    # v * 2**-11 by design (3 fewer mantissa bits), the "2-3 decimal digits"
    # folk description of bfloat16 precision, now as a checked bound.
    v = z3.Real("v")
    n = z3.Int("n")
    half = z3.RealVal(1) / 2
    step = v * (z3.RealVal(1) / 128)  # v * 2**-7

    hypotheses = z3.And(v > 0, n - v / step <= half, v / step - n <= half)
    rounded = n * step
    error = v - rounded
    bound = v * (z3.RealVal(1) / 256)  # v * 2**-8
    prove(z3.Implies(hypotheses, z3.And(error <= bound, -error <= bound)))


def test_quantize_bf16_negative_control_needs_nearest_hypothesis():
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


def _model(body, initializer=(), opset=18, ir_version=10):
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
    # quantize_bf16 replaces a converted weight with a *new* initializer,
    # leaving the old float32 one orphaned in the model, so the initializer
    # actually feeding a node must be looked up by that node's current input
    # name, not by indexing graph.initializer blindly.
    node = next(n for n in model.graph.node if n.op_type == op_type)
    name = node.input[input_index]
    return next(init for init in model.graph.initializer if init.name == name)


def _bf16_bits_reference(array):
    """Independent reference bfloat16 conversion: ml_dtypes' own bfloat16
    dtype when available, else a hand-rolled reimplementation of
    FloatToBFloat16Bits's own round-to-nearest-with-carry formula
    (add 0x8000, truncate to the top 16 bits), vectorized in numpy -- numpy
    itself has no native bfloat16 dtype (unlike float16), so unlike
    ``quantize_fp16``'s file this can't simply be ``array.astype(...)``
    against a numpy builtin.
    """
    array = np.asarray(array, dtype=np.float32)
    if _HAS_ML_DTYPES:
        return array.astype(ml_dtypes.bfloat16).view(np.uint16)
    bits = array.view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x00008000) & 0xFFFFFFFF
    return (rounded >> 16).astype(np.uint16)


def _bf16_bits_to_float32(bits_u16):
    """Decode raw bfloat16 bit patterns back to float32: a bfloat16 value's
    bits are exactly the top 16 bits of a float32 value with its low 16 bits
    zeroed, so left-shifting by 16 and reinterpreting is exact -- no library
    dependency needed for this direction.
    """
    bits_u16 = np.asarray(bits_u16, dtype=np.uint16)
    return (bits_u16.astype(np.uint32) << 16).view(np.float32)


def test_quantize_bf16_is_an_opt_in_pass():
    C = onnxsim.onnxsim_cpp2py_export
    assert "quantize_bf16" in C._list_other_optimizers()
    assert "quantize_bf16" not in C._list_optimizers()


def test_quantize_bf16_keep_io_types_rounds_weight_and_inserts_boundary_casts():
    # Non-tie weight values (arbitrary decimals essentially never land
    # exactly halfway between two representable bfloat16 values) -- checked
    # bit-for-bit against the independent reference above. Concat(X, W)
    # stands in for quantize_fp16's own Add(X, W): see this file's docstring
    # for why (no BFLOAT16 Add/Mul/... kernel exists on CPUExecutionProvider
    # in this environment).
    w = np.array([0.1, -0.2, 1234.25, -6.7], dtype=np.float32)
    model = _model(
        """
        g (float[4] X) => (float[8] Y)
        {
          Y = Concat<axis=0>(X, W)
        }
        """,
        initializer=[_f32(w, "W")],
    )

    quant = onnxsim.quantize_bf16(model)  # keep_io_types defaults to True
    onnx.checker.check_model(quant)

    # Boundary casts, right after the input and right before the output.
    assert [n.op_type for n in quant.graph.node] == ["Cast", "Concat", "Cast"]
    input_cast, concat_node, output_cast = quant.graph.node
    assert input_cast.input[0] == quant.graph.input[0].name
    assert concat_node.input[0] == input_cast.output[0]
    assert output_cast.input[0] == concat_node.output[0]
    assert output_cast.output[0] == quant.graph.output[0].name
    to_attr = {a.name: a.i for a in input_cast.attribute}
    assert to_attr["to"] == onnx.TensorProto.BFLOAT16
    to_attr = {a.name: a.i for a in output_cast.attribute}
    assert to_attr["to"] == onnx.TensorProto.FLOAT

    # The graph's own declared I/O stays float32.
    assert quant.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert quant.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    w_init = _node_input_initializer(quant, "Concat", 1)
    assert w_init.data_type == onnx.TensorProto.BFLOAT16
    w_bits = np.frombuffer(w_init.raw_data, dtype=np.uint16)
    np.testing.assert_array_equal(w_bits, _bf16_bits_reference(w))

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
    np.testing.assert_allclose(quant_y, orig_y, rtol=2e-2, atol=1e-2)


def test_quantize_bf16_ties_away_from_zero():
    # A value hand-constructed to be an EXACT bfloat16 tie: within [1, 2),
    # bfloat16's representable grid is 1 + k/128 for integer k (7 stored
    # mantissa bits, vs. float16's 1 + k/1024 at 10 bits -- same
    # construction as quantize_fp16's own tie test, just at bfloat16's own,
    # coarser grid spacing). Pick k=2 (even -- the "round to even" candidate)
    # and its neighbor k=3 (odd); the exact midpoint 1 + 2.5/128 is a dyadic
    # fraction with denominator 2**8, representable exactly in float32 (and
    # in ordinary Python/double-precision arithmetic), so no incidental
    # rounding sneaks into building the test value itself.
    #
    # Round-half-to-even would pick k=2 (already even); this pass's
    # documented rule (ties away from zero) instead picks k=3 -- the
    # larger-magnitude neighbor -- for both signs. Verified two ways: against
    # a hand-rolled reimplementation of FloatToBFloat16Bits's own bit formula
    # (add 0x8000, truncate) directly in Python, and below against the real
    # compiled pass -- both round this exact tie away from zero, confirming
    # the header comment's claim at face value.
    low = 1.0 + 2 / 128.0
    high = 1.0 + 3 / 128.0
    tie = (low + high) / 2.0
    assert tie == 1.0 + 2.5 / 128.0  # sanity: exact midpoint, no drift

    w = np.array([tie, -tie], dtype=np.float32)
    model = _model(
        """
        g (float[2] X) => (float[4] Y)
        {
          Y = Concat<axis=0>(X, W)
        }
        """,
        initializer=[_f32(w, "W")],
    )

    quant = onnxsim.quantize_bf16(model)
    onnx.checker.check_model(quant)

    w_init = _node_input_initializer(quant, "Concat", 1)
    w_bits = np.frombuffer(w_init.raw_data, dtype=np.uint16)
    w_bf16 = _bf16_bits_to_float32(w_bits)
    assert w_bf16[0] == np.float32(high)
    assert w_bf16[1] == np.float32(-high)
    # Not the round-half-to-even answer -- confirms the tie was actually
    # resolved away from zero, not merely landing on the same value either
    # rule would produce.
    assert w_bf16[0] != np.float32(low)
    assert w_bf16[1] != np.float32(-low)


def test_quantize_bf16_extreme_magnitude_stays_finite_not_clamped():
    # A magnitude far beyond float16's old 65504 clamp point but well within
    # float32's own range: unlike quantize_fp16, this pass has no clamp at
    # all (see the module docstring), so this must convert cleanly to a
    # large but FINITE bfloat16 value of the same order of magnitude --
    # never a smaller clamped value, and never inf/nan.
    w = np.array([1.0e30, -1.0e30, 3.0], dtype=np.float32)
    model = _model(
        """
        g (float[3] X) => (float[6] Y)
        {
          Y = Concat<axis=0>(X, W)
        }
        """,
        initializer=[_f32(w, "W")],
    )

    quant = onnxsim.quantize_bf16(model)
    onnx.checker.check_model(quant)

    w_init = _node_input_initializer(quant, "Concat", 1)
    w_bits = np.frombuffer(w_init.raw_data, dtype=np.uint16)
    w_bf16 = _bf16_bits_to_float32(w_bits)
    assert np.all(np.isfinite(w_bf16))
    # Same order of magnitude as the original -- not clamped/saturated to
    # something far smaller the way quantize_fp16 would (to +-65504).
    np.testing.assert_allclose(w_bf16[0], 1.0e30, rtol=1e-2)
    np.testing.assert_allclose(w_bf16[1], -1.0e30, rtol=1e-2)
    assert w_bf16[2] == 3.0  # exact: 3.0's mantissa needs no rounding at all

    x = np.zeros(3, dtype=np.float32)
    (out,) = ort.InferenceSession(
        quant.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"X": x})
    assert np.all(np.isfinite(out))


def test_quantize_bf16_no_keep_io_types_redeclares_io_directly():
    w = np.array([0.1, -0.2, 1234.25, -6.7], dtype=np.float32)
    model = _model(
        """
        g (float[4] X) => (float[8] Y)
        {
          Y = Concat<axis=0>(X, W)
        }
        """,
        initializer=[_f32(w, "W")],
    )

    quant = onnxsim.quantize_bf16(model, keep_io_types=False)
    onnx.checker.check_model(quant)

    # No boundary casts -- the graph's own I/O is redeclared bfloat16 directly.
    assert [n.op_type for n in quant.graph.node] == ["Concat"]
    assert quant.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
    assert quant.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16

    # numpy has no native bfloat16 dtype for onnxruntime's Python bindings to
    # map to (unlike float16's x.astype(np.float16), which quantize_fp16's
    # own file uses directly) -- feed/read raw uint16 buffers instead, via
    # ortvalue_from_numpy_with_onnx_type on the way in and a raw pointer read
    # (data_ptr/ctypes) on the way out.
    x = np.array([1.0, -2.5, 100.0, 0.3], dtype=np.float32)
    x_bits = _bf16_bits_reference(x)
    x_ortvalue = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
        x_bits, onnx.TensorProto.BFLOAT16
    )

    quant_sess = ort.InferenceSession(
        quant.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y_ortvalue,) = quant_sess.run_with_ort_values(["Y"], {"X": x_ortvalue})
    n = y_ortvalue.shape()[0]
    y_bits = np.ctypeslib.as_array(
        (ctypes.c_uint16 * n).from_address(y_ortvalue.data_ptr())
    ).copy()
    quant_y = _bf16_bits_to_float32(y_bits)

    orig_sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (orig_y,) = orig_sess.run(None, {"X": x})
    np.testing.assert_allclose(quant_y, orig_y, rtol=2e-2, atol=1e-2)
