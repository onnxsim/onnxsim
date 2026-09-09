"""Formal check for QuantizeFp8 (opt-in; ``onnxsim/passes/quantize_fp8.h``):
converts every float32 initializer/``Constant``-node value in the top-level
graph to an 8-bit floating-point format, selectable between two formats
introduced in onnx 1.15.0:

- **E4M3FN** (the pass's own default): 1 sign, 4 exponent, 3 mantissa bits,
  bias 7, max finite magnitude 448. No infinity encoding at all -- the
  exponent-all-ones pattern is reserved entirely for a single NaN encoding
  per sign.
- **E5M2**: 1 sign, 5 exponent, 2 mantissa bits, bias 15, max finite
  magnitude 57344. Does have a real infinity encoding, but this pass
  deliberately never produces it (see the clamp discussion below).

With ``keep_io_types`` (the pass's own default, read from the function-local
static ``QuantizeFp8KeepIoTypes()``), the graph's own external input/output
types stay float32 via a boundary ``Cast``; with it false, inputs/outputs are
redeclared in the target float8 format directly. Full mechanics in the
header comment -- read in full before this file.

``quantize_fp8`` is ``quantize_fp16``/``quantize_bf16``'s sibling: same
whole-graph, no-calibration, floating-point-not-integer "quantization"
design (constant conversion via ``FetchConstantTensor``, the same
``keep_io_types`` boundary-Cast mechanics, top-level-graph-only scope),
reached through Python the same way -- grepping ``onnxsim/*.py`` turns up
``onnxsim.quantize_fp8(model, format="e4m3", keep_io_types=True)``
(``onnxsim/onnx_simplifier.py``) -- but with a genuinely new wrinkle those
two sibling files didn't need: ``format`` (``"e4m3"`` or ``"e5m2"``) selects
which of the two float8 layouts to convert to, plumbed the same way as
``keep_io_types`` -- a second C++ function-local static
(``QuantizeFp8TargetFormat()``) set by ``QuantizeFp8`` in ``onnxsim.cpp``
immediately before invoking ``OptimizeFixed`` -- so every differential test
below calls the real ``onnxsim.quantize_fp8`` entry point directly with
explicit ``format``/``keep_io_types`` values, rather than going through
``simplify_isolated_extra`` (which has no way to plumb either through).

Where float8 genuinely differs from its ``quantize_fp16``/``quantize_bf16``
siblings: rounding here is round-to-nearest **ties-to-even**, not
ties-away-from-zero. Float8's mantissa is only 2-3 bits wide, so an exact
tie between two representable values is common enough on real weight/
activation data (unlike float16/bfloat16, where a tie is a constructed edge
case) that the header comment calls out ties-to-even as float8's own
documented standard (RNE). Both formats share one bit-conversion
implementation (``FloatToFloat8Bits``, parametrized by
``mantissa_bits``/``bias``/``max_value``/``nan_pattern``) -- unlike
``quantize_fp16``/``quantize_bf16``, whose dissimilar bit layouts and
simpler rounding rule made near-duplicate copies the more readable choice
there.

Formal content, mirroring ``test_formal_verify_quantize_fp16.py``'s shape
but adapted for two formats and the different tie-breaking rule:

1. ``test_quantize_fp8_round_to_nearest_is_half_step_bound``: the same
   general, scheme-agnostic "round to nearest is a half-step bound" lemma
   every sibling file proves -- restated here independently (per this
   suite's convention of every file being self-contained even when a
   proof's *shape* is shared): for any real ``v`` and any grid spacing
   ``step > 0``, a value ``n`` steps from the origin that is *nearest* to
   ``v / step`` (within 0.5) reconstructs ``v`` to within ``step / 2``. This
   holds regardless of tie-breaking rule -- a tie only ever occurs exactly
   halfway, where both neighbors already satisfy the bound -- so it covers
   ties-to-even just as well as it covered ``quantize_fp16``'s/
   ``quantize_bf16``'s ties-away-from-zero; only the *differential* tests
   below need to distinguish the two rules.
2. ``test_quantize_fp8_worst_case_relative_bound_e4m3fn`` /
   ``..._e5m2``: TWO format-specific corollaries, one per supported format,
   each instantiating the general lemma with that format's own mantissa-bit
   count. E4M3FN's 3 stored mantissa bits give a worst-case local step (ULP)
   of ``v * 2**-3`` for a normalized ``v > 0`` (within ``[2**e, 2**(e+1))``
   the ULP is exactly ``2**(e-3)``, and ``v >= 2**e`` there), so a
   reconstruction-error bound of ``v * 2**-4``. E5M2's 2 stored mantissa
   bits give ``v * 2**-2`` and ``v * 2**-3`` respectively -- coarser still.
   Both are *much* coarser than float16's ``v * 2**-11`` or bfloat16's
   ``v * 2**-8``, as expected for an 8-bit format: this is the "roughly 1-2
   decimal digits" folk description of float8 precision, now as checked
   bounds rather than an approximation.
3. ``test_quantize_fp8_clamp_saturates_to_max_magnitude`` (parametrized over
   both formats' own max finite value, 448.0 and 57344.0): present here,
   unlike ``quantize_bf16``'s file which correctly omits a clamp lemma
   (bfloat16's exponent range exactly matches float32's, so there is nothing
   to clamp) -- ``quantize_fp8``, like ``quantize_fp16``, saturates: for
   ``|v|`` beyond the target format's max finite value, the clamped value is
   *exactly* that max finite value, not proportionally further off and not
   an infinity/NaN. One parametrized proof covers both formats' actual
   clamp constants rather than two near-identical copies, since the claim's
   shape (and Z3 encoding) is identical for either constant. Worth calling
   out explicitly for E5M2 in particular: E5M2 *does* have a real infinity
   encoding (unlike E4M3FN), but this pass's saturating-clamp design means
   it never actually emits it -- both formats share one clamp code path
   specifically so this holds uniformly (see the header comment).
4. ``test_quantize_fp8_negative_control_needs_nearest_hypothesis``: same
   shape as every sibling file's own negative control -- with only
   ``step > 0`` assumed (dropping "n is nearest to v/step" entirely), the
   half-step bound is not a theorem; Z3 finds a real counterexample,
   confirming the bound genuinely depends on the rounding hypothesis.

No substitution/composition step is attempted here, for the same reason
``quantize_fp16``'s/``quantize_bf16``'s own docstrings give: an arbitrary
downstream consumer need not be Lipschitz, so a per-element numeric bound
doesn't by itself bound anything about a consumer applied to it. The
differential tests below instead check real end-to-end numeric behavior on
concrete models.

Differential tests build models via ``onnx.parser.parse_model()`` (per
``CLAUDE.md``) and call the real ``onnxsim.quantize_fp8`` entry point found
above. Model opsets must be >= 19 (unlike the fp16/bf16 files' opset 13/18)
since ``Cast``'s own float8 support -- and float8 element types at all --
only exist from opset 19 on.

Two environment-specific wrinkles drive the differential tests' shape,
checked empirically while writing this file (the same way
``test_formal_verify_quantize_bf16.py`` checked BFLOAT16 kernel coverage):

- **ml_dtypes** (checked available in this environment) supplies native
  ``float8_e4m3fn``/``float8_e5m2`` numpy dtypes, used the same way
  ``test_formal_verify_quantize_bf16.py`` used ``ml_dtypes.bfloat16`` -- an
  independent reference conversion, confirmed (see
  ``_float8_bits_reference``'s own check while writing this file) to use
  round-to-nearest ties-to-even, matching this pass's documented rule. A
  hand-rolled fallback (a direct Python port of ``FloatToFloat8Bits``) is
  used if ml_dtypes isn't installed.
- **ONNX Runtime's float8 kernel coverage in this environment is even
  narrower than BFLOAT16's own** (``test_formal_verify_quantize_bf16.py``'s
  own finding): unlike that file, whose ``Concat``-based data-movement
  workaround at least had a registered BFLOAT16 kernel, this environment's
  CPU build has *no* registered FLOAT8E4M3FN/FLOAT8E5M2 kernel for any of
  ``Add``/``Mul``/``Concat``/``Transpose``/``Slice``/``Squeeze``/``Flatten``/
  ``Where``/``Expand``/``Gather``/``Reshape`` (checked empirically -- each
  raises ``INVALID_GRAPH``/``NOT_IMPLEMENTED`` for a float8 tensor type) --
  only ``Identity`` and ``CastLike`` have one. So every differential model
  below uses two independent ``Identity`` branches, one from the graph input
  and one from the weight initializer, each round-tripped through the
  target float8 format via boundary ``Cast`` nodes and back, in place of a
  single combining op like ``Add``/``Concat`` -- this still exercises real
  per-element float8 rounding of both the weight and (via the boundary
  Cast) the activation, run end-to-end through ``onnxruntime``, without
  needing a float8 arithmetic/data-movement kernel that doesn't exist here.

The tests: (1) ``quantize_fp8_is_an_opt_in_pass`` -- confirmed via
``_list_other_optimizers()``/``_list_optimizers()``. (2) with the pass's own
default ``keep_io_types=True`` and default ``format="e4m3"``, a non-tie
float32 weight becomes the correct E4M3FN bit pattern (checked against the
independent reference above) and boundary ``Cast`` nodes appear exactly
where the header comment says; the overall numeric output, run through
onnxruntime, stays close to (not equal to) the unquantized model's. (3) A
value hand-constructed to be an *exact* tie at each format's own mantissa
grid confirms **ties-to-even** specifically, for both E4M3FN and E5M2 --
deliberately checking the EVEN neighbor wins, the opposite rounding rule
from ``quantize_fp16``'s/``quantize_bf16``'s own ties-away-from-zero tests.
(4) An out-of-range magnitude, for both formats, clamps to exactly that
format's own max finite value (448 for E4M3FN, 57344 for E5M2), read back
from raw bytes, never inf/nan -- for E5M2 specifically this is the "avoids
ever emitting infinity" behavior the header comment calls out. (5)
``keep_io_types=False`` redeclares I/O in the target format directly, with
no boundary casts. (6) A minimal check that ``format="e4m3"`` and
``format="e5m2"`` are genuinely distinguishable -- the same float32 value
converts to different bit patterns and a different tensor element type
under each.
"""

import math
import struct

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

# mantissa_bits, bias, max_value, canonical-NaN bit pattern, TensorProto
# element type -- mirrors quantize_fp8.h's own per-format constants exactly
# (FloatToFloat8Bits's two-argument overload, kFloat8E4M3FNMax/
# kFloat8E5M2Max, kFloat8E4M3FNCanonicalNaN/kFloat8E5M2CanonicalNaN,
# Float8ElemType).
_FORMAT_PARAMS = {
    "e4m3": (3, 7, 448.0, 0x7F, onnx.TensorProto.FLOAT8E4M3FN),
    "e5m2": (2, 15, 57344.0, 0x7D, onnx.TensorProto.FLOAT8E5M2),
}


def test_quantize_fp8_round_to_nearest_is_half_step_bound():
    # n stands for "the number of grid steps the rounded value sits at" --
    # constrained only by *being nearest* to v / step (within 0.5, the same
    # abstract modeling of round() quantize_round_trip's own proof uses).
    # True regardless of tie-breaking rule -- a tie only ever occurs exactly
    # halfway, where both neighbors already satisfy the bound -- so this
    # single lemma covers quantize_fp8's ties-to-even just as well as it
    # covered quantize_fp16's/quantize_bf16's ties-away-from-zero; only the
    # differential tests below need to pin down which neighbor actually
    # wins. No zero_point term: float8's grid, like float16's/bfloat16's, is
    # symmetric about zero, unlike uniform-affine quantization's offset code.
    v, step = z3.Reals("v step")
    n = z3.Int("n")
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(step > 0, n - v / step <= half, v / step - n <= half)
    rounded = n * step
    error = v - rounded
    prove(z3.Implies(hypotheses, z3.And(error <= step / 2, -error <= step / 2)))


def test_quantize_fp8_worst_case_relative_bound_e4m3fn():
    # E4M3FN's own corollary of the general bound above: for a normalized
    # v > 0, the local grid spacing (ULP) is at most v * 2**-3 (3 stored
    # mantissa bits -- within [2**e, 2**(e+1)) the ULP is exactly 2**(e-3),
    # and v >= 2**e there, so ulp(v)/v <= 2**-3). Instantiating the general
    # lemma's `step` with that worst-case relative spacing gives a
    # reconstruction error of at most v * 2**-4 -- much coarser than
    # float16's v * 2**-11 or bfloat16's v * 2**-8, as expected for an 8-bit
    # format: the "roughly 1-2 decimal digits" folk description of float8
    # precision, now as a checked bound.
    v = z3.Real("v")
    n = z3.Int("n")
    half = z3.RealVal(1) / 2
    step = v * (z3.RealVal(1) / 8)  # v * 2**-3

    hypotheses = z3.And(v > 0, n - v / step <= half, v / step - n <= half)
    rounded = n * step
    error = v - rounded
    bound = v * (z3.RealVal(1) / 16)  # v * 2**-4
    prove(z3.Implies(hypotheses, z3.And(error <= bound, -error <= bound)))


def test_quantize_fp8_worst_case_relative_bound_e5m2():
    # E5M2's own corollary: only 2 stored mantissa bits, so the worst-case
    # local step (ULP) for a normalized v > 0 is v * 2**-2, giving a
    # reconstruction-error bound of v * 2**-3 -- coarser still than
    # E4M3FN's v * 2**-4 above, matching E5M2's own narrower mantissa (2
    # bits vs. 3) at the cost of E4M3FN's smaller max finite magnitude.
    v = z3.Real("v")
    n = z3.Int("n")
    half = z3.RealVal(1) / 2
    step = v * (z3.RealVal(1) / 4)  # v * 2**-2

    hypotheses = z3.And(v > 0, n - v / step <= half, v / step - n <= half)
    rounded = n * step
    error = v - rounded
    bound = v * (z3.RealVal(1) / 8)  # v * 2**-3
    prove(z3.Implies(hypotheses, z3.And(error <= bound, -error <= bound)))


@pytest.mark.parametrize(
    "max_mag", [448.0, 57344.0], ids=["e4m3fn-max-448", "e5m2-max-57344"]
)
def test_quantize_fp8_clamp_saturates_to_max_magnitude(max_mag):
    # This pass's own documented design choice (the header comment: both
    # formats "share one saturating-clamp code path" so E5M2's own real
    # infinity encoding is never actually emitted): a value beyond the
    # target format's largest finite magnitude clamps to *exactly* that
    # magnitude, not to something proportionally further off and not to
    # infinity. FloatToFloat8Bits clamps first and rounds second, so a value
    # just past the max cannot land one ULP beyond it either -- it becomes
    # the clamp bound itself, bit for bit. One parametrized proof covers
    # both formats' own max finite constants (448 for E4M3FN, 57344 for
    # E5M2) since the claim's shape is identical for either constant.
    v = z3.Real("v")
    max_mag_z3 = z3.RealVal(max_mag)
    clamped = z3.If(v > max_mag_z3, max_mag_z3, z3.If(v < -max_mag_z3, -max_mag_z3, v))
    prove(z3.Implies(v > max_mag_z3, clamped == max_mag_z3))
    prove(z3.Implies(v < -max_mag_z3, clamped == -max_mag_z3))


def test_quantize_fp8_negative_control_needs_nearest_hypothesis():
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


def _model(body, initializer=(), opset=19, ir_version=10):
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


def _two_branch_model(x_shape, w):
    # Two independent Identity branches -- one from the graph input, one
    # from the weight initializer -- in place of a single combining op like
    # Add/Concat: see this file's module docstring for why (no float8
    # arithmetic/data-movement kernel exists on CPUExecutionProvider in this
    # environment beyond Identity/CastLike). This still exercises real
    # per-element float8 rounding of both the weight (Y2) and, via the
    # boundary Cast, the activation (Y1).
    w = np.asarray(w, dtype=np.float32)
    return _model(
        f"""
        g (float[{x_shape}] X) => (float[{x_shape}] Y1, float[{w.size}] Y2)
        {{
          Y1 = Identity(X)
          Y2 = Identity(W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )


def _weight_initializer(model):
    # quantize_fp8 replaces a converted weight with a *new* initializer,
    # leaving the old float32 one orphaned in the model, so the initializer
    # actually feeding the W-branch Identity must be looked up by that
    # node's current input name, not by indexing graph.initializer blindly.
    identity_nodes = [n for n in model.graph.node if n.op_type == "Identity"]
    input_cast = next(
        n
        for n in model.graph.node
        if n.op_type == "Cast" and n.input[0] == model.graph.input[0].name
    )
    w_node = next(n for n in identity_nodes if n.input[0] != input_cast.output[0])
    return next(
        init for init in model.graph.initializer if init.name == w_node.input[0]
    )


def _float8_bits_hand_rolled(value, mantissa_bits, bias, max_value, nan_pattern):
    """Pure-Python port of FloatToFloat8Bits (quantize_fp8.h) -- used only
    when ml_dtypes isn't installed. Faithfully mirrors the C++ rounding
    (ties-to-even) and saturating-clamp logic, including the subnormal
    branch, so it also serves as an independent reference for the clamp
    tests below.
    """
    if math.isnan(value):
        return nan_pattern
    if value > max_value:
        value = max_value
    elif value < -max_value:
        value = -max_value

    (bits,) = struct.unpack("<I", struct.pack("<f", value))
    sign8 = (bits >> 24) & 0x80
    exp32 = (bits >> 23) & 0xFF
    mant32 = bits & 0x7FFFFF

    if exp32 == 0 and mant32 == 0:
        return sign8

    exp_narrow = exp32 - 127 + bias
    if exp_narrow <= 0:
        if exp_narrow < -mantissa_bits:
            return sign8
        m = mant32 | 0x800000
        shift = 24 - mantissa_bits - exp_narrow
        rounded_mant = m >> shift
        remainder = m & ((1 << shift) - 1)
        halfway = 1 << (shift - 1)
        if remainder > halfway or (remainder == halfway and (rounded_mant & 1)):
            rounded_mant += 1
        return sign8 | rounded_mant

    shift = 23 - mantissa_bits
    rounded_mant = mant32 >> shift
    remainder = mant32 & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and (rounded_mant & 1)):
        rounded_mant += 1
    if rounded_mant & (1 << mantissa_bits):
        rounded_mant = 0
        exp_narrow += 1
    return sign8 | (exp_narrow << mantissa_bits) | rounded_mant


def _float8_bits_to_float32(bits_u8, format_name):
    """Inverse of ``_float8_bits_reference``: decode raw float8 bit patterns
    back to float32, via ml_dtypes' own dtype (a ``.view`` reinterpret, then
    an ordinary upcast) when available, else a direct decode of the format's
    sign/exponent/mantissa layout (normalized and subnormal cases only --
    sufficient for every value this file's tests actually decode; none is a
    NaN or infinity).
    """
    bits_u8 = np.asarray(bits_u8, dtype=np.uint8)
    if _HAS_ML_DTYPES:
        dtype = (
            ml_dtypes.float8_e4m3fn if format_name == "e4m3" else ml_dtypes.float8_e5m2
        )
        return bits_u8.view(dtype).astype(np.float32)
    mantissa_bits, bias, _, _, _ = _FORMAT_PARAMS[format_name]

    def _decode_one(bits):
        sign = -1.0 if (bits & 0x80) else 1.0
        exp = (bits >> mantissa_bits) & ((1 << (7 - mantissa_bits)) - 1)
        mantissa = bits & ((1 << mantissa_bits) - 1)
        if exp == 0:
            return sign * (mantissa / (1 << mantissa_bits)) * 2.0 ** (1 - bias)
        return sign * (1.0 + mantissa / (1 << mantissa_bits)) * 2.0 ** (exp - bias)

    return np.array([_decode_one(int(b)) for b in bits_u8], dtype=np.float32)


def _float8_bits_reference(array, format_name):
    """Independent reference float8 conversion: ml_dtypes' own
    float8_e4m3fn/float8_e5m2 dtypes when available (confirmed, while
    writing this file, to round-to-nearest ties-to-even -- matching this
    pass's documented rule -- via the exact-tie construction the same way
    ``_float8_bits_hand_rolled`` is verified below), else the hand-rolled
    port above, vectorized with a Python loop (arrays here are always tiny).
    """
    array = np.asarray(array, dtype=np.float32)
    if _HAS_ML_DTYPES:
        dtype = (
            ml_dtypes.float8_e4m3fn if format_name == "e4m3" else ml_dtypes.float8_e5m2
        )
        return array.astype(dtype).view(np.uint8)
    mantissa_bits, bias, max_value, nan_pattern, _ = _FORMAT_PARAMS[format_name]
    return np.array(
        [
            _float8_bits_hand_rolled(
                float(v), mantissa_bits, bias, max_value, nan_pattern
            )
            for v in array
        ],
        dtype=np.uint8,
    )


def test_quantize_fp8_is_an_opt_in_pass():
    C = onnxsim.onnxsim_cpp2py_export
    assert "quantize_fp8" in C._list_other_optimizers()
    assert "quantize_fp8" not in C._list_optimizers()


def test_quantize_fp8_keep_io_types_rounds_weight_and_inserts_boundary_casts():
    # Non-tie weight values (arbitrary decimals essentially never land
    # exactly halfway between two representable E4M3FN values) -- checked
    # bit-for-bit against the independent reference above.
    w = np.array([0.1, -0.2, 12.5, -6.7], dtype=np.float32)
    model = _two_branch_model(4, w)

    quant = onnxsim.quantize_fp8(model)  # format="e4m3", keep_io_types=True
    onnx.checker.check_model(quant)

    # Boundary casts: one right after the input, one right before each
    # output; the two Identity nodes are the two independent branches (see
    # _two_branch_model's own comment for why Identity stands in for a
    # combining op like Add/Concat here).
    assert sorted(n.op_type for n in quant.graph.node) == [
        "Cast",
        "Cast",
        "Cast",
        "Identity",
        "Identity",
    ]
    input_cast = next(
        n
        for n in quant.graph.node
        if n.op_type == "Cast" and n.input[0] == quant.graph.input[0].name
    )
    to_attr = {a.name: a.i for a in input_cast.attribute}
    assert to_attr["to"] == onnx.TensorProto.FLOAT8E4M3FN
    output_casts = [
        n
        for n in quant.graph.node
        if n.op_type == "Cast" and n.output[0] in (o.name for o in quant.graph.output)
    ]
    assert len(output_casts) == 2
    for cast in output_casts:
        to_attr = {a.name: a.i for a in cast.attribute}
        assert to_attr["to"] == onnx.TensorProto.FLOAT

    # The graph's own declared I/O stays float32.
    assert quant.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    for out in quant.graph.output:
        assert out.type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    w_init = _weight_initializer(quant)
    assert w_init.data_type == onnx.TensorProto.FLOAT8E4M3FN
    w_bits = np.frombuffer(w_init.raw_data, dtype=np.uint8)
    np.testing.assert_array_equal(w_bits, _float8_bits_reference(w, "e4m3"))

    # Overall numeric output stays close to (not equal to) the original:
    # Y1 round-trips the activation through E4M3FN, Y2 round-trips the
    # weight -- neither is a no-op, but both stay within E4M3FN's own
    # worst-case relative error (~1/16, per this file's own precision lemma).
    x = np.array([1.0, -2.5, 100.0, 0.3], dtype=np.float32)
    orig_sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    quant_sess = ort.InferenceSession(
        quant.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    orig_y1, orig_y2 = orig_sess.run(None, {"X": x})
    quant_y1, quant_y2 = quant_sess.run(None, {"X": x})
    assert not np.array_equal(orig_y1, quant_y1)  # genuinely lossy, not a no-op
    assert not np.array_equal(orig_y2, quant_y2)
    np.testing.assert_allclose(quant_y1, orig_y1, rtol=6e-2, atol=2e-2)
    np.testing.assert_allclose(quant_y2, orig_y2, rtol=6e-2, atol=2e-2)


@pytest.mark.parametrize("format_name", ["e4m3", "e5m2"])
def test_quantize_fp8_ties_to_even(format_name):
    # A value hand-constructed to be an EXACT tie at each format's own
    # mantissa grid, the same construction quantize_fp16's/quantize_bf16's
    # own tie tests use at *their* mantissa widths: within [1, 2), the
    # representable grid is 1 + k / 2**mantissa_bits for integer k. Pick
    # k=2 (even -- the "round to even" candidate) and its neighbor k=3
    # (odd); the exact midpoint is a dyadic fraction, representable exactly
    # in float32 (and ordinary Python/double-precision arithmetic), so no
    # incidental rounding sneaks into building the test value itself.
    #
    # quantize_fp16/quantize_bf16 round this exact shape of tie AWAY from
    # zero (picking the odd, larger-magnitude k=3 neighbor); this pass's
    # documented rule is the OPPOSITE -- ties-to-EVEN -- so it must pick the
    # already-even k=2 neighbor instead. Getting this backwards (asserting
    # the away-from-zero answer) is the easy mistake this test guards
    # against.
    mantissa_bits = _FORMAT_PARAMS[format_name][0]
    denom = 2**mantissa_bits
    low = 1.0 + 2 / denom  # even mantissa (k=2) -- ties-to-even's answer
    high = 1.0 + 3 / denom  # odd mantissa (k=3) -- ties-away-from-zero's answer
    tie = (low + high) / 2.0
    assert tie == 1.0 + 2.5 / denom  # sanity: exact midpoint, no drift

    w = np.array([tie, -tie], dtype=np.float32)
    model = _two_branch_model(2, w)

    quant = onnxsim.quantize_fp8(model, format=format_name)
    onnx.checker.check_model(quant)

    w_init = _weight_initializer(quant)
    w_bits = np.frombuffer(w_init.raw_data, dtype=np.uint8)
    expected_bits = _float8_bits_reference(w, format_name)
    np.testing.assert_array_equal(w_bits, expected_bits)

    reconstructed = numpy_helper.to_array(w_init).astype(np.float32)
    assert reconstructed[0] == np.float32(low)
    assert reconstructed[1] == np.float32(-low)
    # Not the ties-away-from-zero answer -- confirms the tie was actually
    # resolved to the EVEN neighbor, not merely landing on the same value
    # either rule would produce.
    assert reconstructed[0] != np.float32(high)
    assert reconstructed[1] != np.float32(-high)


@pytest.mark.parametrize("format_name", ["e4m3", "e5m2"])
def test_quantize_fp8_clamps_out_of_range_weight(format_name):
    # A magnitude far beyond the target format's own max finite value must
    # clamp to exactly that value, never inf/nan -- read back from the raw
    # bytes directly, independent of numpy_helper.to_array's own decoding
    # path. For E5M2 specifically, this confirms the pass's saturating
    # clamp keeps it from ever emitting E5M2's own real infinity encoding
    # (unlike E4M3FN, E5M2 *has* one -- see the module docstring).
    max_value = _FORMAT_PARAMS[format_name][2]
    w = np.array([1.0e10, -1.0e10, 3.0], dtype=np.float32)
    model = _two_branch_model(3, w)

    quant = onnxsim.quantize_fp8(model, format=format_name)
    onnx.checker.check_model(quant)

    w_init = _weight_initializer(quant)
    w_reconstructed = numpy_helper.to_array(w_init).astype(np.float32)
    assert np.all(np.isfinite(w_reconstructed))
    assert w_reconstructed[0] == max_value
    assert w_reconstructed[1] == -max_value
    assert w_reconstructed[2] == 3.0  # unaffected, well within range

    x = np.zeros(3, dtype=np.float32)
    _, out = ort.InferenceSession(
        quant.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"X": x})
    assert np.all(np.isfinite(out))


def test_quantize_fp8_no_keep_io_types_redeclares_io_directly():
    w = np.array([0.1, -0.2, 12.5, -6.7], dtype=np.float32)
    model = _two_branch_model(4, w)

    quant = onnxsim.quantize_fp8(model, keep_io_types=False)
    onnx.checker.check_model(quant)

    # No boundary casts -- the graph's own I/O is redeclared FLOAT8E4M3FN
    # directly.
    assert sorted(n.op_type for n in quant.graph.node) == ["Identity", "Identity"]
    assert (
        quant.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT8E4M3FN
    )
    for out in quant.graph.output:
        assert out.type.tensor_type.elem_type == onnx.TensorProto.FLOAT8E4M3FN

    x = np.array([1.0, -2.5, 100.0, 0.3], dtype=np.float32)
    x_bits = _float8_bits_reference(x, "e4m3")
    x_ortvalue = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
        x_bits, onnx.TensorProto.FLOAT8E4M3FN
    )

    quant_sess = ort.InferenceSession(
        quant.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    y1_name, y2_name = (o.name for o in quant.graph.output)
    (y1_ortvalue, y2_ortvalue) = quant_sess.run_with_ort_values(
        [y1_name, y2_name], {"X": x_ortvalue}
    )
    # onnxruntime's OrtValue.numpy() hands back the raw uint8 bytes for a
    # float8 tensor (it has no native float8 numpy dtype to decode into,
    # same reason quantize_bf16's own file reads BFLOAT16 OrtValues via raw
    # bytes rather than a native array dtype) -- decode via the reference
    # above rather than assuming a float dtype comes back directly.
    y1_bits = y1_ortvalue.numpy()
    # Identity is a bit-for-bit pass-through, so the output bytes must
    # exactly equal the fed-in (already-quantized) input bytes.
    np.testing.assert_array_equal(y1_bits, x_bits)
    reconstructed_y1 = _float8_bits_to_float32(y1_bits, "e4m3")

    orig_sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    orig_y1, _ = orig_sess.run(None, {"X": x})
    np.testing.assert_allclose(reconstructed_y1, orig_y1, rtol=6e-2, atol=2e-2)


def test_quantize_fp8_e4m3_and_e5m2_are_distinguishable():
    # Minimal check that format selection actually reaches the pass and
    # actually changes its behavior: the same float32 value converts to a
    # different tensor element type AND a different bit pattern under each
    # format. 15.0 = 1.111(base2) * 2**3 needs all 3 of E4M3FN's mantissa
    # bits to represent exactly; E5M2's narrower 2-bit mantissa cannot
    # represent it exactly and must round it away (to 16.0), so the two
    # formats provably disagree on this value, not merely happening to
    # produce the same bits under a shared type tag.
    w = np.array([15.0], dtype=np.float32)

    model_e4m3 = _two_branch_model(1, w)
    quant_e4m3 = onnxsim.quantize_fp8(model_e4m3, format="e4m3")
    onnx.checker.check_model(quant_e4m3)
    init_e4m3 = _weight_initializer(quant_e4m3)

    model_e5m2 = _two_branch_model(1, w)
    quant_e5m2 = onnxsim.quantize_fp8(model_e5m2, format="e5m2")
    onnx.checker.check_model(quant_e5m2)
    init_e5m2 = _weight_initializer(quant_e5m2)

    assert init_e4m3.data_type == onnx.TensorProto.FLOAT8E4M3FN
    assert init_e5m2.data_type == onnx.TensorProto.FLOAT8E5M2
    assert init_e4m3.data_type != init_e5m2.data_type

    bits_e4m3 = np.frombuffer(init_e4m3.raw_data, dtype=np.uint8)
    bits_e5m2 = np.frombuffer(init_e5m2.raw_data, dtype=np.uint8)
    assert bits_e4m3[0] != bits_e5m2[0]

    value_e4m3 = numpy_helper.to_array(init_e4m3).astype(np.float32)[0]
    value_e5m2 = numpy_helper.to_array(init_e5m2).astype(np.float32)[0]
    assert value_e4m3 == 15.0  # exactly representable in E4M3FN
    assert value_e5m2 == 16.0  # E5M2's coarser 2-bit mantissa rounds it up
    assert value_e4m3 != value_e5m2
