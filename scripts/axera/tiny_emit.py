#!/usr/bin/env python3
"""Emit NPU mcode for tinygrad-traced graphs.

Pipeline: a tinygrad ``Tensor`` graph is traced to its UOp pattern,
matched by one of the ``trace_*`` functions below, and (for Neg so far)
emitted as an mcode stream assembled from a reference Pulsar2 build with
caller-chosen output scales (found by value as float32 words -- Neg
carries its output scale as four stride-7 copies).

``trace_neg``/``trace_add``/``trace_mul`` match single- or two-node UOp
patterns (unary negation lowers to ``MUL(x, CONST(-1.0))``; a genuine
two-tensor add or multiply to ``ADD(a, b)``/``MUL(a, b)`` with neither
operand a ``CONST`` -- the two ``Ops.MUL`` matchers are mutually
exclusive by that const check). ``trace_relu``/``trace_sigmoid`` match
deeper compound patterns tinygrad lowers these ops to, since neither has
a dedicated UOp: Relu is ``WHERE(CMPLT(CONST(0.0), x), x, CONST(0.0))``
(a compare-and-select, with an operand-*identity* check -- not just
shape/dtype equality -- that the ``x`` compared against 0 is the same
UOp used as the true branch), and Sigmoid is
``RECIPROCAL(ADD(CONST(1.0), EXP2(MUL(x, CONST(-log2(e))))))`` (the
standard ``e^-x = 2^(-x*log2(e))`` rewrite, checked by exact float
equality against ``-math.log2(math.e)``). Every pattern here was
confirmed by direct UOp inspection against a real tinygrad install, not
assumed from tinygrad's Python source -- these lowerings can and do
change across tinygrad versions.

The Add matcher and a corresponding Sub scale patcher can transplant the
three confirmed output-scale fields in existing templates, while requiring
the graph shape and input quantization to remain the same. Neg has its own
scale patcher; the other matchers still describe graphs without claiming
that their mcode can be synthesized. See each function's docstring for the
exact boundary.

``patch_site_a`` and ``patch_matmul_a_scale`` (2026-09) generalize the
site-A idea beyond Mul, to Gemm/Conv and MatMul's own operands -- the
first mcode *generation* work built directly on this project's later,
much larger decode effort (site A's cross-op generalization, and
MatMul's exact short/full-form switching rule). See
tests/test_axera_sitea_generator.py for what's verified (byte-exact
against real target builds, both directions of MatMul's same-form
case) and, just as importantly, what's refused and why: a form-crossing
MatMul ``A`` edit is not a local patch -- a controlled build pair
(same ``B`` seed, same table order) shows it reflows 201 bytes across
the stream, not just the quad's own footprint. That refusal is the
general shape of why most of this project's later shape-derived
decode findings (Gemm's ``K*M-1`` field, Conv's dilation/orientation
fields) are *understood* but not yet *generatable*: knowing a field's
value is not the same as knowing it's safe to write in place.

``patch_mul_scales`` extends the same reference-patch idea to two-input
Mul streams' input side (sites A/C/B -- see
tests/test_axera_mcode_reciprocal.py for the field map): given the
reference build's recorded scales and the target scales, it rewrites the
input reciprocal and requant slots by value, verifying each family forms
its exact stride run. ``patch_mul_output_quad`` does the same for the
output-side scale quad (``03 <f32(z_scale)> 81 <tag2>`` x4 stride 7--
``tag2`` varies build to build, unlike the always-``81`` byte before it;
not patched, not relied on). ``patch_mul_zp_x`` patches x's zero point,
but *only* when the reference build happens to use the one zero-point
form this project has actually decoded (see below) -- it raises rather
than silently doing nothing when it doesn't apply, since whether it
applies is not predictable in advance. ``patch_conv_zp_x`` is the same
patch for Conv (the literal unit was confirmed identical across ops);
see its own docstring for a caveat ``patch_mul_zp_x`` does not carry --
patching this unit alone was checked against a real second Conv build
with x_scale held bit-identical and did *not* reproduce that build's
mcode elsewhere in the stream (~20 bytes beyond the unit and the
shape's own noise floor also move with zp_x), unlike the scale-family
patches below, which a real hardware test confirmed reproduce a
rebuild bit-exactly.

What none of this touches: the S-unit programs themselves (magnitude-
adaptive shape, unmodeled ISA), the manifest string table (tensor-name
order varies build to build), y's zero point (no literal encoding found
for it at all, decoded or not), z's own zero point (see the hardware
result below -- it did not need patching, at least once), or x's zero
point when the reference build doesn't use the literal form -- which is
the common case, not an edge case (confirmed non-predictable from zp_x's
value at any magnitude, see ``TestZpXLiteralByteWhenPresent`` in the
test file). Because of that last point, and because z_scale changing
generally moves far more of the stream than the five families patched
here (A, B, C, the output quad, zp_x -- z's own calibration range shifts
the S-unit programs' internal
constants throughout, not just the named slots -- confirmed by diffing
same-shape builds at different z scales: over a thousand bytes move),
**full-stream equality after patching is not a goal and should not be
expected**; what is verified is that each named family lands on the
target build's own bytes for that family.

Status (2026-09-16): scale, output-quad and (when present) zp_x words
are mapped, verified offline (round-trip exactly, pass ``mcode.check``),
and confirmed twice on real AX650N hardware
(tests/test_axera_mul_emit_hardware.py): patching a reference build's
scales + output quad + zp_x (17 bytes total, in that one build pair)
reproduced a real rebuild's device output *bit-exactly*, not just within
quantization noise, and neither zp_y nor zp_z needed touching for that
result to hold. This supersedes an earlier note here about the emitted
stream running ~0.19-vs-0.006 against ORT -- that check pre-dated the
output-quad and zp_x work and used a different code path (``emit_neg``,
not the Mul patch functions); it has not been repeated against these.

**Practical recommendation, also hardware-confirmed**: if the calibration
data can be chosen (as opposed to given), keep it non-negative for both
inputs -- MinMax clipping then guarantees zp_x = zp_y = 0 (see
``TestSiteBFormSelectorIsZpX`` in the test file), which sidesteps
``patch_mul_zp_x`` and its "does this build even use the literal form"
uncertainty entirely: only ``patch_mul_scales`` + ``patch_mul_output_quad``
are needed, and that combination also reproduces a real rebuild
bit-exactly (second test in the hardware file). Two data points, not a
general proof -- see the hardware test file's own docstring for exactly
what is and isn't covered. tinygrad itself is an optional, lazily-imported
dependency. ``emit_add`` also loads ``mcode.py`` lazily for instruction
boundary decoding, so that operation also requires ``onnx`` when called;
``minmax_scale`` lazily uses NumPy.
"""

from __future__ import annotations

import math
import struct


def trace_neg(tensor):
    """Match a tinygrad tensor holding unary negation.

    Returns ``{"shape": [...], "dtype": ...}`` if the tensor's UOp graph
    is ``MUL(x, const(-1.0))`` (what ``-t`` lowers to), else raises
    ``ValueError``. tinygrad is imported lazily so this module stays
    importable without it.
    """
    try:
        from tinygrad.uop.ops import Ops
    except ImportError as exc:
        raise ImportError(
            "tracing needs the tinygrad package (pip install tinygrad)"
        ) from exc

    uop = tensor.uop
    if uop.op is not Ops.MUL or len(uop.src) != 2:
        raise ValueError(f"not a multiply: {uop.op}")
    data, const = uop.src
    if const.op is not Ops.CONST or float(const.arg) != -1.0:
        raise ValueError(f"not a multiply-by-minus-one: {const}")
    shape = list(data.shape)
    return {"shape": shape, "dtype": str(data.dtype)}


def trace_add(tensor):
    """Match a tinygrad tensor holding elementwise two-tensor addition.

    Returns ``{"shape": [...], "dtype": ...}`` if the tensor's UOp graph
    is ``ADD(a, b)`` with neither operand a ``CONST`` (a genuine
    two-tensor add, not a scalar-add lowering, which would need its own
    matcher the way ``trace_neg`` needs one distinct from a general
    two-tensor multiply) -- else raises ``ValueError``.
    """
    try:
        from tinygrad.uop.ops import Ops
    except ImportError as exc:
        raise ImportError(
            "tracing needs the tinygrad package (pip install tinygrad)"
        ) from exc

    uop = tensor.uop
    if uop.op is not Ops.ADD or len(uop.src) != 2:
        raise ValueError(f"not an add: {uop.op}")
    a, b = uop.src
    if a.op is Ops.CONST or b.op is Ops.CONST:
        raise ValueError("a scalar add, not a two-tensor add")
    return {"shape": list(uop.shape), "dtype": str(tensor.dtype)}


def trace_mul(tensor):
    """Match a tinygrad tensor holding elementwise two-tensor multiply.

    Returns ``{"shape": [...], "dtype": ...}`` if the tensor's UOp graph
    is ``MUL(a, b)`` with neither operand a ``CONST``. This is the
    complement of ``trace_neg``'s pattern (``MUL(x, CONST(-1.0))``) over
    the same ``Ops.MUL`` space -- a scalar multiply-by-constant (Neg's
    shape, or any other scalar multiply) is rejected here and matched by
    ``trace_neg`` instead when the constant happens to be -1.0; any other
    scalar multiply matches neither and needs its own matcher, not added
    here since this project has no confirmed byte-level scalar-multiply
    encoding yet.
    """
    try:
        from tinygrad.uop.ops import Ops
    except ImportError as exc:
        raise ImportError(
            "tracing needs the tinygrad package (pip install tinygrad)"
        ) from exc

    uop = tensor.uop
    if uop.op is not Ops.MUL or len(uop.src) != 2:
        raise ValueError(f"not a multiply: {uop.op}")
    a, b = uop.src
    if a.op is Ops.CONST or b.op is Ops.CONST:
        raise ValueError("a scalar multiply, not a two-tensor multiply")
    return {"shape": list(uop.shape), "dtype": str(tensor.dtype)}


def trace_relu(tensor):
    """Match a tinygrad tensor holding ReLU.

    tinygrad has no dedicated Relu UOp -- ``.relu()`` lowers to a
    compare-and-select, ``WHERE(CMPLT(CONST(0.0), x), x, CONST(0.0))``,
    confirmed by direct UOp inspection (not assumed from tinygrad's
    Python source, which can and does change the lowering across
    versions). Matching it means checking that three-node shape *and*
    that the ``x`` referenced in the comparison is the identical UOp
    object used as the true-branch -- tinygrad's UOp graph shares
    subexpression nodes, so ``is`` identity is the correct check, not
    structural equality (two different tensors could legitimately have
    identical shape/dtype without being the same operand).
    """
    try:
        from tinygrad.uop.ops import Ops
    except ImportError as exc:
        raise ImportError(
            "tracing needs the tinygrad package (pip install tinygrad)"
        ) from exc

    uop = tensor.uop
    if uop.op is not Ops.WHERE or len(uop.src) != 3:
        raise ValueError(f"not a where: {uop.op}")
    cmp, true_branch, false_branch = uop.src
    if cmp.op is not Ops.CMPLT or len(cmp.src) != 2:
        raise ValueError(f"not a relu (where's condition isn't a cmplt): {cmp.op}")
    zero_lhs, x = cmp.src
    if zero_lhs.op is not Ops.CONST or float(zero_lhs.arg) != 0.0:
        raise ValueError("not a relu (cmplt's left side isn't the constant 0.0)")
    if true_branch is not x:
        raise ValueError("not a relu (where's true branch isn't cmplt's operand)")
    if false_branch.op is not Ops.CONST or float(false_branch.arg) != 0.0:
        raise ValueError("not a relu (where's false branch isn't the constant 0.0)")
    return {"shape": list(uop.shape), "dtype": str(tensor.dtype)}


def trace_sigmoid(tensor):
    """Match a tinygrad tensor holding Sigmoid.

    Like Relu, tinygrad has no dedicated Sigmoid UOp: ``.sigmoid()``
    lowers to ``RECIPROCAL(ADD(CONST(1.0), EXP2(MUL(x, CONST(c)))))``
    where ``c`` is exactly ``-log2(e)`` (the standard
    ``e^-x = 2^(-x*log2(e))`` rewrite so the hardware/software exp
    lowers through ``EXP2`` rather than a natural-base ``Ops.EXP``),
    confirmed by direct UOp inspection and an exact float equality check
    against ``-math.log2(math.e)`` -- not a tolerance-based comparison,
    since tinygrad computes this constant once at trace time and every
    build should reproduce the identical float64-rounded-to-float32 bit
    pattern. The deepest pattern of the four traced here (four nested
    ops); unlike Relu's compare-and-select, no operand identity check is
    needed beyond following the chain down to a single ``x``.
    """
    try:
        from tinygrad.uop.ops import Ops
    except ImportError as exc:
        raise ImportError(
            "tracing needs the tinygrad package (pip install tinygrad)"
        ) from exc
    import math

    uop = tensor.uop
    if uop.op is not Ops.RECIPROCAL or len(uop.src) != 1:
        raise ValueError(f"not a reciprocal: {uop.op}")
    (add_node,) = uop.src
    if add_node.op is not Ops.ADD or len(add_node.src) != 2:
        raise ValueError(
            f"not a sigmoid (reciprocal's operand isn't an add): {add_node.op}"
        )
    one_const, exp2_node = add_node.src
    if one_const.op is not Ops.CONST or float(one_const.arg) != 1.0:
        raise ValueError("not a sigmoid (add's first operand isn't the constant 1.0)")
    if exp2_node.op is not Ops.EXP2 or len(exp2_node.src) != 1:
        raise ValueError(
            f"not a sigmoid (add's second operand isn't an exp2): {exp2_node.op}"
        )
    (mul_node,) = exp2_node.src
    if mul_node.op is not Ops.MUL or len(mul_node.src) != 2:
        raise ValueError(
            f"not a sigmoid (exp2's operand isn't a multiply): {mul_node.op}"
        )
    x, c = mul_node.src
    if c.op is not Ops.CONST or float(c.arg) != -math.log2(math.e):
        raise ValueError("not a sigmoid (multiply's constant isn't -log2(e))")
    return {"shape": list(uop.shape), "dtype": str(tensor.dtype)}


def _find_all(mcode: bytes, pattern: bytes) -> list[int]:
    """Return every possibly overlapping occurrence of ``pattern``.

    ``bytes.find`` performs the scan in C and avoids allocating a short slice
    at every candidate offset. Advance by one byte after a hit to preserve the
    overlap behavior of the former offset-by-offset comparisons.
    """
    if not pattern:
        raise ValueError("cannot search for an empty pattern")
    hits = []
    start = 0
    while (found := mcode.find(pattern, start)) >= 0:
        hits.append(found)
        start = found + 1
    return hits


def find_scale_words(mcode: bytes, scale: float) -> list:
    """Offsets of every float32 occurrence of ``scale`` in the stream.

    Neg carries its output scale as four stride-7 copies; this returns
    wherever the value literally occurs so the caller can decide which
    copies are output-side. No layout assumptions beyond the byte match.
    """
    if isinstance(mcode, bytearray):
        mcode = bytes(mcode)
    pat = struct.pack("<f", float(scale))
    return _find_all(mcode, pat)


def minmax_scale(samples) -> float:
    """Pulsar2's output-scale formula, verified to ~1e-10 against six
    real Neg builds: ``(max - min) / 255`` over the calibration samples
    (computed in float64). The zero point formula is still open (floor
    fits 4/9 builds), so this returns the scale only."""
    import numpy as np

    flat = np.concatenate([np.asarray(s).reshape(-1) for s in samples]).astype(
        np.float64
    )
    return float((flat.max() - flat.min()) / 255.0)


def emit_neg(reference_mcode: bytes, old_scale: float, new_scale: float) -> bytes:
    """Replace every float32 occurrence of ``old_scale`` with ``new_scale``.

    Both ends must be exactly representable (pass ``float(np.float32(x))``
    -- a float64 with extra digits never matches). Returns bytes that
    decode and round-trip exactly like the reference; whether they
    *compute* identically is a device question (see module docstring).
    """
    import struct as _struct

    new = _struct.pack("<f", float(new_scale))
    offsets = find_scale_words(reference_mcode, float(old_scale))
    if not offsets:
        raise ValueError(f"scale {old_scale!r} occurs nowhere: wrong reference?")
    out = bytearray(reference_mcode)
    for off in offsets:
        out[off : off + 4] = new
    return bytes(out)


def _patch_additive_output_scale(
    reference_mcode: bytes, old_y_scale: float, new_y_scale: float, op_name: str
) -> bytes:
    from mcode import FULL_RULE, decode, stream_bounds

    if not all(math.isfinite(s) and s > 0.0 for s in (old_y_scale, new_y_scale)):
        raise ValueError(f"{op_name} output scales must be finite and positive")
    old_word = struct.pack("<f", float(old_y_scale))
    new_word = struct.pack("<f", float(new_y_scale))
    lo, hi = stream_bounds(reference_mcode)
    records = decode(reference_mcode, start=lo, end=hi, **FULL_RULE)
    fields = [
        record
        for record in records
        if record.get("kind") == "V"
        and record.get("verb") == 0xA1
        and record.get("bank") == 15
        and record.get("field") in (96, 112, 128)
        and record.get("operand") == old_word
    ]
    if sorted(record["field"] for record in fields) != [96, 112, 128]:
        raise ValueError(
            f"{op_name} mcode must contain exactly one output-scale operand"
            " at fields 96, 112 and 128"
        )

    out = bytearray(reference_mcode)
    for record in fields:
        at = record["at"] + 4
        out[at : at + 4] = new_word
    return bytes(out)


def emit_add(reference_mcode: bytes, old_y_scale: float, new_y_scale: float) -> bytes:
    """Patch Add's output scale in an existing two-input Add mcode template.

    The confirmed Add encoding carries its direct output scale at the three
    ``a1``/bank-15 fields 96, 112 and 128. Those field IDs also carry other
    scale families, so the patch selects by both field and exact old float32
    operand. The reference must already have the target's shape, operand
    order, input scales and zero points. Output zero point can be patched
    separately with ``patch_add_output_zero_point``; this does not synthesize
    Add instructions or update its input-side quantization fields.
    """
    return _patch_additive_output_scale(
        reference_mcode, old_y_scale, new_y_scale, "Add"
    )


def patch_sub_output_scale(
    reference_mcode: bytes, old_y_scale: float, new_y_scale: float
) -> bytes:
    """Patch Sub's output scale in an isolated two-input Sub template.

    The tested Sub encoding shares Add's ``a1``/bank-15 output fields 96, 112
    and 128. Since those IDs also carry an input reciprocal, the exact old
    float32 operand is part of the match; a missing or ambiguous set is
    refused. The reference's shape and input quantization must stay fixed.
    Fixture evidence covers one shape, and this field patch has not been
    confirmed against a changed-calibration hardware rebuild.
    """
    return _patch_additive_output_scale(
        reference_mcode, old_y_scale, new_y_scale, "Sub"
    )


def _patch_elementwise_output_zero_point(
    reference_mcode: bytes, old_zp_y: int, new_zp_y: int, op_name: str
) -> bytes:
    from mcode import FULL_RULE, decode, stream_bounds

    if any(
        not isinstance(value, int) or not 0 <= value <= 255
        for value in (old_zp_y, new_zp_y)
    ):
        raise ValueError(
            f"{op_name} output zero points must be integers in [0, 255]"
        )
    if old_zp_y == 4:
        raise ValueError(f"old_zp_y=4 is ambiguous with {op_name}'s fixed payload")

    lo, hi = stream_bounds(reference_mcode)
    records = decode(reference_mcode, start=lo, end=hi, **FULL_RULE)
    locator = [
        record
        for record in records
        if record.get("kind") == "S"
        and record.get("reg") == 14
        and record.get("tag") == 131
    ]
    matches = [record for record in locator if record["payload"][-1] == old_zp_y]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {op_name} zp_y match at reg=14/tag=131, found {len(matches)}"
        )

    record = matches[0]
    out = bytearray(reference_mcode)
    at = record["at"] + record["p"] + 1
    out[at] = new_zp_y
    return bytes(out)


def patch_add_output_zero_point(
    reference_mcode: bytes, old_zp_y: int, new_zp_y: int
) -> bytes:
    """Patch Add's decoded output zero point in an existing mcode template.

    Add stores ``zp_y`` as the last payload byte of an S record at
    ``reg=14, tag=131``. Another record at that locator has a fixed payload
    ending in 4, so the old value must identify exactly one record; in
    particular, ``old_zp_y == 4`` is intentionally refused. This changes only
    the output zero point, preserving the template's shape, operand order and
    input quantization. It does not synthesize Add instructions or establish
    that this isolated field change matches a hardware rebuild.
    """
    return _patch_elementwise_output_zero_point(
        reference_mcode, old_zp_y, new_zp_y, "Add"
    )


def patch_sub_output_zero_point(
    reference_mcode: bytes, old_zp_y: int, new_zp_y: int
) -> bytes:
    """Patch Sub's output zero point at its decoded ``reg=14, tag=131`` field.

    Sub shares Add's locator and unrelated fixed-payload collision at 4.
    Only a unique old-value match is changed; shape and input quantization
    must stay fixed. This is an offline field patch, with no changed-scale
    hardware validation.
    """
    return _patch_elementwise_output_zero_point(
        reference_mcode, old_zp_y, new_zp_y, "Sub"
    )


def patch_matmul_gemm_output_zero_point(
    reference_mcode: bytes, old_zp_y: int, new_zp_y: int
) -> bytes:
    """Patch the output zero point in an isolated Gemm/MatMul template.

    Tested Gemm and MatMul builds store this byte as the last payload byte of
    an S record at ``reg=120, tag=132``. The template must contain exactly one
    such record whose payload ends in ``old_zp_y``; absent or ambiguous cases
    are refused. In particular, builds with ``zp_y == 128`` omit this record,
    so changing to or from that wire form requires a rebuild. Keep this helper
    scoped to an isolated op stream: a full training graph can contain several
    MatMuls, and this locator does not identify which node a record belongs to.

    This patches one decoded field only. The evidence covers Gemm and MatMul
    templates, not Conv, and does not establish bit-exact equivalence to a
    Pulsar2 rebuild for changed calibration data.
    """
    from mcode import FULL_RULE, decode, stream_bounds

    if any(
        not isinstance(value, int) or not 0 <= value <= 255
        for value in (old_zp_y, new_zp_y)
    ):
        raise ValueError("output zero points must be integers in [0, 255]")
    if old_zp_y == 128 or new_zp_y == 128:
        raise ValueError(
            "Gemm/MatMul zp_y=128 uses a different form and cannot be patched in place"
        )

    lo, hi = stream_bounds(reference_mcode)
    records = decode(reference_mcode, start=lo, end=hi, **FULL_RULE)
    matches = [
        record
        for record in records
        if record.get("kind") == "S"
        and record.get("reg") == 120
        and record.get("tag") == 132
        and record["payload"][-1] == old_zp_y
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one Gemm/MatMul zp_y at reg=120/tag=132, found {len(matches)}"
        )

    record = matches[0]
    out = bytearray(reference_mcode)
    at = record["at"] + record["p"] + 1
    out[at] = new_zp_y
    return bytes(out)


def _strided_run(mcode: bytes, pattern: bytes, stride: int, count: int = 4) -> list:
    """Offsets where ``pattern`` occurs as exactly ``count`` stride-run copies.

    Raises ``ValueError`` unless the occurrences are exactly ``count`` and
    land on a perfect stride grid -- an incidental byte collision anywhere
    else in the stream fails loudly instead of patching half a slot family.
    """
    hits = _find_all(mcode, pattern)
    if len(hits) != count or any(b - a != stride for a, b in zip(hits, hits[1:])):
        raise ValueError(
            f"pattern {pattern.hex()} hits {hits}: not a stride-{stride} x{count} run"
        )
    return hits


def patch_mul_scales(reference_mcode: bytes, old_scales, new_scales) -> bytes:
    """Rewrite a two-input Mul stream's input-side scale slots by value.

    ``old_scales``/``new_scales`` are ``(x_scale, y_scale, z_scale)``
    triples -- the reference build's recorded MinMax scales and the
    target's. Three slot families move (see
    tests/test_axera_mcode_reciprocal.py):

    - site A (stride-8 x4): float32(1/x_scale), the x quant multiplier;
    - site C (stride-8 x4): float32(z_scale/(x_scale*y_scale)), the
      integer requant multiplier;
    - site B: float32(1/y_scale) x4 at stride 7 when the reference uses
      the full S-unit form, else the short form's low 3 bytes x4 at
      stride 6 (tag byte preserved). The reference's form is kept: this
      patches values, it does not recompile programs.

    Every family is stride-verified; a missing or ambiguous family
    raises. Output quads, S-unit programs, the string table and zero
    points are untouched (see module docstring).
    """
    old_x, old_y, old_z = (float(s) for s in old_scales)
    new_x, new_y, new_z = (float(s) for s in new_scales)
    # Locate every family on the pristine reference first: patching one
    # family must never disturb another family's search (overlapping
    # values across families would otherwise corrupt the later lookup).
    edits = []

    def locate(old_value: float, new_value: float, stride: int, width: int = 4):
        old_pat = struct.pack("<f", old_value)[:width]
        new_pat = struct.pack("<f", new_value)[:width]
        for off in _strided_run(reference_mcode, old_pat, stride):
            edits.append((off, new_pat))

    locate(1.0 / old_x, 1.0 / new_x, 8)
    locate(old_z / (old_x * old_y), new_z / (new_x * new_y), 8)
    try:
        locate(1.0 / old_y, 1.0 / new_y, 7)
    except ValueError:
        locate(1.0 / old_y, 1.0 / new_y, 6, width=3)
    out = bytearray(reference_mcode)
    for off, new_pat in edits:
        out[off : off + len(new_pat)] = new_pat
    return bytes(out)


def patch_mul_output_quad(
    reference_mcode: bytes, old_z_scale: float, new_z_scale: float
) -> bytes:
    """Rewrite a Mul stream's output scale quads by value.

    The output-side counterpart to ``patch_mul_scales``'s input-side
    families: four copies of ``03 <f32(z_scale)> 81 <tag2>`` at stride 7
    (see ``TestOutputScaleQuads`` in tests/test_axera_mcode_reciprocal.py).
    Verifies the ``03``/``81`` framing on every copy before patching, on
    top of ``_strided_run``'s count/stride check, since those two bytes
    are cheap to confirm and a false match here would silently leave the
    output at the old scale. The second tag byte is NOT checked: it read
    ``82`` on every fixture ``TestOutputScaleQuads`` was written against,
    which that test's docstring stated as if constant, but two more real
    builds (zp_x=33/35, output zp 30/32) came back ``80`` instead -- only
    the *value* (this function's job) and the ``03``/``81`` framing hold
    across all six builds gathered so far. This function preserves
    whatever the second tag byte already is; it never depends on its
    value.
    """
    old_pat = struct.pack("<f", float(old_z_scale))
    new_pat = struct.pack("<f", float(new_z_scale))
    hits = _strided_run(reference_mcode, old_pat, 7)
    for off in hits:
        lead, tag1 = reference_mcode[off - 1], reference_mcode[off + 4]
        if lead != 0x03 or tag1 != 0x81:
            raise ValueError(
                f"quad @{off}: frame {lead:02x}/{tag1:02x} is not 03../81."
            )
    out = bytearray(reference_mcode)
    for off in hits:
        out[off : off + 4] = new_pat
    return bytes(out)


def patch_output_quad(
    reference_mcode: bytes, old_z_scale: float, new_z_scale: float
) -> bytes:
    """Rewrite the output-scale quad for Gemm/Conv/MatMul (and Mul).

    ``patch_mul_output_quad`` only recognizes Mul's own bare frame (``03
    <f32(z_scale)> 81 <tag2>`` x4 stride 7). This session's later decode
    work found the *identical* mechanism on Gemm, Conv and MatMul too
    (``tests/test_axera_output_scale_quad_generalizes.py``,
    ``tests/test_axera_gemm_output_quad.py``), framed with a `05 50 0f`
    lead-in before the first copy instead: ``05 50 0f <f32(z_scale)> 81
    <tag2> 03`` x4 stride 7. This function handles both frames by
    checking for the lead-in first and falling back to Mul's bare-``03``
    frame when it's absent, so one function covers every op this project
    has found the quad on.

    Same defensive posture as ``patch_mul_output_quad``: verifies the
    stride-7 x4 run and the ``0x81`` tail byte (and the lead-in, when
    present) before patching, but never depends on ``tag2`` -- confirmed
    build-specific across every op checked, not a shared constant.
    Preserves whatever ``tag2`` and the trailing 0x03/0x83 high-bit byte
    already are; this only ever rewrites the 4-byte float.

    Verified (2026-09-17), compile-only, no device access needed:
    patching `gemm_1x8x8_tb0.mcode.gz`'s quad to `gemm_1x8x8_tb1.mcode.gz`'s
    real output_scale reproduces `tb1`'s own mcode byte-for-byte outside
    this project's already-known ``~295-330`` noise zone -- see
    ``tests/test_axera_output_quad_generator.py``.
    """
    old_pat = struct.pack("<f", float(old_z_scale))
    new_pat = struct.pack("<f", float(new_z_scale))
    hits = _strided_run(reference_mcode, old_pat, 7)
    lead_in = reference_mcode[hits[0] - 3 : hits[0]]
    has_lead_in = lead_in == bytes.fromhex("05500f")
    if not has_lead_in and reference_mcode[hits[0] - 1] != 0x03:
        raise ValueError(
            f"quad @{hits[0]}: lead-in {lead_in.hex()} is neither the"
            " generalized 05500f frame nor Mul's bare 03 frame"
        )
    for off in hits:
        if reference_mcode[off + 4] != 0x81:
            raise ValueError(
                f"quad @{off}: tail byte {reference_mcode[off + 4]:02x} is not 81"
            )
    out = bytearray(reference_mcode)
    for off in hits:
        out[off : off + 4] = new_pat
    return bytes(out)


def patch_site_a(
    reference_mcode: bytes, old_x_scale: float, new_x_scale: float
) -> bytes:
    """Rewrite site A's full-form quad by value -- the generalized-op
    counterpart of ``patch_mul_scales``'s own site-A edit.

    Site A (``<f32(1/x_scale)> a1 00 <id>`` x4, stride 8) was confirmed
    full-form-only for Gemm and Conv, and for MatMul's ``B`` operand, in
    ``tests/test_axera_site_a_generalizes.py`` (merged). This function is
    literally the same value-only, framing-agnostic edit
    ``patch_mul_scales``'s own ``locate(1/old_x, 1/new_x, 8)`` call
    already performs for Mul's site A -- that call never checked the
    ``a1 00 <id>`` suffix either, so no new framing logic is needed here,
    only a standalone entry point for ops that have no output-side
    scale/y-operand to patch alongside it.

    Does not apply to MatMul's ``A`` operand -- see ``patch_matmul_a_scale``
    for why that one needs its own function.
    """
    old_pat = struct.pack("<f", float(1.0 / old_x_scale))
    new_pat = struct.pack("<f", float(1.0 / new_x_scale))
    hits = _strided_run(reference_mcode, old_pat, 8)
    out = bytearray(reference_mcode)
    for off in hits:
        out[off : off + 4] = new_pat
    return bytes(out)


def patch_reshape_gather_scales(
    reference_mcode: bytes,
    old_x_scale: float,
    new_x_scale: float,
    old_z_scale: float,
    new_z_scale: float,
) -> bytes:
    """Patch the decoded scale fields in a compiled Reshape+Gather stream.

    The verified ``Reshape(X[8]->[2,4])+Gather(axis=1, indices=[0,2])``
    reference contains site A's reciprocal input scale and the generalized
    output-scale quad. This applies those two already-validated edits in
    sequence. Reshape and Gather themselves are still carried by the
    reference program: this does not synthesize their instructions or change
    tensor shapes, indices, zero points, or allocation decisions.

    Both scale fields are independently located as strict stride runs, so a
    stream with a missing or ambiguous field fails instead of silently
    emitting a plausible-looking result. The decode evidence is recorded in
    ``tests/test_axera_reshape_gather_bwd_decode.py``; the underlying patch
    primitives have fixture-level checks in ``tests/test_axera_tiny_emit.py``.

    Reshape and Gather are data-movement ops in the confirmed graph, so they
    preserve the quantization scale. Require the reference and requested
    input/output scales to have identical float32 encodings; allowing them to
    differ would make this patcher emit a byte-valid stream whose output
    quantization metadata contradicts the bytes it moves.
    """
    scales = {
        "old_x_scale": float(old_x_scale),
        "old_z_scale": float(old_z_scale),
        "new_x_scale": float(new_x_scale),
        "new_z_scale": float(new_z_scale),
    }
    words = {}
    for name, value in scales.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive: {value!r}")
        words[name] = struct.pack("<f", value)
    if words["old_x_scale"] != words["old_z_scale"]:
        raise ValueError("reference Reshape+Gather input/output scales must match")
    if words["new_x_scale"] != words["new_z_scale"]:
        raise ValueError("target Reshape+Gather input/output scales must match")

    out = patch_site_a(reference_mcode, old_x_scale, new_x_scale)
    return patch_output_quad(out, old_z_scale, new_z_scale)


def matmul_a_quad_form(a_scale: float) -> str:
    """Which encoding MatMul's rank-3 batched ``A`` quad uses for this
    scale, per the closed-form rule in
    ``tests/test_axera_matmul_quad_form_switch.py`` (merged): the short
    3-byte-truncated form is used exactly when ``1/a_scale``'s float32
    representation has high byte ``0x42`` (the value lies in
    ``[32, 128)``); any other high byte forces the full 4-byte form.
    Returns ``"short"`` or ``"full"``. This is the general IEEE754 fact
    (one high byte covers two consecutive power-of-two exponent groups),
    but the *codec's* choice of ``0x42`` as its one implied-default byte
    was only ever confirmed for values landing in that specific
    ``[32,128)`` window -- other high bytes are assumed (not verified
    against real hardware here) to all fall back to the full form, since
    that is the only fallback this project has ever observed.
    """
    high_byte = struct.pack("<f", float(1.0 / a_scale))[3]
    return "short" if high_byte == 0x42 else "full"


def patch_matmul_a_scale(
    reference_mcode: bytes, old_a_scale: float, new_a_scale: float
) -> bytes:
    """Rewrite MatMul's rank-3 batched ``A`` quad by value -- but only
    within one encoding form; refuses a form-crossing edit outright.

    Same-form edits (old and new scale's ``1/scale`` share a float32 high
    byte, so ``matmul_a_quad_form`` agrees on both) are a plain value
    patch, mirroring ``patch_site_a``: short form is ``<3 bytes,
    low(1/a_scale)> 82 <var> <02|83>`` x4 at stride 6 (the last copy's
    tail byte is ``0x83`` instead of ``0x02`` -- both are accepted and
    each preserved as-is, since the tag/var bytes are not this function's
    job); full form is ``<f32(1/a_scale)> 81 <var>`` x4 at stride 7.

    A form-crossing edit -- e.g. patching a short-form reference to a
    target scale whose ``1/scale`` needs the full form, or vice versa --
    is refused with ``ValueError`` rather than attempted. This was tested
    directly, not assumed: a controlled pair (identical ``B`` calibration
    seed, identical mcode length 3336 bytes both ways, identical
    ``A_offset``/``B_offset`` table order -- ruling out both of this
    project's two known confounds, per
    ``tests/test_axera_matmul_offset_table_coinflip.py`` -- differing
    *only* in whether ``A``'s scale crosses the ``0x42``/``0x43``
    boundary) shows 201 bytes differ, spanning offset 785 to 2896, not
    just the quad's own ~28-36 byte footprint. One concrete piece of that
    footprint: a little-endian u32 count/length field at offset 2896
    reads 1060 in the short-form build and 1064 in the full-form build --
    exactly the difference between the short form's stride-6 x4 = 24-byte
    footprint and the full form's stride-7 x4 = 28-byte footprint. The
    extra 4 bytes the full form needs are drawn from elsewhere in the
    stream (that count field, plus a real content change at offset
    785-802), not local padding -- so despite the *total* stream length
    coincidentally staying equal in this pair, the edit is not a local
    swap and this function does not attempt it.
    """
    old_form = matmul_a_quad_form(old_a_scale)
    new_form = matmul_a_quad_form(new_a_scale)
    if old_form != new_form:
        raise ValueError(
            f"cannot patch MatMul A's quad from {old_a_scale!r} ({old_form} form)"
            f" to {new_a_scale!r} ({new_form} form): a form-crossing edit changes"
            " the quad's own byte width (24 vs 28 bytes total) and reflows at"
            " least 201 bytes elsewhere in the stream (offsets 785-2896 in a"
            " controlled test pair) -- not a local patch. Choose a target scale"
            " on the same side of the 0x42/0x43 float32-high-byte boundary as"
            " the reference, or recompile with Pulsar2."
        )
    if old_form == "full":
        old_pat = struct.pack("<f", float(1.0 / old_a_scale))
        new_pat = struct.pack("<f", float(1.0 / new_a_scale))
        hits = _strided_run(reference_mcode, old_pat, 7)
        out = bytearray(reference_mcode)
        for off in hits:
            out[off : off + 4] = new_pat
        return bytes(out)
    old_pat = struct.pack("<f", float(1.0 / old_a_scale))[:3]
    new_pat = struct.pack("<f", float(1.0 / new_a_scale))[:3]
    hits = _find_all(reference_mcode, old_pat)
    hit_set = set(hits)
    run = []
    for start in hits:
        candidate = [start]
        i = start + 6
        while i in hit_set:
            candidate.append(i)
            i += 6
        if len(candidate) == 4:
            run = candidate
            break
    if len(run) != 4:
        raise ValueError(
            f"short-form A quad for {old_a_scale!r} not found as a stride-6 x4 run"
        )
    out = bytearray(reference_mcode)
    for off in run:
        out[off : off + 3] = new_pat
    return bytes(out)


def patch_mul_zp_x(reference_mcode: bytes, old_zp_x: int, new_zp_x: int) -> bytes:
    """Rewrite x's zero point, where the literal-byte form is present.

    The unit ``02 10 1b <zp_x> 83 36`` carries zp_x verbatim as its
    fourth byte in some Mul builds (see ``TestZpXLiteralByteWhenPresent``)
    -- but whether a given build uses this form is not predictable from
    zp_x's value at any magnitude; roughly as many builds use one of two
    other, still-undecoded forms instead (``TestZpXImmediateRegion``).
    This function only ever does the one thing it has evidence for:
    raises ``ValueError`` if the reference build does not carry
    ``old_zp_x`` in this exact form, rather than silently leaving zp_x
    unpatched or guessing at the opaque forms' encoding.
    """
    if not 0 <= old_zp_x <= 255 or not 0 <= new_zp_x <= 255:
        raise ValueError(f"zp_x must be a uint8: old={old_zp_x!r} new={new_zp_x!r}")
    old_unit = bytes.fromhex("02101b") + bytes([old_zp_x]) + bytes.fromhex("8336")
    hits = _find_all(reference_mcode, old_unit)
    if len(hits) != 1:
        raise ValueError(
            f"literal zp_x unit for {old_zp_x} not found exactly once"
            f" (found {len(hits)}) -- this reference build likely uses one"
            " of the opaque, undecoded forms instead"
        )
    out = bytearray(reference_mcode)
    out[hits[0] + 3] = new_zp_x
    return bytes(out)


def patch_conv_zp_x(reference_mcode: bytes, old_zp_x: int, new_zp_x: int) -> bytes:
    """Rewrite Conv's input zero point, where the literal-byte form is
    present -- the same ``02 10 1b <zp_x> 83 36`` unit ``patch_mul_zp_x``
    patches for Mul, confirmed to generalize byte-for-byte to Conv at the
    identical framing (``tests/test_axera_zpx_generalizes.py``). Delegates
    to ``patch_mul_zp_x`` directly: the unit and its patch are op-agnostic
    once the literal form is present, so there is nothing Conv-specific to
    do here beyond documenting Conv's own generation-scope caveat below.

    **This patches the VALUE inside an already-literal-form unit. It does
    NOT, and per current knowledge cannot, predict whether a genuinely
    fresh Conv build at ``new_zp_x`` would itself land in this literal
    form or one of the two still-undecoded opaque forms** -- presence is
    non-monotone in zp_x for both Mul and Conv (see
    ``TestZpXLiteralByteWhenPresent``), so this function's success at
    finding ``old_zp_x``'s unit says nothing about whether ``new_zp_x``
    "should" have used the same form; it only says the reference happens
    to hold both bytes in the one slot this project knows how to write.

    **A second, newly-confirmed caveat, found verifying this function
    against a real second Conv build (2026-09-17):** even holding
    x_scale exactly fixed (bit-identical between the two builds, not
    just close), patching *only* this 6-byte unit does not reproduce a
    real second build's mcode elsewhere in the stream -- roughly twenty
    bytes beyond this unit and beyond the shape's own ~3-byte rebuild
    noise floor also move with zp_x alone, at offsets this project has
    not decoded. Callers should not expect (and this function does not
    claim) that patching this unit alone reproduces a real rebuild
    bit-exactly, the way Mul's *scale*-family patches were confirmed to
    on real hardware (``tests/test_axera_mul_emit_hardware.py``) --
    zp_x's own patch has not had, and does not currently pass, that same
    bar. See ``tests/test_axera_conv_zpx_generator.py`` for the exact
    build pair and byte counts this was checked against.

    **A third caveat, decoded (not merely observed) 2026-09-18: for
    ``Conv(dilation=3, pad=3, cin=4, cout=4, insz=16)``-shaped inputs
    specifically, the ``<zp_x>`` byte this function writes IS the same
    register as ``reg=60``, part of Conv's own already-decoded 28-byte
    "binary path switch" (``tests/test_axera_conv_reg60_mechanism.py``).**
    Confirmed structurally across all 52 already-committed fixtures of
    that exact shape (this project's own PR #1636 test file, zero
    exceptions): the 6-byte literal unit this function searches for
    is the complete on-the-wire encoding of an ``S``-kind, ``reg=54``,
    ``tag=131`` record, and ``reg=54`` is independently already known to
    be byte-identical to ``reg=60`` in every sample checked. Practical
    consequences for THIS shape family only: (1) this function's own
    ``old_zp_x``/``new_zp_x`` values are not free parameters independent
    of the binary-cluster switch's own state -- patching zp_x here also
    changes which of the switch's own alternate states the stream reads
    as being in, whether or not that is what a caller intended; (2) this
    is very likely part of what the second caveat above ("~20 bytes
    beyond this unit... also move with zp_x alone, at offsets this
    project has not decoded") was actually seeing, since the binary-
    cluster switch is a real, coordinated 28-byte flip, not an isolated
    zp_x-adjacent artifact -- not confirmed by direct byte-offset
    cross-reference here, flagged as a plausible connection for whoever
    chases the remaining ~20 bytes next, not claimed as settled; (3)
    this function's own existing hardware-adjacent verification
    (``tests/test_axera_conv_zpx_generator.py``) used a DIFFERENT,
    unaffected shape (``Conv(cin=1,cout=1,hw=8,k=3)``), confirmed
    directly to sit at a different offset with values that don't
    resemble a switch pair -- that verification is not called into
    question by this caveat, only patches against the `dilation=3`
    shape family are. Whether an analogous collision exists for
    ``patch_mul_zp_x`` on any Mul shape has not been checked.
    """
    return patch_mul_zp_x(reference_mcode, old_zp_x, new_zp_x)


def bank81_field192_operand(k: int) -> bytes:
    """Predict Gemm bank ``0x81`` field=192's V-record operand from the
    contraction dimension ``K`` alone, no compiler in the loop.

    The 3-byte operand is ``4c 05 <1024 // k - 1>``, decoded from Gemm's
    native `Gemm` op (``tests/test_axera_gemm_bank_81_e1_decode.py``,
    merged) and independently reconfirmed byte-identical for a
    constant-weight ``MatMul(x, w)`` (``tests/test_axera_bank81_cross_op_check.py``,
    merged) -- the same formula, same leading bytes, same op-independent
    value, for every ``K`` both files tested. Verified directly against
    real committed fixtures in
    ``tests/test_axera_generator_progress_stocktake.py``.

    **What this narrow capability does NOT give you.** This predicts one
    field's own correct byte value for a shape that would trigger bank
    ``0x81`` at all -- it does not decode, and this function does not
    check, whether a given ``(K, N)`` pair actually lands in the
    ``0x81``-alone regime in the first place (that switch is itself a
    multi-plateau function of both ``K`` and ``N`` together, only
    partially mapped -- see ``tests/test_axera_gemm_e1_threshold_formula.py``
    and its own siblings). Nor does it establish that this field is
    SAFE to patch in place in an arbitrary target stream without the
    rest of the stream reflowing, the same caution ``patch_site_a``'s
    own docstring already raises for this project's other shape-derived
    fields (Gemm's ``K*M-1``, Conv's dilation/orientation fields): a
    decoded formula for a field's own value is a necessary but not
    sufficient condition for that field to be *generatable* in a target
    stream this function did not itself compile.

    **This was tested directly, not left as a hedge (2026-09-18).**
    Patching a real ``K=512`` reference build's own two field=192
    records to this function's own ``K=256``-predicted operand, then
    diffing the result byte-for-byte against a REAL, independently-built
    ``K=256`` reference -- the same ground-truth-comparison bar
    ``patch_matmul_a_scale``'s own form-crossing check and
    ``patch_conv_zp_x``'s own zp_x check used -- finds the two are not
    remotely close: the raw stream lengths themselves already differ
    (4368 vs 3792 bytes, a 576-byte gap no in-place byte patch can ever
    close), and even over their shared 3792-byte prefix, 2776 bytes
    differ (73%), with the first mismatch at byte 36 -- nowhere near
    either copy of the patched field itself (bytes 533 and 1172). The
    reverse direction (``K=256`` source patched to ``K=512``'s own
    predicted value, diffed against a real ``K=512`` build) shows the
    identical picture: same 2776/3792 diff count, same first-mismatch
    offset. This is confined-vs-scattered-diff evidence, and it is
    unambiguously scattered -- worse than ``patch_matmul_a_scale``'s own
    201-byte, single-region reflow for a form-crossing MatMul edit. The
    patched stream still round-trips through ``mcode.decode()``/
    ``mcode.check()`` with zero grammar errors either direction -- a
    weak, uninformative signal on its own (this project's grammar does
    not validate semantic correctness, only structural well-formedness)
    -- but it is NOT a real ``K=256`` build by any byte-level measure.
    See ``tests/test_axera_bank81_field192_patch_verify.py`` for the
    exact fixture pair and byte counts. **Conclusion: this field is
    understood, not generatable** -- changing ``K`` restructures nearly
    the entire compiled stream (consistent with ``K`` being Gemm's own
    real contraction dimension, driving tiling/scheduling throughout),
    not just this one field's own recorded value.
    """
    return b"\x4c\x05" + bytes([1024 // k - 1])


_REG8_QUAD_CANDIDATES = {
    b"\x33\x00\x20",
    b"\x23\x00\x40",
    b"\x23\x00\x30",
    b"\x23\x00\x10",
}

_REG8_QUAD_ANCHOR = bytes.fromhex("a2000000") + bytes.fromhex("12000000")


def emit_matmul_reg8_quad(reference_mcode: bytes, permutation) -> bytes:
    """Rewrite MatMul's 4-slot ``reg=8`` unordered-pool group to a
    caller-chosen permutation of the same 4-member candidate pool, in
    place.

    ``permutation`` is a 4-tuple of the candidate 3-byte payloads
    (``{33 00 20, 23 00 40, 23 00 30, 23 00 10}``, decoded for
    ``MatMul(A[4,8],B[8,8])`` in
    ``tests/test_axera_matmul_reg8_noise_source.py``, merged) in slot
    order -- the three ``V``-kind records at relative offsets +8/+16/+24
    from the group's own anchor, then the one ``S``-kind ``reg=8``
    record at +33. Must be a genuine permutation of the full 4-member
    set (no duplicate, no omission): the one constraint that source file
    found held with zero exceptions across all 10 real Pulsar2 builds it
    checked, unlike Gemm's own duplicate-tolerant 3-slot version
    (``tests/test_axera_gemm_reg8_second_noise_source.py``) or Conv's
    always-omits-one 3-of-4 version
    (``tests/test_axera_conv_reg8_reg60_noise_source.py``).

    Locates the group by its own stable anchor record (``a2 00 00 00
    12 00 00 00``) -- searched for here rather than hardcoded to a fixed
    offset, so this raises loudly if the anchor is not found exactly
    once instead of silently patching the wrong bytes. Only the 4
    candidate-identity fields (12 bytes total) are rewritten; every
    framing byte around them (the verb/bank/field header on each
    V-record, the positional ``0x00``/``0x82`` trailing byte, the raw
    length/link byte between the third V-record and the S-record, the
    S-record's own ``p``/tag/reg bytes) is left completely untouched --
    this project's usual "patch only what varies" discipline (see
    ``patch_output_quad``).

    **What this establishes and does not.** Verified in
    ``tests/test_axera_matmul_reg8_emit_verify.py``: every one of the 7
    distinct permutations actually observed across the source file's own
    10 real builds round-trips through this function exactly (a
    no-op emit reproduces the reference byte-for-byte; emitting a
    *different* observed permutation and re-decoding recovers exactly
    that permutation, at the same fixed offsets, with the rest of the
    stream untouched), and every result still passes ``mcode.check()``
    cleanly. It does NOT establish that a permutation this function CAN
    produce, but that no real Pulsar2 build has ever been observed to
    use, is itself a valid Pulsar2 output -- only that it is
    syntactically well-formed by this project's own grammar (of the 24
    mathematically possible permutations, only 7 have actually been
    observed in the small sample checked; this function does not know
    or enforce which subset a real compiler would choose). Unlike the
    scale-family patches (``tests/test_axera_mul_emit_hardware.py``),
    this has not been checked against real AX650N hardware.
    """
    perm = list(permutation)
    if len(perm) != 4 or set(perm) != _REG8_QUAD_CANDIDATES:
        raise ValueError(
            f"permutation must contain each of {sorted(_REG8_QUAD_CANDIDATES)} "
            f"exactly once, got {perm!r}"
        )
    hits = _find_all(reference_mcode, _REG8_QUAD_ANCHOR)
    if len(hits) != 1:
        raise ValueError(
            f"reg=8 quad anchor found {len(hits)} times in reference_mcode,"
            " expected exactly 1 -- wrong shape/reference?"
        )
    anchor = hits[0]
    out = bytearray(reference_mcode)
    for off, cand in zip((anchor + 8, anchor + 16, anchor + 24), perm[:3]):
        out[off + 4 : off + 7] = cand
    s_off = anchor + 33
    out[s_off + 1 : s_off + 4] = perm[3]
    return bytes(out)


_CONV_REG8_ANCHOR = bytes([0x00]) + b"\x12" + bytes([132, 170])

_CONV_REG8_CLASSES = {
    "P1": b"\x23\x00\x20",
    "P2": b"\x23\x00\x10",
    "P3": b"\x23\x00\x40",
    "P4": b"\x30",
}


def _conv_reg8_slot_bytes(cls: str, tag: int, reg: int) -> bytes:
    payload = _CONV_REG8_CLASSES[cls]
    return bytes([len(payload) - 1]) + payload + bytes([tag, reg])


def emit_conv_reg8_group(
    reference_mcode: bytes,
    slot1: tuple[str, int],
    slot2: tuple[int, str, int],
    slot3: tuple[int, str, int],
) -> bytes:
    """Rewrite Conv's 3-of-4 ``reg=8`` unordered-pool group (anchored at
    a stable ``reg=170`` record) to a caller-chosen configuration,
    completing this same session's generator-progress theme for the
    RICHEST of the three op-specific ``reg=8`` variants decoded so far
    (``emit_matmul_reg8_quad`` did MatMul's own simpler, fixed-length
    4-of-4 case; this is Conv's own, per
    ``tests/test_axera_conv_reg8_reg60_noise_source.py``, PR #1580).

    ``slot1`` is ``(class, tag)`` -- always labeled ``reg=174``.
    ``slot2``/``slot3`` are each ``(reg, class, tag)``, ``reg`` one of
    ``8``/``242``/``176`` (the register-label pool PR #1580 found
    those two slots draw from). ``class`` is one of ``"P1"``/``"P2"``/
    ``"P3"``/``"P4"``; the 3 slots must use 3 DISTINCT classes with
    zero duplicates -- unlike Gemm's own duplicate-tolerant version
    (``tests/test_axera_gemm_reg8_second_noise_source.py``), Conv's
    mechanism was found to hold this with zero exceptions across all 8
    real builds checked. ``P4``'s own payload is always the 1-byte
    short form (``0x30``) -- observed in every one of its 3
    occurrences across the 8 real samples regardless of which slot it
    lands in, never the 3-byte long form the other 3 classes use.

    ``reg=172``'s own tag is derived automatically (``134`` iff
    ``slot1``'s class is ``"P4"``, else ``132``) -- the exact
    biconditional PR #1580 verified with zero exceptions.

    **Every slot's own record tag must be supplied explicitly, not
    just slot1's -- two previously-unnoticed wrinkles, surfaced
    building this function, not by PR #1580's own original decode**
    (whose own ``EXPECTED`` table discarded each slot record's own tag
    entirely, keeping only ``(reg, class)``): slot1's own tag is
    ``130`` in 7 of 8 real builds but ``132`` in one
    (``conv_dilation3_v7stability_r0.mcode.gz``, an otherwise-identical
    long-form ``P2`` slot1); separately, slot2/slot3's own tag is
    ``130`` in every occurrence of ``reg=8``/``242`` but ``132`` in the
    corpus's one ``reg=176`` occurrence
    (``conv_dilation3_v7stability_r1.mcode.gz``). Whether ``132``
    tracks "first time this register label is used" or something else
    is not decoded here -- with only 2 exceptions in 24 total slot
    observations, there isn't enough data in this project's own corpus
    to tell. Rather than silently defaulting past this and risking a
    plausible-looking but wrong emission, every slot's tag is a
    required, explicit argument (matching PR #1580's own established
    value, ``130``, is the safe default for any NEW, not-yet-observed
    configuration, but callers reproducing a specific real build must
    pass that build's own actual per-slot values).

    Locates the group by ``reg=170``'s own stable, byte-identical-
    across-all-8-samples anchor record (``00 12 84 aa``) -- searched
    for, not hardcoded to a fixed offset. **This is a length-changing
    edit, unlike ``emit_matmul_reg8_quad``'s fixed-size splice**: the
    group's own total byte length is 24 or 26 bytes depending on
    whether slot1's class is ``"P4"`` (short form) or not, and the OLD
    group being replaced may be either length regardless of what the
    NEW one is -- the old group's own true length is walked byte-by-
    byte from the reference's own p-bytes (not assumed), and
    everything after it is shifted accordingly.

    **The length change is now handled automatically (2026-09-18).**
    Earlier versions of this function returned the raw length-changed
    stream directly, which left ``mcode.check()`` reporting a real
    error (``"tail: no readable segment table"``) whenever the edit
    actually changed the group's own total length -- ``retarget_tail_vector``
    (below) decoded the shared root cause (a stale FlatBuffers-style
    relative-uoffset header word) and fixed it; this function now calls
    it internally before returning, so callers no longer need a manual
    second step. Verified in
    ``tests/test_axera_conv_reg8_emit_verify.py`` against all 8 real
    Pulsar2 builds this project has of ``Conv(k=3, dilation=3, pad=3,
    cin=4, cout=4, insz=16)``, including every length-changing
    cross-fixture case that used to be broken: the result now passes
    ``mcode.check()`` cleanly regardless of whether the edit changed
    the group's own total length.

    **What this establishes and does not.** A no-op emit (using each
    fixture's own exact observed configuration, including its own
    per-slot tags) reproduces that fixture byte-for-byte in every one
    of the 8 cases. Emitting a DIFFERENT real fixture's own
    configuration into another fixture's base stream always re-decodes
    to exactly that target configuration, with the rest of the stream
    (outside the reg=8 group and the one retargeted header word)
    byte-identical to the original reference. This function does NOT
    establish end-to-end shape-to-mcode generation, that an untested
    ``(class, reg, tag)`` combination not seen in these 8 samples is
    something a real Pulsar2 build would produce, or anything about
    device-level correctness -- the same scope limits
    ``emit_matmul_reg8_quad``'s own docstring already states. Nor does
    the tail-vector fix establish semantic validity, only structural
    well-formedness by this project's own grammar (``retarget_tail_vector``'s
    own docstring).
    """
    slot1_class, slot1_tag = slot1
    if slot1_class not in _CONV_REG8_CLASSES:
        raise ValueError(
            f"slot1 class must be one of {sorted(_CONV_REG8_CLASSES)}, got"
            f" {slot1_class!r}"
        )
    for reg, cls, tag in (slot2, slot3):
        if cls not in _CONV_REG8_CLASSES:
            raise ValueError(
                f"slot class must be one of {sorted(_CONV_REG8_CLASSES)}, got {cls!r}"
            )
        if reg not in (8, 242, 176):
            raise ValueError(f"slot register must be one of 8/242/176, got {reg!r}")
        if tag not in (130, 132):
            raise ValueError(f"slot tag must be 130 or 132, got {tag!r}")
    classes = {slot1_class, slot2[1], slot3[1]}
    if len(classes) != 3:
        raise ValueError(
            "the 3 slots must use 3 DISTINCT classes (Conv's mechanism is never"
            " duplicate-tolerant, unlike Gemm's) -- got"
            f" {(slot1_class, slot2[1], slot3[1])!r}"
        )
    if slot1_tag not in (130, 132):
        raise ValueError(f"slot1 tag must be 130 or 132, got {slot1_tag!r}")

    hits = _find_all(reference_mcode, _CONV_REG8_ANCHOR)
    if len(hits) != 1:
        raise ValueError(
            f"reg=170 anchor found {len(hits)} times in reference_mcode, expected"
            " exactly 1 -- wrong shape/reference?"
        )
    anchor = hits[0]

    reg172_tag = 134 if slot1_class == "P4" else 132
    reg172_bytes = bytes([0x00]) + b'"' + bytes([reg172_tag, 172])
    new_group = (
        reference_mcode[anchor : anchor + 4]
        + reg172_bytes
        + _conv_reg8_slot_bytes(slot1_class, slot1_tag, 174)
        + _conv_reg8_slot_bytes(slot2[1], slot2[2], slot2[0])
        + _conv_reg8_slot_bytes(slot3[1], slot3[2], slot3[0])
    )

    pos = anchor + 8
    for _ in range(3):
        p = reference_mcode[pos]
        pos += p + 4
    old_group_end = pos

    edited = reference_mcode[:anchor] + new_group + reference_mcode[old_group_end:]
    return retarget_tail_vector(reference_mcode, edited)


def _gemm_reg8_group_bounds(reference_mcode: bytes) -> tuple:
    """Locate Gemm's ``reg=78``-anchored 3-slot ``reg=8`` pool group's
    own exact byte span ``[start, end)``.

    Unlike ``emit_matmul_reg8_quad``'s own fixed-byte-pattern anchor
    (unique across the whole MatMul stream it was checked against),
    Gemm's own reg=78 anchor record's closing ``tag+reg`` byte pair
    (``84 4e`` for its short form, ``82 4e`` for its long form,
    decoded in ``tests/test_axera_gemm_reg8_second_noise_source.py``)
    is NOT globally unique in a Gemm stream -- both patterns recur
    several more times elsewhere (verified directly: `84 4e` and
    `82 4e` each appear 2-3 times total across the 8 real
    `Gemm(1,512,1000)` fixtures this was checked against, only one of
    which is the pool's own anchor). A raw literal-byte search
    therefore cannot locate the group unambiguously the way it could
    for MatMul's own case. This function uses ``mcode.decode()``
    instead (imported locally, the one function in this file that
    needs it -- every other function here is deliberately
    decode-free), matching
    ``tests/test_axera_gemm_reg8_second_noise_source.py``'s own
    ``three_slot_group()`` filter exactly: the one ``S``-kind record
    with ``reg=78``, ``tag`` in ``(130, 132)``, and a payload matching
    one of the 3 known candidate classes.

    The group's own end is the first ``verb=162, field=0, bank=0``
    V-record after the anchor -- the same recurring cross-op marker
    this project's reg=8 work has repeatedly used as an anchor for
    Mul's and MatMul's own versions of this mechanism too.

    Raises if the anchor, or its own immediately-following end marker,
    is not found exactly once.
    """
    import mcode  # noqa: PLC0415 -- deliberately local, see docstring

    recs = mcode.decode(reference_mcode, **mcode.FULL_RULE)
    class_of = {
        b"\x93\x00\x40": "A",
        b"\x93\x00\x30": "B",
        b"\x23\x00\x20": "C",
        b"\x23": "C",
    }
    anchor_hits = [
        r
        for r in recs
        if r["kind"] == "S"
        and r.get("reg") == 78
        and r.get("tag") in (130, 132)
        and r.get("payload") in class_of
    ]
    if len(anchor_hits) != 1:
        raise ValueError(
            f"reg=78 pool anchor found {len(anchor_hits)} times, expected exactly 1"
        )
    start = anchor_hits[0]["at"]
    end_hits = [
        r
        for r in recs
        if r["kind"] == "V"
        and r.get("verb") == 162
        and r.get("field") == 0
        and r.get("bank") == 0
        and r["at"] > start
    ]
    if not end_hits:
        raise ValueError(
            "no verb=162,field=0,bank=0 V-record found after the reg=78 anchor"
        )
    end = min(r["at"] for r in end_hits)
    return start, end


def emit_gemm_reg8_group(reference_mcode: bytes, donor_mcode: bytes) -> bytes:
    """Splice Gemm's own 3-slot ``reg=8`` unordered-pool group -- the
    ``reg=78``-anchored region ``tests/test_axera_gemm_reg8_second_noise_source.py``
    decoded, together with its own linked ``reg=0`` biconditional
    indicator -- out of ``donor_mcode`` and into ``reference_mcode``,
    in place of ``reference_mcode``'s own version of the same group.

    Unlike MatMul's own version of this mechanism
    (``emit_matmul_reg8_quad``, a fixed-length 4-slot permutation
    overwritten byte-for-byte in place), Gemm's own group is NOT
    fixed-length: each of its 3 candidate classes has both a 3-byte
    "long" form and (class ``C`` only, observed) a 1-byte "short" form,
    the group can be 3-of-3 distinct or duplicate-tolerant (2 distinct,
    one class repeated or a whole slot dropped), and the source file's
    own ``reg=0`` biconditional indicator is not a separately-inserted
    record at all -- it is a different GRAMMAR INTERPRETATION of a
    fixed-position byte range that follows the pool slots, verified
    directly here: every one of the 8 real builds this was checked
    against has the *exact same* total group span (either 21 or 23
    bytes, correlated with whether the anchor itself uses its short or
    long form) as every other build sharing that same anchor form --
    the indicator's presence/absence changes what THOSE bytes decode
    as, not how many bytes there are.

    Given that real complexity, this function does not attempt to
    SYNTHESIZE an arbitrary caller-chosen slot assignment from
    scratch (which would require independently re-deriving exactly
    which byte encodes what under which of the two anchor forms, a
    real risk of getting subtly wrong) -- it instead extracts a real,
    already-*compiled* group verbatim from a donor build (itself one
    of the 8 already-observed real Pulsar2 outputs, or any other
    stream this function's own bounds-search succeeds on) and splices
    it whole into the reference stream, which is safe precisely
    because the spliced bytes are never anything other than bytes a
    real Pulsar2 build actually produced. This is a genuine, if more
    conservative, generation capability: given a target reference
    build and a *menu* of already-observed donor group states (this
    project's own real fixture corpus already provides several), a
    generator can choose which one to splice in.

    **What this establishes and does not.** Verified in
    ``tests/test_axera_gemm_reg8_emit_verify.py`` against all 8 real
    ``Gemm(1,512,1000)`` fixtures:

    - A no-op splice (any donor identical to the reference, including
      ``donor_mcode is reference_mcode``) reproduces the reference
      byte-for-byte, for all 8.
    - Splicing a donor whose anchor uses the SAME form as the
      reference's own (the 8 fixtures split cleanly into two groups of
      4 by anchor form -- short/21-byte-span, long/23-byte-span; all 12
      ordered same-group cross-pairs checked) preserves the total
      stream length, round-trips through ``mcode.decode()``/
      ``mcode.check()`` with zero hard errors, and reads back exactly
      the donor's own slot assignment and ``reg=0`` indicator state,
      with everything outside the spliced span byte-identical to the
      original reference.
    - Splicing ACROSS anchor forms (a length-changing splice, e.g. a
      21-byte donor group into a 23-byte reference span or vice versa)
      used to reliably produce a specific hard ``mcode.check()`` error
      -- ``"tail: no readable segment table (no header word points at a
      tail table vector)"``. **This is now fixed automatically
      (2026-09-18).** ``retarget_tail_vector`` (below) decoded the
      shared root cause (a stale FlatBuffers-style relative-uoffset
      header word left over from the length change) and this function
      now calls it internally before returning: verified in
      ``tests/test_axera_gemm_reg8_emit_verify.py`` against both
      cross-anchor-form directions, the spliced result now passes
      ``mcode.check()`` cleanly, decodes to exactly the donor's own
      slot assignment and ``reg=0`` indicator state, and leaves
      everything outside the spliced group (and the one retargeted
      header word) byte-identical to the original reference.

    It does NOT establish that every one of the 3^3 = 27 mathematically
    conceivable slot-class combinations (ignoring form/duplicate
    subtleties entirely) is itself a valid Pulsar2 output, nor does it
    attempt the harder problem ``emit_matmul_reg8_quad`` solved of
    accepting an arbitrary caller-chosen assignment directly -- only 8
    real, verified group states (4 same-length pairs each) are known to
    be safe to splice as of this function. Nor does the tail-vector fix
    establish semantic validity, only structural well-formedness by
    this project's own grammar (``retarget_tail_vector``'s own
    docstring).
    """
    ref_start, ref_end = _gemm_reg8_group_bounds(reference_mcode)
    donor_start, donor_end = _gemm_reg8_group_bounds(donor_mcode)
    donor_group = donor_mcode[donor_start:donor_end]
    edited = reference_mcode[:ref_start] + donor_group + reference_mcode[ref_end:]
    return retarget_tail_vector(reference_mcode, edited)


def retarget_tail_vector(reference_mcode: bytes, edited_mcode: bytes) -> bytes:
    """Fix the ONE shared root cause behind every length-changing edit's
    ``mcode.check()`` failure this session's own generator work has hit:
    ``emit_conv_reg8_group``'s length-changing reconfigurations and
    ``emit_gemm_reg8_group``'s cross-anchor-form splices both reliably
    produce the identical hard error, ``"tail: no readable segment table
    (no header word points at a tail table vector)"``
    (``tests/test_axera_reg8_emit_capability_synthesis.py``, PR #1631,
    named this as a recurring, never-decoded failure mode across 3
    independent functions).

    **The mechanism, decoded directly against real broken output from
    both functions.** ``mcode.tail_vector()`` (``scripts/axera/mcode.py``)
    locates an mcode stream's tail FlatBuffers vector by scanning the
    fixed-size header (the first ~297 bytes, or up to the convolution-
    engine channel-extent marker ``a1 00 40 02`` for graphs that have
    one) for a 4-byte little-endian word ``w`` at some offset ``o`` such
    that ``o + w`` points at a valid-looking vector -- a FlatBuffers-
    style *relative* uoffset. Every ``reg=8`` group (and, more generally,
    every one of this project's own decoded resource-model fields) lives
    well AFTER this header, and BEFORE the tail vector itself -- so when
    an edit changes that region's own total length, the true tail vector
    shifts by the same delta, but the header's own stored *relative*
    uoffset does not, since none of this session's own emit functions
    touch bytes before offset ~300. Confirmed directly at a single
    concrete offset (``o=272`` for the `Conv(dilation=3)`/`Gemm(1,512,1000)`
    shape family this was checked against) via the four cases below --
    but this function does not hardcode that offset; it re-locates ``o``
    fresh from ``reference_mcode`` every call, the same defensive
    posture every other locate-by-content function in this file uses.

    ``reference_mcode`` must be the UNEDITED stream ``edited_mcode`` was
    derived from (so this function can locate the tail vector's own
    correct OLD position and offset word before the edit) --
    ``len(edited_mcode) - len(reference_mcode)`` is taken directly as the
    length delta to apply; no assumption is made about WHERE within the
    stream that length changed, only that it happened somewhere after
    the header region this function scans.

    **What this establishes and does not.** Verified in
    ``tests/test_axera_tail_table_mechanism.py`` against all 4 already-
    known broken cases from this session's own prior work -- both
    directions of ``emit_conv_reg8_group``'s own length-changing
    reconfiguration (shrink and grow) and both directions of
    ``emit_gemm_reg8_group``'s own cross-anchor-form splice: applying
    this function to each function's own broken output makes
    ``mcode.check()`` report zero hard errors, ``mcode.segments()``
    tile exactly to the (correctly relocated) tail vector, and
    ``mcode.decode()`` succeed -- a genuine fix, not merely a
    check-passes technicality, confirmed 4/4. It does NOT fix
    ``bank81_field192_operand``'s own K-changing reflow (tested
    directly, and correctly found not to need fixing by this function:
    that edit is length-PRESERVING -- a 1-byte in-place value overwrite
    -- so it never triggers this specific tail-vector error in the
    first place; ``mcode.check()`` already reports it clean, per
    ``tests/test_axera_bank81_field192_patch_verify.py``'s own finding
    that the patched stream "still decodes/checks cleanly" despite being
    semantically wrong. Field192's own problem is that `K` drives the
    ENTIRE stream's tiling/scheduling, not a single stale pointer --
    a fundamentally different, and NOT locally fixable, class of
    failure). This function also does NOT establish that a
    length-changed stream it retargets is a valid Pulsar2 output
    semantically (only that it is structurally well-formed by this
    project's own grammar) -- the same "syntactically valid, not proven
    to be what Pulsar2 would produce" scope every other emit function in
    this file already carries. Composing this with the length-changing
    emit functions it fixes -- so a caller never has to call it
    separately -- is a natural next step, not attempted here.
    """
    # struct is already imported at module level; mcode is deliberately
    # local here, matching _gemm_reg8_group_bounds's own precedent --
    # every OTHER function in this file is decode-free.
    import mcode  # noqa: PLC0415

    t_ref = mcode.tail_vector(reference_mcode)
    first_verb = reference_mcode.find(b"\xa1\x00\x40\x02")
    if first_verb <= 0:
        first_verb = 297
    header_off = None
    for o in range(0, first_verb - 3, 4):
        if o + struct.unpack_from("<I", reference_mcode, o)[0] == t_ref:
            header_off = o
            break
    if header_off is None:
        raise ValueError(
            "could not locate the tail-vector header word in reference_mcode"
        )
    delta = len(edited_mcode) - len(reference_mcode)
    new_t = t_ref + delta
    patched = bytearray(edited_mcode)
    struct.pack_into("<I", patched, header_off, new_t - header_off)
    return bytes(patched)
