"""Record-level emitters for the ResNet18 training step's remaining non-memory
ops: ReduceSum, Greater/Less -> Cast, Softmax, Log, MaxPool, ReduceMean, Neg,
and the Sqrt shape ``[512,512,3,3]`` that ``elementwise_scale_emit`` left out.

Every AX650 MCode segment is an LZ77 stream over 8-byte register records
(``short_unit_codec``, PR #1850). Decompressed, a calibration change touches
whole records only (``docs/axera-misc-op-record-emit.md``):

* **Scale lanes.** Each scale-derived float32 is written once per lane, eight
  lanes, on registers ``0x0f50..0x0fc0`` (or ``0x0fd0..0x1040``). The value is
  Pulsar2's float32 scales combined in float64 and rounded to float32.

  * ReduceSum: ``1/s_x`` (dequantize), ``s_x/s_y`` (requantize) and ``s_y``
    (output), once per tile.
  * Sqrt: ``1/s_x``, ``s_x`` and ``s_y``, once per tile.

* **ReduceSum zero points.** Each of the three stages above is preceded by
  writes of registers ``0x1a90`` and ``0x1b10``:

  * dequantize: ``0x1a90 = 0``, ``0x1b10 = zp_x``;
  * requantize: ``0x1a90 = zp_x * 2**k``, ``0x1b10 = zp_y``, where ``2**k``
    depends only on the shape (the template's own value divided by its own
    ``zp_x``);
  * output: ``0x1a90 = zp_y``, ``0x1b10 = 0``.

  Pulsar2 omits a ``0x1b10`` write when the register already holds that value.
  That is why a zero-point change can move one record: ``reducesum_s1``
  (``zp_x = zp_y = 128``) has no requantize ``0x1b10`` write but does write
  ``0x1b10 = 0`` after its output ``0x1a90``, while ``reducesum_asym``
  (``zp_x = 64, zp_y = 0``) is the other way round. An omitted write goes
  directly after its stage's ``0x1a90`` record. This model turns either build
  into the other exactly.

* **Softmax, Log, MaxPool, ReduceMean** (the step's tail ops) follow the same
  lane pattern, one float32 formula per lane run:

  * Softmax: ``1/s_x``, ``s_x``, ``1/s_y``, ``s_y``, plus one ``0x1b10 = zp_x``
    write before the first run (``zp_y`` is 0 and fixed). The same rules
    serve the last-axis ``[1, 64]`` template; ``softmax_codes`` is the op's
    measured arithmetic and ``softmax_output_scale`` its calibration rule;
  * Log: ``1/s_x`` and ``s_y`` (the dequantize lanes), plus a 258-entry u8
    table (two u16 entries per record, registers ``0x1050..0x1850``;
    ``0x1850`` is written after an unrelated ``0x1860..0x1a50`` block):
    ``clip(rint(log((q - zp_x) s_x) / s_y) + zp_y)`` for ``q`` in 0..255,
    then entry 255 again and a 0;
  * MaxPool: ``1/s_x`` and ``s_x`` (``s_y = s_x``);
  * ReduceMean: ``1/s_x``, ``s_x/(s_y*N)`` with ``N`` the reduced element
    count, and ``s_y``.

  Their zero points are fixed by the template except Softmax's ``zp_x`` and
  ReduceMean's: ReduceMean has ReduceSum's three zero-point stages (its
  requantize ``0x1a90`` is ``zp_x * N``), so the ReduceSum rules below the
  lanes, elision and re-padding included, retarget both of its zero points
  (measured at ``[1, 64]``; ``zp_x = 0`` stays refused, as for ReduceSum).

* **Neg** ``[1,1]`` (``s_y = s_x``, ``zp_y = 255 - zp_x``) compiles to one of
  two programs, picked by the scale alone (float32 ``s < 1/64``: small). Each
  has lanes ``1/s_x`` and ``s_x``, and zero-point writes on ``0x1a90``,
  ``0x1ad0`` and ``0x1b10`` that are omitted when the register already holds
  the value (outside register-block dumps), so any zero point retargets.

* **Sigmoid, Mul, Add, Div** at ``[1, 64]`` (standalone, float32 I/O; scale
  and zero-point keys ``x``/``y``, or ``a``/``b``/``y`` for two inputs). Lane
  runs in program order, each preceded by a fixed list of zero-point writes on
  ``0x1a90``/``0x1ad0``/``0x1b10``:

  * Sigmoid: ``1/s_x`` (``0x1b10 = zp_x``) and ``s_y``, plus Log's table
    layout holding ``clip(rint(sigmoid((q - zp_x) s_x) / s_y) + zp_y)``;
  * Mul: ``1/s_a`` (``0x1b10 = zp_a``), ``1/s_b``, ``s_y/(s_a s_b)``
    (``0x1a90 = 0x1ad0 = zp_a = zp_b``, ``0x1b10 = zp_y``) and ``s_y``
    (``0x1a90 = zp_y``);
  * Add: ``1/s_a`` (``0x1b10 = zp_a = zp_b``), ``1/s_b`` and ``s_y``
    (``0x1a90 = zp_y``), plus the int32 ``add_offset`` on ``0x1ef0..0x1f20``.
    The Q15 input weights ``add_q15`` live in ``npu_params``, which this
    module does not rewrite;
  * Div: ``1/s_a`` (``0x1b10 = zp_a``), ``1/s_b``, ``s_a/(s_b s_y)``
    (``0x1a90 = zp_a``, ``0x1b10 = zp_y``) and ``s_y`` (``0x1a90 = zp_y``).

  Refused because no build measured them: ``zp_a != zp_b`` (Mul, Add), a
  nonzero Div ``zp_b`` or Sigmoid ``zp_y``, a zero point that is 0 on one side
  of the retarget only, a write that would repeat its register's value where
  the template's does not (or the reverse), and an Add target whose Q15
  weights differ from the template's.

* **Greater -> Cast** is not quantized at all. Builds at the same shape and
  different calibrations are record-identical except segment 0's slot table
  (records 2..5, a per-build permutation that is rebuild noise), so a template
  at the exact shape is the whole program.

Shape does not enter any of these edits. A template serves only the exact
shape, ReduceSum axes and ``keepdims`` it was built with. Every tensor
dimension, tile count and address stays as built. Anything the rules above do
not cover raises ``ValueError``:

* an unknown shape;
* a target whose scale-formula values coincide (the lanes could no longer be
  told apart);
* a ReduceSum ``zp_x`` of 0, because the omission rule was not measured for
  the dequantize ``0x1b10`` write;
* a ReduceSum ``0x1a90`` write that would equal the register's current value,
  which was never observed;
* a segment whose records do not end in the ``0xa2`` terminator plus 0..3
  zero pad records (the padding rule below would be ambiguous).

A zero-point change may also add or remove records overall: with ``zp_y = 0``
the output stage's ``0x1b10 = 0`` write (and, on a later tile, the requantize
``0x1b10 = zp_y`` write) is elided because the register already holds 0. The
emitter applies the same elision rule and re-pads the segment: every
decompressed segment is a whole number of 4-record (32-byte) groups, filled
with all-zero records after the ``0xa2`` terminator. Some multi-axis
ReduceSum programs also write ``zp_x`` packed into all four bytes
(``zp_x * 0x01010101``) on eight lanes ``0x0d30..0x0da0``; those are
rewritten with the zero points.
"""

from __future__ import annotations

import gzip
import json
import os
import struct
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import binary_op_scale_emit as bose  # noqa: E402
import short_unit_codec as suc  # noqa: E402

RECORD = suc.RECORD
LANES = 8
LANE_BASES = (0x0F50, 0x0FD0)
ZP_PACKED_BASE = 0x0D30  # zp_x in all four bytes, eight lanes (multi-axis ReduceSum)
PAD_GROUP = 4  # decompressed segments are padded to whole 4-record groups
TERMINATOR = 0xA2
ZP_ACC = 0x1A90
ZP_IN = 0x1B10
FIXTURES = os.path.join(_HERE, "fixtures")
TEMPLATE_INDEX = os.path.join(FIXTURES, "misc_op_record_emit", "index.json")
# segment 0's slot-table records that differ between otherwise identical builds
SEG0_NOISE = range(2, 6)
OPS = (
    "ReduceSum",
    "Sqrt",
    "GreaterCast",
    "LessCast",
    "Softmax",
    "Log",
    "MaxPool",
    "ReduceMean",
    "Neg",
    "Sigmoid",
    "Mul",
    "Add",
    "Div",
)
CALIBRATED = ("ReduceSum", "Sqrt", "Softmax", "Log", "MaxPool", "ReduceMean", "Neg")
# standalone [1,64] ops whose lane runs are matched in program order
STAGED = ("Sigmoid", "Mul", "Add", "Div")
CALIBRATED += STAGED
TWO_INPUT = ("Mul", "Add", "Div")
ZP_AUX = 0x1AD0
ADD_SHIFT = 0x1EA0  # holds 15 before Add's offset block
ADD_OFFSET_BASE = 0x1EF0  # Add's int32 offset, four lanes
ADD_OFFSET_LANES = 4
ADD_Q15 = 15
# verb 0xa8 records on these registers pick a slot of segment 0's slot table
SLOT_VERB = 0xA8
SLOT_REGS = (0x03D0, 0x02B0)
LOG_TABLE_BASE = 0x1050
LOG_TABLE_RECORDS = 129  # 258 u16 entries
CALIBRATION_FREE = (
    "GreaterCast",
    "LessCast",
    "GreaterOrEqualCast",
    "LessOrEqualCast",
)
# Neg [1,1] compiles to one of two programs depending on the scale alone
# (zero points 0..255 on both sides): float32 s < 1/64 gives the small program,
# s >= 1/64 the large one (0.01562 small, 0.015625 and 0.01563 large).
NEG_PROGRAM_SCALE = 2.0**-6


def _f32(x: float) -> float:
    return float(np.float32(x))


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def lane_values(
    op: str, scales: Mapping[str, float], reduce_count: int | None = None
) -> dict[str, int]:
    """Float32 bit patterns of each scale-lane formula of ``op``.
    ``reduce_count`` is ReduceMean's number of reduced elements."""
    if op in TWO_INPUT:
        return _two_input_lane_values(op, scales)
    sx, sy = float(scales["x"]), float(scales["y"])
    if op == "ReduceSum":
        vals = {"1/s_x": 1.0 / sx, "s_x/s_y": sx / sy, "s_y": sy}
    elif op == "Sqrt":
        vals = {"1/s_x": 1.0 / sx, "s_x": sx, "s_y": sy}
    elif op == "Softmax":
        vals = {"1/s_x": 1.0 / sx, "s_x": sx, "1/s_y": 1.0 / sy, "s_y": sy}
    elif op == "Log":
        # the table lookup is dequantized with s_y (both native Log builds share
        # one s_y, so only a device run caught it: docs/axera-emitter-device-check.md)
        vals = {"1/s_x": 1.0 / sx, "s_y": sy}
    elif op == "MaxPool":
        if _f32(sx) != _f32(sy):
            raise ValueError("MaxPool shares one scale between input and output")
        vals = {"1/s_x": 1.0 / sx, "s_x": sx}
    elif op == "ReduceMean":
        if not reduce_count:
            raise ValueError("ReduceMean needs its reduced element count")
        vals = {"1/s_x": 1.0 / sx, "s_x/(s_y*N)": sx / (sy * reduce_count), "s_y": sy}
    elif op == "Neg":
        if _f32(sx) != _f32(sy):
            raise ValueError("Neg shares one scale between input and output")
        vals = {"1/s_x": 1.0 / sx, "s_x": sx}
    elif op == "Sigmoid":
        vals = {"1/s_x": 1.0 / sx, "s_y": sy}
    else:
        raise ValueError(f"{op!r} has no scale lanes")
    return {k: _bits(_f32(v)) for k, v in vals.items()}


def _two_input_lane_values(op: str, scales: Mapping[str, float]) -> dict[str, int]:
    """Mul/Add/Div lane formulas in program order (float32 scales combined in
    float64, rounded to float32)."""
    sa, sb, sy = (_f32(scales[k]) for k in ("a", "b", "y"))
    vals = {"1/s_a": 1.0 / sa, "1/s_b": 1.0 / sb}
    if op == "Mul":
        vals["s_y/(s_a*s_b)"] = sy / (sa * sb)
    elif op == "Div":
        vals["s_a/(s_b*s_y)"] = sa / (sb * sy)
    vals["s_y"] = sy
    return {k: _bits(_f32(v)) for k, v in vals.items()}


def mcode_initializer(model: onnx.ModelProto) -> onnx.TensorProto:
    return bose._mcode_init(model)


def _chunks(raw: bytes) -> list[bytes]:
    if len(raw) % RECORD:
        raise ValueError("decompressed segment is not a whole number of records")
    return [raw[i : i + RECORD] for i in range(0, len(raw), RECORD)]


def _reg(w: bytes) -> int:
    return w[2] | (w[3] << 8)


def _val(w: bytes) -> int:
    return int.from_bytes(w[4:], "little")


def _with_val(w: bytes, v: int) -> bytes:
    return w[:4] + int(v).to_bytes(4, "little")


def lane_runs(words: Sequence[bytes]) -> list[tuple[int, int]]:
    """``(index, value)`` of every run of eight same-valued lane records."""
    runs = []
    i = 0
    while i + LANES <= len(words):
        base = _reg(words[i])
        run = words[i : i + LANES]
        if (
            base in LANE_BASES
            and all(w[0] == words[i][0] and w[1] == words[i][1] for w in run)
            and [_reg(w) for w in run] == [base + 0x10 * k for k in range(LANES)]
            and len({_val(w) for w in run}) == 1
        ):
            runs.append((i, _val(words[i])))
            i += LANES
        else:
            i += 1
    return runs


def _check_distinct(vals: Mapping[str, int], what: str) -> None:
    if len(set(vals.values())) != len(vals):
        raise ValueError(f"{what} scale formulas coincide: {dict(vals)}")


def _retarget_lanes(
    words, op, old, new, reduce_count=None
) -> tuple[list[bytes], list[tuple[int, str]]]:
    """Rewrite scale lanes; returns the words and each lane run's ``(index, kind)``."""
    if op in STAGED:
        return _retarget_ordered_lanes(words, op, old, new)
    old_v = lane_values(op, old, reduce_count)
    new_v = lane_values(op, new, reduce_count)
    _check_distinct(old_v, "template")
    _check_distinct(new_v, "target")
    by_bits = {b: k for k, b in old_v.items()}
    out = list(words)
    kinds = []
    for i, v in lane_runs(words):
        kind = by_bits.get(v)
        if kind is None:
            continue
        kinds.append((i, kind))
        for j in range(i, i + LANES):
            out[j] = _with_val(out[j], new_v[kind])
    return out, kinds


def _zp_stages(words, kinds) -> list[tuple[str, list[int]]]:
    """Each lane run's kind and the zero-point register records before it."""
    stages = []
    prev = 0
    for i, kind in kinds:
        zp = [j for j in range(prev, i) if _reg(words[j]) in (ZP_ACC, ZP_IN)]
        stages.append((kind, zp))
        prev = i + LANES
    return stages


_STAGE = {"1/s_x": "dequantize", "s_x/s_y": "requantize", "s_y": "output"}


def _retarget_reducesum_zps(words, kinds, old_zp, new_zp) -> list[bytes]:
    zx0 = int(old_zp["x"])
    zx, zy = int(new_zp["x"]), int(new_zp["y"])
    if zx0 == 0 or zx == 0:
        raise ValueError("zp_x = 0 is not measured (dequantize 0x1b10 omission)")
    state: dict[int, int | None] = {ZP_ACC: None, ZP_IN: None}
    edits: dict[int, bytes | None] = {}  # index -> new word, or None to drop
    inserts: dict[int, bytes] = {}  # insert after index
    for kind, zp_idx in _zp_stages(words, kinds):
        stage = _STAGE[kind]
        acc = [j for j in zp_idx if _reg(words[j]) == ZP_ACC]
        inp = [j for j in zp_idx if _reg(words[j]) == ZP_IN]
        if len(acc) != 1 or len(inp) > 1:
            raise ValueError(f"{stage} stage has an unmeasured zero-point layout")
        a = acc[0]
        if stage == "dequantize":
            want_acc, want_in = 0, zx
            if not inp:
                raise ValueError("dequantize stage without a 0x1b10 write")
        elif stage == "requantize":
            mult, rem = divmod(_val(words[a]), zx0)
            if rem:
                raise ValueError("requantize 0x1a90 is not a multiple of zp_x")
            want_acc, want_in = zx * mult, zy
        else:
            want_acc, want_in = zy, 0
        if state[ZP_ACC] == want_acc:
            raise ValueError(f"{stage} 0x1a90 write would repeat the register value")
        edits[a] = _with_val(words[a], want_acc)
        state[ZP_ACC] = want_acc
        write_in = state[ZP_IN] != want_in
        if inp and write_in:
            edits[inp[0]] = _with_val(words[inp[0]], want_in)
        elif inp:
            edits[inp[0]] = None
        elif write_in:
            if stage == "dequantize":
                raise ValueError("dequantize 0x1b10 insertion is not measured")
            inserts[a] = _with_val(_zp_in_word(words), want_in)
        state[ZP_IN] = want_in
    out = []
    for j, w in enumerate(words):
        new = edits.get(j, w)
        if new is not None:
            out.append(new)
        if j in inserts:
            out.append(inserts[j])
    return out


def _pad_count(words) -> int:
    """Trailing all-zero pad records after the ``0xa2`` terminator."""
    n = 0
    while n < len(words) and words[-1 - n] == bytes(RECORD):
        n += 1
    if n >= PAD_GROUP or n == len(words) or words[-1 - n][0] != TERMINATOR:
        raise ValueError("segment does not end in a terminator plus 0..3 pad records")
    return n


def _retarget_packed_zp(words, old_zx: int, new_zx: int) -> list[bytes]:
    """Rewrite the eight ``0x0d30..0x0da0`` lanes holding ``zp_x`` in every byte."""
    old_v, new_v = old_zx * 0x01010101, new_zx * 0x01010101
    out = list(words)
    for i in range(len(words) - LANES + 1):
        run = words[i : i + LANES]
        if [_reg(w) for w in run] == [ZP_PACKED_BASE + 0x10 * k for k in range(LANES)]:
            if any(_val(w) != old_v for w in run):
                raise ValueError("0x0d30 lanes do not hold the template's packed zp_x")
            for j in range(i, i + LANES):
                out[j] = _with_val(out[j], new_v)
    return out


def log_table(scales: Mapping[str, float], zero_points: Mapping[str, int]) -> list[int]:
    """Log's 258-entry u8 lookup table: ``clip(rint(log((q - zp_x) * s_x) / s_y)
    + zp_y, 0, 255)`` for ``q`` in 0..255 (``log 0`` clips to 0), then entry 255
    repeated and a 0."""
    sx, sy = float(np.float32(scales["x"])), float(np.float32(scales["y"]))
    q = np.arange(256, dtype=np.float64) - int(zero_points["x"])
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.rint(np.log(q * sx) / sy) + int(zero_points["y"])
    v = np.clip(np.nan_to_num(v, nan=0.0, neginf=0.0), 0, 255).astype(int).tolist()
    return v + [v[255], 0]


def minmax_input_calibration(samples: Sequence[np.ndarray]) -> tuple[float, int]:
    """``(scale, zero_point)`` Pulsar2's MinMax gives a uint8 tensor from its
    calibration samples: the range is widened to include 0, ``s = (hi - lo) /
    255`` and ``zp = rint(-lo / s)``. Bit-exact in float32 on the five
    ``Softmax:1x64:axis1`` held-out builds (zero points 64, 128 and 232)."""
    lo = min(0.0, min(float(np.min(x)) for x in samples))
    hi = max(0.0, max(float(np.max(x)) for x in samples))
    scale = (hi - lo) / 255.0
    return scale, int(np.rint(-lo / scale))


def softmax_output_scale(samples: Sequence[np.ndarray], axis: int = -1) -> float:
    """A Softmax output's scale: the largest probability over the calibration
    samples divided by 255 (zero point 0), not ``1/255``. So the output
    saturates at code 255 on any row whose largest probability exceeds what
    calibration saw. Matches the held-out builds to 1e-7 relative (Pulsar2's
    own float rounding)."""
    top = 0.0
    for x in samples:
        x = np.asarray(x, dtype=np.float64)
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        top = max(top, float((e / e.sum(axis=axis, keepdims=True)).max()))
    return top / 255.0


def softmax_codes(
    x: np.ndarray,
    scales: Mapping[str, float],
    zero_points: Mapping[str, int],
    axis: int = -1,
) -> np.ndarray:
    """Output codes of the compiled Softmax for float input ``x`` (the device
    returns ``codes * s_y``).

    The input is quantized to ``q = clip(rint(x / s_x) + zp_x, 0, 255)``. Each
    element then depends only on its code distance below the row maximum,
    ``d = max(q) - q``: ``p = exp(-d * s_x) / sum(exp(-d * s_x))`` and the code
    is ``clip(rint(p / s_y), 0, 255)``. The quantized row is not renormalized.

    Measured on an AX8850 at ``[1, 64]`` over six calibrations: every code is
    within 1 of this model and 98.8% are equal; the misses all sit at rounding
    ties. Shifting a row by whole input codes leaves the device output
    bit-identical, which is what the distance form predicts
    (``docs/axera-misc-op-record-emit.md``, "Softmax semantics")."""
    sx = np.float32(scales["x"])
    q = np.clip(
        np.rint(np.asarray(x, dtype=np.float32) / sx) + int(zero_points["x"]), 0, 255
    )
    dist = q.max(axis=axis, keepdims=True) - q
    t = np.exp(-dist.astype(np.float64) * float(sx))
    p = t / t.sum(axis=axis, keepdims=True) / float(scales["y"])
    return np.clip(np.rint(p), 0, 255).astype(np.int64)


def _retarget_log_table(words, old_sc, new_sc, zp) -> list[bytes]:
    """Rewrite the ``0x1050..0x1850`` table (two u16 entries per record). Each
    register is written once; ``0x1850`` (entries 256, 257) comes after the
    unrelated ``0x1860..0x1a50`` block, so records are found by register."""
    at: dict[int, list[int]] = {}
    for j, w in enumerate(words):
        r = _reg(w)
        if (
            LOG_TABLE_BASE <= r < LOG_TABLE_BASE + 0x10 * LOG_TABLE_RECORDS
            and w[0] == 0xA1
        ):
            at.setdefault(r, []).append(j)
    regs = [LOG_TABLE_BASE + 0x10 * k for k in range(LOG_TABLE_RECORDS)]
    if any(len(at.get(r, [])) != 1 for r in regs):
        raise ValueError("expected each Log table register written exactly once")
    old_t, new_t = log_table(old_sc, zp), log_table(new_sc, zp)
    out = list(words)
    for k, r in enumerate(regs):
        j = at[r][0]
        if _val(words[j]) != old_t[2 * k] | (old_t[2 * k + 1] << 16):
            raise ValueError("Log table does not match the template's calibration")
        out[j] = _with_val(words[j], new_t[2 * k] | (new_t[2 * k + 1] << 16))
    return out


def _retarget_input_zp(words, kinds, old_zx: int, new_zx: int) -> list[bytes]:
    """Softmax: the one ``0x1b10 = zp_x`` write before its first lane run."""
    if old_zx == new_zx:
        return list(words)
    if 0 in (old_zx, new_zx):
        raise ValueError("zp_x = 0 is not measured (0x1b10 write omission)")
    first = kinds[0][0]
    hits = [
        j for j in range(first) if _reg(words[j]) == ZP_IN and _val(words[j]) == old_zx
    ]
    if len(hits) != 1:
        raise ValueError(f"expected one 0x1b10 = zp_x write, found {len(hits)}")
    out = list(words)
    out[hits[0]] = _with_val(words[hits[0]], new_zx)
    return out


def _retarget_shared_zp(words, old_zp, new_zp) -> list[bytes]:
    """MaxPool: input and output share one zero point (MaxPool is passive).
    A nonzero one is written as whole words to ``0x1b10`` and ``0x1a90``
    (zero writes are elided, so 0 is a different program); rewrite both."""
    old, new = int(old_zp["x"]), int(new_zp["x"])
    if int(old_zp["y"]) != old or int(new_zp["y"]) != new:
        raise ValueError("MaxPool input and output share one zero point")
    if 0 in (old, new):
        raise ValueError("a MaxPool zero point of 0 is a different program")
    hits = [
        j for j, w in enumerate(words) if _reg(w) in (ZP_IN, ZP_ACC) and _val(w) == old
    ]
    if {_reg(words[j]) for j in hits} != {ZP_IN, ZP_ACC}:
        raise ValueError(f"expected 0x1b10 and 0x1a90 = {old} writes, found {hits}")
    out = list(words)
    for j in hits:
        out[j] = _with_val(words[j], new)
    return out


NEG_ZP_REGS = (ZP_ACC, 0x1AD0, ZP_IN)


def _in_block(words, j: int) -> bool:
    """Record ``j`` is part of a register-block dump (consecutive registers
    ``0x10`` apart); Pulsar2 writes those whole, never eliding a record."""
    r = _reg(words[j])
    return (j > 0 and _reg(words[j - 1]) == r - 0x10) or (
        j + 1 < len(words) and _reg(words[j + 1]) == r + 0x10
    )


def _retarget_neg_zps(words, old_zp, new_zp) -> list[bytes]:
    """Neg's zero-point writes on ``0x1a90``/``0x1ad0``/``0x1b10``.

    Each such record holds ``zp_x``, ``zp_y`` or 0. A write outside a register
    block is omitted when the register already holds the value (registers start
    at 0), so the template must be one where no write was omitted: both zero
    points nonzero and distinct."""
    zx0, zy0 = int(old_zp["x"]), int(old_zp["y"])
    if 0 in (zx0, zy0) or zx0 == zy0:
        raise ValueError("a Neg template needs distinct nonzero zero points")
    subst = {zx0: int(new_zp["x"]), zy0: int(new_zp["y"]), 0: 0}
    state = dict.fromkeys(NEG_ZP_REGS, 0)
    out = []
    for j, w in enumerate(words):
        r = _reg(w)
        if r not in NEG_ZP_REGS or w[0] != 0xA1:
            out.append(w)
            continue
        if _val(w) not in subst:
            raise ValueError(f"{r:#06x} holds {_val(w)}, not a zero point or 0")
        v = subst[_val(w)]
        if _in_block(words, j) or state[r] != v:
            out.append(_with_val(w, v))
        state[r] = v
    return out


def neg_program(scale: float) -> str:
    """Which of Pulsar2's two Neg programs a calibration compiles to."""
    return "small" if _f32(scale) < NEG_PROGRAM_SCALE else "large"


def _zp_in_word(words) -> bytes:
    """A ``0x1b10`` record of the stream's own verb/unit, value to be set."""
    for w in words:
        if _reg(w) == ZP_IN:
            return w
    raise ValueError("stream has no 0x1b10 record to copy")


# Zero-point register writes (verb 0xa1) between the previous calibration lane
# run and each lane run, in order: (register, zero-point key or the constant 0).
_ZP_HEAD = ((ZP_ACC, 0), (ZP_AUX, 0))  # part of the 0x1a60.. block dump
ZP_STAGE_WRITES = {
    "Sigmoid": {"1/s_x": (*_ZP_HEAD, (ZP_IN, "x")), "s_y": ((ZP_IN, 0),)},
    "Mul": {
        "1/s_a": (*_ZP_HEAD, (ZP_IN, "a")),
        "1/s_b": (),
        "s_y/(s_a*s_b)": ((ZP_ACC, "a"), (ZP_AUX, "b"), (ZP_IN, "y")),
        "s_y": ((ZP_ACC, "y"), (ZP_AUX, 0), (ZP_IN, 0)),
    },
    "Add": {
        "1/s_a": (*_ZP_HEAD, (ZP_IN, "a")),
        "1/s_b": (),
        "s_y": ((ZP_ACC, "y"), (ZP_IN, 0)),
    },
    "Div": {
        "1/s_a": (*_ZP_HEAD, (ZP_IN, "a")),
        "1/s_b": ((ZP_IN, "b"),),
        "s_a/(s_b*s_y)": ((ZP_ACC, "a"), (ZP_IN, "y")),
        "s_y": ((ZP_ACC, "y"), (ZP_IN, 0)),
    },
}


def _retarget_ordered_lanes(
    words, op, old, new
) -> tuple[list[bytes], list[tuple[int, str]]]:
    """Sigmoid/Mul/Add/Div: the lane runs holding the template's values must
    come in the op's program order, one run per formula. Matching by order
    rather than by value alone lets two formulas coincide (``s_a == s_b``)."""
    old_v, new_v = lane_values(op, old), lane_values(op, new)
    order = list(old_v)
    out = list(words)
    kinds: list[tuple[int, str]] = []
    for i, v in lane_runs(words):
        if len(kinds) < len(order) and v == old_v[order[len(kinds)]]:
            kind = order[len(kinds)]
            kinds.append((i, kind))
            for j in range(i, i + LANES):
                out[j] = _with_val(out[j], new_v[kind])
        elif kinds and v in old_v.values():
            raise ValueError(f"{op} lane runs are not in the measured order {order}")
    if kinds and len(kinds) != len(order):
        raise ValueError(f"{op}: found lane runs {kinds}, expected {order}")
    return out, kinds


def sigmoid_table(
    scales: Mapping[str, float], zero_points: Mapping[str, int]
) -> list[int]:
    """Sigmoid's u8 lookup table in Log's 258-entry layout: ``clip(rint(
    sigmoid((q - zp_x) * s_x) / s_y) + zp_y, 0, 255)`` for ``q`` in 0..255,
    then entry 255 repeated and a 0. Equal to the device on all 19,200 codes
    of three calibrations (``docs/axera-misc-op-record-emit.md``)."""
    sx, sy = _f32(scales["x"]), _f32(scales["y"])
    q = np.arange(256, dtype=np.float64) - int(zero_points["x"])
    v = np.rint(1.0 / (1.0 + np.exp(-q * sx)) / sy) + int(zero_points["y"])
    v = np.clip(v, 0, 255).astype(int).tolist()
    return v + [v[255], 0]


def add_q15(scales: Mapping[str, float]) -> tuple[int, int]:
    """Add's two input weights, ``round(s_a/s_y * 2**15)`` and ``round(s_b/s_y
    * 2**15)``: the first two little-endian u16 of ``npu_params``."""
    sa, sb, sy = (_f32(scales[k]) for k in ("a", "b", "y"))
    return int(np.rint(sa / sy * 2**ADD_Q15)), int(np.rint(sb / sy * 2**ADD_Q15))


def add_offset(scales: Mapping[str, float], zero_points: Mapping[str, int]) -> int:
    """Add's int32 offset on ``0x1ef0..0x1f20``: ``trunc((zp_y - zp_a * r_a -
    zp_b * r_b) * 2**15)`` with ``r = float32(s / s_y)``."""
    sa, sb, sy = (_f32(scales[k]) for k in ("a", "b", "y"))
    ra, rb = _f32(sa / sy), _f32(sb / sy)
    zp = {k: int(zero_points[k]) for k in ("a", "b", "y")}
    return int(np.trunc((zp["y"] - zp["a"] * ra - zp["b"] * rb) * 2**ADD_Q15))


def _check_staged_target(op, old_scales, new_scales, old_zp, new_zp) -> None:
    """Refuse what no Sigmoid/Mul/Add/Div build measured."""
    keys = ("a", "b", "y") if op in TWO_INPUT else ("x", "y")
    for zp in (old_zp, new_zp):
        if set(zp) != set(keys):
            raise ValueError(f"{op} needs zero points {keys}, got {sorted(zp)}")
        if op in ("Mul", "Add") and int(zp["a"]) != int(zp["b"]):
            raise ValueError(f"{op} with zp_a != zp_b is not measured")
        if op == "Div" and int(zp["b"]) != 0:
            raise ValueError("Div with zp_b != 0 is not measured")
        if op == "Sigmoid" and int(zp["y"]) != 0:
            raise ValueError("Sigmoid with zp_y != 0 is not measured")
    for k in keys:
        if (int(old_zp[k]) == 0) != (int(new_zp[k]) == 0):
            raise ValueError(
                f"{op} zp_{k} {old_zp[k]} -> {new_zp[k]}: a zero point of 0 on one "
                "side only is not measured (write elision)"
            )
    if op == "Add" and add_q15(old_scales) != add_q15(new_scales):
        raise ValueError(
            f"Add Q15 weights {add_q15(new_scales)} differ from the template's "
            f"{add_q15(old_scales)}; npu_params is not rewritten"
        )


def _repeated_writes(writes: Sequence[tuple[int, int]]) -> list[bool]:
    """Which writes leave their register's value unchanged (registers start at 0)."""
    state: dict[int, int] = {}
    flags = []
    for reg, v in writes:
        flags.append(state.get(reg, 0) == v)
        state[reg] = v
    return flags


def _retarget_staged_zps(words, kinds, op, old_zp, new_zp) -> list[bytes]:
    """Sigmoid/Mul/Add/Div zero points: each lane run's ``ZP_STAGE_WRITES``
    records are checked against the template's zero points and rewritten."""
    out = list(words)
    old_w: list[tuple[int, int]] = []
    new_w: list[tuple[int, int]] = []
    prev = 0
    for i, kind in kinds:
        want = ZP_STAGE_WRITES[op][kind]
        at = [
            j
            for j in range(prev, i)
            if words[j][0] == 0xA1 and _reg(words[j]) in (ZP_ACC, ZP_AUX, ZP_IN)
        ]
        if [_reg(words[j]) for j in at] != [reg for reg, _ in want]:
            raise ValueError(f"{op} {kind} stage has an unmeasured zero-point layout")
        for j, (reg, key) in zip(at, want):
            old = int(old_zp[key]) if key else 0
            new = int(new_zp[key]) if key else 0
            if _val(words[j]) != old:
                raise ValueError(
                    f"{op} {kind} stage: {reg:#06x} holds {_val(words[j])}, "
                    f"not the template's {old}"
                )
            out[j] = _with_val(words[j], new)
            old_w.append((reg, old))
            new_w.append((reg, new))
        prev = i + LANES
    if any(w[0] == 0xA1 and _reg(w) in (ZP_ACC, ZP_AUX, ZP_IN) for w in words[prev:]):
        raise ValueError(f"{op} writes a zero-point register after its last lane run")
    if _repeated_writes(old_w) != _repeated_writes(new_w):
        raise ValueError(
            f"{op}: a zero-point write would repeat its register's value where the "
            "template's does not (or the reverse); that elision is not measured"
        )
    return out


def _retarget_add_offset(words, old: int, new: int) -> list[bytes]:
    """Add: the four ``0x1ef0..0x1f20`` records right after ``0x1ea0 = 15``."""
    regs = [ADD_OFFSET_BASE + 0x10 * k for k in range(ADD_OFFSET_LANES)]
    hits = [
        j + 1
        for j, w in enumerate(words)
        if _reg(w) == ADD_SHIFT
        and _val(w) != 0
        and [_reg(x) for x in words[j + 1 : j + 1 + ADD_OFFSET_LANES]] == regs
    ]
    if len(hits) != 1 or _val(words[hits[0] - 1]) != ADD_Q15:
        raise ValueError("expected one Add offset block after a 0x1ea0 = 15 write")
    out = list(words)
    for j in range(hits[0], hits[0] + ADD_OFFSET_LANES):
        if _val(words[j]) != old & 0xFFFFFFFF:
            raise ValueError("Add offset does not match the template's calibration")
        out[j] = _with_val(words[j], new & 0xFFFFFFFF)
    return out


def _retarget_u8_table(words, old_t, new_t, what: str) -> list[bytes]:
    """Rewrite a 258-entry table in Log's layout (``_retarget_log_table``):
    two u16 entries per record on ``0x1050..0x1850``, found by register."""
    at: dict[int, list[int]] = {}
    for j, w in enumerate(words):
        r = _reg(w)
        if (
            LOG_TABLE_BASE <= r < LOG_TABLE_BASE + 0x10 * LOG_TABLE_RECORDS
            and w[0] == 0xA1
        ):
            at.setdefault(r, []).append(j)
    regs = [LOG_TABLE_BASE + 0x10 * k for k in range(LOG_TABLE_RECORDS)]
    if any(len(at.get(r, [])) != 1 for r in regs):
        raise ValueError(f"expected each {what} table register written exactly once")
    out = list(words)
    for k, r in enumerate(regs):
        j = at[r][0]
        if _val(words[j]) != old_t[2 * k] | (old_t[2 * k + 1] << 16):
            raise ValueError(f"{what} table does not match the template's calibration")
        out[j] = _with_val(words[j], new_t[2 * k] | (new_t[2 * k + 1] << 16))
    return out


def _retarget_staged(
    words, kinds, op, old_scales, new_scales, old_zp, new_zp
) -> list[bytes]:
    """Everything but the lanes of a Sigmoid/Mul/Add/Div segment."""
    new = _retarget_staged_zps(words, kinds, op, old_zp, new_zp)
    if op == "Sigmoid":
        new = _retarget_u8_table(
            new,
            sigmoid_table(old_scales, old_zp),
            sigmoid_table(new_scales, new_zp),
            "Sigmoid",
        )
    elif op == "Add":
        new = _retarget_add_offset(
            new, add_offset(old_scales, old_zp), add_offset(new_scales, new_zp)
        )
    return new


def retarget(
    mc: bytes,
    op: str,
    old_scales: Mapping[str, float],
    new_scales: Mapping[str, float],
    old_zero_points: Mapping[str, int] | None = None,
    new_zero_points: Mapping[str, int] | None = None,
    reduce_count: int | None = None,
) -> bytes:
    """``mc`` with its calibration moved from ``old_*`` to ``new_*``.

    ReduceSum and ReduceMean may change both zero points, Softmax its ``zp_x``
    and Sigmoid/Mul/Add/Div theirs within ``_check_staged_target``; every
    other op keeps its template's."""
    if op not in CALIBRATED:
        raise ValueError(f"no calibration edit for {op!r}")
    old_zp = dict(old_zero_points or {})
    new_zp = dict(new_zero_points or old_zp)
    changed = {k for k in old_zp if int(new_zp.get(k, old_zp[k])) != int(old_zp[k])}
    movable = {
        "ReduceSum": {"x", "y"},
        "Softmax": {"x"},
        "MaxPool": {"x", "y"},
        "Neg": {"x", "y"},
        "ReduceMean": {"x", "y"},
        "Sigmoid": {"x", "y"},
        "Mul": {"a", "b", "y"},
        "Add": {"a", "b", "y"},
        "Div": {"a", "b", "y"},
    }.get(op, set())
    if op in STAGED:
        _check_staged_target(op, old_scales, new_scales, old_zp, new_zp)
    if op == "Neg":
        if int(new_zp["y"]) != 255 - int(new_zp["x"]):
            raise ValueError("Neg's output zero point is 255 - zp_x")
        if neg_program(old_scales["x"]) != neg_program(new_scales["x"]):
            raise ValueError("the target scale compiles to the other Neg program")
    if changed - movable:
        raise ValueError(
            f"{op} zero points {sorted(changed - movable)} are fixed by the template"
        )
    segs = suc.decode_segments(mc)
    found = 0
    for si, raw in enumerate(segs):
        words = _chunks(raw)
        new, kinds = _retarget_lanes(words, op, old_scales, new_scales, reduce_count)
        if not kinds:
            continue
        found += len(kinds)
        if op == "Log":
            new = _retarget_log_table(new, old_scales, new_scales, old_zp)
        elif op == "Softmax" and old_zp:
            new = _retarget_input_zp(new, kinds, int(old_zp["x"]), int(new_zp["x"]))
        elif op == "MaxPool" and old_zp != new_zp:
            new = _retarget_shared_zp(new, old_zp, new_zp)
        elif op in STAGED:
            new = _retarget_staged(
                new, kinds, op, old_scales, new_scales, old_zp, new_zp
            )
        if op == "ReduceSum" and old_zp and old_zp != new_zp:
            pad = _pad_count(words)
            new = _retarget_reducesum_zps(new[: len(new) - pad], kinds, old_zp, new_zp)
            new = _retarget_packed_zp(new, int(old_zp["x"]), int(new_zp["x"]))
            new += [bytes(RECORD)] * (-len(new) % PAD_GROUP)
        elif op == "Neg":
            pad = _pad_count(words)
            new = _retarget_neg_zps(new[: len(new) - pad], old_zp, new_zp)
            new += [bytes(RECORD)] * (-len(new) % PAD_GROUP)
        elif op == "ReduceMean" and old_zp and old_zp != new_zp:
            # ReduceSum's three stages; the requantize 0x1a90 is zp_x * N
            if any(_reg(w) == ZP_PACKED_BASE for w in words):
                raise ValueError("ReduceMean with packed zp_x lanes is not measured")
            stages = [(i, k.replace("s_x/(s_y*N)", "s_x/s_y")) for i, k in kinds]
            pad = _pad_count(words)
            new = _retarget_reducesum_zps(new[: len(new) - pad], stages, old_zp, new_zp)
            new += [bytes(RECORD)] * (-len(new) % PAD_GROUP)
        new_raw = b"".join(new)
        if new_raw != raw:
            mc = bose.relayout_segment(mc, si, new_raw)
    if not found:
        raise ValueError("no scale lanes of the template calibration were found")
    return mc


@lru_cache(maxsize=None)
def load_index(path: str = TEMPLATE_INDEX) -> dict:
    with open(path) as f:
        return json.load(f)


def template_key(op: str, shape, axes=None, keepdims=None) -> str:
    key = f"{op}:{'x'.join(str(int(d)) for d in shape)}"
    if op == "ReduceSum":
        key += f":axes{','.join(str(int(a)) for a in axes)}:k{int(keepdims)}"
    return key


def load_template(key: str, index_path: str = TEMPLATE_INDEX):
    """``(model, meta)`` for the committed template ``key``."""
    index = load_index(index_path)
    if key not in index:
        raise ValueError(f"no validated template for {key}; have {sorted(index)}")
    meta = index[key]
    with gzip.open(os.path.join(FIXTURES, meta["file"]), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    return model, meta


def emit_model(
    key: str,
    scales: Mapping[str, float] | None = None,
    zero_points: Mapping[str, int] | None = None,
    index_path: str = TEMPLATE_INDEX,
) -> onnx.ModelProto:
    """A compiled model for template ``key`` at the given calibration.

    Greater/Less -> Cast takes no calibration: its template is returned as built."""
    key = equivalent_key(key, index_path) or key
    model, meta = load_template(key, index_path)
    if meta["op"] == "Neg" and scales:
        # one template per program; the target scale picks the program
        want = neg_program(scales["x"])
        if want != meta["program"]:
            key = meta["programs"][want]
            model, meta = load_template(key, index_path)
    if meta["op"] in CALIBRATION_FREE:
        if scales or zero_points:
            raise ValueError(f"{meta['op']} is not quantized; no calibration")
        return model
    import step_recalibrate

    mc = retarget(
        bytes(mcode_initializer(model).raw_data),
        meta["op"],
        meta["scales"],
        scales or meta["scales"],
        meta["zero_points"],
        zero_points or meta["zero_points"],
        meta.get("reduce_count"),
    )
    # a zero-point move can change the blob length: the runtime reads the
    # MCode size from the initializer's dims (0x80300709 on load otherwise)
    return step_recalibrate.with_mcode(model, mc)


def emit_spec(
    op: str,
    shape,
    *,
    axes: Sequence[int] | None = None,
    keepdims: int | None = None,
    scales: Mapping[str, float] | None = None,
    zero_points: Mapping[str, int] | None = None,
    attrs: Mapping[str, Sequence[int] | int] | None = None,
) -> onnx.ModelProto:
    """Emit a measured MCode model from an operator specification.

    This is the generic dispatch boundary used by the Pulsar-free scheduler:
    shape and reduction attributes become a template key, while MCode bytes
    and calibration retargeting remain owned by the validated record emitter.
    It deliberately refuses a missing exact template instead of inventing
    tile counts or register records.
    """
    shape = tuple(int(dim) for dim in shape)
    if op in ("ReduceSum", "ReduceMean"):
        if axes is None or keepdims is None:
            raise ValueError(f"{op} spec requires axes and keepdims")
        axes = tuple(sorted(int(axis) % len(shape) for axis in axes))
        if len(set(axes)) != len(axes):
            raise ValueError(f"{op} spec axes must be unique")
        if any(axis < 0 or axis >= len(shape) for axis in axes):
            raise ValueError(f"{op} spec axis is outside the input rank")
        key = template_key("ReduceSum", shape, axes, int(keepdims))
        if op == "ReduceMean":
            key = key.replace("ReduceSum", "ReduceMean", 1)
    elif op in CALIBRATION_FREE:
        key = f"{op}:{'x'.join(str(dim) for dim in shape)}"
    elif op == "MaxPool":
        if not attrs:
            raise ValueError("MaxPool spec requires kernel_shape, strides and pads")
        kernel = "x".join(str(int(v)) for v in attrs["kernel_shape"])
        strides = "x".join(str(int(v)) for v in attrs["strides"])
        pads = ",".join(str(int(v)) for v in attrs["pads"])
        key = f"MaxPool:{'x'.join(str(dim) for dim in shape)}:k{kernel}:s{strides}:p{pads}"
    elif op == "Softmax":
        if attrs is None or "axis" not in attrs:
            raise ValueError("Softmax spec requires axis")
        axis = int(attrs["axis"]) % len(shape)
        key = f"Softmax:{'x'.join(str(dim) for dim in shape)}:axis{axis}"
    else:
        key = f"{op}:{'x'.join(str(dim) for dim in shape)}"
    return emit_model(key, scales, zero_points)


# ReduceSum nodes Pulsar2 cannot tile ("Can not tile", also inside the step's
# own Reshape -> ReduceSum -> Reshape chain) whose reduction, over the same
# contiguous bytes with different shape labels, does compile: [16,1,64,12544]
# over axes (0,3) is [16,64,112,112] over axes (0,2,3), output [1,64] == [64].
REDUCESUM_EQUIVALENTS = {
    "ReduceSum:16x1x64x12544:axes0,3:k0": "ReduceSum:16x64x112x112:axes0,2,3:k0",
}


def equivalent_key(key: str, index_path: str = TEMPLATE_INDEX) -> str | None:
    """The validated template that computes ``key`` on the same bytes, else ``None``."""
    alt = REDUCESUM_EQUIVALENTS.get(key)
    return alt if alt is not None and alt in load_index(index_path) else None


def normalized_records(mc: bytes) -> list[list[bytes]]:
    """Decompressed records per segment, segment 0's slot table dropped."""
    segs = [_chunks(s) for s in suc.decode_segments(mc)]
    if segs:
        segs[0] = [w for i, w in enumerate(segs[0]) if i not in SEG0_NOISE]
    return segs


def slot_normalized_records(mc: bytes) -> list[list[bytes]]:
    """``normalized_records`` with the slot ids of the ``0x03d0``/``0x02b0``
    select records (verb 0xa8) renumbered by first appearance, 0 kept.

    Pulsar2 may bind a two-input op's inputs to segment 0's slots in either
    order (Add and Div ``[1, 64]``: the ``asym`` builds swap slots 1,2 and
    3,4 against the template). The ids pair with segment 0's slot table, which
    ``normalized_records`` already drops as per-build noise."""
    segs = normalized_records(mc)
    for words in segs:
        ids: dict[int, int] = {0: 0}
        for j, w in enumerate(words):
            if w[0] == SLOT_VERB and _reg(w) in SLOT_REGS:
                words[j] = _with_val(w, ids.setdefault(_val(w), len(ids)))
    return segs


STEP_OPS = (
    "ReduceSum",
    "Sqrt",
    "Greater",
    "Less",
    "Cast",
    "Softmax",
    "Log",
    "MaxPool",
    "ReduceMean",
    "Neg",
)


def step_node_keys(onnx_path: str) -> list[tuple[str, str]]:
    """``(op, template key)`` for each node of a real graph whose op is in
    ``STEP_OPS``. Greater/Less feed a Cast in the step, so both nodes of a pair
    share the pair's key."""
    from onnx import numpy_helper, shape_inference

    model = shape_inference.infer_shapes(onnx.load(onnx_path, load_external_data=False))
    shapes = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in list(model.graph.value_info)
        + list(model.graph.input)
        + list(model.graph.output)
    }
    inits = {i.name: i for i in model.graph.initializer}
    producer = {o: n for n in model.graph.node for o in n.output}
    keys = []
    for node in model.graph.node:
        op = node.op_type
        if op not in STEP_OPS:
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        shape = shapes.get(node.input[0], [])
        if op == "ReduceSum":
            axes = attrs.get("axes")
            if axes is None and len(node.input) > 1 and node.input[1]:
                src = inits.get(node.input[1])
                const = producer.get(node.input[1])
                if src is None and const is not None and const.op_type == "Constant":
                    src = onnx.helper.get_attribute_value(const.attribute[0])
                if src is None:
                    raise ValueError(f"ReduceSum {node.name!r} has non-constant axes")
                axes = numpy_helper.to_array(src).ravel().tolist()
            axes = sorted(int(a) % len(shape) for a in axes or range(len(shape)))
            keys.append((op, template_key(op, shape, axes, attrs.get("keepdims", 1))))
        elif op in ("Sqrt", "Log", "Neg"):
            keys.append((op, template_key(op, shape)))
        elif op == "Softmax":
            axis = int(attrs.get("axis", -1)) % len(shape)
            keys.append((op, f"{template_key(op, shape)}:axis{axis}"))
        elif op == "ReduceMean":
            axes = sorted(
                int(a) % len(shape) for a in attrs.get("axes", range(len(shape)))
            )
            keys.append(
                (
                    op,
                    template_key(
                        "ReduceSum", shape, axes, attrs.get("keepdims", 1)
                    ).replace("ReduceSum", op, 1),
                )
            )
        elif op == "MaxPool":
            k = "x".join(map(str, attrs["kernel_shape"]))
            st = "x".join(
                map(str, attrs.get("strides", [1] * len(attrs["kernel_shape"])))
            )
            pd = ",".join(
                map(str, attrs.get("pads", [0] * 2 * len(attrs["kernel_shape"])))
            )
            keys.append((op, f"{template_key(op, shape)}:k{k}:s{st}:p{pd}"))
        elif op in ("Greater", "Less"):
            keys.append((op, f"{op}Cast:{'x'.join(map(str, shape))}"))
        elif op == "Cast":
            src = producer.get(node.input[0])
            if src is not None and src.op_type in ("Greater", "Less"):
                s = shapes.get(src.input[0], [])
                keys.append((op, f"{src.op_type}Cast:{'x'.join(map(str, s))}"))
            else:
                keys.append((op, f"Cast:{'x'.join(map(str, shape))}"))
    return keys


ELEMENTWISE_INDEX = os.path.join(FIXTURES, "elementwise_scale_emit", "index.json")


def coverage(onnx_path: str, index_path: str = TEMPLATE_INDEX) -> dict:
    """Per op: nodes, nodes with a committed template, and missing keys.

    Sqrt counts ``elementwise_scale_emit``'s zero-point-0 templates too (the
    step's Sqrt nodes calibrate to ``zp = 0``); this module adds the one shape
    that module left out."""
    index = set(load_index(index_path))
    if os.path.exists(ELEMENTWISE_INDEX):
        index |= {
            k.rsplit(":", 1)[0]
            for k in load_index(ELEMENTWISE_INDEX)
            if k.startswith("Sqrt:") and k.endswith(":x0,y0")
        }
    res: dict = {}
    for op, key in step_node_keys(onnx_path):
        r = res.setdefault(op, {"nodes": 0, "covered": 0, "missing": {}})
        r["nodes"] += 1
        if key in index or equivalent_key(key, index_path):
            r["covered"] += 1
        else:
            r["missing"][key] = r["missing"].get(key, 0) + 1
    return res


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step_onnx", help="real training-step graph for a coverage report")
    args = p.parse_args(argv)
    print(json.dumps(coverage(args.step_onnx), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
