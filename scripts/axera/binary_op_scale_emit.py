"""Retarget a compiled standalone AX650 binary elementwise op (Add, Sub, Mul,
Div; two live inputs of the same shape) to new quantization scales without
Pulsar2.

This works on the *decompressed* MCode. ``short_unit_codec.py`` (PR #1850)
showed that each MCode segment is an LZ77 token stream over plain 8-byte
register records ``[verb][00][field][bank][value32]``. A calibration change at
fixed zero points then touches only whole records (measured on held-out builds,
``docs/axera-binary-op-scale-emit.md``):

* Each scale-derived float32 is written once per output lane (8 lanes) per
  tile block, on registers ``0x0f50..0x0fc0`` (``0x0fd0..0x1040`` for Mul's
  divisor); the block count depends on the shape, not the calibration:

  * all four ops: ``1/s_x``, ``1/s_z`` and ``s_y``;
  * Mul additionally ``s_y/(s_x*s_z)``;
  * Div additionally ``s_x/(s_y*s_z)``.

  Each value is Pulsar2's float32 scale combined in float64 and rounded to
  float32, the same arithmetic ``teng_register_census.scale_monomials`` uses.
* Add and Sub also carry a Q15 requant header at the start of ``npu_params``:
  ``round(r * 2**(15 - k))`` for ``r = s_x/s_y`` and ``r = s_z/s_y`` as
  little-endian uint16, where ``k >= 0`` is the smallest shift with both
  ``r * 2**-k < 1`` (so a ratio just below 1 stores 32768; one word when both
  round equal). Register ``0x1ea0`` in
  the TENG segment holds ``15 - k``. PR #1756's ``round(r * 32768)`` is the
  ``k = 0`` case, valid while both ratios are below 1. Registers
  ``0x1ef0..0x1f20`` hold the int32 zero-point offset (``zp_offset``).
* A re-encoded stream that pads to a different size is relaid out
  (``relayout_segment``), as Pulsar2 itself does.

What is **not** calibration: the compiled node's input order (``x, z`` or
``z, x``) is Pulsar2's own per-build choice. It moves the small slot numbers at
registers ``0x03d0``/``0x02b0`` and permutes segment 0's slot table, which is
rebuild noise (``docs/axera-teng-register-census.md``). An emitted model keeps
its template's order, which is self-consistent; validation compares against
native builds with segment 0 excluded and the input order normalized
(``normalized_records``).

Scope: zero points are fixed (a template serves the zero points it was built
with), both inputs are live and have the same shape, and a template only serves
targets whose float equalities (which formula values coincide) and Q15 header
width match its own. Anything else raises ``ValueError``.
"""

from __future__ import annotations

import gzip
import json
import os
import struct
import sys
from collections import Counter
from collections.abc import Mapping

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import short_unit_codec as codec  # noqa: E402

OPS = ("Add", "Sub", "Mul", "Div")
LANES = 8
SCALE_REGS = tuple(range(0x0F50, 0x0FD0, 0x10))
DIVISOR_REGS = tuple(range(0x0FD0, 0x1050, 0x10))
SHIFT_REG = 0x1EA0
OFFSET_REGS = tuple(range(0x1EF0, 0x1F30, 0x10))
# input-order slot numbers: the two inputs' slots trade places when the node's
# inputs are (z, x); the output slot (0x03d0=6) and zero entries stay put
ORDER_SWAP = {0x03D0: {2: 4, 4: 2}, 0x02B0: {1: 3, 3: 1}}
TEMPLATE_DIR = os.path.join(_HERE, "fixtures", "binary_op_scale_emit")


def _f32(x: float) -> float:
    return float(np.float32(x))


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _scales(scales: Mapping[str, float]) -> tuple[float, float, float]:
    out = []
    for name in ("x", "z", "y"):
        if name not in scales:
            raise ValueError(f"missing scale for tensor {name!r}")
        s = _f32(scales[name])
        if not (s > 0 and np.isfinite(s)):
            raise ValueError(f"scale {name!r} must be a positive finite float")
        out.append(s)
    return out[0], out[1], out[2]


def op_values(op: str, scales: Mapping[str, float]) -> list[tuple[str, int, tuple]]:
    """``(name, float32 bits, registers)`` for every scale-derived value
    ``op``'s TENG segment writes (one record per register)."""
    if op not in OPS:
        raise ValueError(f"op {op!r} is not supported; supported: {OPS}")
    sx, sz, sy = _scales(scales)
    vals = [
        ("1/s_x", _bits(1.0 / sx), SCALE_REGS),
        ("1/s_z", _bits(1.0 / sz), SCALE_REGS),
    ]
    # float64 products in the census's order (sorted tensor names x, y, z)
    if op == "Mul":
        vals.append(("s_y/(s_x*s_z)", _bits(sx**-1 * sy * sz**-1), DIVISOR_REGS))
    if op == "Div":
        vals.append(("s_x/(s_y*s_z)", _bits(sx * sy**-1 * sz**-1), SCALE_REGS))
    vals.append(("s_y", _bits(sy), SCALE_REGS))
    return vals


def q15_header(op: str, scales: Mapping[str, float]) -> tuple[list[int], int]:
    """``(words, k)``: Add/Sub's npu_params requant words and shift."""
    if op not in ("Add", "Sub"):
        raise ValueError(f"{op} has no Q15 requant header")
    sx, sz, sy = _scales(scales)
    ratios = (sx / sy, sz / sy)
    # k is picked on the ratio itself, not on the rounded word: a ratio just
    # below 1 keeps k = 0 and stores 32768 (census sub_asym/sub_mix/sub_r3)
    k = 0
    while max(ratios) * 2.0**-k >= 1.0:
        k += 1
        if k > 15:
            raise ValueError(f"scale ratios {ratios} do not fit Q15 at any shift")
    words = [int(round(r * 2.0 ** (15 - k))) for r in ratios]
    if words[0] == words[1]:
        words = words[:1]
    return words, k


def zp_offset(
    op: str, scales: Mapping[str, float], zero_points: Mapping[str, int]
) -> int:
    """Add/Sub's int32 zero-point offset on registers ``0x1ef0..0x1f20``:
    ``z_y - z_x*r_x -/+ z_z*r_z`` in Q15, truncated, with ``r = s/s_y``
    rounded to float32 (fits all 38 measured Add/Sub builds)."""
    if op not in ("Add", "Sub"):
        raise ValueError(f"{op} has no zero-point offset register")
    sx, sz, sy = _scales(scales)
    sign = -1 if op == "Sub" else 1
    c = (
        int(zero_points["y"])
        - int(zero_points["x"]) * _f32(sx / sy)
        - sign * int(zero_points["z"]) * _f32(sz / sy)
    )
    return int(c * 32768)


def _header_bytes(words: list[int]) -> bytes:
    return b"".join(struct.pack("<H", w) for w in words)


def _mcode_init(model: onnx.ModelProto):
    m = [i for i in model.graph.initializer if i.name.endswith("_neu")]
    if len(m) != 1:
        raise ValueError(f"expected one *_neu MCode initializer, found {len(m)}")
    return m[0]


def _params_init(model: onnx.ModelProto):
    m = [i for i in model.graph.initializer if i.name == "npu_params"]
    if len(m) != 1:
        raise ValueError("expected one npu_params initializer")
    return m[0]


def _value_slots(raw: bytes, bits: int, regs: tuple) -> list[int]:
    """Record indices in decompressed segment ``raw`` whose value is ``bits``
    and whose register is in ``regs``."""
    want = struct.pack("<I", bits)
    out = []
    for r in range(0, len(raw) - 7, 8):
        w = raw[r : r + 8]
        if w[4:] == want and (w[2] | w[3] << 8) in regs:
            out.append(r)
    return out


def _check_distinct(vals, what: str) -> None:
    bits = [b for _, b, _ in vals]
    if len(set(bits)) != len(bits):
        raise ValueError(
            f"{what} scale-derived values coincide; the compiler emits a "
            "different program for that case (separate template class)"
        )


def _u32(b, o: int) -> int:
    return struct.unpack_from("<I", b, o)[0]


def relayout_segment(mc: bytes, index: int, raw: bytes) -> bytes:
    """``mc`` with compressed segment ``index`` re-encoded from ``raw``, also
    when the new stream pads to a different multiple of 32 bytes.

    Measured on native Mul builds whose segment 2 pads to 1312 vs 1344 bytes
    (``docs/axera-binary-op-scale-emit.md``): the segments tile a FlatBuffers
    byte vector, so a size change shifts everything after the segment, and
    Pulsar2 updates exactly

    * the segment's table key 2 (padded words) and key 5 (stream bytes), and
      key 3 (start word) of every later segment;
    * the total segment words (``header - 8``, or ``header - 4`` in larger
      programs) and the byte-vector length in the word before it;
    * every header uoffset (and the root table's negative soffset) whose
      target lies past the segment.

    Tail tables use offsets relative to themselves and move as one block."""
    streams = codec.segment_streams(mc)
    start, _, table, comp = streams[index]
    if not comp:
        raise ValueError(f"segment {index} is not compressed; not measured")
    stream = codec.encode(raw)
    room = 8 * table[2]
    new_room = -(-len(stream) // 32) * 32
    if new_room == room:
        return codec.replace_segment(mc, index, raw)
    delta = new_room - room
    header, segs = codec.mcode.segments(mc)
    ins = start + room  # end of the old slot
    words = sum(s[1] for s in segs) // 8
    # the word count sits at header-8 (a trailing word follows it) or header-4
    wat = next((p for p in (header - 8, header - 4) if _u32(mc, p) == words), None)
    if wat is None or wat < 4:
        raise ValueError("unmeasured header layout (no segment word count)")
    lat = wat - 4
    out = bytearray(mc)
    # header pointers first, while offsets still refer to the old layout
    for o in range(0, lat, 4):
        v = _u32(mc, o)
        if v >= 1 << 31:
            target = o - struct.unpack_from("<i", mc, o)[0]
            if ins <= target < len(mc):
                struct.pack_into(
                    "<i", out, o, struct.unpack_from("<i", mc, o)[0] - delta
                )
        elif ins <= o + v < len(mc):
            struct.pack_into("<I", out, o, v + delta)
    struct.pack_into("<I", out, lat, _u32(mc, lat) + delta)
    struct.pack_into("<I", out, wat, _u32(mc, wat) + delta // 8)
    # tables (stored in reverse stream order)
    n = len(streams)
    for k in range(index, n):
        t = n - 1 - k
        key = 2 if k == index else 3
        if k > index and 3 not in streams[k][2]:
            raise ValueError(f"segment {k} has no start-word field; unmeasured")
        at, width = codec._table_field_offset(mc, t, key)
        cur = int.from_bytes(mc[at : at + width], "little")
        out[at : at + width] = (cur + delta // 8).to_bytes(width, "little")
    at, width = codec._table_field_offset(mc, n - 1 - index, 5)
    out[at : at + width] = len(stream).to_bytes(width, "little")
    # the stream itself, then the shift
    out = out[:start] + stream + bytes(new_room - len(stream)) + out[start + room :]
    out = bytes(out)
    if codec.decode_segments(out)[index] != bytes(raw):
        raise ValueError("relayout self-check failed")
    return out


def retarget(
    mcode_bytes: bytes,
    params: bytes,
    op: str,
    old_scales: Mapping[str, float],
    new_scales: Mapping[str, float],
    zero_points: Mapping[str, int] | None = None,
) -> tuple[bytes, bytes]:
    """``(mcode, npu_params)`` rewritten from ``old_scales`` to ``new_scales``
    at the template's ``zero_points`` (needed for Add and Sub)."""
    old = op_values(op, old_scales)
    new = op_values(op, new_scales)
    _check_distinct(old, "template")
    _check_distinct(new, "target")

    segs = codec.decode_segments(mcode_bytes)
    streams = codec.segment_streams(mcode_bytes)
    edited = {i: bytearray(s) for i, s in enumerate(segs)}
    touched = set()

    def lane_hits(name: str, bits: int, regs: tuple) -> list[tuple[int, int]]:
        """Every record holding ``bits`` on ``regs``. Large shapes repeat a
        value once per tile block, so the count is ``n * len(regs)`` with the
        same ``n`` on every lane register, all in one segment."""
        hits = [
            (i, r)
            for i in range(1, len(segs))
            for r in _value_slots(segs[i], bits, regs)
        ]
        per_reg = Counter(segs[i][r + 2] | segs[i][r + 3] << 8 for i, r in hits)
        if (
            not hits
            or len({i for i, _ in hits}) != 1
            or set(per_reg) != set(regs)
            or len(set(per_reg.values())) != 1
        ):
            raise ValueError(
                f"template lacks the measured structure for {name}: {len(hits)} "
                f"records on {sorted(per_reg)} (expected an equal count on each of "
                f"{len(regs)} lane registers, in one segment)"
            )
        return hits

    def put(hits, value: bytes) -> None:
        for i, r in hits:
            if edited[i][r + 4 : r + 8] != value:
                edited[i][r + 4 : r + 8] = value
                touched.add(i)

    for (name, ob, regs), (_, nb, _) in zip(old, new):
        put(lane_hits(name, ob, regs), struct.pack("<I", nb))

    new_params = params
    if op in ("Add", "Sub"):
        ow, ok = q15_header(op, old_scales)
        nw, nk = q15_header(op, new_scales)
        if len(ow) != len(nw):
            raise ValueError(
                "Q15 header width changes (equal-ratio collapse); separate "
                "template class"
            )
        if params[: 2 * len(ow)] != _header_bytes(ow):
            raise ValueError("template npu_params header does not match its scales")
        new_params = _header_bytes(nw) + params[2 * len(ow) :]
        shift = [
            (i, r)
            for i in range(1, len(segs))
            for r in range(0, len(segs[i]) - 7, 8)
            if segs[i][r + 2] | segs[i][r + 3] << 8 == SHIFT_REG
            and _u32(segs[i], r + 4)
        ]
        if 15 in (ok, nk):
            raise ValueError("a Q15 shift of 15 would leave 0x1ea0 zero; not measured")
        if not shift or any(_u32(segs[i], r + 4) != 15 - ok for i, r in shift):
            raise ValueError(
                f"expected every nonzero 0x{SHIFT_REG:04x} record to hold {15 - ok}"
            )
        put(shift, struct.pack("<I", 15 - nk))

        if zero_points is None:
            raise ValueError(f"{op} needs the template's zero points")
        oc = zp_offset(op, old_scales, zero_points)
        nc = zp_offset(op, new_scales, zero_points)
        if oc == 0 and nc != 0:
            raise ValueError(
                "template zero-point offset is 0, so its lanes cannot be "
                "located; use a template with a nonzero offset"
            )
        if oc != 0:
            put(
                lane_hits("zero-point offset", oc & 0xFFFFFFFF, OFFSET_REGS),
                struct.pack("<i", nc),
            )

    out = mcode_bytes
    for i in sorted(touched):
        if not streams[i][3]:
            raise ValueError(f"segment {i} is not compressed; not measured")
        out = relayout_segment(out, i, bytes(edited[i]))
    return out, new_params


def normalized_records(model: onnx.ModelProto) -> list[bytes]:
    """Decompressed segments 1.. of ``model`` with the input-order slot
    numbers (registers ``0x03d0``/``0x02b0``) rewritten to the ``x, z`` order,
    for comparing builds whose node input order differs. Segment 0 (rebuild
    noise) is dropped."""
    segs = codec.decode_segments(bytes(_mcode_init(model).raw_data))
    swap = list(model.graph.node[0].input[:2]) == ["z", "x"]
    out = []
    for raw in segs[1:]:
        raw = bytearray(raw)
        if swap:
            for r in range(0, len(raw) - 7, 8):
                reg = raw[r + 2] | raw[r + 3] << 8
                if reg in ORDER_SWAP:
                    v = struct.unpack_from("<I", raw, r + 4)[0]
                    struct.pack_into("<I", raw, r + 4, ORDER_SWAP[reg].get(v, v))
        out.append(bytes(raw))
    return out


def _key(op: str, shape, zero_points: Mapping[str, int]) -> str:
    zps = ",".join(f"{k}{int(v)}" for k, v in sorted(zero_points.items()))
    return f"{op}:{'x'.join(str(int(d)) for d in shape)}:{zps}"


def template_shape(shape) -> list[int]:
    """The template shape serving ``shape``. Pulsar2 cannot tile a standalone
    rank-1 binary op (``TileFailException`` in ``AxQuantizedAdd``), so a rank-1
    tensor is served by the ``[1, C]`` template: same elements, same contiguous
    layout, only the declared IO shape differs."""
    shape = [int(d) for d in shape]
    return [1, *shape] if len(shape) == 1 else shape


def load_template(op: str, shape, zero_points, template_dir: str = TEMPLATE_DIR):
    with open(os.path.join(template_dir, "index.json")) as f:
        index = json.load(f)
    key = _key(op, template_shape(shape), zero_points)
    if key not in index:
        raise ValueError(f"no validated template for {key}; have {sorted(index)}")
    meta = index[key]
    with gzip.open(os.path.join(template_dir, meta["file"]), "rb") as f:
        return onnx.load_model_from_string(f.read()), meta


def emit_model(
    model: onnx.ModelProto,
    op: str,
    old_scales: Mapping[str, float],
    new_scales: Mapping[str, float],
    zero_points: Mapping[str, int] | None = None,
) -> onnx.ModelProto:
    """A copy of compiled ``model`` retargeted to ``new_scales``."""
    out = onnx.ModelProto()
    out.CopyFrom(model)
    mc, pa = retarget(
        bytes(_mcode_init(out).raw_data),
        bytes(_params_init(out).raw_data),
        op,
        old_scales,
        new_scales,
        zero_points,
    )
    _mcode_init(out).raw_data = mc
    del _mcode_init(out).dims[:]
    _mcode_init(out).dims.append(len(mc))
    _params_init(out).raw_data = pa
    return out


def emit(
    op: str,
    shape,
    scales: Mapping[str, float],
    zero_points: Mapping[str, int],
    out_path: str,
    template_dir: str = TEMPLATE_DIR,
) -> str:
    """Write a compiled standalone ``op(x[shape], z[shape])`` for Pulsar2
    quantization ``scales``/``zero_points`` (per tensor ``x``, ``z``, ``y``)."""
    model, meta = load_template(op, shape, zero_points, template_dir)
    out = emit_model(model, op, meta["scales"], scales, meta["zero_points"])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    onnx.save(out, out_path)
    return out_path
