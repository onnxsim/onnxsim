"""Recalibrate a compiled live-operand MatMul from its quantization scales,
without a reference build at the target calibration.

The ResNet18 training step's 41 MatMuls, its Gemm and its 20 Convs all have
live weights. That makes them live-operand matrix multiplies, so there is no
weight table to patch. For one shape, the MCode is a fixed register program.
Only the calibration records change. On the decompressed records
(``short_unit_codec``), those are:

* float32 lanes at ``0x0f50..0x1000``. Each lane holds a function of the
  tensor scales, such as ``1/s_B``, ``1/s_A``, ``s_Y``, or a ratio or product
  of scales when the MatMul sits inside a fused chain;
* zero-point records at ``0x1a90``/``0x1b10``;
* float32 words in ``npu_params``. For a live MatMul, these are one lane per
  output column of ``float(zp_Y)`` and one of ``s_A * s_B / s_Y``, padded
  per 16 or per output tile.

``recalibrate`` finds each of these in a *template* build. It uses the
template's own recorded scales (Pulsar2's ``quant_axmodel.json``) and
matches every lane value, bit for bit, against a closed set of scale
formulas (``roles``). It then rewrites each lane with the same formula
evaluated at the new scales, and re-encodes and re-lays out the changed
segments (``step_recalibrate.replace_segments``). It refuses the edit when:

* a calibration record's value matches no formula;
* the formulas that match a value disagree on the new value;
* a zero point would go between zero and nonzero. Pulsar2 then emits a
  different record count (``docs/axera-step-recalibrate-relayout.md``).

``step_recalibrate.recalibrate`` needs a Pulsar2 build at the target
calibration. This module needs one build per *shape*, at any calibration.

The shape side does not generalize to the step. The analysis is in
``docs/axera-matmul-record-emit.md``. Record structure changes with every
tile count. The 152 vendor MatMul models fall into 139 structures. In the
small sweeps, values inside one structure are exact functions of M or K,
but with thresholds, and none of the sweeps reaches a step shape. So every
distinct step shape needs one build (``STEP_SHAPES``), and after that its
calibration comes from this module.

Usage::

    matmul_record_emit.py check TEMPLATE_DIR TARGET_DIR   # recalibrate, compare
    matmul_record_emit.py emit TEMPLATE_DIR QUANT.json OUT.axmodel
    matmul_record_emit.py explain TEMPLATE_DIR            # role of every lane

A build directory is Pulsar2's output layout: ``out/compiled.axmodel`` plus
``out/quant/quant_axmodel.json``. A ``.axmodel`` path and a ``.json`` path
also work, gzipped or not.
"""

from __future__ import annotations

import gzip
import itertools
import json
import os
import struct
import sys
from collections.abc import Iterable, Mapping
from functools import lru_cache

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import short_unit_codec as codec  # noqa: E402
import step_recalibrate as sr  # noqa: E402

LANE_REGS = frozenset(range(0x0F50, 0x1010, 0x10))
"""Float32 scale lanes: ``1/s`` and ``s`` groups at ``0x0f50..0x0fc0``
(#1831) and the divisor group at ``0x0fd0..0x1000`` (#1836)."""
ZERO_POINT_REGS = frozenset(sr.ZERO_POINT_REGS) | {0x1EB0}
"""Plus ``0x1eb0``, the third zero-point register the Reshape->Relu layout
writes (docs/axera-step-real-calibration.md): a signed-input 3x3 Conv chain
writes its input zero point there."""
OFFSET_REGS = frozenset(range(0x1EF0, 0x1F30, 0x10))
"""A fused Add (a live bias) carries the int32 zero-point offset of
``binary_op_scale_emit.zp_offset`` on these four lanes (#1869)."""
SHIFT_REG = 0x1EA0
"""... and its Q-format, ``15 - k``, on this register."""
FULL_WRITE = 0xA1
TENG_FLAG = 0xA8
NOISE_REGS = frozenset({0x02B0, 0x03D0})
"""Verb ``0xa8`` writes to these two registers flip between ``1/2`` and
``3/4`` even between rebuilds with identical calibration (``t17
batched_r0`` vs ``r3``). They are rebuild noise, not calibration."""

Scales = dict[str, tuple[float, float]]
Role = tuple


class CalibrationError(ValueError):
    pass


# --------------------------------------------------------------------------
# loading


def _read(path: str) -> bytes:
    with (gzip.open if path.endswith(".gz") else open)(path, "rb") as f:
        return f.read()


def build_paths(where: str) -> tuple[str, str]:
    """``(axmodel, quant json)`` for a build directory or an axmodel path.
    Next to ``X.axmodel[.gz]``, the scales are looked up in
    ``X.quant.json[.gz]``."""
    if os.path.isdir(where):
        return (
            os.path.join(where, "out", "compiled.axmodel"),
            os.path.join(where, "out", "quant", "quant_axmodel.json"),
        )
    stem = where[: -len(".gz")] if where.endswith(".gz") else where
    stem = stem[: -len(".axmodel")] if stem.endswith(".axmodel") else stem
    for q in (stem + ".quant.json", stem + ".quant.json.gz"):
        if os.path.exists(q):
            return where, q
    raise FileNotFoundError(f"no quant json next to {where}")


@lru_cache(maxsize=256)
def load_model(path: str) -> onnx.ModelProto:
    return onnx.load_model_from_string(_read(path))


I8 = "#i8"
"""Suffix of a tensor's second quantization: a uint8 tensor that also feeds a
MatMul is requantized by it to symmetric int8, and that consumer's view
(``quant_min`` -128, its own scale) is a separate scale the MatMul's lanes
use (``1/s`` of it in a dW chain whose activation also has a uint8 use)."""


def quant_scales(quant: dict) -> Scales:
    """``{tensor: (scale, zero_point)}`` from a Pulsar2 ``quant_axmodel.json``.
    ``tensor_configs`` names each tensor per consumer, with the hash of its
    quantization in ``values``. A tensor seen both as uint8 and, by a MatMul,
    as symmetric int8 at another scale keeps the uint8 view under its name
    and the int8 one under ``name + I8``."""
    values = quant["values"]
    views: dict[str, list[tuple[float, float, bool]]] = {}
    for per_op in quant["tensor_configs"].values():
        for name, cfg in per_op.items():
            key = str(cfg.get("dominator", cfg.get("hash")))
            v = values.get(key)
            if (
                isinstance(v, dict)
                and isinstance(v.get("scale"), list)
                and len(v["scale"]) == 1
            ):
                q = (float(v["scale"][0]), float(v["zero_point"][0]))
                views.setdefault(name, []).append((*q, cfg.get("quant_min", 0) < 0))
    out: Scales = {}
    for name, vs in views.items():
        u8 = [v for v in vs if not v[2]]
        i8 = [v for v in vs if v[2]]
        out[name] = (u8[-1] if u8 else vs[-1])[:2]
        if u8 and i8 and i8[-1][0] != u8[-1][0]:
            out[name + I8] = i8[-1][:2]
    return out


@lru_cache(maxsize=256)
def load_scales(path: str) -> Scales:
    return quant_scales(json.loads(_read(path)))


def params_of(model: onnx.ModelProto) -> bytes:
    return sr._init(model, sr.PARAMS)


# --------------------------------------------------------------------------
# roles: closed set of scale formulas a calibration value can be


def _bits(x) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def evaluate(role: Role, scales: Scales) -> int:
    """The 32-bit value ``role`` takes at ``scales``."""
    kind = role[0]
    s = {t: v[0] for t, v in scales.items()}
    if kind == "inv32":
        return _bits(np.float32(1) / np.float32(s[role[1]]))
    if kind == "inv":
        return _bits(1 / s[role[1]])
    if kind == "s":
        return _bits(s[role[1]])
    if kind == "ratio":
        return _bits(s[role[1]] / s[role[2]])
    if kind == "mult":
        return _bits(s[role[1]] * s[role[2]] / s[role[3]])
    if kind == "meanr":
        # float32 arithmetic: in float64 the Gemm chain's lane comes out one
        # ulp high (0x3d84030f for the native 0x3d84030e).
        a, b = np.float32(s[role[1]]), np.float32(s[role[2]])
        return _bits(a / (b * np.float32(role[3])))
    if kind == "zpf":
        return _bits(scales[role[1]][1])
    if kind == "nzpf":
        return _bits(-scales[role[1]][1])
    if kind == "zp":
        return int(scales[role[1]][1]) & 0xFFFFFFFF
    if kind == "zpk":
        return int(scales[role[1]][1]) * role[2] & 0xFFFFFFFF
    if kind == "zp8":
        z = int(scales[role[1]][1])
        if not 0 < z < 256:
            raise CalibrationError(f"zero point {z} of {role[1]} is not one byte")
        return z
    if kind in ("zpoff", "qshift", "q15", "rqoff", "rqshift"):
        return _bias_add(kind, *role[1:], scales)
    if kind == "cat15":
        return _cat15(*role[1:], scales)
    raise ValueError(f"unknown role {role!r}")


def _bias_add(kind: str, *names_and_scales) -> int:
    *names, scales = names_and_scales
    if kind in ("rqoff", "rqshift"):
        return _requant(kind, *names, scales)
    return _add(kind, *names, scales)


def _requant(kind: str, x: str, y: str, scales: Scales) -> int:
    """A one-input requantize's words (``x`` into ``y``): ``rqshift`` is
    ``0x80 | (15 - k)`` and ``rqoff`` is ``int((zp_y - zp_x*r) * 2**(15-k))``
    with ``r = s_x/s_y`` rounded to float32, ``k`` the smallest shift that
    brings ``r`` to at most 1. Seen on the weight's way into the Concat of a
    3x3 Conv's taps (asymmetric uint8 to symmetric)."""
    (sx, zx), (sy, zy) = scales[x], scales[y]
    if x == y or zx == zy:
        raise CalibrationError("not a requantize")
    # r == 1 exactly (a Slice into a Concat at the same scale) keeps k = 0:
    # offset -zp_x * 2**15 and shift 0x8f, seen on the step's stage4 conv1.
    k = 0
    while sx / sy * 2.0**-k > 1.0:
        k += 1
        if k > 15:
            raise CalibrationError(f"requantize ratio of {y} does not fit Q15")
    if kind == "rqshift":
        return 0x80 | (15 - k)
    # zp_x * r is a float32 product: with a double one, 2 of 6 native
    # offsets (signed-input chains, zp_x 98..113) come out one too high.
    r = np.float32(sx / sy)
    c = zy - float(np.float32(np.float32(zx) * r))
    return int(c * 2.0 ** (15 - k)) & 0xFFFFFFFF


def _q15_shift(a: str, b: str, scales: Scales) -> int:
    """``k``, the smallest shift that brings ``s_a/s_b * 2**-k`` to at most 1
    (the requantize's rule: exactly 1 keeps ``k = 0``)."""
    r = scales[a][0] / scales[b][0]
    if a == b or not r > 0:
        raise CalibrationError(f"Concat ratio {a}/{b} = {r} is not positive")
    k = 0
    while r * 2.0**-k > 1.0:
        k += 1
        if k > 15:
            raise CalibrationError(f"Concat ratio {a}/{b} = {r} does not fit Q15")
    return k


def _q15_shifted(a: str, b: str, scales: Scales, k: int | None = None) -> int:
    """``round(s_a/s_b * 2**(15-k))``. With the template's ``k`` given, refuse
    a ratio on the other side of 1: ``k = 0`` and ``k >= 1`` compile to
    different record counts (a shift above 0 adds the ``0x1ea0`` write of
    the activation taps' requantize, which ``k = 0`` elides). Shifts 1 and 2
    are the same program (record-exact both ways in the
    ``docs/axera-conv-concat-shift.md`` sweep)."""
    got = _q15_shift(a, b, scales)
    if k is not None and (got == 0) != (k == 0):
        raise CalibrationError(
            f"Concat ratio {a}/{b} needs shift {got}, the template has {k} "
            "(a ratio on the other side of 1 is a different program)"
        )
    if k is not None and max(got, k) > 2:
        # Only builds at shifts 0..2 were paired (the sweep's shift-3 build
        # pairs with none: it only ever came out as the other allocator
        # variant); the step's convs need at most 1.
        raise CalibrationError(f"Concat ratio {a}/{b} needs shift {got}, unmeasured")
    return int(round(scales[a][0] / scales[b][0] * 2.0 ** (15 - got)))


def _cat15(a: str, b: str, c: str, d: str, *shifts_and_scales) -> int:
    """A 3x3 Conv chain's Concat header in ``npu_params``: the activation
    taps' ratio into their Concat (``s_a/s_b``) and the weight taps' ratio
    into theirs (``s_c/s_d``), each ``round(r * 2**(15-k))`` as a uint16,
    low half first, ``k`` the requantize's shift for that ratio. A ratio at
    most 1 has ``k = 0`` (a hair below 1 stores 32768, one further down
    32767). An unfused Relu's output feeding the taps shares its wider
    pre-activation quantization, so the activation taps' ratio into their
    Concat is above 1 there (stage2/3/4 conv1): ``k = 1`` and the half is
    Q14 (stage2 conv1: ``1.0991 * 2**14`` = 18008). The role carries both
    ``k``: a recalibration that moves either across ``k = 0`` is refused."""
    *shifts, scales = shifts_and_scales
    ka, kw = shifts if shifts else (None, None)
    return _q15_shifted(a, b, scales, ka) | _q15_shifted(c, d, scales, kw) << 16


def _add(kind: str, x: str, z: str, y: str, scales: Scales) -> int:
    """A fused ``y = Add(x, z)``'s calibration words, as
    ``binary_op_scale_emit`` decoded them for a standalone Add (#1869), with
    the Q15 shift ``k`` applied to the offset too:

    * ``q15``: ``npu_params`` header, ``round(s_x/s_y * 2**(15-k))`` and
      ``round(s_z/s_y * 2**(15-k))`` as two little-endian uint16s, ``k``
      the smallest shift that brings both ratios below 1;
    * ``qshift``: ``15 - k``;
    * ``zpoff``: ``int((zp_y - zp_x*r_x - zp_z*r_z) * 2**(15-k))`` with the
      ratios rounded to float32.

    Fits the step's Gemm-as-MatMul+Add builds (``k = 1``) and #1870's
    ``mm_add`` (``k = 0``)."""
    (sx, zx), (sz, zz), (sy, zy) = scales[x], scales[z], scales[y]
    k = 0
    while max(sx / sy, sz / sy) * 2.0**-k >= 1.0:
        k += 1
        if k > 15:
            raise CalibrationError(f"Add ratios of {y} do not fit Q15")
    q = 15 - k
    if kind == "qshift":
        return q
    if kind == "q15":
        wx, wz = (int(round(r * 2.0**q)) for r in (sx / sy, sz / sy))
        if wx == wz or not (0 <= wx < 1 << 16 and 0 <= wz < 1 << 16):
            raise CalibrationError(f"Add header of {y} changes layout ({wx}, {wz})")
        return wx | wz << 16
    c = zy - zx * float(np.float32(sx / sy)) - zz * float(np.float32(sz / sy))
    return int(c * 2.0**q) & 0xFFFFFFFF


def float_roles(tensors: Iterable[str]) -> list[Role]:
    ts = sorted(tensors)
    roles: list[Role] = []
    for t in ts:
        roles += [("inv32", t), ("inv", t), ("s", t), ("zpf", t), ("nzpf", t)]
    roles += [("ratio", a, b) for a, b in itertools.permutations(ts, 2)]
    roles += [
        ("mult", a, b, c)
        for a, b in itertools.combinations_with_replacement(ts, 2)
        for c in ts
    ]
    roles += [
        ("meanr", a, b, k)
        for a, b in itertools.permutations(ts, 2)
        for k in POOL_WINDOWS
    ]
    return roles


def role_table(scales: Scales, roles: Iterable[Role]) -> dict[int, list[Role]]:
    table: dict[int, list[Role]] = {}
    for r in roles:
        table.setdefault(evaluate(r, scales), []).append(r)
    return table


# --------------------------------------------------------------------------
# locating calibration values in a template


def _records(seg: bytes):
    for i in range(0, len(seg), 8):
        verb, reg = seg[i], int.from_bytes(seg[i + 2 : i + 4], "little")
        yield i, verb, reg, struct.unpack_from("<I", seg, i + 4)[0]


def locate(model: onnx.ModelProto, scales: Scales) -> dict:
    """Every calibration value in ``model``, with the roles that explain it.

    Returns ``{"records": [(seg, byte offset, reg, value, roles)], "params":
    [(byte offset, value, roles)]}``. Record entries cover every nonzero
    lane and zero-point record; an empty ``roles`` list means unexplained.
    Param entries cover only the words that some role explains. The other
    words are shape data (DMA descriptors, padding) and stay as they are."""
    segs = codec.decode_segments(sr.get_mcode(model))
    ftable = role_table(scales, float_roles(_representatives(scales)))
    ztable = role_table(
        scales,
        [("zp", t) for t in scales]
        + [("zpk", t, k) for t in scales for k in POOL_WINDOWS if scales[t][1]],
    )
    recs = []
    for si, seg in enumerate(segs):
        for off, verb, reg, value in _records(seg):
            if verb != FULL_WRITE or value == 0:
                continue
            if reg in LANE_REGS:
                recs.append((si, off, reg, value, ftable.get(value, [])))
            elif reg in ZERO_POINT_REGS and (reg != 0x1EB0 or value in ztable):
                # 0x1eb0 also carries non-zero-point words (0x80000000)
                recs.append((si, off, reg, value, ztable.get(value, [])))
    params = params_of(model)
    lanes = _param_lanes(params, ftable)
    offsets = [
        (si, off, reg, value)
        for si, seg in enumerate(segs)
        for off, verb, reg, value in _records(seg)
        if verb == FULL_WRITE and reg in OFFSET_REGS and value
    ]
    if offsets:
        recs, lanes = _locate_bias_add(segs, scales, offsets, recs, params, lanes)
    lanes = sorted(lanes + _zp8_lanes(params, scales, lanes))
    return {
        "segments": segs,
        "params_bytes": params,
        "records": recs,
        "params": lanes,
    }


MIN_LANE_RUN = 4

POOL_WINDOWS = (49, 196, 784, 3136)
"""A mean pool fused into a chain (the classifier's ReduceMean over 7x7)
writes its input zero point times the window size to a zero-point
register (``zpk``): 4508 = 92 * 49 in the Gemm chain. Its mean ratio,
``s_x / (s_y * N)`` in float32, is a scale lane (``meanr``): 0.064459 =
0.072179 / (0.022852 * 49) there. The step's global pools are these
spatial sizes."""


def _zp8_lanes(params: bytes, scales: Scales, taken_lanes: list) -> list:
    """A uint8 input requantized inside the chain also leaves its zero point
    in ``npu_params`` as a run of equal bytes (signed-input Conv chains: 9 or
    32 bytes of ``zp_x``, not word-aligned). One ``zp8`` entry per byte of a
    run of at least ``2 * MIN_LANE_RUN`` bytes, outside the float lanes."""
    zps: dict[int, list[Role]] = {}
    for t, (_, z) in scales.items():
        if 0 < z < 256:
            zps.setdefault(int(z), []).append(("zp8", t))
    taken = {
        o + d for o, v, r in taken_lanes for d in range(1 if r[0][0] == "zp8" else 4)
    }
    out, i = [], 0
    while i < len(params):
        j = i
        while j < len(params) and params[j] == params[i]:
            j += 1
        if params[i] in zps and j - i >= 2 * MIN_LANE_RUN:
            out += [
                (o, params[i], zps[params[i]]) for o in range(i, j) if o not in taken
            ]
        i = j
    return out


def _param_lanes(params: bytes, ftable: dict[int, list[Role]]) -> list:
    """Role-explained float32 words of ``npu_params``, in runs of at least
    ``MIN_LANE_RUN`` equal words. Requant lanes come one per output column,
    so a genuine one is never alone. Blocks are not 4-byte aligned in a
    fused chain (the Gather-MatMul chain's lanes start at byte 4511), so all
    four phases are scanned."""
    out = []
    for phase in range(4):
        words = [
            (off, struct.unpack_from("<I", params, off)[0])
            for off in range(phase, len(params) - 3, 4)
        ]
        i = 0
        while i < len(words):
            j = i
            while j < len(words) and words[j][1] == words[i][1]:
                j += 1
            value = words[i][1]
            if value and value in ftable and j - i >= MIN_LANE_RUN:
                out += [(off, value, ftable[value]) for off, _ in words[i:j]]
            i = j
    out.sort()
    for (a, _, _), (b, _, _) in zip(out, out[1:]):
        if b < a + 4:
            raise CalibrationError(f"npu_params lanes overlap at bytes {a} and {b}")
    return out


def _offset_groups(segs):
    """``(shift record, [offset records])`` per nonzero zero-point offset
    group: the four ``OFFSET_REGS`` lanes and the ``SHIFT_REG`` write that
    last precedes them."""
    groups = []
    for si, seg in enumerate(segs):
        shift = None
        for off, verb, reg, v in _records(seg):
            if verb != FULL_WRITE:
                continue
            if reg == SHIFT_REG:
                shift = (si, off, reg, v)
            elif reg in OFFSET_REGS and v:
                if (
                    groups
                    and groups[-1][1][-1][0] == si
                    and groups[-1][1][-1][1] == off - 8
                ):
                    groups[-1][1].append((si, off, reg, v))
                else:
                    groups.append((shift, [(si, off, reg, v)]))
    return groups


def _representatives(scales: Scales) -> list[str]:
    """One tensor name per distinct ``(scale, zero point)``, the first in
    sort order. A chain's taps share their source's quantization
    (OVERLAPPED), and the stem Conv's 147 taps make the role searches (cubic
    in names) take minutes otherwise; twins move together, so the stand-in
    gives the same value at any calibration."""
    return sorted({v: t for t, v in sorted(scales.items(), reverse=True)}.values())


def _locate_bias_add(segs, scales, offsets, recs, params, lanes):
    """The zero-point offset groups of a fused chain, each with the
    ``SHIFT_REG`` write before it, and a bias Add's ``npu_params`` header.

    A shift with bit 7 set belongs to a one-input requantize (a Concat input
    moving from an asymmetric to a symmetric quantization): ``rqoff`` /
    ``rqshift`` over ``(x, y)`` pairs. Otherwise it is an Add: ``zpoff`` /
    ``qshift`` / ``q15`` over ``(x, z, y)`` triples. Every tuple of tensors
    that explains both the offset and the shift is kept as a role. A group
    whose offset is 0 is not located (there is nothing to match), and
    ``recalibrate`` cannot tell; the step templates have no such group."""
    del offsets  # regrouped with their shift records below
    names = _representatives(scales)
    header: dict[int, list[Role]] = {}
    rq_pairs: set[tuple[str, str]] = set()
    for shift, group in _offset_groups(segs):
        if shift is None:
            raise CalibrationError("zero-point offset lanes without a shift register")
        v, sv = group[0][3], shift[3]
        if sv & 0x80:
            tuples = [("rq", x, y) for x, y in itertools.permutations(names, 2)]
            kinds = ("rqoff", "rqshift")
        else:
            tuples = [("add", *t) for t in itertools.permutations(names, 3)]
            kinds = ("zpoff", "qshift")
        hit = []
        for t in tuples:
            try:
                if _bias_add(kinds[0], *t[1:], scales) == v and (
                    _bias_add(kinds[1], *t[1:], scales) == sv
                ):
                    hit.append(t[1:])
            except CalibrationError:
                pass
        recs = recs + [
            (si, off, reg, w, [(kinds[0], *t) for t in hit])
            for si, off, reg, w in group
        ]
        recs.append((*shift, [(kinds[1], *t) for t in hit]))
        if kinds[0] == "rqoff":
            rq_pairs.update(hit)
        if kinds[0] == "zpoff":
            for t in hit:
                try:
                    header.setdefault(evaluate(("q15", *t), scales), []).append(
                        ("q15", *t)
                    )
                except CalibrationError:
                    pass
    taken = {o + d for o, _, _ in lanes for d in range(-3, 4)}
    found = [
        (off, v, header[v])
        for off in range(len(params) - 3)
        if off not in taken
        for v in [struct.unpack_from("<I", params, off)[0]]
        if v in header
    ]
    if header and len(found) != 1:
        raise CalibrationError(
            f"bias Add header found {len(found)} times in npu_params, want once"
        )
    if rq_pairs:
        found += _locate_cat_header(scales, rq_pairs, params, lanes + found)
    return recs, sorted(lanes + found)


def _locate_cat_header(scales, rq_pairs, params, taken_lanes):
    """The Concat header of a chain that requantizes its weight taps (see
    ``_cat15``). The weight half's ratio is one of the requantize groups'
    ``(x, y)`` pairs; the activation half is any pair whose ratio is at most
    1 and gives the stored value (the taps and their Concat, which usually
    share no zero-point offset group because both zero points are 0). The
    header must be found exactly once. The two builds of every 3x3 Conv
    template before stage2 conv1 stored 32768 in the weight half, so a
    missing header went unnoticed; stage2 conv1's pair is 32768 vs 32767."""
    names = sorted(scales)

    def halves(pairs):
        out: dict[int, list[tuple[str, str, int]]] = {}
        for a, b in pairs:
            try:
                k = _q15_shift(a, b, scales)
                out.setdefault(_q15_shifted(a, b, scales), []).append((a, b, k))
            except CalibrationError:
                pass
        return out

    hi = halves(sorted(rq_pairs))
    lo = halves(itertools.permutations(names, 2))
    taken = {o + d for o, _, _ in taken_lanes for d in range(-3, 4)}
    found = []
    for off in range(len(params) - 3):
        if off in taken:
            continue
        v = struct.unpack_from("<I", params, off)[0]
        wl, wh = v & 0xFFFF, v >> 16
        if wh in hi and wl in lo:
            # The two halves are two different Concats' ratios. One ratio
            # and its inverse (now that a half may be shifted) is a
            # misaligned read two bytes before the stem's real header.
            roles = [
                ("cat15", p[0], p[1], q[0], q[1], p[2], q[2])
                for p in lo[wl]
                for q in hi[wh]
                if (scales[p[0]], scales[p[1]]) != (scales[q[1]], scales[q[0]])
            ]
            if roles:
                found.append((off, v, roles))
    if len(found) != 1:
        raise CalibrationError(
            f"Concat header found {len(found)} times in npu_params, want once"
        )
    return found


PRECEDENCE = (
    "zp",
    "zp8",
    "zpk",
    "zpf",
    "nzpf",
    "s",
    "inv32",
    "inv",
    "ratio",
    "mult",
    "meanr",
    "zpoff",
    "qshift",
    "q15",
    "rqoff",
    "rqshift",
    "cat15",
)
"""Tie-break between formulas that give the same bits at the template's
scales. Across 174 builds, ``inv32``/``inv`` (``1/s`` in float32 or
float64) always matched together, and so did ``s``/``mult(x, y, x)``. No
build tells them apart. For a float32-exact scale they differ only on a
double-rounding tie. So the simplest formula wins, and within a kind the
formula must be unique."""


def _new_value(where: str, value: int, roles: list[Role], new: Scales) -> int:
    if not roles:
        raise CalibrationError(f"{where}: value {value:#010x} matches no scale formula")
    kind = min((r[0] for r in roles), key=PRECEDENCE.index)
    got = {evaluate(r, new) for r in roles if r[0] == kind}
    if len(got) != 1:
        raise CalibrationError(
            f"{where}: value {value:#010x} is ambiguous; {kind} roles "
            f"{[r for r in roles if r[0] == kind][:4]} disagree at the new scales"
        )
    return got.pop()


def _pow2_exponent(r: float) -> int | None:
    e = float(np.log2(r))
    return round(e) if abs(e - round(e)) < 1e-6 else None


def _check_fixed_ratios(old: Scales, new: Scales) -> None:
    """Refuse a calibration that changes an exact power-of-two scale ratio
    (other than 1) between two template tensors. Pulsar2 sets a 3x3 Conv's
    Concat input to exactly twice its uint8 source's scale, and the program
    has no record carrying that ratio: moving the two apart independently
    compiled to outputs 189 LSB off on the device, while keeping the ratio
    stayed within 1-2 LSB (``docs/axera-emitter-device-check.md``)."""
    names = sorted(t for t in old if old[t][0] > 0)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            k = _pow2_exponent(old[b][0] / old[a][0])
            if not k:
                continue
            if _pow2_exponent(new[b][0] / new[a][0]) != k:
                raise CalibrationError(
                    f"scale of {b} is 2**{k} x {a}'s in the template; the new "
                    "calibration breaks that ratio, which the program does not carry"
                )


def recalibrate(
    model: onnx.ModelProto, old: Scales, new: Scales
) -> tuple[onnx.ModelProto, dict]:
    """``model`` (built at scales ``old``) moved to scales ``new``."""
    missing = set(old) - set(new)
    if missing:
        raise CalibrationError(f"new scales miss {sorted(missing)}")
    crossing = [t for t in old if (old[t][1] == 0) != (new[t][1] == 0)]
    if crossing:
        raise CalibrationError(
            f"zero point goes between zero and nonzero for {crossing}; "
            "Pulsar2 emits a different record count then"
        )
    _check_fixed_ratios(old, new)
    groups: dict[tuple, list[str]] = {}
    for t, v in old.items():
        groups.setdefault(v, []).append(t)
    split = [ts for ts in groups.values() if len({new[t] for t in ts}) > 1]
    if split:
        raise CalibrationError(
            f"tensors that share a quantization in the template part at the new "
            f"scales: {sorted(split[0])[:4]}; roles name one of them for all"
        )
    found = locate(model, old)
    segs = [bytearray(s) for s in found["segments"]]
    changed: set[int] = set()
    for si, off, reg, value, roles in found["records"]:
        nv = _new_value(
            f"segment {si} record {off // 8} reg {reg:#06x}", value, roles, new
        )
        if nv != value:
            struct.pack_into("<I", segs[si], off + 4, nv)
            changed.add(si)
    params = bytearray(found["params_bytes"])
    for off, value, roles in found["params"]:
        nv = _new_value(f"npu_params byte {off}", value, roles, new)
        if roles[0][0] == "zp8":  # one byte, not a word
            params[off] = nv
        else:
            struct.pack_into("<I", params, off, nv)
    mc = sr.get_mcode(model)
    new_mc = sr.replace_segments(mc, {i: bytes(segs[i]) for i in changed})
    sr.check_relayout(new_mc, [bytes(s) for s in segs])
    out = sr.with_mcode(model, new_mc)
    for init in out.graph.initializer:
        if init.name == sr.PARAMS:
            init.raw_data = bytes(params)
    report = {
        "records": len(found["records"]),
        "segments_changed": sorted(changed),
        "params_words": len(found["params"]),
        "mcode_bytes": [len(mc), len(new_mc)],
    }
    return out, report


# --------------------------------------------------------------------------
# comparison against a native build


def _is_noise(si: int, verb: int, reg: int) -> bool:
    if si == 0 and verb == 0xA2:  # segment-0 tail rotation (#1836)
        return True
    return verb == TENG_FLAG and reg in NOISE_REGS


def compare(got: onnx.ModelProto, want: onnx.ModelProto) -> dict:
    """Record-level difference, split into rebuild noise and real diffs."""
    a = codec.decode_segments(sr.get_mcode(got))
    b = codec.decode_segments(sr.get_mcode(want))
    if [len(s) for s in a] != [len(s) for s in b]:
        return {"structure": False}
    real, noise = [], 0
    for si, (x, y) in enumerate(zip(a, b)):
        for (off, verb, reg, v), (_, verb2, reg2, w) in zip(_records(x), _records(y)):
            if (verb, reg) != (verb2, reg2):
                return {"structure": False}
            if v != w:
                if _is_noise(si, verb, reg):
                    noise += 1
                else:
                    real.append((si, off // 8, reg, v, w))
    pa, pb = params_of(got), params_of(want)
    return {
        "structure": True,
        "record_diffs": real,
        "noise_diffs": noise,
        "params_diff_bytes": (
            sum(1 for p, q in zip(pa, pb) if p != q) if len(pa) == len(pb) else None
        ),
    }


def check(template: str, target: str) -> dict:
    tm, tq = build_paths(template)
    gm, gq = build_paths(target)
    want = load_model(gm)
    out, report = recalibrate(load_model(tm), load_scales(tq), load_scales(gq))
    report.update(compare(out, want))
    return report


# --------------------------------------------------------------------------
# the ResNet18 training step's live-operand shapes


STEP_SHAPES = {
    # (A shape, B shape): (nodes in step.onnx, role)
    ((16, 1000), (1000, 512)): (1, "fc dX"),
    ((512, 16), (16, 1000)): (1, "fc dW"),
    ((16, 512), (512, 1000)): (1, "fc forward (Gemm, transB folded)"),
    ((1, 1, 512, 4608), (16, 1, 4608, 49)): (3, "dX, 512->512 3x3 @7x7"),
    ((16, 1, 512, 49), (16, 1, 49, 4608)): (3, "dW, 512->512 3x3 @7x7"),
    ((1, 1, 256, 4608), (16, 1, 4608, 196)): (1, "dX, 256->512 3x3 s2"),
    ((16, 1, 512, 49), (16, 1, 49, 2304)): (1, "dW, 256->512 3x3 s2"),
    ((1, 1, 256, 512), (16, 1, 512, 196)): (1, "dX, 256->512 1x1 s2"),
    ((16, 1, 512, 49), (16, 1, 49, 256)): (1, "dW, 256->512 1x1 s2"),
    ((1, 1, 256, 2304), (16, 1, 2304, 196)): (3, "dX, 256->256 3x3 @14x14"),
    ((16, 1, 256, 196), (16, 1, 196, 2304)): (3, "dW, 256->256 3x3 @14x14"),
    ((1, 1, 128, 2304), (16, 1, 2304, 784)): (1, "dX, 128->256 3x3 s2"),
    ((16, 1, 256, 196), (16, 1, 196, 1152)): (1, "dW, 128->256 3x3 s2"),
    ((1, 1, 128, 256), (16, 1, 256, 784)): (1, "dX, 128->256 1x1 s2"),
    ((16, 1, 256, 196), (16, 1, 196, 128)): (1, "dW, 128->256 1x1 s2"),
    ((1, 1, 128, 1152), (16, 1, 1152, 784)): (3, "dX, 128->128 3x3 @28x28"),
    ((16, 1, 128, 784), (16, 1, 784, 1152)): (3, "dW, 128->128 3x3 @28x28"),
    ((1, 1, 64, 1152), (16, 1, 1152, 3136)): (1, "dX, 64->128 3x3 s2"),
    ((16, 1, 128, 784), (16, 1, 784, 576)): (1, "dW, 64->128 3x3 s2"),
    ((1, 1, 64, 128), (16, 1, 128, 3136)): (1, "dX, 64->128 1x1 s2"),
    ((16, 1, 128, 784), (16, 1, 784, 64)): (1, "dW, 64->128 1x1 s2"),
    ((1, 1, 64, 576), (16, 1, 576, 3136)): (4, "dX, 64->64 3x3 @56x56"),
    ((16, 1, 64, 3136), (16, 1, 3136, 576)): (4, "dW, 64->64 3x3 @56x56"),
    ((16, 1, 64, 12544), (16, 1, 12544, 147)): (1, "dW, stem 7x7 s2"),
}
"""The step's 41 MatMuls plus its Gemm, from
``t6-r18fold/step.onnx`` shape inference, as ``(A, B)`` operand shapes.
The labels are read off the shapes. ``dX`` is a Conv's input gradient: the
kernel ``[1, 1, Cin, Cout*k*k]`` against the gathered output gradient, as in
``docs/axera-gather-aggregate-real.md``. ``dW`` is its weight gradient. They
were not traced node by node."""

STEP_CONV_MATMULS = {
    # legalize.act_weight_conv_to_matmul: X[16, Ho, Wo, Cin*k*k] @ W[Cin*k*k, Cout]
    ((16, 112, 112, 147), (147, 64)): 1,
    ((16, 56, 56, 576), (576, 64)): 4,
    ((16, 28, 28, 64), (64, 128)): 1,
    ((16, 28, 28, 576), (576, 128)): 1,
    ((16, 28, 28, 1152), (1152, 128)): 3,
    ((16, 14, 14, 128), (128, 256)): 1,
    ((16, 14, 14, 1152), (1152, 256)): 1,
    ((16, 14, 14, 2304), (2304, 256)): 3,
    ((16, 7, 7, 256), (256, 512)): 1,
    ((16, 7, 7, 2304), (2304, 512)): 1,
    ((16, 7, 7, 4608), (4608, 512)): 3,
}
"""The step's 20 live-weight Convs after ``act_weight_conv_to_matmul``: one
MatMul each, with the taps concatenated along the contraction."""

STEP_TEMPLATE_DIR = os.path.join(_HERE, "fixtures", "matmul_step_templates")
"""One Pulsar2 build per distinct step chain, plus ``manifest.json``: which
template serves which ``step.onnx`` node, and how the node's tensor names
map onto the template's (``docs/axera-matmul-step-templates.md``)."""


STANDALONE_MATMUL_TEMPLATES = {
    ((16, 1000), (1000, 512)): (
        "fc_dX_MatMul_36__v2.axmodel.gz",
        "fc_dX_MatMul_36__v2.quant.json.gz",
    ),
}


def emit_standalone_matmul(
    a_shape,
    b_shape,
    scales: Mapping[str, float],
    zero_points: Mapping[str, int],
) -> onnx.ModelProto:
    """Retarget one exact live-weight MatMul template without Pulsar2.

    The template is a measured full AX program; this adapter only maps the
    caller's ``x/z/y`` calibration onto its three tensor names. Unknown
    operand shapes remain refused until a matching training-chain build is
    validated.
    """
    key = (tuple(int(v) for v in a_shape), tuple(int(v) for v in b_shape))
    paths = STANDALONE_MATMUL_TEMPLATES.get(key)
    if paths is None:
        raise ValueError(f"no measured standalone MatMul template for {key}")
    model_path, quant_path = (os.path.join(STEP_TEMPLATE_DIR, name) for name in paths)
    model = load_model(model_path)
    old = load_scales(quant_path)
    if len(model.graph.input) != 2 or len(model.graph.output) != 1:
        raise ValueError("standalone MatMul template has an unexpected signature")
    names = [item.name for item in model.graph.input] + [model.graph.output[0].name]
    new = {
        names[0]: (float(scales["x"]), float(zero_points["x"])),
        names[1]: (float(scales["z"]), float(zero_points["z"])),
        names[2]: (float(scales["y"]), float(zero_points["y"])),
    }
    return recalibrate(model, old, new)[0]


@lru_cache(maxsize=1)
def step_manifest() -> dict:
    with open(os.path.join(STEP_TEMPLATE_DIR, "manifest.json")) as f:
        return json.load(f)


def step_template(node: str, manifest: dict | None = None) -> dict:
    """The template serving ``step.onnx`` node ``node``: ``axmodel`` and
    ``quant`` paths, and ``names``, the node's tensor names mapped onto the
    template's. A node is listed only if its chain has the template's exact
    ops, attributes, input shapes and constants. Raises ``ValueError`` for
    any other node."""
    m = manifest if manifest is not None else step_manifest()
    entry = m["nodes"].get(node)
    if entry is None:
        raise ValueError(f"no validated live-operand template serves node {node!r}")
    t = m["templates"][entry["template"]]
    return {
        "template": entry["template"],
        "axmodel": os.path.join(STEP_TEMPLATE_DIR, t["axmodel"]),
        "quant": os.path.join(STEP_TEMPLATE_DIR, t["quant"]),
        "names": dict(zip(entry["names"], t["names"])),
        "constants": list(t.get("constants", [])),
        "aliases": dict(t.get("aliases", {})),
        # The node runs as ``batch_split`` invocations of a template built at
        # batch ``N / batch_split`` (dX MatMul_325: Pulsar2 does not finish
        # the batch-16 chain; every op in it is per-sample).
        "batch_split": int(entry.get("batch_split", 1)),
    }


def step_node_scales(entry: dict, template_scales: Scales, scales: Scales) -> Scales:
    """``scales`` (keyed by step tensor names) re-keyed onto the template's
    names, ready for ``recalibrate``. Constants baked into the chain (Gather
    masks) keep the template's own scale; every other template tensor must
    be given."""
    out: Scales = {}
    for step_name, name in entry["names"].items():
        if step_name in scales:
            out[name] = scales[step_name]
        if step_name + I8 in scales and name + I8 in template_scales:
            out[name + I8] = scales[step_name + I8]
    # An int8 view no consumer declares in the step's calibration (the
    # weight's view by the Reshape that starts a dX kernel path) is the
    # quantization of the int8 tensor it overlaps: take that tensor's scale.
    for t, v in template_scales.items():
        if t.endswith(I8) and t not in out:
            twins = [
                u for u, w in template_scales.items() if w == v and u != t and u in out
            ]
            if twins:
                out[t] = out[twins[0]]
    for name in entry["constants"]:
        if name in template_scales:
            out[name] = template_scales[name]
    # A template-only side output (an Identity that keeps a chain input
    # uint8, as its other consumers keep it in the step) shares its source's
    # quantization.
    for name, src in entry.get("aliases", {}).items():
        if src in out:
            out[name] = out[src]
    missing = sorted(set(template_scales) - set(out))
    if missing:
        raise CalibrationError(f"no step scale for template tensors {missing}")
    return out


# --------------------------------------------------------------------------


def _explain(where: str) -> None:
    m, q = build_paths(where)
    model, scales = load_model(m), load_scales(q)
    found = locate(model, scales)
    print("tensors:", {t: v for t, v in sorted(scales.items())})
    for si, off, reg, value, roles in found["records"]:
        print(f"seg{si} rec{off // 8:6d} reg={reg:#06x} {value:#010x} {roles[:3]}")
    by: dict[tuple, int] = {}
    for _, _, roles in found["params"]:
        by[tuple(roles[:2])] = by.get(tuple(roles[:2]), 0) + 1
    for roles, n in by.items():
        print(f"npu_params {n:6d} words {list(roles)}")


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "check":
        print(json.dumps(check(argv[1], argv[2]), indent=1, default=str))
        return 0
    if len(argv) == 4 and argv[0] == "emit":
        m, q = build_paths(argv[1])
        out, report = recalibrate(load_model(m), load_scales(q), load_scales(argv[2]))
        onnx.save(out, argv[3])
        print(json.dumps(report))
        return 0
    if len(argv) == 2 and argv[0] == "explain":
        _explain(argv[1])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
