"""Move a compiled single-op U16 program (AX650, float32 IO) to another calibration.

``retarget(template, op, template_q, target_q, inputs)`` returns the program of
the same op and shape at the target calibration, from one native Pulsar2
build at any other calibration; ``calibrate(directory, op)`` computes that
calibration from the program's samples, so no Pulsar2 build is needed per
calibration. Ops: Mul, Add (also with a constant operand, and the
attention-mask Add), Div, ReduceMean, Sqrt, Sigmoid, Softmax and dynamic
(two-input) MatMul, built with
``layer_configs: [{op_types: [<op>], data_type: U16}]``. A constant-weight
MatMul is a FullyConnected (``linear_emit.emit_linear_u16``).

A fused program (several ops in one native build: an RMSNorm, a SiLU block,
a RoPE apply, a fused attention) goes through ``retarget_graph(template,
model, template_q, target_q)`` with ``calibrate_graph(directory)``: the same
rules per graph node; when the template has one CORE job per compute node,
each job holds the formulas of the node its template records point to (Pulsar2
does not run the jobs in ONNX node order), in any CORE job otherwise. Two Adds
at one shift share one ``0x1ea0`` write, and a target whose Adds shift
differently is refused: the elided write cannot be restored.

How a record is found
---------------------

Every A1 record of the template and every 2-byte-aligned ``npu_params`` word
is compared with candidate formulas evaluated on the template's calibration:
per tensor ``zp``, ``float32(1 / s)`` and ``float32(s)``, and the op's own
formulas (``candidates``). A match is rewritten with the same formula on the
target calibration. In the main engine a QUANT job may only hold an input's
``1 / s`` and ``zp``, a DEQUANT job only the output's ``s`` and ``zp``;
formulas tied to one register are written ``name@0xREG``. A record that
matches two formulas with different target values is refused (pick a
template whose tensors differ).

The formulas are the U8 rules of ``graph_stitch`` at 16 bits, plus:

* Add: ``0x1ea0 = 15 - k``, with ``k`` the smallest shift that stores both
  scale ratios as u16 ``Q(15 - k)`` (``add_shift``; a ratio of exactly 1.0
  keeps ``k = 0``), and the zero-point offset in ``Q(15 - k)``; with a
  constant operand the U16 constant comes first in ``npu_params`` and the
  ratio constant follows at byte 2.
* dynamic MatMul: ``npu_params`` holds lane blocks ``zp_y``, ``M``, ``zp_y``,
  ``256 M`` (``M = s_a s_b / s_y``).
* Sigmoid: a 256-entry u16 table, one entry per 256 input codes
  (``0x1050..0x1840``, two per register; entry 256 in ``0x1850``), in
  single precision.

Write elision
-------------

Pulsar2 leaves out a write of the value a register already holds (the first
write of each register in a segment is always there: the PARAM and QUANT jobs
reset them). After the rewrite, a write that became redundant is dropped and
the segment is re-padded with zero records to a multiple of four. A record
the template already left out cannot be restored, so a template must have
zero points above 255 (smaller ones are not located) and distinct zero points
and scales; ``retarget`` refuses otherwise.

Checked offline against native builds (``tests/test_axera_u16_retarget.py``):
every pair of builds of one op and shape, the SmolLM2-135M layer-0 programs
and calibration variants, gives records equal to the native build of the
target calibration. ``calibrate`` reproduces Pulsar2's U16 MinMax exactly on
all but one of those builds (a ReduceMean output scale two ulps off).
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import tarfile
from collections.abc import Mapping, Sequence

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import graph_stitch as gs  # noqa: E402
import linear_emit as le  # noqa: E402
import misc_op_record_emit as mre  # noqa: E402
import short_unit_codec as suc  # noqa: E402
import width_retarget as wr  # noqa: E402

F = np.float32
OPS = (
    "Mul",
    "Add",
    "Div",
    "ReduceMean",
    "Sqrt",
    "Sigmoid",
    "Softmax",
    "MatMul",
    "Silu",
    "Abs",
    "ReduceMax",
)
#: the row-scaled norm's ops (``pf_rsnorm_probe``): Abs takes the per-tensor
#: formulas of its input and output; ReduceMax is a pass-through in U16 (no
#: record of its own, and its output shares its input's calibration, see
#: ``propagate``). A graph with one of them is calibrated with ``_mul_input``'s
#: zero-point rule and the windowed ReduceMean forward (``calibrate_graph``).
NEW_OPS = ("Abs", "ReduceMax")
SIGMOID_TABLE = 0x1050  # 128 registers, two u16 entries each
SIGMOID_TAIL = 0x1850
#: the table's 16 segments: input codes per entry as ``65536 / step`` (``0x100``:
#: 256) and each segment's first code ``<< 16``; Pulsar2 sometimes spaces them
#: unevenly (a Silu over a wide range), the formulas assume the even layout
TABLE_STEP, TABLE_START = 0x1860, 0x1960
ADD_SHIFT = 0x1EA0


def _fbits(v) -> int:
    return int(np.asarray(F(v)).view(np.uint32))


#: data-movement ops: no calibration of their own (Pulsar2's quant json can leave
#: their output, e.g. a fused program's graph output, without an entry)
PASS_THROUGH = ("Reshape", "Transpose", "Flatten", "Squeeze", "Unsqueeze")
#: data movement with several inputs or outputs (rotate-half inside a program:
#: ``Split`` a head into halves, ``Concat`` them swapped). Pulsar2 runs them
#: without a job and calibrates every side alike
MOVES = PASS_THROUGH + ("Split", "Concat")


def propagate(
    model, q: Mapping[str, tuple[float, int]]
) -> dict[str, tuple[float, int]]:
    """``q`` with every tensor that passes through a data-movement node without
    a calibration of its own given its neighbour's."""
    m = model if isinstance(model, onnx.ModelProto) else onnx.load(model)
    q = dict(q)
    consts = {i.name for i in m.graph.initializer}
    moves = [
        (n.input[0], n.output[0]) for n in m.graph.node if n.op_type in PASS_THROUGH
    ]
    for n in m.graph.node:  # every side of a Split / Concat alike
        if n.op_type in ("Split", "Concat"):
            ts = [t for t in [*n.input, *n.output] if t and t not in consts]
            moves += [(a, b) for a in ts for b in ts if a != b]
    # a ReduceMax's output is not calibrated on its own: Pulsar2 keeps its input's
    # scale and zero point (the quant json lists the output as overlapped with the
    # input), so it overrides whatever a MinMax of the output would give
    shares = [
        (n.input[0], n.output[0]) for n in m.graph.node if n.op_type == "ReduceMax"
    ]
    changed = True
    while changed:
        changed = False
        for a, b in moves:
            if a in q and b not in q:
                q[b], changed = q[a], True
            elif b in q and a not in q:
                q[a], changed = q[b], True
        for a, b in shares:
            if a in q and q.get(b) != q[a]:
                q[b], changed = q[a], True
    return q


def quant_of(directory: str) -> dict[str, tuple[float, int]]:
    """``{tensor: (scale, zero point)}`` of a build's ``quant_axmodel.json``
    (per-tensor entries only; entries without a scale, such as a Reshape's
    shape constant, are skipped). With the build's ``model.onnx`` present, a
    data-movement node's uncalibrated side takes its neighbour's (``propagate``)."""
    q = json.load(open(os.path.join(directory, "out", "quant", "quant_axmodel.json")))
    out = {}
    for tc in q["tensor_configs"].values():
        for t, c in tc.items():
            v = q["values"].get(
                str(c["hash"])
            )  # absent: an inner tensor's consumer side
            if v is not None and v.get("scale") is not None and len(v["scale"]) == 1:
                out[t] = (float(v["scale"][0]), int(v["zero_point"][0]))
    model = os.path.join(directory, "model.onnx")
    return propagate(model, out) if os.path.exists(model) else out


# ---- calibration --------------------------------------------------------------------
#: Pulsar2's smallest scale: an eps constant of 1e-6 gets float32(1e-10), not 1e-6 / 65535
MIN_SCALE = F(1e-10)


def unsigned_minmax(
    lo: float, hi: float, symmetric_32767: bool = True, f32: bool = False
) -> tuple[float, int]:
    """U16 MinMax of a range folded with 0: ``zp = rint(-lo / s)``. At a
    symmetric range (``lo == -hi``, where ``-lo / s`` is exactly 32767.5)
    Pulsar2 gives 32767 for a Mul operand (RoPE's sin table) but rint's 32768
    for an Add input (a [-1, 1] calibration); ``symmetric_32767`` picks the
    former. Fitted on five such builds only; ``calibrate`` passes it per op.
    The scale is at least ``MIN_SCALE`` (a tiny constant such as an RMSNorm's
    eps)."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    s = max(F((hi - lo) / 65535.0), MIN_SCALE)
    if symmetric_32767 and lo == -hi and hi > 0:
        return float(s), 32767
    if f32:  # -lo / s divided in float32: at a half-code boundary this is the rounding
        return float(s), int(np.clip(np.rint(F(-lo) / s), 0, 65535))
    return float(s), int(np.clip(np.rint(-lo / float(s)), 0, 65535))


def signed_absmax(absmax: float) -> tuple[float, int]:
    """A dynamic MatMul operand: signed, ``s = max|x| / 32767.5``, zero point 0."""
    return float(F(F(absmax) / F(32767.5))), 0


def _samples(path: str) -> list[np.ndarray]:
    with tarfile.open(path) as t:
        members = sorted(
            (m for m in t.getmembers() if m.isfile()),
            key=lambda m: int(os.path.basename(m.name).split(".")[0]),
        )
        return [np.load(io.BytesIO(t.extractfile(m).read())) for m in members]


def _data(directory, ins, samples):
    """Each input's samples: ``samples[name]`` when given (the arrays the
    tars were written from), else read from ``dataset/<name>.tar``."""
    if samples is not None:
        return {n: list(samples[n]) for n in ins}
    return {n: _samples(os.path.join(directory, "dataset", f"{n}.tar")) for n in ins}


def calibrate(
    directory: str, op: str, samples: Mapping | None = None
) -> dict[str, tuple[float, int]]:
    """The U16 calibration of a single-op build directory (``model.onnx``,
    ``dataset/<input>.tar``) from its samples: MinMax of each input, of each
    constant operand, and of the float op's output over every sample.
    ``samples``: ``{input: arrays}`` in place of the tars."""
    m = onnx.load(os.path.join(directory, "model.onnx"))
    consts = {i.name for i in m.graph.initializer}
    ins = [i.name for i in m.graph.input if i.name not in consts]
    data = _data(directory, ins, samples)
    ref = ReferenceEvaluator(m)
    ys = [
        ref.run(None, {n: data[n][j].astype(F) for n in ins})[0]
        for j in range(len(data[ins[0]]))
    ]
    q = {}
    for n in ins:
        a = np.concatenate([x.reshape(-1) for x in data[n]])
        q[n] = (
            signed_absmax(np.abs(a).max())
            if op == "MatMul"
            else unsigned_minmax(a.min(), a.max(), symmetric_32767=op == "Mul")
        )
    if op != "MatMul":
        for t in m.graph.initializer:
            a = onnx.numpy_helper.to_array(t)
            q[t.name] = unsigned_minmax(a.min(), a.max())
    y = np.concatenate([v.reshape(-1) for v in ys])
    q[m.graph.output[0].name] = unsigned_minmax(y.min(), y.max())
    return q


# ---- record formulas ----------------------------------------------------------------
def add_shift(ra: float, rb: float) -> int:
    """Add's shift ``k`` (``0x1ea0 = 15 - k``): the smallest ``k`` with
    ``round(max(ra, rb) * 2^(15 - k)) <= 32768``, the ratios stored as u16
    ``Q(15 - k)``. A ratio of exactly 1.0 (an eps Add whose input and output
    share a scale) keeps ``k = 0`` and stores 32768; ``graph_stitch.add_const``'s
    U8 rule (``ratio < 1``) would shift it."""
    k = 0
    while round(max(ra, rb) * 2.0 ** (15 - k)) > 32768:
        k += 1
    return k


def candidates(
    op: str,
    q: Mapping[str, tuple[float, int]],
    inputs: Sequence[str],
    out: str = "y",
    attrs: Mapping | None = None,
) -> dict[str, int]:
    """Formula name -> 32-bit record value on calibration ``q``."""
    if op not in OPS:
        raise ValueError(f"{op}: not a retargetable U16 op ({', '.join(OPS)})")
    if op == "ReduceMax":
        # pass-through: no formula of its own (its output takes its input's calibration)
        return {}
    s = {t: v[0] for t, v in q.items()}
    z = {t: v[1] for t, v in q.items()}
    c = {}
    for t in q:
        c[f"zp:{t}"] = z[t]
        c[f"inv:{t}"] = _fbits(1 / F(s[t]))
        c[f"s:{t}"] = _fbits(s[t])
    if op == "Mul":
        a, b = inputs
        c["mul_lane"] = _fbits(s[out] / (s[a] * s[b]))
    elif op == "Add":
        a, b = inputs
        ra, rb = s[a] / s[out], s[b] / s[out]
        k = add_shift(ra, rb)
        c[f"add_shift@{ADD_SHIFT:#06x}"] = 15 - k
        c["add_offset"] = int(
            (z[out] - z[a] * gs._f32(ra) - z[b] * gs._f32(rb)) * 2 ** (15 - k)
        )
        c["add_const"] = struct.unpack(
            "<I",
            struct.pack("<2H", *[int(round(r * 2.0 ** (15 - k))) for r in (ra, rb)]),
        )[0]
    elif op == "Div":
        a, b = inputs
        c["div_lane"] = _fbits(s[a] / (s[b] * s[out]))
    elif op == "ReduceMean":
        (x,) = inputs
        n = int((attrs or {})["count"])
        c["rm_lane"] = gs.reducemean_lane(s[x], s[out], n)
        c["rm_zp"] = z[x] * min(n, gs.REDUCE_WINDOW)
    elif op == "MatMul":
        a, b = inputs
        m = F(s[a] * s[b] / s[out])
        c["mm_scale"] = _fbits(m)
        c["mm_scale256"] = _fbits(F(m * F(256)))
        c["mm_zp_f"] = _fbits(F(z[out]))
    elif op in ("Sigmoid", "Silu"):
        (x,) = inputs
        codes = np.arange(257) * 256
        if op == "Silu":  # the last entry is the last code's (65535), not 65536's
            codes[256] = 65535
        xf = ((codes - z[x]).astype(F) * F(s[x])).astype(F)
        sg = (F(1) / (F(1) + np.exp(-xf).astype(F))).astype(F)
        if op == "Silu":  # Pulsar2's fused x * sigmoid(x): the same table layout
            sg = (xf * sg).astype(F)
        t = np.clip(np.rint((sg / F(s[out])).astype(F)) + z[out], 0, 65535).astype(int)
        for k in range(128):
            c[f"sig_t{k}@{SIGMOID_TABLE + 16 * k:#06x}"] = int(t[2 * k]) | (
                int(t[2 * k + 1]) << 16
            )
        c[f"sig_tail@{SIGMOID_TAIL:#06x}"] = int(t[256])
    return {k: v & 0xFFFFFFFF for k, v in c.items()}


def _allowed(role: str, name: str, inputs, out: str, reg: int | None) -> bool:
    if "@" in name and int(name.split("@")[1], 16) != reg:
        return False
    if role == "QUANT":
        return name in {f"inv:{t}" for t in inputs} | {f"zp:{t}" for t in inputs}
    if role == "DEQUANT":
        return name in (f"s:{out}", f"zp:{out}")
    return True


def _segs(model) -> list[list[bytes]]:
    mc = bytes(mre.mcode_initializer(model).raw_data)
    return [gs.records(r) for r in suc.decode_segments(mc)]


def _base(name: str) -> str:
    """A candidate's formula name without its node prefix (``n<i>|``)."""
    return name.split("|", 1)[-1]


def _rewrite(template, ct, cn, allowed, n_core=None, fixed=None) -> onnx.ModelProto:
    """Rewrite every A1 record and ``npu_params`` word of ``template`` that
    matches one candidate of ``ct`` (allowed by ``allowed(tag, name, reg)``,
    ``tag`` one of QUANT, DEQUANT, PARAM, CORE<i> (the i-th CORE job of the
    main engine) or OTHER (other segments)) with its ``cn`` value, then
    apply Pulsar2's write elision and re-pad the segments. ``fixed``:
    ``(start, end)`` byte ranges of ``npu_params`` left as they are (weight tables)."""
    model = gs.load_model(template)
    by_val: dict[int, list[str]] = {}
    for k, v in sorted(ct.items()):
        if _base(k).startswith("zp:") and v < 256:
            if cn[k] != v:
                raise ValueError(
                    f"template {k} = {v}: below 256 it cannot be located in the records, "
                    f"and the target needs {cn[k]}; use a template whose zero points are > 255"
                )
            continue
        by_val.setdefault(v, []).append(k)
    segs = _segs(model)
    tag_at, core = {}, 0
    for role, a, b, _ in le._jobs(segs[gs.MAIN]):
        tag = role
        if role == "CORE":
            tag, core = f"CORE{core}", core + 1
        tag_at.update({i: tag for i in range(a, b)})
    if n_core is not None and core != n_core:
        raise ValueError(f"the template has {core} CORE jobs for {n_core} graph nodes")
    touched = [set() for _ in segs]
    for si, seg in enumerate(segs):
        for i, r in enumerate(seg):
            if r[0] != gs.A1:
                continue
            reg, val = gs._reg(r), gs._val(r)
            tag = tag_at.get(i, "OTHER") if si == gs.MAIN else "OTHER"
            names = [n for n in by_val.get(val, []) if allowed(tag, n, reg)]
            if len({cn[n] for n in names}) > 1:
                raise ValueError(
                    f"segment {si} record {i} ({reg:#06x} = {val:#x}) matches {names}: "
                    "ambiguous template"
                )
            if names:
                seg[i] = gs._rec(gs.A1, reg, cn[names[0]])
                touched[si].add(reg)
    # write elision on the calibration registers
    tsegs = _segs(model)
    for si, seg in enumerate(segs):
        if not touched[si]:
            continue
        st_t, st_n, keep = {}, {}, []
        for r, rt in zip(seg, tsegs[si]):
            if r[0] == gs.A1 and gs._reg(r) in touched[si]:
                reg = gs._reg(r)
                if st_t.get(reg) == gs._val(rt):
                    raise ValueError(
                        f"the template repeats a write to {reg:#06x}; a template with "
                        "distinct zero points and scales is needed"
                    )
                st_t[reg] = gs._val(rt)
                if st_n.get(reg) == gs._val(r):
                    continue
                st_n[reg] = gs._val(r)
            keep.append(r)
        while keep and keep[-1] == bytes(8):
            keep.pop()
        segs[si] = keep + [bytes(8)] * (-len(keep) % 4)
    result = wr.with_segments(model, segs)
    params = next((t for t in result.graph.initializer if t.name == "npu_params"), None)
    if params is not None:
        lut = {}
        for val, ns in by_val.items():
            names = [n for n in ns if not _base(n).startswith(("zp:", "inv:", "s:"))]
            if len(names) == 1 and 0 <= val < 1 << 32:
                lut[val] = cn[names[0]]
        params.raw_data = _rewrite_params(bytes(params.raw_data), lut, fixed or ())
    return result


def _rewrite_params(raw: bytes, lut: Mapping[int, int], fixed) -> bytes:
    """Every 32-bit word of ``npu_params`` at an even offset whose value is a key
    of ``lut`` rewritten to ``lut[value]``, in offset order (a rewrite changes
    the word two bytes on, which is then read rewritten), outside the
    ``fixed`` byte ranges. The candidate offsets are found with numpy on the
    original bytes; after a rewrite only the next offset is re-read (a plain
    scan of a 10 MB table took seconds per program)."""
    p = bytearray(raw)
    if len(p) < 4 or not lut:
        return bytes(p)
    a = np.frombuffer(raw, np.uint8)
    off = np.arange(0, len(p) - 3, 2)
    w = (
        a[off].astype(np.uint32)
        | a[off + 1].astype(np.uint32) << 8
        | a[off + 2].astype(np.uint32) << 16
        | a[off + 3].astype(np.uint32) << 24
    )
    free = np.ones(len(off), bool)
    for lo, hi in fixed:
        free &= ~((off + 4 > lo) & (off < hi))
    keys = np.fromiter(lut, np.uint32, len(lut))
    cand = off[free & np.isin(w, keys)].tolist()
    last = len(p) - 4

    def visit(o):
        val = struct.unpack_from("<I", p, o)[0]
        if val in lut:
            struct.pack_into("<I", p, o, lut[val])
            return True
        return False

    ci, redo = 0, None
    while ci < len(cand) or redo is not None:
        if redo is not None and (ci >= len(cand) or redo <= cand[ci]):
            o, redo = redo, None
            if ci < len(cand) and cand[ci] == o:
                ci += 1
            free_o = o <= last and not any(o + 4 > lo and o < hi for lo, hi in fixed)
            wrote = free_o and visit(o)
        else:
            o, ci = cand[ci], ci + 1
            wrote = visit(o)
        if wrote:
            redo = o + 2
    return bytes(p)


def retarget(
    template,
    op: str,
    template_q: Mapping[str, tuple[float, int]],
    target_q: Mapping[str, tuple[float, int]],
    inputs: Sequence[str],
    out: str = "y",
    attrs: Mapping | None = None,
) -> onnx.ModelProto:
    """The template program at calibration ``target_q``. Both calibrations
    are keyed by the template's tensor names (``inputs`` and ``out``); the
    result keeps the template's IO names."""
    ct = candidates(op, template_q, inputs, out, attrs)
    cn = candidates(op, target_q, inputs, out, attrs)

    def allowed(tag, name, reg):
        return _allowed(tag, name, inputs, out, reg)

    return _rewrite(template, ct, cn, allowed)


# ---- fused (multi-op) programs ---------------------------------------------------------
UNARY = ("Sqrt", "Sigmoid", "Softmax", "ReduceMean", "Silu")
#: a node folded into another one's job (``fuse_silu``): no job, no formulas
FOLDED = "Folded"


def fuse_silu(nodes, q) -> list:
    """``nodes`` (``graph_nodes``) with each ``sg = Sigmoid(x)``,
    ``y = Mul(x, sg)`` pair whose ``sg`` has no calibration in ``q`` as one
    ``Silu`` node. Pulsar2 fuses the pair into one onnx ``Silu`` op when ``x``
    is a FullyConnected's output (a SiLU after the gate linear), and quantizes
    ``x`` at U16 only with ``Silu`` in the build's U16 ``op_types`` (else U8).
    Its job is a U16 Sigmoid-layout table of ``x * sigmoid(x)``. The Mul
    becomes a ``FOLDED`` node (no job, no formulas)."""
    out = list(nodes)
    for i, (op, ins, sg, _) in enumerate(nodes):
        if op != "Sigmoid" or sg in q:
            continue
        x = ins[0]
        js = [
            j
            for j, n in enumerate(nodes)
            if n[0] == "Mul" and sorted(n[1]) == sorted((x, sg))
        ]
        users = [n for n in nodes if sg in n[1]]
        if len(js) == 1 and len(users) == 1:
            j = js[0]
            out[i] = ("Silu", (x,), nodes[j][2], {})
            out[j] = (FOLDED, (), nodes[j][2], {})
    return out


def graph_nodes(model) -> list[tuple[str, tuple, str, dict]]:
    """``(op, inputs, output, attrs)`` per node of a fused program's float
    graph (``model.onnx``); ReduceMean's ``count`` is the reduced width."""
    m = onnx.shape_inference.infer_shapes(
        model if isinstance(model, onnx.ModelProto) else onnx.load(model)
    )
    dims = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in [*m.graph.input, *m.graph.value_info, *m.graph.output]
    }
    out = []
    for n in m.graph.node:
        attrs = {}
        if n.op_type == "ReduceMean":
            sh = dims[n.input[0]]
            axes = next((list(a.ints) for a in n.attribute if a.name == "axes"), [-1])
            attrs["count"] = int(np.prod([sh[a] for a in axes]))
        if len(n.output) > 1:  # a Split: every output
            attrs["outputs"] = list(n.output)
        out.append((n.op_type, tuple(n.input), n.output[0], attrs))
    return out


def _tensor_candidates(q) -> dict[str, int]:
    """Per-tensor formulas only: a data-movement node's (it has none of its own)."""
    c = {}
    for t, (s, z) in q.items():
        c[f"zp:{t}"], c[f"inv:{t}"], c[f"s:{t}"] = z, _fbits(1 / F(s)), _fbits(s)
    return {k: v & 0xFFFFFFFF for k, v in c.items()}


def _graph_candidates(nodes, q) -> dict[str, int]:
    c = {}
    for i, (op, ins, out, attrs) in enumerate(nodes):
        sub = {t: q[t] for t in {*ins, out, *attrs.get("outputs", ())} if t in q}
        if op == FOLDED:
            continue
        if op in MOVES or op == "FullyConnected":
            got = _tensor_candidates(sub)
        else:
            used = ins[:1] if op in UNARY else ins
            got = candidates(op, sub, used, out, attrs)
        for k, v in got.items():
            c[f"n{i}|{k}"] = v
    return c


def _core_jobs(template) -> int:
    segs = _segs(gs.load_model(template))
    return sum(1 for role, *_ in le._jobs(segs[gs.MAIN]) if role == "CORE")


#: calibration registers (lanes, zero points, the Sigmoid table, Add's shift and
#: offset) all sit at or above this; below it are source / destination address
#: writes, which a candidate value can coincide with
MIN_CALIB_REG = 0x0F00
#: at most this many pairs of compute nodes run as one CORE job (strict mode)
MAX_MERGED = 4


def _assign_core_jobs(main, comp, ct, nodes=None, first=0) -> dict[int, str]:
    """CORE job j -> node ``n<i>``, one job per compute node. Pulsar2 orders
    the jobs topologically, not in ONNX node order (a fused q / k norm + RoPE
    runs ``x * x`` first). The assignment is the topological order of the
    graph whose nodes' template formulas explain most of the jobs'
    calibration records in total (dynamic programming over the sets of
    computed nodes; ONNX order among equals). Restricting it to topological
    orders matters when tensors share a calibration (a Div by an RMS of
    normalized values, ``d = u / r``: ``u`` and ``d`` alike, ``m``, ``e``
    and ``r`` alike): a score alone then puts a Div before its Sqrt."""
    cores = [(a, b) for role, a, b, _ in le._jobs(main) if role == "CORE"]
    cores = cores[first:]
    by_node: dict[str, set[int]] = {}
    for k, v in ct.items():
        node, base = k.split("|", 1)
        if not (base.startswith("zp:") and v < 256):
            by_node.setdefault(node, set()).add(v)
    vals = [
        [
            gs._val(main[r])
            for r in range(a, b)
            if main[r][0] == gs.A1 and gs._reg(main[r]) >= MIN_CALIB_REG
        ]
        for a, b in cores
    ]
    sets = [by_node.get(f"n{i}", set()) for i in comp]

    def score(j, cs):
        u = set().union(*(sets[c] for c in cs))
        return sum(v in u for v in vals[j])

    # tensor -> the compute node that produces it (through data-movement nodes)
    prod: dict[str, int | None] = {}
    for i, (op, ins, out, attrs) in enumerate(nodes or []):
        if op == FOLDED:
            continue
        if op in MOVES:  # a Concat's halves come from one node in practice
            src = next((prod[t] for t in ins if prod.get(t) is not None), None)
            for o in attrs.get("outputs", [out]):
                prod[o] = src
        else:  # a FullyConnected runs outside the main engine's CORE jobs
            prod[out] = None if op == "FullyConnected" else i
    in_comp = set(comp)  # (with ``first``, the earlier jobs' nodes are not in it)
    deps = {
        i: {prod[t] for t in nodes[i][1] if prod.get(t) is not None} & in_comp
        if nodes
        else set()
        for i in comp
    }

    def ready(done):
        return [c for c, i in enumerate(comp) if i not in done and deps[i] <= done]

    # best total over topological orders: DP over the sets of computed nodes
    # (order ideals of the graph); ties to the lexicographically first ONNX
    # order. With fewer jobs than nodes, Pulsar2 ran some pairs of nodes as one
    # job: a job may then take two nodes (the second ready after the first).
    # With more jobs than nodes, the extra jobs hold no calibration record (data
    # movement, e.g. around a FullyConnected): such a job may take no node.
    merges = max(len(comp) - len(cores), 0)
    empties = max(len(cores) - len(comp), 0)
    allv = set().union(*sets) if sets else set()
    blank = [not any(v in allv for v in vals[j]) for j in range(len(cores))]
    if sum(blank) < empties:
        raise ValueError(
            f"{len(cores)} CORE jobs for {len(comp)} compute nodes, only "
            f"{sum(blank)} without calibration records"
        )
    states: dict[frozenset, tuple[float, tuple]] = {frozenset(): (0.0, ())}
    for j in range(len(cores)):
        nxt: dict[frozenset, tuple[float, tuple]] = {}

        def offer(key, cand):
            best = nxt.get(key)
            if (
                best is None
                or cand[0] > best[0]
                or (cand[0] == best[0] and cand[1] < best[1])
            ):
                nxt[key] = cand

        for done, (sc, path) in states.items():
            skipped = sum(1 for p in path if not p)
            spare = merges - (len(done) - (j - skipped))
            if blank[j] and skipped < empties:
                offer(done, (sc, path + ((),)))
            for c in ready(done):
                d1 = done | {comp[c]}
                offer(d1, (sc + score(j, (c,)), path + ((c,),)))
                if spare > 0:
                    for c2 in ready(d1):
                        if c2 > c or comp[c] in deps[comp[c2]]:
                            offer(
                                d1 | {comp[c2]},
                                (sc + score(j, (c, c2)), path + ((c, c2),)),
                            )
        if not nxt:  # the jobs are not a topological order of this graph
            raise ValueError("no topological order of the graph fits the CORE jobs")
        states = nxt
    full = {k: v for k, v in states.items() if len(k) == len(comp)}
    if not full:
        raise ValueError("no topological order of the graph fits the CORE jobs")
    _, path = max(
        full.values(), key=lambda v: (v[0], [-c[0] if c else 1 for c in v[1]])
    )
    return {first + j: "+".join(f"n{comp[c]}" for c in cs) for j, cs in enumerate(path)}


def _assign_after_attention(main, comp, ct, nodes) -> tuple[dict, set]:
    """A whole layer with its attention first (``alyrqk``): Pulsar2 runs the
    attention's jobs (a Softmax in two, data movement around them) before
    the rest, whose nodes are then assigned strictly (``_assign_core_jobs``)
    from the latest CORE job that fits; the attention's nodes may sit in any
    job before it. ``({}, set())`` when no such split fits."""
    fc = [i for i, n in enumerate(nodes) if n[0] == "FullyConnected"]
    if not fc or not any(n[0] == "Softmax" for n in nodes):
        return {}, set()
    up, todo = set(), [nodes[fc[0]][1][0]]  # the nodes before the first linear
    while todo:
        t = todo.pop()
        for i, (op, ins, out, attrs) in enumerate(nodes):
            if (out == t or t in attrs.get("outputs", ())) and i not in up:
                up.add(i)
                todo += list(ins)
    head = {f"n{i}" for i in comp if i in up}
    rest = [i for i in comp if i not in up]
    # the rest starts at the first CORE job holding a record that only one of
    # its first nodes (no producer among the rest: the residual Add) explains
    prod = {}
    for i, (op, ins, out, attrs) in enumerate(nodes):
        for o in attrs.get("outputs", [out]):
            prod[o] = i
    starts = {
        f"n{i}"
        for i in rest
        if not any(prod.get(t) in rest for t in nodes[i][1] if t in prod)
    }
    owners: dict[int, set] = {}
    for k, v in ct.items():
        owners.setdefault(v, set()).add(k.split("|", 1)[0])
    own = {v for v, o in owners.items() if len(o) == 1 and o <= starts}
    cores = [(a, b) for role, a, b, _ in le._jobs(main) if role == "CORE"]
    for first, (a, b) in enumerate(cores):
        hit = any(
            main[r][0] == gs.A1
            and gs._reg(main[r]) >= MIN_CALIB_REG
            and gs._val(main[r]) in own
            for r in range(a, b)
        )
        if hit:
            try:
                return _assign_core_jobs(main, rest, ct, nodes, first=first), head
            except ValueError:
                return {}, set()
    return {}, set()


def _check_even_tables(main) -> None:
    """Refuse a template whose Sigmoid / Silu table is not spaced evenly (one
    entry per 256 input codes): its entries sit at other codes than the
    formulas'. Pulsar2 picks uneven segments for some calibrations (seen on a
    Silu over [-14, 16]); a retarget always writes the even table. A
    Sigmoid / Silu job's steps stay near ``0x100`` (seen 0xe2..0x10c); a job with
    other steps is another table op's (a Softmax in the same program)."""
    for _, a, b, _ in le._jobs(main):
        regs = {
            gs._reg(main[r]): gs._val(main[r])
            for r in range(a, b)
            if main[r][0] == gs.A1
        }
        if SIGMOID_TABLE not in regs:
            continue
        steps = [regs.get(TABLE_STEP + 16 * k, 0x100) for k in range(16)]
        if not all(0x80 <= v <= 0x200 for v in steps):
            continue  # another table op's job (a Softmax's exp: steps of 1 and more)
        for k in range(16):
            step = regs.get(TABLE_STEP + 16 * k, 0x100)
            start = regs.get(TABLE_START + 16 * k, k << 28)
            if (step, start) != (0x100, k << 28):
                raise ValueError(
                    f"the template's table is not spaced evenly (segment {k}: "
                    f"step {step:#x}, start {start:#x}); use another template"
                )


def _check_add_shift_writes(main, nodes, assign, cn) -> None:
    """Refuse a template that elided an Add-shift write the target needs.
    Pulsar2 writes ``0x1ea0`` in an Add's CORE job only when the register
    does not already hold that Add's shift (PARAM jobs reset it to 0, other
    jobs may write it too). Walking the main segment with the target's
    shifts, an Add whose shift differs from the register's value needs a
    write in its own job; when the template left it out, it cannot be
    restored (the Add would run at the previous shift)."""
    state, core = None, 0
    for role, a, b, _ in le._jobs(main):
        writes = [
            gs._val(main[r])
            for r in range(a, b)
            if main[r][0] == gs.A1 and gs._reg(main[r]) == ADD_SHIFT
        ]
        node = None
        if role == "CORE":
            node, core = assign.get(core), core + 1
        adds = [
            n for n in (node or "").split("+") if n and nodes[int(n[1:])][0] == "Add"
        ]
        if len(adds) > 1:
            raise ValueError(f"one CORE job holds two Adds ({node})")
        if adds:
            want = cn[f"{adds[0]}|add_shift@{ADD_SHIFT:#06x}"]
            if want != state and not writes:
                raise ValueError(
                    f"the target's {node} Add needs {ADD_SHIFT:#06x} = {want} (the "
                    f"register holds {state}): an elided Add-shift write in the "
                    "template; use a template whose Adds shift differently"
                )
            state = want
        elif writes:
            state = writes[-1]


def retarget_graph(
    template,
    model,
    template_q: Mapping[str, tuple[float, int]],
    target_q: Mapping[str, tuple[float, int]],
    out: str | None = None,
    weights=None,
    attention_first: bool = False,
) -> onnx.ModelProto:
    """``retarget`` for a fused program (several ops in one native build).
    ``model`` is its float graph; the calibrations cover every tensor, inner
    ones included (``quant_of`` / ``calibrate_graph``).

    When the template has one CORE job per compute node (RMSNorm, SiLU, RoPE
    apply, a per-head q / k norm with RoPE; data-movement nodes have no job),
    each CORE job holds one node's formulas only (``_assign_core_jobs``) and
    QUANT jobs hold a graph input's ``1 / s`` and zero point: the strongest
    disambiguation. Otherwise (fused attention: Softmax takes two CORE jobs, a
    Transpose quantizes its own output; a whole-layer attention with
    Reshapes), any node's formulas may sit in any CORE job and a QUANT job may
    hold any tensor's ``1 / s`` and zero point; a record matching two formulas
    with different targets is still refused. A DEQUANT job holds an
    output's ``s`` and zero point (any graph output unless ``out`` names one); PARAM jobs and the other segments any
    node's. Records below ``MIN_CALIB_REG`` are never rewritten.
    Data-movement nodes (``MOVES``) contribute per-tensor formulas
    only. ``attention_first``: a whole layer with its attention first
    (``alyrqk``, too many jobs for the strict rule):
    ``_assign_after_attention``."""
    m = model if isinstance(model, onnx.ModelProto) else onnx.load(model)
    nodes = graph_nodes(m)
    consts = {i.name for i in m.graph.initializer}
    # a MatMul by a constant is a FullyConnected: no main-engine CORE job of its
    # own; its weights / requantization are a table in npu_params (_linear_table)
    fc = [i for i, n in enumerate(nodes) if n[0] == "MatMul" and n[1][1] in consts]
    for i in fc:
        nodes[i] = ("FullyConnected", nodes[i][1][:1], nodes[i][2], {})
        # the weights carry no calibration record (the table has them)
        wn = m.graph.node[i].input[1]
        template_q = {k: v for k, v in template_q.items() if k != wn}
        target_q = {k: v for k, v in target_q.items() if k != wn}
    graph_ins = [i.name for i in m.graph.input if i.name not in consts]
    outs = {out} if out else {o.name for o in m.graph.output}
    template_q, target_q = propagate(m, template_q), propagate(m, target_q)
    nodes = fuse_silu(nodes, template_q)
    ct, cn = _graph_candidates(nodes, template_q), _graph_candidates(nodes, target_q)
    main = _segs(gs.load_model(template))[gs.MAIN]
    if any(n[0] in ("Sigmoid", "Silu") for n in nodes):
        _check_even_tables(main)
    comp = [
        i
        for i, n in enumerate(nodes)
        if n[0] not in MOVES and n[0] not in ("FullyConnected", FOLDED)
    ]
    # a CORE job per compute node, a few pairs of nodes run as one job, or a few
    # extra jobs without calibration records
    # (a Split / Concat may take a job without records of its own)
    moves = sum(1 for n in nodes if n[0] in ("Split", "Concat"))
    strict = abs(len(comp) - _core_jobs(template)) <= MAX_MERGED + moves
    assign = {}
    if strict:
        try:
            assign = _assign_core_jobs_memo(main, comp, ct, nodes)
        except ValueError:  # e.g. a Softmax in two jobs: any node in any CORE job
            strict = False
    head: set = set()  # the attention's nodes when only the rest is assigned
    if attention_first and not strict:
        assign, head = _assign_after_attention(main, comp, ct, nodes)
        strict = bool(assign)
    if strict:
        _check_add_shift_writes(main, nodes, assign, cn)

    def allowed(tag, name, reg):
        node, base = name.split("|", 1)
        if reg is not None and reg < MIN_CALIB_REG:
            return False
        if "@" in base and int(base.split("@")[1], 16) != reg:
            return False
        if tag == "QUANT":
            if strict and not head:
                return base in {f"inv:{t}" for t in graph_ins} | {
                    f"zp:{t}" for t in graph_ins
                }
            return base.split(":", 1)[0] in ("inv", "zp")
        if tag == "DEQUANT":
            return base in {f"{p}:{o}" for o in outs for p in ("s", "zp")}
        if tag.startswith("CORE") and strict:
            j = int(tag[4:])
            if head and j not in assign:  # a job before the assigned ones
                return node in head
            return node in assign[j].split("+")  # "": none
        return True

    if not fc:
        return _rewrite(template, ct, cn, allowed)
    tmodel = gs.load_model(template)
    tp = bytes(
        next(i for i in tmodel.graph.initializer if i.name == "npu_params").raw_data
    )
    tables, zps = [], {}
    for i in fc:
        x, y = nodes[i][1][0], nodes[i][2]
        wname = m.graph.node[i].input[1]
        w = onnx.numpy_helper.to_array(
            next(t for t in m.graph.initializer if t.name == wname)
        )
        if weights is not None:
            w = weights[wname] if isinstance(weights, Mapping) else weights
        old_tab = _template_table(template, wname, template_q[x], template_q[y])
        at = tp.find(old_tab)
        if at < 0 or tp.find(old_tab, at + 1) >= 0:
            raise ValueError(f"{wname}: the template's weight table is not found once")
        new_tab = le.linear_params_u16(
            b"", np.asarray(w, F), *target_q[x], *target_q[y], table_only=True
        )
        tables.append((at, at + len(old_tab), new_tab))
        tz, nz = template_q[x][1], target_q[x][1]
        if zps.get(tz, nz) != nz:
            raise ValueError(
                f"two linears' inputs share zero point {tz} in the template: their "
                f"core {le.REG_CORE_ZP_X:#06x} records cannot be told apart"
            )
        zps[tz] = nz
    res = _rewrite(template, ct, cn, allowed, fixed=[t[:2] for t in tables])
    params = next(i for i in res.graph.initializer if i.name == "npu_params")
    p = bytearray(params.raw_data)
    for a, b, tab in tables:
        p[a:b] = tab
    params.raw_data = bytes(p)
    # each linear's core copy of zp_x (low half of 0x03d0, outside the main engine)
    segs, hits = _segs(res), dict.fromkeys(zps, 0)
    for si, seg in enumerate(segs):
        if si == gs.MAIN:
            continue
        for i, r in enumerate(seg):
            if r[0] == gs.A1 and gs._reg(r) == le.REG_CORE_ZP_X:
                tz = gs._val(r) & 0xFFFF
                if tz in zps:
                    v = (gs._val(r) & 0xFFFF0000) | zps[tz]
                    seg[i] = gs._rec(gs.A1, le.REG_CORE_ZP_X, v)
                    hits[tz] += 1
    if not all(hits.values()):
        raise ValueError(
            f"no core {le.REG_CORE_ZP_X:#06x} record holds a linear's zp_x"
        )
    return wr.with_segments(res, segs)


def exact_add_offset(
    q: Mapping[str, tuple[float, int]], a: str, b: str, out: str
) -> int:
    """An Add's offset word without Pulsar2's bias: the card computes
    ``(qa * A + qb * B + off) >> n`` with the ratios ``A``, ``B`` rounded to
    u16 ``Q(n)``; Pulsar2's ``off`` (``candidates``' ``add_offset``) takes the
    unrounded ratios, which leaves ``zp_a * (A / 2^n - r_a) + zp_b * (B / 2^n -
    r_b)`` output codes on every element (up to ~2 codes). This ``off =
    zp_y * 2^n - zp_a * A - zp_b * B`` makes the sum exact at the zero points."""
    (sa, za), (sb, zb), (sy, zy) = q[a], q[b], q[out]
    ra, rb = sa / sy, sb / sy
    n = 15 - add_shift(ra, rb)
    return (zy * 2**n - za * round(ra * 2.0**n) - zb * round(rb * 2.0**n)) & 0xFFFFFFFF


def exact_add_offsets(program, nodes, q) -> tuple[onnx.ModelProto, int]:
    """``program`` (a native or retargeted U16 program at calibration ``q``,
    ``nodes`` its ``graph_nodes``) with every Add's offset word replaced by
    ``exact_add_offset``; the records are found by value (Pulsar2's offset)
    as ``retarget`` finds them. Returns the program and the number of Adds
    whose offset changed."""
    ct, cn = {}, {}
    for i, (op, ins, out, _) in enumerate(nodes):
        if op != "Add":
            continue
        sub = {t: q[t] for t in (*ins, out)}
        old = candidates("Add", sub, ins, out)["add_offset"]
        if old != exact_add_offset(sub, *ins, out):
            ct[f"n{i}|add_offset"] = old
            cn[f"n{i}|add_offset"] = exact_add_offset(sub, *ins, out)

    def allowed(tag, name, reg):
        return reg in gs.ADD_OFFSET_REGS

    model = gs.load_model(program)
    params = next((i for i in model.graph.initializer if i.name == "npu_params"), None)
    # the offset is never in npu_params: leave its words alone
    fixed = [(0, len(params.raw_data))] if params is not None else None
    return _rewrite(model, ct, cn, allowed, fixed=fixed), len(ct)


#: template-only results reused across the programs of one shape (every layer
#: retargets the same template variants): CORE job assignments, weight tables
_MEMO: dict = {}
_MEMO_MAX = 256


def _memo(key, fn):
    if key not in _MEMO:
        if len(_MEMO) >= _MEMO_MAX:
            _MEMO.pop(next(iter(_MEMO)))
        _MEMO[key] = fn()
    return _MEMO[key]


def _assign_core_jobs_memo(main, comp, ct, nodes=None) -> dict[int, str]:
    """``_assign_core_jobs``, memoized on its whole input (a template's main
    segment, its candidates and the graph): the same for every layer."""
    key = (
        "assign",
        b"".join(main),
        tuple(comp),
        tuple(sorted(ct.items())),
        repr(nodes),
    )
    return dict(_memo(key, lambda: _assign_core_jobs(main, comp, ct, nodes)))


def _template_table(template, name, qx, qy) -> bytes:
    """``linear_params_u16``'s table of the template's own weights ``name`` at
    the template's calibration (``qx``, ``qy``), memoized per template file."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(template))), "model.onnx"
    )
    st = os.stat(path)
    key = ("table", path, st.st_mtime_ns, st.st_size, name, tuple(qx), tuple(qy))
    return _memo(
        key,
        lambda: le.linear_params_u16(
            b"", _template_weight(template, name), *qx, *qy, table_only=True
        ),
    )


def _template_weight(template, name) -> np.ndarray:
    """The template build's own weights, from the ``model.onnx`` beside its ``out/``."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(template))), "model.onnx"
    )
    tm = onnx.load(path)
    return onnx.numpy_helper.to_array(
        next(i for i in tm.graph.initializer if i.name == name)
    )


def _mul_input(m: onnx.ModelProto, t: str) -> bool:
    """A graph input all of whose consumers are Mul: its symmetric range's zero
    point is 32767 (``unsigned_minmax``'s rule for a Mul operand); any other
    tensor's is rint's 32768 (seen on a graph input read by Abs and Div, and on
    an inner tensor (``xs = x / s``) read by Mul). Used by graphs with
    ``NEW_OPS`` only: the older graphs keep ``calibrate_graph``'s default."""
    if t not in {i.name for i in m.graph.input}:
        return False
    uses = [n.op_type for n in m.graph.node if t in n.input]
    return bool(uses) and all(u == "Mul" for u in uses)


#: Pulsar2's ReduceMean sums in windows of this many elements (``REDUCE_WINDOW``):
#: a window's pairwise sum, the window sums' pairwise sum, then / n in float32.
#: Calibrated that way the scales of a ReduceMean and of the Add and Sqrt after it
#: match Pulsar2's MinMax to the ulp on the row-scaled norm's builds (seeds 3, 7, 99;
#: a plain float32 mean is one or two ulps off there). Used by ``calibrate_graph``
#: for graphs with ``NEW_OPS`` only (the older graphs' calibrations are unchanged).
CAL_WINDOW = 256


def _windowed_forward(m: onnx.ModelProto, strict: bool = True) -> onnx.ModelProto:
    """``m`` with each ReduceMean (last axis, width ``k * CAL_WINDOW``) replaced by
    the window sums and their division, for the float forward of ``calibrate_graph``.
    ``strict=False``: a ReduceMean of another width stays a plain mean (a width-128
    norm in a graph with ``NEW_OPS``), instead of raising."""
    m = onnx.ModelProto.FromString(m.SerializeToString())
    shaped = onnx.shape_inference.infer_shapes(m)
    dims = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in [*shaped.graph.input, *shaped.graph.value_info, *shaped.graph.output]
    }
    nodes, inits = [], []
    for n in m.graph.node:
        if n.op_type != "ReduceMean":
            nodes.append(n)
            continue
        src = dims[n.input[0]]
        width = src[-1]
        if width % CAL_WINDOW or len(src) != 2 or src[0] != 1:
            if not strict:
                nodes.append(n)
                continue
            raise ValueError(
                f"{n.output[0]}: ReduceMean width {width} is not a window multiple"
            )
        k, out = width // CAL_WINDOW, n.output[0]
        t = lambda name: f"__cal_{out}_{name}"  # noqa: E731
        inits += [
            onnx.numpy_helper.from_array(np.array([k, CAL_WINDOW], np.int64), t("shp")),
            onnx.numpy_helper.from_array(np.array([1], np.int64), t("ax1")),
            onnx.numpy_helper.from_array(np.array([1, k], np.int64), t("shp2")),
            onnx.numpy_helper.from_array(np.array([float(width)], F), t("n")),
        ]
        nodes += [
            onnx.helper.make_node("Reshape", [n.input[0], t("shp")], [t("w")]),
            onnx.helper.make_node(
                "ReduceSum", [t("w"), t("ax1")], [t("ws")], keepdims=0
            ),
            onnx.helper.make_node("Reshape", [t("ws"), t("shp2")], [t("ws2")]),
            onnx.helper.make_node(
                "ReduceSum", [t("ws2"), t("ax1")], [t("tot")], keepdims=1
            ),
            onnx.helper.make_node("Div", [t("tot"), t("n")], [out]),
        ]
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    m.graph.initializer.extend(inits)
    return m


def _fma(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``a * b + c`` in float32 with one rounding (float64 holds the exact float32
    product; the double rounding of the sum is the only emulation error)."""
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(
        F
    )


#: Sleef's expf (``expf_u10``, the FMA form), the exp of Pulsar2's float Sigmoid and
#: Softmax: range reduction by ln 2 (split L2U + L2L), a degree-6 polynomial, 2**n.
_SLEEF_INV_LN2 = F(1.442695040888963407359924681001892137426645954152985934135449406931)
_SLEEF_L2U = F(0.693145751953125)
_SLEEF_L2L = F(1.428606765330187045e-06)
_SLEEF_POLY = (
    0.000198527617612853646278381,
    0.00139304355252534151077271,
    0.00833336077630519866943359,
    0.0416664853692054748535156,
    0.166666671633720397949219,
    0.5,
)


def _sleef_exp(x: np.ndarray) -> np.ndarray:
    """Pulsar2's float exp (see ``_SLEEF_*``). Bit-exact on the Sigmoid probes (65,536
    and 4,010 inputs over [-88, 88], subnormal tail included) and the Softmax probes."""
    d = np.asarray(x, F)
    q = np.rint((d * _SLEEF_INV_LN2).astype(F)).astype(F)
    s = _fma(q, -_SLEEF_L2U, d)
    s = _fma(q, -_SLEEF_L2L, s)
    u = F(_SLEEF_POLY[0])
    for c in _SLEEF_POLY[1:]:
        u = _fma(u, s, F(c))
    u = _fma((s * s).astype(F), u, s)
    u = (u + F(1)).astype(F)
    return (u * np.exp2(q.astype(np.float64)).astype(F)).astype(F)


def _sigmoid_pulsar(x: np.ndarray) -> np.ndarray:
    """Pulsar2's float Sigmoid: ``1 / (1 + exp(-x))`` with ``_sleef_exp``."""
    x = np.asarray(x, F)
    return (F(1) / (F(1) + _sleef_exp(-x))).astype(F)


def _lanes_sum(v: np.ndarray, lanes: int) -> np.ndarray:
    """Sum of the last axis as Pulsar2's vector kernels do: 16 lanes (a zero-padded
    tail), the chunks added lane-wise in order, then the lanes halved pairwise."""
    n = v.shape[-1]
    pad = (-n) % lanes
    pv = np.concatenate([v, np.zeros(v.shape[:-1] + (pad,), F)], axis=-1)
    pv = pv.reshape(v.shape[:-1] + (-1, lanes))
    acc = pv[..., 0, :].copy()
    for j in range(1, pv.shape[-2]):
        acc = (acc + pv[..., j, :]).astype(F)
    while acc.shape[-1] > 1:
        h = acc.shape[-1] // 2
        acc = (acc[..., :h] + acc[..., h:]).astype(F)
    return acc[..., 0]


def _softmax_pulsar(x: np.ndarray) -> np.ndarray:
    """Pulsar2's float Softmax over the last axis: ``x - max``, ``_sleef_exp``, the
    16-lane sum (``_lanes_sum``), then every exp times the reciprocal of the sum."""
    x = np.asarray(x, F)
    e = _sleef_exp((x - x.max(axis=-1, keepdims=True)).astype(F))
    s = _lanes_sum(e, 16)[..., None]
    return (e * (F(1) / s)).astype(F)


def _model4(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """A MatMul of one or two rows as Pulsar2 accumulates it: four lanes, the first
    four terms into lane 0, then term k into lane (k - 4) % 4, each step a float32
    product and sum; the result is (lane0 + lane2) + (lane1 + lane3). Exact on the
    probes at K = 16, 32, 64, 2560 and 4096 (N = 256, 1024, 4096)."""
    R, K = X.shape
    acc = [np.zeros((R, W.shape[1]), F) for _ in range(4)]

    def add(a, k):
        return (a + (X[:, k : k + 1] * W[k][None, :]).astype(F)).astype(F)

    for k in range(min(4, K)):
        acc[0] = add(acc[0], k)
    for k in range(4, K):
        acc[(k - 4) % 4] = add(acc[(k - 4) % 4], k)
    return ((acc[0] + acc[2]).astype(F) + (acc[1] + acc[3]).astype(F)).astype(F)


def _seqfma_matmul(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """A MatMul of four rows: one sequential chain per output, each term a fused
    multiply-add (the product exact, one rounding per step)."""
    R, K = X.shape
    acc = np.zeros((R, W.shape[1]), F)
    for k in range(K):
        acc = (
            acc.astype(np.float64)
            + X[:, k : k + 1].astype(np.float64) * W[k][None, :].astype(np.float64)
        ).astype(F)
    return acc


def _matmul_pulsar(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Pulsar2's float MatMul by row count (2-D, or 3-D over a batch): 1-2 rows
    ``_model4``, 4 rows ``_seqfma_matmul``, other row counts numpy (not modelled)."""
    X = np.asarray(X, F)
    W = np.asarray(W, F)
    if X.ndim == 3:
        return np.stack(
            [
                _matmul_pulsar(X[b], W[b] if W.ndim == 3 else W)
                for b in range(X.shape[0])
            ]
        )
    if X.shape[0] in (1, 2):
        return _model4(X, W)
    if X.shape[0] == 4:
        return _seqfma_matmul(X, W)
    return (X @ W).astype(F)


#: Pulsar2's ReduceMean sum: 512-element blocks, 32 lanes (lane b takes index
#: b mod 32 of a block, sequentially), lanes r, r+8, r+16, r+24 combined per group
#: r, the 8 groups summed in order; then / n in float32. Revealed at K = 128, 256,
#: 2560 (``_RM_*``); the row-scaled norms of the Qwen3 build match it to the ulp.
_RM_BLOCK, _RM_LANES, _RM_GROUPS = 512, 32, 8


def _rowmean_sum(x: np.ndarray) -> np.ndarray:
    """``sum`` over the last axis of a (rows, K) float32 array (K >= 32)."""
    rows, K = x.shape
    lanes = [None] * _RM_LANES
    for base in range(0, K, _RM_BLOCK):
        for b in range(_RM_LANES):
            tmp = None
            for k in range(base + b, min(base + _RM_BLOCK, K), _RM_LANES):
                tmp = x[:, k].copy() if tmp is None else (tmp + x[:, k]).astype(F)
            if tmp is None:
                continue
            lanes[b] = tmp if lanes[b] is None else (lanes[b] + tmp).astype(F)
    per = _RM_LANES // _RM_GROUPS
    groups = []
    for r in range(_RM_GROUPS):
        acc = lanes[r]
        for j in range(1, per):
            acc = (acc + lanes[r + _RM_GROUPS * j]).astype(F)
        groups.append(acc)
    tot = groups[0]
    for r in range(1, _RM_GROUPS):
        tot = (tot + groups[r]).astype(F)
    return tot


def _rowmean_pulsar(x: np.ndarray) -> np.ndarray:
    """ReduceMean over the last axis as Pulsar2 sums it (``_rowmean_sum``, / K)."""
    x = np.asarray(x, F)
    K = x.shape[-1]
    return (_rowmean_sum(x.reshape(-1, K)) / F(K)).reshape(x.shape[:-1]).astype(F)


def _pulsar_op(n: onnx.NodeProto, env: dict) -> np.ndarray | None:
    """The modelled value of node ``n`` (a list of outputs), or None when ``n`` is
    not one of the modelled ops in the form the model covers."""
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
    x = env[n.input[0]] if n.input else None
    if n.op_type == "Sigmoid":
        return [_sigmoid_pulsar(x)]
    if (
        n.op_type == "Softmax"
        and x.ndim >= 1
        and attrs.get("axis", -1) in (-1, x.ndim - 1)
    ):
        return [_softmax_pulsar(x)]
    if n.op_type == "MatMul":
        w = env[n.input[1]]
        batched = x.ndim == 3 and w.ndim == 3 and w.shape[0] == x.shape[0]
        if (x.ndim == 2 and w.ndim == 2) or batched or (x.ndim == 3 and w.ndim == 2):
            return [_matmul_pulsar(x, w)]
        return None
    if (
        n.op_type == "ReduceMean"
        and len(n.input) == 1
        and x.ndim >= 2
        and x.shape[-1] >= _RM_LANES
    ):
        axes = list(attrs.get("axes", [-1]))
        if axes not in ([-1], [x.ndim - 1]) or attrs.get("keepdims", 1) != 1:
            return None
        return [_rowmean_pulsar(x)[..., None]]
    return None


def _pulsar_forward(m: onnx.ModelProto):
    """``run(feeds)``: every tensor of ``m`` (inputs and node outputs) for one sample,
    the float forward of ``calibrate_graph`` for ``NEW_OPS`` graphs: Sigmoid, Softmax,
    MatMul and ReduceMean as Pulsar2 computes them (the ``_*_pulsar`` models above),
    every other node by the onnx reference op of its own."""
    g = m.graph
    consts = {i.name: onnx.numpy_helper.to_array(i) for i in g.initializer}
    refs = []
    for n in g.node:
        ins = [i for i in n.input if i]
        nn = onnx.helper.make_node(n.op_type, ins, list(n.output), domain=n.domain)
        nn.attribute.extend(n.attribute)
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [nn],
                "node",
                [
                    onnx.helper.make_tensor_value_info(i, onnx.TensorProto.FLOAT, None)
                    for i in ins
                ],
                [
                    onnx.helper.make_tensor_value_info(o, onnx.TensorProto.FLOAT, None)
                    for o in n.output
                ],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 17)],
        )
        model.ir_version = 8
        refs.append(ReferenceEvaluator(model))

    def run(feeds: dict) -> dict:
        env = {**consts, **feeds}
        for n, ref in zip(g.node, refs):
            res = _pulsar_op(n, env)
            if res is None:
                res = ref.run(None, {i: env[i] for i in n.input if i})
            for o, v in zip(n.output, res):
                env[o] = np.asarray(v)
        return env

    return run


def _u8_weights(m: onnx.ModelProto) -> onnx.ModelProto:
    """A copy of ``m`` with every constant MatMul operand (a 2-D weight) rounded to
    its per-column 8-bit grid: ``s = max|w| / 127.5`` per output column,
    ``q = rint(w / s)`` clipped to [-128, 127], ``w = q * s``. Pulsar2 calibrates a
    row-scaled norm's MatMul-fed tensors on these weights: with the float weights
    the ReduceMean outputs of the Qwen3 ``ident`` build sit 0.1-0.24% off the native
    scales, with them 9-19 ulp (and the other ReduceMeans 0-2 ulp)."""
    m = onnx.ModelProto.FromString(m.SerializeToString())
    consts = {i.name for i in m.graph.initializer}
    ops = {t for n in m.graph.node if n.op_type == "MatMul" for t in n.input}
    for i in m.graph.initializer:
        if i.name not in ops or i.name not in consts:
            continue
        w = onnx.numpy_helper.to_array(i)
        if w.dtype != F or w.ndim != 2:
            continue
        s = np.abs(w).max(axis=0, keepdims=True) / F(127.5)
        s = np.where(s > 0, s, F(1)).astype(F)
        q = (np.clip(np.rint(w / s), -128, 127) * s).astype(F)
        i.CopyFrom(onnx.numpy_helper.from_array(q, i.name))
    return m


def _moves_share(m: onnx.ModelProto, q: dict) -> dict:
    """``q`` with every data-movement output (``MOVES``: Reshape, Transpose, Split,
    Concat) given its first input's calibration. Pulsar2 keeps one quant across such
    a chain (the native quant json lists the outputs with the producer's scale and
    zero point, even where a MatMul reads them as a dynamic operand: ``zk0`` = ``z``,
    ``att`` = ``o_a``, ``k1`` = ``k``, ``qr3`` = ``q``), so the MinMax of the output
    is not its quant. Graph order, so a chain takes its source's final value."""
    q = dict(q)
    consts = {i.name for i in m.graph.initializer}
    # a dynamic MatMul operand (both inputs variable) keeps its own signed MinMax:
    # native gives ``qh_a`` (a Reshape of ``q_a``) zero point 0 and its own scale
    dyn = {
        t
        for n in m.graph.node
        if n.op_type == "MatMul" and not any(t in consts for t in n.input)
        for t in n.input
    }
    for n in m.graph.node:
        if n.op_type in MOVES and n.input and n.input[0] in q:
            for o in n.output:
                if o not in dyn:
                    q[o] = q[n.input[0]]
    return q


def calibrate_graph(
    directory: str,
    samples: Mapping | None = None,
    weight_u8: bool | None = None,
    pulsar_ref: bool | None = None,
) -> dict[str, tuple[float, int]]:
    """``calibrate`` for a fused program: U16 MinMax of every tensor (inputs,
    constants, inner tensors, output) over the float forward of the whole
    graph; both operands of every MatMul are signed (``signed_absmax``), as
    Pulsar2 quantizes them. Pulsar2's inner scales can be a float32 ulp or two
    off this (seen on fused RMSNorms and attention); the zero points agree.
    ``samples``: ``{input: arrays}`` in place of the tars.
    ``weight_u8``: the forward on the constant MatMul weights rounded to 8 bits
    (``_u8_weights``). ``None`` (the default) turns it on for graphs with
    ``NEW_OPS`` only; the older graphs' calibrations are unchanged. For ``NEW_OPS``
    graphs also: a data-movement output takes its source's calibration (``_moves_share``),
    a FullyConnected's activation is unsigned MinMax (only a dynamic MatMul operand is
    signed), and an unsigned zero point divides ``-lo`` by the scale in float32.
    ``pulsar_ref``: the float forward models Pulsar2's own float ops (``_pulsar_forward``:
    Sigmoid and Softmax by Sleef's exp, MatMul and ReduceMean by their accumulation
    order). ``None`` (the default) turns it on for graphs with ``NEW_OPS`` only."""
    m = onnx.load(os.path.join(directory, "model.onnx"))
    consts = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
    ins = [i.name for i in m.graph.input if i.name not in consts]
    data = _data(directory, ins, samples)
    new = any(n.op_type in NEW_OPS for n in m.graph.node)
    signed = {t for n in m.graph.node if n.op_type == "MatMul" for t in n.input}
    if new:  # a FullyConnected's activation (one constant operand) is unsigned MinMax
        signed = {
            t
            for n in m.graph.node
            if n.op_type == "MatMul" and not any(i in consts for i in n.input)
            for t in n.input
        }
    u8 = new if weight_u8 is None else weight_u8
    fwd = _u8_weights(m) if u8 else m
    # ``pulsar_ref`` (default: NEW_OPS graphs): the float forward with Pulsar2's
    # Sigmoid, Softmax, MatMul and ReduceMean (``_pulsar_forward``) in place of the
    # windowed ReduceMean of ``_windowed_forward``; the older graphs keep the reference.
    use_pulsar = new if pulsar_ref is None else pulsar_ref
    if use_pulsar:
        run = _pulsar_forward(fwd)
    else:
        ref = ReferenceEvaluator(_windowed_forward(fwd, strict=False) if new else fwd)
        run = lambda feeds: ref.run(None, feeds, intermediate=True)  # noqa: E731
    lo, hi = {}, {}
    for j in range(len(data[ins[0]])):
        res = run({n: data[n][j].astype(F) for n in ins})
        for k, v in res.items():
            a = np.asarray(v)
            if (
                k in consts
                or k.startswith("__cal_")
                or a.dtype.kind != "f"
                or a.size == 0
            ):
                continue
            lo[k] = min(lo.get(k, np.inf), float(a.min()))
            hi[k] = max(hi.get(k, -np.inf), float(a.max()))
    q = {
        k: (
            signed_absmax(max(abs(lo[k]), abs(hi[k])))
            if k in signed
            else unsigned_minmax(
                lo[k], hi[k], symmetric_32767=not new or _mul_input(m, k), f32=new
            )
        )
        for k in lo
    }
    for k, a in consts.items():
        q[k] = unsigned_minmax(a.min(), a.max(), f32=new)
    if new:
        q = _moves_share(m, q)
    return q


def compare(got, want) -> list[str]:
    """Differences that carry calibration between two compiled single-op
    programs: ``npu_params`` and, per segment, every record but the A8 ones in
    order (segment 0's slot table as a set) and the A8 records as a set. The
    IO names and their order (which flips between builds, with the A8 slot
    records that follow it) are not compared; when the two programs list their
    inputs in different orders, their A8 slot records follow different orders
    too (with more than two inputs even their multiset changes), so the A8
    records are then not compared either."""
    diffs = []

    def order(m):
        blob = gs.parse_blob(bytes(mre.mcode_initializer(m).raw_data))
        return [name for name, _ in blob["inputs"]]

    same_io = order(got) == order(want)
    pa = next(i.raw_data for i in got.graph.initializer if i.name == "npu_params")
    pb = next(i.raw_data for i in want.graph.initializer if i.name == "npu_params")
    if pa != pb:
        diffs.append("npu_params")
    for si, (ra, rb) in enumerate(zip(_segs(got), _segs(want))):
        if len(ra) != len(rb):
            diffs.append(f"segment {si}: {len(ra)} records vs {len(rb)}")
            continue
        lo, hi = gs.slot_table(rb) if si == 0 else (0, 0)
        if sorted(ra[lo:hi]) != sorted(rb[lo:hi]):
            diffs.append("segment 0 slot table")
        if same_io and sorted(r for r in ra if r[0] == gs.A8) != sorted(
            r for r in rb if r[0] == gs.A8
        ):
            diffs.append(f"segment {si}: A8 records")
        diffs += [
            f"segment {si} record {i}: {x.hex()} vs {y.hex()}"
            for i, (x, y) in enumerate(zip(ra, rb))
            if x != y and not lo <= i < hi and x[0] != gs.A8
        ]
    return diffs


def _io_order(model) -> list[str]:
    blob = gs.parse_blob(bytes(mre.mcode_initializer(model).raw_data))
    return [name for name, _ in blob["inputs"]]


def _canonical_jobs(model, keep_a8: bool) -> list:
    """Per segment: segment 0's slot table as a set, then every job (records
    up to and including its ``a9`` launch, zero padding dropped) as its
    records in order but the IO-slot ones (A8, and the A7 that follows one),
    and those as a sorted multiset (``keep_a8``) or their count."""
    out = []
    for si, seg in enumerate(_segs(model)):
        lo, hi = gs.slot_table(seg) if si == 0 else (0, 0)
        jobs, cur = [], []
        for r in seg[:lo] + seg[hi:]:
            if r == bytes(8):
                continue
            cur.append(r)
            if r[0] == gs.A9:
                jobs.append(cur)
                cur = []
        jobs.append(cur)
        out.append(
            (
                tuple(sorted(seg[lo:hi])),
                tuple(
                    (
                        tuple(r for r in j if r[0] not in (gs.A7, gs.A8)),
                        tuple(sorted(r for r in j if r[0] in (gs.A7, gs.A8)))
                        if keep_a8
                        else sum(r[0] in (gs.A7, gs.A8) for r in j),
                    )
                    for j in jobs
                ),
            )
        )
    return out


def compare_jobs(got, want) -> list[str]:
    """``compare`` up to Pulsar2's rebuild noise: two native builds of one
    calibration list their graph inputs in different orders (a per-build coin
    flip), and the A8 slot records (with the A7 records that follow them)
    follow that order, also in their position among a job's other records.
    Here every job's other records must match in order, its A7 / A8 records
    as a multiset (only their count when the IO orders differ), the slot
    table as a set, and ``npu_params`` byte for byte. Returns the segments
    (and ``npu_params``) that differ."""
    same_io = _io_order(got) == _io_order(want)
    diffs = []
    pa = next(i.raw_data for i in got.graph.initializer if i.name == "npu_params")
    pb = next(i.raw_data for i in want.graph.initializer if i.name == "npu_params")
    if pa != pb:
        diffs.append("npu_params")
    ca, cb = _canonical_jobs(got, same_io), _canonical_jobs(want, same_io)
    if len(ca) != len(cb):
        return diffs + [f"{len(ca)} segments vs {len(cb)}"]
    diffs += [f"segment {si}" for si, (a, b) in enumerate(zip(ca, cb)) if a != b]
    return diffs


def add_const_sharing(nodes, q) -> tuple[int, ...]:
    """Which Adds of a fused graph share one ``npu_params`` constant at
    calibration ``q``: for each Add (graph order) the index of the first Add
    with the same ``add_const`` word. Pulsar2 stores each distinct constant
    once and loads each with its own DMA job, so two calibrations with
    different sharing compile to programs with different job counts (an
    eps Add's ratio ``s_eps / s_e`` rounds to 0 or to a few units; a residual
    update's ratio can round to 0 and meet the eps Adds' ``(0x8000, 0)``).
    A retarget keeps the template's jobs."""
    vals = [
        candidates("Add", {t: q[t] for t in (*ins, out)}, ins, out)["add_const"]
        for op, ins, out, _ in nodes
        if op == "Add"
    ]
    first: dict[int, int] = {}
    return tuple(first.setdefault(v, i) for i, v in enumerate(vals))
