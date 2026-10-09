"""Move a compiled ``[1, 64]`` AX650 op program to another width, and emit
ReduceMean at any measured width from two templates.

``retarget_width(template_model, n, op)`` rewrites the width-dependent records
of a standalone compiled ``[1, 64]`` program (Sigmoid, Mul, Add, Div, Sqrt,
Softmax, Neg) so that it is the program Pulsar2 compiles for ``[1, n]``. The
calibration stays the template's; ``calibrate_program`` then moves a program to
another calibration (and chooses the two things a native build picks freely:
the input slot order and where the output's PARAM job runs).
``emit_reducemean`` does both steps for ReduceMean, whose core job has three
forms (``docs/axera-width-and-linear-emit.md``).

This lives next to ``graph_stitch`` rather than in ``misc_op_record_emit``:
it needs ``graph_stitch``'s blob rebuild and one-op stitch (and
``graph_stitch`` imports ``misc_op_record_emit``), and that module's contract
is the opposite one, "shape does not enter any of these edits".

Derived (read off every program)
--------------------------------

Checked by rebuilding native builds record for record:

* the scratch layout: buffers are bump-allocated from ``0x2f7000`` in address
  order, each ``k * n`` bytes rounded up to 32 (``k`` 1, 2 or 4), the
  32-byte PARAM buffers unchanged;
* ``a7 0x0100 = 0x17c00 | (PARAM buffer - 0x2f7000) / 0x20`` (``graph_stitch``
  checks it on every program);
* the IO byte sizes of the blob, the tensor shapes and ``outputs_info`` of the
  model.

Fitted (which builds pin what)
------------------------------

Fitted on native builds at n = 64, 100, 384, 512, 576, 2048 (Sigmoid, Mul,
Softmax; Add, Div, Sqrt at 64, 100, 576; Neg at 64 and 576, two calibrations,
one per Neg program):

* byte counts on ``0x02a0 0x03c0 0x04e0 0x0710 0x1b60``: ``k * n - 1``;
* ``0x09b0 0x0ad0 0x0b60``: ``ceil16(n) - 1``; ``0x0bf0``: ``ceil8(n) - 1``;
  ``0x0e80``: ``ceil(n / 8) - 1``;
* block counts on ``0x1fe0 0x1ff0 0x1bf0 0x1c00 0x1ca0 0x1cb0 0x1d50 ..
  0x1d80``: ``ceil(n / 16) - 1``, or ``ceil(n / 32) - 1`` when the word carries
  flag ``0x01000000`` or ``0x04000000``;
* a buffer larger than 1024 bytes leaves the scratch window and is
  bump-allocated from address 0 in the same order. Seen in Softmax at 576
  (one buffer of 1152 bytes) and in Sigmoid, Mul and Softmax at 2048 (two or
  three buffers of 2048 and 4096 bytes), so a relocated buffer followed by
  another one is accepted only when its size is a multiple of 128;
* ``0x1a70`` / ``0x1a80`` (a stride ``W - 1``, ``W`` 8 or 32) become ``n - 1``
  when ``n`` is not a multiple of ``W``. Only n = 100 shows it.

ReduceMean (``emit_reducemean``), fitted on n = 64, 100, 128, 256, 288, 384,
512, 576, 2048 (11 builds with the second calibration at 64 and 576):

* n <= 256: the small template's core with ``0x0310 = n - 1`` and
  ``0x1ed0 = 0x40000 | (n - 1)``;
* n > 256: the large template's core (256-wide pooling windows, the last one
  padded) with ``0x0310 = n - 1``, ``0x0cd0 = ceil256(n) - n``,
  ``0x0cf0 = n``, ``0x0e80 = ceil(n / 256) - 1``;
* n > 256 and a multiple of 256: the large core without its pad group
  (``0x0c20``, ``0x0cd0..0x0da0``) and with ``0x0280`` and ``0x0c10`` of the
  small template's core.

Any pair of templates gives the same programs: (64, 576), (128, 288),
(256, 384) and (256, 576) were compared at all nine widths.

Not predicted
-------------

Where the graph output's PARAM job runs. Native ReduceMean at 512 and 576 and
Sigmoid at 512 run it before the core job, every other build after. No rule
was found, so ``emit_reducemean`` takes it as a required argument and
``calibrate_program`` as an optional one. Both orders are valid programs.

Refused (``ValueError``)
------------------------

* a template that is not ``[1, 64]`` (``retarget_width``);
* a width below 64, above the op's largest measured width (2048 for Sigmoid,
  Mul, Softmax and ReduceMean; 576 for Add, Div, Sqrt and Neg), or not a
  multiple of 32. The one exception is 100, the only such width measured, for
  Sigmoid, Mul, Add, Div, Sqrt and ReduceMean;
* Softmax at a width that is not a multiple of 32: the native ``[1, 100]``
  build is a different, padded program (20 more records);
* Neg at 100 (never built);
* a relocated buffer whose size is not a multiple of 128 followed by another
  relocated buffer (the alignment was never observed);
* a template that already holds a buffer outside the scratch window;
* ReduceMean templates outside the measured kinds: the small one ``[1, n]``
  with n a multiple of 32 up to 256, the large one a multiple of 32 above 256
  that is not a multiple of 256, at most 576.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import graph_stitch as gs  # noqa: E402
import misc_op_record_emit as mre  # noqa: E402
import short_unit_codec as suc  # noqa: E402

TEMPLATE_WIDTH = 64
OPS = ("Sigmoid", "Mul", "Add", "Div", "Sqrt", "Softmax", "Neg")
# largest width a native build was compared at, per op
MAX_WIDTH = {
    "Sigmoid": 2048,
    "Mul": 2048,
    "Softmax": 2048,
    "Add": 576,
    "Div": 576,
    "Sqrt": 576,
    "Neg": 576,
    "ReduceMean": 2048,
}
# the only measured width that is not a multiple of 32, and the ops built there
ODD_WIDTH = 100
ODD_WIDTH_OPS = ("Sigmoid", "Mul", "Add", "Div", "Sqrt", "ReduceMean")

BASE, UNIT = gs.SCRATCH_BASE, gs.SCRATCH_UNIT
SIZE_REGS = (0x02A0, 0x03C0, 0x04E0, 0x0710, 0x1B60)
SIZE16_REGS = (0x09B0, 0x0B60, 0x0AD0)
SIZE8_REGS = (0x0BF0,)
EIGHTH_REG = 0x0E80
COUNT_REGS = (
    0x1FE0,
    0x1FF0,
    0x1BF0,
    0x1C00,
    0x1CA0,
    0x1CB0,
    0x1D50,
    0x1D60,
    0x1D70,
    0x1D80,
)
COUNT32_FLAGS, COUNT16_FLAGS = (0x1000000, 0x4000000), (0, 0x8000000)
STRIDE_REGS = (0x1A70, 0x1A80)
STRIDES = (8, 32)
ADDRESS_REGS = (gs.REG_SRC_A, gs.REG_DST)  # a1: a main-engine buffer address
BIG_LIMIT = 1024  # a larger buffer leaves the scratch window
BIG_ALIGN = 128  # every relocated size seen is a multiple of this

# ReduceMean
REDUCE_WINDOW = gs.REDUCE_WINDOW
REG_REDUCE_LAST = 0x0310  # n - 1
REG_REDUCE_SMALL = 0x1ED0  # 0x40000 | (n - 1), small core
REG_REDUCE_PAD, REG_REDUCE_LEN = 0x0CD0, 0x0CF0  # pad bytes, n (large core)
REDUCE_FROM_SMALL = (0x0280, 0x0C10)  # large core, n a multiple of 256
REDUCE_PAD_GROUP = (0x0C20, *range(0x0CD0, 0x0DA1, 0x10))


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _cup(a: int, b: int) -> int:
    return _cdiv(a, b) * b


def _segments(model: onnx.ModelProto) -> list[list[bytes]]:
    mc = bytes(mre.mcode_initializer(model).raw_data)
    return [gs.records(raw) for raw in suc.decode_segments(mc)]


def model_width(model: onnx.ModelProto) -> int:
    """``n`` of a compiled program whose graph inputs are all ``[1, n]``."""
    shapes = {
        tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
        for v in model.graph.input
    }
    if len(shapes) != 1 or len(next(iter(shapes))) != 2 or next(iter(shapes))[0] != 1:
        raise ValueError(f"expected graph inputs of one shape [1, n], found {shapes}")
    return next(iter(shapes))[1]


def check_width(op: str, n: int) -> None:
    """Raise ``ValueError`` unless ``op`` at ``[1, n]`` is inside the measured
    conditions (module docstring, "Refused")."""
    if op not in MAX_WIDTH:
        raise ValueError(f"no width rules for {op!r}; have {sorted(MAX_WIDTH)}")
    if not TEMPLATE_WIDTH <= n <= MAX_WIDTH[op]:
        raise ValueError(
            f"{op} at width {n}: only {TEMPLATE_WIDTH}..{MAX_WIDTH[op]} was measured"
        )
    if n % 32 == 0 or (n == ODD_WIDTH and op in ODD_WIDTH_OPS):
        return
    if op == "Softmax":
        raise ValueError(
            f"Softmax at width {n}: a width that is not a multiple of 32 compiles "
            "to a different, padded program"
        )
    raise ValueError(
        f"{op} at width {n}: not a multiple of 32, and only {ODD_WIDTH} was "
        f"measured among those (for {', '.join(ODD_WIDTH_OPS)})"
    )


# ---- the record rewrite ----------------------------------------------------------
def _layout(segs: Sequence[Sequence[bytes]], n_old: int, n_new: int) -> dict[int, int]:
    """Scratch address of the template -> address at width ``n_new``."""
    addrs = set()
    for s in segs:
        for w in s:
            if w[0] == gs.A1 and gs._is_scratch(gs._val(w)):
                addrs.add(gs._val(w))
            elif w[0] == gs.A7 and gs._reg(w) == gs.REG_A7_COUNT:
                addrs.add(BASE + UNIT * (gs._val(w) & 0xFF))
    for w in segs[gs.MAIN]:
        if w[0] == gs.A1 and gs._reg(w) in ADDRESS_REGS and gs._val(w) < BASE:
            raise ValueError(
                "the template already holds a buffer outside the scratch "
                f"window ({gs._reg(w):#06x} = {gs._val(w):#x})"
            )
    addrs = sorted(addrs)
    amap, cur, big, big_sizes = {}, BASE, 0, []
    for a, nxt in zip(addrs, addrs[1:] + [None]):
        if nxt is None:  # the last buffer's size does not matter
            amap[a] = cur
            break
        size = nxt - a
        if size != UNIT:
            if size % n_old:
                raise ValueError(
                    f"scratch buffer {a:#x} of {size} bytes is not a multiple "
                    f"of the template width {n_old}"
                )
            size = _cup(size // n_old * n_new, UNIT)
        if size > BIG_LIMIT:
            amap[a] = big
            big += size
            big_sizes.append(size)
        else:
            amap[a] = cur
            cur += size
    if any(s % BIG_ALIGN for s in big_sizes[:-1]):
        raise ValueError(
            f"width {n_new}: relocated buffers of {big_sizes} bytes; the "
            f"alignment after a size that is not a multiple of {BIG_ALIGN} "
            "was never observed"
        )
    return amap


def _rewrite(segs: Sequence[Sequence[bytes]], n_old: int, n_new: int):
    """The records of a ``[1, n_old]`` program at width ``n_new``."""
    if n_old % 32:
        raise ValueError(f"template width {n_old} is not a multiple of 32")
    amap = _layout(segs, n_old, n_new)
    out = []
    for s in segs:
        new = []
        for w in s:
            verb, reg, v = w[0], gs._reg(w), gs._val(w)
            nv = v
            if verb == gs.A1 and gs._is_scratch(v):
                nv = amap[v]
            elif verb == gs.A7 and reg == gs.REG_A7_COUNT and v >> 8 == gs.A7_HIGH >> 8:
                nv = gs.A7_HIGH | ((amap[BASE + UNIT * (v & 0xFF)] - BASE) // UNIT)
            elif verb != gs.A1:
                pass
            elif reg in SIZE_REGS and v + 1 in (n_old, 2 * n_old, 4 * n_old):
                nv = (v + 1) // n_old * n_new - 1
            elif reg in SIZE16_REGS and v + 1 == n_old:
                nv = _cup(n_new, 16) - 1
            elif reg in SIZE8_REGS and v + 1 == n_old:
                nv = _cup(n_new, 8) - 1
            elif reg == EIGHTH_REG and v == n_old // 8 - 1:
                nv = _cdiv(n_new, 8) - 1
            elif reg in COUNT_REGS:
                hi, lo = v & ~0xFFFF, v & 0xFFFF
                if hi in COUNT32_FLAGS and lo == n_old // 32 - 1:
                    nv = hi | (_cdiv(n_new, 32) - 1)
                elif hi in COUNT16_FLAGS and lo == n_old // 16 - 1:
                    nv = hi | (_cdiv(n_new, 16) - 1)
            elif reg in STRIDE_REGS and v + 1 in STRIDES and n_new % (v + 1):
                nv = n_new - 1
            new.append(gs._rec(verb, reg, nv) if nv != v else w)
        out.append(new)
    return out


def with_segments(
    model: onnx.ModelProto,
    segs: Sequence[Sequence[bytes]],
    n_old: int | None = None,
    n_new: int | None = None,
) -> onnx.ModelProto:
    """``model`` with its MCode blob rebuilt around ``segs`` (decompressed
    records per segment; the segment count and which segments are compressed
    stay the template's); with ``n_old`` and ``n_new`` the float IO byte
    sizes, tensor shapes and ``outputs_info`` move from width ``n_old`` to
    ``n_new``."""
    blob = gs.parse_blob(bytes(mre.mcode_initializer(model).raw_data))

    def io(descs):
        if n_old is None:
            return descs
        return [(k, 4 * n_new if s == 4 * n_old else s) for k, s in descs]

    raws = [b"".join(s) for s in segs]
    # an unchanged segment keeps its stream: suc.encode does not always pick
    # the matches Pulsar2's compressor picked (same records, other bytes)
    old = suc.decode_segments(bytes(mre.mcode_initializer(model).raw_data))
    streams = [
        stream if raw == was else (suc.encode(raw) if comp else raw)
        for raw, was, stream, comp in zip(raws, old, blob["streams"], blob["comp"])
    ]
    mc = gs.build_blob(
        dict(
            blob,
            inputs=io(blob["inputs"]),
            outputs=io(blob["outputs"]),
            streams=streams,
        )
    )
    if suc.decode_segments(mc) != raws:
        raise ValueError("the rebuilt MCode blob does not decode to its segments")
    out = onnx.ModelProto()
    out.CopyFrom(model)
    g = out.graph
    neu = mre.mcode_initializer(out)
    neu.raw_data = mc
    del neu.dims[:]
    neu.dims.append(len(mc))
    for v in [*g.input, *g.output, *g.value_info]:
        dims = v.type.tensor_type.shape.dim
        if v.name == neu.name:
            dims[0].dim_value = len(mc)
        elif (
            n_old is not None and v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        ):
            for d in dims:
                if d.dim_value == n_old:
                    d.dim_value = n_new
    if n_old is not None:
        for a in g.node[0].attribute:
            if a.name == "outputs_info":
                info = json.loads(a.s)
                for k in info:
                    info[k][1] = [n_new if x == n_old else x for x in info[k][1]]
                a.s = json.dumps(info).encode()
    return out


def _retarget(model: onnx.ModelProto, n_old: int, n_new: int) -> onnx.ModelProto:
    if n_new == n_old:
        out = onnx.ModelProto()
        out.CopyFrom(model)
        return out
    return with_segments(model, _rewrite(_segments(model), n_old, n_new), n_old, n_new)


def retarget_width(template_model, n: int, op: str) -> onnx.ModelProto:
    """The standalone ``[1, 64]`` program ``template_model`` of ``op`` at width
    ``n``, in the template's own calibration, slot order and job order.

    ``op`` is one of ``OPS``; it selects the measured conditions
    (``check_width``), the compiled model does not name its op."""
    if op not in OPS:
        raise ValueError(
            f"retarget_width serves {OPS}; ReduceMean has emit_reducemean, got {op!r}"
        )
    model = gs.load_model(template_model)
    if model_width(model) != TEMPLATE_WIDTH:
        raise ValueError(
            f"the template is [1, {model_width(model)}]; the formulas were "
            f"fitted from [1, {TEMPLATE_WIDTH}] templates only"
        )
    check_width(op, int(n))
    return _retarget(model, TEMPLATE_WIDTH, int(n))


# ---- calibration of a one-op program ---------------------------------------------
def calibrate_program(
    model,
    op: str,
    scales: Mapping[str, float],
    zero_points: Mapping[str, int],
    *,
    signed: Iterable[str] = (),
    count: int | None = None,
    slot_order: Sequence[str] | None = None,
    output_param: str | None = None,
    old_scales: Mapping[str, float] | None = None,
    old_zero_points: Mapping[str, int] | None = None,
) -> onnx.ModelProto:
    """The standalone program ``model`` of ``op`` at another calibration.

    Every op but Neg is a one-op ``graph_stitch``: ``slot_order`` is the
    compiled input slot order (default: the program's own) and
    ``output_param`` places the output's PARAM job ``"early"`` (before the
    core job) or ``"late"`` (before the DEQUANT job); ``None`` keeps the
    program's. ``count`` is ReduceMean's reduced element count. Neg is not a
    ``graph_stitch`` op: it goes through ``misc_op_record_emit.retarget`` and
    needs the program's own calibration (``old_scales``, ``old_zero_points``);
    its two knobs must stay at their defaults."""
    model = gs.load_model(model)
    if op == "Neg":
        if slot_order is not None or output_param is not None:
            raise ValueError("Neg keeps its template's slot and job order")
        if old_scales is None or old_zero_points is None:
            raise ValueError("Neg needs the program's own calibration (old_*)")
        return _calibrate_neg(model, old_scales, old_zero_points, scales, zero_points)
    if output_param not in (None, "early", "late"):
        raise ValueError(f"output_param={output_param!r}: 'early', 'late' or None")
    prog = gs.Program(model, op)
    out = prog.out_names[0]
    step = dict(
        op=op,
        program=prog,
        out=(out, out),
        onnx_inputs=[v.name for v in model.graph.input],
    )
    step["in"] = {k: k for k in prog.in_names}
    if count is not None:
        step["attrs"] = {"count": int(count)}
    wiring = {
        "inputs": list(slot_order) if slot_order is not None else prog.in_names,
        "graph_inputs": [v.name for v in model.graph.input],
        "output": out,
        "ops": [step],
    }
    if output_param is not None:
        wiring["output_param"] = 0 if output_param == "early" else "late"
    return gs.stitch_model(wiring, scales, zero_points, signed)


def _calibrate_neg(model, old_scales, old_zps, scales, zps) -> onnx.ModelProto:
    """Records through ``mre.retarget``; the large program's ``npu_params``
    words through ``mre.neg_params``. A calibration that compiles to the other
    Neg program is refused there."""
    mc = bytes(mre.mcode_initializer(model).raw_data)
    mc = mre.retarget(mc, "Neg", old_scales, scales, old_zps, zps)
    out = with_segments(model, [gs.records(r) for r in suc.decode_segments(mc)])
    if mre.neg_program(scales["x"]) == "large":
        params = next(i for i in out.graph.initializer if i.name == "npu_params")
        old = bytes(params.raw_data)
        if old != mre.neg_params(old_scales["x"], len(old)):
            raise ValueError("the Neg template's npu_params are not neg_params")
        params.raw_data = mre.neg_params(scales["x"], len(old))
    return out


# ---- ReduceMean --------------------------------------------------------------------
def _core_job(prog: gs.Program) -> int:
    cores = [j.index for j in prog.jobs if j.role == "CORE"]
    if len(cores) != 1:
        raise ValueError(f"{prog.label}: expected one CORE job, found {len(cores)}")
    return cores[0]


def _core_registers(model: onnx.ModelProto) -> dict[int, int]:
    prog = gs.Program(model, "ReduceMean")
    recs = prog.jobs[_core_job(prog)].records
    return {gs._reg(w): gs._val(w) for w in recs if w[0] == gs.A1}


def reducemean_template(n: int, small=None, large=None) -> onnx.ModelProto:
    """A ReduceMean ``[1, n] -> [1, 1]`` program in its source template's
    calibration and job order (module docstring for the three core forms).

    ``small`` is a native build at a width up to 256, ``large`` one above 256
    that is not a multiple of 256. ``n <= 256`` needs ``small``, ``n > 256``
    needs ``large``, and a multiple of 256 above 256 needs both."""
    n = int(n)
    check_width("ReduceMean", n)
    need_small = n <= REDUCE_WINDOW or n % REDUCE_WINDOW == 0
    need_large = n > REDUCE_WINDOW
    if (need_small and small is None) or (need_large and large is None):
        raise ValueError(
            f"ReduceMean at width {n} needs the "
            + " and the ".join(
                k for k, need in (("small", need_small), ("large", need_large)) if need
            )
            + " template"
        )
    if need_small:
        small = gs.load_model(small)
        w = model_width(small)
        if w % 32 or not TEMPLATE_WIDTH <= w <= REDUCE_WINDOW:
            raise ValueError(
                f"small ReduceMean template is [1, {w}]: measured kinds are "
                f"multiples of 32 in {TEMPLATE_WIDTH}..{REDUCE_WINDOW}"
            )
    if need_large:
        large = gs.load_model(large)
        w = model_width(large)
        if w % 32 or w % REDUCE_WINDOW == 0 or not REDUCE_WINDOW < w <= 576:
            raise ValueError(
                f"large ReduceMean template is [1, {w}]: measured kinds are "
                f"multiples of 32 in {REDUCE_WINDOW + 1}..576 that are not "
                f"multiples of {REDUCE_WINDOW}"
            )
    source = large if need_large else small
    model = _retarget(source, model_width(source), n)
    prog = gs.Program(model, "ReduceMean")
    core = _core_job(prog)
    jobs = [list(j.records) for j in prog.jobs]
    tail = prog.segs[gs.MAIN][sum(len(j) for j in jobs) :]
    drop: tuple[int, ...] = ()
    if not need_large:
        values = {REG_REDUCE_LAST: n - 1, REG_REDUCE_SMALL: 0x40000 | (n - 1)}
    else:
        values = {
            REG_REDUCE_LAST: n - 1,
            REG_REDUCE_PAD: _cup(n, REDUCE_WINDOW) - n,
            REG_REDUCE_LEN: n,
            EIGHTH_REG: _cdiv(n, REDUCE_WINDOW) - 1,
        }
        if need_small:  # no padded window: the small core's mode words
            small_core = _core_registers(small)
            values.update({r: small_core[r] for r in REDUCE_FROM_SMALL})
            drop = REDUCE_PAD_GROUP
    missing = sorted(
        r
        for r in values
        if r not in drop
        and not any(w[0] == gs.A1 and gs._reg(w) == r for w in jobs[core])
    )
    if missing:
        raise ValueError(
            "the ReduceMean template's core job does not write registers "
            + ", ".join(f"{r:#06x}" for r in missing)
        )
    main = []
    for i, job in enumerate(jobs):
        for w in job:
            reg = gs._reg(w)
            if w[0] == gs.A1 and i >= core and reg in drop:
                continue
            if w[0] == gs.A1 and i == core and reg in values:
                w = gs._rec(gs.A1, reg, values[reg])
            main.append(w)
    main += [w for w in tail if any(w)]
    main += [bytes(gs.REC)] * (-len(main) % gs.PAD_RECORDS)
    segs = [list(s) for s in prog.segs]
    segs[gs.MAIN] = main
    return with_segments(model, segs)


def emit_reducemean(
    n: int,
    scales: Mapping[str, float],
    zero_points: Mapping[str, int],
    *,
    output_param: str,
    small=None,
    large=None,
) -> onnx.ModelProto:
    """ReduceMean over the last axis of ``[1, n]`` (``keepdims=1``) at the
    given calibration (tensor names of the templates, ``x`` and ``y`` in the
    committed ones).

    ``output_param`` is required: ``"early"`` or ``"late"`` (module docstring,
    "Not predicted")."""
    if output_param not in ("early", "late"):
        raise ValueError(
            "emit_reducemean needs output_param='early' or 'late': where the "
            "output's PARAM job runs is not predicted"
        )
    template = reducemean_template(n, small, large)
    return calibrate_program(
        template,
        "ReduceMean",
        scales,
        zero_points,
        count=int(n),
        output_param=output_param,
    )
