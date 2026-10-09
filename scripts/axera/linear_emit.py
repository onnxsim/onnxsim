"""Emit an AX650 linear layer (``y = x @ w``, constant ``w``) for new weights.

``emit_linear(template_model, w, scales, zero_points)`` returns the compiled
model of ``MatMul(x, w)`` for a weight matrix Pulsar2 never saw, from one
native build of the same shape (any weights, any calibration). Two shapes are
served, ``[1,64] @ [64,64]`` and ``[1,576] @ [576,576]``; the weight-table
layout differs between them and no other shape was built.
``emit_linear_from_samples`` also derives the calibration, with
``pulsar_free_calibration`` (``docs/axera-width-and-linear-emit.md``).

What a weight set changes
-------------------------

Between native builds of one shape only these differ (six weight sets per
shape, and the 576 layer at a second calibration):

* ``npu_params``: one tile per 64 output channels, each a weight block
  followed by a 512-byte requantization block, then 20 zero bytes per IO
  tensor;
* nine main-engine records: the DEQUANT job's eight scale lanes
  (``0x0f50..0x0fc0 = float32(s_y)``) and its ``0x1a90 = zp_y``;
* with another input calibration, the QUANT job's eight lanes
  (``float32(1 / s_x)``; its ``0x1b10 = zp_x`` follows the QUANT rule of every
  ``graph_stitch`` program, but no pair of linear builds varies ``zp_x``);
* segment 0's slot table order (rebuild noise, kept from the template).

Derived
-------

* Weight codes, per output channel ``c``: ``s_w = max|w[:, c]| / 127.5``,
  ``code = clip(rint(w / s_w), -128, 127) + 128`` in float32
  (``emitter.codes_of``).
* Requantization block of a tile (``emitter.requant_block``): 64 float32
  biases ``zp_y - zp_x * sum(q_c) * M_c`` then 64 float32 multipliers
  ``M_c = s_x * s_w[c] / s_y``.

Fitted
------

The byte address of every weight code, closed form, equal to the table bytes
of all six native builds of each shape (``weight_addresses``). With ``o`` the
output channel inside its tile and ``i`` the input channel:

* 64x64 (plain bytes, one tile of 4608 weight bytes)::

      byte = code[o, i]
      at 288 * ((o >> 1) & 15) + 36 * (o & 1) + 72 * ((o >> 5) & 1)
         + 144 * (i // 36) + i % 36

* 576x576 (two nibble planes, nine tiles of 36864 weight bytes), for the
  input pair ``j = i // 2`` and plane ``p`` (0: low nibbles, 1: high)::

      byte = nib_p(code[o, 2j]) | nib_p(code[o, 2j + 1]) << 4
      at 1152 * (o & 15) + 72 * ((o >> 4) & 1) + 18432 * (o >> 5)
         + 144 * (j // 36) + j % 36 + 36 * p

The weight-block bytes outside these addresses (512 per tile at 64x64, none
at 576x576) are the same in every build and are kept from the template.

Refused (``ValueError``)
------------------------

* a template that is not one of the two shapes, or whose ``npu_params`` size
  or main-engine jobs (PARAM, QUANT, PARAM, DEQUANT) are not the measured
  ones;
* a weight matrix of another shape, with a non-finite value or with an
  all-zero output channel (its scale would be 0; never built);
* a zero point of 0 for ``x`` or ``y``: the native program would then leave
  out a zero-point write (the register already holds 0), which no build shows;
* a template whose QUANT or DEQUANT job does not write its eight lanes or its
  zero point.

Limits: one native build of the shape is still needed as the template; the
template fixes everything a weight set does not change (job order, addresses,
the slot table order); only weights drawn from one distribution
(uniform +-0.1) and inputs calibrated to about +-1 and +-4 were built.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import emitter  # noqa: E402
import graph_stitch as gs  # noqa: E402
import misc_op_record_emit as mre  # noqa: E402
import pulsar_free_calibration as pfc  # noqa: E402
import short_unit_codec as suc  # noqa: E402
import width_retarget as wr  # noqa: E402

TILE = 64  # output channels per npu_params tile
BLOCK = 8 * TILE  # requantization block: 64 float32 biases, 64 float32 multipliers
# weight bytes per tile, by layer width
WEIGHT_BYTES = {64: 4608, 576: 36864}
SHAPES = tuple(WEIGHT_BYTES)
IO_BYTES = 20  # npu_params tail bytes per IO tensor
JOBS = ("PARAM", "QUANT", "PARAM", "DEQUANT")


def tile_bytes(n: int) -> int:
    return WEIGHT_BYTES[n] + BLOCK


def params_bytes(n: int) -> int:
    """``npu_params`` size of the ``n x n`` layer."""
    return n // TILE * tile_bytes(n) + 2 * IO_BYTES


def _check_shape(n: int) -> None:
    if n not in WEIGHT_BYTES:
        raise ValueError(
            f"linear layer {n}x{n}: only {' and '.join(f'{k}x{k}' for k in SHAPES)} "
            "were built, and their weight-table layouts differ"
        )


def weight_addresses(n: int) -> np.ndarray:
    """``npu_params`` byte address of every weight code.

    64: ``[out, in]``, one plain byte per code. 576: ``[out, in // 2, 2]``,
    for each input pair the low-nibble and the high-nibble byte."""
    _check_shape(n)
    o = np.arange(n) % TILE
    tile = (np.arange(n) // TILE) * tile_bytes(n)
    if n == 64:
        i = np.arange(n)
        row = 288 * ((o >> 1) & 15) + 36 * (o & 1) + 72 * ((o >> 5) & 1)
        return (tile + row)[:, None] + (144 * (i // 36) + i % 36)[None, :]
    j = np.arange(n // 2)
    row = 1152 * (o & 15) + 72 * ((o >> 4) & 1) + 18432 * (o >> 5)
    col = 144 * (j // 36) + j % 36
    return (tile + row)[:, None, None] + col[None, :, None] + 36 * np.arange(2)


def weight_bytes(codes: np.ndarray) -> np.ndarray:
    """The table bytes of ``codes [out, in]`` (u8), shaped like
    ``weight_addresses``."""
    n = codes.shape[0]
    _check_shape(n)
    if n == 64:
        return codes
    c = codes.reshape(n, n // 2, 2)
    planes = [
        ((c[..., 0] >> (4 * p)) & 15) | (((c[..., 1] >> (4 * p)) & 15) << 4)
        for p in (0, 1)
    ]
    return np.stack(planes, -1).astype(np.uint8)


def weight_codes(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(codes [out, in] u8, s_w [out] float32)`` of float weights
    ``w [in, out]``."""
    w = np.asarray(w, np.float32)
    if not np.isfinite(w).all():
        raise ValueError("the weight matrix holds a non-finite value")
    if not np.abs(w).max(axis=0).all():
        raise ValueError(
            "an output channel of the weight matrix is all zero; its scale "
            "would be 0 and no such layer was built"
        )
    return emitter.codes_of(w.T, axis=0), emitter.weight_scales(w.T, axis=0)


def linear_params(
    template_params: bytes,
    w: np.ndarray,
    s_x: float,
    zp_x: int,
    s_y: float,
    zp_y: int,
) -> bytes:
    """``npu_params`` of the layer for weights ``w [in, out]``: the weight
    codes at their addresses and each tile's requantization block; every other
    byte is the template's."""
    w = np.asarray(w, np.float32)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"expected square weights [n, n], got {w.shape}")
    n = w.shape[0]
    _check_shape(n)
    if len(template_params) != params_bytes(n):
        raise ValueError(
            f"the template's npu_params hold {len(template_params)} bytes, a "
            f"{n}x{n} linear layer {params_bytes(n)}"
        )
    codes, s_w = weight_codes(w)
    table = np.frombuffer(template_params, np.uint8).copy()
    table[weight_addresses(n).ravel()] = weight_bytes(codes).ravel()
    for t in range(n // TILE):
        ch = slice(TILE * t, TILE * (t + 1))
        at = t * tile_bytes(n) + WEIGHT_BYTES[n]
        table[at : at + BLOCK] = emitter.requant_block(
            codes[ch], float(s_x), int(zp_x), float(s_y), int(zp_y), s_w[ch]
        )
    return table.tobytes()


# ---- the calibration records -----------------------------------------------------
def _jobs(main: Sequence[bytes]) -> list[tuple[str, int, int, dict]]:
    """``(role, first record, last record + 1, state before the job)`` of
    each main-engine job, with ``graph_stitch.Program``'s role rules."""
    jobs, state, start = [], {}, 0
    before = {}
    for i, w in enumerate(main):
        if w[0] in (gs.A1, gs.A8):
            state[gs._reg(w)] = (w[0], gs._val(w))
        if w[0] != gs.A9:
            continue
        if state.get(gs.REG_JOBTYPE, (0, 0))[1] == gs.JOB_PARAM:
            role = "PARAM"
        elif state.get(gs.REG_SRC_A, (0, 0))[0] == gs.A8:
            role = "QUANT"
        elif state.get(gs.REG_DST, (0, 0))[0] == gs.A8:
            role = "DEQUANT"
        else:
            role = "CORE"
        jobs.append((role, start, i + 1, before))
        start, before = i + 1, dict(state)
    return jobs


def _set_job(main, start, end, before, lanes: int, zp_reg: int, zp: int, what: str):
    """Rewrite one job's eight lane records and its zero-point record."""
    span = range(start, end)
    hit = [i for i in span if main[i][0] == gs.A1 and gs._reg(main[i]) in gs.LANES_LO]
    if sorted(gs._reg(main[i]) for i in hit) != list(gs.LANES_LO):
        raise ValueError(f"the template's {what} job does not write its eight lanes")
    zps = [i for i in span if main[i][0] == gs.A1 and gs._reg(main[i]) == zp_reg]
    if len(zps) != 1:
        raise ValueError(
            f"the template's {what} job does not write its zero point "
            f"({zp_reg:#06x}) exactly once"
        )
    if before.get(zp_reg, (gs.A1, 0)) == (gs.A1, int(zp)):
        raise ValueError(
            f"{what} zero point {zp}: register {zp_reg:#06x} already holds it, "
            "so the native program would leave the write out; no build shows that"
        )
    for i in hit:
        main[i] = gs._rec(gs.A1, gs._reg(main[i]), lanes)
    main[zps[0]] = gs._rec(gs.A1, zp_reg, int(zp))


def template_width(model: onnx.ModelProto) -> int:
    """``n`` of a native ``[1, n] @ [n, n]`` linear build, after checking its
    IO shapes, ``npu_params`` size and main-engine jobs."""
    g = model.graph
    shapes = [
        tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
        for v in [*g.input, *g.output]
    ]
    if len(g.input) != 1 or len(g.output) != 1 or len(set(shapes)) != 1:
        raise ValueError(f"expected one input and one output of one shape: {shapes}")
    if len(shapes[0]) != 2 or shapes[0][0] != 1:
        raise ValueError(f"expected IO shape [1, n], found {list(shapes[0])}")
    n = shapes[0][1]
    _check_shape(n)
    params = next(i for i in g.initializer if i.name == "npu_params")
    if len(params.raw_data) != params_bytes(n):
        raise ValueError(
            f"the template's npu_params hold {len(params.raw_data)} bytes, a "
            f"{n}x{n} linear layer {params_bytes(n)}"
        )
    return n


def emit_linear(
    template_model,
    w: np.ndarray,
    scales: Mapping[str, float],
    zero_points: Mapping[str, int],
) -> onnx.ModelProto:
    """The compiled model of ``y = x @ w`` for float weights ``w [in, out]``.

    ``template_model`` is a native build of the same shape (path, bytes or
    ``ModelProto``); ``scales`` and ``zero_points`` are keyed by its input and
    output tensor names (``x`` and ``y`` in the committed builds). The
    per-channel weight scales follow from ``w``."""
    model = gs.load_model(template_model)
    n = template_width(model)
    w = np.asarray(w, np.float32)
    if w.shape != (n, n):
        raise ValueError(f"the template is {n}x{n}; the weights are {w.shape}")
    x, y = model.graph.node[0].input[0], model.graph.node[0].output[0]
    missing = sorted({x, y} - (set(scales) & set(zero_points)))
    if missing:
        raise ValueError(f"no scale or zero point for tensors {missing}")
    s_x, s_y = float(np.float32(scales[x])), float(np.float32(scales[y]))
    zp_x, zp_y = int(zero_points[x]), int(zero_points[y])
    if not (0 < zp_x <= 255 and 0 < zp_y <= 255):
        raise ValueError(
            f"zero points x={zp_x}, y={zp_y}: both must be in 1..255 (with 0 "
            "the native program leaves a zero-point write out; never built)"
        )
    if not (s_x > 0 and s_y > 0):
        raise ValueError(f"scales must be positive: x={s_x}, y={s_y}")

    mc = bytes(mre.mcode_initializer(model).raw_data)
    segs = [gs.records(raw) for raw in suc.decode_segments(mc)]
    main = segs[gs.MAIN]
    jobs = _jobs(main)
    if tuple(j[0] for j in jobs) != JOBS:
        raise ValueError(
            f"the template's main-engine jobs are {[j[0] for j in jobs]}, a "
            f"linear layer's {list(JOBS)}"
        )
    _, q0, q1, q_before = jobs[1]
    _, d0, d1, d_before = jobs[3]
    _set_job(main, q0, q1, q_before, gs._f32bits(1 / s_x), gs.REG_ZP_OUT, zp_x, "QUANT")
    _set_job(main, d0, d1, d_before, gs._f32bits(s_y), gs.REG_ZP_A, zp_y, "DEQUANT")
    out = wr.with_segments(model, segs)
    params = next(i for i in out.graph.initializer if i.name == "npu_params")
    params.raw_data = linear_params(bytes(params.raw_data), w, s_x, zp_x, s_y, zp_y)
    return out


def linear_float_model(w: np.ndarray, x: str = "x", y: str = "y") -> onnx.ModelProto:
    """The float graph ``y = MatMul(x, w)`` for ``w [in, out]``, as Pulsar2
    was given it."""
    w = np.asarray(w, np.float32)
    n_in, n_out = w.shape
    model = onnx.parser.parse_model(
        '<ir_version: 9, opset_import: ["" : 17]> '
        f"g (float[1,{n_in}] {x}) => (float[1,{n_out}] {y}) {{ {y} = MatMul({x}, w) }}"
    )
    # the weights are a large random array: attached as a numpy initializer
    model.graph.initializer.append(onnx.numpy_helper.from_array(w, "w"))
    return model


def emit_linear_from_samples(
    template_model,
    w: np.ndarray,
    samples: Sequence[np.ndarray],
) -> tuple[onnx.ModelProto, dict[str, float], dict[str, int]]:
    """``emit_linear`` with the calibration derived from input ``samples`` by
    ``pulsar_free_calibration`` (MinMax). Returns ``(model, scales,
    zero_points)``. The output scale can be up to two float32 ulps from the
    one Pulsar2 computes (``pulsar_free_calibration``'s docstring)."""
    model = gs.load_model(template_model)
    x, y = model.graph.node[0].input[0], model.graph.node[0].output[0]
    quants = pfc.calibrate(linear_float_model(w, x, y), {x: list(samples)})
    scales, zero_points, _ = pfc.stitch_calibration(quants, (x, y))
    return emit_linear(model, w, scales, zero_points), scales, zero_points
