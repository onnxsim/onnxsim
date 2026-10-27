"""Emit an AX650 linear layer (``y = x @ w``, constant ``w``) for new weights.

``emit_linear(template_model, w, scales, zero_points)`` returns the compiled
model of ``MatMul(x, w)`` for a weight matrix Pulsar2 never saw, from one
native build of the same shape (any weights, any calibration). The weight-table
layout depends on the shape; the families are listed under "Families" below,
and ``linear_params`` does the ``npu_params`` of any of them (a template build
of the shape is still needed for the main-engine records and the IO tail).
``emit_linear_from_samples`` also derives the calibration, with
``pulsar_free_calibration`` (``docs/axera-width-and-linear-emit.md``).

What a weight set changes
-------------------------

Families (``family(k, n)``)
---------------------------

* ``square64`` / ``square576`` (``[1, n] @ [n, n]``, the two layouts above);
* ``tall576`` (``k = 576``, ``n`` a multiple of 64): the square-576 tile
  layout, one tile per 64 outputs (36864 weight bytes + 512 requantization
  bytes each), 40 IO bytes;
* ``plain128`` (``128 x 128``): plain code bytes, two 64-output tiles of 9216
  bytes, then ONE requantization block for all 128 channels;
* ``wide`` (``k`` in 768, 1024, 1536, 2048, 3072; ``n`` a multiple of 32): input
  pairs in segments of at most 560 pairs (see ``wide_layout``); per 32-output
  group, each segment's 16 weight rows are followed by a 256-byte
  requantization block. The 1024 x 151936 build also has a 100-byte IO tail.

The non-square families are built from scratch (weights, pads, requantization
blocks) with the template's IO tail; ``emit_linear`` of the 151936-wide layer
is refused (its output is four DEQUANT jobs, not two).

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
(uniform +-0.1) and inputs calibrated to about +-1 and +-4 were built. The
non-square families were fitted and checked on the held-out builds listed in
``tests/test_axera_linear_emit_families.py``; the 1024 x 151936 layer has
``npu_params`` only (its output is four DEQUANT jobs, see ``emit_linear``).
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


#: input widths of the ``wide`` family that were built (any multiple-of-32 ``n``)
WIDE_K = (768, 1024, 1536, 2048, 3072)
WIDE_SEGMENT = 560  # input pairs per segment, except the first
SHAPE_TAIL = {(1024, 151936): 100}  # other shapes: 40 zero bytes


def family(k: int, n: int) -> str:
    """The layout family of the ``[1, k] @ [k, n]`` linear layer."""
    if (k, n) == (64, 64):
        return "square64"
    if (k, n) == (128, 128):
        return "plain128"
    if k == 576 and n % TILE == 0 and n > 0:
        return "square576" if n == 576 else "tall576"
    if k in WIDE_K and n % 32 == 0:
        return "wide"
    raise ValueError(
        f"linear layer {k}x{n}: no layout built (64x64, 128x128, 576 x 64m, and "
        f"k in {list(WIDE_K)} with n a multiple of 32)"
    )


def wide_segments(k: int) -> list[int]:
    """Input pairs per segment of a ``wide`` layer: the first holds
    ``(k/2) mod 560`` pairs (560 if that is 0), the rest 560 each."""
    pairs = k // 2
    if pairs <= WIDE_SEGMENT:
        return [pairs]
    first = pairs % WIDE_SEGMENT or WIDE_SEGMENT
    return [first] + [WIDE_SEGMENT] * ((pairs - first) // WIDE_SEGMENT)


def _row_bytes(pairs: int) -> int:
    """``S1``: one weight row of a segment, ``144`` bytes per 36 pairs."""
    return 144 * -(-pairs // 36)


def wide_layout(k: int) -> tuple[list[int], list[int], int]:
    """``(segment pairs, segment offsets in a group, group bytes G)``, with
    ``G = sum over segments of (16 * S1 + 256)``."""
    segs = wide_segments(k)
    offs, off = [], 0
    for pairs in segs:
        offs.append(off)
        off += (
            16 * _row_bytes(pairs) + 256
        )  # weight rows, then the requantization block
    return segs, offs, off


def _table_bytes(k: int, n: int) -> int:
    """Bytes of the weight-and-requantization table (the IO tail excluded)."""
    fam = family(k, n)
    if fam in ("square64", "square576"):
        return params_bytes(n) - IO_BYTES * 2
    if fam == "plain128":
        return 18432 + 1024
    if fam == "tall576":
        return n // TILE * tile_bytes(576)
    return n // 32 * wide_layout(k)[2]


def _tail_bytes(k: int, n: int) -> int:
    return SHAPE_TAIL.get((k, n), 2 * IO_BYTES)


def layer_params_bytes(k: int, n: int) -> int:
    """``npu_params`` size of the ``[1, k] @ [k, n]`` linear layer."""
    return _table_bytes(k, n) + _tail_bytes(k, n)


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


def gptq_weight(
    w: np.ndarray,
    x: np.ndarray,
    percdamp: float = 0.01,
    device: str | None = None,
) -> np.ndarray:
    """GPTQ rounding of float weights ``w [in, out]`` on the card's own s8
    grid, with layer inputs ``x [samples, in]``: weights ``w'`` whose
    ``weight_codes`` are GPTQ's codes at ``w``'s own scales
    (``max|w_c| / 127.5``), for ``emit_linear_u16``. Each output channel's
    largest element keeps its float value and its code (it sets the scale
    ``weight_codes`` recomputes); every other element is ``code * s`` with
    ``|code| <= 127``, since a -128 elsewhere would become the channel's
    largest and move the scale. GPTQ runs in onnxsim's C++ column loop, or
    in torch on ``device`` (``"cuda"``) (``onnxsim.gptq``). Qwen3-0.6B on the card: median logit error 4.4% ->
    3.4% and 4.0% -> 2.9% on two held-out passages."""
    from onnxsim.gptq import _gptq_quantize_columns

    w = np.asarray(w, np.float32)
    codes0, s_w = weight_codes(w)
    w_nk = w.T.astype(np.float64)
    keep = np.abs(w_nk) == np.abs(w_nk).max(axis=1, keepdims=True)
    fixed = codes0.astype(np.float64) - 128
    lo = np.where(keep, fixed, -127.0)
    hi = np.where(keep, fixed, 127.0)
    xs = np.asarray(x, np.float64).reshape(-1, w.shape[0])
    codes = _gptq_quantize_columns(
        w_nk,
        s_w.astype(np.float64)[:, None],
        w.shape[0],
        xs.T @ xs,
        percdamp,
        128,
        code_bounds=(lo, hi),
        device=device,
    )
    out = (codes * s_w[:, None].astype(np.float64)).astype(np.float32)
    out[keep] = w_nk[keep].astype(np.float32)
    return out.T.copy()


def _requant_f64(codes, s_w, s_x, zp_x, s_y, zp_y):
    """``(bias, M)`` float32 per output channel, computed in float64 and cast
    once (float32 arithmetic is an ulp off on about a third of the channels)."""
    m = (np.float64(s_x) * np.asarray(s_w, np.float64) / np.float64(s_y)).astype(
        np.float32
    )
    qsum = (codes.astype(np.int64) - 128).sum(axis=1).astype(np.float64)
    bias = (np.float64(zp_y) - np.float64(zp_x) * qsum * m.astype(np.float64)).astype(
        np.float32
    )
    return bias, m


def _plain128_table(codes, bias, m):
    """128 x 128: plain code bytes, two 64-output tiles, one requantization block."""
    n, k = codes.shape
    t = np.full(2 * 9216, 128, np.uint8)  # unused bytes hold code 128
    o, i = np.arange(n), np.arange(k)
    row = 9216 * (o >> 6) + 576 * ((o >> 1) & 15) + 36 * (o & 1) + 72 * ((o >> 5) & 1)
    col = 144 * (i // 36) + i % 36
    t[(row[:, None] + col[None, :]).ravel()] = codes.ravel()
    return np.concatenate([t, bias.view(np.uint8), m.view(np.uint8)])


def _tall576_table(codes, bias, m):
    """576 x 64m: the square-576 tile per 64 outputs (36864 weight bytes, then
    the 512-byte requantization block)."""
    n, k = codes.shape
    tile = tile_bytes(576)
    t = np.zeros(n // TILE * tile, np.uint8)
    o = np.arange(n) % TILE
    tb = (np.arange(n) // TILE) * tile
    j = np.arange(k // 2)
    row = 1152 * (o & 15) + 72 * ((o >> 4) & 1) + 18432 * (o >> 5)
    col = 144 * (j // 36) + j % 36
    c = codes.reshape(n, k // 2, 2)
    addr = (tb + row)[:, None, None] + col[None, :, None] + 36 * np.arange(2)
    planes = [
        ((c[..., 0] >> (4 * p)) & 15) | (((c[..., 1] >> (4 * p)) & 15) << 4)
        for p in (0, 1)
    ]
    t[addr.ravel()] = np.stack(planes, -1).astype(np.uint8).ravel()
    for tt in range(n // TILE):
        at = tt * tile + WEIGHT_BYTES[576]
        ch = slice(TILE * tt, TILE * (tt + 1))
        t[at : at + BLOCK] = np.concatenate(
            [bias[ch].view(np.uint8), m[ch].view(np.uint8)]
        )
    return t


def _wide_table(codes, bias, m):
    """``wide``: per 32-output group, each segment's 16 weight rows (its pairs
    in 36-pair blocks of 144 bytes; low and high nibble planes at +0 and +36),
    then the group's 256-byte requantization block."""
    n, k = codes.shape
    segs, offs, g_bytes = wide_layout(k)
    ng = n // 32
    t = np.zeros(ng * g_bytes, np.uint8)
    o = np.arange(n)
    g, op = o >> 5, o & 31
    u = codes.astype(np.int64)
    rq = np.concatenate(
        [bias.reshape(ng, 32).view(np.uint8), m.reshape(ng, 32).view(np.uint8)], axis=1
    )  # [group, 256]
    start = 0
    for pairs, off in zip(segs, offs):
        s1 = _row_bytes(pairs)
        nslot = -(-pairs // 36) * 36
        jp = np.arange(nslot)
        slot = 144 * (jp // 36) + jp % 36
        ue = np.full((n, nslot), 128, np.int64)
        uo = np.full((n, nslot), 128, np.int64)
        ue[:, :pairs] = u[:, 2 * start : 2 * (start + pairs) : 2]
        uo[:, :pairs] = u[:, 2 * start + 1 : 2 * (start + pairs) : 2]
        row = s1 * (op & 15) + 72 * ((op >> 4) & 1)
        for p in (0, 1):
            byte = (((ue >> (4 * p)) & 15) | (((uo >> (4 * p)) & 15) << 4)).astype(
                np.uint8
            )
            t[(g * g_bytes + off)[:, None] + row[:, None] + slot[None, :] + 36 * p] = (
                byte
            )
        at = np.arange(ng) * g_bytes + off + 16 * s1
        t[at[:, None] + np.arange(256)[None, :]] = rq
        start += pairs
    return t


def _family_table(codes, s_w, s_x, zp_x, s_y, zp_y) -> np.ndarray:
    """The weight-and-requantization table of a non-square layer (the IO tail
    is the template's)."""
    n, k = codes.shape
    fam = family(k, n)
    bias, m = _requant_f64(codes, s_w, s_x, zp_x, s_y, zp_y)
    if fam == "plain128":
        return _plain128_table(codes, bias, m)
    if fam == "tall576":
        return _tall576_table(codes, bias, m)
    return _wide_table(codes, bias, m)


def linear_params(
    template_params: bytes,
    w: np.ndarray,
    s_x: float,
    zp_x: int,
    s_y: float,
    zp_y: int,
) -> bytes:
    """``npu_params`` of the layer for weights ``w [in, out]``: the weight
    codes at their addresses and the requantization blocks; the IO tail is the
    template's, and for the square layers every other byte is the template's."""
    w = np.asarray(w, np.float32)
    if w.ndim != 2:
        raise ValueError(f"expected weights [in, out], got {w.shape}")
    k, n = w.shape
    fam = family(k, n)
    if len(template_params) != layer_params_bytes(k, n):
        raise ValueError(
            f"the template's npu_params hold {len(template_params)} bytes, a "
            f"{k}x{n} linear layer {layer_params_bytes(k, n)}"
        )
    codes, s_w = weight_codes(w)
    table = np.frombuffer(template_params, np.uint8).copy()
    if fam in ("square64", "square576"):
        table[weight_addresses(n).ravel()] = weight_bytes(codes).ravel()
        for t in range(n // TILE):
            ch = slice(TILE * t, TILE * (t + 1))
            at = t * tile_bytes(n) + WEIGHT_BYTES[n]
            table[at : at + BLOCK] = emitter.requant_block(
                codes[ch], float(s_x), int(zp_x), float(s_y), int(zp_y), s_w[ch]
            )
        return table.tobytes()
    head = _table_bytes(k, n)
    table[:head] = _family_table(codes, s_w, s_x, zp_x, s_y, zp_y)
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


def template_shape(model: onnx.ModelProto) -> tuple[int, int]:
    """``(k, n)`` of a native ``[1, k] @ [k, n]`` linear build, after checking
    its IO shapes and ``npu_params`` size."""
    g = model.graph
    if len(g.input) != 1 or len(g.output) != 1:
        raise ValueError(
            f"expected one input and one output, found {len(g.input)} and {len(g.output)}"
        )
    shapes = [
        tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
        for v in [*g.input, *g.output]
    ]
    if any(len(sh) != 2 or sh[0] != 1 for sh in shapes):
        raise ValueError(
            f"expected IO shapes [1, n], found {[list(sh) for sh in shapes]}"
        )
    (_, k), (_, n) = shapes
    family(k, n)  # refuses a layout that was never built
    params = next(i for i in g.initializer if i.name == "npu_params")
    if len(params.raw_data) != layer_params_bytes(k, n):
        raise ValueError(
            f"the template's npu_params hold {len(params.raw_data)} bytes, a "
            f"{k}x{n} linear layer {layer_params_bytes(k, n)}"
        )
    return k, n


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
    k, n = template_shape(model)
    w = np.asarray(w, np.float32)
    if w.shape != (k, n):
        raise ValueError(f"the template is {k}x{n}; the weights are {w.shape}")
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


# ---- 16-bit activations (U16) -------------------------------------------------------
# A linear layer built with ``layer_configs: [{op_types: [FullyConnected], data_type: U16}]``
# keeps s8 per-channel weights (same codes, same requantization formula) but has its own
# table layout and two more calibration fields. Fitted on native U16 builds of 576 x 192,
# 576 x 576, 576 x 1536, 576 x 49152 and 1536 x 576 (template and held-out weights), and on
# three builds with real-activation calibration (another zp_x); see
# ``tests/test_axera_linear_emit_u16.py``.
#
# * Table: the ``wide`` layout with half-size blocks. Per 32-output group, per input
#   segment: 16 weight rows of ``72 * ceil(pairs / 18)`` bytes, then the group's 256-byte
#   requantization block. Input pair ``j`` of a row sits in 72-byte block ``j // 18`` at
#   ``j % 18``: low-nibble plane at +0, high plane at +18, outputs 16..31 of the group at
#   +36; unused slots hold code 128.
# * Segments (input pairs): ``nseg = ceil(pairs / 288)`` segments, the first holding
#   ``pairs - 272 * (nseg - 1)`` and the rest 272 each: ``k = 576`` is ``[288]``,
#   ``768`` is ``[112, 272]``, ``1536`` is ``[224, 272, 272]`` and ``3072`` is
#   ``[176, 272 x 5]``; Qwen3-0.6B's ``1024`` is ``[240, 272]`` and ``2048`` is
#   ``[208, 272 x 3]``. Checked (native U16 template / held-out pairs, byte-equal
#   emits) at those six ``k`` only; at any other ``k`` the template's ``npu_params``
#   size check (``linear_params_u16``) refuses a layout the rule gets wrong.
# * IO tail: 40 bytes, 100 for the 49152-wide layer, 180 for Qwen3-0.6B's 1024 x 151936
#   lm_head (its output split over 9 DEQUANT jobs); kept from the template.
# * Main engine: QUANT and the first DEQUANT job as in the U8 layer, zero points up to
#   65535; a wide output is split over CORE / DEQUANT pairs whose later DEQUANT jobs reuse
#   the first one's lanes and zero point.
# * Core (segments other than the main engine): ``0x03d0`` holds ``zp_x`` in its low half
#   (the upper half is per-record mode bits), so another input calibration rewrites it.

#: input widths the segment rule was checked on
U16_K = (576, 768, 1024, 1536, 2048, 3072)
U16_BLOCK = 18  # input pairs per 72-byte block
U16_SEGMENT = 272  # input pairs per segment after the first
U16_FIRST_MAX = 288  # input pairs a single segment holds
U16_TAILS = (40, 100, 180)
REG_CORE_ZP_X = 0x03D0


def u16_segments(k: int) -> list[int]:
    """Input pairs per segment of a U16 layer: ``ceil(pairs / 288)`` segments,
    the first ``pairs - 272 * (n - 1)``, the rest 272 (checked at ``U16_K``)."""
    if k <= 0 or k % 2:
        raise ValueError(
            f"U16 linear layer with k = {k}: k must be a positive even number"
        )
    pairs = k // 2
    nseg = -(-pairs // U16_FIRST_MAX)
    return [pairs - U16_SEGMENT * (nseg - 1)] + [U16_SEGMENT] * (nseg - 1)


def _u16_row_bytes(pairs: int) -> int:
    return 72 * -(-pairs // U16_BLOCK)


def u16_layout(k: int) -> tuple[list[int], list[int], int]:
    """``(segment pairs, segment offsets in a group, group bytes)`` of a U16 layer."""
    segs, offs, off = u16_segments(k), [], 0
    for pairs in segs:
        offs.append(off)
        off += 16 * _u16_row_bytes(pairs) + 256
    return segs, offs, off


def u16_table_bytes(k: int, n: int) -> int:
    """Bytes of the U16 weight-and-requantization table (the IO tail excluded)."""
    if n % 32 or n <= 0:
        raise ValueError(
            f"U16 linear layer {k}x{n}: n must be a positive multiple of 32"
        )
    return n // 32 * u16_layout(k)[2]


def _u16_table(codes, bias, m) -> np.ndarray:
    """The U16 weight-and-requantization table. Per group of 32 outputs and per
    segment: 16 rows of ``s1`` bytes (output ``o & 15``), each row 72-byte blocks
    of 18 input pairs, a block holding ``[o < 16: low nibbles, high nibbles][o >=
    16: low, high]`` (a byte packs pair ``(2j, 2j + 1)``'s nibbles), then the
    256-byte requantization block (bias, M of the 32 outputs). Written with
    reshapes rather than per-byte indices (an index per byte took seconds per
    table)."""
    n, k = codes.shape
    segs, offs, g_bytes = u16_layout(k)
    u16_table_bytes(k, n)
    ng = n // 32
    t = np.zeros((ng, g_bytes), np.uint8)
    u = np.asarray(codes).astype(np.uint8)
    rq = np.concatenate(
        [bias.reshape(ng, 32).view(np.uint8), m.reshape(ng, 32).view(np.uint8)], axis=1
    )  # [group, 256]
    start = 0
    for pairs, off in zip(segs, offs):
        s1 = _u16_row_bytes(pairs)
        nb = s1 // 72
        nslot = nb * U16_BLOCK
        ue = np.full((n, nslot), 128, np.uint8)
        uo = np.full((n, nslot), 128, np.uint8)
        ue[:, :pairs] = u[:, 2 * start : 2 * (start + pairs) : 2]
        uo[:, :pairs] = u[:, 2 * start + 1 : 2 * (start + pairs) : 2]
        byte = np.stack(
            [(ue & 15) | ((uo & 15) << 4), (ue >> 4) | ((uo >> 4) << 4)], axis=1
        )  # [n, p, nslot]
        # o = 32 g + 16 h + r -> [g, r, block, h, p, 18]
        byte = byte.reshape(ng, 2, 16, 2, nb, U16_BLOCK).transpose(0, 2, 4, 1, 3, 5)
        t[:, off : off + 16 * s1] = byte.reshape(ng, 16 * s1)
        t[:, off + 16 * s1 : off + 16 * s1 + 256] = rq
        start += pairs
    return t.reshape(-1)


def linear_params_u16(
    template_params: bytes,
    w: np.ndarray,
    s_x: float,
    zp_x: int,
    s_y: float,
    zp_y: int,
    table_only: bool = False,
) -> bytes:
    """``npu_params`` of a U16 layer for weights ``w [in, out]``; the IO tail is the
    template's. ``table_only``: the weight / requantization table alone (a
    fused program's ``npu_params`` holds it next to other constants)."""
    w = np.asarray(w, np.float32)
    if w.ndim != 2:
        raise ValueError(f"expected weights [in, out], got {w.shape}")
    k, n = w.shape
    if table_only:
        codes, s_w = weight_codes(w)
        bias, m = _requant_f64(codes, s_w, s_x, zp_x, s_y, zp_y)
        return _u16_table(codes, bias, m).tobytes()
    head = u16_table_bytes(k, n)
    if len(template_params) - head not in U16_TAILS:
        raise ValueError(
            f"the template's npu_params hold {len(template_params)} bytes, a U16 "
            f"{k}x{n} table {head} plus an IO tail of {' or '.join(map(str, U16_TAILS))}"
        )
    codes, s_w = weight_codes(w)
    bias, m = _requant_f64(codes, s_w, s_x, zp_x, s_y, zp_y)
    return _u16_table(codes, bias, m).tobytes() + template_params[head:]


def emit_linear_u16(
    template_model,
    w: np.ndarray,
    scales: Mapping[str, float],
    zero_points: Mapping[str, int],
) -> onnx.ModelProto:
    """``emit_linear`` for a U16 template (a native U16 build of the same shape):
    zero points ``1..65535``, the U16 table, and the core's copy of ``zp_x``."""
    model = gs.load_model(template_model)
    g = model.graph
    (_, k), (_, n) = [
        tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
        for v in (g.input[0], g.output[0])
    ]
    w = np.asarray(w, np.float32)
    if w.shape != (k, n):
        raise ValueError(f"the template is {k}x{n}; the weights are {w.shape}")
    x, y = g.node[0].input[0], g.node[0].output[0]
    missing = sorted({x, y} - (set(scales) & set(zero_points)))
    if missing:
        raise ValueError(f"no scale or zero point for tensors {missing}")
    s_x, s_y = float(np.float32(scales[x])), float(np.float32(scales[y]))
    zp_x, zp_y = int(zero_points[x]), int(zero_points[y])
    if not (0 < zp_x <= 65535 and 0 < zp_y <= 65535):
        raise ValueError(
            f"zero points x={zp_x}, y={zp_y}: both must be in 1..65535 (with 0 the "
            "native program leaves a zero-point write out; never built)"
        )
    if not (s_x > 0 and s_y > 0):
        raise ValueError(f"scales must be positive: x={s_x}, y={s_y}")

    mc = bytes(mre.mcode_initializer(model).raw_data)
    segs = [gs.records(raw) for raw in suc.decode_segments(mc)]
    main = segs[gs.MAIN]
    jobs = _jobs(main)
    roles = [j[0] for j in jobs]
    deq = [i for i, r in enumerate(roles) if r == "DEQUANT"]
    if (
        roles[:3] != ["PARAM", "QUANT", "PARAM"]
        or not deq
        or set(roles[3:]) - {"CORE", "DEQUANT"}
    ):
        raise ValueError(f"the template's main-engine jobs are {roles}")
    for j in deq[1:]:
        _, a, b, _ = jobs[j]
        if any(
            main[i][0] == gs.A1 and gs._reg(main[i]) in (*gs.LANES_LO, gs.REG_ZP_A)
            for i in range(a, b)
        ):
            raise ValueError("a later DEQUANT job writes its own lanes or zero point")
    _, q0, q1, q_before = jobs[1]
    zq = [
        gs._val(main[i])
        for i in range(q0, q1)
        if main[i][0] == gs.A1 and gs._reg(main[i]) == gs.REG_ZP_OUT
    ]
    if len(zq) != 1:
        raise ValueError("the template's QUANT job does not write zp_x exactly once")
    hits = 0
    for si, seg in enumerate(segs):
        if si == gs.MAIN:
            continue
        for i, r in enumerate(seg):
            if (
                r[0] == gs.A1
                and gs._reg(r) == REG_CORE_ZP_X
                and gs._val(r) & 0xFFFF == zq[0]
            ):
                seg[i] = gs._rec(gs.A1, REG_CORE_ZP_X, (gs._val(r) & 0xFFFF0000) | zp_x)
                hits += 1
    if not hits:
        raise ValueError(
            f"no core {REG_CORE_ZP_X:#06x} record holds the template's zp_x {zq[0]}"
        )
    _, d0, d1, d_before = jobs[deq[0]]
    _set_job(main, q0, q1, q_before, gs._f32bits(1 / s_x), gs.REG_ZP_OUT, zp_x, "QUANT")
    _set_job(main, d0, d1, d_before, gs._f32bits(s_y), gs.REG_ZP_A, zp_y, "DEQUANT")
    out = wr.with_segments(model, segs)
    params = next(i for i in out.graph.initializer if i.name == "npu_params")
    params.raw_data = linear_params_u16(bytes(params.raw_data), w, s_x, zp_x, s_y, zp_y)
    return out
