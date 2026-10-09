"""Stitch a fused op chain's AX650 MCode model from standalone per-op programs.

Pulsar2 compiles a graph of several ops into one ``neu mode`` node. This
module builds that node's model without Pulsar2's compiler: from one compiled
*standalone* program per op (a one-op graph at the op's shape, any
calibration) plus the fused graph's scales and zero points. ``stitch`` returns
the five decompressed segments and ``npu_params``; ``stitch_model`` wraps them
into a complete model. On the committed graphs (SiLU, three more two-op
graphs, RMSNorm, a single-head attention block; two calibrations each) the
result equals the native fused build in every decompressed record outside
segment 0's slot table, in ``npu_params`` and in the rest of the model proto
(``docs/axera-graph-stitch.md``, ``tests/test_axera_graph_stitch.py``).

Method
------

Every MCode segment is an LZ77 stream over 8-byte register records
(``short_unit_codec``). Segment 2 drives the main engine, segments 0 and 1 the
two matrix engines, segments 3 and 4 the two copy engines.

* **Jobs split at a9 launches.** A segment is a list of jobs, each a run of
  register writes ended by an ``a9`` launch. A standalone program's main
  engine jobs get a role from their register state at launch: PARAM (job type
  ``0x100601``, loads a quantization parameter word from ``npu_params``),
  QUANT (reads a graph input), DEQUANT (writes the graph output), CORE (the
  op itself; scratch to scratch). The fused program keeps every CORE job, the
  PARAM+QUANT pair of each graph input at its first consumer and the
  PARAM+DEQUANT pair of the graph output. The jobs that convert an
  intermediate tensor are dropped.
* **Register-state delta encoding.** The compiler writes only the registers
  whose value differs from the running state, in one fixed register order.
  The stitcher replays each op's own register state job by job (with the fused
  calibration and addresses substituted), merges the write orders of all
  source jobs into one canonical order, and emits each kept job as the
  difference to the fused running state.
* **Liveness gates.** A kept job must also restore a register that an earlier
  op left at another value, but only when the register is live in that job.
  A register's gates are the conditions that hold in every standalone job
  writing it: bits 8..11 of the job's ``0x0150`` select word and "mode
  register g is nonzero". A register is restored when all its gates are on.
* **Scratch allocation.** Scratch buffers are bump-allocated from
  ``0x2f7000`` in op order; inside an op in its own address order, with the
  sizes of its standalone layout. A tensor already held in a buffer is never
  quantized again.
* **Constant loaders.** An initializer operand (and Add's or MatMul's
  synthesized constant) lives in ``npu_params`` and is copied into scratch by
  a loader sub-program on a copy engine. Loaders are re-emitted from the
  standalone loader of that engine with length, offset and destination
  replaced. Initializer bytes are copied from the standalone program; the
  synthesized constants are recomputed from the fused calibration.
* **Sync records.** ``a2`` records carry "signal k of this engine" and "wait
  for signal k of engine e". Each job's dependencies are read from the
  standalone program's own waits and from the tensors it reads; the fused
  waits and signal numbers are derived from them.
* **FlatBuffer blob rebuild.** The MCode blob is an ordinary FlatBuffer (IO
  descriptors, one table per segment, the segment streams, a symbol list).
  ``parse_blob`` plus ``build_blob`` reproduce every committed blob byte for
  byte, so the fused blob is rebuilt from the fused IO list and segments.

Derived from standalone programs
--------------------------------

Checked on every standalone program loaded (``Program``) or by rebuilding it:
job roles; the IO slot numbering (``a8`` slots, the ``a7`` head, PARAM
offsets); calibration formulas of each op (Sigmoid table, Sqrt, Mul, Add,
ReduceMean, Div lanes and zero points, Softmax lanes and output data type,
MatMul ``npu_params`` lanes, QUANT/DEQUANT lanes and zero points); the
register order; the liveness gates of every register outside the two learned
groups below; scratch sizes; loader fields; the matrix-engine operand offset
(``0x0330``/``0x0320``); the sync mechanism (what a job waits for); segment
compression (segments 0 and 2 always, 4 never, 1 and 3 unless empty); the
FlatBuffer layout and the model wrapper.

Learned from a single native fused build
----------------------------------------

These are not derivable from the standalone programs. Each was read off one
native fused graph (both of its calibrations agree) and is marked ``LEARNED``
at its definition:

* **Always-live mode-register group** (SiLU): the registers written between
  the select strobe and ``0x0df0`` are restored whenever they differ, without
  gates.
* **Port-block gates** (RMSNorm): the registers of one input-port descriptor
  block share one gate set.
* **npu_params constant ordering** (RMSNorm): all initializer operands in op
  order, then all synthesized constants in op order.
* **Loader alternation across ops** (RMSNorm): the k-th loader of the whole
  graph runs on copy engine 4, 3, 4, ... regardless of the engine it used in
  its standalone program.
* **Sync/wait placement** (RMSNorm, attention): a job's waits precede all its
  register writes; the main engine does not repeat a wait for a signal it
  already waited for, the other engines repeat theirs.
* **Graph-input jobs before the first core** (attention): the PARAM+QUANT
  jobs of every graph input run first, whichever op consumes the input.
* **Attention's engine placement**: which engine runs each MatMul's
  sub-programs (``place`` in the wiring). The standalone MatMuls use other
  engines.
* **The main-engine transpose job** (attention): the second MatMul's operand
  transpose runs as a 30-record main-engine job that no standalone program
  contains. Its register list and four constants are ``MAIN_TRANSPOSE_JOB``,
  copied from the native fused attention build; only its operand addresses
  and permutation are taken from the standalone copy-engine job.

* **Where the output's PARAM job runs** (RMSNorm at ``[1,576]``): in the
  ``[1,64]`` graphs the graph output's PARAM job runs where the last op's
  standalone program has it. The native ``[1,576]`` RMSNorm runs it just
  before ReduceMean's core job instead, and standalone builds differ too
  (Sigmoid at 512 and ReduceMean at 512 and 576 run it before the core, the
  other widths after). Nothing found predicts it, so it is a wiring input
  (``output_param``). A job that moves is emitted from the register state its
  op has at that job in the standalone order.
* **Wide ReduceMean** (standalone ReduceMean at 64 to 2048, RMSNorm at 576):
  the lane is single precision, ``float32(float32(s_x / s_y) / n)``; the
  accumulator zero point is ``zp_x * min(n, 256)`` (one 256-wide pooling
  window); a core that writes the eight packed lanes ``0x0d30..0x0da0`` (its
  pad bytes, widths that are not a multiple of 256) holds ``zp_x`` in all four
  bytes of each. The pad group ``0x0cd0..0x0da0`` is live only while the pad
  mode word ``0x0c20`` is nonzero (an exception to the always-live mode group).

Three structural choices are assumptions: no build contradicts them and
none isolates them. Scratch buffers, PARAM words and each engine's
sub-programs are numbered in op order.

Refused
-------

``NotImplementedError``: an op outside ``SUPPORTED_OPS``; a Sqrt with a
nonzero input or output zero point and a Div with a nonzero divisor zero
point (no standalone build has one, so their zero-point registers are
unknown); two ops on one matrix engine; a copy-engine sub-program that would
need a register restore; a sub-program placed on an engine of another kind;
several MatMul-like ops without an explicit ``place`` for each;
a main-engine CORE job of an op with no calibration model; a copy-engine
loader with no standalone template. ``ValueError``: an ``output_param`` that
names no op with a core job; a tensor without a scale
or zero point; constants that do not account for a program's ``npu_params``;
constant values that do not quantize to the standalone program's bytes; a
standalone program that breaks one of the checked rules.

Limits: the fused scales and zero points are inputs (Pulsar2's
``quant_axmodel.json``, or ``pulsar_free_calibration`` from the float graph
and its samples); every program serves one shape (``width_retarget`` moves a
``[1,64]`` program to another width); the learned rules
rest on one native graph each; an engine placement other than the native
build's is accepted when no rule refuses it, but nothing has verified it.
Add's two ``npu_params`` scale words are written in ONNX input order, which
holds in both fused Add graphs and four of five standalone Add builds; the
fifth (``add``, asym) has them swapped.
"""

from __future__ import annotations

import base64
import gzip
import heapq
import json
import os
import struct
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import misc_op_record_emit as mre  # noqa: E402
import short_unit_codec as suc  # noqa: E402

FIXTURES = os.path.join(_HERE, "fixtures", "graph_stitch")
FIXTURE_INDEX = os.path.join(FIXTURES, "index.json")

REC = suc.RECORD
A1, A2, A3, A7, A8, A9 = 0xA1, 0xA2, 0xA3, 0xA7, 0xA8, 0xA9
SUPPORTED_OPS = (
    "Sigmoid",
    "Sqrt",
    "Mul",
    "Add",
    "ReduceMean",
    "Div",
    "Softmax",
    "MatMul",
)

# ---- main-engine register roles ------------------------------------------------
REG_CTL = 0x0150  # phase/strobe register: every write is emitted
REG_JOBTYPE = 0x0160
REG_SRC_A_SLOT, REG_SRC_A = 0x02B0, 0x02C0
REG_PARAM_SLOT = 0x03D0
REG_SRC_B_SLOT, REG_SRC_B = 0x04F0, 0x0500
REG_DST_SLOT, REG_DST = 0x0720, 0x0730
REG_A7_COUNT = 0x0100
REG_ZP_A, REG_ZP_B, REG_ZP_OUT = 0x1A90, 0x1AD0, 0x1B10
REG_DTYPE = 0x1B50  # data type of the tensor a job writes: 2 = u8, 3 = s8
REG_MODE_END = 0x0DF0
LANES_LO = tuple(range(0x0F50, 0x0FC1, 0x10))
LANES_HI = tuple(range(0x0FD0, 0x1041, 0x10))
# ReduceMean's pad lanes: the input zero point in all four bytes (a core over
# a width that is not a multiple of REDUCE_WINDOW pads its last window)
LANES_PACKED = tuple(range(0x0D30, 0x0DA1, 0x10))
REDUCE_WINDOW = 256
# LEARNED (standalone ReduceMean at 288 and 384): the pad group is live only
# while the pad mode word 0x0c20 is nonzero. The PARAM job after the padded
# core writes 0x0c20 = 0 and leaves the pad group alone.
REG_PAD_MODE = 0x0C20
PAD_GROUP = tuple(range(0x0CD0, 0x0DA1, 0x10))
TABLE_REGS = tuple(range(0x1050, 0x1851, 0x10))
ADD_OFFSET_REGS = (0x1EF0, 0x1F00, 0x1F10, 0x1F20)
JOB_PARAM = 0x100601
SELECT_BITS = (8, 9, 10, 11)
CTL_PHASES = (1, 0x100000, 0x1000000)  # any other 0x0150 value is a select word

# ---- scratch ---------------------------------------------------------------------
SCRATCH_BASE, SCRATCH_END = 0x2F7000, 0x2F8000
SCRATCH_UNIT = 0x20
A7_HIGH = 0x17C00
PAD_RECORDS = 4

# ---- engines ---------------------------------------------------------------------
MAIN, MATRIX_ENGINES, COPY_ENGINES = 2, (0, 1), (3, 4)
# Field registers of a copy-engine loader: copy ``len + 1`` bytes at
# ``npu_params`` offset ``off`` into scratch ``dst``.
LOADER_FIELDS = {
    3: dict(len=(0x0160, 0x0390), slot=0x0170, off=0x0180, dst=0x03B0, ctl=0x04B0),
    4: dict(len=(0x0220, 0x0330), slot=0x0230, off=0x0240, dst=0x0350, ctl=0x0150),
}
# LEARNED (RMSNorm): the k-th loader of the fused graph runs on this engine,
# whatever engine it used standalone.
LOADER_ENGINE_ORDER = (4, 3)
# A matrix-engine sub-program that writes 0x0330 holds there
# (MATRIX_B_ANCHOR - address in 0x0260) / 0x20, and its 30-bit negative in 0x0320.
MATRIX_B_ANCHOR, MATRIX_ADDR = 0x2F7E00, 0x0260
MATRIX_OFF_POS, MATRIX_OFF_NEG, MATRIX_CTL = 0x0330, 0x0320, 0x0220

# LEARNED (RMSNorm): the input-port descriptor registers form four blocks of
# 0x120 (A 0x0290.., B 0x03b0.., C 0x04d0.., D 0x05f0..). All registers of a
# block share one gate set, the intersection of their inferred gate sets. No
# standalone job shows it (none writes 0x02d0/0x0310 with 0x0df0 == 0).
PORT_BLOCK_BASE, PORT_BLOCK_SIZE, PORT_BLOCKS = 0x0290, 0x120, 4

# LEARNED (attention). Taken from a native Pulsar2 build, not derived: the
# fused attention block (fixtures/graph_stitch/fused_attention_pm1 and _pm4,
# identical there) transposes the second MatMul's B operand in a main-engine
# job. No standalone program has such a job (none has 0x0c20 != 0), so the
# job's register list, its record order (``_transpose_job``) and the four
# constants below are copied from that build. Only the operand values are
# taken from the standalone copy-engine (segment 3) transpose, register by
# register through the three ``(first, last, shift)`` maps.
MAIN_TRANSPOSE_JOB = {
    "constants": {
        0x0280: 0x00000001,
        0x0290: 0x0003FD00,
        0x0C10: 0x00100040,
        0x0C20: 0x00000104,
    },
    "select": 0x100,
    "source": (0x0160, 0x0210, 0x140),  # 0x0160..0x0210 -> 0x02a0..0x0350
    "permutation": (0x04E0, 0x0570, 0x750),  # 0x04e0..0x0570 -> 0x0c30..0x0cc0
    "destination": (0x0390, 0x0440, 0x380),  # 0x0390..0x0440 -> 0x0710..0x07c0
    "job_type_from": 0x04C0,  # copy-engine register holding the job type
}


def _f32(v: float) -> float:
    return float(np.float32(v))


def _f32bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(v)))[0]


def _reg(w: bytes) -> int:
    return w[2] | (w[3] << 8)


def _val(w: bytes) -> int:
    return int.from_bytes(w[4:], "little")


def _rec(verb: int, reg: int, value: int) -> bytes:
    return bytes([verb, 0, reg & 0xFF, reg >> 8]) + (value & 0xFFFFFFFF).to_bytes(
        4, "little"
    )


def records(raw: bytes) -> list[bytes]:
    """The 8-byte records of a decompressed segment."""
    return [bytes(raw[i : i + REC]) for i in range(0, len(raw), REC)]


def _is_scratch(v: int) -> bool:
    return SCRATCH_BASE <= v < SCRATCH_END


def _require(ok: bool, where: str, rule: str) -> None:
    if not ok:
        raise ValueError(f"{where}: does not follow the stitcher's rule: {rule}")


# ---- IO numbering (checked on every program loaded) --------------------------------
def slot_io(io: int) -> int:
    """``a8`` slot of graph IO tensor ``io`` (inputs, then outputs)."""
    return 2 * io + 1


def slot_param(io: int) -> int:
    """``a8`` slot (``0x03d0``) of the PARAM job paired with IO tensor ``io``."""
    return 2 * (io + 1)


def slot_params_blob(n_io: int) -> int:
    """``a8`` slot of the ``npu_params`` initializer."""
    return 2 * n_io + 1


def a7_head(n_io: int) -> int:
    """Register field of the ``a7`` record that opens every segment."""
    return 0x200 + 0x400 * n_io


def param_offset(n_io: int, const_len: int, k: int) -> int:
    """``npu_params`` offset read by the k-th PARAM job."""
    return const_len + 8 * n_io + 4 * k


def params_len(n_io: int, const_len: int) -> int:
    return const_len + 20 * n_io


def sync_signal(engine: int, k: int) -> int:
    """``a2`` value: signal ``k`` of ``engine``."""
    return (engine << 20) | (k << 4) | 2


def sync_wait(engine: int, k: int) -> int:
    """``a2`` value: wait for signal ``k`` of ``engine``."""
    return (engine << 20) | (k << 4) | 3


SYNC_END = 1  # ``a2`` value closing segment 0


def is_stub(recs: Sequence[bytes], engine: int) -> bool:
    """An engine with no sub-program: head, final signal 1, two pad records."""
    return (
        len(recs) == 4
        and recs[0][0] == A7
        and recs[1][0] == A2
        and _val(recs[1]) == sync_signal(engine, 1)
        and not any(recs[2])
        and not any(recs[3])
    )


def compress_shape(segs: Sequence[Sequence[bytes]]) -> list[bool]:
    """Which segments Pulsar2 stores compressed: 0 and 2 always, 4 never, 1
    and 3 unless they are stubs."""
    return [
        True if k in (0, 2) else (k != 4 and not is_stub(segs[k], k))
        for k in range(len(segs))
    ]


def slot_table(recs: Sequence[bytes]) -> tuple[int, int]:
    """``(first, last + 1)`` record indices of segment 0's slot table: its
    waits for the other engines' final signals. Their order is a per-build
    permutation (rebuild noise), so two builds are compared there as sets."""
    end = next(i for i, w in enumerate(recs) if w[0] == A2 and _val(w) == SYNC_END)
    i = end
    while i > 0 and recs[i - 1][0] == A2 and _val(recs[i - 1]) & 0xF == 3:
        i -= 1
    return i, end


# ---- calibration formulas (each checked on standalone builds) ------------------------
def sigmoid_table(s_x: float, zp_x: int, s_y: float, zp_y: int) -> dict[int, int]:
    """Sigmoid's 258-entry u8 table, two u16 entries per register."""
    q = np.arange(256) - zp_x
    t = np.clip(np.rint(1 / (1 + np.exp(-q * _f32(s_x))) / _f32(s_y)) + zp_y, 0, 255)
    t = t.astype(int).tolist()
    t += [t[255], 0]
    return {r: t[2 * k] | (t[2 * k + 1] << 16) for k, r in enumerate(TABLE_REGS)}


def add_offset(s_a, s_b, s_y, zp_a, zp_b, zp_y) -> int:
    """Add's zero-point offset word (``0x1ef0..0x1f20``), Q15."""
    v = (zp_y - zp_a * _f32(s_a / s_y) - zp_b * _f32(s_b / s_y)) * 2**15
    return int(v) & 0xFFFFFFFF


def add_const(s_a: float, s_b: float, s_y: float) -> bytes:
    """Add's ``npu_params`` constant: the two scale ratios as u16 fixed point,
    shifted down together until both are below 1."""
    ratios, k = (s_a / s_y, s_b / s_y), 0
    while max(ratios) * 2.0**-k >= 1.0:
        k += 1
    return struct.pack("<2H", *[int(round(x * 2.0 ** (15 - k))) for x in ratios])


def matmul_const(template: bytes, s_a, s_b, s_y, zp_y) -> bytes:
    """MatMul's ``npu_params`` constant: two equal blocks of float32 lanes, N
    lanes of ``float(zp_y)`` then N lanes of ``s_a * s_b / s_y``, each zero
    padded. N and the block size are read from the standalone constant."""
    half = len(template) // 2
    lanes = np.frombuffer(template[half : 2 * half], "<f4")
    n = int(np.count_nonzero(lanes))
    head = np.frombuffer(template[:half], "<f4")
    if not n or lanes[n:].any() or head[n:].any():
        raise ValueError("MatMul npu_params constant is not two padded lane blocks")

    def block(v):
        return np.full(n, v, "<f4").tobytes() + bytes(half - 4 * n)

    return block(np.float32(zp_y)) + block(np.float32(s_a * s_b / s_y))


def matrix_b_offset(addr: int) -> tuple[int, int]:
    """``(0x0330, 0x0320)`` values of a matrix-engine sub-program whose B
    operand sits at scratch ``addr``."""
    d = (MATRIX_B_ANCHOR - addr) // SCRATCH_UNIT
    return d, (-d) & 0x3FFFFFFF


def quantize_constant(values, scale: float, zero_point: int) -> bytes:
    """u8 bytes of an initializer operand: ``clip(rint(v / s) + zp, 0, 255)``."""
    v = np.asarray(values, dtype=np.float64).ravel()
    return (
        np.clip(np.rint(v / float(scale)) + int(zero_point), 0, 255)
        .astype(np.uint8)
        .tobytes()
    )


def reducemean_lane(s_x: float, s_y: float, n: int) -> int:
    """ReduceMean's lane word: ``float32(float32(s_x / s_y) / n)``, every step
    in single precision. The float64 form ``s_x / (s_y * n)`` is one ulp off
    on some builds (the standalone ``[1,384]`` build is one)."""
    f = np.float32
    return int(f(f(f(s_x) / f(s_y)) / f(n)).view(np.uint32))


def _core_values(
    op, core_index, ports, in_ts, out_t, sc, zp, signed, attrs, written=()
):
    """Register -> value owned by CORE job ``core_index`` of ``op``. ``ports``
    maps "A"/"B" to the fused tensor read through ``0x02c0``/``0x0500``;
    ``written`` is the set of registers the standalone job writes."""

    def port(name):
        if name not in ports:
            raise ValueError(f"{op} core job reads no tensor on port {name}")
        return ports[name]

    out: dict[int, int] = {}
    if op == "Sigmoid":
        t = port("A")
        out.update(sigmoid_table(sc[t], zp[t], sc[out_t], zp[out_t]))
    elif op == "Sqrt":
        t = port("A")
        if zp[t] or zp[out_t]:
            raise NotImplementedError(
                "Sqrt with a nonzero zero point: its zero-point registers are "
                "unknown (every standalone build has zero points 0)"
            )
        out.update({r: _f32bits(sc[t]) for r in LANES_LO})
        out.update({r: _f32bits(sc[out_t]) for r in LANES_HI})
    elif op == "Mul":
        a, b = port("A"), port("B")
        out[REG_ZP_A], out[REG_ZP_B], out[REG_ZP_OUT] = zp[a], zp[b], zp[out_t]
        out.update({r: _f32bits(sc[out_t] / (sc[a] * sc[b])) for r in LANES_HI})
    elif op == "Add":
        a, b = in_ts  # ONNX input order (the formula is symmetric)
        off = add_offset(sc[a], sc[b], sc[out_t], zp[a], zp[b], zp[out_t])
        out.update({r: off for r in ADD_OFFSET_REGS})
    elif op == "ReduceMean":
        if not attrs.get("count"):
            raise ValueError("ReduceMean needs attrs['count'], its reduced elements")
        t, n = port("A"), int(attrs["count"])
        out[REG_ZP_A], out[REG_ZP_OUT] = zp[t] * min(n, REDUCE_WINDOW), zp[out_t]
        lane = reducemean_lane(sc[t], sc[out_t], n)
        out.update({r: lane for r in LANES_LO})
        if LANES_PACKED[0] in written:
            out.update({r: zp[t] * 0x01010101 for r in LANES_PACKED})
    elif op == "Div":
        a, b = port("A"), port("B")
        if zp[b]:
            raise NotImplementedError(
                "Div with a nonzero divisor zero point: its register is "
                "unknown (every standalone build has divisor zero point 0)"
            )
        out[REG_ZP_A], out[REG_ZP_OUT] = zp[a], zp[out_t]
        out.update({r: _f32bits(sc[a] / (sc[b] * sc[out_t])) for r in LANES_LO})
    elif op == "Softmax":
        # two core jobs: s_x (the zero point of x does not enter, Softmax is
        # shift invariant), then 1/s_y and the output data type
        if core_index == 0:
            out.update({r: _f32bits(sc[in_ts[0]]) for r in LANES_LO})
        else:
            out.update({r: _f32bits(1 / sc[out_t]) for r in LANES_LO})
            out[REG_DTYPE] = 3 if out_t in signed else 2
    else:
        raise NotImplementedError(f"{op}: no calibration model for a main-engine job")
    return out


def _synth_const(op, program, in_ts, out_t, sc, zp, n_init: int) -> bytes:
    """The constant ``op`` synthesizes at the end of its ``npu_params`` head."""
    if op == "Add":
        return add_const(sc[in_ts[0]], sc[in_ts[1]], sc[out_t])
    if op == "MatMul":
        template = program.params[n_init : program.const_len]
        return matmul_const(template, sc[in_ts[0]], sc[in_ts[1]], sc[out_t], zp[out_t])
    return b""


# ---- standalone programs ---------------------------------------------------------
@dataclass
class Job:
    """One main-engine job of a standalone program."""

    index: int
    records: list[bytes]  # including the a2 waits that open it
    launch: dict[int, tuple[int, int]]  # register -> (verb, value) at the a9
    type: int
    written: set[int]
    written_a1: set[int]
    select: int
    role: str = "CORE"
    io: int | None = None
    tensor: str | None = None
    param_index: int | None = None
    waits: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Task:
    """One sub-program of a matrix or copy engine of a standalone program."""

    seg: int
    index: int
    records: list[bytes]
    waits: list[tuple[int, int]]
    launch: dict[int, tuple[int, int]]
    kind: str = "raw"  # "loader": copies npu_params bytes into scratch
    len: int = 0
    off: int = 0
    dst: int = 0


def _ctl_class(v: int):
    return v if v in CTL_PHASES else "select"


def _order_keys(job: Sequence[bytes]) -> list[tuple]:
    """Ordering key per record: ``0x0150`` by phase, an ``a7`` by the record it
    follows, an ``a2`` by its value, any other write by its register."""
    out, prev = [], None
    for w in job:
        if w[0] == A1 and _reg(w) == REG_CTL:
            k = ("ctl", _ctl_class(_val(w)))
        elif w[0] in (A1, A8):
            k = ("reg", _reg(w))
        elif w[0] == A7:
            k = ("a7", prev)
        elif w[0] == A2:
            k = ("a2", _val(w))
        else:
            k = ("verb", w[0])
        out.append(k)
        prev = k
    return out


def _parse_engine(recs, seg: int, n_io: int, where: str):
    """One segment as ``(tasks, signal positions, final waits)``: an ``a7``
    head; tasks ended by ``a9``, each preceded by its ``a2`` waits; signal k
    after the task another engine waits for; a final signal; segment 0 then
    waits for every other engine's final signal and ends."""
    _require(recs[0][0] == A7 and _reg(recs[0]) == a7_head(n_io), where, "a7 head")
    tasks, waits, cur, sigs, ended = [], [], [], [], False
    for w in recs[1:]:
        if not any(w):
            continue
        if w[0] == A2:
            _require(not cur, where, "no a2 record inside a job")
            v = _val(w)
            engine, k, kind = v >> 20, (v >> 4) & 0xFFFF, v & 0xF
            if kind == 3:
                waits.append((engine, k))
            elif kind == 2:
                _require(engine == seg, where, "an engine signals only itself")
                sigs.append((k, len(tasks) - 1))
            else:
                _require(v == SYNC_END and seg == 0, where, "a2 value")
                ended = True
            continue
        cur.append(w)
        if w[0] == A9:
            tasks.append((cur, waits))
            cur, waits = [], []
    _require(not cur and ended == (seg == 0), where, "segment end")
    _require(
        bool(sigs)
        and [k for k, _ in sigs] == list(range(1, len(sigs) + 1))
        and sigs[-1][1] == len(tasks) - 1,
        where,
        "signals are numbered from 1 and the last follows the last job",
    )
    _require(seg == 0 or not waits, where, "only segment 0 ends with waits")
    return tasks, {k: at for k, at in sigs[:-1]}, waits


class Program:
    """A standalone compiled single-op program, split into jobs with roles.

    Roles come from the register state at each job's launch:

    * PARAM: ``0x0160 == 0x100601``;
    * QUANT: source port A is a relocated external (``0x02c0`` written by ``a8``);
    * DEQUANT: the destination is a relocated external (``0x0730`` by ``a8``);
    * CORE: anything else (reads and writes scratch only).

    Loading checks the rules the stitcher relies on and raises ``ValueError``
    naming the first one a program breaks."""

    def __init__(self, model: onnx.ModelProto, label: str = "program"):
        self.model, self.label = model, label
        nodes = [n for n in model.graph.node if n.op_type == "neu mode"]
        _require(len(model.graph.node) == 1 and len(nodes) == 1, label, "one node")
        node = nodes[0]
        self.in_names, self.out_names = list(node.input), list(node.output)
        self.io_names = self.in_names + self.out_names
        self.n_io = len(self.io_names)
        self.mc = bytes(mre.mcode_initializer(model).raw_data)
        self.segs = [records(d) for d in suc.decode_segments(self.mc)]
        _require(len(self.segs) == 5, label, "five segments")
        self.params = bytes(
            next(i for i in model.graph.initializer if i.name == "npu_params").raw_data
        )
        self.const_len = len(self.params) - params_len(self.n_io, 0)
        _require(self.const_len >= 0, label, "npu_params holds 20 bytes per IO")
        self._parse_main()
        self._parse_engines()

    def _parse_main(self) -> None:
        label, n_in = self.label, len(self.in_names)
        tasks, sigpos, join = _parse_engine(self.segs[MAIN], MAIN, self.n_io, label)
        self.eng = {MAIN: dict(sigpos=sigpos, join=join)}
        raw_jobs, cur = [], []
        for w in self.segs[MAIN]:
            cur.append(w)
            if w[0] == A9:
                raw_jobs.append(cur)
                cur = []
        _require(len(raw_jobs) == len(tasks), label, "jobs end at a9 launches")
        state: dict[int, tuple[int, int]] = {}
        self.jobs: list[Job] = []
        self.addr_owner: dict[int, int] = {}  # scratch address -> first job using it
        for i, rj in enumerate(raw_jobs):
            for w in rj:
                if w[0] in (A1, A8):
                    state[_reg(w)] = (w[0], _val(w))
                if w[0] == A1 and _is_scratch(_val(w)):
                    self.addr_owner.setdefault(_val(w), i)
            selects = [
                _val(w)
                for w in rj
                if w[0] == A1 and _reg(w) == REG_CTL and _ctl_class(_val(w)) == "select"
            ]
            _require(bool(selects) and REG_JOBTYPE in state, label, "select + type")
            a2s = [n for n, w in enumerate(rj) if w[0] == A2]
            _require(a2s == list(range(len(a2s))), label, "a2 waits open a job")
            job = Job(
                index=i,
                records=rj,
                launch=dict(state),
                type=state[REG_JOBTYPE][1],
                written={_reg(w) for w in rj if w[0] in (A1, A8)},
                written_a1={_reg(w) for w in rj if w[0] == A1 and _reg(w) != REG_CTL},
                select=selects[0],
                waits=tasks[i][1],
            )

            def got(r):
                return state.get(r, (None, None))

            if job.type == JOB_PARAM:
                job.role = "PARAM"
                job.io = got(REG_PARAM_SLOT)[1] // 2 - 1
                _require(
                    got(REG_PARAM_SLOT) == (A8, slot_param(job.io))
                    and got(REG_SRC_B_SLOT) == (A8, slot_params_blob(self.n_io))
                    and got(REG_SRC_B)[0] == A1,
                    label,
                    "PARAM job slots",
                )
            elif got(REG_SRC_A)[0] == A8:
                job.role = "QUANT"
                job.io = (got(REG_SRC_A_SLOT)[1] - 1) // 2
                _require(
                    got(REG_SRC_A_SLOT) == (A8, slot_io(job.io)) and job.io < n_in,
                    label,
                    "QUANT job reads an input slot",
                )
            elif got(REG_DST)[0] == A8:
                job.role = "DEQUANT"
                job.io = (got(REG_DST_SLOT)[1] - 1) // 2
                _require(
                    got(REG_DST_SLOT) == (A8, slot_io(job.io))
                    and n_in <= job.io < self.n_io,
                    label,
                    "DEQUANT job writes an output slot",
                )
            if job.io is not None:
                _require(0 <= job.io < self.n_io, label, "IO index")
                job.tensor = self.io_names[job.io]
            self.jobs.append(job)
        # The PARAM jobs read the 4-byte words at const_len + 8 n_io + 4 k, k a
        # permutation of 0..n-1 (job order for the element-wise ops, ONNX
        # input order for MatMul, whose jobs run B first).
        params = [j for j in self.jobs if j.role == "PARAM"]
        base = param_offset(self.n_io, self.const_len, 0)
        offs = [j.launch[REG_SRC_B][1] for j in params]
        _require(
            sorted(offs) == [base + 4 * k for k in range(len(params))],
            label,
            "PARAM jobs read consecutive npu_params words",
        )
        for j, o in zip(params, offs):
            j.param_index = (o - base) // 4
        # a7 0x0100 of a QUANT/DEQUANT job = A7_HIGH | index (0x20 units) of
        # the buffer its paired PARAM job wrote
        param_of = {j.io: j for j in params}
        for j in self.jobs:
            for w in j.records:
                if w[0] == A7 and _reg(w) == REG_A7_COUNT:
                    p = param_of.get(j.io)
                    _require(
                        p is not None
                        and _val(w)
                        == A7_HIGH
                        | ((p.launch[REG_DST][1] - SCRATCH_BASE) // SCRATCH_UNIT),
                        label,
                        "a7 0x0100 names the PARAM job's buffer",
                    )
        self.dequant = [j for j in self.jobs if j.role == "DEQUANT"]
        _require(len(self.dequant) == 1, label, "one DEQUANT job")
        # tensor held by each scratch address: QUANT outputs and the op output
        self.addr_tensor = {
            j.launch[REG_DST][1]: j.tensor for j in self.jobs if j.role == "QUANT"
        }
        self.addr_tensor[self.dequant[0].launch[REG_SRC_A][1]] = self.dequant[0].tensor

    def _parse_engines(self) -> None:
        label = self.label
        self.etasks: list[Task] = []
        self.templates: dict[int, list[bytes]] = {}  # engine -> full loader
        for seg in (*MATRIX_ENGINES, *COPY_ENGINES):
            tasks, sigpos, join = _parse_engine(self.segs[seg], seg, self.n_io, label)
            self.eng[seg] = dict(sigpos=sigpos, join=join)
            state: dict[int, tuple[int, int]] = {}
            for ti, (recs, waits) in enumerate(tasks):
                for w in recs:
                    if w[0] in (A1, A8):
                        state[_reg(w)] = (w[0], _val(w))
                t = Task(seg, ti, recs, waits, dict(state))
                if seg in LOADER_FIELDS:
                    f, got = LOADER_FIELDS[seg], state.get
                    if (
                        got(f["slot"]) == (A8, slot_params_blob(self.n_io))
                        and got(f["off"], (0, 0))[0] == A1
                        and got(f["dst"], (0, 0))[0] == A1
                        and _is_scratch(got(f["dst"])[1])
                        and got(f["len"][0]) == got(f["len"][1])
                    ):
                        t.kind = "loader"
                        t.len = got(f["len"][0])[1]
                        t.off, t.dst = got(f["off"])[1], got(f["dst"])[1]
                        if ti == 0:
                            self.templates.setdefault(seg, recs)
                else:
                    for w in recs:
                        if w[0] == A1 and _reg(w) == MATRIX_OFF_POS:
                            d, neg = matrix_b_offset(state[MATRIX_ADDR][1])
                            _require(
                                _val(w) == d and state[MATRIX_OFF_NEG] == (A1, neg),
                                label,
                                "matrix-engine operand offset",
                            )
                self.etasks.append(t)
        self.loaders = sorted(
            (t for t in self.etasks if t.kind == "loader"), key=lambda t: t.dst
        )
        _require(
            [t.off for t in self.loaders] == sorted(t.off for t in self.loaders),
            label,
            "loader buffers are in npu_params offset order",
        )
        # bump allocator: a buffer's size is the distance to the next address
        addrs = set(self.addr_owner)
        for t in self.etasks:
            addrs |= {_val(w) for w in t.records if w[0] == A1 and _is_scratch(_val(w))}
        self.all_addrs = sorted(addrs)
        self.addr_size = {
            a: (self.all_addrs[n + 1] - a if n + 1 < len(self.all_addrs) else None)
            for n, a in enumerate(self.all_addrs)
        }


def load_model(source) -> onnx.ModelProto:
    """A compiled model from a ``ModelProto``, serialized bytes or a path
    (``.axmodel``, gzipped or not)."""
    if isinstance(source, onnx.ModelProto):
        return source
    if isinstance(source, (bytes, bytearray)):
        return onnx.load_model_from_string(bytes(source))
    opener = gzip.open if str(source).endswith(".gz") else open
    with opener(source, "rb") as f:
        return onnx.load_model_from_string(f.read())


def load_program(source, label: str | None = None) -> Program:
    """``Program`` for a standalone compiled single-op model."""
    if isinstance(source, Program):
        return source
    if label is None:
        label = source if isinstance(source, (str, os.PathLike)) else "program"
    return Program(load_model(source), os.path.basename(str(label)))


# ---- register order and liveness -------------------------------------------------
def canonical_rank(job_lists: Iterable[Iterable[Sequence[bytes]]]) -> dict:
    """One total register order: the topological merge of every source job's
    own write order (ties: first seen)."""
    seq: dict = {}
    succ: dict = {}
    indeg: dict = {}
    for jobs in job_lists:
        for job in jobs:
            keys = _order_keys(job)
            for k in keys:
                seq.setdefault(k, len(seq))
                succ.setdefault(k, set())
                indeg.setdefault(k, 0)
            for a, b in zip(keys, keys[1:]):
                if a != b and b not in succ[a]:
                    succ[a].add(b)
                    indeg[b] += 1
    by_seq = {n: k for k, n in seq.items()}
    heap = [seq[k] for k in seq if indeg[k] == 0]
    heapq.heapify(heap)
    rank = {}
    while heap:
        k = by_seq[heapq.heappop(heap)]
        rank[k] = len(rank)
        for b in succ[k]:
            indeg[b] -= 1
            if indeg[b] == 0:
                heapq.heappush(heap, seq[b])
    if len(rank) != len(seq):
        raise ValueError("source jobs disagree on register order (cycle)")
    return rank


def mode_group(rank: Mapping) -> frozenset[int]:
    """A job's mode words: the registers written between the ``0x0150`` select
    strobe and ``0x0df0`` (inclusive) in canonical order."""
    lo, hi = rank[("ctl", "select")], rank[("reg", REG_MODE_END)]
    return frozenset(k[1] for k, n in rank.items() if k[0] == "reg" and lo < n <= hi)


def job_gates(select: int, launch: Mapping, mode: Iterable[int]) -> frozenset:
    """The gates that are on in a job: ``("sel", k)`` for bit k of its select
    word, ``("en", g)`` for mode register g nonzero at launch."""
    on = {("sel", k) for k in SELECT_BITS if select >> k & 1}
    on |= {("en", g) for g in mode if launch.get(g, (A1, 0))[1]}
    return frozenset(on)


def infer_gates(programs: Iterable[Program], mode: frozenset[int]) -> dict:
    """Register -> gate set: the gates on in every standalone job writing it.

    LEARNED (SiLU): the mode group itself is always live (empty gate set).
    LEARNED (ReduceMean at 288/384): except its pad group, live only while
    ``0x0c20`` is nonzero.
    LEARNED (RMSNorm): the registers of one input-port block share the
    intersection of their gate sets."""
    gates: dict[int, frozenset] = {}
    for prog in programs:
        for j in prog.jobs:
            on = job_gates(j.select, j.launch, mode)
            for r in j.written_a1:
                gates[r] = gates[r] & on if r in gates else on
    for r in mode:
        gates[r] = frozenset()
    for r in PAD_GROUP:
        if r in gates:
            gates[r] = frozenset({("en", REG_PAD_MODE)})
    for b in range(PORT_BLOCKS):
        lo = PORT_BLOCK_BASE + b * PORT_BLOCK_SIZE
        regs = [r for r in gates if lo <= r < lo + PORT_BLOCK_SIZE and r not in mode]
        if regs:
            common = frozenset.intersection(*[gates[r] for r in regs])
            for r in regs:
                gates[r] = common
    return gates


# ---- the stitcher ------------------------------------------------------------------
@dataclass
class _Fused:
    """One job or sub-program of the fused program."""

    fid: int
    eng: int
    op: int
    kind: str  # "job" | "loader" | "raw" | "transpose"
    job: Job | None = None
    src: Task | None = None
    len: int = 0
    off: int = 0
    dst: int | None = None
    deps: set = field(default_factory=set)
    pos: int = 0
    req: dict = field(default_factory=dict)
    waits: list = field(default_factory=list)


@dataclass
class _OpState:
    index: int
    op: dict
    program: Program
    name: dict  # standalone tensor name -> fused tensor name
    known: frozenset  # fused tensors already in scratch before this op
    amap: dict = field(default_factory=dict)  # standalone -> fused scratch address
    addr_fused: dict = field(default_factory=dict)  # standalone address -> tensor
    param_k: dict = field(default_factory=dict)  # job index -> fused PARAM number
    shadow: dict = field(default_factory=dict)  # the op's own replayed state
    cursor: int = 0
    n_core: int = 0


@dataclass
class Stitched:
    """``stitch`` output: decompressed segments 0..4 and ``npu_params``."""

    segments: list[bytes]
    params: bytes
    programs: list[Program]
    wiring: dict
    log: list[dict]  # per emitted main-engine job: label, start, n, restores


def _normalize(wiring: Mapping) -> dict:
    ops = []
    for o in wiring["ops"]:
        o = dict(o)
        if o["op"] not in SUPPORTED_OPS:
            raise NotImplementedError(
                f"op {o['op']!r} is not modelled; supported: {SUPPORTED_OPS}"
            )
        o["in"] = dict(o["in"])
        o["out"] = tuple(o["out"])
        o["onnx_inputs"] = list(o.get("onnx_inputs") or o["in"].values())
        o["consts"] = [dict(c) for c in o.get("consts") or ()]
        o["attrs"] = dict(o.get("attrs") or {})
        o["place"] = dict(o.get("place") or {})
        ops.append(o)
    return dict(wiring, ops=ops, inputs=list(wiring["inputs"]))


def _scratch_refs(recs: Iterable[bytes]) -> set[int]:
    return {_val(w) for w in recs if w[0] == A1 and _is_scratch(_val(w))}


class _Stitcher:
    """One ``stitch`` call: constants, a plan (allocation, fused task list,
    dependencies), sync numbering, then the five segment streams."""

    def __init__(self, wiring, scales, zero_points, signed, gate_corpus):
        self.wiring = wiring = _normalize(wiring)
        self.sc = {k: float(v) for k, v in scales.items()}
        self.zp = {k: int(v) for k, v in zero_points.items()}
        self.signed = frozenset(signed)
        self.ops = wiring["ops"]
        self.graph_io = wiring["inputs"] + [wiring["output"]]
        self.n_io = len(self.graph_io)
        self.progs = [load_program(o["program"]) for o in self.ops]
        for o in self.ops:
            used = [*o["in"].values(), o["out"][1], *o["onnx_inputs"]]
            missing = sorted({t for t in used if t not in self.sc or t not in self.zp})
            if missing:
                raise ValueError(f"no scale or zero point for tensors {missing}")
        # Engine placement is not derivable (LEARNED, attention): with more than
        # one op using a matrix or copy engine the caller has to give it.
        engine_ops = [
            o
            for o, p in zip(self.ops, self.progs)
            if any(t.kind == "raw" for t in p.etasks)
        ]
        if len(engine_ops) > 1 and any(
            set(o["place"]) != {"matrix", "loader", "transpose"} for o in engine_ops
        ):
            raise NotImplementedError(
                "several ops with matrix-engine sub-programs: each needs an "
                "explicit place = {matrix, loader, transpose}"
            )
        self._lay_out_constants()
        # register order, liveness gates and loader templates
        corpus = list(self.progs) + [load_program(p) for p in gate_corpus]
        self.rank = canonical_rank([[j.records for j in p.jobs] for p in corpus])
        self.mode = mode_group(self.rank)
        self.gates = infer_gates(corpus, self.mode)
        self.templates: dict[int, list[bytes]] = {}
        for prog in corpus:
            for seg, tpl in prog.templates.items():
                self.templates.setdefault(seg, tpl)
        # the plan
        self.next_scratch = SCRATCH_BASE
        self.env: dict[str, int] = {}  # fused tensor -> scratch address
        self.producers: dict[str, set] = {}  # fused tensor -> producing task ids
        self.tasks: list[_Fused] = []
        self.eng_tasks: dict[int, list[_Fused]] = {e: [] for e in range(5)}
        self.states: list[_OpState] = []
        self.n_loader = self.n_param = 0
        first, rest = [], []  # main engine: graph-input jobs, then the others
        for oi in range(len(self.ops)):
            a, b = self._plan_op(oi)
            first += a
            rest += b
        self.eng_tasks[MAIN] = self._place_output_param(first + rest)
        for e in range(5):
            for n, t in enumerate(self.eng_tasks[e]):
                t.pos = n
        self._number_signals()

    def _place_output_param(self, main: list[_Fused]) -> list[_Fused]:
        """LEARNED (RMSNorm at 576; not predicted, see the module docstring):
        ``wiring["output_param"]`` moves the graph output's PARAM job to just
        before the first CORE job of op k (an int), or to just before the
        DEQUANT job (``"late"``). Without it the job stays where the last
        op's standalone program has it."""
        where = self.wiring.get("output_param")
        if where is None:
            return main

        def find(what, test):
            at = next((i for i, t in enumerate(main) if test(t)), None)
            if at is None:
                raise ValueError(f"output_param={where!r}: the graph has no {what}")
            return at

        def role(t, name):
            return t.kind == "job" and t.job.role == name

        out = self.wiring["output"]
        at = find(
            "PARAM job of its output",
            lambda t: role(t, "PARAM") and self.states[t.op].name[t.job.tensor] == out,
        )
        if where == "late":
            task = main.pop(at)
            main.insert(find("DEQUANT job", lambda t: role(t, "DEQUANT")), task)
            return main
        if isinstance(where, bool) or not isinstance(where, int):
            raise ValueError(f"output_param={where!r}: expected an op index or 'late'")
        core = find(
            f"CORE job of op {where}", lambda t: role(t, "CORE") and t.op == where
        )
        task = main.pop(at)
        main.insert(core - (at < core), task)
        return main

    def result(self) -> Stitched:
        main = self._main_segment()
        segments = [
            main if e == MAIN else b"".join(self._engine_segment(e)) for e in range(5)
        ]
        params = self.const_bytes + bytes(20 * self.n_io)
        return Stitched(segments, params, self.progs, self.wiring, self.log)

    # ---- constants -----------------------------------------------------------------
    def _lay_out_constants(self) -> None:
        """An op's ``npu_params`` head is its initializer operands (quantized
        bytes), then its synthesized constant."""
        sc, zp = self.sc, self.zp
        inits, synths, self.items = [], [], []
        for oi, (o, prog) in enumerate(zip(self.ops, self.progs)):
            at, mine = 0, []
            for c in o["consts"]:
                n = int(c["size"])
                data = prog.params[at : at + n]  # copied from the standalone program
                if c.get("values") is not None:
                    want = quantize_constant(c["values"], sc[c["name"]], zp[c["name"]])
                    if want != data:
                        raise ValueError(
                            f"constant {c['name']!r} of op {oi} does not quantize "
                            "to the bytes its standalone program was built with"
                        )
                mine.append(dict(kind="init", name=c["name"], base=at, data=data))
                at += n
            synth = _synth_const(
                o["op"], prog, o["onnx_inputs"], o["out"][1], sc, zp, at
            )
            if synth:
                mine.append(dict(kind="synth", name=None, base=at, data=synth))
                at += len(synth)
            if at != prog.const_len:
                raise ValueError(
                    f"op {oi} ({o['op']}): its constants account for {at} of the "
                    f"{prog.const_len} constant bytes in the program's npu_params"
                )
            self.items.append(mine)
            inits += [m for m in mine if m["kind"] == "init"]
            synths += [m for m in mine if m["kind"] == "synth"]
        # LEARNED (RMSNorm): all initializer operands, then all synthesized ones
        at = 0
        for m in inits + synths:
            m["fused_base"] = at
            at += len(m["data"])
        self.const_bytes = b"".join(m["data"] for m in inits + synths)

    # ---- plan: allocation, fused tasks, dependencies -----------------------------------
    def _kept(self, st: _OpState, j: Job) -> bool:
        """CORE always; PARAM+DEQUANT of the graph output; PARAM+QUANT of a
        graph input at its first consumer, whatever the op."""
        t = st.name[j.tensor] if j.tensor else None
        if j.role == "CORE" or t == self.wiring["output"]:
            return True
        if t in st.known:
            return False
        return t in self.wiring["inputs"]

    def _new_task(self, **kw) -> _Fused:
        t = _Fused(fid=len(self.tasks), **kw)
        self.tasks.append(t)
        return t

    def _plan_op(self, oi: int) -> tuple[list[_Fused], list[_Fused]]:
        """Plan op ``oi``; returns its main-engine tasks as (graph-input jobs
        and main-engine transposes, the other jobs)."""
        o, prog = self.ops[oi], self.progs[oi]
        name = dict(o["in"])
        name[o["out"][0]] = o["out"][1]
        unknown = sorted(set(prog.io_names) - set(name))
        if unknown:
            raise ValueError(f"op {oi} ({o['op']}): no wiring for tensors {unknown}")
        st = _OpState(oi, o, prog, name, frozenset(self.env))
        self.states.append(st)
        st.addr_fused = {a: name[t] for a, t in prog.addr_tensor.items()}
        loader_off = {}  # the loaders' offsets in the fused npu_params
        for ld in prog.loaders:
            m = next(
                (
                    m
                    for m in self.items[oi]
                    if m["base"] <= ld.off < m["base"] + len(m["data"])
                ),
                None,
            )
            if m is None:
                raise ValueError(f"op {oi}: a loader reads outside its constants")
            loader_off[ld.seg, ld.index] = m["fused_base"] + (ld.off - m["base"])
            if m["kind"] == "init":
                _require(
                    ld.off == m["base"] and ld.len == len(m["data"]) - 1,
                    prog.label,
                    "one loader per initializer operand",
                )
                st.addr_fused[ld.dst] = m["name"]

        # Allocation: per op, the op's own address order and sizes; a buffer is
        # allocated when a kept job or an engine sub-program references it.
        env = self.env
        amap = {a: env[name[t]] for a, t in prog.addr_tensor.items() if name[t] in env}
        kept_jobs = [j for j in prog.jobs if self._kept(st, j)]
        used = set()
        for x in [*kept_jobs, *prog.etasks]:
            used |= _scratch_refs(x.records)
        for a in prog.all_addrs:
            if a in amap:
                continue
            if a in used:
                size = prog.addr_size[a]
                amap[a] = self.next_scratch
                self.next_scratch += size if size is not None else SCRATCH_UNIT
            else:
                amap[a] = None
        st.amap = amap

        # PARAM numbering: inside an op its own offset order, across ops op order
        for j in sorted(
            (j for j in kept_jobs if j.role == "PARAM"), key=lambda j: j.param_index
        ):
            st.param_k[j.index] = self.n_param
            self.n_param += 1

        # this op's jobs and engine sub-programs
        by_src: dict[tuple[int, int], _Fused] = {}
        for j in kept_jobs:
            by_src[MAIN, j.index] = self._new_task(eng=MAIN, op=oi, kind="job", job=j)
        mine: dict[int, list[_Fused]] = {e: [] for e in (0, 1, 3, 4)}
        place = o["place"]
        for ld in prog.loaders:
            # LEARNED (RMSNorm): loaders alternate over the whole graph
            seg = place.get("loader", LOADER_ENGINE_ORDER[self.n_loader % 2])
            self.n_loader += 1
            t = self._new_task(
                eng=seg,
                op=oi,
                kind="loader",
                src=ld,
                len=ld.len,
                off=loader_off[ld.seg, ld.index],
                dst=amap[ld.dst],
            )
            by_src[ld.seg, ld.index] = t
            mine[seg].append(t)
        transposes = []
        for src in prog.etasks:
            if src.kind != "raw":
                continue
            key = "matrix" if src.seg in MATRIX_ENGINES else "transpose"
            seg = place.get(key, src.seg)
            t = self._new_task(eng=seg, op=oi, kind="raw", src=src)
            by_src[src.seg, src.index] = t
            if seg == MAIN:
                if src.seg not in COPY_ENGINES:
                    raise NotImplementedError(
                        "only a copy-engine transpose can run on the main engine"
                    )
                t.kind = "transpose"
                transposes.append(t)
            else:
                mine[seg].append(t)
        for e in mine:  # engine task order: op order; in an op loaders first
            self.eng_tasks[e] += mine[e]

        # Dependencies: a wait on signal k of engine E means "after every task
        # of E up to that signal". A dropped QUANT job stands for the producers
        # of its tensor.
        producers = self.producers

        def fused_of(seg, idx):
            if (seg, idx) in by_src:
                return {by_src[seg, idx].fid}
            if seg == MAIN:
                j = prog.jobs[idx]
                if j.role == "QUANT" and name[j.tensor] in producers:
                    return set(producers[name[j.tensor]])
            return set()

        def prefix(waits):
            out = set()
            for e, k in waits:
                _require(k in prog.eng[e]["sigpos"], prog.label, "wait target")
                for i in range(prog.eng[e]["sigpos"][k] + 1):
                    out |= fused_of(e, i)
            return out

        def addr_deps(recs):
            out = set()
            for w in recs:
                t = prog.addr_tensor.get(_val(w)) if w[0] == A1 else None
                if t is not None and name[t] in st.known:
                    out |= producers.get(name[t], set())
            return out

        for j in kept_jobs:
            by_src[MAIN, j.index].deps = prefix(j.waits) | addr_deps(j.records)
        for src in prog.etasks:
            deps = prefix(src.waits) | addr_deps(src.records)
            by_src[src.seg, src.index].deps = deps
        for j in kept_jobs:
            if j.role == "QUANT":
                producers[name[j.tensor]] = {by_src[MAIN, j.index].fid}
        dq = prog.dequant[0]
        made = prefix(dq.waits)
        for i in range(dq.index):
            made |= fused_of(MAIN, i)
        producers[o["out"][1]] = made

        for a, t in st.addr_fused.items():
            if amap.get(a) is not None:
                env.setdefault(t, amap[a])

        # LEARNED (attention): the PARAM+QUANT jobs of every graph input (op
        # order; then the op's main-engine transpose) run before the first core.
        p1 = [
            j
            for j in kept_jobs
            if j.role in ("PARAM", "QUANT") and name[j.tensor] in self.wiring["inputs"]
        ]
        p2 = [j for j in kept_jobs if j not in p1]
        _require(
            not p1 or not p2 or max(j.index for j in p1) < min(j.index for j in p2),
            prog.label,
            "input jobs come first",
        )
        return (
            [by_src[MAIN, j.index] for j in p1] + transposes,
            [by_src[MAIN, j.index] for j in p2],
        )

    # ---- waits and signals ---------------------------------------------------------
    def _number_signals(self) -> None:
        """A task waits, per other engine, for the last task of that engine
        among its transitive dependencies.

        LEARNED (attention; RMSNorm agrees): the main engine does not repeat a
        wait it already performed (equal or later signal); the other engines
        emit every wait."""
        tasks = self.tasks

        def need(t: _Fused) -> dict[int, int]:
            seen, todo = set(), list(t.deps)
            while todo:
                f = todo.pop()
                if f not in seen:
                    seen.add(f)
                    todo += tasks[f].deps
            req: dict[int, int] = {}
            for f in seen:
                x = tasks[f]
                if x.eng != t.eng:
                    req[x.eng] = max(req.get(x.eng, -1), x.pos)
            return req

        sigset: dict[int, set] = {e: set() for e in range(5)}
        for e in range(5):
            waited: dict[int, int] = {}
            for t in self.eng_tasks[e]:
                req = need(t)
                if e == MAIN:
                    req = {x: p for x, p in req.items() if p > waited.get(x, -1)}
                    waited.update(req)
                t.req = req
                for x, p in req.items():
                    sigset[x].add(p)
        # signal number of each waited-for task position, and the final signal
        self.signo = {
            e: {p: n + 1 for n, p in enumerate(sorted(sigset[e]))} for e in range(5)
        }
        self.n_final = {e: len(sigset[e]) + 1 for e in range(5)}
        for e in range(5):
            for t in self.eng_tasks[e]:
                t.waits = [
                    sync_wait(x, self.signo[x][p]) for x, p in sorted(t.req.items())
                ]

    def _signals_after(self, t: _Fused) -> list[bytes]:
        if t.pos not in self.signo[t.eng]:
            return []
        return [_rec(A2, 0, sync_signal(t.eng, self.signo[t.eng][t.pos]))]

    # ---- main engine (segment 2) -------------------------------------------------------
    def _main_segment(self) -> bytes:
        self.out: list[bytes] = []
        self.running: dict[int, tuple[int, int]] = {}  # fused register state
        self.param_dst: dict[int, int] = {}  # fused IO index -> PARAM job's buffer
        self.log: list[dict] = []
        out = self.out
        # (op, job index) -> the op's (shadow, core count) before that job; a
        # job emitted after later jobs of its op (``output_param``) is emitted
        # from the state its op has at that job in the standalone order
        before: dict[tuple[int, int], tuple[dict, int]] = {}
        for t in self.eng_tasks[MAIN]:
            st = self.states[t.op]
            if t.kind == "job" and t.job.index < st.cursor:
                now = st.shadow, st.n_core
                shadow, st.n_core = before[t.op, t.job.index]
                st.shadow = dict(shadow)
                self._run_job(st, t.job, t.waits)
                st.shadow, st.n_core = now
            elif t.kind == "job":
                for skipped in st.program.jobs[st.cursor : t.job.index]:
                    before[t.op, skipped.index] = dict(st.shadow), st.n_core
                    self._run_job(st, skipped, None)
                self._run_job(st, t.job, t.waits)
                st.cursor = t.job.index + 1
            else:
                start = len(out)
                out += [_rec(A2, 0, v) for v in t.waits]
                for verb, r, v in _transpose_job(t.src.launch, st.amap):
                    self._emit(verb, r, v)
                label = f"op{t.op}:{st.op['op']}:transpose"
                self.log.append(
                    dict(label=label, start=start, n=len(out) - start, restores=[])
                )
            out += self._signals_after(t)
        out.append(_rec(A2, 0, sync_signal(MAIN, self.n_final[MAIN])))
        out += [bytes(REC)] * (-len(out) % PAD_RECORDS)
        return b"".join(out)

    def _emit(self, verb: int, r: int, v: int) -> None:
        """An a1 write is dropped when the register already holds the value;
        0x0150 and every other verb are always emitted."""
        v &= 0xFFFFFFFF
        if verb == A1 and r != REG_CTL and self.running.get(r) == (A1, v):
            return
        self.out.append(_rec(verb, r, v))
        if verb in (A1, A8):
            self.running[r] = (verb, v)

    def _owned_values(self, st: _OpState, j: Job) -> dict[int, int]:
        """The calibration registers job ``j`` owns, whatever its standalone
        delta happened to write."""
        sc, zp = self.sc, self.zp
        t = st.name[j.tensor] if j.tensor else None
        if j.role == "QUANT":
            return {**{r: _f32bits(1 / sc[t]) for r in LANES_LO}, REG_ZP_OUT: zp[t]}
        if j.role == "DEQUANT":
            return {**{r: _f32bits(sc[t]) for r in LANES_LO}, REG_ZP_A: zp[t]}
        if j.role != "CORE":
            return {}
        ports = {}
        for port, r in (("A", REG_SRC_A), ("B", REG_SRC_B)):
            verb, v = j.launch.get(r, (None, None))
            if verb == A1 and v in st.addr_fused:
                ports[port] = st.addr_fused[v]
        o = st.op
        st.n_core += 1
        return _core_values(
            o["op"],
            st.n_core - 1,
            ports,
            o["onnx_inputs"],
            o["out"][1],
            sc,
            zp,
            self.signed,
            o["attrs"],
            j.written,
        )

    def _fused_record(self, st, j, w, owned, keep):
        """``(verb, register, value)`` of standalone record ``w`` in the fused
        program; value ``None`` when the fused program has no such value."""
        n_io, prog = self.n_io, st.program
        verb, r, v = w[0], _reg(w), _val(w)
        t = st.name[j.tensor] if j.tensor else None
        io = self.graph_io.index(t) if t in self.graph_io else None
        if verb == A1 and r in owned:
            return verb, r, owned[r]
        if verb == A1 and _is_scratch(v):
            return verb, r, st.amap[v]
        if verb == A7 and r == a7_head(prog.n_io):
            return verb, a7_head(n_io), v
        if verb == A7 and r == REG_A7_COUNT:
            if io is None or io not in self.param_dst:
                return verb, r, None
            index = (self.param_dst[io] - SCRATCH_BASE) // SCRATCH_UNIT
            return verb, r, A7_HIGH | index
        if verb == A8 and v:
            if j.role == "PARAM" and r == REG_PARAM_SLOT:
                return verb, r, slot_param(io) if io is not None else None
            if j.role == "PARAM" and r == REG_SRC_B_SLOT:
                return verb, r, slot_params_blob(n_io)
            if (j.role, r) in (("QUANT", REG_SRC_A_SLOT), ("DEQUANT", REG_DST_SLOT)):
                return verb, r, slot_io(io) if io is not None else None
        if verb == A1 and j.role == "PARAM" and r == REG_SRC_B:
            if not keep:
                return verb, r, None
            k = st.param_k[j.index]
            return verb, r, param_offset(n_io, len(self.const_bytes), k)
        return verb, r, v

    def _run_job(self, st: _OpState, j: Job, waits: Sequence[int] | None) -> None:
        """Replay job ``j`` into the op's shadow state; emit it (``waits`` not
        ``None``) as the difference to the fused running state."""
        keep, rank, out = waits is not None, self.rank, self.out
        owned = self._owned_values(st, j)
        todo = []  # (canonical rank, tie break, (verb, register, value))
        for n, (k, w) in enumerate(zip(_order_keys(j.records), j.records)):
            if w[0] == A2:
                continue  # sync records are regenerated
            if w[0] == A7 and _reg(w) == a7_head(st.program.n_io) and out:
                continue  # the a7 head opens the segment, not each op
            todo.append((rank[k], n, self._fused_record(st, j, w, owned, keep)))
        shadow = st.shadow
        for _, _, (verb, r, v) in todo:
            if verb in (A1, A8):
                shadow[r] = (verb, v)
        for r, v in owned.items():
            shadow[r] = (A1, v)
        if not keep:
            return
        # LEARNED (RMSNorm): the waits precede every register write of the job
        for n, v in enumerate(waits):
            todo.append((-1, -10 + n, (A2, 0, v)))
        # restore every live register an earlier op left at another value
        restores = []
        on = job_gates(j.select, shadow, self.mode)
        for r, (verb, v) in shadow.items():
            if r in j.written or r == REG_CTL or verb != A1 or v is None:
                continue
            if self.running.get(r) == (verb, v):
                continue
            if not self.gates.get(r, frozenset()) <= on:
                continue  # not live in this job
            restores.append(r)
            todo.append((rank[("reg", r)], -1, (verb, r, v)))
        todo.sort(key=lambda it: (it[0], it[1]))
        start = len(out)
        label = f"op{st.index}:{st.op['op']}:job{j.index}:{j.role}"
        for _, _, (verb, r, v) in todo:
            if v is None:
                raise ValueError(f"{label}: no fused value for register {r:#06x}")
            self._emit(verb, r, v)
        self.log.append(
            dict(label=label, start=start, n=len(out) - start, restores=restores)
        )
        if j.role == "PARAM":
            t = st.name[j.tensor]
            self.param_dst[self.graph_io.index(t)] = self.running[REG_DST][1]

    # ---- matrix and copy engines (segments 0, 1, 3, 4) -----------------------------------
    def _loader_records(self, seg: int, t: _Fused) -> list[tuple[int, int, int]]:
        """The engine's full standalone loader with its fields replaced."""
        f = LOADER_FIELDS[seg]
        if seg not in self.templates:
            raise NotImplementedError(
                f"no standalone program has a loader on copy engine {seg}"
            )
        full = []
        for w in self.templates[seg]:
            verb, r, v = w[0], _reg(w), _val(w)
            if verb == A1 and r in f["len"]:
                v = t.len
            elif verb == A1 and r == f["off"]:
                v = t.off
            elif verb == A1 and r == f["dst"]:
                v = t.dst
            elif verb == A8 and r == f["slot"]:
                v = slot_params_blob(self.n_io)
            full.append((verb, r, v))
        return full

    def _moved_records(self, seg: int, t: _Fused) -> list[tuple[int, int, int]]:
        """A standalone matrix or transpose sub-program at its fused addresses."""
        src, amap = t.src, self.states[t.op].amap
        if (seg in MATRIX_ENGINES) != (src.seg in MATRIX_ENGINES) or (
            seg in COPY_ENGINES and seg != src.seg
        ):
            raise NotImplementedError(
                f"an engine-{src.seg} sub-program cannot run on engine {seg}: "
                "the register layouts differ"
            )
        full = []
        for w in src.records:
            verb, r, v = w[0], _reg(w), _val(w)
            if verb == A1 and _is_scratch(v):
                v = amap[v]
            if (
                seg in MATRIX_ENGINES
                and verb == A1
                and r in (MATRIX_OFF_POS, MATRIX_OFF_NEG)
            ):
                addr = amap[src.launch[MATRIX_ADDR][1]]
                v = matrix_b_offset(addr)[r == MATRIX_OFF_NEG]
            full.append((verb, r, v))
        return full

    def _engine_segment(self, seg: int) -> list[bytes]:
        recs = [_rec(A7, a7_head(self.n_io), 0)]
        state: dict[int, int] = {}  # the engine's running a1 state
        ctl = LOADER_FIELDS[seg]["ctl"] if seg in LOADER_FIELDS else MATRIX_CTL
        owner = None
        for t in self.eng_tasks[seg]:
            recs += [_rec(A2, 0, v) for v in t.waits]
            delta = True
            if t.kind == "loader":
                full = self._loader_records(seg, t)
            elif seg in MATRIX_ENGINES:
                # Matrix engines 0 and 1 share one register layout. An op's
                # sub-programs keep their own order and deltas, which is only
                # valid when the op has the engine to itself.
                full = self._moved_records(seg, t)
                own = [x.src.index for x in self.eng_tasks[seg] if x.op == t.op]
                if owner not in (None, t.op) or own != list(range(len(own))):
                    raise NotImplementedError(
                        "two ops on one matrix engine: the register restore "
                        "rules are unknown"
                    )
                owner, delta = t.op, False
            else:
                # A copy-engine sub-program is written as the difference to the
                # engine's running state. One that was not first on its
                # standalone engine is only usable when the registers it left
                # unwritten already hold its launch values.
                full = self._moved_records(seg, t)
                amap, written = self.states[t.op].amap, {r for _, r, _ in full}
                for r, (verb, v) in t.src.launch.items() if t.src.index else ():
                    if verb == A1 and r != ctl and r not in written:
                        v = amap[v] if _is_scratch(v) else v
                        if state.get(r) != v:
                            raise NotImplementedError(
                                f"copy-engine sub-program: register {r:#06x} "
                                "needs a restore (order unknown)"
                            )
            for verb, r, v in full:
                if delta and verb == A1 and r != ctl:
                    if state.get(r) == v:
                        continue
                    state[r] = v
                recs.append(_rec(verb, r, v))
            recs += self._signals_after(t)
        recs.append(_rec(A2, 0, sync_signal(seg, self.n_final[seg])))
        if seg == 0:
            # segment 0's slot-table order is per-build noise: the first
            # program's order is used
            order = [e for e, _ in self.progs[0].eng[0]["join"]]
            recs += [_rec(A2, 0, sync_wait(e, self.n_final[e])) for e in order]
            recs.append(_rec(A2, 0, SYNC_END))
        recs += [bytes(REC)] * (-len(recs) % PAD_RECORDS)
        return recs


def stitch(
    wiring: Mapping,
    scales: Mapping[str, float],
    zero_points: Mapping[str, int],
    signed: Iterable[str] = (),
    gate_corpus: Iterable = (),
) -> Stitched:
    """The fused program of an op chain, from standalone programs only.

    ``wiring``::

        {"inputs": [graph inputs in compiled slot order], "output": name,
         "graph_inputs": [the source graph's input order],   # optional
         "output_param": op index | "late",                  # optional
         "ops": [{"op": "Mul", "program": path | bytes | ModelProto | Program,
                  "in": {standalone input name: fused tensor},
                  "out": (standalone output name, fused tensor),
                  "onnx_inputs": [fused names in ONNX input order],  # with consts
                  "consts": [{"name": fused name, "size": bytes, "values": [...]}],
                  "attrs": {"count": 64},              # ReduceMean
                  "place": {"matrix": e, "loader": e, "transpose": e}}, ...]}

    ``scales`` and ``zero_points`` are keyed by fused tensor name; ``signed``
    names the tensors quantized to signed 8 bit (it sets a Softmax output's
    data type). The compiled input slot order is a per-build permutation, so
    the caller chooses it (``inputs``). ``consts`` lists an op's initializer
    operands in its ``npu_params`` order; their bytes are copied from the
    standalone program, and ``values`` (optional) only checks that the
    standalone build used the same constant. ``place`` pins an op's engine
    sub-programs to engines (a LEARNED input: attention's placement cannot be
    derived); without it the matrix and transpose sub-programs keep their
    standalone engine and loaders alternate. ``output_param`` places the graph
    output's PARAM job before the first core job of that op, or (``"late"``)
    before the DEQUANT job; like the slot order it is not predicted, so the
    caller chooses it (default: where the last op's standalone program has
    it). ``gate_corpus`` adds standalone
    programs used only to infer the register order, the liveness gates and
    loader templates."""
    return _Stitcher(wiring, scales, zero_points, signed, gate_corpus).result()


def _transpose_job(launch: Mapping, amap: Mapping) -> list[tuple[int, int, int]]:
    """Records of the LEARNED main-engine transpose job (``MAIN_TRANSPOSE_JOB``)
    for the standalone copy-engine transpose whose launch state is ``launch``.
    The record order is the native fused attention build's."""
    job = MAIN_TRANSPOSE_JOB
    const = job["constants"]

    def got(r):
        return launch.get(r, (A1, 0))

    def moved(first, last, shift):
        res = []
        for r in range(first, last + 1, 0x10):
            verb, v = got(r)
            res.append(
                (verb, r + shift, amap[v] if verb == A1 and _is_scratch(v) else v)
            )
        return res

    first, last, shift = job["permutation"]
    out = [(A1, 0x0280, const[0x0280]), (A1, 0x0290, const[0x0290])]
    out += moved(*job["source"])
    out += [
        (A1, REG_CTL, job["select"]),
        (A1, 0x0C10, const[0x0C10]),
        (A1, 0x0C20, const[0x0C20]),
    ]
    out += [(A1, r + shift, got(r)[1]) for r in range(first, last + 1, 0x10)]
    out.append((A1, REG_CTL, 1))
    out += moved(*job["destination"])
    out += [
        (A1, REG_CTL, 0x100000),
        (A1, REG_JOBTYPE, got(job["job_type_from"])[1]),
        (A3, 0, 0),
        (A1, REG_CTL, 0x1000000),
        (A9, 0, 0),
    ]
    return out


# ---- the MCode blob as a FlatBuffer ------------------------------------------------
# The blob is an ordinary FlatBuffer written by a back-to-front builder
# (objects created first sit at the highest addresses, vtables are shared,
# trailing absent fields trimmed). Creation order (= descending address):
#   X1 {u8}, X2 {u8}                                    root fields 0, 1
#   inputs: per graph input (slot order) desc(name, bytes), desc(name_offset, 4);
#           per output desc(name_offset, 4); desc("params", len(npu_params))
#   outputs: desc(name, bytes)
#   empty vector, name string, json string, IO table
#   segment tables 0..4 {f0 u8 = 1, f1 u8 kind, f2 u32 words, f3 u32 start word,
#                        f4 u8 compressed, f5 u32 stream bytes}, vector (4..0)
#   u64 vector holding the segment streams (each zero padded to 32 bytes)
#   table {segment vector, stream vector}
#   symbols {name, index}: _ocm_base, per IO (name, name_offset), params; vector
#   table {u32, u32, symbols}; root table; finish
SEGMENT_PAD = 32


class _FlatBuilder:
    """The part of a back-to-front FlatBuffer builder this blob needs. Offsets
    are distances from the end of the buffer."""

    def __init__(self):
        self.buf = bytearray()
        self.minalign = 1
        self.vtables: dict[tuple, int] = {}
        self.slots: list[int] = []
        self.object_end = 0

    def _prep(self, size: int, extra: int) -> None:
        self.minalign = max(self.minalign, size)
        self.buf[:0] = bytes(-(len(self.buf) + extra) % size)

    def _put(self, fmt: str, v: int) -> None:
        self._prep(struct.calcsize(fmt), 0)
        self.buf[:0] = struct.pack("<" + fmt, v)

    def _put_ref(self, off: int) -> None:
        self._prep(4, 0)
        self.buf[:0] = struct.pack("<I", len(self.buf) - off + 4)

    def string(self, s: str) -> int:
        data = s.encode()
        self._prep(4, len(data) + 1)
        self.buf[:0] = data + b"\0"
        self.buf[:0] = struct.pack("<I", len(data))
        return len(self.buf)

    def ref_vector(self, offsets: Sequence[int]) -> int:
        self._prep(4, 4 * len(offsets))
        for off in reversed(offsets):
            self._put_ref(off)
        self.buf[:0] = struct.pack("<I", len(offsets))
        return len(self.buf)

    def u64_vector(self, data: bytes) -> int:
        self._prep(4, len(data))
        self._prep(8, len(data))
        self.buf[:0] = data
        self.buf[:0] = struct.pack("<I", len(data) // 8)
        return len(self.buf)

    def start(self, n_fields: int) -> None:
        self.slots = [0] * n_fields
        self.object_end = len(self.buf)

    def scalar(self, slot: int, fmt: str, v: int) -> None:
        if v:  # a default (0) field is left out
            self._put(fmt, v)
            self.slots[slot] = len(self.buf)

    def ref(self, slot: int, off: int) -> None:
        self._put_ref(off)
        self.slots[slot] = len(self.buf)

    def end(self) -> int:
        self._put("i", 0)
        obj = len(self.buf)
        fields = [obj - s if s else 0 for s in self.slots]
        while fields and not fields[-1]:
            fields.pop()
        key = (tuple(fields), obj - self.object_end)
        vt = self.vtables.get(key)
        if vt is None:
            for f in reversed(fields):
                self._put("H", f)
            self._put("H", obj - self.object_end)
            self._put("H", 2 * (len(fields) + 2))
            vt = self.vtables[key] = len(self.buf)
        struct.pack_into("<i", self.buf, len(self.buf) - obj, vt - obj)
        return obj

    def finish(self, root: int) -> bytes:
        self._prep(self.minalign, 4)
        self._put_ref(root)
        return bytes(self.buf)


def _u(fmt, mc, o):
    return struct.unpack_from(fmt, mc, o)[0]


def _fb_fields(mc, pos):
    vt = pos - _u("<i", mc, pos)
    n = (_u("<H", mc, vt) - 4) // 2
    return [(_u("<H", mc, vt + 4 + 2 * i) or None) for i in range(n)]


def _fb_scalar(mc, pos, i, fmt):
    f = _fb_fields(mc, pos)
    return _u(fmt, mc, pos + f[i]) if i < len(f) and f[i] is not None else 0


def _fb_ref(mc, pos, i):
    f = _fb_fields(mc, pos)
    if i >= len(f) or f[i] is None:
        raise ValueError(f"MCode blob: table at {pos} has no field {i}")
    o = pos + f[i]
    return o + _u("<I", mc, o)


def _fb_vector(mc, pos):
    n = _u("<I", mc, pos)
    return [pos + 4 + 4 * k + _u("<I", mc, pos + 4 + 4 * k) for k in range(n)]


def _fb_string(mc, pos):
    return mc[pos + 4 : pos + 4 + _u("<I", mc, pos)].decode()


def parse_blob(mc: bytes) -> dict:
    """The fields of an MCode blob; ``build_blob`` inverts it byte for byte."""
    root = _u("<I", mc, 0)
    if len(_fb_fields(mc, root)) != 10:
        raise ValueError("MCode blob: the root table does not have 10 fields")
    io, exe, sym = _fb_ref(mc, root, 2), _fb_ref(mc, root, 5), _fb_ref(mc, root, 6)

    def desc(p):
        return (_fb_string(mc, _fb_ref(mc, p, 0)), _fb_scalar(mc, p, 2, "<Q"))

    segs = [
        dict(
            kind=_fb_scalar(mc, p, 1, "<B"),
            words=_fb_scalar(mc, p, 2, "<I"),
            start=_fb_scalar(mc, p, 3, "<I"),
            comp=_fb_scalar(mc, p, 4, "<B"),
            nbytes=_fb_scalar(mc, p, 5, "<I"),
        )
        for p in reversed(_fb_vector(mc, _fb_ref(mc, exe, 0)))
    ]
    data = _fb_ref(mc, exe, 1)
    body = mc[data + 4 : data + 4 + 8 * _u("<I", mc, data)]
    streams = []
    for s in segs:
        raw = body[8 * s["start"] : 8 * (s["start"] + s["words"])]
        streams.append(raw[: s["nbytes"]] if s["comp"] else raw)
    return dict(
        x1=_fb_scalar(mc, _fb_ref(mc, root, 0), 0, "<B"),
        x2=_fb_scalar(mc, _fb_ref(mc, root, 1), 0, "<B"),
        inputs=[desc(p) for p in _fb_vector(mc, _fb_ref(mc, io, 0))],
        outputs=[desc(p) for p in _fb_vector(mc, _fb_ref(mc, io, 1))],
        n_extra=len(_fb_vector(mc, _fb_ref(mc, io, 2))),
        name=_fb_string(mc, _fb_ref(mc, root, 8)),
        json=_fb_string(mc, _fb_ref(mc, root, 9)),
        kinds=[s["kind"] for s in segs],
        comp=[bool(s["comp"]) for s in segs],
        streams=streams,
        g0=_fb_scalar(mc, sym, 0, "<I"),
        g1=_fb_scalar(mc, sym, 1, "<I"),
        syms=[
            (_fb_string(mc, _fb_ref(mc, p, 0)), _fb_scalar(mc, p, 1, "<i"))
            for p in _fb_vector(mc, _fb_ref(mc, sym, 2))
        ],
    )


def build_blob(d: Mapping) -> bytes:
    """An MCode blob from ``parse_blob``-shaped fields."""
    if d["n_extra"]:
        raise NotImplementedError("MCode blob with a non-empty third IO vector")
    b = _FlatBuilder()

    def byte_table(v):
        b.start(1)
        b.scalar(0, "B", v)
        return b.end()

    def desc(name, size):
        s = b.string(name)
        b.start(3)
        b.ref(0, s)
        b.scalar(2, "Q", size)
        return b.end()

    x1, x2 = byte_table(d["x1"]), byte_table(d["x2"])
    ins = b.ref_vector([desc(n, s) for n, s in d["inputs"]])
    outs = b.ref_vector([desc(n, s) for n, s in d["outputs"]])
    extra = b.ref_vector([])
    name, js = b.string(d["name"]), b.string(d["json"])
    b.start(3)
    b.ref(0, ins)
    b.ref(1, outs)
    b.ref(2, extra)
    io = b.end()
    body, tabs = bytearray(), []
    for kind, comp, stream in zip(d["kinds"], d["comp"], d["streams"]):
        room = -(-len(stream) // SEGMENT_PAD) * SEGMENT_PAD
        start = len(body) // 8
        body += stream + bytes(room - len(stream))
        b.start(6)
        b.scalar(0, "B", 1)
        b.scalar(1, "B", kind)
        b.scalar(2, "I", room // 8)
        b.scalar(3, "I", start)
        if comp:
            b.scalar(4, "B", 1)
            b.scalar(5, "I", len(stream))
        tabs.append(b.end())
    segv = b.ref_vector(list(reversed(tabs)))
    datav = b.u64_vector(bytes(body))
    b.start(2)
    b.ref(0, segv)
    b.ref(1, datav)
    exe = b.end()
    syms = []
    for nm, idx in d["syms"]:
        s = b.string(nm)
        b.start(2)
        b.ref(0, s)
        b.scalar(1, "i", idx)
        syms.append(b.end())
    symv = b.ref_vector(syms)
    b.start(3)
    b.scalar(0, "I", d["g0"])
    b.scalar(1, "I", d["g1"])
    b.ref(2, symv)
    sym = b.end()
    b.start(10)
    b.ref(0, x1)
    b.ref(1, x2)
    b.ref(2, io)
    b.ref(5, exe)
    b.ref(6, sym)
    b.ref(8, name)
    b.ref(9, js)
    return b.finish(b.end())


def io_tables(inputs, outputs, n_params: int):
    """``(input descriptors, output descriptors, symbols)`` of a blob from its
    IO list: ``[(name, byte size)]`` in slot order."""
    ins = []
    for n, s in inputs:
        ins += [(n, s), (n + "_offset", 4)]
    ins += [(n + "_offset", 4) for n, _ in outputs] + [("params", n_params)]
    names = ["_ocm_base"]
    for n, _ in [*inputs, *outputs]:
        names += [n, n + "_offset"]
    names.append("params")
    return ins, list(outputs), [(n, i) for i, n in enumerate(names)]


def _graph_io(st: Stitched) -> dict[str, tuple]:
    """Fused graph IO tensor -> ``(byte size, ValueInfo, outputs_info entry)``,
    from the standalone program that quantizes or dequantizes the tensor."""
    wiring, found = st.wiring, {}
    for o, prog in zip(wiring["ops"], st.programs):
        d = parse_blob(prog.mc)
        sizes = dict(d["inputs"] + d["outputs"])
        g = prog.model.graph
        infos = {v.name: v for v in [*g.input, *g.output]}
        info = json.loads(
            next(a.s for a in g.node[0].attribute if a.name == "outputs_info")
        )
        for sname, fname in [*o["in"].items(), o["out"]]:
            if fname in wiring["inputs"] and sname in prog.in_names:
                found.setdefault(fname, (sizes[sname], infos[sname], None))
            if fname == wiring["output"] and sname in prog.out_names:
                found[fname] = (sizes[sname], infos[sname], info[sname])
    missing = sorted(set(wiring["inputs"] + [wiring["output"]]) - set(found))
    if missing:
        raise ValueError(f"no standalone program quantizes graph tensors {missing}")
    return found


def stitched_blob(st: Stitched) -> bytes:
    """The fused MCode blob of ``stitch``'s result."""
    wiring, d0 = st.wiring, parse_blob(st.programs[0].mc)
    found = _graph_io(st)
    out = wiring["output"]
    ins, outs, syms = io_tables(
        [(n, found[n][0]) for n in wiring["inputs"]],
        [(out, found[out][0])],
        len(st.params),
    )
    comp = compress_shape([records(r) for r in st.segments])
    streams = [suc.encode(r) if c else r for r, c in zip(st.segments, comp)]
    mc = build_blob(
        dict(d0, inputs=ins, outputs=outs, syms=syms, comp=comp, streams=streams)
    )
    if suc.decode_segments(mc) != st.segments:
        raise ValueError("the rebuilt MCode blob does not decode to its segments")
    return mc


def _varint(n: int) -> bytes:
    out = b""
    while True:
        out += bytes([(n & 0x7F) | (0x80 if n > 0x7F else 0)])
        n >>= 7
        if not n:
            return out


def _extra_data(blob: bytes) -> tuple[list[str], bytes]:
    """``extra_data`` metadata = repeated field 1 {1: tensor name} (graph
    inputs, then outputs) and a constant tail."""
    names, i = [], 0
    while i < len(blob) and blob[i] == 0x0A:
        n = blob[i + 1]
        if blob[i + 2] != 0x0A or blob[i + 3] != n - 2:
            raise ValueError("unexpected extra_data layout")
        names.append(blob[i + 4 : i + 2 + n].decode())
        i += 2 + n
    return names, blob[i:]


def stitched_model(st: Stitched) -> onnx.ModelProto:
    """The complete fused model of ``stitch``'s result. Every field is copied
    from, or shaped like, the standalone models: node inputs in slot order,
    graph inputs in the source graph's order, value_info = ``npu_params``,
    ``npu_dyn_params``, inputs (slot order), the MCode blob, the output."""
    wiring, mc, found = st.wiring, stitched_blob(st), _graph_io(st)
    out = wiring["output"]
    m = onnx.ModelProto()
    m.CopyFrom(st.programs[0].model)
    g = m.graph
    node = g.node[0]
    del node.input[:]
    node.input.extend(wiring["inputs"])
    del node.output[:]
    node.output.append(out)

    def vi(name):
        v = onnx.ValueInfoProto()
        v.CopyFrom(found[name][1])
        v.name = name
        return v

    for a in node.attribute:
        if a.name == "outputs_info":
            a.s = json.dumps({out: found[out][2]}).encode()
    meta = next(p for p in m.metadata_props if p.key == "extra_data")
    names0, tail = _extra_data(base64.b64decode(meta.value))
    order = wiring.get("graph_inputs")
    if order is None:  # all inputs enter at the first op: its own input order
        inv = {s: f for s, f in wiring["ops"][0]["in"].items()}
        order = [inv[n] for n in names0[:-1]]
    if sorted(order) != sorted(wiring["inputs"]):
        raise ValueError(
            f"graph_inputs {list(order)} are not the graph inputs {wiring['inputs']}"
        )
    del g.input[:]
    g.input.extend(vi(n) for n in order)
    del g.output[:]
    g.output.append(vi(out))
    neu = mre.mcode_initializer(m).name
    data = {"npu_params": st.params, "npu_dyn_params": b"", neu: mc}
    old_vi = {v.name: v for v in g.value_info}
    for init in g.initializer:
        init.raw_data = data[init.name]
        del init.dims[:]
        init.dims.append(len(data[init.name]))
        old_vi[init.name].type.tensor_type.shape.dim[0].dim_value = len(data[init.name])
    new_vi = [
        old_vi["npu_params"],
        old_vi["npu_dyn_params"],
        *[vi(n) for n in wiring["inputs"]],
        old_vi[neu],
        vi(out),
    ]
    del g.value_info[:]
    g.value_info.extend(new_vi)
    entries = b"".join(
        b"\x0a"
        + _varint(len(n.encode()) + 2)
        + b"\x0a"
        + _varint(len(n.encode()))
        + n.encode()
        for n in [*order, out]
    )
    meta.value = base64.b64encode(entries + tail).decode()
    return m


def stitch_model(
    wiring: Mapping,
    scales: Mapping[str, float],
    zero_points: Mapping[str, int],
    signed: Iterable[str] = (),
    gate_corpus: Iterable = (),
) -> onnx.ModelProto:
    """``stitch`` and wrap the result into a compiled model."""
    return stitched_model(stitch(wiring, scales, zero_points, signed, gate_corpus))


def compare_models(got: onnx.ModelProto, want: onnx.ModelProto) -> list[str]:
    """Differences between two compiled models outside segment 0's slot table
    (compared as a set of records): blob fields, header, tail, stream layout,
    every decompressed record, and the model proto outside the MCode bytes
    (``npu_params`` included). An empty list means equal."""
    mc, omc = (bytes(mre.mcode_initializer(m).raw_data) for m in (got, want))
    diffs = []
    if len(mc) != len(omc):
        diffs.append(f"blob length {len(mc)} vs {len(omc)}")
    da, db = parse_blob(mc), parse_blob(omc)
    for k in da:
        if k != "streams" and da[k] != db[k]:
            diffs.append(f"blob field {k}: {da[k]} vs {db[k]}")
    la, lb = suc.segment_streams(mc), suc.segment_streams(omc)
    if [(a, b, c) for a, b, _, c in la] != [(a, b, c) for a, b, _, c in lb]:
        diffs.append("segment stream layout")
    if mc[: la[0][0]] != omc[: lb[0][0]]:
        diffs.append("blob header bytes")
    ea, eb = (s[-1][0] + 8 * s[-1][2][2] for s in (la, lb))
    if mc[ea:] != omc[eb:]:
        diffs.append("blob tail bytes")
    segs = zip(suc.decode_segments(mc), suc.decode_segments(omc))
    for si, (ra, rb) in enumerate(segs):
        ra, rb = records(ra), records(rb)
        if len(ra) != len(rb):
            diffs.append(f"segment {si}: {len(ra)} records vs {len(rb)}")
            continue
        lo, hi = slot_table(rb) if si == 0 else (0, 0)
        if sorted(ra[lo:hi]) != sorted(rb[lo:hi]):
            diffs.append("segment 0 slot table: different record set")
        diffs += [
            f"segment {si} record {i}: {x.hex()} vs {y.hex()}"
            for i, (x, y) in enumerate(zip(ra, rb))
            if x != y and not lo <= i < hi
        ]
    a, b = onnx.ModelProto(), onnx.ModelProto()
    a.CopyFrom(got)
    b.CopyFrom(want)
    for m in (a, b):
        mre.mcode_initializer(m).raw_data = b""
    if a.SerializeToString(deterministic=True) != b.SerializeToString(
        deterministic=True
    ):
        diffs.append("model proto outside the MCode bytes")
    return diffs


# ---- committed graphs ------------------------------------------------------------
@lru_cache(maxsize=None)
def load_index(path: str = FIXTURE_INDEX) -> dict:
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=None)
def fixture_program(component: str, calibration: str) -> Program:
    """A committed standalone program (``fixtures/graph_stitch``)."""
    files = load_index()["components"][component]["files"]
    if calibration not in files:
        raise ValueError(
            f"{component} has no {calibration} build; have {sorted(files)}"
        )
    return load_program(os.path.join(FIXTURES, files[calibration]))


def fixture_case(graph: str, calibration: str, sources: str | None = None):
    """``(wiring, scales, zero_points, signed, native fused model)`` for a
    committed graph at ``calibration``. ``sources`` picks the calibration the
    standalone programs were built with (default: the same one)."""
    index = load_index()
    if graph not in index["graphs"]:
        raise ValueError(f"no graph {graph!r}; have {sorted(index['graphs'])}")
    g = index["graphs"][graph]
    cal = g["calibrations"][calibration]
    values = g.get("constants", {})
    ops = []
    for o in g["ops"]:
        o = dict(o, program=fixture_program(o["component"], sources or calibration))
        o["consts"] = [
            dict(c, values=values.get(c["name"])) for c in o.get("consts", [])
        ]
        ops.append(o)
    wiring = {
        "inputs": cal["inputs"],
        "graph_inputs": g["graph_inputs"],
        "output": g["output"],
        "ops": ops,
    }
    oracle = load_model(os.path.join(FIXTURES, cal["oracle"]))
    return wiring, cal["scales"], cal["zero_points"], cal["signed"], oracle


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("graph", nargs="*", help="committed graphs (default: all)")
    args = p.parse_args(argv)
    bad = 0
    for graph in args.graph or sorted(load_index()["graphs"]):
        for cal in sorted(load_index()["graphs"][graph]["calibrations"]):
            wiring, sc, zp, signed, oracle = fixture_case(graph, cal)
            st = stitch(wiring, sc, zp, signed)
            diffs = compare_models(stitched_model(st), oracle)
            counts = [len(s) // REC for s in st.segments]
            print(f"{graph} {cal}: records {counts}, differences {len(diffs)}")
            bad += bool(diffs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
