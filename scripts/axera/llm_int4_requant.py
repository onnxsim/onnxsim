#!/usr/bin/env python3
"""Re-quantize the int4 weights of a compiled ``pulsar2 llm_build --weight_type s4``
model in place, without recompiling.

A compiled layer file (``{prefix}_p{N}_l{i}_together.axmodel``) keeps its seven
Linear weights (q/k/v/o/gate/up/down) as int4 codes inside the ``npu_params``
initializer. This module finds those bytes, decodes them, computes better codes
in the *same* format (one float32 scale per output row, 4-bit codes), and writes
a copy of the directory in which only the weight-block bytes differ. The ``post``
model is s8 whatever ``--weight_type`` says and is copied unchanged. See
``docs/axera-llm-int4-requant.md`` for the byte map, the evaluation and what was
and was not run on a device.

Layout (generalises ``llm_build_dtype_analysis``'s 32-row s4 block)
-------------------------------------------------------------------
A Linear ``[rows, cin]`` is stored as ``rows / RB`` *row blocks*, ``RB`` = 32 or
64 (64 was seen for SmolLM2-135M's q/k/v). A row block is split by columns into
*column parts* (``llm_build_dtype_analysis.column_blocks``: the first part holds
up to 576 columns, every later one 544). One column part of width ``w``, with
``ch = ceil(w / 36)`` chunks, is::

    body   RB * 18 * ch bytes   nibble codes ``q + 8``; byte = (code[2j+1] << 4) | code[2j];
                                columns from w up to 36 * ch hold 8 (q = 0)
    sums   RB int32             Q16.16 of ``-sum(q) / 2`` over this part's columns
    zeros  4 * RB bytes
    scales RB float32           the row's scale, repeated in every part of the row

The body is a sequence of ``36 * RB / 32``-byte units. Unit ``(k, c)`` (row pair
``k`` in 0..15, chunk ``c``) holds chunk ``c`` of rows ``2k``, ``2k + 1`` and, for
``RB`` = 64, ``2k + 32``, ``2k + 33``; 18 bytes each.

Quantizers
----------
``plain``   what the compiler does. Two rules were observed, and which one a
            build uses is read from its own codes (``detect_rule``):
            A: ``s = -w[argmax|w|] / 8``, ``q = floor(w / s + 0.5)``, codes in [-8, 7];
            B: ``s = max|w| / 7``, ``q = rint(w / s)``, codes in [-7, 7].
``gptq``    GPTQ/OBS column-sequential rounding with error compensation against
            ``H = X^T X`` of the Linear's input, for several per-row scale
            candidates, then an H-weighted closed-form refit of the row scale.
``gptqa``   ``gptq`` of a re-targeted weight ``W'`` that maps the *quantized*
            path's input to the *float* path's output (``retarget``, ridge
            parameter ``ridge`` times the mean diagonal of ``H``). It fits one
            regression per Linear on the calibration tokens, so it can overfit a
            small calibration set; ``gptq`` is the conservative choice.
``identity`` writes the decoded codes back; the output must equal the input
            byte for byte (a self-check of the encoder and the layout).

Calibration is layer-sequential and sequential inside a layer (q/k/v, then o,
then gate/up, then down): every ``H`` is collected through the numpy reference
(``llm_reference``) with the already-quantized Linears in place and bf16 rounding
of the hidden state at layer boundaries.

Usage::

    python llm_int4_requant.py --src DIR --checkpoint DIR --dst DIR --method gptq \\
        --calibration FILE_OR_TEXT [FILE_OR_TEXT ...]

Each ``--calibration`` item is a ``.json`` file of token-id lists (``[[1, 2], ...]``
or ``[{"ids": [1, 2]}, ...]``), a text file (tokenized with the checkpoint's
``tokenizer.json`` and cut into ``--seq-len`` token sequences), or literal text.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import llm_build_dtype_analysis as lda  # noqa: E402
import llm_layer_loop as loop  # noqa: E402
import llm_reference as ref  # noqa: E402

# Short key -> checkpoint tensor stem, in the order the tensors appear in npu_params.
PROJECTIONS = {
    "v": "self_attn.v_proj",
    "k": "self_attn.k_proj",
    "q": "self_attn.q_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
ROW_BLOCKS = (32, 64)  # rows per row block seen in compiled files
CODE_RANGE = {"A": (-8, 7), "B": (-7, 7)}
METHODS = ("plain", "gptq", "gptqa", "identity")
SCALE_CANDIDATES = (1.0, 0.85, 0.72, 0.61, 0.52, 0.44, 0.37)
PARAMS = "npu_params"


class LayoutError(ValueError):
    """A file, block or checkpoint does not fit the s4 layout described above."""


# --------------------------------------------------------------------------
# Row blocks: encode / decode
# --------------------------------------------------------------------------
column_blocks = lda.column_blocks


def _chunks(width: int) -> int:
    return -(-width // lda.CHUNK_COLS)


def _body_bytes(width: int, row_block: int) -> int:
    return row_block * lda.CHUNK * _chunks(width)


def part_bytes(width: int, row_block: int) -> int:
    """Bytes of one column part: the nibble body and the 12-bytes-per-row tail."""
    return _body_bytes(width, row_block) + 12 * row_block


def block_bytes(cin: int, row_block: int) -> int:
    """Bytes of one row block of a Linear with ``cin`` inputs."""
    return sum(part_bytes(w, row_block) for w in column_blocks(cin))


def _check_row_block(row_block: int) -> None:
    if row_block not in ROW_BLOCKS:
        raise LayoutError(
            f"row blocks of {row_block} rows: only {ROW_BLOCKS} are known"
        )


@functools.lru_cache(maxsize=None)
def part_offsets(width: int, row_block: int) -> tuple[np.ndarray, np.ndarray]:
    """``(byte offset, shift)``, each ``[row_block, 36 * chunks]``, of every code in one
    column part's body. For ``row_block`` = 32 this is ``lda.s4_offset``."""
    _check_row_block(row_block)
    ch = _chunks(width)
    unit = lda.CHUNK_COLS * (row_block // 32)
    r = np.arange(row_block)[:, None]
    c = np.arange(lda.CHUNK_COLS * ch)[None, :]
    off = (
        unit * ch * ((r % 32) // 2)  # the row pair's units, one per chunk
        + unit * (c // lda.CHUNK_COLS)  # the chunk's unit
        + lda.CHUNK_COLS * (r // 32)  # rows 32..63 take the unit's second half
        + lda.CHUNK * (r % 2)
        + (c % lda.CHUNK_COLS) // 2
    )
    return off, np.broadcast_to(4 * (c % 2), off.shape).copy()


def row_sums_q16(q: np.ndarray) -> np.ndarray:
    """Per-row ``-sum(q) / 2`` as int32 Q16.16."""
    return (-q.sum(axis=1) * (1 << 15)).astype("<i4")


def _encode_part(q: np.ndarray, scale: np.ndarray, row_block: int) -> bytes:
    width = q.shape[1]
    off, sh = part_offsets(width, row_block)
    code = np.full(off.shape, 8, np.int64)  # pad columns hold the zero point
    code[:, :width] += q
    body = np.zeros(_body_bytes(width, row_block), np.int64)
    np.add.at(body, off.ravel(), (code << sh).ravel())
    return (
        body.astype(np.uint8).tobytes()
        + row_sums_q16(q).tobytes()
        + bytes(4 * row_block)
        + scale.astype("<f4").tobytes()
    )


def encode_block(
    q: np.ndarray, scale: np.ndarray, row_block: Optional[int] = None
) -> bytes:
    """The bytes of one row block: signed codes ``q`` ``[row_block, cin]`` in [-8, 7]
    and one float32 scale per row."""
    q = np.asarray(q)
    scale = np.asarray(scale, np.float32)
    row_block = len(q) if row_block is None else row_block
    _check_row_block(row_block)
    if q.ndim != 2 or q.shape[0] != row_block or scale.shape != (row_block,):
        raise ValueError(
            f"codes {q.shape} / scales {scale.shape} for a {row_block}-row block"
        )
    if not np.issubdtype(q.dtype, np.integer) or q.min() < -8 or q.max() > 7:
        raise ValueError("codes must be integers in [-8, 7]")
    if not np.isfinite(scale).all() or (scale == 0).any():
        raise ValueError("scales must be finite and non-zero")
    q = q.astype(np.int64)
    out, start = [], 0
    for width in column_blocks(q.shape[1]):
        out.append(_encode_part(q[:, start : start + width], scale, row_block))
        start += width
    return b"".join(out)


def _decode(data: bytes, cin: int, row_block: int):
    """-> (q [RB, cin], scale [parts, RB], rowsum [parts, RB] int32 Q16.16, problem or None)."""
    _check_row_block(row_block)
    if len(data) != block_bytes(cin, row_block):
        raise LayoutError(
            f"{len(data)} bytes for a {row_block}-row block of {cin} columns "
            f"({block_bytes(cin, row_block)} expected)"
        )
    b = np.frombuffer(data, np.uint8).astype(np.int64)
    qs, scales, sums, pos, problem = [], [], [], 0, None
    for width in column_blocks(cin):
        off, sh = part_offsets(width, row_block)
        code = (b[off + pos] >> sh) & 15
        q = code[:, :width] - 8
        tail = pos + _body_bytes(width, row_block)
        rowsum = np.frombuffer(data, "<i4", row_block, tail)
        scale = np.frombuffer(data, "<f4", row_block, tail + 8 * row_block)
        if problem is None:
            if b[tail + 4 * row_block : tail + 8 * row_block].any():
                problem = "non-zero bytes between the row sums and the scales"
            elif (code[:, width:] != 8).any():
                problem = "a pad nibble is not 8"
            elif not np.array_equal(rowsum, row_sums_q16(q)):
                problem = "a row sum is not -sum(q)/2 of its column part"
            elif not np.isfinite(scale).all() or (scale == 0).any():
                problem = "a scale is zero or not finite"
            elif scales and not np.array_equal(scale, scales[0]):
                problem = "the column parts of a row disagree on its scale"
        qs.append(q)
        scales.append(scale.copy())
        sums.append(rowsum.copy())
        pos += part_bytes(width, row_block)
    return np.concatenate(qs, axis=1), np.stack(scales), np.stack(sums), problem


def block_problem(data: bytes, cin: int, row_block: int) -> Optional[str]:
    """Why ``data`` is not a well-formed row block, or None when it is: the zero bytes,
    the pad nibbles, the row sums and the per-part copies of the scales all check."""
    return _decode(data, cin, row_block)[3]


def decode_block(data: bytes, cin: int, row_block: int) -> dict:
    """Inverse of ``encode_block``: ``q`` (signed, ``[row_block, cin]``), ``scale``
    (``[row_block]``) and ``rowsum`` (``-sum(q)/2`` per column part, ``[row_block,
    parts]``, like ``lda.decode_block``). Raises ``LayoutError`` when the block is not
    well formed."""
    q, scales, sums, problem = _decode(data, cin, row_block)
    if problem:
        raise LayoutError(f"not an s4 row block ({row_block} rows x {cin}): {problem}")
    return {"q": q, "scale": scales[0], "rowsum": sums.T / 65536.0}


# --------------------------------------------------------------------------
# Layout discovery
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TensorLayout:
    """Where one Linear's row blocks are inside ``npu_params``."""

    key: str
    rows: int
    cin: int
    row_block: int
    offsets: tuple  # byte offset of each row block, ascending

    @property
    def block_bytes(self) -> int:
        return block_bytes(self.cin, self.row_block)

    @property
    def ranges(self) -> list:
        return [(o, o + self.block_bytes) for o in self.offsets]


def projection_shapes(model: ref.Model) -> dict:
    """``{key: (rows, cin)}`` of the seven Linears, from the checkpoint's architecture."""
    return {
        "v": (model.kvdim, model.H),
        "k": (model.kvdim, model.H),
        "q": (model.qdim, model.H),
        "o": (model.H, model.qdim),
        "gate": (model.cfg["intermediate_size"], model.H),
        "up": (model.cfg["intermediate_size"], model.H),
        "down": (model.H, model.cfg["intermediate_size"]),
    }


def _locate(params: bytes, zeros: np.ndarray, cursor: int, rows, cin, row_block):
    """Offsets of ``rows / row_block`` well-formed row blocks at or after ``cursor``, in
    order, or None. A block normally starts where the previous one ended; when it does
    not (a norm weight or the RoPE table sits inside q and k), candidates are the
    offsets whose first tail has its ``4 * row_block`` zero bytes. ``zeros[i]`` counts
    the non-zero bytes before byte ``i``."""
    size = block_bytes(cin, row_block)
    gap = 4 * row_block
    tail0 = _body_bytes(column_blocks(cin)[0], row_block) + gap
    offsets = []
    for _ in range(rows // row_block):
        last = len(params) - size
        found = None
        if cursor <= last and not block_problem(
            params[cursor : cursor + size], cin, row_block
        ):
            found = cursor
        elif cursor < last:
            p = np.arange(cursor + 1, last + 1) + tail0
            # ... and whose scales are not zero bytes (each of them has a non-zero byte).
            fits = (zeros[p + gap] == zeros[p]) & (
                zeros[p + 2 * gap] - zeros[p + gap] >= row_block
            )
            for o in np.flatnonzero(fits) + cursor + 1:
                if not block_problem(params[o : o + size], cin, row_block):
                    found = int(o)
                    break
        if found is None:
            return None
        offsets.append(found)
        cursor = found + size
    return offsets


def discover_layout(params: bytes, shapes: dict) -> dict:
    """Locate every Linear's row blocks in a layer file's ``npu_params``.

    ``shapes`` is ``projection_shapes(model)``. Nothing is searched by value: the
    tensors are taken in the order the compiler writes them (``PROJECTIONS``), each as
    ``rows / RB`` row blocks whose zero bytes, pad nibbles and row sums check, with
    ``RB`` the candidate (32 or 64) whose first block comes first. Raises
    ``LayoutError`` when a tensor cannot be placed. Whether the codes belong to a given
    checkpoint is a separate check (``check_against_checkpoint``)."""
    arr = np.frombuffer(params, np.uint8)
    zeros = np.concatenate([[0], np.cumsum(arr != 0, dtype=np.int64)])
    layout, cursor = {}, 0
    for key in PROJECTIONS:
        rows, cin = shapes[key]
        best = None
        for row_block in ROW_BLOCKS:
            if rows % row_block:
                continue
            offsets = _locate(params, zeros, cursor, rows, cin, row_block)
            if offsets and (best is None or offsets[0] < best.offsets[0]):
                best = TensorLayout(key, rows, cin, row_block, tuple(offsets))
            if offsets and offsets[0] == cursor:
                break  # nothing can start earlier
        if best is None:
            raise LayoutError(
                f"{PROJECTIONS[key]} [{rows}, {cin}]: no run of well-formed s4 row blocks "
                f"at or after byte {cursor} of {len(params)} (not an s4 llm_build layer "
                "file of this checkpoint, or it stores deduplicated blocks)"
            )
        layout[key] = best
        cursor = best.ranges[-1][1]
    return layout


def decode_tensor(params: bytes, t: TensorLayout) -> tuple[np.ndarray, np.ndarray]:
    """``(q [rows, cin] int8, scale [rows] float32)`` of one located Linear."""
    blocks = [decode_block(params[a:b], t.cin, t.row_block) for a, b in t.ranges]
    return (
        np.concatenate([b["q"] for b in blocks]).astype(np.int8),
        np.concatenate([b["scale"] for b in blocks]),
    )


def encode_tensor(q: np.ndarray, scale: np.ndarray, t: TensorLayout) -> list:
    """One ``encode_block`` per row block of a Linear, in ``t.offsets`` order."""
    if q.shape != (t.rows, t.cin) or scale.shape != (t.rows,):
        raise ValueError(f"{t.key}: codes {q.shape} for a [{t.rows}, {t.cin}] Linear")
    rb = t.row_block
    return [
        encode_block(q[i : i + rb], scale[i : i + rb], rb) for i in range(0, t.rows, rb)
    ]


# --------------------------------------------------------------------------
# Quantizers
# --------------------------------------------------------------------------
def quantize_plain(w: np.ndarray, rule: str) -> tuple[np.ndarray, np.ndarray]:
    """The compiler's own per-row quantizer, ``(q int64, scale float32)``.

    A (tiny Llama, SmolLM2-135M builds): ``s = -w[argmax|w|] / 8``,
      ``q = floor(w / s + 0.5)``, [-8, 7]. The scale is negative when the row's
      peak is positive, so the peak is always code -8. On SmolLM2 this misses
      3,535 of 106,168,320 codes: all are exact .5 ties that the compiler rounded
      down, and its tie rule is not identified.
    B (Qwen3-0.6B build): ``s = max|w| / 7`` (positive), ``q = rint(w / s)`` (half
      to even), [-7, 7]. Matches every code of that build."""
    w = np.asarray(w, np.float32)
    if rule == "A":
        peak = w[np.arange(len(w)), np.argmax(np.abs(w), axis=1)]
        s = (-peak / np.float32(8)).astype(np.float32)
        q = np.floor(w / s[:, None] + np.float32(0.5))
    elif rule == "B":
        s = (np.abs(w).max(axis=1) / np.float32(7)).astype(np.float32)
        q = np.rint(w / s[:, None])
    else:
        raise ValueError(f"quantizer rule {rule!r}: 'A' or 'B'")
    return np.clip(q, *CODE_RANGE[rule]).astype(np.int64), s


def detect_rule(q: np.ndarray, scale: np.ndarray) -> str:
    """Which plain rule produced these codes, from the codes alone: under A every row
    holds code -8 (its peak); under B no code is -8, every scale is positive and every
    row's largest magnitude is 7."""
    q = np.asarray(q)
    if (q.min(axis=1) == -8).all():
        return "A"
    if (
        q.min() >= -7
        and (np.asarray(scale) > 0).all()
        and (np.abs(q).max(axis=1) == 7).all()
    ):
        return "B"
    raise LayoutError(
        "the codes fit neither observed llm_build rule (every row peaking at -8, or "
        "[-7, 7] with positive scales): not the compiler's own s4 codes"
    )


def dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) * scale[:, None]).astype(np.float32)


def h_error(w: np.ndarray, wq: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Per-row ``(w - wq) H (w - wq)^T``: the squared output error on the inputs ``H``
    was collected from."""
    d = w - wq
    return ((d @ h) * d).sum(axis=1)


def _base_scale(w: np.ndarray, rule: str) -> np.ndarray:
    if rule == "A":  # the build's sign convention: the peak sits on the negative side
        peak = w[np.arange(len(w)), np.argmax(np.abs(w), axis=1)]
        return (-peak / 8).astype(np.float64)
    return (np.abs(w).max(axis=1) / 7).astype(np.float64)


def _gptq_codes(w, u, s, lo, hi, block=128):
    """Column-sequential rounding of ``w`` (float64, ``[rows, cin]``) at per-row scale
    ``s``. ``u`` is the upper Cholesky factor of ``H^-1``; each column's rounding error
    is spread over the columns not yet rounded."""
    w = w.copy()
    rows, cin = w.shape
    q = np.zeros_like(w)
    for i1 in range(0, cin, block):
        i2 = min(i1 + block, cin)
        w1 = w[:, i1:i2].copy()
        err = np.zeros((rows, i2 - i1))
        u1 = u[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = w1[:, i]
            code = np.clip(np.rint(col / s), lo, hi)
            q[:, i1 + i] = code
            e = (col - code * s) / u1[i, i]
            w1[:, i:] -= e[:, None] * u1[i, i:][None, :]
            err[:, i] = e
        if i2 < cin:
            w[:, i2:] -= err @ u[i1:i2, i2:]
    return q


def gptq(
    w: np.ndarray,
    h: np.ndarray,
    rule: str,
    damping: float = 0.01,
    scale_candidates: Sequence[float] = SCALE_CANDIDATES,
) -> tuple[np.ndarray, np.ndarray]:
    """Activation-aware codes for ``w`` ``[rows, cin]`` given ``h = X^T X`` ``[cin, cin]``.

    For each candidate (a fraction of the plain scale) the row is rounded column by
    column with GPTQ error compensation (``damping`` times the mean diagonal added to
    ``H``), then its scale is refit in closed form, ``s = (q H w) / (q H q)``, keeping
    the plain scale's sign. Each row keeps the candidate with the lowest H-weighted
    error. Codes stay in the rule's range, so nothing appears that the compiled build
    does not already contain. Returns ``(q int8, scale float32)``."""
    lo, hi = CODE_RANGE[rule]
    w64 = w.astype(np.float64)
    h = h.astype(np.float64).copy()
    dead = np.diag(h) == 0  # inputs that were never non-zero
    h[dead, dead] = 1.0
    w64[:, dead] = 0.0
    hinv = np.linalg.inv(h + damping * np.mean(np.diag(h)) * np.eye(len(h)))
    u = np.linalg.cholesky((hinv + hinv.T) / 2).T  # H^-1 = u^T u
    s0 = _base_scale(w.astype(np.float64), rule)
    best_e = np.full(len(w64), np.inf)
    best_s = s0.copy()
    best_q = np.zeros_like(w64)
    for a in scale_candidates:
        s = s0 * a
        q = _gptq_codes(w64, u, s, lo, hi)
        qh = q @ h
        den = (qh * q).sum(axis=1)
        num = (qh * w64).sum(axis=1)
        s2 = np.where(den > 0, num / np.maximum(den, 1e-30), s)
        s2 = np.where(np.sign(s2) == np.sign(s0), s2, s)
        e = h_error(w64, q * s2[:, None], h)
        better = e < best_e
        best_e[better] = e[better]
        best_s[better] = s2[better]
        best_q[better] = q[better]
    return best_q.astype(np.int8), best_s.astype(np.float32)


def retarget(w: np.ndarray, c: np.ndarray, h: np.ndarray, ridge: float) -> np.ndarray:
    """``W'`` minimising ``|W x_float - W' x_quant|^2`` over the calibration tokens,
    pulled towards ``W`` by a ridge of ``ridge * mean(diag H)``:
    ``W' = W (C + lam I)(H + lam I)^-1`` with ``C = X_float^T X_quant`` and
    ``H = X_quant^T X_quant``. This is what ``gptqa`` quantizes instead of ``W``."""
    lam = ridge * np.mean(np.diag(h))
    eye = np.eye(len(h))
    return (
        w.astype(np.float64) @ (c + lam * eye) @ np.linalg.inv(h + lam * eye)
    ).astype(np.float32)


# --------------------------------------------------------------------------
# Calibration statistics, layer-sequential through the numpy reference
# --------------------------------------------------------------------------
def gram(xs: Sequence[np.ndarray]) -> np.ndarray:
    """``H = X^T X`` over the concatenated sequences (float32 product, float64 result)."""
    x = np.concatenate(xs).astype(np.float32)
    return (x.T @ x).astype(np.float64)


def cross_gram(xf: Sequence[np.ndarray], xq: Sequence[np.ndarray]) -> np.ndarray:
    """``C = X_float^T X_quant``."""
    a = np.concatenate(xf).astype(np.float32)
    b = np.concatenate(xq).astype(np.float32)
    return (a.T @ b).astype(np.float64)


class Calibration:
    """The calibration sequences' hidden states, carried through the model one layer at
    a time. The four methods return the inputs of the layer's Linears in the order they
    are quantized, each computed with whatever weights ``lw`` holds at that moment:
    ``norm1`` (input of q/k/v), ``attn`` (input of o), ``after_o`` (input of gate/up),
    ``act`` (input of down); ``finish`` advances to the next layer with bf16 rounding,
    which is what crosses a layer boundary in the compiled pipeline."""

    def __init__(self, model: ref.Model, token_ids: Sequence[Sequence[int]]):
        self.m = model
        self.x = [loop.bf16_round(model.embed(list(s))) for s in token_ids]
        self.tokens = sum(len(s) for s in token_ids)

    def norm1(self, lw):
        return [ref.rmsnorm(x, lw["n1"], self.m.eps) for x in self.x]

    def attn(self, lw, a_list):
        m, out = self.m, []
        g = m.nh // m.nkv
        for a in a_list:
            t = len(a)
            cos, sin = m.rope_cos_sin(np.arange(t))
            causal = np.triu(np.full((t, t), -np.inf, dtype=np.float32), 1)
            q, k, v = m._qkv(lw, a, cos[:, None], sin[:, None])
            sc = np.einsum("tkgd,skd->kgts", q.reshape(t, m.nkv, g, m.hd), k)
            p = ref.softmax(sc / np.float32(np.sqrt(m.hd)) + causal)
            att = np.einsum("kgts,skd->tkgd", p, v).reshape(t, m.qdim)
            out.append(att.astype(np.float32))
        return out

    def after_o(self, lw, att_list):
        x1 = [x + att @ lw["o"].T for x, att in zip(self.x, att_list)]
        return x1, [ref.rmsnorm(x, lw["n2"], self.m.eps) for x in x1]

    def act(self, lw, m_list):
        return [
            (ref.silu(mm @ lw["gate"].T) * (mm @ lw["up"].T)).astype(np.float32)
            for mm in m_list
        ]

    def finish(self, lw, x1_list, act_list):
        self.x = [
            loop.bf16_round((x1 + a @ lw["down"].T).astype(np.float32))
            for x1, a in zip(x1_list, act_list)
        ]


class SequentialQuantizer:
    """Codes for every layer of ``model``, one ``layer(li)`` call per layer, in order.

    ``plain`` needs no calibration. ``gptq`` / ``gptqa`` carry ``calibration_token_ids``
    through the layers quantized so far (and ``gptqa`` a second copy through the float
    layers, for its targets)."""

    def __init__(
        self,
        model: ref.Model,
        rule: str,
        method: str,
        calibration_token_ids: Optional[Sequence[Sequence[int]]] = None,
        ridge: float = 0.3,
        damping: float = 0.01,
        scale_candidates: Sequence[float] = SCALE_CANDIDATES,
    ):
        if method not in ("plain", "gptq", "gptqa"):
            raise ValueError(f"method {method!r}: one of plain, gptq, gptqa")
        if rule not in CODE_RANGE:
            raise ValueError(f"quantizer rule {rule!r}: 'A' or 'B'")
        self.model, self.rule, self.method = model, rule, method
        self.ridge, self.damping, self.candidates = (
            ridge,
            damping,
            tuple(scale_candidates),
        )
        self.next_layer = 0
        self.cal = self.cal_float = None
        if method != "plain":
            seqs = [list(map(int, s)) for s in (calibration_token_ids or []) if len(s)]
            if not seqs:
                raise ValueError(f"method {method!r} needs calibration token ids")
            if max(map(max, seqs)) >= model.vocab or min(map(min, seqs)) < 0:
                raise ValueError("a calibration token id is outside the vocabulary")
            self.cal = Calibration(model, seqs)
            if method == "gptqa":
                self.cal_float = Calibration(model, seqs)

    def layer(self, li: int, reference: Optional[dict] = None) -> tuple[dict, dict]:
        """``({key: (q int8, scale float32)}, stats)`` for layer ``li``. ``stats[key]`` is
        the H-weighted output error relative to the Linear's output energy, for the new
        codes and, when ``reference`` (e.g. the compiled file's own codes) is given, for
        those."""
        if li != self.next_layer:
            raise ValueError(
                f"layers must be quantized in order: expected {self.next_layer}"
            )
        self.next_layer += 1
        float_lw = self.model.layer_weights(li)
        fw = {k: np.asarray(float_lw[k], np.float32) for k in PROJECTIONS}
        if self.method == "plain":
            return {
                k: _as_codes(*quantize_plain(fw[k], self.rule)) for k in PROJECTIONS
            }, {}
        lw = dict(float_lw)  # the layer as quantized so far
        codes, stats = {}, {}

        def do(keys, h, c):
            for k in keys:
                target = fw[k] if c is None else retarget(fw[k], c, h, self.ridge)
                codes[k] = gptq(target, h, self.rule, self.damping, self.candidates)
                w64 = fw[k].astype(np.float64)
                energy = h_error(w64, 0 * w64, h).sum()
                stats[k] = {
                    self.method: float(
                        h_error(w64, dequantize(*codes[k]).astype(np.float64), h).sum()
                        / energy
                    )
                }
                if reference is not None:
                    was = dequantize(*reference[k]).astype(np.float64)
                    stats[k]["reference"] = float(h_error(w64, was, h).sum() / energy)
                lw[k] = dequantize(*codes[k])

        cal, calf = self.cal, self.cal_float
        af = attf = mf = actf = None
        if calf is not None:
            af = calf.norm1(float_lw)
            attf = calf.attn(float_lw, af)
            x1f, mf = calf.after_o(float_lw, attf)
            actf = calf.act(float_lw, mf)
            calf.finish(float_lw, x1f, actf)
        a = cal.norm1(lw)
        do(("q", "k", "v"), gram(a), None if calf is None else cross_gram(af, a))
        att = cal.attn(lw, a)
        do(("o",), gram(att), None if calf is None else cross_gram(attf, att))
        x1, mm = cal.after_o(lw, att)
        do(("gate", "up"), gram(mm), None if calf is None else cross_gram(mf, mm))
        act = cal.act(lw, mm)
        do(("down",), gram(act), None if calf is None else cross_gram(actf, act))
        cal.finish(lw, x1, act)
        return codes, stats


def _as_codes(q: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return q.astype(np.int8), scale.astype(np.float32)


# --------------------------------------------------------------------------
# Compiled files
# --------------------------------------------------------------------------
def read_axmodel(path: str) -> tuple[bytes, dict]:
    """``(file bytes, {initializer name: raw bytes})``."""
    import onnx

    with open(path, "rb") as f:
        raw = f.read()
    model = onnx.load_from_string(raw)
    return raw, {i.name: bytes(i.raw_data) for i in model.graph.initializer}


def _params_offset(raw: bytes, inits: dict, path: str) -> int:
    if PARAMS not in inits:
        raise LayoutError(f"{path}: no {PARAMS} initializer")
    at = raw.find(inits[PARAMS])
    if at < 0 or raw.find(inits[PARAMS], at + 1) >= 0:
        raise LayoutError(f"{path}: {PARAMS} is not one unique byte run of the file")
    return at


def check_against_checkpoint(
    codes: dict, weights: dict, where: str = ""
) -> tuple[str, int]:
    """The rule of a compiled layer's ``codes`` and the number of codes that differ from
    the plain quantizer of the checkpoint ``weights`` (rule A: exact ties only, see
    ``quantize_plain``). Raises ``LayoutError`` unless every scale equals the plain
    scale bit for bit, i.e. unless the file was compiled from this checkpoint and still
    holds the compiler's own codes."""
    rules = {k: detect_rule(*codes[k]) for k in PROJECTIONS}
    if len(set(rules.values())) != 1:
        raise LayoutError(
            f"{where}: the Linears do not share one quantizer rule: {rules}"
        )
    rule = rules["v"]
    mismatched = 0
    for k in PROJECTIONS:
        q, s = quantize_plain(weights[k], rule)
        if not np.array_equal(s, codes[k][1]):
            raise LayoutError(
                f"{where}: {PROJECTIONS[k]}'s stored scales are not the rule-{rule} scales "
                "of the checkpoint's weight (a different checkpoint?)"
            )
        mismatched += int((q != codes[k][0]).sum())
    return rule, mismatched


def read_layer_codes(path: str, model: ref.Model) -> dict:
    """``{key: (q, scale)}`` decoded from one compiled (or re-quantized) layer file."""
    _, inits = read_axmodel(path)
    if PARAMS not in inits:
        raise LayoutError(f"{path}: no {PARAMS} initializer")
    layout = discover_layout(inits[PARAMS], projection_shapes(model))
    return {k: decode_tensor(inits[PARAMS], layout[k]) for k in PROJECTIONS}


# --------------------------------------------------------------------------
# Emulator
# --------------------------------------------------------------------------
def s8_head(w: np.ndarray, rule: str, chunk_rows: int = 8192) -> np.ndarray:
    """``lm_head`` as the post model stores it: s8 with one scale per row (the post
    model of an s4 build is byte-identical to the s8 build's). Rule A is
    ``lda.quantize_int(w, 8)``; rule B is ``s = max|w| / 127``, ``q = rint(w / s)``."""
    out = np.empty(w.shape, np.float32)
    for i in range(0, len(w), chunk_rows):
        rows = np.asarray(w[i : i + chunk_rows], np.float32)
        if rule == "A":
            q, s = lda.quantize_int(rows, 8)
        else:
            s = (np.abs(rows).max(axis=1) / np.float32(127)).astype(np.float32)
            q = np.rint(rows / s[:, None])
        out[i : i + chunk_rows] = q.astype(np.float32) * s[:, None]
    return out


def emulator(model: ref.Model, layer_codes: Sequence[dict], rule: Optional[str] = None):
    """A numpy model whose Linears are the dequantized codes ``q * s``.

    ``layer_codes[li]`` is ``{key: (q, scale)}``. With ``rule`` the head is the s8
    ``lm_head`` of the post model, otherwise the float one. Run it with bf16 rounding
    at layer boundaries, ``emu.greedy(ids, n, rnd=llm_layer_loop.bf16_round)``, to model
    the compiled pipeline; arithmetic inside a layer stays float32, which the device's
    is not (see the doc for how closely the two agree). Holds every weight as float32
    in memory."""
    if len(layer_codes) != model.L:
        raise ValueError(
            f"{len(layer_codes)} layers of codes for a {model.L}-layer model"
        )
    stems = {f"{stem}.weight": key for key, stem in PROJECTIONS.items()}
    w = {}
    for name in model._w.keys():
        parts = name.split(".")
        key = (
            stems.get(".".join(parts[3:])) if name.startswith("model.layers.") else None
        )
        if key is None:
            w[name] = np.asarray(model.weight(name), np.float32)
        else:
            w[name] = dequantize(*layer_codes[int(parts[2])][key])
    if rule is not None:
        w["lm_head.weight"] = s8_head(model.weight(model._head), rule)
    emu = ref.Model(model.cfg, w)
    emu.eos_ids, emu.eos = model.eos_ids, model.eos
    return emu


def emulate_directory(model_dir: str, checkpoint_dir: str, rule: Optional[str] = None):
    """``emulator`` over the codes stored in a compiled or re-quantized directory.
    ``rule`` (for the s8 head) is detected from layer 0 when the directory holds the
    compiler's own codes; pass it for a re-quantized one."""
    model = ref.Model.load(checkpoint_dir)
    files = loop.list_model_dir(model_dir)
    codes = [read_layer_codes(p, model) for p in files.layer_paths(model_dir)]
    if rule is None:
        rule = detect_rule(*codes[0]["v"])
    return emulator(model, codes, rule)


# --------------------------------------------------------------------------
# The patcher
# --------------------------------------------------------------------------
def _verify_layer(src_raw, src_inits, path, base, layout, codes) -> dict:
    """Read ``path`` back and check it against the source file and the intended codes."""
    new_raw, new_inits = read_axmodel(path)  # still parses
    rewritable = np.zeros(len(src_raw), bool)
    for t in layout.values():
        for a, b in t.ranges:
            rewritable[base + a : base + b] = True
    same_size = len(new_raw) == len(src_raw)
    changed = outside = -1
    if same_size:
        diff = np.frombuffer(src_raw, np.uint8) != np.frombuffer(new_raw, np.uint8)
        changed, outside = int(diff.sum()), int((diff & ~rewritable).sum())
    others_same = set(new_inits) == set(src_inits) and all(
        new_inits[k] == src_inits[k] for k in src_inits if k != PARAMS
    )
    decode_ok = True
    try:
        for k, t in layout.items():
            q, s = decode_tensor(new_inits[PARAMS], t)  # zeros, pads, row sums
            decode_ok &= np.array_equal(q, codes[k][0]) and np.array_equal(
                s, codes[k][1]
            )
    except (LayoutError, KeyError):
        decode_ok = False
    return {
        "bytes_changed": changed,
        "bytes_rewritable": int(rewritable.sum()),
        "changed_outside_weight_blocks": outside,
        "other_initializers_identical": bool(others_same),
        "decode_matches_intended": bool(decode_ok),
        "ok": bool(same_size and outside == 0 and others_same and decode_ok),
    }


def requantize_directory(
    src_dir: str,
    checkpoint_dir: str,
    dst_dir: str,
    method: str = "gptq",
    calibration_token_ids: Optional[Sequence[Sequence[int]]] = None,
    *,
    ridge: float = 0.3,
    damping: float = 0.01,
    scale_candidates: Sequence[float] = SCALE_CANDIDATES,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Copy the compiled s4 directory ``src_dir`` to ``dst_dir`` (which must not exist
    or be empty) and rewrite each layer file's weight blocks with ``method``'s codes
    for ``checkpoint_dir``'s weights.

    Before a layer is touched its blocks are located and decoded, and the stored scales
    must be the plain-rule scales of the checkpoint. After a file is written it is read
    back: it must parse, have the same size, differ from the source only inside the
    weight blocks (so the MCode and every other constant are identical), and decode to
    exactly the intended codes and scales with consistent zero bytes, pad nibbles and
    row sums. Every other file, the s8 post model included, is copied unchanged.
    Raises ``LayoutError`` when the source does not fit and ``RuntimeError`` when a
    written file fails verification. Returns a report dict."""
    if method not in METHODS:
        raise ValueError(f"method {method!r}: one of {', '.join(METHODS)}")
    say = log or (lambda _line: None)
    model = ref.Model.load(checkpoint_dir)
    files = loop.list_model_dir(src_dir)
    if files.num_layers != model.L:
        raise LayoutError(
            f"{src_dir} has {files.num_layers} layer files, the checkpoint {model.L} layers"
        )
    biases = [n for n in model._w.keys() if n.endswith("_proj.bias")]
    if biases:
        raise ValueError(f"Linear biases are not supported (e.g. {biases[0]})")
    if os.path.realpath(src_dir) == os.path.realpath(dst_dir):
        raise ValueError("dst_dir is src_dir")
    if os.path.exists(dst_dir) and os.listdir(dst_dir):
        raise FileExistsError(f"{dst_dir} exists and is not empty")
    shapes = projection_shapes(model)
    layer_names = files.layer_names()

    # Layer 0 decides the rule; it is needed before any calibration starts.
    def load(li):
        path = os.path.join(src_dir, layer_names[li])
        raw, inits = read_axmodel(path)
        base = _params_offset(raw, inits, path)
        layout = discover_layout(inits[PARAMS], shapes)
        codes = {k: decode_tensor(inits[PARAMS], layout[k]) for k in PROJECTIONS}
        lw = model.layer_weights(li)
        rule, mismatched = check_against_checkpoint(
            codes, {k: lw[k] for k in PROJECTIONS}, layer_names[li]
        )
        return raw, inits, base, layout, codes, rule, mismatched

    first = load(0)
    rule = first[5]
    quantizer = None
    if method != "identity":
        quantizer = SequentialQuantizer(
            model, rule, method, calibration_token_ids, ridge, damping, scale_candidates
        )
    os.makedirs(dst_dir, exist_ok=True)
    report = {
        "method": method,
        "rule": rule,
        "calibration_tokens": 0
        if quantizer is None or quantizer.cal is None
        else quantizer.cal.tokens,
        "layers": [],
        "copied": [],
        "ok": True,
    }
    for name in sorted(os.listdir(src_dir)):
        if name not in layer_names:
            shutil.copy2(os.path.join(src_dir, name), os.path.join(dst_dir, name))
            report["copied"].append(name)
    for li, name in enumerate(layer_names):
        t0 = time.time()
        raw, inits, base, layout, src_codes, layer_rule, mismatched = (
            first if li == 0 else load(li)
        )
        first = None
        if layer_rule != rule:
            raise LayoutError(f"{name} uses rule {layer_rule}, layer 0 rule {rule}")
        if quantizer is None:
            codes, stats = src_codes, {}
        else:
            codes, stats = quantizer.layer(li, src_codes)
        lo, hi = CODE_RANGE[rule]
        buf = bytearray(raw)
        for k, t in layout.items():
            q, s = codes[k]
            if q.min() < lo or q.max() > hi:
                raise RuntimeError(f"{name} {k}: codes outside [{lo}, {hi}]")
            for (a, b), block in zip(t.ranges, encode_tensor(q, s, t)):
                assert len(block) == b - a
                buf[base + a : base + b] = block
        out = os.path.join(dst_dir, name)
        with open(out, "wb") as f:
            f.write(bytes(buf))
        entry = _verify_layer(raw, inits, out, base, layout, codes)
        entry.update(
            layer=li,
            file=name,
            row_blocks={k: t.row_block for k, t in layout.items()},
            source_codes_not_plain=mismatched,
            relative_output_error=stats,
        )
        report["layers"].append(entry)
        report["ok"] &= entry["ok"]
        say(
            f"layer {li}: {entry['bytes_changed']} of {entry['bytes_rewritable']} weight-block "
            f"bytes changed, {entry['changed_outside_weight_blocks']} outside, "
            f"{'ok' if entry['ok'] else 'FAILED'} ({time.time() - t0:.0f} s)"
            + "".join(
                f"  {k} {v.get('reference', float('nan')):.3f}->{v[method]:.3f}"
                for k, v in stats.items()
            )
        )
        if not entry["ok"]:
            raise RuntimeError(f"{out} failed verification: {entry}")
    report["ok"] &= sorted(os.listdir(dst_dir)) == sorted(os.listdir(src_dir))
    if not report["ok"]:
        raise RuntimeError(f"{dst_dir} does not hold the same files as {src_dir}")
    return report


# --------------------------------------------------------------------------
# Calibration input and the CLI
# --------------------------------------------------------------------------
def load_calibration(
    items: Iterable[str], checkpoint_dir: str, seq_len: int = 128, min_len: int = 8
) -> list:
    """Token-id sequences from ``--calibration`` items.

    A ``.json`` file holds a list of token-id lists, or of objects with an ``ids`` list;
    its sequences are used as they are (so chat-templated prompts keep their template).
    Any other existing file is read as text; anything else is literal text. Text is
    tokenized with the checkpoint's ``tokenizer.json`` without special tokens, paragraph
    by paragraph (blank-line separated), and cut into sequences of at most ``seq_len``
    tokens; pieces shorter than ``min_len`` are dropped."""
    tok, out = None, []
    for item in items:
        if item.endswith(".json") and os.path.isfile(item):
            with open(item) as f:
                data = json.load(f)
            seqs = [d["ids"] if isinstance(d, dict) else d for d in data]
            if not all(
                isinstance(s, list) and all(isinstance(t, int) for t in s) for s in seqs
            ):
                raise ValueError(f"{item}: expected a JSON list of token-id lists")
            out += [s for s in seqs if s]
            continue
        if os.path.isfile(item):
            with open(item, errors="replace") as f:
                text = f.read()
        else:
            text = item
        if tok is None:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
        for para in text.split("\n\n"):
            ids = tok.encode(para.strip(), add_special_tokens=False).ids
            pieces = [ids[i : i + seq_len] for i in range(0, len(ids), seq_len)]
            out += [p for p in pieces if len(p) >= min(min_len, seq_len)]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Re-quantize the int4 weights of a compiled llm_build s4 directory.",
        epilog="See docs/axera-llm-int4-requant.md.",
    )
    ap.add_argument("--src", required=True, help="compiled --weight_type s4 directory")
    ap.add_argument(
        "--checkpoint", required=True, help="the checkpoint it was built from"
    )
    ap.add_argument("--dst", required=True, help="output directory (must not exist)")
    ap.add_argument("--method", default="gptq", choices=METHODS)
    ap.add_argument(
        "--calibration",
        nargs="+",
        default=[],
        metavar="FILE_OR_TEXT",
        help="JSON files of token-id lists, text files, or literal text (gptq, gptqa)",
    )
    ap.add_argument("--seq-len", type=int, default=128, help="tokens per text sequence")
    ap.add_argument("--ridge", type=float, default=0.3, help="gptqa: ridge towards W")
    ap.add_argument("--damping", type=float, default=0.01, help="gptq: Hessian damping")
    ap.add_argument("--report", help="write the verification report to this JSON file")
    args = ap.parse_args(argv)
    seqs = None
    if args.method in ("gptq", "gptqa"):
        seqs = load_calibration(args.calibration, args.checkpoint, args.seq_len)
        print(
            f"calibration: {len(seqs)} sequences, {sum(map(len, seqs))} tokens",
            flush=True,
        )
    report = requantize_directory(
        args.src,
        args.checkpoint,
        args.dst,
        args.method,
        seqs,
        ridge=args.ridge,
        damping=args.damping,
        log=lambda line: print(line, flush=True),
    )
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
    changed = sum(e["bytes_changed"] for e in report["layers"])
    print(
        f"{args.dst}: {len(report['layers'])} layer files rewritten (rule {report['rule']}, "
        f"{changed} bytes changed), {len(report['copied'])} files copied unchanged; verified"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
