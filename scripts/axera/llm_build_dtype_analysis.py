#!/usr/bin/env python3
"""How ``pulsar2 llm_build`` stores each ``--weight_type`` in an AX650 engine.

This is black-box analysis of our own builds. A tiny synthetic Llama
checkpoint is compiled once for each of ``s4``, ``s8``, ``fp16``, ``bf16``,
``fp8_e4m3`` and ``fp8_e5m2``. The ``npu_params`` weight table is then
compared, byte for byte, with the encoders below. The encoders reproduce the
compiler's bytes for every Linear in the layer. The one exception is a
handful of s8 codes that sit within a few ulps of a rounding tie (see
``quantize_int``). Details are in ``docs/axera-llm-build-dtype-analysis.md``.

Weights are stored in blocks of 32 output rows (``ROW_BLOCK``). Within a
block the byte layout is as follows:

* ``s8``: two nibble planes per row (``s8_offset``), then a 512-byte tail.
  The tail is 32 x (int32 Q16.16 ``-sum(q)/2``, 4 zero bytes), then 128 zero
  bytes, then 32 float32 per-row scales.
* ``s4``: one nibble plane per row (``s4_offset``), then a 384-byte tail.
  The tail is 32 x int32 Q16.16 ``-sum(q)/2``, then 128 zero bytes, then 32
  float32 per-row scales.
* float types: one float32 word per weight, column-major inside the block
  (``float_offset``). The word holds the weight rounded to the requested
  type. There are no scales.

Byte-identical 32-row blocks are stored only once (``dedup_blocks``). The
decode and prefill subgraphs share a single ``npu_params``.

Everything above was derived on the tiny 256-hidden model. Three things differ
on real models and are handled in ``llm_int4_requant.py`` (see
``docs/axera-llm-int4-requant.md``): the Qwen3-0.6B s4 build uses another
quantizer (``scale = max|w| / 7``, round half to even, codes in [-7, 7]);
SmolLM2-135M's q/k/v are stored in blocks of 64 rows, not 32; and the ``post``
model is s8 whatever ``--weight_type`` says. ``column_blocks`` below already
has the rule that fits the real widths.

Usage::

    python llm_build_dtype_analysis.py CHECKPOINT_DIR OUT_DIR:WEIGHT_TYPE ...
"""

from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

IMAGE = "pulsar2:7.0-lite"  # every build in the doc used this image
ROW_BLOCK = 32
CHUNK = 18  # bytes of one nibble plane per 36-column chunk
CHUNK_COLS = 2 * CHUNK
COL_BLOCK = 544  # s8/s4 width of every column block after the first
FIRST_COL_BLOCK = 576  # the first column block holds up to 16 chunks of 36
WEIGHT_TYPES = ("s4", "s8", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2")
FLOAT_TYPES = ("fp16", "bf16", "fp8_e4m3", "fp8_e5m2")
PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


# --- quantizers -------------------------------------------------------------


def quantize_int(w: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric quantizer that llm_build uses for s8/s4.

    ``scale = -w[argmax |w|] / 2**(bits-1)``, which is positive when the
    largest-magnitude weight is negative. ``q = floor(w / scale + 0.5)``, so
    ties round toward +inf. ``q`` is clipped to the signed range. Returns
    ``(q, scale)`` where ``q`` is signed. There is one scale per output row
    and no k-groups, even at ``cin = 512``.

    This matches every s4 code in our builds. For s8 it misses 3 of 557,056
    codes, and 10 of 1,736,704 in the ``inter=2048`` build. Every miss has
    ``w / scale`` within 4e-6 of a .5 tie (for example 31.4999943,
    -8.5000019, -0.5000037), and each time the compiler's code is one higher
    than ours. Its ratio arithmetic differs from float32 ``w / scale`` in the
    last few ulps, and we have not identified how."""
    w = np.asarray(w, np.float32)
    half = 1 << (bits - 1)
    peak = w[np.arange(len(w)), np.argmax(np.abs(w), axis=1)]
    scale = (-peak / np.float32(half)).astype(np.float32)
    q = np.floor(w / scale[:, None] + np.float32(0.5))
    return np.clip(q, -half, half - 1).astype(np.int64), scale


def _round_mantissa(x: np.ndarray, mant: int, emin: int, maxv: float) -> np.ndarray:
    a = np.abs(x.astype(np.float64))
    e = np.maximum(np.floor(np.log2(np.maximum(a, 1e-300))), emin)
    step = 2.0 ** (e - mant)
    return (np.sign(x) * np.minimum(np.rint(a / step) * step, maxv)).astype(np.float32)


def round_float(w: np.ndarray, weight_type: str) -> np.ndarray:
    """The value stored for a float weight type. It is a float32 holding the
    weight rounded, round-to-nearest-even, straight to the type. fp8 has no
    scale, so small weights go subnormal or zero."""
    w = np.asarray(w, np.float32)
    if weight_type == "fp16":
        return w.astype(np.float16).astype(np.float32)
    if weight_type == "bf16":
        u = w.view(np.uint32).astype(np.uint64)
        u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
        return u.astype(np.uint32).view(np.float32)
    if weight_type == "fp8_e4m3":
        return _round_mantissa(w, 3, -6, 448.0)
    if weight_type == "fp8_e5m2":
        return _round_mantissa(w, 2, -14, 57344.0)
    raise ValueError(weight_type)


# --- in-block layouts -------------------------------------------------------


def _chunks(cin: int) -> int:
    return -(-cin // CHUNK_COLS)


def s8_row_bytes(cin: int) -> int:
    """Two planes of 18 bytes per 36-column chunk."""
    return 2 * CHUNK * _chunks(cin)


def s4_row_bytes(cin: int) -> int:
    return CHUNK * _chunks(cin)


def s8_offset(row: int, col: int, cin: int) -> tuple[int, int]:
    """``(byte, shift)`` of the low-nibble-plane entry of ``w[row, col]``
    within its 32-row block. The high nibble sits ``CHUNK`` bytes later at the
    same shift. The unit is 72 bytes: row ``r``'s two planes, then row
    ``r + 16``'s two planes. The external "36 metadata bytes per 72-byte unit"
    are the other row. ``cin`` here is the width of one column block (see
    ``column_blocks``)."""
    a = 4 * CHUNK * _chunks(cin)
    r = row % ROW_BLOCK
    return (
        a * (r % 16)
        + 2 * CHUNK * (r // 16)
        + 4 * CHUNK * (col // CHUNK_COLS)
        + (col % CHUNK_COLS) // 2
    ), 4 * (col % 2)


def s4_offset(row: int, col: int, cin: int) -> tuple[int, int]:
    """``(byte, shift)`` of ``w[row, col]`` within its 32-row block. The unit
    is 36 bytes: 18 bytes of row ``2k``, then 18 bytes of row ``2k + 1``. Each
    byte holds two columns, ``(code[2j+1] << 4) | code[2j]``."""
    r = row % ROW_BLOCK
    per_pair = 2 * s4_row_bytes(cin)
    return (
        per_pair * (r // 2)
        + CHUNK * (r % 2)
        + 2 * CHUNK * (col // CHUNK_COLS)
        + (col % CHUNK_COLS) // 2
    ), 4 * (col % 2)


def float_offset(row: int, col: int, cin: int) -> int:
    """Byte offset of ``w[row, col]``'s float32 word within its block. The
    block is column-major: column ``c`` holds 32 consecutive rows."""
    return 4 * (col * ROW_BLOCK + row % ROW_BLOCK)


# --- block encoders ---------------------------------------------------------


def _rowsum_q16(q: np.ndarray) -> np.ndarray:
    """Per-row ``-sum(q) / 2`` in Q16.16, where ``q`` is the signed code."""
    return (-q.sum(axis=1) * (1 << 15)).astype("<i4")


def _int_tail(q: np.ndarray, scale: np.ndarray, sum_stride: int) -> bytes:
    sums = np.zeros((ROW_BLOCK, sum_stride // 4), "<i4")
    sums[:, 0] = _rowsum_q16(q)
    return sums.tobytes() + bytes(128) + scale.astype("<f4").tobytes()


def column_blocks(cin: int) -> list[int]:
    """Widths of the column blocks an s8/s4 weight is split into.

    The first block holds up to ``FIRST_COL_BLOCK`` (576, i.e. 16 chunks of
    36) columns and every later block exactly ``COL_BLOCK`` (544). There are
    as few 544-wide blocks as leave the first one at most 576 wide, so the
    first block takes the remainder: 576 is a single block, 1024 gives 480,
    544, 1536 gives 448, 544, 544, 2048 gives 416, 544, 544, 544, 3072 gives
    352 then five 544s, and 4096 gives 288 then seven 544s (the README's "The
    LLM layout at 4096 hidden"). Observed at exactly those widths plus the
    single blocks 256 and 512. ``cin = 576`` (SmolLM2-135M) is the only one
    that separates this rule from "split once ``cin > 544``", which was the
    rule here before and would give 32, 544. No width with ``cin % 544`` in
    1..31 has been built.

    Each block is laid out as though it were a whole matrix of its own width,
    with its own tail. The tail repeats the full row's scale, and its row sums
    cover only that block's columns. Float types are never split."""
    n = max(0, -(-(cin - FIRST_COL_BLOCK) // COL_BLOCK))
    return [cin - n * COL_BLOCK] + [COL_BLOCK] * n


def _int_offsets(weight_type: str, width: int) -> tuple[np.ndarray, np.ndarray]:
    padded = CHUNK_COLS * _chunks(width)
    fn = s8_offset if weight_type == "s8" else s4_offset
    off = np.zeros((ROW_BLOCK, padded), np.int64)
    sh = np.zeros((ROW_BLOCK, padded), np.int64)
    for r in range(ROW_BLOCK):
        for c in range(padded):
            off[r, c], sh[r, c] = fn(r, c, width)
    return off, sh


def _part_bytes(weight_type: str, width: int) -> int:
    if weight_type == "s8":
        return 16 * 4 * CHUNK * _chunks(width) + 512
    return ROW_BLOCK * s4_row_bytes(width) + 384


def _encode_part(q: np.ndarray, scale: np.ndarray, weight_type: str) -> bytes:
    bits = 8 if weight_type == "s8" else 4
    width = q.shape[1]
    off, sh = _int_offsets(weight_type, width)
    # Columns past the block width, up to the next 36-column chunk, hold the
    # zero point.
    code = np.full(off.shape, 1 << (bits - 1), np.int64)
    code[:, :width] += q
    body = np.zeros(
        _part_bytes(weight_type, width) - (512 if bits == 8 else 384), np.int64
    )
    np.add.at(body, off.ravel(), ((code & 15) << sh).ravel())
    if bits == 8:
        np.add.at(body, (off + CHUNK).ravel(), ((code >> 4) << sh).ravel())
    return body.astype(np.uint8).tobytes() + _int_tail(q, scale, 8 if bits == 8 else 4)


def encode_block(w: np.ndarray, weight_type: str) -> bytes:
    """The exact bytes llm_build writes for one 32-row block of a Linear
    weight ``w`` (``[32, cin]``, float32)."""
    w = np.asarray(w, np.float32)
    assert w.shape[0] == ROW_BLOCK, w.shape
    cin = w.shape[1]
    rows = np.arange(ROW_BLOCK)[:, None]
    cols = np.arange(cin)[None, :]
    if weight_type in FLOAT_TYPES:
        out = np.zeros(ROW_BLOCK * cin, "<f4")
        out[(cols * ROW_BLOCK + rows).ravel()] = round_float(w, weight_type).ravel()
        return out.tobytes()
    q, scale = quantize_int(w, 8 if weight_type == "s8" else 4)
    out, start = b"", 0
    for width in column_blocks(cin):
        out += _encode_part(q[:, start : start + width], scale, weight_type)
        start += width
    return out


def block_bytes(weight_type: str, cin: int) -> int:
    if weight_type in FLOAT_TYPES:
        return 4 * ROW_BLOCK * cin
    return sum(_part_bytes(weight_type, w) for w in column_blocks(cin))


def decode_block(data: bytes, weight_type: str, cin: int) -> dict:
    """Inverse of ``encode_block``. Returns ``values`` for float types. For
    s8/s4 it returns ``q`` (signed), ``scale``, and ``rowsum`` (one column
    per column block)."""
    if weight_type in FLOAT_TYPES:
        f = np.frombuffer(data[: block_bytes(weight_type, cin)], "<f4")
        return {"values": f.reshape(cin, ROW_BLOCK).T.copy()}
    bits = 8 if weight_type == "s8" else 4
    stride = 8 if bits == 8 else 4
    b = np.frombuffer(data, np.uint8).astype(np.int64)
    qs, sums, scale, pos = [], [], None, 0
    for width in column_blocks(cin):
        off, sh = _int_offsets(weight_type, width)
        off, sh = off[:, :width] + pos, sh[:, :width]
        code = (b[off] >> sh) & 15
        if bits == 8:
            code |= ((b[off + CHUNK] >> sh) & 15) << 4
        qs.append(code - (1 << (bits - 1)))
        tail = pos + _part_bytes(weight_type, width) - (512 if bits == 8 else 384)
        sums.append(
            np.frombuffer(data[tail : tail + ROW_BLOCK * stride], "<i4")[:: stride // 4]
            / 65536.0
        )
        s0 = tail + ROW_BLOCK * stride + 128
        scale = np.frombuffer(data[s0 : s0 + 4 * ROW_BLOCK], "<f4").copy()
        pos += _part_bytes(weight_type, width)
    return {
        "q": np.concatenate(qs, axis=1),
        "scale": scale,
        "rowsum": np.stack(sums, axis=1),
    }


def encode_matrix(w: np.ndarray, weight_type: str) -> list[bytes]:
    """One ``encode_block`` per 32-row block of ``w``."""
    return [
        encode_block(w[i : i + ROW_BLOCK], weight_type)
        for i in range(0, len(w), ROW_BLOCK)
    ]


def dedup_blocks(blocks: list[bytes]) -> list[bytes]:
    """The blocks the compiler actually stores. A block byte-identical to an
    earlier one of the same Linear is dropped (verified for all-identical
    rows at s4 and bf16: the region shrinks by exactly the repeated blocks)."""
    seen, out = set(), []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


# --- the synthetic checkpoint and the build ----------------------------------


def tiny_llama_weights(
    inter: int = 512, q_seed: int = 11, seed: int = 3
) -> dict[str, np.ndarray]:
    """The one-layer Llama we built: hidden 256, 8 heads, 2 KV heads, vocab
    512. ``q_proj`` is ``RandomState(q_seed)``. Everything else draws from
    ``RandomState(seed)`` in checkpoint order. Norm weights are ones."""
    hidden, kv, vocab = 256, 64, 512
    q = (np.random.RandomState(q_seed).randn(hidden, hidden) * 0.02).astype(np.float32)
    rng = np.random.RandomState(seed)

    def s(*shape):
        return (rng.randn(*shape) * 0.02).astype(np.float32)

    ones = np.ones(hidden, np.float32)
    return {
        "model.embed_tokens.weight": s(vocab, hidden),
        "model.layers.0.self_attn.q_proj.weight": q,
        "model.layers.0.self_attn.k_proj.weight": s(kv, hidden),
        "model.layers.0.self_attn.v_proj.weight": s(kv, hidden),
        "model.layers.0.self_attn.o_proj.weight": s(hidden, hidden),
        "model.layers.0.mlp.gate_proj.weight": s(inter, hidden),
        "model.layers.0.mlp.up_proj.weight": s(inter, hidden),
        "model.layers.0.mlp.down_proj.weight": s(hidden, inter),
        "model.layers.0.input_layernorm.weight": ones,
        "model.layers.0.post_attention_layernorm.weight": ones,
        "model.norm.weight": ones,
        "lm_head.weight": s(vocab, hidden),
    }


def write_checkpoint(path: str, weights: dict[str, np.ndarray]) -> None:
    """A float32 safetensors checkpoint and a Llama ``config.json``."""
    os.makedirs(path, exist_ok=True)
    header, blobs, off = {}, [], 0
    for name, a in weights.items():
        data = np.ascontiguousarray(a, np.float32).tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(a.shape),
            "data_offsets": [off, off + len(data)],
        }
        off += len(data)
        blobs.append(data)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(os.path.join(path, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    hidden = weights["model.norm.weight"].shape[0]
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden,
        "intermediate_size": weights["model.layers.0.mlp.gate_proj.weight"].shape[0],
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": weights["lm_head.weight"].shape[0],
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "torch_dtype": "float32",
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "attention_bias": False,
        "mlp_bias": False,
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)


def build_command(
    work_dir: str, ckpt_rel: str, out_rel: str, weight_type: str, name: str
) -> list[str]:
    """The memory-capped ``llm_build`` invocation we used. Run it under
    ``flock /tmp/pulsar2-build.lock``, one build at a time. The prefill and KV
    lengths are the smallest the other llm_build tests use."""
    return [
        "systemd-run", "--user", "--wait", "--collect", "--pipe",
        "-p", "MemoryMax=24G", "-p", "MemorySwapMax=0",
        "docker", "run", "--rm", "--name", name, "--memory", "24g", "--memory-swap", "24g",
        "-v", f"{work_dir}:/data", IMAGE,
        "pulsar2", "llm_build", "--input_path", f"/data/{ckpt_rel}", "--output_path", f"/data/{out_rel}",
        "--hidden_state_type", "bf16", "--weight_type", weight_type,
        "--prefill_len", "64", "--kv_cache_len", "127", "--chip", "AX650", "--parallel", "1",
    ]  # fmt: skip


# --- engine helpers ----------------------------------------------------------


def load_safetensors(path: str) -> dict[str, np.ndarray]:
    raw = open(path, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + n])
    out = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        assert v["dtype"] == "F32", v["dtype"]
        s, e = v["data_offsets"]
        out[k] = np.frombuffer(raw[8 + n + s : 8 + n + e], np.float32).reshape(
            v["shape"]
        )
    return out


def layer_engine(out_dir: str) -> str:
    return next(
        os.path.join(out_dir, f)
        for f in sorted(os.listdir(out_dir))
        if "_l0_" in f and f.endswith(".axmodel")
    )


def engine_parts(axmodel_path: str) -> dict:
    """``npu_params`` and each subgraph's MCode, plus which constant table
    each subgraph reads."""
    import onnx

    m = onnx.load(axmodel_path)
    inits = {i.name: bytes(i.raw_data) for i in m.graph.initializer}
    subgraphs = []
    for node in m.graph.node:
        info = json.loads(
            next(a for a in node.attribute if a.name == "npu_graph_info").s
        )
        for d in info["dotneus"]:
            subgraphs.append(
                {
                    "name": node.name,
                    "mcode": inits[d["neu_key"]],
                    "params_keys": [
                        e["const_data_key"] for e in d.get("extra_inputs", [])
                    ],
                }
            )
    return {"npu_params": inits["npu_params"], "subgraphs": subgraphs}


def locate(
    params: bytes, weights: dict[str, np.ndarray], weight_type: str
) -> list[dict]:
    """Find every projection's encoded blocks in ``params``. Each entry has
    the offset of every stored block, or ``None`` if the exact bytes are
    absent."""
    res = []
    for name in PROJECTIONS:
        w = weights[f"model.layers.0.{name}.weight"]
        blocks = dedup_blocks(encode_matrix(w, weight_type))
        offs = [params.find(b) for b in blocks]
        size = block_bytes(weight_type, w.shape[1])
        mismatched_codes, ratios = 0, []
        for i, o in enumerate(offs):
            # A block whose bytes are absent is compared at the offset its
            # found neighbour implies, to count how many codes differ.
            if (
                o >= 0
                or weight_type in FLOAT_TYPES
                or len(blocks) != len(w) // ROW_BLOCK
            ):
                continue
            near = [j for j in range(len(offs)) if offs[j] >= 0]
            if not near:
                continue
            j = min(near, key=lambda k: abs(k - i))
            at = offs[j] + (i - j) * size
            got = decode_block(params[at : at + size], weight_type, w.shape[1])["q"]
            wb = w[i * ROW_BLOCK : (i + 1) * ROW_BLOCK]
            want, scale = quantize_int(wb, 8 if weight_type == "s8" else 4)
            bad = got != want
            mismatched_codes += int(bad.sum())
            ratios.extend(float(x) for x in (wb / scale[:, None])[bad])
        res.append(
            {
                "name": name,
                "shape": list(w.shape),
                "blocks": len(blocks),
                "block_bytes": block_bytes(weight_type, w.shape[1]),
                "offsets": [o if o >= 0 else None for o in offs],
                "exact": all(o >= 0 for o in offs),
                "mismatched_codes": mismatched_codes,
                "mismatched_ratios": ratios,
                "contiguous": all(
                    b - a == block_bytes(weight_type, w.shape[1])
                    for a, b in zip(offs, offs[1:])
                ),
            }
        )
    return res


def mcode_summary(mcode: bytes) -> list[dict]:
    """Per-segment record counts of an MCode blob, decoded with
    ``short_unit_codec``. It parses llm_build engines unchanged."""
    import short_unit_codec as suc

    return [
        {"records": len(recs), "regs": len({r["reg"] for r in recs})}
        for recs in (suc.records(d) for d in suc.decode_segments(mcode))
    ]


def mcode_record_diff(a: bytes, b: bytes) -> list[dict]:
    """Per segment, the ``(verb, reg)`` records whose value differs between
    two MCodes, plus the change in record count. Records are compared by
    position within each ``(verb, reg)`` stream."""
    import short_unit_codec as suc

    out = []
    for i, (da, db) in enumerate(zip(suc.decode_segments(a), suc.decode_segments(b))):
        ra, rb = suc.records(da), suc.records(db)
        sa, sb = {}, {}
        for r in ra:
            sa.setdefault((r["verb"], r["reg"]), []).append(r["value"])
        for r in rb:
            sb.setdefault((r["verb"], r["reg"]), []).append(r["value"])
        changed = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
        out.append(
            {"segment": i, "records": (len(ra), len(rb)), "changed_streams": changed}
        )
    return out


def analyze(checkpoint_dir: str, builds: dict[str, str]) -> dict:
    """``builds`` maps weight type to llm_build output dir."""
    weights = load_safetensors(os.path.join(checkpoint_dir, "model.safetensors"))
    n_params = sum(weights[f"model.layers.0.{p}.weight"].size for p in PROJECTIONS)
    report = {}
    for wt, out_dir in builds.items():
        parts = engine_parts(layer_engine(out_dir))
        params = parts["npu_params"]
        located = locate(params, weights, wt)
        weight_bytes = sum(e["blocks"] * e["block_bytes"] for e in located)
        report[wt] = {
            "npu_params_bytes": len(params),
            "layer_params": n_params,
            "bytes_per_param": len(params) / n_params,
            "weight_bytes": weight_bytes,
            "weight_bytes_per_param": weight_bytes / n_params,
            "other_bytes": len(params) - weight_bytes,
            "projections": located,
            "shared_params": all(
                s["params_keys"] == ["npu_params"] for s in parts["subgraphs"]
            ),
            "mcode": {s["name"]: mcode_summary(s["mcode"]) for s in parts["subgraphs"]},
        }
    return report


def main(argv: list[str]) -> int:
    ckpt, *pairs = argv
    builds = {}
    for p in pairs:
        out_dir, wt = p.rsplit(":", 1)
        builds[wt] = out_dir
    report = analyze(ckpt, builds)
    for wt, r in report.items():
        print(
            f"{wt:9s} npu_params={r['npu_params_bytes']:8d} B "
            f"({r['bytes_per_param']:.3f} B/param; weights {r['weight_bytes_per_param']:.4f}) "
            f"shared={r['shared_params']} "
            f"exact={all(p['exact'] for p in r['projections'])}"
        )
        for p in r["projections"]:
            print(
                f"    {p['name']:18s} {p['shape']} blocks={p['blocks']} "
                f"offsets={p['offsets'][:2]}... exact={p['exact']} "
                f"mismatched_codes={p['mismatched_codes']} {p['mismatched_ratios']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
