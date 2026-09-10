#!/usr/bin/env python3
"""Emit an `.axmodel`'s weight table without a compiler, by learning its encoding.

The AX650N stores a convolution's weights as **bit planes**: a byte of
`npu_params` is a mixture of single bits taken from several different weight
codes, at offsets that this repository's README recovers rule by rule -- gaps
of 36, strides of 144, per-tap padding, polyphase reversal for transposes, and
a per-output-channel cost that is charged per *bit* rather than per row. Those
rules were worth deriving because they explain the format. They are a poor
foundation for an emitter, because each new shape or dtype adds another.

This module skips the rules. Every bit of the table is either constant for a
given shape, or *equal to one specific bit of one specific weight code* -- the
encoding is a bit permutation, whatever the arithmetic behind it. So compile
the same shape `k` times with different weights, and read the permutation off
directly: each table bit's column of `k` observed values is a signature, and
the code bit carrying the same signature is the one it came from.

`k` random builds give `8k` independent sample bits per code, and naming one
code bit out of `Cout*Cin*K*8` needs about 15, so four builds already make a
wrong match a ~1e-5 event and six leave no doubt. That is the entire method,
and it does not care which of the seven layouts the shape happens to use.

What it buys: one set of reference builds per *shape*, after which any weights
at that shape can be written without Pulsar2 -- which is what
`README.md`'s "An emitter for the llm_build path" already showed for the LLM
path by hand, generalised and made shape-agnostic.

Usage::

    emitter.py learn out_a/ out_b/ ... --weights a.npy b.npy ... -o map.npz
    emitter.py emit reference.axmodel map.npz new_weights.npy -o new.axmodel
"""

from __future__ import annotations

import argparse

import numpy as np
import onnx

#: A table bit that is the same in every sample carries no information about
#: the weights; it is part of the shape's own scaffolding.
CONST = -1


def codes_of(w, bits=8, axis=0):
    """The unsigned codes Pulsar2 stores for weight `w`.

    Per output channel, symmetric, step ``max|w_c| / 127.5``, then offset by
    ``2**(bits-1)`` -- the quantiser `README.md` recovered byte-exactly from
    real weight tables. `precision.quantise_dequantise` is the same rule
    without the storage offset.
    """
    w = np.asarray(w, dtype=np.float32)
    red = tuple(i for i in range(w.ndim) if i != axis)
    peak = np.abs(w).max(axis=red, keepdims=True)
    step = np.where(peak == 0, 1.0, peak / (2.0 ** (bits - 1) - 0.5))
    half = 2 ** (bits - 1)
    q = np.clip(np.rint(w / step), -half, half - 1) + half
    return q.astype(np.uint8)


def table_of(axmodel_path, name="npu_params"):
    """The raw weight table out of a compiled `.axmodel`."""
    model = onnx.load(axmodel_path, load_external_data=False)
    for init in model.graph.initializer:
        if init.name == name:
            return np.frombuffer(bytes(init.raw_data), dtype=np.uint8)
    raise KeyError(f"{name} not in {axmodel_path}")


def _bits(arr):
    """`(n, 8*len)` bit matrix, LSB first within each byte."""
    a = np.asarray(arr, dtype=np.uint8).reshape(len(arr), -1)
    return np.unpackbits(a, axis=1, bitorder="little").astype(np.uint8)


def _signatures(bit_matrix):
    """One integer per column, packing that column's value in every sample."""
    n = bit_matrix.shape[0]
    if n > 63:
        raise ValueError("more than 63 samples would overflow the signature")
    weights = (1 << np.arange(n, dtype=np.uint64)).reshape(-1, 1)
    return (bit_matrix.astype(np.uint64) * weights).sum(axis=0)


def learn(code_samples, table_samples):
    """Learn each table bit's origin from `k` builds of one shape.

    `code_samples` and `table_samples` are equal-length sequences of uint8
    arrays -- the codes (`codes_of`) and the table (`table_of`) of each build.

    Returns `(origin, const, ambiguous)`:

    * `origin[i]` -- the code-bit index table bit `i` copies, or `CONST`,
    * `const[i]` -- that bit's fixed value where `origin[i]` is `CONST`,
    * `ambiguous` -- table bit indices whose signature matched no code bit,
      i.e. everything the shape's scaffolding computes rather than copies
      (per-channel scales and biases live here).
    """
    code_bits = _bits([c.ravel() for c in code_samples])
    table_bits = _bits(table_samples)
    n = code_bits.shape[0]
    if n != table_bits.shape[0]:
        raise ValueError("need one table per code sample")

    code_sig = _signatures(code_bits)
    table_sig = _signatures(table_bits)
    all_zero, all_one = np.uint64(0), np.uint64((1 << n) - 1)

    lookup = {}
    for idx, sig in enumerate(code_sig):
        if sig in (all_zero, all_one):
            continue  # this code bit is itself constant: uninformative
        lookup.setdefault(sig, idx)

    origin = np.full(table_bits.shape[1], CONST, dtype=np.int64)
    const = np.zeros(table_bits.shape[1], dtype=np.uint8)
    ambiguous = []
    for i, sig in enumerate(table_sig):
        if sig == all_zero or sig == all_one:
            const[i] = 1 if sig == all_one else 0
            continue
        hit = lookup.get(sig)
        if hit is None:
            ambiguous.append(i)
        else:
            origin[i] = hit
    return origin, const, np.asarray(ambiguous, dtype=np.int64)


def collisions(code_samples, origin):
    """How many table bits have more than one consistent source.

    A learned map is only trustworthy when this is zero -- it is the direct
    measure of whether `k` was large enough.
    """
    code_bits = _bits([c.ravel() for c in code_samples])
    sig = _signatures(code_bits)
    n = code_bits.shape[0]
    all_zero, all_one = np.uint64(0), np.uint64((1 << n) - 1)
    counts = {}
    for s in sig:
        if s in (all_zero, all_one):
            continue
        counts[s] = counts.get(s, 0) + 1
    used = origin[origin != CONST]
    return int(sum(counts[sig[u]] - 1 for u in np.unique(used)))


def emit_table(reference_table, origin, codes):
    """Write `codes` into a copy of `reference_table` through a learned map.

    Bits the map calls `CONST` -- and anything it could not explain -- keep the
    reference's value, so the result is the reference table with exactly the
    weight bits replaced.
    """
    table_bits = _bits([np.asarray(reference_table, dtype=np.uint8)])[0]
    code_bits = _bits([np.asarray(codes, dtype=np.uint8).ravel()])[0]
    mapped = origin != CONST
    out = table_bits.copy()
    out[mapped] = code_bits[origin[mapped]]
    return np.packbits(out, bitorder="little")


def requant_block(codes, x_scale, x_zero, y_scale, y_zero, w_scale):
    """The per-output-channel float32 pair the table carries after the codes.

    A quantised convolution's output in code units is

        y_code = zy + sum_i (x_code_i - zx) * q_i * (x_scale * w_scale / y_scale)

    and the constant half of that is precomputed per channel. The AX650N stores
    both terms, back to back, as float32:

        [0 : C]      bias[c] = zy - zx * sum(q_c) * M_c
        [C : 2C]     M_c     = x_scale * w_scale_c / y_scale

    Read off 48 builds of one shape: the formula reproduces every one of
    48 x 32 stored values to 6.1e-05, which is float32 rounding.
    """
    codes = np.asarray(codes, dtype=np.int64)
    q = codes.reshape(codes.shape[0], -1) - 2 ** (8 - 1)
    m = (np.asarray(x_scale) * np.asarray(w_scale) / np.asarray(y_scale)).astype(
        np.float32
    )
    bias = (y_zero - x_zero * q.sum(axis=1) * m).astype(np.float32)
    return np.concatenate([bias, m]).view(np.uint8)


def weight_scales(w, bits=8, axis=0):
    """The per-output-channel step Pulsar2 quantises `w` with."""
    w = np.asarray(w, dtype=np.float32)
    red = tuple(i for i in range(w.ndim) if i != axis)
    peak = np.abs(w).max(axis=red)
    return np.where(peak == 0, 1.0, peak / (2.0 ** (bits - 1) - 0.5)).astype(np.float32)


def emit(reference_table, origin, w, x_scale, x_zero, y_scale, y_zero, block_at=None):
    """A whole table for new weights: codes through the map, block in closed form.

    `block_at` is the byte offset of the requantisation block, which `learn`
    reports as the run of unexplained bits. Left `None`, only the codes are
    written and the reference's block is kept -- correct only if the weights
    happen to share its scales.
    """
    codes = codes_of(w)
    table = emit_table(reference_table, origin, codes)
    if block_at is not None:
        block = requant_block(codes, x_scale, x_zero, y_scale, y_zero, weight_scales(w))
        table[block_at : block_at + len(block)] = block
    return table


def emit_mcode(reference_mcode, fields, y_scale, y_zero):
    """A copy of `reference_mcode` carrying a new output quantisation.

    Raises if the new zero point is `RESERVED_OPERAND`, which the stream
    cannot hold as a literal -- there is no in-place edit for it, and writing
    it anyway would produce a stream that decodes as something else.
    """
    zero = int(round(float(y_zero))) & 0xFF
    scale_bytes = np.float32(y_scale).tobytes()
    bad = fields.get("unpatchable", {})
    if zero in set(bad.get("zero", ())):
        raise ValueError(
            f"output zero point {zero} is one this shape's mcode does not "
            "encode as a plain literal -- it shifts the stream, so no in-place "
            "patch expresses it. Nudge the calibration range, or build it."
        )
    if scale_bytes[0] in set(bad.get("scale_low_byte", ())):
        raise ValueError(
            f"output scale {float(y_scale)!r} has low byte "
            f"0x{scale_bytes[0]:02x}, which this shape's mcode does not encode "
            "as a plain literal -- it shifts the stream. Nudge the scale, or "
            "build it."
        )
    out = bytearray(np.asarray(reference_mcode, dtype=np.uint8).tobytes())
    for off in fields.get("scale_offsets", ()):
        out[off : off + 4] = scale_bytes
    for off in fields.get("zero_offsets", ()):
        out[off] = zero
    return np.frombuffer(bytes(out), dtype=np.uint8)


def learn_mcode(mcodes, y_scales, y_zeros, min_agreement=0.9):
    """Find the mcode fields that follow the output quantisation.

    A convolution's mcode is otherwise a function of shape alone -- across 48
    builds of one shape only 24 of 2824 bytes move at all, once the one build
    with a shifted layout is set aside. Of those, this identifies the ones that
    are the output scale or zero point written down literally:

    * `scale_offsets` -- little-endian float32 equal to `y_scale`; the AX650N
      writes four copies,
    * `zero_offsets` -- one byte equal to `round(y_zero)`,
    * `free_offsets` -- varying, and neither. On the reference shape these are
      seven bytes near 303-326 holding a permutation of 0x10/0x20/0x30/0x40
      with 0x13/0x23 separators. **They do not affect the result**: emitting
      eight held-out models with the reference's values left in place produced
      device output bit-identical to Pulsar2's own build every time. They are
      scheduling, not semantics,
    * `outliers` -- builds whose stream does not agree with the majority
      layout, which is how the 0x80 case announces itself.

    A field is accepted when at least `min_agreement` of the builds carry it,
    so one shifted stream does not hide the other 47.
    """
    stack = np.stack([np.asarray(m, dtype=np.uint8) for m in mcodes])
    y_scales = np.asarray(y_scales, dtype=np.float32)
    y_zeros = np.round(np.asarray(y_zeros)).astype(np.int64)
    need = int(np.ceil(min_agreement * len(stack)))

    varying = np.flatnonzero((stack != stack[0]).any(axis=0))
    scale_offsets, zero_offsets, free = [], [], []
    agree = np.zeros(len(stack), dtype=np.int64)
    claimed = set()
    for off in varying:
        off = int(off)
        if off in claimed:
            continue
        if off + 4 <= stack.shape[1]:
            word = stack[:, off : off + 4].copy().view(np.float32).ravel()
            hit = word == y_scales
            if hit.sum() >= need:
                scale_offsets.append(off)
                claimed.update(range(off, off + 4))
                agree += hit
                continue
        hit = stack[:, off].astype(np.int64) == (y_zeros & 0xFF)
        if hit.sum() >= need:
            zero_offsets.append(off)
            agree += hit
            continue
        free.append(off)
    # An outlier is a build the reference cannot be patched into: re-emit each
    # one and see whether anything outside the free offsets survives. That is
    # the property an emitter actually needs, and it does not care why a
    # stream disagrees.
    fields = {"scale_offsets": scale_offsets, "zero_offsets": zero_offsets}
    allowed = set(free)
    outliers = []
    for i in range(len(stack)):
        try:
            got = emit_mcode(stack[0], fields, y_scales[i], y_zeros[i])
        except ValueError:
            outliers.append(i)
            continue
        if not set(np.flatnonzero(got != stack[i]).tolist()) <= allowed:
            outliers.append(i)
    if outliers:
        # a shifted stream makes every byte after the shift look weight-
        # dependent, which would bury the handful that really are. Recompute
        # what varies using only the builds that share the majority layout.
        keep = np.setdiff1d(np.arange(len(stack)), np.asarray(outliers))
        sub = stack[keep]
        claimed = set(o for off in scale_offsets for o in range(off, off + 4))
        free = [
            int(o)
            for o in np.flatnonzero((sub != sub[0]).any(axis=0))
            if int(o) not in claimed and int(o) not in zero_offsets
        ]
    # The stream refuses some literals -- two of 48 builds of the reference
    # shape shift everything after the field rather than writing the value
    # inline, one with an output zero point of exactly 128 and one with a
    # scale whose low byte is 0x9e. No rule tried here predicts which
    # (plenty of non-outliers also carry bytes in 0x80-0x9f), so record the
    # values that actually failed and let `emit_mcode` refuse those.
    ok_zero = {int(y_zeros[i]) & 0xFF for i in range(len(stack)) if i not in outliers}
    ok_low = {
        np.float32(y_scales[i]).tobytes()[0]
        for i in range(len(stack))
        if i not in outliers
    }
    bad_zero = sorted({int(y_zeros[i]) & 0xFF for i in outliers} - ok_zero)
    # attribute each outlier to one field: a zero point no good build uses
    # explains it on its own, and only what is left is blamed on the scale.
    rest = [i for i in outliers if (int(y_zeros[i]) & 0xFF) not in bad_zero]
    unpatchable = {
        "zero": bad_zero,
        "scale_low_byte": sorted(
            {np.float32(y_scales[i]).tobytes()[0] for i in rest} - ok_low
        ),
    }
    return {
        "scale_offsets": scale_offsets,
        "zero_offsets": zero_offsets,
        "free_offsets": free,
        "outliers": outliers,
        "unpatchable": unpatchable,
    }


def mcode_name(axmodel_path):
    """The name of the initializer holding the command stream.

    Pulsar2 names it after the subgraph (`subgraph_npu_0_b1_neu`); the `neu`
    suffix is the marker `pulsar2_ops.AXERA_NPU_OP_TYPE` is built around.
    """
    model = onnx.load(axmodel_path, load_external_data=False)
    for init in model.graph.initializer:
        if init.name.endswith("_neu"):
            return init.name
    raise KeyError(f"no *_neu initializer in {axmodel_path}")


def emit_axmodel(
    reference_path,
    out_path,
    w,
    x_scale,
    x_zero,
    y_scale,
    y_zero,
    origin,
    block_at,
    fields,
    table_name="npu_params",
):
    """Write a whole `.axmodel` for new weights, from a reference build.

    Everything the compiler decides from the *shape* is inherited from the
    reference; everything it decides from the *weights* is recomputed here.
    On the reference shape this reproduces Pulsar2's own weight table byte for
    byte and its own device output bit for bit -- see `README.md`.
    """
    model = onnx.load(reference_path, load_external_data=False)
    name = mcode_name(reference_path)
    table = mc = None
    for init in model.graph.initializer:
        if init.name == table_name:
            table = np.frombuffer(bytes(init.raw_data), dtype=np.uint8)
        elif init.name == name:
            mc = np.frombuffer(bytes(init.raw_data), dtype=np.uint8)
    if table is None or mc is None:
        raise KeyError(f"{reference_path} is missing {table_name} or {name}")

    new_table = emit(
        table, origin, w, x_scale, x_zero, y_scale, y_zero, block_at=block_at
    )
    new_mcode = emit_mcode(mc, fields, y_scale, y_zero)
    for init in model.graph.initializer:
        if init.name == table_name:
            init.raw_data = new_table.tobytes()
        elif init.name == name:
            init.raw_data = new_mcode.tobytes()
    onnx.save(model, out_path)
    return new_table, new_mcode


def save_map(path, origin, const, ambiguous, shape):
    np.savez_compressed(
        path,
        origin=origin,
        const=const,
        ambiguous=ambiguous,
        shape=np.asarray(shape, dtype=np.int64),
    )


def load_map(path):
    z = np.load(path)
    return z["origin"], z["const"], z["ambiguous"], tuple(z["shape"].tolist())


def _replace_initializer(model, name, data):
    for init in model.graph.initializer:
        if init.name == name:
            init.raw_data = bytes(data)
            return True
    return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("learn", help="learn the map from k builds of one shape")
    lp.add_argument("axmodels", nargs="+")
    lp.add_argument(
        "--weights",
        nargs="+",
        required=True,
        help=".npy of each build's float weights, same order",
    )
    lp.add_argument("-o", "--out", required=True)

    ep = sub.add_parser("emit", help="write new weights through a learned map")
    ep.add_argument("reference")
    ep.add_argument("map")
    ep.add_argument("weights")
    ep.add_argument("-o", "--out", required=True)

    args = p.parse_args(argv)
    if args.cmd == "learn":
        ws = [np.load(w) for w in args.weights]
        codes = [codes_of(w) for w in ws]
        tables = [table_of(a) for a in args.axmodels]
        origin, const, ambiguous = learn(codes, tables)
        save_map(args.out, origin, const, ambiguous, ws[0].shape)
        mapped = int((origin != CONST).sum())
        print(
            f"{len(tables)} builds, table {len(tables[0])} B "
            f"({8 * len(tables[0])} bits)"
        )
        print(f"  mapped to a weight bit : {mapped}")
        print(
            f"  constant across builds : {int((origin == CONST).sum()) - len(ambiguous)}"
        )
        print(f"  unexplained            : {len(ambiguous)}")
        print(f"  colliding sources      : {collisions(codes, origin)}")
        return 0

    origin, const, ambiguous, shape = load_map(args.map)
    model = onnx.load(args.reference, load_external_data=False)
    ref = table_of(args.reference)
    codes = codes_of(np.load(args.weights))
    table = emit_table(ref, origin, codes)
    if not _replace_initializer(model, "npu_params", table.tobytes()):
        raise SystemExit("no npu_params in the reference")
    onnx.save(model, args.out)
    print(f"wrote {args.out}: {int((table != ref).sum())} of {len(ref)} bytes changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
