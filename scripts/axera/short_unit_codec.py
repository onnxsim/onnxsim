"""Codec for AX650 MCode segment compression -- the "short units".

The variable-length short units in the MCode stream
(``[p][p+1 bytes][tag][reg]``, bare ``[tag][reg]`` pairs, the ``0x9f`` extra
byte; see ``scripts/axera/README.md``) are the tokens of a byte-oriented LZ77
scheme. Once a segment is decompressed, it is a plain array of 8-byte records
``[verb][00][field][bank][value32 LE]``, and every register value appears
there in full. That includes zero points, scale ratios, and ``1/s``. See
``docs/axera-short-unit-encoding.md`` for the evidence.

Token grammar (``t`` is the token byte):

* ``t < 0x80``: literal run. The next ``t + 1`` bytes are copied through.
* ``t >= 0x80``: back-reference.

  * length: ``(t & 0x1f) + 3``. If ``t & 0x1f == 0x1f``, one extra byte ``e``
    follows the offset byte and the length becomes ``34 + e``.
  * offset: ``(b << 2) | ((t >> 5) & 3)``, where ``b`` is the byte after
    ``t``. This is a 10-bit window of 1..1023 bytes. Copies may overlap
    their own output.

Framing: a compressed segment is one whose tail table carries key 5. Key 5
is the exact length of the token stream in bytes. The stream always starts
with a literal token followed by the segment's ``a7 00 00 XX`` header
record. ``mcode.segments`` used to place every start 4 bytes past the true
stream start in 424 of the 1,063 fixture blobs (those with 4 bytes of tail
padding before the FlatBuffers vector); that is fixed, see
``docs/axera-mcode-segments-fix.md``, and ``stream_start`` now only checks
the header record is where ``mcode.segments`` says. Each stream is zero-padded to a multiple of 32 bytes, which is table key 2
in 8-byte words. Uncompressed segments (no key 5) are raw records.
"""

from __future__ import annotations

import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402

MIN_MATCH = 3
MAX_SHORT = 0x1E + MIN_MATCH  # 33: longest copy without the extension byte
MAX_MATCH = 0x1F + MIN_MATCH + 0xFF  # 289
MAX_OFFSET = 0x3FF  # 1023
MAX_LITERAL = 0x80  # 128
RECORD = 8

# Envelope of Pulsar2's own encoder, measured over the fixture corpus; see
# ``encode``.
NATIVE_MIN_OFFSET = 25
NATIVE_MIN_MATCH = 4
NATIVE_TAIL_LITERALS = 12


class CodecError(ValueError):
    pass


def decode(stream: bytes) -> bytes:
    """Decompress a whole token stream (exactly ``table[5]`` bytes)."""
    out = bytearray()
    i, n = 0, len(stream)
    while i < n:
        t = stream[i]
        if t < 0x80:
            j = i + 1 + t + 1
            if j > n:
                raise CodecError(f"literal run at {i} overruns the stream")
            out += stream[i + 1 : j]
            i = j
            continue
        if i + 1 >= n:
            raise CodecError(f"truncated back-reference at {i}")
        length = (t & 0x1F) + MIN_MATCH
        off = (stream[i + 1] << 2) | ((t >> 5) & 3)
        i += 2
        if t & 0x1F == 0x1F:
            if i >= n:
                raise CodecError(f"missing length extension at {i}")
            length += stream[i]
            i += 1
        if not 0 < off <= len(out):
            raise CodecError(f"offset {off} at {i} exceeds history {len(out)}")
        start = len(out) - off
        for k in range(length):
            out.append(out[start + k])
    return bytes(out)


def tokens(stream: bytes) -> list[tuple]:
    """``("L", pos, data)`` / ``("M", pos, length, offset, size)`` per token."""
    res, i = [], 0
    while i < len(stream):
        t = stream[i]
        if t < 0x80:
            res.append(("L", i, stream[i + 1 : i + t + 2]))
            i += t + 2
            continue
        length = (t & 0x1F) + MIN_MATCH
        off = (stream[i + 1] << 2) | ((t >> 5) & 3)
        size = 2
        if t & 0x1F == 0x1F:
            length += stream[i + 2]
            size = 3
        res.append(("M", i, length, off, size))
        i += size
    return res


def _match_token(length: int, off: int) -> bytes:
    hi = (off & 3) << 5
    if length <= MAX_SHORT:
        return bytes([0x80 | hi | (length - MIN_MATCH), off >> 2])
    return bytes([0x80 | hi | 0x1F, off >> 2, length - MAX_SHORT - 1])


def _flush_literals(out: bytearray, lit: bytearray) -> None:
    for k in range(0, len(lit), MAX_LITERAL):
        chunk = lit[k : k + MAX_LITERAL]
        out.append(len(chunk) - 1)
        out += chunk
    lit.clear()


def encode(raw: bytes) -> bytes:
    """Compress ``raw`` into a token stream that ``decode`` inverts exactly.

    The encoder stays inside the envelope that Pulsar2's own streams use.
    Across the fixture corpus, native streams never use an offset below
    ``NATIVE_MIN_OFFSET``, a copy shorter than ``NATIVE_MIN_MATCH``, or a copy
    that overlaps its own output, and they end with at least
    ``NATIVE_TAIL_LITERALS`` literal bytes. A copy outside that envelope
    decodes correctly in software, but it has never been shown to decode on
    the device.

    Parsing is greedy: at each position take the longest copy, with ties
    going to the nearest offset, and cap it so it stops short of the tail.
    This matches the native bytes exactly for most small segments, but not
    all. ``docs/axera-short-unit-encoding.md`` gives the counts.
    """
    out = bytearray()
    lit = bytearray()
    chains: dict[bytes, list[int]] = {}
    n = len(raw)
    key = NATIVE_MIN_MATCH
    i = 0
    while i < n:
        best_len, best_off = 0, 0
        if i + key <= n:
            for p in reversed(chains.get(raw[i : i + key], ())):
                off = i - p
                if off < NATIVE_MIN_OFFSET:
                    continue
                if off > MAX_OFFSET:
                    break
                limit = min(MAX_MATCH, n - i, off)
                length = key
                while length < limit and raw[p + length] == raw[i + length]:
                    length += 1
                if length > best_len:
                    best_len, best_off = length, off
        best_len = min(best_len, n - NATIVE_TAIL_LITERALS - i)
        step = best_len if best_len >= NATIVE_MIN_MATCH else 1
        for p in range(i, i + step):
            if p + key <= n:
                chains.setdefault(raw[p : p + key], []).append(p)
        if step > 1:
            _flush_literals(out, lit)
            out += _match_token(best_len, best_off)
        else:
            lit.append(raw[i])
        i += step
    _flush_literals(out, lit)
    return bytes(out)


def stream_start(mc: bytes, pos: int, table: dict) -> int:
    """Start of the segment that ``mcode.segments`` places at ``pos``,
    checked against the stream's opening header record."""
    if 5 in table:
        if mc[pos] < 0x80 and mc[pos + 1 : pos + 4] == b"\xa7\x00\x00":
            return pos
    elif mc[pos : pos + 3] == b"\xa7\x00\x00":
        return pos
    raise CodecError(f"no segment header near {pos}")


def segment_streams(mc: bytes) -> list[tuple[int, int, dict, bool]]:
    """``(start, size, table, compressed)`` for each segment in stream order.

    ``size`` is the token-stream length (``table[5]``) for compressed
    segments and the tiled length for raw ones."""
    _, segs = mcode.segments(mc)
    res = []
    for pos, length, table in segs:
        start = stream_start(mc, pos, table)
        comp = 5 in table
        res.append((start, table[5] if comp else length, table, comp))
    return res


#: ``decode_segments`` results by blob (a template's blob is decoded several times
#: per retarget, and once per layer); a bounded memo of immutable results
_DECODED: dict[bytes, tuple[bytes, ...]] = {}
_DECODED_MAX = 64


def decode_segments(mc: bytes) -> list[bytes]:
    """Decompressed bytes of every segment, compressed or not."""
    mc = bytes(mc)
    got = _DECODED.get(mc)
    if got is None:
        got = tuple(
            decode(mc[s : s + size]) if comp else mc[s : s + size]
            for s, size, _, comp in segment_streams(mc)
        )
        if len(_DECODED) >= _DECODED_MAX:
            _DECODED.pop(next(iter(_DECODED)))
        _DECODED[mc] = got
    return list(got)


def records(data: bytes) -> list[dict]:
    """Split decompressed segment bytes into 8-byte records.

    Each record is ``verb``, ``field``, ``bank``, ``reg`` (``field |
    bank << 8``), ``value`` (uint32 LE) and ``raw``. All-zero padding words
    are skipped."""
    res = []
    for r in range(0, len(data) - RECORD + 1, RECORD):
        w = data[r : r + RECORD]
        if not any(w):
            continue
        res.append(
            {
                "index": r // RECORD,
                "verb": w[0],
                "field": w[2],
                "bank": w[3],
                "reg": w[2] | (w[3] << 8),
                "value": int.from_bytes(w[4:], "little"),
                "raw": bytes(w),
            }
        )
    return res


def token_bytes(mc: bytes) -> set[int]:
    """Absolute offsets of every LZ77 token byte (literal-run headers and
    back-references, not literal payload) in ``mc``'s compressed segments.

    An in-place patch that overwrites one of these changes how the rest of
    the segment decompresses, not the value it meant to write."""
    res = set()
    for start, size, _, comp in segment_streams(mc):
        if not comp:
            continue
        for tok in tokens(mc[start : start + size]):
            if tok[0] == "L":
                res.add(start + tok[1])
            else:
                res.update(range(start + tok[1], start + tok[1] + tok[4]))
    return res


def register_values(mc: bytes) -> list[list[dict]]:
    """``records`` of every decompressed segment of ``mc``."""
    return [records(d) for d in decode_segments(mc)]


def _table_field_offset(mc: bytes, table_index: int, field: int) -> tuple[int, int]:
    """``(byte offset, width)`` of one tail-table field, which is the same
    vtable walk as ``mcode._tables_at``."""
    u32 = lambda o: struct.unpack_from("<I", mc, o)[0]  # noqa: E731
    i32 = lambda o: struct.unpack_from("<i", mc, o)[0]  # noqa: E731
    u16 = lambda o: struct.unpack_from("<H", mc, o)[0]  # noqa: E731
    vec = mcode.tail_vector(mc)
    p = vec + 4 + 4 * table_index
    tpos = p + u32(p)
    vt = tpos - i32(tpos)
    vsz, tsz = u16(vt), u16(vt + 2)
    if 4 + 2 * field >= vsz or not u16(vt + 4 + 2 * field):
        raise CodecError(f"table {table_index} has no field {field}")
    off = u16(vt + 4 + 2 * field)
    return tpos + off, 4 if off + 4 <= tsz else 2


def replace_segment(mc: bytes, index: int, raw: bytes) -> bytes:
    """Return ``mc`` with compressed segment ``index`` (stream order)
    re-encoded from decompressed bytes ``raw``.

    Every native segment's token stream is zero-padded to a multiple of 32
    bytes (table key 2, in words). This rewrites the stream and its key 5 in
    place, so the blob layout does not change. It raises ``CodecError`` when
    the new stream would need a different padded size, because that means
    re-laying out the blob. This path is not device-verified
    (``docs/axera-short-unit-encoding.md``)."""
    streams = segment_streams(mc)
    start, _, table, comp = streams[index]
    if not comp:
        raise CodecError(f"segment {index} is not compressed")
    stream = encode(raw)
    room = 8 * table[2]
    if -(-len(stream) // 32) * 32 != room:
        raise CodecError(
            f"segment {index}: new stream is {len(stream)} bytes; "
            f"it no longer pads to the allotted {room}"
        )
    out = bytearray(mc)
    out[start : start + room] = stream + bytes(room - len(stream))
    # Tables are stored in reverse stream order.
    at, width = _table_field_offset(mc, len(streams) - 1 - index, 5)
    out[at : at + width] = len(stream).to_bytes(width, "little")
    return bytes(out)


if __name__ == "__main__":
    import gzip

    path = sys.argv[1]
    blob = gzip.open(path).read() if path.endswith(".gz") else open(path, "rb").read()
    if ".mcode" not in path:
        import onnx

        model = onnx.load_model_from_string(blob)
        blob = next(
            bytes(i.raw_data)
            for i in model.graph.initializer
            if i.name.endswith("_neu")
        )
    for seg, recs in enumerate(register_values(blob)):
        print(f"segment {seg}: {len(recs)} records")
        for r in recs:
            f = struct.unpack("<f", r["raw"][4:])[0]
            print(
                f"  {r['index']:5d} {r['raw'].hex(' ')}  reg=0x{r['reg']:04x} value={r['value']:#010x} f32={f:.7g}"
            )
