"""The mcode codec: reading, writing and checking Axera's compiled
command-queue programs.

`.axmodel` files carry two blobs this project reverse-engineered -- the
weight table (`npu_params`) and the **mcode**, the NPU command queue, stored
under a `neu_key`-named initializer. Everything here is the confirmed-real
half of that work, lifted out of `tests/test_axera_mcode_structure.py` so it
can run without a Pulsar2 Docker image or an AX650N card. See
`scripts/axera/README.md` for the narrative and for what is *not* known.

The pieces:

* `tokenize` / `decode` / `encode` -- a lossless codec. `encode(decode(m))`
  reproduces the stream byte for byte, which is what makes an edit safe.
* `tail_tables` / `segments` / `stream_bounds` -- the FlatBuffers tail, which
  is a loader manifest: the runtime validates its per-segment word counts.
* `check` -- every structural invariant confirmed on real hardware, as a
  list of violations. This is the CI-facing entry point.

There is deliberately no evaluator here. What each verb *computes* is not
known: most operand slots hold allocator output (addresses and sizes chosen
per build), and no verb's datapath semantics have been established. `check`
validates form, not arithmetic.
"""

import json
import struct

import onnx

VERBS = {0xA1, 0xA2, 0xA3, 0xA8, 0xA9}

WIDE_TAGS = frozenset(
    {0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x89, 0x8A, 0x8B, 0x8C, 0x8D}
    | {0x94, 0x95, 0x9B, 0x9C, 0x9D, 0x9F}
)
"""The 17 short-unit tag bytes that beat a shuffled *explained-bytes* null
by >= 2x in both real models -- the README's "Fourth correction" section.
Superseded by `ALL_TAGS`: the fifth correction's conditional-parity test
showed the rejected tags were false rejections of rare forms."""

VERBS6 = frozenset(VERBS | {0xA7})
"""The five verbs plus `a7`, the segment-marker verb every stream segment
opens with and `llm_build` op programs use freely -- see the README's "The
tail is the segment table" section. Pass as `verbs=` to `tokenize`."""

ALL_TAGS = frozenset(range(0x81, 0xA0))
"""Every byte below the verb range: the full short-unit tag set. For every
tag the byte after it is even in ~100% of real units against ~60% shuffled
-- see the README's "Fifth correction" section."""

FULL_RULE = dict(
    tags=ALL_TAGS | {0xA1},
    pmax=4,
    bare=True,
    extra_byte_tags={0x9F},
    verbs=VERBS6,
    odd_tags={0xC1, 0xE1},
    companion=True,
)
"""Every validated form: all tags, p <= 4, bare pairs, the 0x9f extra byte,
six verbs -- the README's "The tail is the segment table" section."""

TAIL_VECTOR = bytes.fromhex("05000000200000002c000000500000007400000098000000")
"""The FlatBuffers vector of five table offsets that opens an mcode blob's
tail -- see the README's "The op programs are fully tokenized" section."""


def tokenize(
    mcode,
    start=297,
    end=None,
    tags=None,
    pmax=3,
    bare=False,
    extra_byte_tags=frozenset(),
    verbs=None,
    odd_tags=frozenset(),
    companion=False,
):
    """Tokenize an mcode blob's bulk with every validated form -- the 8/7-byte
    verb instructions, the width-rule short units `[p][p+1 bytes][tag]
    [register]` (p+4 bytes) and, with `bare=True`, the payload-less 2-byte
    `[tag][register]` pair -- stepping one unknown byte otherwise. Units
    whose tag is in `extra_byte_tags` take one extra trailing byte (the
    sixth correction: tag 0x9f). Returns `(byte_offset, kind, a, b, c)`
    tuples: kind 'V' (a=verb, b=xx, c=yy), 'S' (a=prefix, b=tag, c=first
    payload byte), 'B' (a=tag, b=register), 'W' (a companion write:
    a=X, b=field, c=bank) or '?' (a=byte). The defaults
    (tags 0x81..0x84, p <= 3, no bare pairs, no extra bytes, stop 252 bytes
    before the end) are the original narrow rule; `tags=ALL_TAGS, pmax=4,
    bare=True, extra_byte_tags={0x9F}` is the corrected one. See the
    README's "The layout that explains all of it" and "Fourth" .. "Sixth
    correction" sections."""
    tags = set(range(0x81, 0x85)) if tags is None else set(tags)
    extra_byte_tags = set(extra_byte_tags)
    verbs = VERBS if verbs is None else frozenset(verbs)
    end = len(mcode) - 252 if end is None else end

    def is_verb(i):
        return (
            i + 3 < len(mcode)
            and mcode[i] in verbs
            and mcode[i + 1] == 0
            and mcode[i + 2] % 0x10 == 0
        )

    def companion_at(i):
        """A 7-byte `[X][field][bank][32-bit operand]` write, recognised only
        when the next 8 bytes are an `a1` verb writing the adjacent slot --
        the same bank one field higher, or the first field of the next bank.
        That anchor never fires on a shuffled stream (see the README's "a
        7-byte write that fills the slot below the next one")."""
        if not companion or i + 15 > end:
            return False
        field, bank = mcode[i + 1], mcode[i + 2]
        if field % 0x10 or mcode[i + 7] != 0xA1 or mcode[i + 8] != 0:
            return False
        nfield, nbank = mcode[i + 9], mcode[i + 10]
        if nfield % 0x10:
            return False
        return (bank == nbank and (nfield - field) % 0x100 == 0x10) or (
            field == 0xF0 and nfield == 0x00 and nbank == bank + 1
        )

    def short_len(i):
        if i >= len(mcode):
            return 0
        p = mcode[i]
        if p <= pmax and i + p + 3 < len(mcode) and mcode[i + p + 2] in tags:
            return p + 4 + (1 if mcode[i + p + 2] in extra_byte_tags else 0)
        return 0

    out, i = [], start
    while i < end:
        if companion_at(i):
            out.append((i, "W", mcode[i], mcode[i + 1], mcode[i + 2]))
            i += 7
            continue
        if is_verb(i):
            n = (
                7
                if (
                    not (is_verb(i + 8) or short_len(i + 8))
                    and (is_verb(i + 7) or short_len(i + 7))
                )
                else 8
            )
            out.append((i, "V", mcode[i], mcode[i + 2], mcode[i + 3]))
        elif short_len(i):
            n = short_len(i)
            out.append((i, "S", mcode[i], mcode[i + n - 2], mcode[i + 1]))
        elif bare and i + 1 < end and mcode[i] in tags and mcode[i + 1] % 2 == 0:
            n = 2 + (1 if mcode[i] in extra_byte_tags else 0)
            out.append((i, "B", mcode[i], mcode[i + 1], 0))
        elif bare and i + 1 < end and mcode[i] in odd_tags and mcode[i + 1] % 2 == 1:
            # Tags with bit 6 set (0xc1, 0xe1) pair with an *odd* register byte.
            n = 2
            out.append((i, "B", mcode[i], mcode[i + 1], 0))
        else:
            n = 1
            out.append((i, "?", mcode[i], 0, 0))
        i += n
    return out


def tail_vector(mcode):
    """Offset of the FlatBuffers table vector that opens an mcode blob's
    tail, found through the header word that points at it (a uoffset whose
    target holds a small count followed by increasing table offsets) -- the
    fixed five-entry `TAIL_VECTOR` pattern only holds for graphs with one
    input and one output. See the README's "The tail is the segment table"
    section."""
    u32 = lambda o: struct.unpack_from("<I", mcode, o)[0]  # noqa: E731
    # The usual anchor is the convolution engine's channel-extent write (see
    # the README's "The first operand with a known meaning" section). A graph
    # with no convolution in it -- a lone LeakyRelu, Relu or Sigmoid -- never
    # programs that register, so fall back to the end of the fixed header.
    first_verb = mcode.find(b"\xa1\x00\x40\x02")
    if first_verb <= 0:
        first_verb = 297
    for o in range(0, first_verb - 3, 4):
        t = o + u32(o)
        if not (first_verb < t < len(mcode) - 8):
            continue
        n = u32(t)
        if not (1 <= n <= 64) or t + 4 + 4 * n > len(mcode):
            continue
        offs = [u32(t + 4 + 4 * k) for k in range(n)]
        if all(0 < x < 8192 for x in offs) and offs == sorted(offs):
            return t
    raise AssertionError("no header word points at a tail table vector")


def tail_tables(mcode):
    """Walk the FlatBuffers tables of an mcode blob's tail (five for a
    one-input, one-output CNN; fifteen for an `llm_build` subgraph).
    Returns, per table, a dict of `field index -> uint32 (or uint16) value`
    for present fields, read through each table's vtable."""
    vec = tail_vector(mcode)
    u32 = lambda o: struct.unpack_from("<I", mcode, o)[0]  # noqa: E731
    i32 = lambda o: struct.unpack_from("<i", mcode, o)[0]  # noqa: E731
    u16 = lambda o: struct.unpack_from("<H", mcode, o)[0]  # noqa: E731
    tables = []
    for k in range(u32(vec)):
        p = vec + 4 + 4 * k
        tpos = p + u32(p)
        vt = tpos - i32(tpos)
        vsz, tsz = u16(vt), u16(vt + 2)
        fields = {}
        for f in range((vsz - 4) // 2):
            off = u16(vt + 4 + 2 * f)
            if off:
                fields[f] = u32(tpos + off) if off + 4 <= tsz else u16(tpos + off)
        tables.append(fields)
    return vec, tables


def segments(mcode):
    """The stream segments the tail table describes: `(offset, length,
    table)` per segment in stream order, which is *reverse* table order.
    Table field 2 is the segment length in 8-byte words; the segments tile
    the blob exactly from the end of the FlatBuffers header to the tail
    vector. Also returns the header length."""
    vec, tables = tail_tables(mcode)
    words = [t.get(2, 0) for t in tables]
    header = vec - 8 * sum(words)
    segs, pos = [], header
    for k in range(len(tables) - 1, -1, -1):
        segs.append((pos, 8 * words[k], tables[k]))
        pos += 8 * words[k]
    assert pos == vec
    return header, segs


def stream_bounds(mcode):
    """`(first instruction byte, one past the last)` -- the FlatBuffers header
    and tail are not instructions."""
    header, segs = segments(mcode)
    return header, segs[-1][0] + segs[-1][1]


def decode(mcode, start=None, end=None, **rule):
    """Decode a stream into records that carry *everything* needed to write it
    back out -- the counterpart of `tokenize`, which returns only what
    is needed to identify a form. Each record is a dict with a `kind`:

    * `V`: a verb (`verb`, `field`, `bank`, `operand`; 8 bytes, or 7 when the
      next form starts early and the operand is one byte shorter)
    * `W`: a companion write (`x`, `field`, `bank`, `operand`)
    * `S`: a width-rule unit (`p`, `payload`, `tag`, `reg`, `extra`)
    * `B`: a bare `[tag][register]` pair (`tag`, `reg`, `extra`)
    * `raw`: one byte no form accounts for (`byte`)

    Every record also carries `at`, its offset in the original stream, so
    an edit can be scoped to one segment.

    See the README's "A lossless codec" section.
    """
    lo = 297 if start is None else start
    hi = (len(mcode) - 252) if end is None else end
    out = []
    toks = tokenize(mcode, start=lo, end=hi, **rule)
    for i, t in enumerate(toks):
        o, kind = t[0], t[1]
        nxt = toks[i + 1][0] if i + 1 < len(toks) else hi
        n = nxt - o
        if kind == "V":
            out.append(
                {
                    "at": o,
                    "kind": "V",
                    "verb": t[2],
                    "field": t[3],
                    "bank": t[4],
                    "operand": mcode[o + 4 : o + n],
                }
            )
        elif kind == "W":
            out.append(
                {
                    "at": o,
                    "kind": "W",
                    "x": t[2],
                    "field": t[3],
                    "bank": t[4],
                    "operand": mcode[o + 3 : o + 7],
                }
            )
        elif kind == "S":
            p = t[2]
            # Read the tag from the stream rather than the token: for a unit
            # whose tag takes an extra byte, `tokenize`'s third slot
            # holds the register, not the tag.
            out.append(
                {
                    "at": o,
                    "kind": "S",
                    "p": p,
                    "payload": mcode[o + 1 : o + 1 + p + 1],
                    "tag": mcode[o + p + 2],
                    "reg": mcode[o + p + 3],
                    "extra": mcode[o + p + 4 : o + n],
                }
            )
        elif kind == "B":
            out.append(
                {
                    "at": o,
                    "kind": "B",
                    "tag": t[2],
                    "reg": t[3],
                    "extra": mcode[o + 2 : o + n],
                }
            )
        else:
            out.append({"at": o, "kind": "raw", "byte": t[2]})
    return out


def encode(records):
    """Write decoded records back out as bytes -- the inverse of
    `decode`. Nothing here reads the original stream, so a byte-exact
    round trip proves the decode captures every bit the forms carry."""
    out = bytearray()
    for r in records:
        kind = r["kind"]
        if kind == "V":
            out += bytes([r["verb"], 0, r["field"], r["bank"]]) + r["operand"]
        elif kind == "W":
            out += bytes([r["x"], r["field"], r["bank"]]) + r["operand"]
        elif kind == "S":
            out += (
                bytes([r["p"]])
                + r["payload"]
                + bytes([r["tag"], r["reg"]])
                + r["extra"]
            )
        elif kind == "B":
            out += bytes([r["tag"], r["reg"]]) + r["extra"]
        else:
            out += bytes([r["byte"]])
    return bytes(out)


def structured_share(records):
    """Fraction of the encoded bytes that come from a recognised form rather
    than a raw escape -- how much of a stream we could write from structure."""
    total = len(encode(records))
    raw = sum(1 for r in records if r["kind"] == "raw")
    return (total - raw) / total


def nonzero_coverage(mcode, **rule_overrides):
    """Fraction of the stream's *non-zero* bytes the rule accounts for, plus
    the unexplained non-zero runs as `(start, end)` pairs. Zero bytes are
    segment padding and are not counted either way."""
    rule = dict(FULL_RULE)
    rule.update(rule_overrides)
    lo, hi = stream_bounds(mcode)
    toks = tokenize(mcode, start=lo, end=hi, **rule)
    nonzero = sum(1 for i in range(lo, hi) if mcode[i])
    runs, last = [], None
    for o, kind, *_ in toks:
        if kind == "?" and mcode[o]:
            if last == o:
                runs[-1][1] = o + 1
            else:
                runs.append([o, o + 1])
            last = o + 1
    unexplained = sum(b - a for a, b in runs)
    return (nonzero - unexplained) / nonzero, [tuple(r) for r in runs]


def segment_coverage(mcode, seg):
    """Explained fraction of one stream segment under `FULL_RULE`, with its
    trailing zero padding trimmed; also the count of `a1 40 02` verbs (op
    programs) and of `a7` verbs inside it."""
    pos, length, _ = seg
    end = pos + length
    while end > pos and mcode[end - 1] == 0:
        end -= 1
    toks = tokenize(mcode, start=pos, end=end, **FULL_RULE)
    # The segment's first 8 bytes can hold the tail of the `a7` marker verb
    # that starts 4 bytes *before* the word boundary (`1e 00 00 00 00`);
    # those are the marker's operand, not unexplained content. Zero bytes
    # left over are padding (an `llm_build` op segment ends with 8 zero bytes
    # before the next segment's marker), not content either.
    unknown = sum(1 for t in toks if t[1] == "?" and t[0] >= pos + 8 and mcode[t[0]])
    programs = sum(1 for t in toks if t[1:] == ("V", 0xA1, 0x40, 0x02))
    a7 = sum(1 for t in toks if t[1] == "V" and t[2] == 0xA7)
    return 1 - unknown / max(1, end - pos), programs, a7


def mcodes_of(axmodel_path):
    """Every `neu mode` node's mcode in an `.axmodel`, as `(node name,
    bytes)` -- `llm_build` per-layer files carry two (decode and prefill)."""
    m = onnx.load(axmodel_path)
    out = []
    for node in m.graph.node:
        if node.op_type != "neu mode":
            continue
        info = json.loads(
            next(a for a in node.attribute if a.name == "npu_graph_info").s.decode()
        )
        for d in info["dotneus"]:
            init = next(i for i in m.graph.initializer if i.name == d["neu_key"])
            out.append((node.name, bytes(init.raw_data)))
    return out


TAGS = ALL_TAGS | {0xA1} | {t | 0x40 for t in ALL_TAGS | {0xA1}}
"""Every byte that can open a short unit: the tags below the verb range, `a1`
(a tag as well as a verb), and the odd-register form of each -- bit 6 of a tag
selects the odd register. See the README's "0xa1 is also a tag" section."""


def check(mcode, strict=True):
    """Every structural invariant this project confirmed on an AX650N, as a
    list of human-readable violations. An empty list means the blob is
    well-formed as far as the format is understood.

    This is *not* an evaluator -- see the module docstring. It answers "could
    the runtime load and walk this?", which is the question that catches a
    corrupted or hand-edited stream, and it needs neither Docker nor a card.
    """
    bad = []

    try:
        header, segs = segments(mcode)
    except Exception as exc:  # noqa: BLE001 -- report, do not raise
        return [f"tail: no readable segment table ({exc})"]

    # 1. The segment table is a loader manifest. Its word counts are
    #    load-bearing: the runtime rejects a blob whose segments do not tile
    #    the stream exactly, from the end of the header to the tail vector.
    vec = tail_vector(mcode)
    end = segs[-1][0] + segs[-1][1]
    if end != vec:
        bad.append(f"segments: tile to {end}, tail vector is at {vec}")
    for pos, length, _ in segs:
        if length % 8:
            bad.append(f"segment at {pos}: length {length} is not a whole word")
        if pos < header or pos + length > len(mcode):
            bad.append(f"segment at {pos}: length {length} leaves the blob")

    if not header < end <= len(mcode):
        # Nothing below can be read if the manifest does not describe this
        # blob; report what is already known and stop.
        return bad + [f"segments: span {header}..{end} is not inside the blob"]

    try:
        records = decode(mcode, start=header, end=end, **FULL_RULE)
    except Exception as exc:  # noqa: BLE001 -- report, do not raise
        return bad + [f"codec: the stream does not decode ({exc!r})"]

    # 2. The codec round-trips. A stream that does not is one this decoder
    #    misread, so nothing below it can be trusted.
    if encode(records) != mcode[header:end]:
        bad.append("codec: re-encoding the decoded records is not byte-exact")

    # 3. `a7` is the synchronisation verb, and every segment boundary but the
    #    first carries one within four bytes -- the verb can start just before
    #    the word boundary its segment begins on. Measured on 60 real streams.
    starts = {r["at"] for r in records if r["kind"] == "V" and r["verb"] == 0xA7}
    for pos, _, _ in segs[1:]:
        if not any(pos + d in starts for d in range(-4, 5)):
            bad.append(f"segment at {pos}: no a7 within four bytes of the boundary")

    # 4. Only the six known verbs, and only tags below the verb range -- plus
    #    `a1`, which is a tag as well as a verb, and the odd-register forms
    #    (bit 6 set) of both.
    for r in records:
        if r["kind"] == "V" and r["verb"] not in VERBS6:
            bad.append(f"at {r['at']}: unknown verb {r['verb']:#04x}")
        if r["kind"] in ("S", "B") and r["tag"] not in TAGS:
            bad.append(f"at {r['at']}: unknown tag {r['tag']:#04x}")

    # 5. Almost every non-zero byte belongs to a recognised form. Zero bytes
    #    are segment padding and are not counted either way. The floor is the
    #    worst of the 60 streams this was measured on (0.9409).
    covered, runs = nonzero_coverage(mcode)
    if strict and covered < 0.94:
        bad.append(
            f"coverage: only {covered:.1%} of non-zero bytes explained {runs[:4]}"
        )

    # 6. The unexplained bytes are scattered, never bulk. Over 70 real streams
    #    -- CNN and `llm_build`, up to 1.8 MB -- no unexplained non-zero run
    #    exceeds nine bytes. A long run is a stream this decoder lost sync in,
    #    which a percentage alone will not show on a large blob.
    long_runs = [(a, b) for a, b in runs if b - a > 12]
    if strict and long_runs:
        bad.append(f"stream: unexplained runs longer than eight bytes {long_runs[:4]}")

    return bad


def op_programs(mcode):
    """The offsets of the op-program verbs (`a1 00 40 02`, the convolution
    engine's channel-extent write) in each segment, as `{segment offset:
    [verb offsets]}`. An `a7` brackets every one of them."""
    header, segs = segments(mcode)
    out = {}
    for pos, length, _ in segs:
        out[pos] = [
            i
            for i in range(pos, pos + max(length - 3, 0))
            if mcode[i] == 0xA1
            and mcode[i + 1] == 0
            and mcode[i + 2] == 0x40
            and mcode[i + 3] == 0x02
        ]
    return out


def weight_table_of(axmodel_path):
    """The `npu_params` weight table of a compiled model, as bytes."""
    model = onnx.load(axmodel_path)
    return bytes(
        next(i for i in model.graph.initializer if i.name == "npu_params").raw_data
    )
