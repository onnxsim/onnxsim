"""Host-only checks for the ``pf_generate.py`` build-time changes
(``docs/axera-llm-pulsar-free-decode.md``, "Build time"): each fast path gives
the bytes of the code it replaced. No Docker, card, checkpoint or template
build is needed."""

import os
import struct
import sys
import tarfile

import numpy as np
import pytest
from onnx import parser

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.join(HERE, "..", "scripts", "axera")
for _p in (AXERA, os.path.join(AXERA, "llm_pulsar_free")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import linear_emit as le  # noqa: E402
import short_unit_codec as suc  # noqa: E402
import u16_retarget as ur  # noqa: E402


def _scan(raw, lut, fixed):
    """The plain scan ``_rewrite`` used: every even offset in order, reading
    the bytes as already rewritten."""
    p = bytearray(raw)
    for off in range(0, len(p) - 3, 2):
        if any(off + 4 > a and off < b for a, b in fixed):
            continue
        val = struct.unpack_from("<I", p, off)[0]
        if val in lut:
            struct.pack_into("<I", p, off, lut[val])
    return bytes(p)


def test_rewrite_params_equals_the_plain_scan():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        n = int(rng.integers(0, 64))
        # few distinct byte values: words repeat and rewrites chain into the next word
        raw = bytes(rng.choice([0, 1, 2, 255], n).astype(np.uint8))
        words = [struct.unpack_from("<I", raw, o)[0] for o in range(0, n - 3, 2)]
        pool = words + [int(rng.integers(0, 1 << 32))]
        lut = {
            int(rng.choice(pool)): int(rng.choice(pool))
            for _ in range(int(rng.integers(0, 6)))
        }
        fixed = []
        for _ in range(int(rng.integers(0, 3))):
            a = int(rng.integers(0, n + 1))
            fixed.append((a, a + int(rng.integers(0, 10))))
        assert ur._rewrite_params(raw, lut, fixed) == _scan(raw, lut, fixed)


def _u16_table_by_index(codes, bias, m):
    """The U16 table written byte by byte through an index per byte (the
    formula ``_u16_table`` replaced)."""
    n, k = codes.shape
    segs, offs, g_bytes = le.u16_layout(k)
    ng = n // 32
    t = np.zeros(ng * g_bytes, np.uint8)
    o = np.arange(n)
    g, op = o >> 5, o & 31
    u = codes.astype(np.int64)
    rq = np.concatenate(
        [bias.reshape(ng, 32).view(np.uint8), m.reshape(ng, 32).view(np.uint8)], axis=1
    )
    start = 0
    for pairs, off in zip(segs, offs):
        s1 = le._u16_row_bytes(pairs)
        nslot = -(-pairs // le.U16_BLOCK) * le.U16_BLOCK
        jp = np.arange(nslot)
        slot = 72 * (jp // le.U16_BLOCK) + jp % le.U16_BLOCK
        ue = np.full((n, nslot), 128, np.int64)
        uo = np.full((n, nslot), 128, np.int64)
        ue[:, :pairs] = u[:, 2 * start : 2 * (start + pairs) : 2]
        uo[:, :pairs] = u[:, 2 * start + 1 : 2 * (start + pairs) : 2]
        row = s1 * (op & 15) + 36 * ((op >> 4) & 1)
        for p in (0, 1):
            byte = (((ue >> (4 * p)) & 15) | (((uo >> (4 * p)) & 15) << 4)).astype(
                np.uint8
            )
            idx = (g * g_bytes + off)[:, None] + row[:, None] + slot[None, :]
            t[idx + le.U16_BLOCK * p] = byte
        at = np.arange(ng) * g_bytes + off + 16 * s1
        t[at[:, None] + np.arange(256)[None, :]] = rq
        start += pairs
    return t


@pytest.mark.parametrize("k", le.U16_K + (36, 38, 4096))
def test_u16_table_equals_the_byte_index_formula(k):
    rng = np.random.default_rng(k)
    for n in (32, 96):
        codes = rng.integers(0, 256, (n, k), dtype=np.uint8)
        bias = rng.standard_normal(n).astype(np.float32)
        m = rng.standard_normal(n).astype(np.float32)
        got = le._u16_table(codes, bias, m)
        assert got.dtype == np.uint8
        assert np.array_equal(got, _u16_table_by_index(codes, bias, m))


def _members(path):
    with tarfile.open(path) as t:
        return [
            (
                i.name,
                i.size,
                i.mode,
                i.uid,
                i.gid,
                i.uname,
                i.gname,
                i.type,
                sorted(i.pax_headers),
                t.extractfile(i).read(),
            )
            for i in t.getmembers()
        ]


def test_npy_tar_equals_pulsar2_docker_but_the_time_stamp(tmp_path):
    import pf_generate

    P = pytest.importorskip("pulsar2_docker")  # imports onnxsim
    rng = np.random.default_rng(1)
    arrs = [rng.standard_normal((3, 5)).astype(np.float32) for _ in range(12)]
    P.make_numpy_calibration_tar(str(tmp_path / "a.tar"), arrs)
    pf_generate.write_npy_tar(str(tmp_path / "b.tar"), arrs)
    assert _members(tmp_path / "a.tar") == _members(tmp_path / "b.tar")


def test_calibration_from_samples_equals_the_tars(tmp_path):
    import pf_generate

    m = parser.parse_model(
        """<ir_version: 8, opset_import: ["" : 17]>
        g (float[1, 8] a, float[1, 8] b) => (float[1, 8] y) {
          s = Add(a, b)
          y = Mul(s, a)
        }"""
    )
    (tmp_path / "model.onnx").write_bytes(m.SerializeToString())
    rng = np.random.default_rng(2)
    samples = {}
    for n in ("a", "b"):
        samples[n] = pf_generate.Generator._tar(
            str(tmp_path), n, list(rng.standard_normal((6, 1, 8)))
        )
    d = str(tmp_path)
    assert ur.calibrate_graph(d, samples=samples) == ur.calibrate_graph(d)
    assert ur.calibrate(d, "Add", samples=samples) == ur.calibrate(d, "Add")


def test_decode_segments_memo_returns_fresh_equal_lists():
    import graph_stitch as gs
    import misc_op_record_emit as mre

    m = gs.load_model(
        os.path.join(AXERA, "fixtures", "sqrt_1x49_0p01_9p9_oracle.axmodel.gz")
    )
    mc = bytes(mre.mcode_initializer(m).raw_data)
    suc._DECODED.clear()
    a = suc.decode_segments(mc)
    a.append(b"x")
    b = suc.decode_segments(bytearray(mc))
    assert b == a[:-1] and isinstance(b, list)
    suc._DECODED.clear()
    assert suc.decode_segments(mc) == b
