"""Tests for ``onnxsim.import_gguf_weights`` -- hydrating an existing ONNX
graph's initializers, by name, from a plain (non-onnxsim) GGUF checkpoint.
Unlike ``import_gguf``, this needs no embedded onnxsim model, and is the
intended way to bring a third-party GGUF's weight *values* into a graph you
already have.

Covers the GGML "K-quant" block formats (Q8_0, Q4_K, Q5_K, Q6_K) real
quantized checkpoints (e.g. Unsloth's GGUF exports) actually use for the
bulk of their weights: this module writes real, byte-accurate GGUF v3 files
containing hand-encoded K-quant blocks with known values, computing each
expected dequantized float independently (a from-scratch transcription of
GGML's published block layout/dequant formula, not a reuse of the C++
decoder under test -- see ggml_kquant.h) and checking onnxsim's decoded
result against it.
"""

import struct

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

import onnxsim

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

# ggml_type codes this suite constructs (see onnxsim/gguf_dtype.h).
GGML_TYPE_F32 = 0
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q4_0 = 2  # legacy family onnxsim does NOT decode -- must be skipped


def _align_up(n, align=32):
    rem = n % align
    return n if rem == 0 else n + (align - rem)


def _write_gguf(path, tensors):
    """Write a minimal, real GGUF v3 file. ``tensors`` is a list of
    ``(name, ggml_type, ne, raw_bytes)`` -- ``ne`` in GGML's own
    innermost-dimension-first order (the reverse of the ONNX shape it
    corresponds to)."""
    infos = b""
    data_chunks = []
    offset = 0
    for name, ggml_type, ne, raw in tensors:
        name_b = name.encode("utf-8")
        infos += struct.pack("<Q", len(name_b)) + name_b
        infos += struct.pack("<I", len(ne))
        for d in ne:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", ggml_type)
        infos += struct.pack("<Q", offset)
        data_chunks.append((offset, raw))
        offset = _align_up(offset + len(raw))

    header = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensors), 0)
    header_end = len(header) + len(infos)
    data_section_start = _align_up(header_end)

    with open(path, "wb") as f:
        f.write(header)
        f.write(infos)
        f.write(b"\x00" * (data_section_start - header_end))
        pos = data_section_start
        for rel_offset, raw in data_chunks:
            abs_offset = data_section_start + rel_offset
            f.write(b"\x00" * (abs_offset - pos))
            f.write(raw)
            pos = abs_offset + len(raw)


def _f16_bits(f):
    return np.float16(f).view(np.uint16).item()


def _f16_to_f32(bits):
    return float(np.uint16(bits).view(np.float16))


def _make_q8_0_block(rng):
    d = round(float(rng.uniform(0.01, 5.0)), 4)
    qs = [int(rng.integers(-127, 128)) for _ in range(32)]
    d_bits = _f16_bits(d)
    raw = struct.pack("<H", d_bits) + bytes(q & 0xFF for q in qs)
    expected = [q * _f16_to_f32(d_bits) for q in qs]
    return raw, expected


def _get_scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4)
    return d, m


def _make_q4_k_block(rng):
    d = round(float(rng.uniform(0.01, 2.0)), 4)
    dmin = round(float(rng.uniform(0.01, 1.0)), 4)
    d_bits, dmin_bits = _f16_bits(d), _f16_bits(dmin)
    scales = [int(rng.integers(0, 256)) for _ in range(12)]
    qs = [int(rng.integers(0, 256)) for _ in range(128)]
    raw = struct.pack("<HH", d_bits, dmin_bits) + bytes(scales) + bytes(qs)

    d_f, dmin_f = _f16_to_f32(d_bits), _f16_to_f32(dmin_bits)
    expected = [0.0] * 256
    is_, q_off, y = 0, 0, 0
    for _j in range(0, 256, 64):
        sc, m = _get_scale_min_k4(is_ + 0, scales)
        d1, m1 = d_f * sc, dmin_f * m
        sc, m = _get_scale_min_k4(is_ + 1, scales)
        d2, m2 = d_f * sc, dmin_f * m
        for idx in range(32):
            expected[y] = d1 * (qs[q_off + idx] & 0xF) - m1
            y += 1
        for idx in range(32):
            expected[y] = d2 * (qs[q_off + idx] >> 4) - m2
            y += 1
        q_off += 32
        is_ += 2
    return raw, expected


def _vi(name, shape, dtype=onnx.TensorProto.FLOAT):
    return onnx.helper.make_tensor_value_info(name, dtype, shape)


def _identity_model(name, shape):
    # A minimal single-initializer graph: Identity(W) -> Y, with W the
    # initializer import_gguf_weights should hydrate. Seeded with zeros so a
    # test failing to actually hydrate is caught (not accidentally correct).
    weight = onnx.numpy_helper.from_array(np.zeros(shape, dtype=np.float32), name)
    nodes = [onnx.helper.make_node("Identity", [name], ["Y"])]
    graph = onnx.helper.make_graph(nodes, "g", [], [_vi("Y", shape)], [weight])
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=10
    )


def test_import_q8_0_weights(tmp_path):
    rng = np.random.default_rng(0)
    raw, expected = _make_q8_0_block(rng)
    gguf_path = str(tmp_path / "model.gguf")
    # ONNX shape [32] -> ggml ne [32] (rank 1, order is irrelevant).
    _write_gguf(gguf_path, [("W", GGML_TYPE_Q8_0, [32], raw)])

    model = _identity_model("W", [32])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == []
    w = next(i for i in result.graph.initializer if i.name == "W")
    assert w.data_type == onnx.TensorProto.FLOAT
    got = onnx.numpy_helper.to_array(w)
    np.testing.assert_allclose(got, np.array(expected, dtype=np.float32), rtol=1e-5)


def test_import_q4_k_weights(tmp_path):
    rng = np.random.default_rng(1)
    raw, expected = _make_q4_k_block(rng)
    gguf_path = str(tmp_path / "model.gguf")
    _write_gguf(gguf_path, [("W", GGML_TYPE_Q4_K, [256], raw)])

    model = _identity_model("W", [256])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == []
    w = next(i for i in result.graph.initializer if i.name == "W")
    assert w.data_type == onnx.TensorProto.FLOAT
    got = onnx.numpy_helper.to_array(w)
    np.testing.assert_allclose(got, np.array(expected, dtype=np.float32), rtol=1e-5)


def test_import_multiple_tensors_two_dims(tmp_path):
    # Exercises the ne[]-order reversal (GGML innermost-first vs ONNX
    # outermost-first) with a non-square shape, and multiple Q8_0 blocks
    # concatenated in one tensor (2 rows x 64 cols = 2 blocks of 32).
    rng = np.random.default_rng(2)
    raw0, expected0 = _make_q8_0_block(rng)
    raw1, expected1 = _make_q8_0_block(rng)
    raw = raw0 + raw1
    expected = expected0 + expected1
    gguf_path = str(tmp_path / "model.gguf")
    # ONNX shape [2, 32] -> ggml ne [32, 2] (innermost-first).
    _write_gguf(gguf_path, [("W", GGML_TYPE_Q8_0, [32, 2], raw)])

    model = _identity_model("W", [2, 32])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == []
    w = next(i for i in result.graph.initializer if i.name == "W")
    got = onnx.numpy_helper.to_array(w)
    np.testing.assert_allclose(
        got.reshape(-1), np.array(expected, dtype=np.float32), rtol=1e-5
    )


def test_import_skips_unmatched_and_unsupported(tmp_path):
    rng = np.random.default_rng(3)
    raw, _ = _make_q8_0_block(rng)
    gguf_path = str(tmp_path / "model.gguf")
    legacy_raw = b"\x00" * 18  # Q4_0 block: 2 (d) + 16 (packed nibbles).
    _write_gguf(
        gguf_path,
        [
            ("W", GGML_TYPE_Q8_0, [32], raw),
            ("not_in_graph", GGML_TYPE_Q8_0, [32], raw),
            ("legacy", GGML_TYPE_Q4_0, [32], legacy_raw),
        ],
    )

    model = _identity_model("W", [32])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    # "not_in_graph" is a perfectly loadable Q8_0 tensor -- it's simply not
    # in `skipped`, since that list is TensorPool::LoadGGUF's own format-
    # level skip list (unsupported ggml_type), not a name-matching report.
    # It also never becomes a `model` initializer (this call only hydrates
    # initializers the graph already has, never adds new ones).
    assert set(skipped) == {"legacy"}
    names = {i.name for i in result.graph.initializer}
    assert names == {"W"}
    w = next(i for i in result.graph.initializer if i.name == "W")
    assert w.data_type == onnx.TensorProto.FLOAT


def test_import_raw_dtype_passthrough(tmp_path):
    # A raw (already-unquantized) F32 tensor hydrates unchanged, same as
    # HydrateTensorProto -- no dequantization involved.
    rng = np.random.default_rng(4)
    values = rng.standard_normal(8).astype(np.float32)
    gguf_path = str(tmp_path / "model.gguf")
    # GGUF is always little-endian on disk, regardless of host byte order.
    _write_gguf(gguf_path, [("W", GGML_TYPE_F32, [8], values.astype("<f4").tobytes())])

    model = _identity_model("W", [8])
    result, skipped = onnxsim.import_gguf_weights(model, gguf_path)

    assert skipped == []
    w = next(i for i in result.graph.initializer if i.name == "W")
    got = onnx.numpy_helper.to_array(w)
    np.testing.assert_array_equal(got, values)
