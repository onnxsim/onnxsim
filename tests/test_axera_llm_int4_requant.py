"""Regression coverage for ``scripts/axera/llm_int4_requant.py``.

The script rewrites the int4 weight blocks of a compiled ``pulsar2 llm_build
--weight_type s4`` directory. No Docker, device or network is needed.

``fixtures/llm_int4_requant/slices.npz`` holds four row blocks cut from our own
builds' ``npu_params`` (``*_block``) and the matching checkpoint weight rows as
bf16 bit patterns (``*_bf16``); ``summary.json`` says where each came from:

* ``smol_v``: SmolLM2-135M ``v_proj`` rows 0..63, a 64-row block, one 576-wide
  column part, quantizer rule A.
* ``smol_o``: SmolLM2-135M ``o_proj`` rows 192..223, a 32-row block holding one
  code the plain quantizer misses (an exact tie).
* ``smol_down``: SmolLM2-135M ``down_proj`` rows 0..31, column parts 448, 544, 544.
* ``qwen3_q``: Qwen3-0.6B ``q_proj`` rows 0..31, column parts 480, 544, rule B.

The end-to-end tests run on a synthetic "compiled" directory: the tiny Llama of
``llm_build_dtype_analysis`` with its weight blocks placed at the offsets the real
s4 build of that checkpoint has (``fixtures/llm_build_dtype_analysis/summary.json``).
See ``docs/axera-llm-int4-requant.md``."""

import json
import os
import sys

import numpy as np
import onnx
import pytest
from onnx import parser

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import llm_build_dtype_analysis as lda  # noqa: E402
import llm_int4_requant as rq  # noqa: E402
import llm_layer_loop as loop  # noqa: E402
import llm_reference as ref  # noqa: E402

_FIX = os.path.join(_AXERA_DIR, "fixtures", "llm_int4_requant")
_SLICES = dict(np.load(os.path.join(_FIX, "slices.npz")))
_SUMMARY = json.load(open(os.path.join(_FIX, "summary.json")))
_TINY = json.load(
    open(
        os.path.join(_AXERA_DIR, "fixtures", "llm_build_dtype_analysis", "summary.json")
    )
)
_TAGS = sorted(_SUMMARY)


def _slice(tag):
    """(block bytes, checkpoint rows as float32, summary entry)."""
    return (
        _SLICES[f"{tag}_block"].tobytes(),
        ref.widen(_SLICES[f"{tag}_bf16"], "BF16"),
        _SUMMARY[tag],
    )


# --- layout -----------------------------------------------------------------


def test_column_blocks_fit_every_width_seen():
    assert lda.column_blocks(576) == [576]  # SmolLM2 hidden: one block, not 32 + 544
    assert lda.column_blocks(1024) == [480, 544]
    assert lda.column_blocks(1536) == [448, 544, 544]
    assert lda.column_blocks(2048) == [416, 544, 544, 544]
    assert lda.column_blocks(3072) == [352] + [544] * 5
    assert lda.column_blocks(4096) == [288] + [544] * 7
    assert [lda.column_blocks(c) for c in (256, 512, 544)] == [[256], [512], [544]]
    for cin in range(1, 5000, 7):
        blocks = lda.column_blocks(cin)
        assert sum(blocks) == cin and 0 < blocks[0] <= 576
        assert all(b == 544 for b in blocks[1:])
    for tag in _TAGS:
        assert rq.column_blocks(_SUMMARY[tag]["cin"]) == _SUMMARY[tag]["column_blocks"]


@pytest.mark.parametrize("tag", _TAGS)
def test_real_slice_decodes_to_the_plain_quantizer_of_the_checkpoint(tag):
    data, w, info = _slice(tag)
    got = rq.decode_block(data, info["cin"], info["row_block"])
    assert rq.detect_rule(got["q"], got["scale"]) == info["rule"]
    q, scale = rq.quantize_plain(w, info["rule"])
    np.testing.assert_array_equal(got["scale"], scale)
    bad = got["q"] != q
    assert bad.sum() == info["mismatched_codes"]
    lo, hi = rq.CODE_RANGE[info["rule"]]
    assert got["q"].min() == lo and got["q"].max() == hi
    # The only misses (rule A) are exact .5 ties the compiler rounded down.
    ratio = (w / scale[:, None])[bad]
    np.testing.assert_array_equal(ratio - np.floor(ratio), 0.5)
    np.testing.assert_array_equal(got["q"][bad], q[bad] - 1)


def test_fixture_covers_both_rules_both_row_blocks_and_a_tie():
    assert {_SUMMARY[t]["rule"] for t in _TAGS} == {"A", "B"}
    assert {_SUMMARY[t]["row_block"] for t in _TAGS} == {32, 64}
    assert _SUMMARY["smol_o"]["mismatched_codes"] == 1
    assert sum(_SUMMARY[t]["mismatched_codes"] for t in _TAGS) == 1


@pytest.mark.parametrize("tag", _TAGS)
def test_encode_of_decode_is_byte_identical(tag):
    data, w, info = _slice(tag)
    assert len(data) == rq.block_bytes(info["cin"], info["row_block"])
    got = rq.decode_block(data, info["cin"], info["row_block"])
    assert rq.encode_block(got["q"], got["scale"], info["row_block"]) == data
    if not info["mismatched_codes"]:
        assert rq.encode_block(*rq.quantize_plain(w, info["rule"])) == data


@pytest.mark.parametrize("tag", _TAGS)
def test_tail_is_row_sums_zeros_and_repeated_scales(tag):
    data, _, info = _slice(tag)
    rb, widths = info["row_block"], info["column_blocks"]
    got = rq.decode_block(data, info["cin"], rb)
    edges = np.cumsum([0] + widths)
    sums = [got["q"][:, a:b].sum(axis=1) for a, b in zip(edges, edges[1:])]
    np.testing.assert_array_equal(got["rowsum"], -np.stack(sums, axis=1) / 2)
    pos = 0
    for width, part_sum in zip(widths, sums):
        tail = data[pos + rb * 18 * -(-width // 36) : pos + rq.part_bytes(width, rb)]
        assert len(tail) == 12 * rb
        np.testing.assert_array_equal(
            np.frombuffer(tail[: 4 * rb], "<i4"), -part_sum * 32768
        )
        assert tail[4 * rb : 8 * rb] == bytes(4 * rb)
        # One scale per full row, repeated in every column part's tail.
        assert tail[8 * rb :] == got["scale"].astype("<f4").tobytes()
        pos += rq.part_bytes(width, rb)
    assert pos == len(data)


def test_64_row_unit_is_rows_2k_2k1_2k32_2k33():
    data, _, info = _slice("smol_v")
    q = rq.decode_block(data, 576, 64)["q"]

    def plane(row, chunk):
        c = q[row, 36 * chunk : 36 * chunk + 36] + 8
        return bytes((c[1::2] << 4 | c[0::2]).astype(np.uint8))

    chunks = 576 // 36
    for k, chunk in ((0, 0), (3, 5), (15, chunks - 1)):
        unit = data[72 * (chunks * k + chunk) :][:72]
        rows = (2 * k, 2 * k + 1, 2 * k + 32, 2 * k + 33)
        assert unit == b"".join(plane(r, chunk) for r in rows)
    # Read as two 32-row blocks the same bytes are not well formed.
    with pytest.raises(rq.LayoutError):
        rq.decode_block(data[: rq.block_bytes(576, 32)], 576, 32)


def test_32_row_blocks_are_the_dtype_analysis_layout():
    for inter, name in ((512, "self_attn.q_proj"), (2048, "mlp.down_proj")):
        w = lda.tiny_llama_weights(inter)[f"model.layers.0.{name}.weight"][:32]
        data = lda.encode_block(w, "s4")
        q, scale = rq.quantize_plain(w, "A")
        np.testing.assert_array_equal(q, lda.quantize_int(w, 4)[0])
        assert rq.encode_block(q, scale) == data
        got, want = (
            rq.decode_block(data, w.shape[1], 32),
            lda.decode_block(data, "s4", w.shape[1]),
        )
        for field in ("q", "scale", "rowsum"):
            np.testing.assert_array_equal(got[field], want[field])


def test_malformed_blocks_and_codes_raise():
    data, _, info = _slice("qwen3_q")
    cin = info["cin"]
    q, scale = (rq.decode_block(data, cin, 32)[k] for k in ("q", "scale"))
    body0 = 32 * 18 * -(-480 // 36)
    for at, why in (
        (body0, "row sum"),  # first row sum of the first part
        (body0 + 4 * 32 + 5, "non-zero"),  # the zero run
        (body0 - 1, "pad nibble"),  # last byte of the body: columns 502, 503 are pad
        (len(data) - 1, "scale"),  # the second part's copy of a scale
    ):
        bad = bytearray(data)
        bad[at] ^= 0x10
        assert why in rq.block_problem(bytes(bad), cin, 32)
        with pytest.raises(rq.LayoutError, match=why):
            rq.decode_block(bytes(bad), cin, 32)
    with pytest.raises(rq.LayoutError):
        rq.decode_block(data[:-4], cin, 32)
    with pytest.raises(rq.LayoutError):
        rq.decode_block(data, cin, 48)
    with pytest.raises(ValueError):
        rq.encode_block(q + 1, scale)  # 8 does not fit a nibble code
    with pytest.raises(ValueError):
        rq.encode_block(q, np.zeros(32, np.float32))
    with pytest.raises(ValueError):
        rq.encode_block(q[:16], scale[:16])
    with pytest.raises(ValueError):
        rq.encode_block(q.astype(np.float32), scale)
    with pytest.raises(ValueError):
        rq.quantize_plain(np.ones((2, 4), np.float32), "C")


# --- quantizers -------------------------------------------------------------


def _linear_problem(seed=0, rows=24, cin=96, tokens=400):
    rng = np.random.RandomState(seed)
    w = (rng.randn(rows, cin) * 0.05).astype(np.float32)
    # Correlated inputs with a few dominant directions, like a hidden state.
    mix = rng.randn(cin, cin) * (rng.rand(cin) ** 4 + 0.02)
    x = (rng.randn(tokens, cin) @ mix).astype(np.float32)
    return w, x, rq.gram([x])


@pytest.mark.parametrize("rule", ["A", "B"])
def test_gptq_lowers_the_h_weighted_error_and_stays_in_range(rule):
    w, x, h = _linear_problem()
    q0, s0 = rq.quantize_plain(w, rule)
    q, s = rq.gptq(w, h, rule)
    lo, hi = rq.CODE_RANGE[rule]
    assert q.dtype == np.int8 and s.dtype == np.float32
    assert q.min() >= lo and q.max() <= hi
    np.testing.assert_array_equal(np.sign(s), np.sign(s0))  # the build's convention
    w64 = w.astype(np.float64)
    plain = rq.h_error(w64, rq.dequantize(q0, s0).astype(np.float64), h)
    new = rq.h_error(w64, rq.dequantize(q, s).astype(np.float64), h)
    assert new.sum() < 0.5 * plain.sum()
    assert (new < plain).mean() > 0.9
    # The same statement on the outputs themselves.
    out_err = np.square(x @ (w - rq.dequantize(q, s)).T).sum()
    assert out_err < 0.5 * np.square(x @ (w - rq.dequantize(q0, s0)).T).sum()
    # The result survives the byte format.
    block = rq.encode_block(
        q.astype(np.int64)[:, :64].repeat(2, 0)[:32], s.repeat(2)[:32]
    )
    np.testing.assert_array_equal(
        rq.decode_block(block, 64, 32)["q"], q[:, :64].repeat(2, 0)[:32]
    )


def test_detect_rule_refuses_codes_that_are_not_the_compilers():
    w = _linear_problem()[0]
    qa, sa = rq.quantize_plain(w, "A")
    qb, sb = rq.quantize_plain(w, "B")
    assert rq.detect_rule(qa, sa) == "A" and rq.detect_rule(qb, sb) == "B"
    # Rows that neither peak at -8 nor reach +-7 with a positive scale.
    with pytest.raises(rq.LayoutError):
        rq.detect_rule(np.clip(qa, -6, 6), sa)
    with pytest.raises(rq.LayoutError):
        rq.detect_rule(qb, -sb)


def test_retarget_is_the_identity_when_both_paths_agree():
    w, x, h = _linear_problem(1)
    np.testing.assert_allclose(rq.retarget(w, h, h, 0.3), w, rtol=1e-4, atol=1e-6)
    # With a perturbed quantized-path input it moves the output towards the float one.
    xq = (x + 0.05 * np.random.RandomState(2).randn(*x.shape) * x.std()).astype(
        np.float32
    )
    w2 = rq.retarget(w, rq.cross_gram([x], [xq]), rq.gram([xq]), 0.3)
    assert np.square(x @ w.T - xq @ w2.T).sum() < np.square(x @ w.T - xq @ w.T).sum()


# --- a synthetic compiled directory -----------------------------------------

_LAYER = "llama_p64_l0_together.axmodel"
_POST = "llama_post.axmodel"
_STEMS = {key: f"model.layers.0.{stem}.weight" for key, stem in rq.PROJECTIONS.items()}


def _axmodel(path, initializers):
    """A stand-in for an .axmodel: the patcher only reads the initializers' bytes."""
    model = parser.parse_model(
        '<ir_version: 8, opset_import: ["" : 13]>'
        " g (float[1] x) => (float[1] y) { y = Identity(x) }"
    )
    for name, data in initializers.items():
        # Raw byte blobs (a megabyte-scale weight table, MCode) are attached after
        # parsing; they are not tensor literals.
        t = model.graph.initializer.add()
        t.name, t.data_type, t.raw_data = name, onnx.TensorProto.UINT8, bytes(data)
        t.dims.append(len(data))
    onnx.save(model, path)


def _tiny_params(weights):
    """``npu_params`` of the tiny Llama's s4 build: every weight block at the offset the
    real build has, in between the same kind of constants."""
    params = np.zeros(_TINY["npu_params_bytes"]["s4"], np.uint8)
    spans = []
    for name, offsets in _TINY["offsets"]["s4"].items():
        blocks = lda.encode_matrix(weights[f"model.layers.0.{name}.weight"], "s4")
        for o, block in zip(offsets, blocks, strict=True):
            params[o : o + len(block)] = np.frombuffer(block, np.uint8)
            spans.append((o, o + len(block)))
    spans.sort()
    gaps = [(a, b) for a, b in zip([0] + [e for _, e in spans], [s for s, _ in spans])]
    gaps = [g for g in gaps if g[0] < g[1]] + [(spans[-1][1], len(params))]
    rng = np.random.RandomState(5)
    for a, b in gaps:  # 1/sqrt(head_dim) tables, a RoPE-like table
        fill = np.cos(rng.rand((b - a) // 4) * 6.0).astype("<f4")
        params[a : a + 4 * len(fill)] = np.frombuffer(fill.tobytes(), np.uint8)
    return params.tobytes(), spans


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("int4")
    weights = lda.tiny_llama_weights(512)
    ckpt, src = str(root / "checkpoint"), str(root / "out_s4")
    lda.write_checkpoint(ckpt, weights)
    os.makedirs(src)
    params, spans = _tiny_params(weights)
    rng = np.random.RandomState(9)
    mcode = {
        f"mcode_{i}": rng.randint(0, 256, 4096 + i, dtype=np.uint8) for i in (0, 1)
    }
    _axmodel(os.path.join(src, _LAYER), {"npu_params": params, **mcode})
    _axmodel(os.path.join(src, _POST), {"npu_params": rng.bytes(2048)})
    calib = np.random.RandomState(4).randint(0, 512, (12, 64)).tolist()
    return {
        "root": root,
        "ckpt": ckpt,
        "src": src,
        "weights": weights,
        "params": params,
        "spans": spans,
        "calib": calib,
    }


def _read(directory, name=_LAYER):
    with open(os.path.join(directory, name), "rb") as f:
        return f.read()


def test_discover_layout_finds_the_real_builds_offsets(tiny):
    model = ref.Model.load(tiny["ckpt"])
    layout = rq.discover_layout(tiny["params"], rq.projection_shapes(model))
    assert list(layout) == list(rq.PROJECTIONS)
    for key, t in layout.items():
        assert t.row_block == 32
        assert list(t.offsets) == _TINY["offsets"]["s4"][rq.PROJECTIONS[key]]
    # k_proj's two blocks straddle the RoPE table: the second one is found past it.
    assert layout["k"].offsets[1] - layout["k"].offsets[0] > layout["k"].block_bytes
    assert sorted(r for t in layout.values() for r in t.ranges) == tiny["spans"]
    codes = {k: rq.decode_tensor(tiny["params"], t) for k, t in layout.items()}
    weights = {k: tiny["weights"][_STEMS[k]] for k in rq.PROJECTIONS}
    assert rq.check_against_checkpoint(codes, weights) == ("A", 0)


@pytest.mark.parametrize("method", ["identity", "plain"])
def test_requantize_directory_reproduces_the_compiled_bytes(tiny, method):
    dst = str(tiny["root"] / f"dst_{method}")
    report = rq.requantize_directory(tiny["src"], tiny["ckpt"], dst, method)
    assert report["ok"] and report["rule"] == "A" and report["copied"] == [_POST]
    assert sorted(os.listdir(dst)) == [_LAYER, _POST]
    assert _read(dst) == _read(tiny["src"])
    assert _read(dst, _POST) == _read(tiny["src"], _POST)
    assert report["layers"][0]["bytes_changed"] == 0


@pytest.mark.parametrize("method", ["gptq", "gptqa"])
def test_requantize_directory_rewrites_only_weight_block_bytes(tiny, method):
    dst = str(tiny["root"] / f"dst_{method}")
    report = rq.requantize_directory(
        tiny["src"], tiny["ckpt"], dst, method, tiny["calib"]
    )
    (entry,) = report["layers"]
    assert report["ok"] and entry["ok"] and report["calibration_tokens"] == 12 * 64
    assert entry["changed_outside_weight_blocks"] == 0
    assert entry["bytes_rewritable"] == sum(b - a for a, b in tiny["spans"])
    assert entry["row_blocks"] == dict.fromkeys(rq.PROJECTIONS, 32)
    assert _read(dst, _POST) == _read(tiny["src"], _POST)

    # The same guarantees, re-derived here from the files alone.
    old, new = _read(tiny["src"]), _read(dst)
    assert len(old) == len(new) and old != new
    base = old.find(tiny["params"])
    inside = np.zeros(len(old), bool)
    for a, b in tiny["spans"]:
        inside[base + a : base + b] = True
    diff = np.frombuffer(old, np.uint8) != np.frombuffer(new, np.uint8)
    assert diff.sum() == entry["bytes_changed"] > 1000 and not (diff & ~inside).any()
    inits = {
        i.name: i.raw_data
        for i in onnx.load(os.path.join(dst, _LAYER)).graph.initializer
    }
    src_inits = {
        i.name: i.raw_data
        for i in onnx.load(os.path.join(tiny["src"], _LAYER)).graph.initializer
    }
    assert list(inits) == list(src_inits)
    assert all(inits[k] == src_inits[k] for k in inits if k.startswith("mcode"))

    # The new codes are well formed, in the build's range, and better on the
    # calibration inputs than the compiler's.
    model = ref.Model.load(tiny["ckpt"])
    codes = rq.read_layer_codes(os.path.join(dst, _LAYER), model)
    for key, (q, s) in codes.items():
        assert q.min() >= -8 and q.max() <= 7 and np.isfinite(s).all()
        err = entry["relative_output_error"][key]
        if method == "gptq":
            assert err["gptq"] < err["reference"]
    if method == "gptqa":  # its target is the float path's output, not W's
        down = entry["relative_output_error"]["down"]
        assert 0 < down["gptqa"] < 1
    # A re-quantized directory is not the compiler's any more: refuse to redo it.
    with pytest.raises(rq.LayoutError):
        rq.requantize_directory(dst, tiny["ckpt"], dst + "_again", "plain")
    assert not os.path.exists(dst + "_again")


def test_calibration_path_is_the_reference_forward(tiny):
    model = ref.Model.load(tiny["ckpt"])
    quantizer = rq.SequentialQuantizer(model, "A", "plain")
    codes, _ = quantizer.layer(0)
    emu = rq.emulator(model, [codes], "A")
    name = _STEMS["q"]
    np.testing.assert_array_equal(emu.weight(name), rq.dequantize(*codes["q"]))
    head = model.weight("lm_head.weight")
    step = np.abs(head).max(axis=1, keepdims=True) / 128
    err = np.abs(emu.weight("lm_head.weight") - head)
    # s8: half a step, or up to a whole one where the side opposite the peak clips.
    assert err.max() > 0 and (err <= step * 1.001).all()
    assert (err <= step / 2 * 1.001).mean() > 0.999
    # One layer of the calibration stream equals the emulator's own forward.
    ids = tiny["calib"][0]
    cal = rq.Calibration(model, [ids])
    lw = emu.layer_weights(0)
    a = cal.norm1(lw)
    att = cal.attn(lw, a)
    x1, m = cal.after_o(lw, att)
    cal.finish(lw, x1, cal.act(lw, m))
    want = emu.forward_full(ids)[1][1]
    assert np.abs(cal.x[0] - want).max() < 0.02 * np.abs(want).max()
    np.testing.assert_array_equal(cal.x[0], loop.bf16_round(cal.x[0]))
    with pytest.raises(ValueError):
        quantizer.layer(0)  # layers go in order
    with pytest.raises(ValueError):
        rq.emulator(model, [], "A")


def test_unsupported_inputs_raise(tiny):
    root, src, ckpt = tiny["root"], tiny["src"], tiny["ckpt"]
    with pytest.raises(ValueError, match="method"):
        rq.requantize_directory(src, ckpt, str(root / "x0"), "awq")
    with pytest.raises(ValueError, match="calibration"):
        rq.requantize_directory(src, ckpt, str(root / "x1"), "gptq")
    with pytest.raises(ValueError, match="vocabulary"):
        rq.requantize_directory(src, ckpt, str(root / "x2"), "gptq", [[1, 2, 512]])
    with pytest.raises(ValueError):
        rq.requantize_directory(src, ckpt, src, "plain")
    full = root / "full"
    full.mkdir()
    (full / "something").write_text("x")
    with pytest.raises(FileExistsError):
        rq.requantize_directory(src, ckpt, str(full), "plain")

    # Another checkpoint of the same shape: the layout fits, the scales do not.
    other = str(root / "other_checkpoint")
    lda.write_checkpoint(other, lda.tiny_llama_weights(512, seed=4))
    with pytest.raises(rq.LayoutError, match="checkpoint"):
        rq.requantize_directory(src, other, str(root / "x3"), "plain")
    # A checkpoint of another shape: nothing can be placed.
    wide = str(root / "wide_checkpoint")
    lda.write_checkpoint(wide, lda.tiny_llama_weights(2048))
    with pytest.raises(rq.LayoutError, match="gate_proj"):
        rq.requantize_directory(src, wide, str(root / "x4"), "plain")

    # A weight table that is not s4 (here: one flipped row-sum byte, then none at all).
    model = ref.Model.load(ckpt)
    shapes = rq.projection_shapes(model)
    broken = bytearray(tiny["params"])
    last_block_end = tiny["spans"][-1][1]
    broken[last_block_end - 12 * 32] ^= 1
    with pytest.raises(rq.LayoutError, match="down_proj"):
        rq.discover_layout(bytes(broken), shapes)
    with pytest.raises(rq.LayoutError, match="v_proj"):
        rq.discover_layout(bytes(len(tiny["params"])), shapes)
    bad_dir = root / "bad_src"
    bad_dir.mkdir()
    _axmodel(str(bad_dir / _LAYER), {"mcode_0": b"abc"})
    _axmodel(str(bad_dir / _POST), {"npu_params": b"abc"})
    with pytest.raises(rq.LayoutError, match="npu_params"):
        rq.requantize_directory(str(bad_dir), ckpt, str(root / "x5"), "plain")
    for name in ("x0", "x1", "x2", "x3", "x4", "x5"):  # nothing half-written
        assert not (root / name).exists()


def test_load_calibration_reads_token_id_json(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps([[1, 2, 3], [], [4, 5]]))
    b.write_text(json.dumps([{"kind": "chat", "ids": [7, 8]}]))
    got = rq.load_calibration([str(a), str(b)], str(tmp_path))
    assert got == [[1, 2, 3], [4, 5], [7, 8]]
    a.write_text(json.dumps([["not", "ids"]]))
    with pytest.raises(ValueError):
        rq.load_calibration([str(a)], str(tmp_path))
