"""Quark's ``MATMUL_NBITS`` preset (onnxsim.quark_matmul_nbits and the
``quark_compat`` preset). No AMD Quark needed; the Quark parity side is the last
section of tests/test_quark_parity.py."""

import warnings

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_compat as qc
from onnxsim import quark_matmul_nbits as nb


def _model(body, shapes, inputs="float[3,64] x", outputs="float[3,16] y", seed=0):
    model = parser.parse_model(
        f'<ir_version: 9, opset_import: ["": 17]> g ({inputs}) => ({outputs}) '
        f"{{ {body} }}"
    )
    rng = np.random.default_rng(seed)
    model.graph.initializer.extend(
        numpy_helper.from_array((rng.standard_normal(s) * 0.5).astype(np.float32), n)
        for n, s in shapes
    )
    return model


def _mlp():
    return _model(
        "h = MatMul(x, w1)\n h2 = Relu(h)\n y = MatMul(h2, w2)",
        [("w1", (64, 96)), ("w2", (96, 16))],
    )


def _run(model, x):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"x": x})[0]


def _inits(model):
    return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}


def _attrs(node):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _rtn_reference(w, block, sym):
    """Slow, independent float64 round-to-nearest of ``[K, N]`` weights in blocks
    along ``K``: the dequantized weights."""
    k, n = w.shape
    out = np.zeros_like(w, dtype=np.float64)
    for j in range(n):
        for k0 in range(0, k, block):
            v = w[k0 : k0 + block, j].astype(np.float64)
            if sym:
                ext = v[np.argmax(np.abs(v))]
                scale = ext / -8.0 if ext else 1.0
                q = np.clip(np.floor(v / scale + 8.5), 0, 15)
                out[k0 : k0 + block, j] = scale * (q - 8)
            else:
                lo, hi = min(v.min(), 0.0), max(v.max(), 0.0)
                scale = (hi - lo) / 15 or 1.0
                zp = np.clip(np.floor(-lo / scale + 0.5), 0, 15)
                q = np.clip(np.floor(v / scale + zp + 0.5), 0, 15)
                out[k0 : k0 + block, j] = scale * (q - zp)
    return out


# -- packing and the block quantizer ---------------------------------------------


def test_pack_unpack_roundtrip_and_nibble_order():
    codes = np.random.default_rng(0).integers(0, 16, (5, 3, 32)).astype(np.uint8)
    packed = nb.pack_int4(codes)
    assert packed.shape == (5, 3, 16) and packed.dtype == np.uint8
    np.testing.assert_array_equal(nb.unpack_int4(packed), codes)
    # even index in the low nibble, odd in the high one (MatMulNBits layout)
    assert nb.pack_int4(np.array([[0x3, 0xA]], np.uint8))[0, 0] == 0xA3
    with pytest.raises(ValueError):
        nb.pack_int4(np.zeros((2, 3), np.uint8))


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("k, block", [(64, 32), (100, 32), (300, 128), (48, 16)])
def test_block_quantizer_layout_and_error_bounds(k, block, sym):
    w = (np.random.default_rng(k).standard_normal((k, 24)) * 0.7).astype(np.float32)
    packed, scales, zps = nb.block_quantize_int4(w, block, sym)
    blocks = -(-k // block)
    assert packed.shape == (24, blocks, block // 2) and packed.dtype == np.uint8
    assert scales.shape == (24 * blocks,) and scales.dtype == np.float32
    if sym:
        assert zps is None
    else:
        assert zps is not None and zps.shape == (24 * ((blocks + 1) // 2),)
    deq = nb.dequantize_int4(packed, scales, zps, k, 24, block)
    # error: half a step (asymmetric) / one step (symmetric: the extreme of the
    # larger sign can land one code outside [0, 15] and is clipped)
    step = np.repeat(np.abs(scales.reshape(24, blocks)), block, axis=1)[:, :k].T
    bound = step * (1.0 if sym else 0.5) * (1 + 1e-5) + 1e-7
    assert np.all(np.abs(deq - w) <= bound)
    # the same numbers as an independent float64 round-to-nearest
    ref = _rtn_reference(w, block, sym)
    assert np.mean(np.abs(deq - ref) > 1e-5) < 0.002  # float32 vs float64 ties only
    assert np.max(np.abs(deq - ref)) <= step.max() * 1.0001


def test_block_quantizer_edge_cases():
    w = np.zeros((32, 4), np.float32)
    w[:, 1] = 0.25
    w[:, 2] = -1.0
    for sym in (True, False):
        packed, scales, zps = nb.block_quantize_int4(w, 32, sym)
        deq = nb.dequantize_int4(packed, scales, zps, 32, 4, 32)
        np.testing.assert_allclose(deq, w, atol=1e-6)  # constants are exact
        assert np.all(np.isfinite(scales))
    with pytest.raises(ValueError):
        nb.block_quantize_int4(w, 24, True)  # not a power of two
    with pytest.raises(ValueError):
        nb.block_quantize_int4(w, 8, True)  # below the kernel minimum


def test_zero_point_padding_nibble_is_eight():
    w = np.random.default_rng(1).standard_normal((32, 3)).astype(np.float32)
    _, _, zps = nb.block_quantize_int4(w, 32, False)  # one block: odd count
    assert np.all(zps >> 4 == 8)


@pytest.mark.parametrize("sym", [True, False])
def test_block_quantizer_is_bit_identical_to_onnxruntime(sym):
    """ORT's own MLAS block quantizer, which Quark calls, on random and on
    tie-heavy (values on a coarse grid) weights."""
    pybind = pytest.importorskip("onnxruntime.capi._pybind_state")
    if not hasattr(pybind, "quantize_matmul_4bits"):
        pytest.skip("this ONNX Runtime has no quantize_matmul_4bits")
    rng = np.random.default_rng(3)
    for trial in range(12):
        block = int(rng.choice([16, 32, 64]))
        k, n = int(rng.integers(20, 200)), 7
        if trial % 2:
            w = (rng.integers(-16, 17, (k, n)) * 0.125).astype(np.float32)
        else:
            w = (rng.standard_normal((k, n)) * rng.choice([0.01, 1, 30])).astype(
                np.float32
            )
        blocks = -(-k // block)
        padded = np.pad(w, ((0, blocks * block - k), (0, 0)))
        packed = np.zeros((n, blocks, block // 2), np.uint8)
        scales = np.zeros(n * blocks, np.float32)
        zp = np.zeros(n * ((blocks + 1) // 2), np.uint8)
        pybind.quantize_matmul_4bits(packed, padded, scales, zp, block, n, k, sym)
        mp, ms, mz = nb.block_quantize_int4(w, block, sym)
        np.testing.assert_array_equal(mp, packed)
        np.testing.assert_array_equal(ms, scales)
        if not sym:
            np.testing.assert_array_equal(mz, zp)


# -- the graph rewrite ------------------------------------------------------------


def test_emission_matches_the_documented_layout():
    model = _mlp()
    model.graph.node[0].name = "fc1"
    out, rep = nb.quantize_matmul_nbits(
        model, group_size=32, accuracy_level=1, symmetric=True
    )
    nodes = list(out.graph.node)
    assert [n.op_type for n in nodes] == ["MatMulNBits", "Relu", "MatMulNBits"]
    assert [n.name for n in nodes] == ["fc1_Q4", "", ""]  # unnamed stays unnamed
    assert rep.converted == ["fc1_Q4", ""] and rep.skipped == []
    first = nodes[0]
    assert first.domain == "com.microsoft"
    assert list(first.input) == ["x", "w1_Q4", "w1_scales"]
    assert list(first.output) == ["h"]
    assert _attrs(first) == {
        "K": 64,
        "N": 96,
        "bits": 4,
        "block_size": 32,
        "accuracy_level": 1,
    }
    inits = _inits(out)
    assert set(inits) == {"w1_Q4", "w1_scales", "w2_Q4", "w2_scales"}  # floats gone
    assert inits["w1_Q4"].shape == (96, 2, 16) and inits["w1_Q4"].dtype == np.uint8
    assert inits["w1_scales"].shape == (192,)
    assert ("com.microsoft", 1) in {(o.domain, o.version) for o in out.opset_import}
    assert len(model.graph.initializer) == 2  # input untouched
    assert model.graph.node[0].op_type == "MatMul"
    onnx.checker.check_model(out)


def test_asymmetric_emits_packed_zero_points():
    out, _ = nb.quantize_matmul_nbits(_mlp(), group_size=32, symmetric=False)
    node = out.graph.node[0]
    assert list(node.input) == ["x", "w1_Q4", "w1_scales", "w1_zero_points"]
    assert _inits(out)["w1_zero_points"].shape == (96,)  # 96 columns * ceil(2 / 2)


def test_selection_rules():
    model = _model(
        "a = MatMul(x, w)\n"
        "b = MatMul(a, w3)\n"  # 3-D weight: skipped
        "c = MatMul(b, w3)\n"
        "d = Gemm(x, g)\n"  # Gemm: never converted
        "e = MatMul(ca, x2)\n"  # constant A: skipped
        "y = Identity(a)",
        [("w", (64, 16)), ("w3", (2, 16, 16)), ("g", (64, 16)), ("ca", (3, 64))],
        inputs="float[3,64] x, float[64,16] x2",
    )
    out, rep = nb.quantize_matmul_nbits(model, group_size=32)
    assert [n.op_type for n in out.graph.node] == [
        "MatMulNBits",
        "MatMul",
        "MatMul",
        "Gemm",
        "MatMul",
        "Identity",
    ]
    assert len(rep.converted) == 1 and len(rep.skipped) == 3


def test_exclude_keeps_node_and_weight():
    model = _mlp()
    model.graph.node[2].name = "head"
    out, rep = nb.quantize_matmul_nbits(model, group_size=32, exclude_nodes=["head"])
    assert [n.op_type for n in out.graph.node] == ["MatMulNBits", "Relu", "MatMul"]
    assert "w2" in _inits(out) and "w1" not in _inits(out)
    assert len(rep.converted) == 1


def test_shared_weight_converted_once():
    model = _model(
        "h = MatMul(x, w)\n h2 = Relu(h)\n y = MatMul(h2, w)",
        [("w", (64, 64))],
        outputs="float[3,64] y",
    )
    out, _ = nb.quantize_matmul_nbits(model, group_size=32)
    assert [n.op_type for n in out.graph.node] == ["MatMulNBits", "Relu", "MatMulNBits"]
    assert sorted(_inits(out)) == ["w_Q4", "w_scales"]
    assert out.graph.node[0].input[1] == out.graph.node[2].input[1]
    x = np.random.default_rng(0).standard_normal((3, 64)).astype(np.float32)
    assert _run(out, x).shape == (3, 64)


def test_weight_also_used_elsewhere_is_kept():
    model = _model(
        "h = MatMul(x, w)\n t = Transpose<perm=[1,0]>(w)\n y = Identity(h)",
        [("w", (64, 16))],
    )
    out, _ = nb.quantize_matmul_nbits(model, group_size=32)
    assert "w" in _inits(out) and "w_Q4" in _inits(out)


def test_subgraph_matmul_uses_outer_weight():
    model = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[3,64] x, bool c) => (float[3,16] y) {
            y = If<then_branch = t () => (float[3,16] r) { r = MatMul(x, w) },
                   else_branch = e () => (float[3,16] r2) { r2 = MatMul(x, w) }>(c)
        }
        """
    )
    w = (np.random.default_rng(0).standard_normal((64, 16)) * 0.5).astype(np.float32)
    model.graph.initializer.append(numpy_helper.from_array(w, "w"))
    out, rep = nb.quantize_matmul_nbits(model, group_size=32)
    assert len(rep.converted) == 2
    branches = [a.g for a in out.graph.node[0].attribute]
    assert all(b.node[0].op_type == "MatMulNBits" for b in branches)
    assert sorted(_inits(out)) == ["w_Q4", "w_scales"]
    sess = ort.InferenceSession(
        out.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    x = np.random.default_rng(1).standard_normal((3, 64)).astype(np.float32)
    got = sess.run(None, {"x": x, "c": np.array(True)})[0]
    assert np.linalg.norm(got - x @ w) / np.linalg.norm(x @ w) < 0.2


def test_unsupported_options_raise():
    with pytest.raises(NotImplementedError, match="4-bit"):
        nb.quantize_matmul_nbits(_mlp(), bits=8)
    with pytest.raises(ValueError, match="power of two"):
        nb.quantize_matmul_nbits(_mlp(), group_size=48)
    with pytest.raises(ValueError, match="unknown"):
        nb.quantize_matmul_nbits(_mlp(), algorithm="AWQ")


# -- numerics under ONNX Runtime --------------------------------------------------


@pytest.mark.parametrize("algorithm", ["DEFAULT", "HQQ"])
@pytest.mark.parametrize("sym", [True, False])
def test_outputs_equal_dequantized_weights_under_ort(algorithm, sym):
    """MatMulNBits (ORT 1.30, CPU) computes x @ dequantize(weights)."""
    model = _mlp()
    out, _ = nb.quantize_matmul_nbits(
        model, group_size=32, symmetric=sym, algorithm=algorithm
    )
    x = np.random.default_rng(5).standard_normal((3, 64)).astype(np.float32)
    inits = {t.name: t for t in out.graph.initializer}
    deq = {}
    for node in out.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        a = _attrs(node)
        packed, scales = (numpy_helper.to_array(inits[i]) for i in node.input[1:3])
        zp = (
            numpy_helper.to_array(inits[node.input[3]]) if len(node.input) > 3 else None
        )
        if algorithm == "HQQ":
            blocks = -(-a["K"] // 32)
            codes = nb.unpack_int4(packed).reshape(a["N"], -1)[:, : a["K"]]
            s = scales.reshape(a["N"], blocks).repeat(32, axis=1)[:, : a["K"]]
            z = zp.reshape(a["N"], blocks).repeat(32, axis=1)[:, : a["K"]]
            deq[node.output[0]] = (s * (codes - z)).T.astype(np.float32)
        else:
            deq[node.output[0]] = nb.dequantize_int4(
                packed, scales, zp, a["K"], a["N"], a["block_size"]
            )
    ref = np.maximum(x @ deq["h"], 0) @ deq["y"]
    np.testing.assert_allclose(_run(out, x), ref, rtol=2e-3, atol=2e-3)
    exact = _run(model, x)
    assert np.linalg.norm(_run(out, x) - exact) / np.linalg.norm(exact) < 0.25


def test_smaller_blocks_are_more_accurate():
    model = _mlp()
    x = np.random.default_rng(2).standard_normal((3, 64)).astype(np.float32)
    exact = _run(model, x)
    errs = []
    for block in (16, 64):
        out, _ = nb.quantize_matmul_nbits(model, group_size=block, symmetric=False)
        errs.append(np.linalg.norm(_run(out, x) - exact))
    assert errs[0] < errs[1]


def test_float16_weights_use_float16_scales():
    model = _model("y = MatMul(x, w)", [("w", (64, 16))], inputs="float[3,64] x")
    w = numpy_helper.to_array(model.graph.initializer[0]).astype(np.float16)
    model.graph.ClearField("initializer")
    model.graph.initializer.append(numpy_helper.from_array(w, "w"))
    out, _ = nb.quantize_matmul_nbits(model, group_size=32)
    assert _inits(out)["w_scales"].dtype == np.float16


# -- GPTQ ---------------------------------------------------------------------------


def _gptq_setup():
    rng = np.random.default_rng(0)
    model = _model(
        "y = MatMul(x, w)",
        [("w", (128, 48))],
        inputs="float[64,128] x",
        outputs="float[64,48] y",
    )
    base = rng.standard_normal((64, 12)).astype(np.float32)
    x = base @ rng.standard_normal((12, 128)).astype(np.float32)
    x += 0.1 * rng.standard_normal((64, 128)).astype(np.float32)
    return model, x


def test_gptq_layout_defaults():
    model, x = _gptq_setup()
    out, _ = nb.quantize_matmul_nbits(
        model, algorithm="GPTQ", calibration=[{"x": x}], group_size=32
    )
    node = out.graph.node[0]
    # Quark: GPTQParams.GroupSize (-1 -> one block of K), no accuracy_level,
    # symmetric (no zero points), 2-D scales
    assert _attrs(node) == {"K": 128, "N": 48, "bits": 4, "block_size": 128}
    assert list(node.input) == ["x", "w_Q4", "w_scales"]
    inits = _inits(out)
    assert inits["w_Q4"].shape == (48, 1, 64) and inits["w_scales"].shape == (48, 1)


def test_gptq_grouped_asymmetric_layout():
    model, x = _gptq_setup()
    out, _ = nb.quantize_matmul_nbits(
        model,
        algorithm="GPTQ",
        calibration=[{"x": x}],
        gptq_params={"GroupSize": 32, "WeightSymmetric": False, "PerChannel": True},
    )
    inits = _inits(out)
    assert list(out.graph.node[0].input)[-1] == "w_zero_points"
    assert inits["w_Q4"].shape == (48, 4, 16)
    assert inits["w_scales"].shape == (48, 4)
    assert inits["w_zero_points"].shape == (48, 2)
    assert (
        float(
            np.linalg.norm(_run(out, x) - x @ _w(model)) / np.linalg.norm(x @ _w(model))
        )
        < 0.3
    )


def _w(model):
    return numpy_helper.to_array(model.graph.initializer[0])


def test_gptq_compensation_beats_round_to_nearest():
    model, x = _gptq_setup()
    ref = x @ _w(model)
    errs = {}
    for comp in (False, True):
        out, _ = nb.quantize_matmul_nbits(
            model,
            algorithm="GPTQ",
            calibration=[{"x": x}],
            gptq_params={"GroupSize": 32, "PerChannel": True, "Compensate": comp},
        )
        errs[comp] = np.linalg.norm(_run(out, x) - ref)
    assert errs[True] < errs[False]


def test_gptq_needs_calibration_and_unshared_weights():
    model, x = _gptq_setup()
    with pytest.raises(ValueError, match="calibration"):
        nb.quantize_matmul_nbits(model, algorithm="GPTQ")
    shared = _model(
        "h = MatMul(x, w)\n y = MatMul(h, w)",
        [("w", (64, 64))],
        outputs="float[3,64] y",
    )
    with pytest.raises(NotImplementedError, match="shared"):
        nb.quantize_matmul_nbits(
            shared,
            algorithm="GPTQ",
            calibration=[{"x": np.zeros((3, 64), np.float32)}],
        )


# -- the compat layer ----------------------------------------------------------------


def _quantize(cfg, model, reader=None, **kw):
    if reader is None:
        x = np.random.default_rng(1).standard_normal((3, 64)).astype(np.float32)
        reader = [{"x": x}]
    q = qc.ModelQuantizer(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = q.quantize_model(model, calibration_data_reader=reader, **kw)
    return q, out


def test_preset_is_registered_with_quarks_parameters():
    cfg = qc.QConfig.get_default_config("MATMUL_NBITS")
    assert cfg.extra_options["UseMatMulNBits"] is True
    assert cfg.extra_options["MatMulNBitsParams"] == {
        "GroupSize": 128,
        "Symmetric": True,
        "Bits": 4,
        "AccuracyLevel": 1,
    }
    assert "MATMUL_NBITS_ADAROUND" not in qc._PRESETS
    # a fresh dict per call: editing one config does not leak into the next
    cfg.extra_options["MatMulNBitsParams"]["GroupSize"] = 32
    assert (
        qc.QConfig.get_default_config("MATMUL_NBITS").extra_options[
            "MatMulNBitsParams"
        ]["GroupSize"]
        == 128
    )


def test_preset_end_to_end_and_options(tmp_path):
    model = _mlp()
    cfg = qc.QConfig.get_default_config("MATMUL_NBITS")
    q, out = _quantize(cfg, model, model_output=str(tmp_path / "o.onnx"))
    assert [n.op_type for n in out.graph.node] == ["MatMulNBits", "Relu", "MatMulNBits"]
    assert _attrs(out.graph.node[0])["block_size"] == 128
    assert _attrs(out.graph.node[0])["accuracy_level"] == 1
    assert onnx.load(str(tmp_path / "o.onnx")).graph.node[0].op_type == "MatMulNBits"
    assert len(q.last_matmul_nbits.converted) == 2

    cfg = qc.QConfig.get_default_config("MATMUL_NBITS")
    cfg.extra_options["MatMulNBitsParams"].update(
        GroupSize=32, Symmetric=False, AccuracyLevel=4
    )
    _, out = _quantize(cfg, model)
    assert _attrs(out.graph.node[0])["accuracy_level"] == 4
    assert len(out.graph.node[0].input) == 4
    x = np.random.default_rng(5).standard_normal((3, 64)).astype(np.float32)
    exact = _run(model, x)
    assert np.linalg.norm(_run(out, x) - exact) / np.linalg.norm(exact) < 0.25


def test_preset_algorithms_and_algo_config():
    model, x = _gptq_setup()
    cfg = qc.QConfig.get_default_config("MATMUL_NBITS")
    cfg.extra_options["MatMulNBitsParams"]["Algorithm"] = "HQQ"
    cfg.extra_options["MatMulNBitsParams"]["GroupSize"] = 32
    _, out = _quantize(cfg, model)
    assert "accuracy_level" not in _attrs(out.graph.node[0])  # HQQ never writes it

    # A GPTQConfig fills GPTQParams but, as in Quark, only Algorithm=GPTQ runs it
    cfg = qc.QConfig.get_default_config("MATMUL_NBITS")
    cfg.algo_config = [qc.GPTQConfig(group_size=32)]
    _, out = _quantize(cfg, model)
    assert _attrs(out.graph.node[0])["block_size"] == 128  # DEFAULT, not GPTQ

    cfg = qc.QConfig.get_default_config("MATMUL_NBITS")
    cfg.extra_options["MatMulNBitsParams"]["Algorithm"] = "GPTQ"
    cfg.algo_config = [qc.GPTQConfig(group_size=32, per_channel=True)]
    q = qc.ModelQuantizer(cfg)
    with pytest.warns(UserWarning, match="GPTQConfig is not read"):
        out = q.quantize_model(model, calibration_data_reader=[{"x": x}])
    assert _inits(out)["w_scales"].shape == (48, 1)  # ignored: ungrouped defaults

    # GPTQParams is what configures it
    cfg = qc.QConfig.get_default_config("MATMUL_NBITS")
    cfg.extra_options["MatMulNBitsParams"]["Algorithm"] = "GPTQ"
    cfg.extra_options["GPTQParams"] = {"GroupSize": 32, "PerChannel": True}
    _, out = _quantize(cfg, model, [{"x": x}])
    assert _inits(out)["w_scales"].shape == (48, 4)

    cfg.algo_config = [qc.CLEConfig()]
    with pytest.raises(NotImplementedError, match="MatMulNBits"):
        qc.ModelQuantizer(cfg).quantize_model(model, calibration_data_reader=[{"x": x}])


def test_preset_exclude_bits_and_matmul_add_note():
    model = _model(
        "h = MatMul(x, w1)\n y = Add(h, b)", [("w1", (64, 16)), ("b", (16,))]
    )
    cfg = qc.QConfig.get_default_config("MATMUL_NBITS")
    # (without ONNX Runtime's optimizer here nothing fuses the pair; with it, the
    # Gemm is left alone as in Quark: tests/test_quark_block_preproc_parity.py)
    cfg.extra_options["UseRuntimeOptimizers"] = False
    with pytest.warns(UserWarning, match="Gemm"):
        qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=[{"x": np.zeros((3, 64), np.float32)}]
        )
    cfg = qc.QConfig.get_default_config("MATMUL_NBITS")
    cfg.extra_options["SkipPreprocess"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to warn about
        out = qc.ModelQuantizer(cfg).quantize_model(model, calibration_data_reader=[])
    assert [n.op_type for n in out.graph.node] == ["MatMulNBits", "Add"]

    model = _mlp()
    model.graph.node[2].name = "head"
    cfg = qc.QConfig.get_default_config("MATMUL_NBITS")
    cfg.exclude = ["head"]
    _, out = _quantize(cfg, model)
    assert [n.op_type for n in out.graph.node] == ["MatMulNBits", "Relu", "MatMul"]

    cfg = qc.QConfig.get_default_config("MATMUL_NBITS")
    cfg.extra_options["MatMulNBitsParams"]["Bits"] = 8
    with pytest.raises(NotImplementedError, match="4-bit"):
        _quantize(cfg, model)
