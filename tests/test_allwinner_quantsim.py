"""scripts/allwinner/transformer_accuracy/quantsim.py emulates Acuity's quantization schemes with fake-quantize nodes.

The accuracy study's conclusions rest on these primitives, so each is checked against an independent numpy computation: the weight
fake-quantizers, the nodes inserted into the graph (run through onnxruntime and compared with the numpy result), the keep / keep_fp16
semantics, calibration, and the structural sets that hybrid schemes use.
"""

import importlib.util
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser, shape_inference

ort = pytest.importorskip("onnxruntime")

ROOT = Path(__file__).resolve().parents[1] / "scripts" / "allwinner"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qs = _load(ROOT / "transformer_accuracy" / "quantsim.py", "quantsim")
npu = _load(ROOT / "npu_rewrite.py", "npu_rewrite")


def _model(body, initializers=(), opset=17):
    model = parser.parse_model(f'<ir_version: 9, opset_import: ["": {opset}]> {body}')
    model.graph.initializer.extend(initializers)
    return shape_inference.infer_shapes(model)


def _run(model, x):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"x": x})[0]


RELU = "g (float[1, 64] x) => (float[1, 64] y) { y = Relu(x) }"


# ---- numpy fake-quantizers ----------------------------------------------------------------------------------------------------
def test_fq_affine_clamps_and_keeps_zero_exact():
    x = np.array([0.0, 0.1, 2.9, 6.0, 7.5, -3.0], np.float32)
    out = qs.fq_affine(x, 0.0, 6.0)
    step = 6.0 / 255
    assert out[0] == 0.0
    assert out[3] == pytest.approx(6.0, abs=1e-6) and out[4] == pytest.approx(
        6.0, abs=1e-6
    )  # saturates at the top
    assert out[5] == 0.0  # and at the bottom
    np.testing.assert_allclose(out[1:3], x[1:3], atol=step / 2 + 1e-7)


def test_fq_affine_widens_the_range_to_include_zero_so_zero_is_exact():
    # a range that does not contain zero is widened (activation range [2, 5] -> [0, 5]); a negative range keeps zero exactly representable
    assert qs.fq_affine(np.array([0.0], np.float32), 2.0, 5.0)[0] == 0.0
    out = qs.fq_affine(np.array([0.0, -1.0, 3.0], np.float32), -1.0, 3.0)
    assert out[0] == 0.0
    np.testing.assert_allclose(out, [0.0, -1.0, 3.0], atol=4.0 / 255)


def test_fq_affine_degenerate_range_is_the_identity():
    x = np.zeros(4, np.float32)
    np.testing.assert_array_equal(qs.fq_affine(x, 0.0, 0.0), x)


def test_per_channel_scales_are_independent():
    # channel 0 is small and channel 1 is large: a per-tensor scale would round channel 0 to zero, per-channel keeps it
    w = np.stack([np.linspace(-0.01, 0.01, 9), np.linspace(-10, 10, 9)]).astype(
        np.float32
    )
    per_channel = qs.fq_sym_perchannel(w, axis=0)
    np.testing.assert_allclose(per_channel[0], w[0], atol=0.01 / 127)
    np.testing.assert_allclose(per_channel[1], w[1], atol=10 / 127)
    per_tensor = qs.fq_affine(w, float(w.min()), float(w.max()))
    assert np.abs(per_tensor[0] - w[0]).max() > np.abs(per_channel[0] - w[0]).max() * 10


def test_dynamic_fixed_point_uses_power_of_two_scales():
    assert qs.dfp_fl(3.0, 16) == 13  # 3 < 2^2, so 15 - 2 fractional bits
    x = np.array([3.0, -2.7, 0.001, 1.23456], np.float32)
    np.testing.assert_allclose(qs.fq_dfp(x), x, atol=2.0**-14)
    assert qs.fq_dfp(np.array([4.0, 1.0], np.float32))[0] == pytest.approx(
        4.0 - 2.0**-13
    )  # max is clipped to the top code
    assert qs.dfp_fl(0.0) == 0


def test_half_rounding_matches_numpy_and_bf16_rounds_to_nearest_even():
    x = np.random.default_rng(0).standard_normal(1000).astype(np.float32) * 100
    np.testing.assert_array_equal(
        qs.to_half(x), x.astype(np.float16).astype(np.float32)
    )
    one, ulp = np.float32(1.0), np.float32(2.0**-7)  # bfloat16 has 7 fraction bits
    assert (
        qs.to_bf16(np.array([one + ulp / 2], np.float32))[0] == one
    )  # tie between 1.0 (even) and 1+ulp (odd) -> 1.0
    assert (
        qs.to_bf16(np.array([one + 3 * ulp / 2], np.float32))[0] == one + 2 * ulp
    )  # tie between odd and even -> even
    assert (
        qs.to_bf16(np.array([one + ulp / 2 + 2.0**-20], np.float32))[0] == one + ulp
    )  # just above a tie -> up
    special = qs.to_bf16(np.array([np.inf, -np.inf, np.nan, 0.0, -0.0], np.float32))
    assert (
        np.isinf(special[0])
        and np.isinf(special[1])
        and np.isnan(special[2])
        and special[3] == 0
        and np.signbit(special[4])
    )


# ---- the inserted graph nodes agree with numpy ---------------------------------------------------------------------------------
def test_uint8_chain_matches_the_numpy_reference():
    rng = (0.0, 6.0)
    scale = 6.0 / 255
    x = ((np.arange(64) + 0.3) * scale * 4 - 2).astype(np.float32)[
        None
    ]  # off the rounding ties; includes values outside the range
    model = _model(RELU)
    out = _run(qs.build(model, {"y": rng}, "uint8"), x)
    np.testing.assert_allclose(out, qs.fq_affine(np.maximum(x, 0), *rng), atol=1e-6)


def test_int16_chain_matches_the_numpy_reference():
    x = (np.linspace(-1, 9.7, 64) + 0.0123).astype(np.float32)[None]
    out = _run(qs.build(_model(RELU), {"y": (0.0, 9.7)}, "int16"), x)
    s = 2.0 ** qs.dfp_fl(9.7, 16)
    want = np.clip(np.round(np.maximum(x, 0) * s), -32768, 32767) / s
    np.testing.assert_allclose(out, want, atol=1e-6)


@pytest.mark.parametrize(
    "scheme,reference", [("fp16", qs.to_half), ("bf16", qs.to_bf16)]
)
def test_float_format_chains_round_through_the_format(scheme, reference):
    x = (np.random.default_rng(1).standard_normal((1, 64)) * 37 + 0.3).astype(
        np.float32
    )
    out = _run(qs.build(_model(RELU), {"y": (0.0, 1.0)}, scheme), x)
    np.testing.assert_array_equal(out, reference(np.maximum(x, 0)))


def test_keep_leaves_a_tensor_exactly_float_and_keep_fp16_rounds_it_through_fp16():
    x = (np.random.default_rng(2).standard_normal((1, 64)) * 5).astype(np.float32)
    model, rng = _model(RELU), {"y": (0.0, 20.0)}
    relu = np.maximum(x, 0)
    np.testing.assert_array_equal(
        _run(qs.build(model, rng, "uint8", keep={"y"}), x), relu
    )
    np.testing.assert_array_equal(
        _run(qs.build(model, rng, "uint8", keep_fp16={"y"}), x), qs.to_half(relu)
    )
    np.testing.assert_array_equal(_run(qs.build(model, rng, "fp32"), x), relu)


def test_a_zero_width_range_leaves_the_tensor_unchanged():
    x = np.zeros((1, 64), np.float32)
    np.testing.assert_array_equal(
        _run(qs.build(_model(RELU), {"y": (0.0, 0.0)}, "uint8"), x), x
    )


# ---- weights ----------------------------------------------------------------------------------------------------------------------
def _weights(model, name):
    return numpy_helper.to_array(
        next(i for i in model.graph.initializer if i.name == name)
    )


def test_weight_quantization_follows_the_scheme_and_the_operator_layout():
    w = (
        np.random.default_rng(3).standard_normal((8, 4)) * np.array([0.01, 1, 5, 0.1])
    ).astype(np.float32)
    wt = np.ascontiguousarray(w.T)  # Gemm transB layout: [out, in]
    b = np.arange(4, dtype=np.float32) + 0.123456
    model = _model(
        "g (float[2, 8] x) => (float[2, 4] y, float[2, 4] z) { y = MatMul(x, w) z = Gemm<transB = 1>(x, wt, b) }",
        [
            numpy_helper.from_array(w, "w"),
            numpy_helper.from_array(wt, "wt"),
            numpy_helper.from_array(b, "b"),
        ],
    )
    pcq = qs.build(model, {}, "pcq")
    np.testing.assert_array_equal(
        _weights(pcq, "w"), qs.fq_sym_perchannel(w, axis=-1)
    )  # MatMul: per output column
    np.testing.assert_array_equal(
        _weights(pcq, "wt"), qs.fq_sym_perchannel(wt, axis=0)
    )  # Gemm transB: per output row
    np.testing.assert_array_equal(_weights(pcq, "b"), b)  # the bias stays float
    u8 = qs.build(model, {}, "uint8")
    np.testing.assert_array_equal(
        _weights(u8, "w"), qs.fq_affine(w, float(w.min()), float(w.max()))
    )
    np.testing.assert_array_equal(
        _weights(qs.build(model, {}, "fp16"), "w"), qs.to_half(w)
    )
    np.testing.assert_array_equal(
        _weights(qs.build(model, {}, "int16"), "w"), qs.fq_dfp(w)
    )
    assert _weights(model, "w").tobytes() == w.tobytes()  # build() works on a copy


def test_pcq_beats_per_tensor_when_output_channels_have_very_different_scales():
    w = (
        np.random.default_rng(4).standard_normal((64, 16)) * np.geomspace(0.001, 3, 16)
    ).astype(np.float32)
    model = _model(
        "g (float[2, 64] x) => (float[2, 16] y) { y = MatMul(x, w) }",
        [numpy_helper.from_array(w, "w")],
    )

    def worst_channel_relative_error(scheme):
        q = _weights(qs.build(model, {}, scheme), "w")
        return (np.abs(q - w).mean(0) / np.abs(w).mean(0)).max()

    # per-channel keeps every column within a percent; one per-tensor scale rounds the smallest columns to zero (100% error)
    assert worst_channel_relative_error("pcq") < 0.02
    assert worst_channel_relative_error("uint8") > 0.5


# ---- calibration ------------------------------------------------------------------------------------------------------------------
def test_collect_ranges_gives_minmax_and_the_mean_of_per_batch_ranges():
    model = _model(RELU)
    batches = [{"x": np.full((1, 64), v, np.float32)} for v in (-1.0, 2.0, 6.0)]
    ranges = qs.collect_ranges(model, batches, ["y"])
    assert ranges["minmax"]["y"] == (0.0, 6.0)  # relu clamps the -1 batch to 0
    # each batch is constant, so its min == max == relu(value) = 0, 2, 6; the moving-average range is their mean at both ends
    lo, hi = ranges["ema"]["y"]
    assert lo == pytest.approx((0 + 2 + 6) / 3) and hi == pytest.approx((0 + 2 + 6) / 3)


def test_collect_percentile_clips_outliers():
    model = _model(RELU)
    x = np.linspace(0, 1, 64, dtype=np.float32)[None].copy()
    x[0, 0] = 1000.0
    lo, hi = qs.collect_percentile(model, [{"x": x}], ["y"], 0.0, 90.0)["y"]
    assert (
        hi < 2.0
    )  # the 90th percentile ignores the outlier that min/max would have used
    assert qs.collect_ranges(model, [{"x": x}], ["y"])["minmax"]["y"][1] == 1000.0


# ---- structure for hybrid schemes ------------------------------------------------------------------------------------------------
def test_hybrid_sets_find_softmax_inputs_layernorm_and_gelu_interiors_and_matmul_inputs():
    body = """g (float[2, 8] x, float[2, 8] mask) => (float[2, 8] y, float[2, 8] p) {
        s = Add(x, mask)
        p = Softmax<axis = -1>(s)
        n = LayerNormalization<axis = -1, epsilon = 1e-5>(x, sc)
        h = Gemm(n, w1)
        t = Mul(h, c1)
        e = Erf(t)
        e1 = Add(e, one)
        hx = Mul(h, e1)
        a = Mul(hx, half)
        y = Gemm(a, w2)
    }"""
    init = [
        numpy_helper.from_array(np.ones(s, np.float32), n)
        for n, s in (("sc", (8,)), ("w1", (8, 8)), ("w2", (8, 8)))
    ]
    init += [
        numpy_helper.from_array(np.float32(v).reshape(()), n)
        for n, v in (("c1", 0.7), ("one", 1.0), ("half", 0.5))
    ]
    model = _model(body, init)
    model, _ = npu.rewrite(model)
    sets = qs.hybrid_sets(shape_inference.infer_shapes(model))
    assert sets["softmax_in"] == {"s"}
    assert (
        sets["ln_interior"]
        and all("__" in t for t in sets["ln_interior"])
        and "n" not in sets["ln_interior"]
    )
    assert {"e", "e1", "hx", "t"} <= sets[
        "gelu_interior"
    ]  # the interior of the erf-GELU
    assert (
        "a" not in sets["gelu_interior"] and "h" not in sets["gelu_interior"]
    )  # its input and output stay quantized
    assert {"n", "a"} <= sets["matmul_in"]
    onnx.checker.check_model(model)
