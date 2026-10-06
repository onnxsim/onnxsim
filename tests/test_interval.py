"""Tests for onnxsim.ranges (annotations) and onnxsim.interval (propagation + quantization bounds).

The central property test: sample inputs inside the box, run the model in
onnxruntime with *every* intermediate tensor exposed, and require each observed
value to sit inside the propagated interval (up to float32 rounding slack).
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

import onnxsim
from onnxsim import interval as I
from onnxsim import ranges as R


def _model(body, initializer=None, opset=15, ir_version=8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    onnx.checker.check_model(model)
    return model


def _f32(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32)


def _all_tensors(model, x):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    produced = [o for n in m.graph.node for o in n.output if o]
    del m.graph.output[:]
    m.graph.output.extend(onnx.helper.make_empty_tensor_value_info(o) for o in produced)
    sess = ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return dict(zip(produced, sess.run(None, x)))


def _cnn(rng):
    k = 4
    return _model(
        """
        m (float[1,3,6,6] x) => (float[1,5] y) {
          c = Conv<pads=[1,1,1,1]>(x, W, B)
          b = BatchNormalization<epsilon=1e-5>(c, g, be, mu, var)
          r = Relu(b)
          s = Sigmoid(c)
          a = Add(r, s)
          p = GlobalAveragePool(a)
          f = Flatten(p)
          y = Gemm<transB=1>(f, W2, B2)
        }""",
        dict(
            W=_f32(rng, k, 3, 3, 3), B=_f32(rng, k), g=rng.uniform(0.5, 1.5, k).astype(np.float32),
            be=_f32(rng, k), mu=_f32(rng, k), var=rng.uniform(0.5, 2, k).astype(np.float32),
            W2=_f32(rng, 5, k), B2=_f32(rng, 5),
        ),
    )  # fmt: skip


# ---- ranges ------------------------------------------------------------------


def test_range_roundtrip_per_channel_and_unbounded():
    m = _model("m (float[1,3,2,2] x) => (float[1,3,2,2] y) { y = Relu(x) }")
    R.set_range(m, "x", np.array([0.0, -1.0, 0.5]).reshape(1, 3, 1, 1), 1.0)
    R.set_range(m, "y", 0.0, None)
    got = R.get_ranges(m)
    np.testing.assert_array_equal(got["x"][0].reshape(-1), [0.0, -1.0, 0.5])
    assert got["x"][1] == 1.0 and got["y"][0] == 0.0 and np.isposinf(got["y"][1])
    R.set_range(m, "x", 0.0, 2.0)  # replaces, does not duplicate
    assert sum(p.key == R.KEY_PREFIX + "x" for p in m.metadata_props) == 1
    R.clear_range(m, "x")
    assert "x" not in R.get_ranges(m) and "y" in R.get_ranges(m)
    with pytest.raises(ValueError):
        R.set_range(m, "x", 1.0, 0.0)


def test_sample_stays_inside_and_handles_one_sided():
    rng = np.random.default_rng(0)
    v = R.sample((np.asarray(-2.0), np.asarray(3.0)), (1000,), rng=rng)
    assert v.min() >= -2.0 and v.max() <= 3.0 and v.std() > 0.5
    w = R.sample((np.asarray(0.0), np.asarray(np.inf)), (100,), rng=rng)
    assert w.min() >= 0.0 and np.isfinite(w).all()


def test_check_outputs_reports_violations_only():
    rg = {"p": (np.asarray(0.0), np.asarray(1.0))}
    assert R.check_outputs(rg, {"p": np.array([0.0, 0.5, 1.0])}) == []
    assert (
        "leaves the annotated range"
        in R.check_outputs(rg, {"p": np.array([0.5, 1.5])})[0]
    )
    assert R.check_outputs(rg, {"other": np.array([99.0])}) == []


def test_annotation_survives_simplify():
    m = _cnn(np.random.default_rng(1))
    R.set_range(m, "x", 0.0, 1.0)
    sim, _ = onnxsim.simplify(m, certify=False)
    assert R.get_ranges(sim)["x"][1] == 1.0


# ---- interval propagation ----------------------------------------------------


def test_intervals_enclose_every_intermediate_tensor():
    rng = np.random.default_rng(2)
    model = _cnn(rng)
    res = I.propagate(model, {"x": (-1.0, 1.0)})
    assert res.unsupported == []
    for _ in range(40):
        x = rng.uniform(-1, 1, (1, 3, 6, 6)).astype(np.float32)
        for name, val in _all_tensors(model, {"x": x}).items():
            assert res.contains(name, val), f"{name} escaped its interval"


def test_unbounded_input_gives_unbounded_not_wrong():
    res = I.propagate(_cnn(np.random.default_rng(3)))
    lo, hi = res.hull("y")
    assert lo == -np.inf and hi == np.inf
    s_lo, s_hi = res.hull("s")  # Sigmoid is bounded whatever the input
    assert 0.0 <= s_lo and s_hi <= 1.0


def test_unsupported_op_falls_back_to_unbounded():
    m = _model("m (float[4] x) => (float[4] y) { a = Abs(x)  y = Relu(a) }")
    res = I.propagate(m, {"x": (-1.0, 1.0)})
    assert res.unsupported == ["Abs"]
    assert res.hull("a") == (-np.inf, np.inf) and res.hull("y")[0] == 0.0


def test_bilinear_both_sides_intervals_enclose():
    rng = np.random.default_rng(4)
    m = _model("m (float[2,3] a, float[3,2] b) => (float[2,2] y) { y = MatMul(a, b) }")
    res = I.propagate(m, {"a": (-1.0, 2.0), "b": (-0.5, 1.5)})
    for _ in range(200):
        a = rng.uniform(-1, 2, (2, 3)).astype(np.float32)
        b = rng.uniform(-0.5, 1.5, (3, 2)).astype(np.float32)
        assert res.contains("y", a @ b)


# ---- quantization bounds -----------------------------------------------------


def _matmul(k, w):
    return _model(
        f"m (float[1,{k}] x) => (float[1,2] y) {{ y = MatMul(x, W) }}", dict(W=w)
    )


def test_accumulator_bound_matches_hand_computation_and_tightens():
    k = 100
    w = np.full((k, 2), 0.5, np.float32)  # every weight quantizes to +-127
    # [0, 1]: scale 1/255, zero point 0 -> |xq - zp| <= 255 -> same as K*127*255
    (b,) = I.quantization_bounds(_matmul(k, w), {"x": (0.0, 1.0)})
    assert (
        b.acc_bound == k * 127 * 255
        and b.int32_safe
        and b.tightening == pytest.approx(1.0)
    )
    # [-1, 1]: zero point 128 -> |xq - zp| <= 128, about 2x tighter than the full range
    (b,) = I.quantization_bounds(_matmul(k, w), {"x": (-1.0, 1.0)})
    assert b.act_zero_point == 128 and b.acc_bound == k * 127 * 128
    assert b.tightening == pytest.approx(255 / 128)


def test_deep_reduction_overflows_int32_and_fp32_exactness():
    k = 140_000  # K*127*255 > 2**31
    (b,) = I.quantization_bounds(
        _matmul(k, np.full((k, 1), 1.0, np.float32)), {"x": (0.0, 1.0)}
    )
    assert not b.int32_safe and not b.fp32_cast_exact
    (b2,) = I.quantization_bounds(
        _matmul(8, np.ones((8, 1), np.float32)), {"x": (0.0, 1.0)}
    )
    assert b2.int32_safe and b2.fp32_cast_exact


def test_certified_error_bound_holds_for_real_fake_quantization():
    rng = np.random.default_rng(5)
    k = 64
    w = rng.standard_normal((k, 3)).astype(np.float32)
    model = _matmul(k, w)
    model.graph.node[0].output[0] = "y"
    (b,) = I.quantization_bounds(model, {"x": (-1.0, 1.0)})
    lo, hi = b.act_range
    s_x = b.act_scale
    s_w = np.abs(w).max(axis=0) / 127.0
    wq = np.round(w / s_w) * s_w
    worst = 0.0
    for _ in range(300):
        x = rng.uniform(-1, 1, (1, k))
        xq = (
            np.clip(np.round(x / s_x) + b.act_zero_point, 0, 255) - b.act_zero_point
        ) * s_x
        worst = max(worst, np.abs(x @ w - xq @ wq).max())
    assert worst <= b.max_abs_error  # a real bound is never exceeded
    assert (
        worst > 0.05 * b.max_abs_error
    )  # and is not absurdly loose for a linear layer


def test_unbounded_activation_layers_are_omitted():
    assert I.quantization_bounds(_matmul(4, np.ones((4, 2), np.float32))) == []


def test_report_formats():
    out = I.format_quantization_report(
        I.quantization_bounds(
            _matmul(16, np.ones((16, 2), np.float32)), {"x": (0.0, 1.0)}
        )
    )
    assert "acc bound" in out and "MatMul" in out
