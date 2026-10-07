"""Tests for onnxsim.backward_diff: a backward (CROWN-style) bound on |orig - converted|.

Soundness is the point, so almost every test samples inputs (and noise values) inside the boxes,
runs both graphs in onnxruntime, and requires every observed difference to sit below the bound.
Tightness is checked against the zonotope engine on the cases where that finishes: the backward
bound may be somewhat looser (it relaxes the float graph's own Relu once per neuron), never
unsound and never orders of magnitude looser on the headline cases.
"""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

pytest.importorskip("onnxruntime")
import onnxruntime as ort  # noqa: E402

import onnxsim  # noqa: E402
from onnxsim import accuracy, interval, quant_verify  # noqa: E402
from onnxsim import backward_diff as BD  # noqa: E402
from onnxsim import zonotope as Z  # noqa: E402

# float32 execution can exceed a real-arithmetic bound by float32 rounding
SLACK = 1e-4


def _model(body, initializer=None, opset=15, ir_version=9):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    return model


def _f32(x):
    return np.asarray(x, dtype=np.float32)


def _session(model):
    return ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def _observed(a, b, boxes, n=400, seed=0):
    """Largest |a(x) - b(x)| over random points of the boxes (graph inputs of ``a`` and ``b``)."""
    rng = np.random.default_rng(seed)
    sa, sb = _session(a), _session(b)
    na = {i.name for i in sa.get_inputs()}
    nb = {i.name for i in sb.get_inputs()}
    worst = 0.0
    for _ in range(n):
        feed = {
            k: rng.uniform(lo, hi).astype(np.float32) for k, (lo, hi) in boxes.items()
        }
        ya = sa.run(None, {k: v for k, v in feed.items() if k in na})[0]
        yb = sb.run(None, {k: v for k, v in feed.items() if k in nb})[0]
        worst = max(worst, float(np.abs(ya - yb).max()))
    return worst


def _box(shape, r):
    return (-r * np.ones(shape), r * np.ones(shape))


def _mlp(ws, bs, noise_at=(), width_in=16):
    """x -> (MatMul, Add, Relu)* -> y, with an additive noise input after the listed layers."""
    lines, prev, inputs = [], "x", [f"float[1,{width_in}] x"]
    for i, (w, b) in enumerate(zip(ws, bs)):
        lines += [f"m{i} = MatMul({prev}, W{i})", f"a{i} = Add(m{i}, B{i})"]
        out = f"a{i}"
        if i < len(ws) - 1:
            lines.append(f"r{i} = Relu(a{i})")
            out = f"r{i}"
        if i in noise_at:
            inputs.append(f"float[1,{w.shape[1]}] e{i}")
            lines.append(f"n{i} = Add({out}, e{i})")
            out = f"n{i}"
        prev = out
    lines.append(f"y = Identity({prev})")
    consts = {f"W{i}": w for i, w in enumerate(ws)} | {
        f"B{i}": b for i, b in enumerate(bs)
    }
    return _model(
        f"g ({', '.join(inputs)}) => (float[1,{ws[-1].shape[1]}] y) {{\n"
        + "\n".join(lines)
        + "\n}",
        {k: _f32(v) for k, v in consts.items()},
    )


def _with_unused_noise(model, names_shapes):
    """quant_verify's float reference carries the noise inputs unused; do the same."""
    out = onnx.ModelProto()
    out.CopyFrom(model)
    for name, shape in names_shapes.items():
        out.graph.input.append(
            onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, list(shape)
            )
        )
    return out


def _pair(seed=0, dw=0.01, noise_r=0.02, dims=(16, 24, 24, 8)):
    rng = np.random.default_rng(seed)
    ws = [
        _f32(rng.standard_normal((dims[i], dims[i + 1])) / np.sqrt(dims[i]))
        for i in range(len(dims) - 1)
    ]
    bs = [_f32(rng.standard_normal(dims[i + 1]) * 0.1) for i in range(len(dims) - 1)]
    wq = [w + _f32(rng.standard_normal(w.shape) * dw) for w in ws]
    noise_at = (0, 1)
    a = _with_unused_noise(_mlp(ws, bs), {f"e{i}": (1, dims[i + 1]) for i in noise_at})
    b = _mlp(wq, bs, noise_at=noise_at)
    boxes = {"x": _box((1, dims[0]), 1.0)} | {
        f"e{i}": _box((1, dims[i + 1]), noise_r) for i in noise_at
    }
    return a, b, boxes


# ---- soundness and tightness on MLPs ---------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mlp_with_perturbed_weights_and_noise_is_sound_and_close_to_the_zonotope(seed):
    a, b, boxes = _pair(seed)
    bd = BD.bound_difference(a, b, boxes)
    zd = Z.bound_difference(a, b, boxes)
    assert bd.notes == []
    got = float(np.max(bd.max_abs["y"]))
    assert _observed(a, b, boxes, seed=seed) <= got * (1 + SLACK)
    assert got <= 3.0 * float(
        np.max(zd.max_abs["y"])
    )  # measured ~1.4x; guard, not a claim
    ia = interval.propagate(a, boxes)
    ib = interval.propagate(b, boxes)
    plain = np.maximum(
        np.abs(ia.intervals["y"][0] - ib.intervals["y"][1]),
        np.abs(ia.intervals["y"][1] - ib.intervals["y"][0]),
    )
    assert got < float(
        np.max(plain)
    )  # better than treating the two graphs independently


def test_identical_models_have_exactly_zero_difference():
    a, _, boxes = _pair()
    b = onnx.ModelProto()
    b.CopyFrom(a)
    got = BD.bound_difference(a, b, boxes)
    assert float(np.max(got.max_abs["y"])) == 0.0


def test_pure_noise_difference_is_the_noise_radius():
    a = _model("g (float[1,8] x, float[1,8] e) => (float[1,8] y) { y = Identity(x) }")
    b = _model("g (float[1,8] x, float[1,8] e) => (float[1,8] y) { y = Add(x, e) }")
    got = BD.bound_difference(a, b, {"x": _box((1, 8), 1.0), "e": _box((1, 8), 0.125)})
    # the reference uses the noise input only on one side, as quant_verify's float reference does
    assert np.allclose(got.max_abs["y"], 0.125)


def test_linear_weight_difference_is_exact():
    rng = np.random.default_rng(3)
    w = _f32(rng.standard_normal((8, 4)))
    dw = _f32(rng.standard_normal((8, 4)) * 0.05)
    a = _model("g (float[1,8] x) => (float[1,4] y) { y = MatMul(x, W) }", {"W": w})
    b = _model("g (float[1,8] x) => (float[1,4] y) { y = MatMul(x, W) }", {"W": w + dw})
    got = BD.bound_difference(a, b, {"x": _box((1, 8), 2.0)})
    expected = 2.0 * np.abs(dw.astype(np.float64)).sum(axis=0)
    assert np.allclose(got.max_abs["y"].reshape(-1), expected, rtol=1e-6)


def test_bias_shift_behind_a_relu_is_not_hidden():
    a, b, boxes = _pair(seed=5, dw=0.0, noise_r=0.0)
    wrong = onnx.ModelProto()
    wrong.CopyFrom(b)
    for t in wrong.graph.initializer:
        if t.name == "B1":
            t.CopyFrom(
                numpy_helper.from_array(numpy_helper.to_array(t) + _f32(0.25), "B1")
            )
    bd = BD.bound_difference(a, wrong, boxes)
    got = float(np.max(bd.max_abs["y"]))
    seen = _observed(a, wrong, boxes, n=600)
    assert seen > 0.05  # the rewrite really is different
    assert seen <= got * (1 + SLACK)
    assert not bd.within(
        atol=1e-3, rtol=0.0
    )  # and the verifier does not pretend otherwise


# ---- conv nets, residuals, real simplify output ------------------------------------------------


def _conv_bn_relu(seed=0, k=4):
    rng = np.random.default_rng(seed)
    return _model(
        f"""
        g (float[1,3,8,8] x) => (float[1,{k},6,6] y) {{
          c = Conv(x, W, B)
          b = BatchNormalization<epsilon=1e-5>(c, gm, be, mu, var)
          y = Relu(b)
        }}""",
        {
            "W": _f32(rng.standard_normal((k, 3, 3, 3)) * 0.5),
            "B": _f32(rng.standard_normal(k) * 0.5),
            "gm": _f32(rng.uniform(0.5, 1.5, k)),
            "be": _f32(rng.standard_normal(k) * 0.5),
            "mu": _f32(rng.standard_normal(k) * 0.5),
            "var": _f32(rng.uniform(0.5, 2.0, k)),
        },
    )


def test_real_simplify_of_conv_bn_relu_is_proved_far_tighter_than_intervals():
    model = _conv_bn_relu()
    simplified, _ = onnxsim.simplify(model, certify=False)
    assert "BatchNormalization" not in {n.op_type for n in simplified.graph.node}
    boxes = {"x": _box((1, 3, 8, 8), 1.0)}
    bd = BD.bound_difference(model, simplified, boxes)
    zd = Z.bound_difference(model, simplified, boxes)
    got = float(np.max(bd.max_abs["y"]))
    assert _observed(model, simplified, boxes, n=100) <= got * (1 + SLACK) + 1e-6
    ia, ib = interval.propagate(model, boxes), interval.propagate(simplified, boxes)
    plain = float(
        np.max(
            np.maximum(
                np.abs(ia.intervals["y"][0] - ib.intervals["y"][1]),
                np.abs(ia.intervals["y"][1] - ib.intervals["y"][0]),
            )
        )
    )
    assert (
        got < 1e-4 and plain > 100 * got
    )  # float32 rounding of the folded constants only
    assert got <= 10.0 * float(np.max(zd.max_abs["y"])) + 1e-12


def _residual(seed=1, delta=0.0):
    rng = np.random.default_rng(seed)
    w1 = _f32(rng.standard_normal((4, 4, 3, 3)) * 0.4)
    w2 = _f32(rng.standard_normal((4, 4, 3, 3)) * 0.4)
    body = (
        "g (float[1,4,6,6] x) => (float[1,4,6,6] y) "
        "{ c = Conv<pads=[1,1,1,1]>(x, W1, B1) r = Relu(c) d = Conv<pads=[1,1,1,1]>(r, W2, B2) "
        "a = Add(d, x) y = Relu(a) }"
    )
    return _model(
        body,
        {
            "W1": w1,
            "B1": _f32(rng.standard_normal(4) * 0.2),
            "W2": w2 + _f32(delta * rng.standard_normal(w2.shape)),
            "B2": _f32(rng.standard_normal(4) * 0.2),
        },
    )


def test_residual_network_is_sound():
    a, b = _residual(), _residual(delta=0.02)
    boxes = {"x": _box((1, 4, 6, 6), 1.0)}
    bd = BD.bound_difference(a, b, boxes)
    got = float(np.max(bd.max_abs["y"]))
    assert 0 < got < np.inf
    assert _observed(a, b, boxes) <= got * (1 + SLACK)


def test_conv_flatten_gemm_head_is_sound():
    rng = np.random.default_rng(2)
    body = (
        "g (float[1,2,6,6] x) => (float[1,5] y) "
        "{ c = Conv(x, W, B) r = Relu(c) p = AveragePool<kernel_shape=[2,2], strides=[2,2]>(r) "
        "f = Flatten(p) y = Gemm<transB=1>(f, V, C) }"
    )

    def build(eps):
        return _model(
            body,
            {
                "W": _f32(rng_w + eps * rng_d1),
                "B": _f32(rng_b),
                "V": _f32(rng_v + eps * rng_d2),
                "C": _f32(rng_c),
            },
        )

    rng_w = rng.standard_normal((3, 2, 3, 3)) * 0.4
    rng_b, rng_c = rng.standard_normal(3) * 0.1, rng.standard_normal(5) * 0.1
    rng_v = rng.standard_normal((5, 12)) * 0.3
    rng_d1, rng_d2 = rng.standard_normal(rng_w.shape), rng.standard_normal(rng_v.shape)
    a, b = build(0.0), build(0.03)
    boxes = {"x": _box((1, 2, 6, 6), 1.0)}
    bd = BD.bound_difference(a, b, boxes)
    assert bd.notes == []
    assert _observed(a, b, boxes) <= float(np.max(bd.max_abs["y"])) * (1 + SLACK)


def test_real_simplify_of_matmul_add_relu_matmul_add_to_gemm_is_tight():
    rng = np.random.default_rng(2)
    model = _model(
        """
        g (float[1,16] x) => (float[1,4] y) {
          a = MatMul(x, W1)
          b = Add(a, B1)
          r = Relu(b)
          c = MatMul(r, W2)
          y = Add(c, B2)
        }""",
        {
            "W1": _f32(rng.standard_normal((16, 8))),
            "B1": _f32(rng.standard_normal(8)),
            "W2": _f32(rng.standard_normal((8, 4))),
            "B2": _f32(rng.standard_normal(4)),
        },
    )
    simplified, _ = onnxsim.simplify(model, certify=False)
    assert "Gemm" in {n.op_type for n in simplified.graph.node}
    boxes = {"x": _box((1, 16), 1.0)}
    bd = BD.bound_difference(model, simplified, boxes)
    assert bd.notes == []
    assert float(np.max(bd.max_abs["y"])) < 1e-4
    assert _observed(model, simplified, boxes) <= float(np.max(bd.max_abs["y"])) + 1e-6


def test_maxpool_and_slice_are_handled_soundly():
    rng = np.random.default_rng(4)
    body = (
        "g (float[1,2,8,8] x) => (float[1,3] y) "
        "{ c = Conv(x, W, B) r = Relu(c) p = MaxPool<kernel_shape=[2,2], strides=[2,2]>(r) "
        "f = Flatten(p) h = Gemm<transB=1>(f, V, C) "
        "s = Constant<value=int64[1] {0}>() e = Constant<value=int64[1] {3}>() ax = Constant<value=int64[1] {1}>() "
        "y = Slice(h, s, e, ax) }"
    )
    w = rng.standard_normal((4, 2, 3, 3)) * 0.4
    v = rng.standard_normal((5, 4 * 3 * 3)) * 0.3
    dw, dv = rng.standard_normal(w.shape), rng.standard_normal(v.shape)

    def build(eps):
        return _model(
            body,
            {
                "W": _f32(w + eps * dw),
                "B": _f32(np.linspace(-0.2, 0.2, 4)),
                "V": _f32(v + eps * dv),
                "C": _f32(np.linspace(-0.1, 0.1, 5)),
            },
        )

    a, b = build(0.0), build(0.03)
    boxes = {"x": _box((1, 2, 8, 8), 1.0)}
    bd = BD.bound_difference(a, b, boxes)
    assert not any("alignment lost" in n or "not aligned" in n for n in bd.notes)
    assert _observed(a, b, boxes) <= float(np.max(bd.max_abs["y"])) * (1 + SLACK)


def test_outputs_argument_restricts_the_analysis():
    a, b, boxes = _pair(seed=1)
    got = BD.bound_difference(a, b, boxes, outputs="y")
    assert list(got.max_abs) == ["y"]


# ---- fallbacks ----------------------------------------------------------------------------------


def test_unaligned_output_falls_back_to_a_sound_interval_difference_with_a_note():
    a = _model("g (float[1,6] x) => (float[1,6] y) { y = Relu(x) }")
    b = _model("g (float[1,6] x) => (float[1,6] y) { y = Sigmoid(x) }")
    boxes = {"x": _box((1, 6), 2.0)}
    bd = BD.bound_difference(a, b, boxes)
    assert any("not aligned" in n or "alignment lost" in n for n in bd.notes)
    assert _observed(a, b, boxes) <= float(np.max(bd.max_abs["y"])) * (1 + SLACK)


def test_aligned_op_without_a_rule_is_a_sound_box():
    a = _model(
        "g (float[1,6] x) => (float[1,6] y) { s = Softmax<axis=1>(x) y = Identity(s) }"
    )
    b = _model(
        "g (float[1,6] x) => (float[1,6] y) { s = Softmax<axis=1>(x) y = Identity(s) }"
    )
    # identical inputs, identical op: exactly equal even though Softmax has no rule here
    got = BD.bound_difference(a, b, {"x": _box((1, 6), 1.0)})
    assert float(np.max(got.max_abs["y"])) == 0.0


def test_missing_input_range_gives_infinity_not_a_guess():
    a, b, boxes = _pair()
    boxes = {k: v for k, v in boxes.items() if k != "x"}
    got = BD.bound_difference(a, b, boxes)
    assert not np.isfinite(got.worst) and any("unbounded input" in n for n in got.notes)


def test_mismatched_outputs_raise():
    a = _model("g (float[1,6] x) => (float[1,6] y) { y = Relu(x) }")
    b = _model("g (float[1,6] x) => (float[1,6] z) { z = Relu(x) }")
    with pytest.raises(ValueError):
        BD.bound_difference(a, b, {"x": _box((1, 6), 1.0)})


def test_a_differing_constant_node_is_never_reported_as_equal():
    def build(value):
        return _model(
            f"g (float[1,6] x) => (float[1,6] y) {{ c = Constant<value=float {{{value}}}>() "
            "r = Relu(x) y = Add(r, c) }"
        )

    a, b = build(1.0), build(2.0)
    boxes = {"x": _box((1, 6), 1.0)}
    bd = BD.bound_difference(a, b, boxes)
    assert np.allclose(bd.max_abs["y"], 1.0, atol=1e-6)
    assert _observed(a, b, boxes) == pytest.approx(1.0)


# ---- quant_verify integration -------------------------------------------------------------------


def _quant_models():
    rng = np.random.default_rng(1)
    cnn = _model(
        "g (float[1,3,8,8] x) => (float[1,4,4,4] y) "
        "{ c = Conv(x, W1, B1) r = Relu(c) y = Conv(r, W2, B2) }",
        {
            "W1": _f32(rng.standard_normal((6, 3, 3, 3)) * 0.5),
            "B1": _f32(rng.standard_normal(6) * 0.5),
            "W2": _f32(rng.standard_normal((4, 6, 3, 3)) * 0.5),
            "B2": _f32(rng.standard_normal(4) * 0.5),
        },
        opset=21,
    )
    mlp = _model(
        "g (float[2,32] x) => (float[2,8] y) "
        "{ a = MatMul(x, W1) b = Add(a, B1) r = Relu(b) c = MatMul(r, W2) d = Relu(c) y = MatMul(d, W3) }",
        {
            "W1": _f32(rng.standard_normal((32, 64)) * 0.5),
            "B1": _f32(rng.standard_normal(64) * 0.5),
            "W2": _f32(rng.standard_normal((64, 64)) * 0.5),
            "W3": _f32(rng.standard_normal((64, 8)) * 0.5),
        },
        opset=21,
    )
    return {"cnn": cnn, "mlp": mlp}


def _quantize(model, scheme, dtype):
    cfg = accuracy.QuantizationConfig(
        scheme=scheme, dtype=dtype, num_calibration_samples=16
    )
    return accuracy.quantize(model, cfg)


@pytest.mark.parametrize(
    "scheme,dtype",
    [
        ("static", "int8"),
        ("static_int16", "int16"),
        ("qoperator", "int8"),
        ("weight_only", "int8"),
        ("dynamic", "int8"),
    ],
)
@pytest.mark.parametrize("name", ["cnn", "mlp"])
def test_quant_verify_backward_engine_is_sound_and_close_to_the_zonotope(
    name, scheme, dtype
):
    model = _quant_models()[name]
    q = _quantize(model, scheme, dtype)
    box = {model.graph.input[0].name: (-0.5, 0.5)}
    rb = quant_verify.verify(model, q, box, breakdown=False, engine="backward")
    rz = quant_verify.verify(model, q, box, breakdown=False)
    observed = quant_verify.observed_error(model, q, box, n=150)
    assert observed <= rb.worst + SLACK * (1.0 + rb.worst)
    assert np.isfinite(rb.worst) == np.isfinite(rz.worst)
    if rz.worst > 0 and np.isfinite(rz.worst):
        assert rb.worst <= 2.5 * rz.worst  # measured 1.0-1.64x over the same grid


def test_quant_verify_default_engine_is_unchanged_and_the_argument_is_validated():
    model = _quant_models()["mlp"]
    q = _quantize(model, "static", "int8")
    box = {"x": (-0.5, 0.5)}
    default = quant_verify.verify(model, q, box, breakdown=False)
    explicit = quant_verify.verify(model, q, box, breakdown=False, engine="zonotope")
    assert default.worst == explicit.worst
    with pytest.raises(ValueError, match="engine must be"):
        quant_verify.verify(model, q, box, engine="nonsense")
