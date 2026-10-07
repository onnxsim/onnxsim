"""Tests for the box part of onnxsim.zonotope (``box_inputs`` / ``box_fresh``).

A box part carries independent per-element noise as a radius instead of one dense symbol per
element. The properties pinned here:

* Default behaviour is untouched: without the options no tensor has a box part and every
  statistic that counts it is zero.
* Where each element's noise passes through at most ONE summing layer (Conv / MatMul / Gemm)
  before reaching an output, with only elementwise and shape ops around it, the box bound equals
  the dense bound exactly -- nothing is lost.
* Where the same noise reaches an element along more than one path -- a residual ``Add``, two
  consecutive summing layers (``sum_m |W2||W1| >= |W2 W1|``: signed paths through different
  middle elements cancel in a zonotope but not in a box), or pooling that merges elements -- the
  box treats the paths as independent: sound, and strictly looser than the dense bound.
* Soundness: every value onnxruntime produces for inputs and noise inside their boxes lies in the
  bounds, for every intermediate tensor, on MLPs, Conv+BN+Relu, residual nets and
  Sigmoid/Tanh chains, in both the single-model (``propagate``) and the paired two-model
  (``bound_difference``) setting, and on QDQ-converted models from onnxsim's own quantizers.
* It is cheaper where it is meant to be: fewer symbols and a smaller dense generator array.
* Anything without a rule falls back soundly (never a guessed bound).
"""

import contextlib
import io
import pathlib
import re

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import accuracy, quant_verify
from onnxsim import zonotope as Z

BOX = {"x": (-1.0, 1.0)}
SLACK = 1e-4  # float32 execution can exceed a real-arithmetic bound by float32 rounding


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


def _all_tensors(model, feeds):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    produced = [o for n in m.graph.node for o in n.output if o]
    del m.graph.output[:]
    m.graph.output.extend(onnx.helper.make_empty_tensor_value_info(o) for o in produced)
    sess = ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return dict(zip(produced, sess.run(None, feeds)))


def _contains(res, name, value):
    lo, hi = res.bounds(name)
    v = np.asarray(value, dtype=np.float64)
    pad = SLACK * (1.0 + np.abs(v))
    return bool(np.all(v >= lo - pad) and np.all(v <= hi + pad))


def _no_fallback(notes):
    """A supported op must not have fallen back to an interval box (sound, but a silent loss)."""
    assert not [n for n in notes if "precision lost" in n], notes


def _assert_sound(model, rng, shape, n=40, **kw):
    res = Z.propagate(model, BOX, **kw)
    _no_fallback(res.notes)
    for _ in range(n):
        x = rng.uniform(-1, 1, shape).astype(np.float32)
        for name, val in _all_tensors(model, {"x": x}).items():
            assert _contains(res, name, val), f"{name} escaped its bounds"
    return res


# ---- the representation ------------------------------------------------------------------


def test_zero_box_is_no_box_and_negative_is_rejected():
    z = Z.Zonotope(np.zeros(3), np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(3))
    assert (
        z.r is None
    )  # an all-zero box is dropped, so results stay identical to the dense path
    with pytest.raises(ValueError):
        Z.Zonotope(
            np.zeros(2), np.zeros(0, np.int64), np.zeros((0, 2)), np.array([1.0, -1.0])
        )


def test_bounds_include_the_box_part():
    z = Z.Zonotope(
        np.array([1.0, 2.0]),
        np.array([0], np.int64),
        np.array([[0.5, 0.0]]),
        np.array([0.25, 0.75]),
    )
    lo, hi = z.bounds(rel_slack=0.0)
    np.testing.assert_allclose(lo, [1.0 - 0.5 - 0.25, 2.0 - 0.75])
    np.testing.assert_allclose(hi, [1.0 + 0.5 + 0.25, 2.0 + 0.75])


def test_add_and_sub_add_radii_and_scale_uses_abs():
    a = Z.Zonotope(
        np.zeros(2), np.zeros(0, np.int64), np.zeros((0, 2)), np.array([1.0, 2.0])
    )
    b = Z.Zonotope(
        np.ones(2), np.zeros(0, np.int64), np.zeros((0, 2)), np.array([0.5, 0.5])
    )
    np.testing.assert_allclose(Z._add(a, b, 1.0).r, [1.5, 2.5])
    np.testing.assert_allclose(
        Z._add(a, b, -1.0).r, [1.5, 2.5]
    )  # |e_a - e_b| <= r_a + r_b
    np.testing.assert_allclose(Z._scale(a, np.array([-3.0, 2.0])).r, [3.0, 4.0])
    # a box added to itself: independent treatment doubles the radius (sound, looser than 2e)
    np.testing.assert_allclose(Z._add(a, a, -1.0).r, [2.0, 4.0])


# ---- default behaviour is untouched ------------------------------------------------------


def test_without_the_options_nothing_has_a_box_part():
    rng = np.random.default_rng(0)
    model = _model(
        "m (float[1,3,6,6] x) => (float[1,4,4,4] y) { c = Conv(x, W, B)  y = Relu(c) }",
        dict(W=_f32(rng, 4, 3, 3, 3), B=_f32(rng, 4)),
    )
    res = Z.propagate(model, BOX)
    assert res.stats["max_box_elements"] == 0
    assert all(v.r is None for v in res.tensors.values() if isinstance(v, Z.Zonotope))
    d = Z.bound_difference(model, model, BOX)
    assert (
        d.stats["orig"]["max_box_elements"]
        == d.stats["simplified"]["max_box_elements"]
        == 0
    )
    assert d.worst == 0.0


# ---- single path: nothing is lost --------------------------------------------------------

_LINEAR_NETS = {
    "conv_bn_flatten": (
        """m (float[1,3,6,6] x) => (float[1,144] y) {
             c = Conv<pads=[1,1,1,1]>(x, W, B)
             b = BatchNormalization<epsilon=1e-5>(c, g, be, mu, var)
             y = Flatten(b)
           }""",
        "bn",
    ),
    "gemm_alpha_beta_transB": (
        """m (float[2,4] x) => (float[2,5] y) {
             y = Gemm<transB=1, alpha=0.5, beta=2.0>(x, W2, B2)
           }""",
        "gemm",
    ),
    "strided_grouped_conv": (
        """m (float[1,4,8,8] x) => (float[1,4,4,4] y) {
             y = Conv<strides=[2,2], pads=[1,1,1,1], group=2, dilations=[1,1]>(x, Wg, Bg)
           }""",
        "grouped",
    ),
    "matmul_transpose_reshape": (
        """m (float[2,6] x) => (float[2,5] y) {
             a = MatMul(x, Wm)
             t = Transpose<perm=[1,0]>(a)
             y = Reshape(t, shp)
           }""",
        "matmul",
    ),
    "concat_of_two_convs_of_x": (
        """m (float[1,3,5,5] x) => (float[1,6,5,5] y) {
             a = Conv<pads=[1,1,1,1]>(x, Wa)
             b = Conv<pads=[1,1,1,1]>(x, Wb)
             y = Concat<axis=1>(a, b)
           }""",
        "concat",
    ),
}


def _linear_inits(rng):
    return dict(
        W=_f32(rng, 4, 3, 3, 3), B=_f32(rng, 4),
        g=rng.uniform(0.5, 1.5, 4).astype(np.float32), be=_f32(rng, 4),
        mu=_f32(rng, 4), var=rng.uniform(0.5, 2, 4).astype(np.float32),
        W2=_f32(rng, 5, 4), B2=_f32(rng, 5),
        Wg=_f32(rng, 4, 2, 3, 3), Bg=_f32(rng, 4),
        Wm=_f32(rng, 6, 5), shp=np.array([2, 5], np.int64),
        Wa=_f32(rng, 3, 3, 3, 3), Wb=_f32(rng, 3, 3, 3, 3),
    )  # fmt: skip


@pytest.mark.parametrize("name", sorted(_LINEAR_NETS))
def test_box_equals_dense_when_the_noise_passes_one_summing_layer(name):
    rng = np.random.default_rng(1)
    model = _model(_LINEAR_NETS[name][0], _linear_inits(rng))
    dense = Z.propagate(model, BOX)
    boxed = Z.propagate(model, BOX, box_inputs=["x"])
    _no_fallback(boxed.notes)
    assert boxed.stats["max_symbols"] == 0 and dense.stats["max_symbols"] > 0
    for out in (o.name for o in model.graph.output):
        d, b = dense.bounds(out), boxed.bounds(out)
        np.testing.assert_allclose(b[0], d[0], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(b[1], d[1], rtol=1e-9, atol=1e-12)


_MERGING_NETS = {
    "two_convs": """m (float[1,3,6,6] x) => (float[1,4,6,6] y) {
             a = Conv<pads=[1,1,1,1]>(x, W1)
             y = Conv<pads=[1,1,1,1]>(a, W2)
           }""",
    "two_matmuls": """m (float[2,6] x) => (float[2,5] y) {
             a = MatMul(x, Wm1)
             y = MatMul(a, Wm2)
           }""",
    "conv_then_global_average_pool": """m (float[1,3,6,6] x) => (float[1,4,1,1] y) {
             a = Conv<pads=[1,1,1,1]>(x, W1)
             y = GlobalAveragePool(a)
           }""",
}


@pytest.mark.parametrize("name", sorted(_MERGING_NETS))
def test_consecutive_summing_layers_and_pooling_lose_cancellation_but_stay_sound(name):
    rng = np.random.default_rng(10)
    inits = dict(
        W1=_f32(rng, 4, 3, 3, 3), W2=_f32(rng, 4, 4, 3, 3),
        Wm1=_f32(rng, 6, 8), Wm2=_f32(rng, 8, 5),
    )  # fmt: skip
    model = _model(_MERGING_NETS[name], inits)
    dense = Z.propagate(model, BOX)
    boxed = Z.propagate(model, BOX, box_inputs=["x"])
    out = model.graph.output[0].name
    (dlo, dhi), (blo, bhi) = dense.bounds(out), boxed.bounds(out)
    assert np.all(blo <= dlo + 1e-9) and np.all(
        bhi >= dhi - 1e-9
    )  # a box contains the zonotope
    assert np.mean(bhi - blo) > 1.05 * np.mean(
        dhi - dlo
    )  # and here it is genuinely looser
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    _assert_sound(model, rng, tuple(shape), n=20, box_inputs=["x"])


def test_two_paths_to_one_element_are_independent_in_a_box_so_it_is_looser_but_sound():
    rng = np.random.default_rng(2)
    model = _model(
        """m (float[1,3,5,5] x) => (float[1,3,5,5] y) {
             c = Conv<pads=[1,1,1,1]>(x, W)
             y = Sub(c, c)
           }""",
        dict(W=_f32(rng, 3, 3, 3, 3)),
    )
    dense = Z.propagate(model, BOX).bounds("y")
    boxed = Z.propagate(model, BOX, box_inputs=["x"]).bounds("y")
    np.testing.assert_allclose(
        dense[0], 0.0, atol=1e-6
    )  # x - x is exactly the zero form
    assert np.all(
        boxed[1] > 0.1
    )  # the box cannot know the two operands are the same noise
    assert np.all(boxed[0] <= dense[0] + 1e-9) and np.all(boxed[1] >= dense[1] - 1e-9)


# ---- soundness vs onnxruntime ------------------------------------------------------------


def test_sound_mlp_with_a_box_input():
    rng = np.random.default_rng(3)
    model = _model(
        """m (float[1,16] x) => (float[1,4] y) {
             a = MatMul(x, W1)
             b = Add(a, B1)
             r = Relu(b)
             c = MatMul(r, W2)
             y = Add(c, B2)
           }""",
        dict(W1=_f32(rng, 16, 8), B1=_f32(rng, 8), W2=_f32(rng, 8, 4), B2=_f32(rng, 4)),
    )
    _assert_sound(model, rng, (1, 16), box_inputs=["x"])


def test_sound_conv_bn_relu_and_residual_and_unimodal_chain():
    rng = np.random.default_rng(4)
    conv_bn_relu = _model(
        """m (float[1,3,6,6] x) => (float[1,4,4,4] y) {
             c = Conv(x, W, B)
             b = BatchNormalization<epsilon=1e-5>(c, g, be, mu, var)
             y = Relu(b)
           }""",
        _linear_inits(rng),
    )
    _assert_sound(conv_bn_relu, rng, (1, 3, 6, 6), box_inputs=["x"])
    residual = _model(
        """m (float[1,4,6,6] x) => (float[1,4,6,6] y) {
             c = Conv<pads=[1,1,1,1]>(x, W1, B1)
             r = Relu(c)
             d = Conv<pads=[1,1,1,1]>(r, W2, B2)
             a = Add(d, x)
             y = Relu(a)
           }""",
        dict(
            W1=_f32(rng, 4, 4, 3, 3) * 0.3, B1=_f32(rng, 4),
            W2=_f32(rng, 4, 4, 3, 3) * 0.3, B2=_f32(rng, 4),
        ),
    )  # fmt: skip
    _assert_sound(residual, rng, (1, 4, 6, 6), box_inputs=["x"])
    unimodal = _model(
        """m (float[1,8] x) => (float[1,3] y) {
             a = MatMul(x, W1)
             s = Sigmoid(a)
             b = MatMul(s, W2)
             y = Tanh(b)
           }""",
        dict(W1=_f32(rng, 8, 6), W2=_f32(rng, 6, 3)),
    )
    _assert_sound(unimodal, rng, (1, 8), box_inputs=["x"])


def test_box_inputs_without_box_fresh_is_also_sound_and_matches_the_dense_relaxation():
    rng = np.random.default_rng(5)
    model = _model(
        """m (float[1,8] x) => (float[1,3] y) {
             a = MatMul(x, W1)
             r = Relu(a)
             y = MatMul(r, W2)
           }""",
        dict(W1=_f32(rng, 8, 6), W2=_f32(rng, 6, 3)),
    )
    res = _assert_sound(model, rng, (1, 8), box_inputs=["x"], box_fresh=False)
    assert res.stats["max_symbols"] > 0  # the Relu error symbols stayed dense


def test_patterns_and_exact_names_select_inputs_and_unknown_patterns_are_harmless():
    rng = np.random.default_rng(6)
    model = _model(
        "m (float[1,4] x, float[1,4] n) => (float[1,4] y) { y = Add(x, n) }", {}
    )
    rng_in = {"x": (-1.0, 1.0), "n": (-0.5, 0.5)}
    exact = Z.propagate(model, rng_in, box_inputs=["n"])
    glob = Z.propagate(model, rng_in, box_inputs=["n*"])
    none = Z.propagate(model, rng_in, box_inputs=["nothing_matches_*"], box_fresh=False)
    np.testing.assert_allclose(exact.bounds("y")[1], glob.bounds("y")[1])
    assert exact.stats["max_symbols"] == 4 and exact.stats["max_box_elements"] == 4
    assert none.stats["max_box_elements"] == 0  # nothing matched: pure dense
    # all three give the same sound enclosure of x + n here
    for r in (exact, glob, none):
        np.testing.assert_allclose(r.bounds("y")[1], 1.5, rtol=1e-6)
    del rng


# ---- the paired two-model setting with one-sided noise -----------------------------------


def _noise_pair(rng, size=6, ch=4):
    """A float net and the same net with additive per-element noise at two sites.

    The noise inputs are graph inputs of BOTH models (the float one ignores them), exactly the
    way onnxsim.quant_verify lays out a quantizer's rounding noise.
    """
    inits = dict(
        W1=_f32(rng, ch, 3, 3, 3) * 0.4, B1=_f32(rng, ch) * 0.1,
        W2=_f32(rng, ch, ch, 3, 3) * 0.3, B2=_f32(rng, ch) * 0.1,
    )  # fmt: skip
    o = (size + 1) // 2
    sig = f"float[1,3,{size},{size}] x, float[1,{ch},{size},{size}] n0, float[1,{ch},{o},{o}] n1"
    out = f"float[1,{ch},{o},{o}] y"
    ref = _model(
        f"""m ({sig}) => ({out}) {{
              c = Conv<pads=[1,1,1,1]>(x, W1, B1)
              r = Relu(c)
              y = Conv<strides=[2,2], pads=[1,1,1,1]>(r, W2, B2)
            }}""",
        inits,
    )
    noisy = _model(
        f"""m ({sig}) => ({out}) {{
              c = Conv<pads=[1,1,1,1]>(x, W1, B1)
              cn = Add(c, n0)
              r = Relu(cn)
              d = Conv<strides=[2,2], pads=[1,1,1,1]>(r, W2, B2)
              y = Add(d, n1)
            }}""",
        inits,
    )
    return ref, noisy, (size, ch, o)


def _noise_ranges(size, ch, o, s0=0.08, s1=0.05):
    return {
        "x": (-1.0, 1.0),
        "n0": (np.full((1, ch, size, size), -s0), np.full((1, ch, size, size), s0)),
        "n1": (np.full((1, ch, o, o), -s1), np.full((1, ch, o, o), s1)),
    }


def _observed_difference(ref, noisy, rng, rngs, n=60):
    sa = ort.InferenceSession(
        ref.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    sb = ort.InferenceSession(
        noisy.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    worst = 0.0
    for i in range(n):
        feed = {}
        for name, (lo, hi) in rngs.items():
            shape = [
                d.dim_value
                for d in next(
                    v for v in ref.graph.input if v.name == name
                ).type.tensor_type.shape.dim
            ]
            lo = np.broadcast_to(np.asarray(lo, np.float64), shape)
            hi = np.broadcast_to(np.asarray(hi, np.float64), shape)
            if i % 3 == 0:  # box vertices are where a linear difference is largest
                feed[name] = np.where(rng.random(shape) < 0.5, lo, hi).astype(
                    np.float32
                )
            else:
                feed[name] = (lo + (hi - lo) * rng.random(shape)).astype(np.float32)
        worst = max(
            worst, float(np.abs(sa.run(None, feed)[0] - sb.run(None, feed)[0]).max())
        )
    return worst


@pytest.mark.parametrize("fresh", [None, False])
def test_difference_bound_with_boxed_noise_is_sound_and_not_much_looser_than_dense(
    fresh,
):
    rng = np.random.default_rng(7)
    ref, noisy, (size, ch, o) = _noise_pair(rng)
    rngs = _noise_ranges(size, ch, o)
    dense = Z.bound_difference(ref, noisy, rngs)
    boxed = Z.bound_difference(ref, noisy, rngs, box_inputs=["n*"], box_fresh=fresh)
    obs = _observed_difference(ref, noisy, rng, rngs)
    assert np.isfinite(boxed.worst)
    _no_fallback(boxed.notes)
    assert obs <= boxed.worst * (1 + SLACK) + SLACK, (obs, boxed.worst)
    assert obs <= dense.worst * (1 + SLACK) + SLACK
    # tightness is a measured property, not an assumption: pin a generous regression ceiling
    assert boxed.worst <= 3.0 * dense.worst, (boxed.worst, dense.worst)
    assert (
        boxed.worst >= dense.worst * 0.999
    )  # a box cannot beat the zonotope it contains


def test_boxing_noise_cuts_symbols_and_the_dense_generator_array():
    rng = np.random.default_rng(8)
    ref, noisy, (size, ch, o) = _noise_pair(rng, size=8, ch=8)
    rngs = _noise_ranges(size, ch, o)
    dense = Z.bound_difference(ref, noisy, rngs)
    boxed = Z.bound_difference(ref, noisy, rngs, box_inputs=["n*"])
    d, b = dense.stats["simplified"], boxed.stats["simplified"]
    assert b["max_symbols"] < 0.7 * d["max_symbols"], (b, d)
    assert b["max_generator_elements"] < 0.5 * d["max_generator_elements"], (b, d)
    assert b["max_box_elements"] > 0 and d["max_box_elements"] == 0


def test_a_box_that_both_models_read_does_not_cancel_but_stays_sound():
    rng = np.random.default_rng(9)
    model = _model(
        "m (float[1,3,5,5] x) => (float[1,4,5,5] y) { y = Conv<pads=[1,1,1,1]>(x, W) }",
        dict(W=_f32(rng, 4, 3, 3, 3)),
    )
    same = Z.bound_difference(model, model, BOX)
    boxed = Z.bound_difference(model, model, BOX, box_inputs=["x"])
    assert same.worst == 0.0  # dense: the shared symbols cancel exactly
    assert (
        boxed.worst > 0.0
    )  # boxed: documented loss -- list only one-sided noise in box_inputs
    assert boxed.bounded


# ---- QDQ-converted models from onnxsim's own quantizers ----------------------------------


def _quantized_cases():
    rng = np.random.default_rng(1)
    cnn = _model(
        "m (float[1,3,8,8] x) => (float[1,4,4,4] y) "
        "{ c = Conv(x, W1, B1) r = Relu(c) y = Conv(r, W2, B2) }",
        dict(
            W1=_f32(rng, 6, 3, 3, 3) * 0.5, B1=_f32(rng, 6) * 0.5,
            W2=_f32(rng, 4, 6, 3, 3) * 0.5, B2=_f32(rng, 4) * 0.5,
        ),
        opset=21,
        ir_version=9,
    )  # fmt: skip
    mlp = _model(
        "m (float[2,32] x) => (float[2,8] y) "
        "{ a = MatMul(x, W1) b = Add(a, B1) r = Relu(b) c = MatMul(r, W2) d = Relu(c) y = MatMul(d, W3) }",
        dict(
            W1=_f32(rng, 32, 64) * 0.5, B1=_f32(rng, 64) * 0.5,
            W2=_f32(rng, 64, 64) * 0.5, W3=_f32(rng, 64, 8) * 0.5,
        ),
        opset=21,
        ir_version=9,
    )  # fmt: skip
    return {"cnn": cnn, "mlp": mlp}


@pytest.mark.parametrize("scheme,dtype", [("static", "int8"), ("qoperator", "int8")])
@pytest.mark.parametrize("name", ["cnn", "mlp"])
@pytest.mark.parametrize("fresh", [True, False])
def test_box_engine_is_sound_on_real_quantized_models(
    monkeypatch, name, scheme, dtype, fresh
):
    model = _quantized_cases()[name]
    cfg = accuracy.QuantizationConfig(
        scheme=scheme, dtype=dtype, num_calibration_samples=16
    )
    q = accuracy.quantize(model, cfg)
    inp = model.graph.input[0].name
    rngs = {inp: (-1.0, 1.0)}
    real = Z.bound_difference

    def boxed(a, b, r=None, *x, **k):
        return real(a, b, r, *x, box_inputs=["__qnoise_*"], box_fresh=fresh, **k)

    dense_rep = quant_verify.verify(model, q, rngs, breakdown=False)
    monkeypatch.setattr(Z, "bound_difference", boxed)
    box_rep = quant_verify.verify(model, q, rngs, breakdown=False)
    observed = quant_verify.observed_error(model, q, rngs, n=60, adversarial=20)
    assert np.isfinite(box_rep.worst)
    assert observed <= box_rep.worst * (1 + SLACK) + SLACK, (observed, box_rep.worst)
    assert (
        box_rep.worst <= 4.0 * dense_rep.worst + 1e-9
    )  # measured ceiling, see the doc


# ---- fallbacks ---------------------------------------------------------------------------


def test_an_op_without_a_rule_falls_back_to_an_interval_box_including_the_box_part():
    model = _model("m (float[1,4] x) => (float[1,4] y) { y = Abs(x) }")
    res = Z.propagate(model, {"x": (-2.0, 1.0)}, box_inputs=["x"])
    lo, hi = res.bounds("y")
    assert np.all(lo <= 0.0 + 1e-9) and np.all(
        hi >= 2.0 - 1e-9
    )  # encloses |[-2, 1]| = [0, 2]
    assert any("precision lost at Abs" in n for n in res.notes)


def test_unbounded_input_is_still_rejected_not_boxed_into_nan():
    model = _model("m (float[1,4] x) => (float[1,4] y) { y = Relu(x) }")
    d = Z.bound_difference(model, model, {}, box_inputs=["x"])
    assert not d.bounded and not np.isnan(d.worst)


# ---- the doc must stay true --------------------------------------------------------------

_DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "zonotope-box-noise.md"
_BLOCK = re.compile(
    r"<!-- doctest -->\n```python\n(.*?)```\n```text\n(.*?)```", re.DOTALL
)


def _doc_blocks():
    return _BLOCK.findall(_DOC.read_text())


def test_the_doc_has_runnable_examples():
    assert len(_doc_blocks()) >= 2


@pytest.mark.parametrize("index", range(len(_doc_blocks())))
def test_doc_example_prints_what_the_doc_says(index):
    code, expected = _doc_blocks()[index]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, f"{_DOC.name}[{index}]", "exec"), {"__name__": "__doc__"})
    assert out.getvalue().strip() == expected.strip()
