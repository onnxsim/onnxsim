"""Weight-residual splitting for the AX650N, checked offline.

`scripts/axera/precision.py` splits a weight into an INT8 part and an INT8
residual so that two INT8 convolutions carry about sixteen bits of weight
between them. The tests here pin the two facts a hardware probe had to teach
this module the hard way -- the scale is per output *channel*, and the
residual has to be taken against what the compiler will re-quantise the high
half to, not against the high half as written.

Neither needs Docker or a card.
"""

import os
import sys

import numpy as np
import onnx
import onnx.parser
from onnx import TensorProto, helper, numpy_helper

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import precision  # noqa: E402


def _heavy_tailed(shape, seed=3):
    """A weight with the outlier tail a real conv weight has (peak/rms ~ 11)."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape).astype(np.float32)
    w *= rng.standard_gamma(0.35, shape).astype(np.float32)
    return w


def _snr(ref, got):
    ref = np.asarray(ref, np.float64)
    err = ref - np.asarray(got, np.float64)
    return 10 * np.log10((ref ** 2).sum() / max((err ** 2).sum(), 1e-30))


def _model(body, initializer=(), opset=17, ir_version=10):
    model = onnx.parser.parse_model(
        f"<ir_version: {ir_version}, opset_import: [\"\": {opset}]> {body}"
    )
    model.graph.initializer.extend(initializer)
    return model


def _conv_model(w, b=None):
    inits = [numpy_helper.from_array(w, "w")]
    sig = "float[1, %d, 8] x" % w.shape[1]
    if b is None:
        body = f"""
            g ({sig}) => (float[1, {w.shape[0]}, 8] y) {{
                y = Conv <kernel_shape = [3], pads = [1, 1]> (x, w)
            }}
        """
    else:
        inits.append(numpy_helper.from_array(b, "b"))
        body = f"""
            g ({sig}) => (float[1, {w.shape[0]}, 8] y) {{
                y = Conv <kernel_shape = [3], pads = [1, 1]> (x, w, b)
            }}
        """
    return _model(body, inits)


def test_the_weight_quantiser_is_per_channel_at_peak_over_127_5():
    """The scale Pulsar2 writes into `quant_axmodel.json` for a conv weight is
    `max|w_c| / 127.5`, one per output channel, zero point 0."""
    w = _heavy_tailed((8, 4, 3))
    got = precision.quantise_dequantise(w, axis=0)
    step = np.abs(w).max(axis=(1, 2)) / 127.5
    for c in range(w.shape[0]):
        codes = got[c] / step[c]
        assert np.allclose(codes, np.rint(codes), atol=1e-3)
        assert codes.min() >= -128 and codes.max() <= 127
    # per channel, not per tensor: a channel far below the global peak keeps
    # its own resolution
    quiet = w.copy()
    quiet[0] *= 1e-3
    per_channel = precision.quantise_dequantise(quiet, axis=0)
    per_tensor = precision.quantise_dequantise(quiet, axis=None)
    assert _snr(quiet[0], per_channel[0]) > _snr(quiet[0], per_tensor[0]) + 40


def test_the_quantiser_is_not_idempotent():
    """A quantised channel peaks at 127 steps where the scale assumed 127.5, so
    re-quantising moves values by up to half a step. This is the whole reason
    `compiler_view` exists."""
    w = _heavy_tailed((16, 8, 3))
    once = precision.quantise_dequantise(w, axis=0)
    twice = precision.compiler_view(once, axis=0)
    assert not np.array_equal(once, twice)
    step = np.abs(w).max(axis=(1, 2), keepdims=True) / 127.5
    assert np.abs(twice - once).max() <= 0.51 * step.max()


def test_split_carries_far_more_than_one_int8_pass():
    """Two INT8 weights reconstruct the float weight to about sixteen bits."""
    w = _heavy_tailed((32, 32, 3))
    plain = _snr(w, precision.quantise_dequantise(w, axis=0))
    hi, lo = precision.split_weight(w, axis=0)
    got = precision.compiler_view(hi, axis=0) + precision.quantise_dequantise(lo, axis=0)
    assert 35 < plain < 42, plain
    assert _snr(w, got) > plain + 35


def test_the_residual_must_be_taken_against_the_compilers_view():
    """`W - quantise(W)` is the intuitive residual and it throws away most of
    the gain -- the compiler re-rounds the high half, and that half-step is
    left uncorrected. A hardware probe built this way measured no gain at all."""
    w = _heavy_tailed((32, 32, 3))
    hi = precision.quantise_dequantise(w, axis=0)
    naive_lo = w - hi
    naive = precision.compiler_view(hi, axis=0) + precision.quantise_dequantise(
        naive_lo, axis=0)
    _, lo = precision.split_weight(w, axis=0)
    right = precision.compiler_view(hi, axis=0) + precision.quantise_dequantise(lo, axis=0)
    assert _snr(w, right) > _snr(w, naive) + 25


def test_channel_axis_follows_each_ops_weight_layout():
    """`Conv` is (Cout, Cin, k) but `ConvTranspose` is (Cin, Cout, k), and
    `Gemm`'s output axis moves with `transB`."""
    conv = helper.make_node("Conv", ["x", "w"], ["y"])
    deconv = helper.make_node("ConvTranspose", ["x", "w"], ["y"])
    matmul = helper.make_node("MatMul", ["x", "w"], ["y"])
    gemm_t = helper.make_node("Gemm", ["x", "w"], ["y"], transB=1)
    gemm_n = helper.make_node("Gemm", ["x", "w"], ["y"])
    assert precision.channel_axis(conv, 3) == 0
    assert precision.channel_axis(deconv, 3) == 1
    assert precision.channel_axis(matmul, 2) == 1
    assert precision.channel_axis(gemm_t, 2) == 0
    assert precision.channel_axis(gemm_n, 2) == 1


def test_a_conv_becomes_two_convs_joined_by_an_add():
    w = _heavy_tailed((6, 4, 3))
    model = _conv_model(w)
    assert precision.weight_residual_split(model) == 1
    kinds = [n.op_type for n in model.graph.node]
    assert kinds == ["Conv", "Conv", "Add"]
    hi, lo, add = model.graph.node
    assert add.input == list(hi.output) + list(lo.output)
    assert add.output[0] == "y"
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    got = precision.compiler_view(inits[hi.input[1]], axis=0) + \
        precision.quantise_dequantise(inits[lo.input[1]], axis=0)
    assert _snr(w, got) > 70
    onnx.checker.check_model(model)


def test_the_split_graph_runs_and_stays_within_half_an_int8_step():
    """Run in float the split is *not* exact -- the residual is deliberately
    taken against the compiler's re-rounding of the high half, so the two
    halves sum to `W` plus that half-step. What matters is that the miss is
    smaller than the INT8 pass it replaces, and that onnxruntime accepts the
    graph."""
    ort = __import__("onnxruntime")
    w = _heavy_tailed((6, 4, 3))
    before = _conv_model(w)
    after = _conv_model(w)
    assert precision.weight_residual_split(after) == 1
    x = np.random.default_rng(11).standard_normal((1, 4, 8)).astype(np.float32)
    run = lambda m: ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    assert run(after).shape == run(before).shape
    # the miss is exactly the compiler's re-rounding of the high half, which is
    # bounded by half an INT8 step -- the same size as the error a plain INT8
    # build makes, and it disappears once the pair is quantised
    inits = {i.name: numpy_helper.to_array(i) for i in after.graph.initializer}
    hi, lo, _ = after.graph.node
    step = np.abs(w).max(axis=(1, 2), keepdims=True) / 127.5
    delta = inits[hi.input[1]] + inits[lo.input[1]] - w
    assert np.abs(delta).max() <= 0.51 * step.max()
    assert _snr(run(before), run(after)) > 40


def test_the_bias_rides_on_the_high_half_only():
    """Copying the bias to both halves would double it."""
    w = _heavy_tailed((6, 4, 3))
    b = np.arange(6, dtype=np.float32) + 1.0
    model = _conv_model(w, b)
    assert precision.weight_residual_split(model) == 1
    hi, lo, _ = model.graph.node
    assert len(hi.input) == 3 and hi.input[2] == "b"
    assert len(lo.input) == 2


def test_min_peak_to_rms_leaves_well_behaved_weights_alone():
    """A weight with no outlier tail has little to gain, and the filter is the
    calibration-free way to say so."""
    rng = np.random.default_rng(5)
    tame = rng.standard_normal((8, 8, 3)).astype(np.float32)
    assert precision.peak_to_rms(tame, 0) < 5.0
    model = _conv_model(tame)
    assert precision.weight_residual_split(model, min_peak_to_rms=6.0) == 0
    assert [n.op_type for n in model.graph.node] == ["Conv"]


def test_weight_error_db_predicts_the_measured_loss():
    """The closed form is what picks layers without running calibration; it has
    to track the real number, not just rank layers."""
    for shape, seed in (((32, 32, 3), 3), ((16, 64, 1), 7), ((64, 8, 3), 11)):
        w = _heavy_tailed(shape, seed)
        got = precision.quantise_dequantise(w, axis=0)
        # the weight's own SNR is the output SNR the quantiser costs, because
        # both ride the same activations
        assert abs(precision.weight_error_db(w, axis=0) - _snr(w, got)) < 2.0


def test_split_op_types_names_the_add_that_must_stay_wide():
    """The join is the op that undoes the split if it requantises to INT8, so
    it belongs in the `layer_configs` entry the caller writes."""
    model = _conv_model(_heavy_tailed((6, 4, 3)))
    assert precision.split_op_types(model) == ["Add", "Conv"]


def test_a_matmul_splits_on_its_last_axis():
    w = _heavy_tailed((4, 6))
    model = _model(
        """
            g (float[2, 4] x) => (float[2, 6] y) {
                y = MatMul (x, w)
            }
        """,
        [numpy_helper.from_array(w, "w")],
    )
    assert precision.weight_residual_split(model) == 1
    assert [n.op_type for n in model.graph.node] == ["MatMul", "MatMul", "Add"]
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    hi, lo, _ = model.graph.node
    got = precision.compiler_view(inits[hi.input[1]], axis=1) + \
        precision.quantise_dequantise(inits[lo.input[1]], axis=1)
    assert _snr(w, got) > 70
