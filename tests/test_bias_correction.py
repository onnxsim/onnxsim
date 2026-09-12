"""Tests for ``onnxsim.correct_bias`` (``onnxsim/bias_correction.py``) --
AIMET's empirical Bias Correction.

The most important test here isn't against a realistic quantized model --
real per-channel symmetric weight quantization tends to round fairly
symmetrically already, so the bias it leaves behind can be too small to
distinguish from measurement noise in a small hand-built test. Instead,
``test_correct_bias_recovers_known_injected_gemm_bias`` (and its Conv
counterpart) fabricate a "quantized" model that is the float model with a
*known* per-channel bias error injected directly, and check that
``correct_bias`` recovers it almost exactly -- a precise check of the
measurement-and-graph-surgery mechanism itself, independent of whether any
particular quantize_* scheme happens to leave a large enough bias to see.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim import backend

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=8):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _gemm_model(w, b, K, N, batch=4):
    return _model(
        f"""
        g (float[{batch},{K}] x) => (float[{batch},{N}] y)
        {{
          y = Gemm(x, w, b)
        }}
        """,
        initializer=[_f32(w, "w"), _f32(b, "b")],
        opset=17,
    )


def _conv_model(w, b, c_in, c_out, batch=1, spatial=8):
    return _model(
        f"""
        g (float[{batch},{c_in},{spatial},{spatial}] x) => (float[{batch},{c_out},{spatial},{spatial}] y)
        {{
          y = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(x, w, b)
        }}
        """,
        initializer=[_f32(w, "w"), _f32(b, "b")],
        opset=13,
    )


def test_correct_bias_recovers_known_injected_gemm_bias():
    rng = np.random.default_rng(0)
    K, N = 16, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    b = rng.standard_normal(N).astype(np.float32) * 0.05
    injected_error = np.array(
        [0.5, -0.3, 0.2, 0.1, -0.4, 0.05, 0.15, -0.1], dtype=np.float32
    )
    float_model = _gemm_model(w, b, K, N)
    fake_quantized = _gemm_model(w, (b + injected_error).astype(np.float32), K, N)

    rng2 = np.random.default_rng(1)
    calib = [{"x": rng2.standard_normal((4, K)).astype(np.float32)} for _ in range(8)]

    corrected = onnxsim.correct_bias(
        float_model, fake_quantized, calibration_data=calib
    )
    onnx.checker.check_model(corrected)
    assert [n.op_type for n in corrected.graph.node] == ["Gemm", "Add"]

    before = onnxsim.measure_accuracy_drop(
        float_model, fake_quantized, calibration_data=calib
    )
    after = onnxsim.measure_accuracy_drop(
        float_model, corrected, calibration_data=calib
    )
    assert after.worst_relative_l2 < before.worst_relative_l2 * 0.01
    assert after.worst_relative_l2 < 1e-5


def _resize_model(mode, c=4, spatial=4, opset=19, ir_version=9):
    scaled = spatial * 2
    return _model(
        f"""
        g (float[1,{c},{spatial},{spatial}] x) => (float[1,{c},{scaled},{scaled}] y)
        {{
          scales = Constant<value = float[4] {{1.0, 1.0, 2.0, 2.0}}>()
          y = Resize<mode = "{mode}">(x, , scales)
        }}
        """,
        opset=opset,
        ir_version=ir_version,
    )


def test_correct_bias_recovers_resize_mode_swap_bias():
    # Not a quantization scenario: `swapped` is `float_model` with its
    # Resize's interpolation mode changed (e.g. because a deployment
    # accelerator doesn't implement "linear"), which -- like quantization
    # rounding -- leaves a systematic per-channel mean shift that
    # correct_bias should measure and cancel.
    float_model = _resize_model("linear")
    swapped = _resize_model("nearest")

    rng = np.random.default_rng(10)
    calib = [
        {"x": rng.standard_normal((1, 4, 4, 4)).astype(np.float32)} for _ in range(16)
    ]

    corrected = onnxsim.correct_bias(float_model, swapped, calibration_data=calib)
    onnx.checker.check_model(corrected)
    assert [n.op_type for n in corrected.graph.node] == [
        "Constant",
        "Resize",
        "Add",
    ]

    before = onnxsim.measure_accuracy_drop(
        float_model, swapped, calibration_data=calib
    )
    after = onnxsim.measure_accuracy_drop(
        float_model, corrected, calibration_data=calib
    )
    # A per-channel constant can't undo a genuinely different resampling
    # algorithm, only its systematic (mean) component -- so this only
    # checks that correction never makes the measured error worse, not
    # that it drives it to zero the way the injected-bias tests do.
    assert after.worst_relative_l2 <= before.worst_relative_l2 + 1e-9


def test_correct_bias_recovers_known_injected_conv_bias():
    rng = np.random.default_rng(2)
    c_in, c_out = 4, 6
    w = rng.standard_normal((c_out, c_in, 3, 3)).astype(np.float32) * 0.1
    b = rng.standard_normal(c_out).astype(np.float32) * 0.02
    injected_error = np.array([0.3, -0.2, 0.15, -0.1, 0.25, -0.05], dtype=np.float32)
    float_model = _conv_model(w, b, c_in, c_out)
    fake_quantized = _conv_model(
        w, (b + injected_error).astype(np.float32), c_in, c_out
    )

    rng2 = np.random.default_rng(3)
    calib = [
        {"x": rng2.standard_normal((1, c_in, 8, 8)).astype(np.float32)}
        for _ in range(8)
    ]

    corrected = onnxsim.correct_bias(
        float_model, fake_quantized, calibration_data=calib
    )
    onnx.checker.check_model(corrected)
    assert [n.op_type for n in corrected.graph.node] == ["Conv", "Add"]

    before = onnxsim.measure_accuracy_drop(
        float_model, fake_quantized, calibration_data=calib
    )
    after = onnxsim.measure_accuracy_drop(
        float_model, corrected, calibration_data=calib
    )
    assert after.worst_relative_l2 < before.worst_relative_l2 * 0.01
    assert after.worst_relative_l2 < 1e-5


def test_correct_bias_end_to_end_with_real_quantization():
    rng = np.random.default_rng(4)
    K, N = 64, 32
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    b = rng.standard_normal(N).astype(np.float32) * 0.05
    float_model = _gemm_model(w, b, K, N, batch=8)
    quantized = onnxsim.quantize_weight_only(float_model)
    onnx.checker.check_model(quantized)

    rng2 = np.random.default_rng(5)
    calib = [{"x": rng2.standard_normal((8, K)).astype(np.float32)} for _ in range(16)]

    corrected = onnxsim.correct_bias(float_model, quantized, calibration_data=calib)
    onnx.checker.check_model(corrected)

    before = onnxsim.measure_accuracy_drop(
        float_model, quantized, calibration_data=calib
    )
    after = onnxsim.measure_accuracy_drop(
        float_model, corrected, calibration_data=calib
    )
    # A correction fit to this exact calibration data should never make the
    # measured-on-the-same-data error worse.
    assert after.worst_relative_l2 <= before.worst_relative_l2 + 1e-9


def test_correct_bias_is_a_noop_when_quantized_model_is_identical():
    rng = np.random.default_rng(6)
    K, N = 8, 4
    w = rng.standard_normal((K, N)).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    model = _gemm_model(w, b, K, N)

    corrected = onnxsim.correct_bias(model, model)
    assert [n.op_type for n in corrected.graph.node] == ["Gemm"]


def test_correct_bias_is_a_noop_on_a_model_with_no_candidate_ops():
    model = _model(
        """
        g (float[2,4] x) => (float[2,4] y)
        {
          y = Relu(x)
        }
        """
    )

    corrected = onnxsim.correct_bias(model, model)
    assert [n.op_type for n in corrected.graph.node] == ["Relu"]


def test_correct_bias_generates_calibration_data_when_omitted():
    rng = np.random.default_rng(7)
    K, N = 8, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    b = rng.standard_normal(N).astype(np.float32) * 0.02
    float_model = _gemm_model(w, b, K, N)
    fake_quantized = _gemm_model(w, (b + np.full(N, 0.4, dtype=np.float32)), K, N)

    corrected = onnxsim.correct_bias(float_model, fake_quantized, num_samples=8, seed=0)
    onnx.checker.check_model(corrected)
    assert [n.op_type for n in corrected.graph.node] == ["Gemm", "Add"]


def test_correct_bias_preserves_existing_graph_output():
    # The corrected node's output is also a genuine graph output (not just
    # an intermediate) -- the correction Add node must take over that name
    # so the graph output still resolves to the corrected value.
    rng = np.random.default_rng(8)
    K, N = 8, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    b = rng.standard_normal(N).astype(np.float32) * 0.02
    injected_error = np.full(N, 0.6, dtype=np.float32)
    float_model = _gemm_model(w, b, K, N)
    fake_quantized = _gemm_model(w, (b + injected_error).astype(np.float32), K, N)
    assert fake_quantized.graph.output[0].name == "y"

    rng2 = np.random.default_rng(9)
    calib = [{"x": rng2.standard_normal((4, K)).astype(np.float32)} for _ in range(8)]
    corrected = onnxsim.correct_bias(
        float_model, fake_quantized, calibration_data=calib
    )
    onnx.checker.check_model(corrected)
    assert corrected.graph.output[0].name == "y"

    (out,) = backend.run_model(corrected, calib[0]).values()
    (expected,) = backend.run_model(float_model, calib[0]).values()
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)
