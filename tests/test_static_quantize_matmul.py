"""Tests for ``onnxsim.quantize_static`` (the ``static_quantize_matmul`` C++
pass) and its calibration-data helpers in ``onnxsim.calibration``.

Each model is built directly with ``onnx.parser`` (no torch dependency),
calibrated with random data, quantized, and then actually run through ONNX
Runtime -- both before and after quantization -- so these tests double as a
minimal end-to-end calibrate/quantize/deploy check: the quantized graph must
load and execute under a real inference engine, and its outputs must stay
close to the float baseline.
"""

import collections

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=10):
    # Pin a low IR version so the model loads under older onnxruntime builds
    # (which cap at IR version 11), matching test_fusion_patterns.py.
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
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _assert_close(float_outputs, quant_outputs):
    # INT8/uint8 quantization is lossy by design; see
    # test_dynamic_quantize_matmul.py's identically-named helper for why this
    # checks aggregate relative L2 error rather than a tight per-element bound.
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < 0.1, f"relative L2 error too large: {rel_l2:.4f}"


def _assert_explicit_weight_zp(quant, out_channels):
    # The weight's DequantizeLinear spells its (all-zeros, symmetric) INT8
    # zero-point out explicitly, same shape as the per-channel scale: runtimes
    # that fuse the QDQ pattern into an integer kernel (ORT's QGemm, the
    # VitisAI EP's MatMulAddFusion) require scale and zero_point to match and
    # reject/abort the fused node when it is omitted.
    inits = {t.name: t for t in quant.graph.initializer}
    wdqs = [
        n
        for n in quant.graph.node
        if n.op_type == "DequantizeLinear"
        and n.input[0] in inits  # weight branch (activation DQs read a node)
    ]
    assert len(wdqs) == 1
    assert len(wdqs[0].input) == 3
    wq = onnx.numpy_helper.to_array(inits[wdqs[0].input[0]])
    ws = onnx.numpy_helper.to_array(inits[wdqs[0].input[1]])
    wzp = onnx.numpy_helper.to_array(inits[wdqs[0].input[2]])
    assert wq.dtype == np.int8
    assert ws.shape == (out_channels,)
    assert wzp.dtype == np.int8
    assert wzp.shape == ws.shape
    assert bool((wzp == 0).all())


def test_quantize_matmul():
    rng = np.random.default_rng(0)
    K, N = 32, 16
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    model = _model(
        """
        g (float[4,32] X) => (float[4,16] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[weight],
    )

    quant = onnxsim.quantize_static(model, num_calibration_samples=16, seed=0)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 1  # the MatMul node itself is kept (QDQ format)
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 2  # one for X, one for W
    _assert_explicit_weight_zp(quant, N)

    x = rng.standard_normal((4, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_matmul_add_fusion_runs():
    # MatMul+Add is what ORT fuses into an integer QGemm at load time -- the
    # exact shape that failed before the weight zero-point was spelled out
    # ("zero point and scale of input b should have same shape size"), and
    # what the VitisAI EP fuses into MatMulAddFusion (it aborted partitioning
    # the 2-input form). The quantized model must load *and execute*.
    rng = np.random.default_rng(2)
    K, N = 32, 16
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    bias = _f32(rng.standard_normal(N) * 0.5, "B")
    model = _model(
        """
        g (float[4,32] X) => (float[4,16] Y)
        {
          T = MatMul(X, W)
          Y = Add(T, B)
        }
        """,
        initializer=[weight, bias],
    )

    quant = onnxsim.quantize_static(model, num_calibration_samples=16, seed=2)
    onnx.checker.check_model(quant)
    _assert_explicit_weight_zp(quant, N)

    x = rng.standard_normal((4, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_gemm_transb_with_bias():
    # PyTorch's nn.Linear layout: weight is [out_features, in_features], i.e.
    # [N, K], exported as Gemm(X, W, B, transB=1) -- the common real-world case.
    rng = np.random.default_rng(1)
    K, N = 24, 12
    weight = _f32(rng.standard_normal((N, K)) * 0.5, "W")
    bias = _f32(rng.standard_normal(N), "B")
    model = _model(
        """
        g (float[3,24] X) => (float[3,12] Y)
        {
          Y = Gemm<transB = 1>(X, W, B)
        }
        """,
        initializer=[weight, bias],
    )

    quant = onnxsim.quantize_static(model, num_calibration_samples=16, seed=1)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    # The Gemm node itself (bias included) is kept; only its X/W inputs are
    # rerouted through QDQ pairs.
    assert ops["Gemm"] == 1
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 2

    x = rng.standard_normal((3, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_skips_non_constant_weight():
    # Both MatMul operands are graph inputs (neither is a constant), so there
    # is nothing to quantize ahead of time.
    model = _model(
        """
        g (float[4,8] X, float[8,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    quant = onnxsim.quantize_static(model)
    assert _op_counts(quant)["MatMul"] == 1


def test_quantize_skips_non_default_gemm_attrs():
    # alpha != 1 falls outside the "vanilla" Gemm shape this pass handles.
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    model = _model(
        """
        g (float[4,8] X) => (float[4,4] Y)
        {
          Y = Gemm<alpha = 2.0>(X, W)
        }
        """,
        initializer=[weight],
    )
    quant = onnxsim.quantize_static(model)
    assert _op_counts(quant)["Gemm"] == 1


def test_quantize_rejects_old_opset():
    # DequantizeLinear's per-channel `axis` attribute needs opset >= 13.
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    model = _model(
        """
        g (float[4,8] X) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[weight],
        opset=12,
    )
    # Below opset 13 the QDQ rewrite cannot run, so this is an error, not a
    # silently unquantized model.
    with pytest.raises(ValueError, match="opset >= 13"):
        onnxsim.quantize_static(model)


def test_generate_random_calibration_data_shapes():
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    model = _model(
        """
        g (float[2,8] X) => (float[2,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[weight],
    )
    batches = onnxsim.generate_random_calibration_data(model, num_samples=3, seed=0)
    assert len(batches) == 3
    for batch in batches:
        assert set(batch.keys()) == {"X"}
        assert batch["X"].shape == (2, 8)
        assert batch["X"].dtype == np.float32


def test_generate_random_calibration_data_keeps_static_zero_dim():
    # A genuinely static, zero-length dimension (e.g. an empty KV-cache
    # sentinel some exporters emit -- see pocket-tts's flow_lm_main.onnx,
    # whose "state_1" input is exactly this) must survive as 0, not get
    # silently promoted to 1: `dim.dim_value` reads back as 0 both when a
    # dim is genuinely fixed to 0 and when it's unset/symbolic, so telling
    # them apart requires `HasField("dim_value")`, not a bare `> 0` check.
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    # X's second dim is dynamic (symbolic "seq"); "state" has a real,
    # static empty dimension at index 1.
    # "state" is unused by any node -- fine, only its declared shape matters
    # for _input_specs, which looks at model.graph.input directly.
    model = _model(
        """
        g (float[2,seq] X, float[1,0,4] state) => (float[2,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[weight],
    )

    batches = onnxsim.generate_random_calibration_data(model, num_samples=1, seed=0)
    assert batches[0]["X"].shape == (2, 1)  # symbolic dim -> defaults to 1
    assert batches[0]["state"].shape == (1, 0, 4)  # static 0 dim -> stays 0


def test_calibrate_returns_ranges_for_quantizable_tensors():
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    model = _model(
        """
        g (float[2,8] X) => (float[2,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[weight],
    )
    data = onnxsim.generate_random_calibration_data(model, num_samples=4, seed=0)
    ranges = onnxsim.calibrate(model, data)
    assert set(ranges.keys()) == {"X"}
    lo, hi = ranges["X"]
    assert lo < hi
