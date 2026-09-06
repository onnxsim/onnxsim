"""Tests for ``onnxsim.apply_deepseek_fp8``/``onnxsim.quantize_dequantize_block_fp8``
(see ``onnxsim/deepseek_fp8.py``) -- DeepSeek-V3-style fine-grained block
FP8 quantization (128x128 weight blocks, per-token-per-128-channel-group
activation scaling), the format SGLang/vLLM both ship as their own
"fp8_w8a8" DeepSeek-V3/R1 serving format.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21, ir_version=10):
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


def _matmul_model(w, K, N, batch="batch", opset=21):
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
        opset=opset,
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9)


def _current_weight(model, weight_input_index=1):
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def test_block_fp8_round_trip_has_at_most_one_scale_per_128_block():
    rng = np.random.default_rng(0)
    # Two very different magnitude regimes in different 128x128 blocks --
    # one shared per-tensor scale would badly under-resolve one of them.
    w = np.zeros((256, 256))
    w[:128, :128] = rng.standard_normal((128, 128)) * 0.01
    w[128:, 128:] = rng.standard_normal((128, 128)) * 100.0

    dequant = onnxsim.quantize_dequantize_block_fp8(w, block_size=128)
    assert dequant.shape == w.shape
    # The small-magnitude block should still resolve to non-zero, distinct
    # values -- if a single global scale (sized for the 100x block) were
    # used instead, this block would quantize to all-zero.
    assert np.any(dequant[:128, :128] != 0.0)
    assert len(np.unique(dequant[:128, :128])) > 4


def test_block_fp8_beats_naive_per_tensor_fp8_on_mixed_magnitude_weight():
    # The core empirical claim: block-wise scaling reduces reconstruction
    # error vs. a single per-tensor FP8 scale, on a weight whose magnitude
    # varies a lot block-to-block (exactly DeepSeek-V3's own motivation).
    from onnxsim.deepseek_fp8 import _fp8_round_trip

    rng = np.random.default_rng(1)
    w = np.zeros((256, 256))
    for i in range(2):
        for j in range(2):
            scale = rng.uniform(0.01, 50.0)
            w[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = (
                rng.standard_normal((128, 128)) * scale
            )

    block_dequant = onnxsim.quantize_dequantize_block_fp8(w, block_size=128)
    per_tensor_scale = max(float(np.max(np.abs(w))), 1e-12) / 448.0
    naive_dequant = _fp8_round_trip(w / per_tensor_scale) * per_tensor_scale

    block_err = np.linalg.norm(w - block_dequant)
    naive_err = np.linalg.norm(w - naive_dequant)
    assert block_err < naive_err


def test_block_fp8_padding_does_not_change_already_aligned_values():
    rng = np.random.default_rng(2)
    aligned = rng.standard_normal((128, 128))
    padded = np.zeros((150, 140))
    padded[:128, :128] = aligned

    dequant_aligned = onnxsim.quantize_dequantize_block_fp8(aligned, block_size=128)
    dequant_padded = onnxsim.quantize_dequantize_block_fp8(padded, block_size=128)
    np.testing.assert_array_equal(dequant_padded[:128, :128], dequant_aligned)


def test_apply_deepseek_fp8_replaces_weight_and_inserts_activation_cast():
    rng = np.random.default_rng(3)
    K, N = 128, 128
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_deepseek_fp8(model)
    onnx.checker.check_model(q)

    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert not np.array_equal(new_w, w)

    cast_nodes = [n for n in q.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 2
    to_attrs = {a.i for n in cast_nodes for a in n.attribute if a.name == "to"}
    assert onnx.TensorProto.FLOAT8E4M3FN in to_attrs


def test_apply_deepseek_fp8_output_stays_close_to_float_via_onnxruntime():
    rng = np.random.default_rng(4)
    K, N = 256, 128
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_deepseek_fp8(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_apply_deepseek_fp8_gemm_with_bias():
    rng = np.random.default_rng(5)
    K, N = 128, 64
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.4
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.apply_deepseek_fp8(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_apply_deepseek_fp8_respects_skip_names():
    rng = np.random.default_rng(6)
    K, N = 128, 128
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_deepseek_fp8(model, skip_names={"W"})
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_deepseek_fp8_noop_below_opset19():
    rng = np.random.default_rng(7)
    K, N = 128, 128
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N, opset=13)

    result = onnxsim.apply_deepseek_fp8(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_deepseek_fp8_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_deepseek_fp8(model)
    assert result.SerializeToString() == model.SerializeToString()
