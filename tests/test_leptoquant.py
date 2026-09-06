"""Tests for ``onnxsim.apply_leptoquant``/
``onnxsim.leptoquant.quantize_dequantize_block_fp8_leptoquant`` (see
``onnxsim/leptoquant.py``) -- AngelSlim's LeptoQuant outlier-aware block
FP8 weight scale search, a weight-only refinement of
``onnxsim.deepseek_fp8``'s own absmax-scaled 128x128 block FP8.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.deepseek_fp8 import quantize_dequantize_block_fp8
from onnxsim.leptoquant import quantize_dequantize_block_fp8_leptoquant

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


def _outlier_weight(rows=128, cols=128, seed=0, num_outliers=30):
    # A Laplacian-peaked bulk (exactly the distribution LeptoQuant's own
    # motivation describes) plus a small planted outlier population, one to
    # two orders of magnitude above the bulk: the absmax scale is then set
    # by those few elements, coarsening the FP8 grid every other (far more
    # numerous) element in the block has to land on.
    rng = np.random.default_rng(seed)
    w = rng.laplace(0.0, 0.05, size=(rows, cols))
    idx = rng.choice(w.size, size=num_outliers, replace=False)
    w.flat[idx] = rng.uniform(0.5, 2.0, num_outliers) * rng.choice(
        [-1.0, 1.0], num_outliers
    )
    return w


def test_leptoquant_block_round_trip_matches_deepseek_fp8_with_alpha_zero_grid():
    # alpha = 0 is literally deepseek_fp8's own absmax choice, so a
    # single-candidate grid forcing it must reproduce that pass exactly.
    rng = np.random.default_rng(0)
    w = rng.standard_normal((150, 140)) * 0.3

    lepto = quantize_dequantize_block_fp8_leptoquant(
        w, block_size=64, alpha_grid=(0.0,)
    )
    deepseek = quantize_dequantize_block_fp8(w, block_size=64)
    np.testing.assert_array_equal(lepto, deepseek)


def test_leptoquant_beats_absmax_block_fp8_on_outlier_weight():
    # The core empirical claim: on a weight with real outlier structure,
    # the per-block grid search finds a strictly better scale than the
    # block's own true absmax.
    w = _outlier_weight(seed=4)

    lepto = quantize_dequantize_block_fp8_leptoquant(w, block_size=128)
    deepseek = quantize_dequantize_block_fp8(w, block_size=128)

    lepto_mse = float(np.mean(np.square(w - lepto)))
    deepseek_mse = float(np.mean(np.square(w - deepseek)))
    assert lepto_mse < deepseek_mse


def test_leptoquant_never_worse_than_absmax_block_fp8_per_block():
    # alpha = 0 stays in the default grid, so no block can come out worse
    # than plain absmax scaling, whatever the weight looks like.
    rng = np.random.default_rng(2)
    w = np.vstack(
        [
            rng.standard_normal((128, 256)) * 0.01,
            _outlier_weight(rows=128, cols=256, seed=3),
        ]
    )

    lepto = quantize_dequantize_block_fp8_leptoquant(w, block_size=128)
    deepseek = quantize_dequantize_block_fp8(w, block_size=128)

    for r in range(0, w.shape[0], 128):
        for c in range(0, w.shape[1], 128):
            block = w[r : r + 128, c : c + 128]
            lepto_mse = np.mean(np.square(block - lepto[r : r + 128, c : c + 128]))
            deepseek_mse = np.mean(
                np.square(block - deepseek[r : r + 128, c : c + 128])
            )
            assert lepto_mse <= deepseek_mse


def test_leptoquant_block_padding_does_not_change_already_aligned_values():
    rng = np.random.default_rng(4)
    aligned = rng.standard_normal((128, 128))
    padded = np.zeros((150, 140))
    padded[:128, :128] = aligned

    dequant_aligned = quantize_dequantize_block_fp8_leptoquant(aligned, block_size=128)
    dequant_padded = quantize_dequantize_block_fp8_leptoquant(padded, block_size=128)
    np.testing.assert_array_equal(dequant_padded[:128, :128], dequant_aligned)


def test_apply_leptoquant_replaces_weight_without_adding_any_node():
    rng = np.random.default_rng(5)
    K, N = 128, 128
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_leptoquant(model)
    onnx.checker.check_model(q)

    # Weight-only: no Cast (or any other) node is inserted at all.
    assert [n.op_type for n in q.graph.node] == [n.op_type for n in model.graph.node]
    assert [n.op_type for n in q.graph.node] == ["MatMul"]

    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert not np.array_equal(new_w, w)


def test_apply_leptoquant_output_stays_close_to_float_via_onnxruntime():
    rng = np.random.default_rng(6)
    K, N = 256, 128
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_leptoquant(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_apply_leptoquant_gemm_with_bias():
    rng = np.random.default_rng(7)
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
    q = onnxsim.apply_leptoquant(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_apply_leptoquant_gemm_transb():
    rng = np.random.default_rng(8)
    K, N = 128, 64
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    q = onnxsim.apply_leptoquant(model)
    onnx.checker.check_model(q)

    new_w = _current_weight(q)
    assert new_w.shape == weight.shape

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.2


def test_apply_leptoquant_respects_skip_names():
    rng = np.random.default_rng(9)
    K, N = 128, 128
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_leptoquant(model, skip_names={"W"})
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_leptoquant_noop_when_weight_is_not_constant():
    model = _model(
        """
        g (float[4,8] X, float[8,8] W) => (float[4,8] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    result = onnxsim.apply_leptoquant(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_leptoquant_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_leptoquant(model)
    assert result.SerializeToString() == model.SerializeToString()
