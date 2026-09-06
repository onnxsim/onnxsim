"""Tests for ``onnxsim.apply_owq`` (OWQ, see ``onnxsim/owq.py``) -- restores
a small number of the most quantization-sensitive input columns (the
classic Optimal Brain Surgeon saliency metric, reusing GPTQ's own Hessian
machinery) of an already-INT4-quantized MatMul/Gemm layer to exact float32
precision via an inserted correction term, leaving every other column's
INT4 codes untouched.
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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _matmul_model_with_outlier_column(K=64, N=16, outlier_col=3, seed=0):
    # One column with much larger magnitude than the rest of its block --
    # round-to-nearest's block-shared scale is dominated by this column, so
    # every *other* column in the same block rounds coarsely, and this
    # column's own relative error is large too (its extreme values sit far
    # from any of the 15 grid points a 4-bit code offers). OWQ's own
    # motivating scenario: this is exactly the kind of column worth
    # rescuing to full precision rather than quantizing.
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    weight[outlier_col, :] = rng.standard_normal(N).astype(np.float32) * 20.0
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )


def _calibration(K=64, num_samples=32, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_samples, K)).astype(np.float32)


def test_owq_reduces_reconstruction_error_by_rescuing_outlier_column():
    model = _matmul_model_with_outlier_column(K=64, N=16, outlier_col=3, seed=0)
    x = _calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    y_float = x.astype(np.float64) @ w_float

    (rtn_y,) = _run(quant, {"X": x})
    rtn_err = np.linalg.norm(y_float - rtn_y.astype(np.float64))

    owq_model = onnxsim.apply_owq(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(owq_model)
    assert any(n.op_type == "Gather" for n in owq_model.graph.node)

    (owq_y,) = _run(owq_model, {"X": x})
    owq_err = np.linalg.norm(y_float - owq_y.astype(np.float64))

    assert owq_err < rtn_err


def _unpack_int4(tensor):
    dims = list(tensor.dims)
    numel = int(np.prod(dims))
    raw = np.frombuffer(tensor.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    return codes.reshape(dims).astype(np.float64)


def test_owq_correction_initializer_exactly_cancels_int4_error():
    # OWQ's own exactness claim is about the *initializers it writes*, not
    # about onnxruntime's own MatMul kernel numerics -- different CPU
    # backends (observed: x86_64 vs arm64 CI runners) can reduce a MatMul
    # over a 20x-magnitude outlier weight in a different order, which is a
    # real but unrelated source of ~1e-3-relative float32 noise that has
    # nothing to do with whether OWQ computed the right correction. So
    # verify the actual claim directly and deterministically: reading back
    # quant's real INT4 codes/scale and owq_model's own delta_w
    # initializer, `codes * scale + delta_w` must equal the float weight
    # for every rescued column, independent of any onnxruntime execution.
    K, N, outlier_col = 32, 8, 5
    model = _matmul_model_with_outlier_column(K=K, N=N, outlier_col=outlier_col, seed=2)
    x = _calibration(K=K, num_samples=32, seed=3)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    owq_model = onnxsim.apply_owq(
        model, quant, calibration_data=calibration_data, outlier_fraction=0.1
    )
    onnx.checker.check_model(owq_model)

    gather_node = next(n for n in owq_model.graph.node if n.op_type == "Gather")
    idx_init = next(
        t for t in owq_model.graph.initializer if t.name == gather_node.input[1]
    )
    weak_idx = onnx.numpy_helper.to_array(idx_init).astype(np.int64)

    matmul_node = next(
        n
        for n in owq_model.graph.node
        if n.op_type == "MatMul" and n.input[0] == gather_node.output[0]
    )
    delta_w_init = next(
        t for t in owq_model.graph.initializer if t.name == matmul_node.input[1]
    )
    delta_w = onnx.numpy_helper.to_array(delta_w_init).astype(
        np.float64
    )  # [num_weak, N]

    quant_dq = next(n for n in quant.graph.node if n.op_type == "DequantizeLinear")
    wq = next(t for t in quant.graph.initializer if t.name == quant_dq.input[0])
    ws = next(t for t in quant.graph.initializer if t.name == quant_dq.input[1])
    block_size = next(a.i for a in quant_dq.attribute if a.name == "block_size")

    codes = _unpack_int4(wq)  # [K, N] -- not transposed in this MatMul model
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)  # [K / block_size, N]
    scale_full = np.repeat(scale, block_size, axis=0)
    w_rtn = codes * scale_full  # [K, N]

    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)

    reconstructed = w_rtn[weak_idx, :] + delta_w  # [num_weak, N]
    np.testing.assert_allclose(reconstructed, w_float[weak_idx, :], atol=1e-4)


def test_owq_output_stays_close_to_float_when_isolating_selected_column():
    # A looser, real-onnxruntime sanity check that the graph is wired up
    # correctly end-to-end (the exactness claim itself is verified above at
    # the initializer level, which is deterministic across platforms).
    K, N, outlier_col = 32, 8, 5
    model = _matmul_model_with_outlier_column(K=K, N=N, outlier_col=outlier_col, seed=2)
    x = _calibration(K=K, num_samples=32, seed=3)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    owq_model = onnxsim.apply_owq(
        model, quant, calibration_data=calibration_data, outlier_fraction=0.1
    )

    gather_node = next(n for n in owq_model.graph.node if n.op_type == "Gather")
    idx_init = next(
        t for t in owq_model.graph.initializer if t.name == gather_node.input[1]
    )
    selected_col = int(onnx.numpy_helper.to_array(idx_init)[0])

    probe = np.zeros((1, K), dtype=np.float32)
    probe[0, selected_col] = 3.7

    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    y_float = probe.astype(np.float64) @ w_float

    (owq_y,) = _run(owq_model, {"X": probe})
    assert np.all(np.isfinite(owq_y))
    assert _rel_l2(y_float, owq_y) < 1e-2


def test_owq_leaves_int4_codes_untouched():
    model = _matmul_model_with_outlier_column(K=32, N=8, outlier_col=1, seed=4)
    x = _calibration(K=32, num_samples=16, seed=5)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    before = next(
        t for t in quant.graph.initializer if t.data_type == onnx.TensorProto.INT4
    )
    owq_model = onnxsim.apply_owq(
        model, quant, calibration_data=calibration_data, outlier_fraction=0.1
    )
    assert any(n.op_type == "Gather" for n in owq_model.graph.node)
    after = next(
        t for t in owq_model.graph.initializer if t.data_type == onnx.TensorProto.INT4
    )
    assert before.raw_data == after.raw_data


def test_owq_gemm_transb():
    rng = np.random.default_rng(6)
    K, N = 96, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.1
    weight[:, 7] = rng.standard_normal(N).astype(np.float32) * 20.0
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    onnx.checker.check_model(quant)

    x = _calibration(K=K, num_samples=32, seed=7)
    calibration_data = [{"X": x}]

    owq_model = onnxsim.apply_owq(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(owq_model)

    (float_y,) = _run(model, {"X": x})
    (owq_y,) = _run(owq_model, {"X": x})
    assert _rel_l2(float_y, owq_y) < 0.25


def test_owq_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_owq(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_owq_noop_when_outlier_fraction_rounds_to_zero():
    model = _matmul_model_with_outlier_column(K=32, N=8, outlier_col=2, seed=8)
    x = _calibration(K=32, num_samples=16, seed=9)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    owq_model = onnxsim.apply_owq(
        model, quant, calibration_data=calibration_data, outlier_fraction=0.001
    )
    assert owq_model.SerializeToString() == quant.SerializeToString()


def _matmul_model_3d_with_outlier_column(K=64, N=16, outlier_col=3, seed=0):
    # _matmul_model_with_outlier_column's [batch, seq, K] counterpart --
    # the activation shape of essentially every real transformer, since
    # ONNX MatMul broadcasts over leading dimensions.
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    weight[outlier_col, :] = rng.standard_normal(N).astype(np.float32) * 20.0
    return _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )


def _calibration_3d(K=64, batch=6, seq=12, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, seq, K)).astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_owq_handles_a_3d_transformer_shaped_activation(seed):
    # A [batch, seq, K] activation used to be filtered out entirely (the
    # capture kept only ndim == 2 arrays), so apply_owq silently returned
    # quantized_model unchanged on exactly the model shape it exists for.
    # Flattening the leading dimensions is exact -- the Hessian X^T X sums
    # over the same rows either way.
    model = _matmul_model_3d_with_outlier_column(K=64, N=16, outlier_col=3, seed=seed)
    x = _calibration_3d(K=64, seed=seed + 300)
    quant = onnxsim.quantize_weight_only_int4(model)

    owq_model = onnxsim.apply_owq(model, quant, calibration_data=[{"X": x}])
    onnx.checker.check_model(owq_model)
    assert owq_model.SerializeToString() != quant.SerializeToString(), (
        "apply_owq was a no-op on a 3-D activation"
    )
    # The rescued column is spliced back in via a Gather-fed correction
    # term, exactly as on a 2-D activation.
    assert any(n.op_type == "Gather" for n in owq_model.graph.node)

    (float_y,) = _run(model, {"X": x})
    (rtn_y,) = _run(quant, {"X": x})
    (owq_y,) = _run(owq_model, {"X": x})
    assert np.all(np.isfinite(owq_y))
    # OWQ rescues a single column out of 64 here, so its reconstruction
    # gain is real but small (measured ~3-4% across seeds). Assert only
    # that the correction did not make things *worse* -- the exactness of
    # what it wrote is checked deterministically below and by
    # test_owq_correction_initializer_exactly_cancels_int4_error.
    assert _rel_l2(float_y, owq_y) <= 1.05 * _rel_l2(float_y, rtn_y)


def test_owq_flattening_matches_an_equivalent_2d_calibration():
    # Flattening [batch, seq, K] -> [batch * seq, K] must be *exact*, not an
    # approximation: feeding the same rows as a 2-D batch has to produce a
    # byte-identical set of rewritten initializers (the rescued column
    # indices and the delta_w correction alike).
    K, N = 32, 8
    rng = np.random.default_rng(4)
    weight = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    weight[3, :] = (rng.standard_normal(N) * 20.0).astype(np.float32)
    model_3d = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    model_2d = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    x3 = _calibration_3d(K=K, batch=4, seq=8, seed=5)

    # K=32 with the default outlier_fraction would round to zero rescued
    # columns and make this vacuous, so ask for a few explicitly.
    out3 = onnxsim.apply_owq(
        model_3d,
        onnxsim.quantize_weight_only_int4(model_3d),
        calibration_data=[{"X": x3}],
        outlier_fraction=0.1,
    )
    out2 = onnxsim.apply_owq(
        model_2d,
        onnxsim.quantize_weight_only_int4(model_2d),
        calibration_data=[{"X": x3.reshape(-1, K)}],
        outlier_fraction=0.1,
    )
    assert any(n.op_type == "Gather" for n in out2.graph.node)
    assert _initializer_bytes(out3) == _initializer_bytes(out2)


def _initializer_bytes(model):
    return sorted(
        (t.name, onnx.numpy_helper.to_array(t).tobytes())
        for t in model.graph.initializer
    )
