"""Tests for ``onnxsim.quantize_qoperator_gemm`` (the
``qoperator_quantize_gemm`` C++ pass) -- the fully-general analogue of
``test_qoperator_quantize_matmul.py``'s ``QLinearMatMul`` coverage, using
ONNX Runtime's "com.microsoft" contrib op ``QGemm`` instead, which handles
any transA/transB/alpha (not just "vanilla" Gemm) and quantizes the bias
directly rather than adding it back in float.
"""

import collections

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(nodes, inputs, outputs, initializer, opset=13):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _assert_close(float_outputs, quant_outputs, rel_l2_tol=0.1):
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < rel_l2_tol, f"relative L2 error too large: {rel_l2:.4f}"


def test_quantize_vanilla_gemm_with_bias():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 4)).astype(np.float32)  # [K, N]
    b = rng.standard_normal((4,)).astype(np.float32)
    nodes = [onnx.helper.make_node("Gemm", ["X", "W", "B"], ["Y"])]
    model = _model(
        nodes, [_vi("X", [3, 8])], [_vi("Y", [3, 4])], [_f32(w, "W"), _f32(b, "B")]
    )

    quant = onnxsim.quantize_qoperator_gemm(model, num_calibration_samples=16, seed=0)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Gemm"] == 0
    assert ops["QGemm"] == 1
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 1
    domains = {o.domain for o in quant.opset_import}
    assert "com.microsoft" in domains

    qgemm = next(n for n in quant.graph.node if n.op_type == "QGemm")
    assert len(qgemm.input) == 9
    assert qgemm.input[6] != ""  # C is present (quantized bias)

    x = rng.standard_normal((3, 8)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_gemm_no_bias():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((8, 4)).astype(np.float32)
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [3, 8])], [_vi("Y", [3, 4])], [_f32(w, "W")])

    quant = onnxsim.quantize_qoperator_gemm(model, num_calibration_samples=16, seed=1)
    onnx.checker.check_model(quant)
    qgemm = next(n for n in quant.graph.node if n.op_type == "QGemm")
    assert qgemm.input[6] == ""  # C omitted -- empty-string placeholder

    x = rng.standard_normal((3, 8)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_gemm_transa_transb_alpha():
    # The exact case quantize_qoperator's QLinearMatMul path cannot handle
    # (transA != 0 and alpha != 1) -- QGemm carries these as its own
    # attributes instead.
    rng = np.random.default_rng(2)
    w = rng.standard_normal((4, 8)).astype(np.float32)  # [N, K] since transB=1
    b = rng.standard_normal((4,)).astype(np.float32)
    nodes = [
        onnx.helper.make_node(
            "Gemm",
            ["X", "W", "B"],
            ["Y"],
            transA=1,
            transB=1,
            alpha=2.5,
        )
    ]
    # X is [K, M] = [8, 3] since transA=1.
    model = _model(
        nodes, [_vi("X", [8, 3])], [_vi("Y", [3, 4])], [_f32(w, "W"), _f32(b, "B")]
    )

    quant = onnxsim.quantize_qoperator_gemm(model, num_calibration_samples=16, seed=2)
    onnx.checker.check_model(quant)
    qgemm = next(n for n in quant.graph.node if n.op_type == "QGemm")
    trans_a = next(a.i for a in qgemm.attribute if a.name == "transA")
    trans_b = next(a.i for a in qgemm.attribute if a.name == "transB")
    alpha = next(a.f for a in qgemm.attribute if a.name == "alpha")
    assert trans_a == 1
    assert trans_b == 1
    assert alpha == pytest.approx(2.5)

    x = rng.standard_normal((8, 3)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_skips_non_vector_bias():
    # A 2-D (or otherwise non-1-D) C is outside this pass's handled scope.
    rng = np.random.default_rng(3)
    w = rng.standard_normal((8, 4)).astype(np.float32)
    c2d = rng.standard_normal((3, 4)).astype(np.float32)
    nodes = [onnx.helper.make_node("Gemm", ["X", "W", "C"], ["Y"])]
    model = _model(
        nodes, [_vi("X", [3, 8])], [_vi("Y", [3, 4])], [_f32(w, "W"), _f32(c2d, "C")]
    )

    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_gemm_quantizable_tensors(model.SerializeToString())
    assert names == []

    quant = onnxsim.quantize_qoperator_gemm(model)
    assert _op_counts(quant)["Gemm"] == 1
    assert _op_counts(quant)["QGemm"] == 0


def test_quantize_skips_non_default_beta_with_bias():
    rng = np.random.default_rng(4)
    w = rng.standard_normal((8, 4)).astype(np.float32)
    b = rng.standard_normal((4,)).astype(np.float32)
    nodes = [onnx.helper.make_node("Gemm", ["X", "W", "B"], ["Y"], beta=0.5)]
    model = _model(
        nodes, [_vi("X", [3, 8])], [_vi("Y", [3, 4])], [_f32(w, "W"), _f32(b, "B")]
    )
    quant = onnxsim.quantize_qoperator_gemm(model)
    assert _op_counts(quant)["Gemm"] == 1
    assert _op_counts(quant)["QGemm"] == 0


def test_quantize_skips_non_float():
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"])]
    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT16, [3, 8])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, [3, 4])],
        [
            onnx.numpy_helper.from_array(
                np.zeros((8, 4), dtype=np.float32).astype(np.float16), "W"
            )
        ],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=10
    )
    quant = onnxsim.quantize_qoperator_gemm(model)
    assert _op_counts(quant)["Gemm"] == 1
    assert _op_counts(quant)["QGemm"] == 0


def test_list_qoperator_gemm_quantizable_tensors():
    rng = np.random.default_rng(5)
    w = rng.standard_normal((8, 4)).astype(np.float32)
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [3, 8])], [_vi("Y", [3, 4])], [_f32(w, "W")])
    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_gemm_quantizable_tensors(model.SerializeToString())
    assert set(names) == {"X", "Y"}
