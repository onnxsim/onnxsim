import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "apple" / "benchmark_onnx_pipeline.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("benchmark_onnx_pipeline", MODULE_PATH)
PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PIPELINE)


def _two_input_model(name, input_names, output_name):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", input_names, [output_name])],
        name,
        [
            onnx.helper.make_tensor_value_info(
                input_name, onnx.TensorProto.FLOAT, [2, 3]
            )
            for input_name in input_names
        ],
        [
            onnx.helper.make_tensor_value_info(
                output_name, onnx.TensorProto.FLOAT, [2, 3]
            )
        ],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return model


def _model(name, input_name, output_name, op):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node(op, [input_name], [output_name])],
        name,
        [
            onnx.helper.make_tensor_value_info(
                input_name, onnx.TensorProto.FLOAT, [2, 3]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                output_name, onnx.TensorProto.FLOAT, [2, 3]
            )
        ],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return model


def test_fuse_stage_models_connects_outputs_and_explicit_inputs():
    first = _model("first", "x", "features", "Identity")
    second = _model("second", "features", "head", "Relu")
    fused, sources, outputs = PIPELINE._fuse_stage_models(
        ["first", "second"],
        {"first": first, "second": second},
        [{"from": ["first", "features"], "to": ["second", "features"]}],
        [],
    )
    assert [value.name for value in fused.graph.output] == ["fused_1_head"]
    assert outputs == ["head"]
    assert sources == {"fused_0_x": ("first", "x")}
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = ort.InferenceSession(
        fused.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"fused_0_x": x})[0]
    np.testing.assert_array_equal(result, x)


def test_fuse_stage_models_connects_explicit_shared_input():
    first = _model("first", "x", "features", "Identity")
    second = _two_input_model("second", ["features", "shared"], "head")
    fused, sources, _ = PIPELINE._fuse_stage_models(
        ["first", "second"],
        {"first": first, "second": second},
        [{"from": ["first", "features"], "to": ["second", "features"]}],
        [{"from": ["first", "x"], "to": ["second", "shared"]}],
    )
    assert sources["fused_0_x"] == ("first", "x")
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = ort.InferenceSession(
        fused.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"fused_0_x": x})[0]
    np.testing.assert_array_equal(result, x * 2)


def test_fuse_stage_models_rejects_forward_input_connection_without_mapping():
    first = _model("first", "x", "features", "Identity")
    second = _model("second", "features", "head", "Relu")
    with pytest.raises(ValueError, match="no connection"):
        PIPELINE._fuse_stage_models(
            ["first", "second"],
            {"first": first, "second": second},
            [],
            [],
        )
