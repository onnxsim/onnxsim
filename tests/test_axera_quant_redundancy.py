import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import quant_redundancy  # noqa: E402


def _model():
    scale = numpy_helper.from_array(np.array([0.25], dtype=np.float32), "scale")
    zero = numpy_helper.from_array(np.array([0], dtype=np.uint8), "zero")
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["qx0"]),
        helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["qx1"]),
        helper.make_node("DequantizeLinear", ["qx0", "scale", "zero"], ["dx"]),
        helper.make_node("Add", ["dx", "qx1"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "qdq",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
        [scale, zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_reports_only_exact_duplicates_as_safe():
    report = quant_redundancy.analyze(_model())
    assert report["quantize_nodes"] == 2
    assert report["dequantize_nodes"] == 1
    assert report["safe_duplicate_removals"] == 1
    assert len(report["unsafe_roundtrips"]) == 1


def test_duplicate_dequantizers_are_safe_candidates():
    model = _model()
    model.graph.node.append(
        helper.make_node("DequantizeLinear", ["qx0", "scale", "zero"], ["dx2"])
    )
    report = quant_redundancy.analyze(model)
    assert report["safe_duplicate_removals"] == 2
    assert any(group["canonical"] == "dx" for group in report["duplicate_groups"])
