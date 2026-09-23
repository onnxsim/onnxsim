"""Smallest float-in/float-out QDQ int8 conv: Q(x)->DQ->Conv(DQ(w))->Q->DQ.

The standard QDQ form QNN's HTP backend lowers to a real int8 conv (the backbone's own
input is float too). Smoke test for qnn_run before the real backbone.
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(out_path: str) -> None:
    rng = np.random.default_rng(0)
    w = rng.integers(-40, 40, (32, 32, 3, 3)).astype(np.int8)
    inits = [
        numpy_helper.from_array(w, "w"),
        numpy_helper.from_array(np.float32(0.02), "xs"),
        numpy_helper.from_array(np.uint8(114), "xz"),
        numpy_helper.from_array(np.float32(0.01), "ws"),
        numpy_helper.from_array(np.int8(0), "wz"),
        numpy_helper.from_array(np.float32(0.05), "ys"),
        numpy_helper.from_array(np.uint8(120), "yz"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "xs", "xz"], ["xq"]),
        helper.make_node("DequantizeLinear", ["xq", "xs", "xz"], ["xf"]),
        helper.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wf"]),
        helper.make_node("Conv", ["xf", "wf"], ["yf"], pads=[1, 1, 1, 1]),
        helper.make_node("QuantizeLinear", ["yf", "ys", "yz"], ["yq"]),
        helper.make_node("DequantizeLinear", ["yq", "ys", "yz"], ["y"]),
    ]
    g = helper.make_graph(
        nodes,
        "tiny_qdq_conv_f32io",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32, 64, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32, 64, 64])],
        inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, out_path)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "tiny_qdq_conv_f32io.onnx")
