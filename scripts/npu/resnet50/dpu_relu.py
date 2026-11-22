"""conv (optionally followed by Relu) compiled as one OpenVINO blob with its real weights. Experiment module on top of
dpu_layers (same compile path, same cache idea, relu in the graph)."""
import hashlib
import os
import tempfile
import subprocess
import shutil
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import dpu_layers as dpu  # noqa: E402

_MODELS = {}
_WORK = os.path.join(tempfile.gettempdir(), "onnxsim_dpu_relu")


def _compile(W, B, cin, cout, k, hin, pad, stride, relu):
    key = (hashlib.sha1(np.ascontiguousarray(W).tobytes() + np.ascontiguousarray(B).tobytes()).hexdigest(), cin, cout, k, hin, pad, stride, relu)
    if key in _MODELS:
        return _MODELS[key]
    os.makedirs(_WORK, exist_ok=True)
    name = "R_" + key[0][:16] + "_%d_%d_%d_%d_%d_%d_%d" % (cin, cout, k, hin, pad, stride, int(relu))
    hout = (hin + 2 * pad - k) // stride + 1
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, cin, hin, hin])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, cout, hout, hout])
    nodes = [helper.make_node("Conv", ["x", "W", "B"], ["c"], kernel_shape=[k, k], pads=[pad] * 4, strides=[stride, stride])]
    nodes.append(helper.make_node("Relu", ["c"], ["y"]) if relu else helper.make_node("Identity", ["c"], ["y"]))
    g = helper.make_graph(nodes, "conv", [x], [y], initializer=[numpy_helper.from_array(W.astype(np.float16), "W"), numpy_helper.from_array(B.astype(np.float16), "B")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx_path = os.path.join(_WORK, name + ".onnx")
    blob_path = os.path.join(_WORK, name + ".blob")
    onnx.save(m, onnx_path)
    tool = os.environ.get("OPENVINO_COMPILE_TOOL") or shutil.which("compile_tool")
    r = subprocess.run([tool, "-m", onnx_path, "-d", "NPU", "-o", blob_path, "-ip", "FP16", "-op", "FP16"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("compile_tool failed: " + r.stderr[-300:])
    cm = dpu._core().import_model(open(blob_path, "rb").read(), "NPU")
    _MODELS[key] = (cm.create_infer_request(), (1, cout, hout, hout))
    return _MODELS[key]


def conv_r(W, B, X, pad, stride, relu):
    cout, cin, k, _ = W.shape
    req, shape = _compile(W, B, cin, cout, k, X.shape[2], pad, stride, relu)
    out = req.infer({0: X.astype(np.float16)})
    return np.asarray(next(iter(out.values())), dtype=np.float32).reshape(shape)
