"""Non-conv ops on the DPU: MaxPool, GlobalAveragePool, and Add with an optional fused Relu. Each op shape is compiled once by
OpenVINO's compile_tool (the census showed these compile to DPU-only blobs), then run in-process. Cached by shape and attrs."""
import hashlib
import os
import shutil
import subprocess
import tempfile

import numpy as np
import onnx
from onnx import helper, TensorProto

import dpu_layers as dpu

_WORK = os.path.join(tempfile.gettempdir(), "onnxsim_dpu_ops")
_MODELS = {}


def _compile(key, name, nodes, inputs, out):
    if key in _MODELS:
        return _MODELS[key]
    os.makedirs(_WORK, exist_ok=True)
    g = helper.make_graph(nodes, "op", inputs, [out])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    tag = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
    onnx_path = os.path.join(_WORK, "%s_%s.onnx" % (name, tag))
    blob_path = onnx_path[:-5] + ".blob"
    onnx.save(m, onnx_path)
    tool = os.environ.get("OPENVINO_COMPILE_TOOL") or shutil.which("compile_tool")
    if not tool:
        raise RuntimeError("compile_tool not found: set OPENVINO_COMPILE_TOOL")
    r = subprocess.run([tool, "-m", onnx_path, "-d", "NPU", "-o", blob_path, "-ip", "FP16", "-op", "FP16"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("compile_tool failed: " + r.stderr[-300:])
    req = dpu._core().import_model(open(blob_path, "rb").read(), "NPU").create_infer_request()
    _MODELS[key] = req
    return req


def _run(req, feeds, out_shape):
    out = req.infer(feeds)
    return np.asarray(next(iter(out.values())), dtype=np.float32).reshape(out_shape)


def add(a, b, relu=False):
    shape = list(a.shape)
    x0 = helper.make_tensor_value_info("a", TensorProto.FLOAT16, shape)
    x1 = helper.make_tensor_value_info("b", TensorProto.FLOAT16, shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape)
    nodes = [helper.make_node("Add", ["a", "b"], ["s"])]
    if relu:
        nodes.append(helper.make_node("Relu", ["s"], ["y"]))
    else:
        nodes.append(helper.make_node("Identity", ["s"], ["y"]))
    req = _compile(("add", tuple(shape), relu), "add%s" % ("_relu" if relu else ""), nodes, [x0, x1], y)
    return _run(req, {0: a.astype(np.float16), 1: b.astype(np.float16)}, shape)


def maxpool(x, k, s, p):
    shape = list(x.shape)
    hout = (shape[2] + 2 * p - k) // s + 1
    wout = (shape[3] + 2 * p - k) // s + 1
    oshape = [shape[0], shape[1], hout, wout]
    xi = helper.make_tensor_value_info("x", TensorProto.FLOAT16, shape)
    yo = helper.make_tensor_value_info("y", TensorProto.FLOAT16, oshape)
    node = helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[k, k], strides=[s, s], pads=[p] * 4)
    req = _compile(("maxpool", tuple(shape), k, s, p), "maxpool", [node], [xi], yo)
    return _run(req, {0: x.astype(np.float16)}, oshape)


def global_avg_pool(x):
    shape = list(x.shape)
    oshape = [shape[0], shape[1], 1, 1]
    xi = helper.make_tensor_value_info("x", TensorProto.FLOAT16, shape)
    yo = helper.make_tensor_value_info("y", TensorProto.FLOAT16, oshape)
    node = helper.make_node("GlobalAveragePool", ["x"], ["y"])
    req = _compile(("gap", tuple(shape)), "gap", [node], [xi], yo)
    return _run(req, {0: x.astype(np.float16)}, oshape)
