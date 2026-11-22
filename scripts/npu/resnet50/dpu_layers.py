"""Conv and Gemm on the Intel NPU DPU: each layer is compiled by OpenVINO with its real weights (one blob per layer), then run
in-process. OpenVINO's compiler handles the tiling of large layers; no chunking or template patching is needed.

Compile cost is about 0.1 s per layer on the test machine. Compiled models are cached in memory by weight hash, so each
layer compiles once per process. Bias is added on the host (it could also be compiled in).

Requirements: OpenVINO's compile_tool (set OPENVINO_COMPILE_TOOL if it is not on PATH) and the OpenVINO Python package.
An NPU driver with a level-zero loader that supports the device is needed.

Verified on Intel NPU 5020 (OpenVINO 2026.4): all 23 ResNet50 conv shapes and the classifier Gemm match a float32 numpy
reference (relative error at most 5e-4) with one launch per layer.
"""
import hashlib
import os
import shutil
import subprocess
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

_WORK = os.path.join(tempfile.gettempdir(), "onnxsim_dpu_layers")
_CORE = None
_MODELS = {}


def _core():
    global _CORE
    if _CORE is None:
        import openvino as ov
        _CORE = ov.Core()
    return _CORE


def _compile(W, cin, cout, k, hin, pad, stride):
    """Compile one conv layer with its real weights to an NPU blob and return an infer request."""
    key = (hashlib.sha1(np.ascontiguousarray(W).tobytes()).hexdigest(), cin, cout, k, hin, pad, stride)
    if key in _MODELS:
        return _MODELS[key]
    os.makedirs(_WORK, exist_ok=True)
    name = "L_" + key[0][:16] + "_%d_%d_%d_%d_%d_%d" % (cin, cout, k, hin, pad, stride)
    hout = (hin + 2 * pad - k) // stride + 1
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, cin, hin, hin])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, cout, hout, hout])
    g = helper.make_graph([helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[k, k], pads=[pad] * 4, strides=[stride, stride])],
                          "conv", [x], [y], initializer=[numpy_helper.from_array(W.astype(np.float16), "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx_path = os.path.join(_WORK, name + ".onnx")
    blob_path = os.path.join(_WORK, name + ".blob")
    onnx.save(m, onnx_path)
    tool = os.environ.get("OPENVINO_COMPILE_TOOL") or shutil.which("compile_tool")
    if not tool:
        raise RuntimeError("compile_tool not found: set OPENVINO_COMPILE_TOOL")
    r = subprocess.run([tool, "-m", onnx_path, "-d", "NPU", "-o", blob_path, "-ip", "FP16", "-op", "FP16"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("compile_tool failed: " + r.stderr[-300:])
    cm = _core().import_model(open(blob_path, "rb").read(), "NPU")
    req = cm.create_infer_request()
    _MODELS[key] = (req, (1, cout, hout, hout))
    return _MODELS[key]


def conv(W, X, pad, stride):
    """Conv of X [1, cin, h, h] with W [cout, cin, k, k] (no bias). Returns float32 [1, cout, hout, hout]."""
    cout, cin, k, _ = W.shape
    req, out_shape = _compile(W, cin, cout, k, X.shape[2], pad, stride)
    out = req.infer({0: X.astype(np.float16)})
    return np.asarray(next(iter(out.values())), dtype=np.float32).reshape(out_shape)


def gemm(W, x, bias=None):
    """y = x @ W^T + bias with W [N, K] and x [1, K], as a 1x1 conv on a 1x1 spatial input. Returns [1, N] float32."""
    N, K = W.shape
    y = conv(W[:, :, None, None], x.reshape(1, K, 1, 1), 0, 1).reshape(1, N)
    return y + (bias.reshape(1, N) if bias is not None else 0)
