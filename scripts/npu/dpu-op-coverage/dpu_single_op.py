"""Run one ONNX op on the Intel NPU through a blob that OpenVINO's compiler produces for that op alone.

Used for ops that the compiler places on the DPU without SHAVE code (see NOTES.md for the census). Each (op, attrs, shapes)
is compiled once with compile_tool (FP32 I/O) and cached in memory; calls then import the blob and run in-process.

Requires OPENVINO_COMPILE_TOOL (or compile_tool on PATH), the OpenVINO Python package, and the level-zero loader on
LD_LIBRARY_PATH if it is not in the default path.
"""
import hashlib
import os
import shutil
import subprocess
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

_WORK = os.path.join(tempfile.gettempdir(), "onnxsim_dpu_single_op")
_CORE = None
_CACHE = {}


def _core():
    global _CORE
    if _CORE is None:
        import openvino as ov
        _CORE = ov.Core()
    return _CORE


def run(op_type, inputs, out_shape, attrs=None, consts=None):
    """Run `op_type` on `inputs` (numpy float32 arrays, in order) with `consts` (dict: input position -> numpy array,
    passed as initializers). Returns a float32 array of `out_shape`.

    Example: run("AveragePool", [x], [1, 64, 14, 14], attrs={"kernel_shape": [2, 2], "strides": [2, 2]})
    """
    attrs = attrs or {}
    consts = consts or {}
    names = []
    graph_inputs, inits = [], []
    k = 0
    for pos in range(len(inputs) + len(consts)):
        if pos in consts and consts[pos] is None:
            names.append("")  # omitted optional input (for example the ROI of Resize)
        elif pos in consts:
            nm = "c%d" % pos
            inits.append(numpy_helper.from_array(np.asarray(consts[pos], np.float32), nm))
            names.append(nm)
        else:
            nm = "x%d" % pos
            shape = list(inputs[k].shape)
            graph_inputs.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, shape))
            names.append(nm)
            k += 1
    key = (op_type, tuple(sorted((a, repr(v)) for a, v in attrs.items())),
           tuple((tuple(i.shape)) for i in inputs), tuple(out_shape),
           tuple((p, None if consts[p] is None else tuple(np.asarray(consts[p]).shape)) for p in sorted(consts)))
    req = _CACHE.get(key)
    if req is None:
        req = _compile(key, op_type, names, attrs, graph_inputs, inits, out_shape)
        _CACHE[key] = req
    feeds = {i: np.ascontiguousarray(a, np.float32) for i, a in enumerate(inputs)}
    out = req.infer(feeds)
    return np.asarray(next(iter(out.values())), np.float32).reshape(out_shape)


def _compile(key, op_type, names, attrs, graph_inputs, inits, out_shape):
    os.makedirs(_WORK, exist_ok=True)
    tag = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
    path = os.path.join(_WORK, "%s_%s.onnx" % (op_type.lower(), tag))
    blob = path[:-5] + ".blob"
    node = helper.make_node(op_type, names, ["y"], **attrs)
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)
    g = helper.make_graph([node], op_type, graph_inputs, [out], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.save(m, path)
    tool = os.environ.get("OPENVINO_COMPILE_TOOL") or shutil.which("compile_tool")
    if not tool:
        raise RuntimeError("compile_tool not found: set OPENVINO_COMPILE_TOOL")
    r = subprocess.run([tool, "-m", path, "-d", "NPU", "-o", blob, "-ip", "FP32", "-op", "FP32"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("compile_tool failed for %s: %s" % (op_type, r.stderr[-300:]))
    return _core().import_model(open(blob, "rb").read(), "NPU").create_infer_request()
