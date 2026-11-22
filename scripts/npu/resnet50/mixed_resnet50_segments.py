"""ResNet50 as segments on the Intel NPU: the graph is cut at residual block boundaries, and each segment (the stem plus
blocks, a group of consecutive blocks, or the classifier tail) is extracted as a sub-model and compiled by OpenVINO into
one NPU blob. The walk makes one device call per segment. Numbers and output are compared with onnxruntime.

Usage: python mixed_resnet50_segments.py [--group N]   (N consecutive residual blocks per segment; default 1)

The segment compile is OpenVINO's work; this module only chooses the cut points and runs the blobs in order.
Requires OPENVINO_COMPILE_TOOL (or compile_tool on PATH), the OpenVINO Python package, and LD_LIBRARY_PATH for the
level-zero loader if it is not in the default path.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference
from onnx.utils import Extractor

MODEL = os.path.expanduser("~/models/resnet50_opset17.onnx")
_WORK = os.path.join(tempfile.gettempdir(), "onnxsim_resnet50_segments")
_CORE = None


def _core():
    global _CORE
    if _CORE is None:
        import openvino as ov
        _CORE = ov.Core()
    return _CORE


def block_cuts(model):
    """Tensor names where a residual block ends: outputs of a Relu whose input is an Add."""
    prod = {o: n for n in model.graph.node for o in n.output}
    return [n.output[0] for n in model.graph.node
            if n.op_type == "Relu" and prod.get(n.input[0]) is not None and prod[n.input[0]].op_type == "Add"]


def compile_segment(inf, name, inp, out):
    """Extract the nodes between inp and out and compile them to one NPU blob. Returns an infer request."""
    os.makedirs(_WORK, exist_ok=True)
    tag = hashlib.sha1(("%s|%s|%s" % (name, inp, out)).encode()).hexdigest()[:16]
    path = os.path.join(_WORK, "seg_%s.onnx" % tag)
    blob = path[:-5] + ".blob"
    if not os.path.exists(blob):
        onnx.save(Extractor(inf).extract_model([inp], [out]), path)
        tool = os.environ.get("OPENVINO_COMPILE_TOOL") or shutil.which("compile_tool")
        if not tool:
            raise RuntimeError("compile_tool not found: set OPENVINO_COMPILE_TOOL")
        r = subprocess.run([tool, "-m", path, "-d", "NPU", "-o", blob, "-ip", "FP32", "-op", "FP32"], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("compile failed for %s: %s" % (name, r.stderr[-300:]))
    return _core().import_model(open(blob, "rb").read(), "NPU").create_infer_request()


def build_segments(model, group):
    """Return a list of (name, input tensor, output tensor, infer request), cut every `group` blocks."""
    inf = shape_inference.infer_shapes(model)
    cuts = block_cuts(model)
    bounds = [model.graph.input[0].name] + cuts[group - 1::group]
    if bounds[-1] != cuts[-1]:
        bounds.append(cuts[-1])
    segs = []
    for i in range(len(bounds) - 1):
        segs.append(("seg%02d" % i, bounds[i], bounds[i + 1]))
    segs.append(("tail", bounds[-1], model.graph.output[0].name))
    return [(n, a, b, compile_segment(inf, n, a, b)) for n, a, b in segs]


def run(segs, x):
    feed = x
    for _, _, _, req in segs:
        feed = np.asarray(next(iter(req.infer({0: np.ascontiguousarray(feed, np.float32)}).values())), np.float32)
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=10)
    args = ap.parse_args()
    model = onnx.load(MODEL)
    t = time.time()
    segs = build_segments(model, args.group)
    t_build = time.time() - t
    x = np.random.default_rng(0).standard_normal((1, 3, 224, 224), dtype=np.float32)
    ref = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"]).run(None, {model.graph.input[0].name: x})[0]
    got = run(segs, x).reshape(ref.shape)
    ts = []
    for _ in range(args.repeat):
        t = time.time(); run(segs, x); ts.append(time.time() - t)
    print("group=%d | device calls %d | build %.1fs | warm median %.4fs | max diff %.4g (ref max %.3g) | top-1 %d vs %d" % (
        args.group, len(segs), t_build, float(np.median(ts)), np.abs(got - ref).max(), np.abs(ref).max(),
        int(got.argmax()), int(ref.argmax())), flush=True)


if __name__ == "__main__":
    main()
