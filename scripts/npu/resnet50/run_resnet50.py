"""Run ResNet50 on the Intel NPU through one of two routes, on the same random 224x224 input, and compare with onnxruntime.

  --backend whole   the whole graph compiled to ONE NPU blob by OpenVINO's compiler (one device call per image)
  --backend dpu     the per-op walk with Conv and Gemm on the DPU (dpu_layers.py, one launch per layer; mixed_resnet50_dpu.py)

Prints the first call (includes compiles) and the warm median over --repeat calls.

Usage:
  OPENVINO_COMPILE_TOOL=/path/to/compile_tool python run_resnet50.py --backend whole
  OPENVINO_COMPILE_TOOL=/path/to/compile_tool LD_LIBRARY_PATH=$HOME/ze-loader/lib python run_resnet50.py --backend dpu
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MODEL = os.path.expanduser("~/models/resnet50_opset17.onnx")


def run_whole(x, repeat):
    import openvino as ov
    tool = os.environ.get("OPENVINO_COMPILE_TOOL") or shutil.which("compile_tool")
    if not tool:
        raise SystemExit("compile_tool not found: set OPENVINO_COMPILE_TOOL")
    blob = os.path.join(tempfile.gettempdir(), "onnxsim_resnet50_whole.blob")
    t = time.time()
    if not os.path.exists(blob):
        r = subprocess.run([tool, "-m", MODEL, "-d", "NPU", "-o", blob, "-ip", "FP32", "-op", "FP32"], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("compile failed: " + r.stderr[-300:])
    t_compile = time.time() - t
    req = ov.Core().import_model(open(blob, "rb").read(), "NPU").create_infer_request()
    t = time.time()
    out = next(iter(req.infer({0: x}).values()))
    first = time.time() - t
    ts = []
    for _ in range(repeat):
        t = time.time(); req.infer({0: x}); ts.append(time.time() - t)
    return np.asarray(out, np.float32), t_compile, first, float(np.median(ts))


def run_dpu(x, repeat):
    import mixed_resnet50_dpu as w
    model = onnx.load(MODEL)
    t = time.time()
    out = w.walk(model, x)
    first = time.time() - t
    ts = []
    for _ in range(repeat):
        t = time.time(); w.walk(model, x); ts.append(time.time() - t)
    return out, 0.0, first, float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["whole", "dpu"], required=True)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    x = np.random.default_rng(0).standard_normal((1, 3, 224, 224), dtype=np.float32)
    ref = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    fn = run_whole if args.backend == "whole" else run_dpu
    got, t_compile, first, warm = fn(x, args.repeat)
    got = got.reshape(ref.shape)
    print("backend=%s | compile %.1fs | first call %.2fs | warm median %.3fs | max diff %.4g (ref max %.3g) | top-1 %d vs %d"
          % (args.backend, t_compile, first, warm, np.abs(got - ref).max(), np.abs(ref).max(), int(got.argmax()), int(ref.argmax())), flush=True)


if __name__ == "__main__":
    main()
